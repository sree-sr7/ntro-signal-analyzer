"""Bounded candidate search across supported FEC and interleaver hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.bitstream.crc import verify_crc16
from core.bitstream.extractor import bits_to_bytes, bytes_to_bits, validate_binary_bits
from core.fec.concatenated import CONCAT_CODED_BITS, CONCAT_FEC_NAME, CONCAT_PAYLOAD_BITS, decode_concatenated
from core.fec.convolutional import CONVOLUTIONAL_CODE_SPECS, decode_viterbi
from core.fec.ldpc_decoder import LDPC_CONFIGS, LDPC_DECODER_ALGORITHM, decode_ldpc_frame
from core.fec.reed_solomon import RS_CODE_SPECS, rs_decode
from core.interleaver.block import deinterleave_block
from core.interleaver.convolutional import deinterleave_convolutional
from core.interleaver.diagonal import (
    DIAGONAL_BLOCK_SIZE,
    DIAGONAL_COLUMNS,
    DIAGONAL_ROWS,
    DIAGONAL_STEP,
    deinterleave as deinterleave_diagonal,
)
from core.interleaver.pseudorandom import (
    DEFAULT_PSEUDORANDOM_CONFIG,
    PseudoRandomInterleaver,
    PseudoRandomInterleaverConfig,
)


_INTERLEAVERS: tuple[tuple[str | None, str, int | None, int | None], ...] = (
    (None, "none", None, None),
    ("Block_8x8", "block", 8, 8),
    ("Block_16x16", "block", 16, 16),
    ("Block_32x32", "block", 32, 32),
    ("Convolutional_Depth_4", "convolutional", 4, None),
    ("Convolutional_Depth_8", "convolutional", 8, None),
    ("Convolutional_Depth_12", "convolutional", 12, None),
    ("Convolutional_Depth_16", "convolutional", 16, None),
    ("Diag_16x16_S1", "diagonal", DIAGONAL_ROWS, DIAGONAL_COLUMNS),
    (
        "PseudoRandom_256_Seed_26147",
        "pseudo_random",
        DEFAULT_PSEUDORANDOM_CONFIG.frame_length,
        DEFAULT_PSEUDORANDOM_CONFIG.seed,
    ),
)
_FEC_TYPES: tuple[str | None, ...] = (
    None,
    *CONVOLUTIONAL_CODE_SPECS.keys(),
    *RS_CODE_SPECS.keys(),
    CONCAT_FEC_NAME,
    *LDPC_CONFIGS.keys(),
)


@dataclass
class TrialResult:
    """One FEC/interleaver hypothesis and its decoded frame/payload."""

    fec_type: str | None
    interleaver_type: str | None
    interleaver_depth: int | None = None
    interleaver_branches: int | None = None
    decoded_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    frame_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    crc_pass: bool | None = None
    valid: bool = False
    structure_match: bool = False
    accepted: bool = False
    compatible: bool = True
    frame_length_compatible: bool = True
    codeword_count: int | None = None
    codeword_length_bits: int | None = None
    successful_fec_decodes: int | None = None
    failed_fec_decodes: int | None = None
    path_metric: int | None = None
    phase_rotation_rad: float | None = None
    llr_source: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TrialSearchResult:
    """Ranked candidate trials and the leading candidate's decoded payload."""

    ranked_trials: list[TrialResult] = field(default_factory=list)
    best_result: TrialResult | None = None
    fec_type: str | None = None
    interleaver_type: str | None = None
    decoded_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    crc_pass: bool | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
    trial_log: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _structural_preflight_error(
    received_bit_count: int,
    *,
    fec_type: str | None,
    interleaver_kind: str,
    rows: int | None,
    cols: int | None,
    expected_frame_length: int | None,
) -> tuple[str, str] | None:
    """Reject only candidates that cannot satisfy exact length structure.

    Interleavers in this engine preserve length. Checking their block shape and
    each code's required codeword length before materializing a deinterleaved
    copy avoids work on hypotheses that cannot possibly be accepted.
    """
    count = int(received_bit_count)
    if interleaver_kind == "block":
        assert rows is not None and cols is not None
        block_size = rows * cols
        if count == 0 or count % block_size:
            return "interleaver", (
                f"bit count must be a positive multiple of matrix size {block_size}"
            )
    elif interleaver_kind == "diagonal":
        if count == 0 or count % DIAGONAL_BLOCK_SIZE:
            return "interleaver", (
                "bit count must be a positive multiple of diagonal block size "
                f"{DIAGONAL_BLOCK_SIZE}"
            )
    elif interleaver_kind == "pseudo_random":
        frame_length = int(rows or DEFAULT_PSEUDORANDOM_CONFIG.frame_length)
        if count == 0 or count % frame_length:
            return "interleaver", (
                f"bit count must be a positive multiple of pseudorandom frame size {frame_length}"
            )

    if fec_type is None:
        if expected_frame_length is not None and count != expected_frame_length:
            return "frame", (
                f"no FEC requires exactly {expected_frame_length} frame bits; received {count}"
            )
    elif fec_type in CONVOLUTIONAL_CODE_SPECS:
        if count == 0 or count % 2:
            return "frame", (
                f"{fec_type} requires a positive even number of coded bits; received {count}"
            )
        if expected_frame_length is not None:
            spec = CONVOLUTIONAL_CODE_SPECS[fec_type]
            expected_coded = 2 * (expected_frame_length + spec.termination_tail_bits)
            if count != expected_coded:
                return "frame", (
                    f"{fec_type} requires exactly {expected_coded} coded bits "
                    f"for the expected frame; received {count}"
                )
    elif fec_type in RS_CODE_SPECS:
        spec = RS_CODE_SPECS[fec_type]
        codeword_bits = spec.n * 8
        if count == 0 or count % codeword_bits:
            return "frame", (
                f"{fec_type} requires a positive multiple of {codeword_bits} codeword bits; "
                f"received {count}"
            )
        if expected_frame_length is not None:
            decoded_length = (count // codeword_bits) * spec.k * 8
            if decoded_length != expected_frame_length:
                return "frame", (
                    f"{fec_type} produces {decoded_length} frame bits, not the expected "
                    f"{expected_frame_length}"
                )
    elif fec_type == CONCAT_FEC_NAME:
        if count == 0 or count % CONCAT_CODED_BITS:
            return "frame", (
                f"{CONCAT_FEC_NAME} requires a positive multiple of {CONCAT_CODED_BITS} "
                f"coded bits; received {count}"
            )
        if expected_frame_length is not None:
            decoded_length = (count // CONCAT_CODED_BITS) * CONCAT_PAYLOAD_BITS
            if decoded_length != expected_frame_length:
                return "frame", (
                    f"{CONCAT_FEC_NAME} produces {decoded_length} frame bits, not the expected "
                    f"{expected_frame_length}"
                )
    elif fec_type in LDPC_CONFIGS:
        spec = LDPC_CONFIGS[fec_type]
        if count == 0 or count % spec.N:
            return "frame", (
                f"{fec_type} requires a positive multiple of {spec.N} codeword bits; "
                f"received {count}"
            )
        if expected_frame_length is not None:
            decoded_length = (count // spec.N) * spec.K
            if decoded_length != expected_frame_length:
                return "frame", (
                    f"{fec_type} produces {decoded_length} frame bits, not the expected "
                    f"{expected_frame_length}"
                )
    return None


def _run_candidate(
    received: np.ndarray,
    *,
    fec_type: str | None,
    interleaver_type: str | None,
    interleaver_kind: str,
    rows: int | None,
    cols: int | None,
    depth: int | None,
    convolutional_branches: int,
    expected_payload_length: int | None,
    crc_present: bool,
    phase_rotation_rad: float | None,
    soft_llrs: np.ndarray | None,
) -> TrialResult:
    result = TrialResult(
        fec_type=fec_type,
        interleaver_type=interleaver_type,
        phase_rotation_rad=phase_rotation_rad,
        interleaver_depth=depth if interleaver_kind == "convolutional" else None,
        interleaver_branches=(
            convolutional_branches if interleaver_kind == "convolutional" else None
        ),
    )
    deinterleaved = received
    deinterleaved_llrs = soft_llrs
    diagonal_diagnostics: dict[str, object] = {}
    pseudo_random_diagnostics: dict[str, object] = {}
    if interleaver_kind == "diagonal":
        diagonal_diagnostics = {
            "interleaver_rows": DIAGONAL_ROWS,
            "interleaver_columns": DIAGONAL_COLUMNS,
            "interleaver_step": DIAGONAL_STEP,
            "interleaver_frame_length": int(received.size),
            "interleaver_complete_blocks": int(received.size // DIAGONAL_BLOCK_SIZE),
            "interleaver_block_remainder": int(received.size % DIAGONAL_BLOCK_SIZE),
            "downstream_fec": fec_type,
        }
    if interleaver_kind == "pseudo_random":
        frame_length = int(rows or DEFAULT_PSEUDORANDOM_CONFIG.frame_length)
        seed = int(cols if cols is not None else DEFAULT_PSEUDORANDOM_CONFIG.seed)
        pseudo_random_diagnostics = {
            "interleaver_seed": seed,
            "interleaver_configuration": {
                "frame_length": frame_length,
                "seed": seed,
                "rng": "SplitMix64",
                "permutation": "Durstenfeld descending Fisher-Yates",
            },
            "interleaver_frame_length": int(received.size),
            "interleaver_frame_size": frame_length,
            "interleaver_processed_frames": int(received.size // frame_length),
            "interleaver_frame_remainder": int(received.size % frame_length),
            "downstream_fec": fec_type,
        }
    expected_frame_length = (
        expected_payload_length + (16 if crc_present else 0)
        if expected_payload_length is not None else None
    )
    try:
        preflight = _structural_preflight_error(
            int(received.size),
            fec_type=fec_type,
            interleaver_kind=interleaver_kind,
            rows=rows,
            cols=cols,
            expected_frame_length=expected_frame_length,
        )
        if preflight is not None:
            category, reason = preflight
            if category == "frame":
                result.frame_length_compatible = False
            raise ValueError(reason)
        if interleaver_kind == "block":
            assert rows is not None and cols is not None
            deinterleaved = deinterleave_block(received, rows, cols)
        elif interleaver_kind == "convolutional":
            assert depth is not None
            deinterleaved = deinterleave_convolutional(
                received, depth=depth, branches=convolutional_branches
            )
        elif interleaver_kind == "diagonal":
            deinterleaved = deinterleave_diagonal(received)
        elif interleaver_kind == "pseudo_random":
            assert rows is not None
            configuration = PseudoRandomInterleaverConfig(
                frame_length=rows,
                seed=int(cols if cols is not None else DEFAULT_PSEUDORANDOM_CONFIG.seed),
            )
            pseudo_random = PseudoRandomInterleaver(configuration)
            deinterleaved = pseudo_random.deinterleave(received)
            if soft_llrs is not None:
                deinterleaved_llrs = pseudo_random.deinterleave_values(soft_llrs)
        else:
            deinterleaved = received.copy()
        if expected_frame_length is not None:
            expected_coded_length: int | None = None
            expected_decoded_length: int | None = None
            if fec_type is None:
                expected_decoded_length = expected_frame_length
            elif fec_type in CONVOLUTIONAL_CODE_SPECS:
                spec = CONVOLUTIONAL_CODE_SPECS[fec_type]
                expected_coded_length = 2 * (
                    expected_frame_length + spec.termination_tail_bits
                )
                expected_decoded_length = expected_frame_length
            elif fec_type in RS_CODE_SPECS:
                spec = RS_CODE_SPECS[fec_type]
                codeword_length_bits = spec.n * 8
                if deinterleaved.size == 0 or deinterleaved.size % codeword_length_bits:
                    result.frame_length_compatible = False
                    raise ValueError(
                        f"{fec_type} frame length must be a positive multiple of "
                        f"{codeword_length_bits} codeword bits; received {deinterleaved.size}"
                    )
                result.codeword_count = int(deinterleaved.size // codeword_length_bits)
                result.codeword_length_bits = codeword_length_bits
                expected_coded_length = result.codeword_count * codeword_length_bits
                expected_decoded_length = result.codeword_count * spec.k * 8
            elif fec_type == CONCAT_FEC_NAME:
                if deinterleaved.size == 0 or deinterleaved.size % CONCAT_CODED_BITS:
                    result.frame_length_compatible = False
                    raise ValueError(
                        f"{CONCAT_FEC_NAME} frame length must be a positive multiple of "
                        f"{CONCAT_CODED_BITS} coded bits; received {deinterleaved.size}"
                    )
                result.codeword_count = int(deinterleaved.size // CONCAT_CODED_BITS)
                result.codeword_length_bits = CONCAT_CODED_BITS
                expected_coded_length = result.codeword_count * CONCAT_CODED_BITS
                expected_decoded_length = result.codeword_count * CONCAT_PAYLOAD_BITS
            elif fec_type in LDPC_CONFIGS:
                spec = LDPC_CONFIGS[fec_type]
                if deinterleaved.size == 0 or deinterleaved.size % spec.N:
                    result.frame_length_compatible = False
                    raise ValueError(
                        f"{fec_type} frame length must be a positive multiple of "
                        f"{spec.N} codeword bits; received {deinterleaved.size}"
                    )
                result.codeword_count = int(deinterleaved.size // spec.N)
                result.codeword_length_bits = spec.N
                expected_coded_length = result.codeword_count * spec.N
                expected_decoded_length = result.codeword_count * spec.K
            if (
                expected_coded_length is not None
                and deinterleaved.size != expected_coded_length
            ):
                result.frame_length_compatible = False
                raise ValueError(
                    f"{fec_type} requires exactly {expected_coded_length} coded bits "
                    f"for the expected frame; received {deinterleaved.size}"
                )
            if (
                expected_decoded_length is not None
                and expected_decoded_length != expected_frame_length
            ):
                result.frame_length_compatible = False
                raise ValueError(
                    f"{fec_type or 'no FEC'} produces {expected_decoded_length} frame bits, "
                    f"not the expected {expected_frame_length}"
                )
        if fec_type in CONVOLUTIONAL_CODE_SPECS:
            decoded = decode_viterbi(deinterleaved, fec_type, terminated=True)
            frame_bits = decoded.decoded_bits
            result.path_metric = decoded.path_metric
            result.valid = decoded.valid and decoded.terminated
            result.diagnostics["fec_diagnostics"] = decoded.diagnostics
            result.warnings.extend(decoded.warnings)
        elif fec_type in RS_CODE_SPECS:
            spec = RS_CODE_SPECS[fec_type]
            expected_codeword_bits = spec.n * 8
            if deinterleaved.size == 0 or deinterleaved.size % expected_codeword_bits:
                result.frame_length_compatible = False
                raise ValueError(
                    f"{fec_type} requires a positive multiple of {expected_codeword_bits} codeword bits; "
                    f"received {deinterleaved.size}"
                )
            codeword_count = int(deinterleaved.size // expected_codeword_bits)
            result.codeword_count = codeword_count
            result.codeword_length_bits = expected_codeword_bits
            decoded_frames: list[np.ndarray] = []
            codeword_diagnostics: list[dict[str, object]] = []
            for codeword_index in range(codeword_count):
                start = codeword_index * expected_codeword_bits
                stop = start + expected_codeword_bits
                codeword_bytes = bits_to_bytes(deinterleaved[start:stop])
                decoded_rs = rs_decode(codeword_bytes, fec_type)
                success = bool(decoded_rs.decode_success and decoded_rs.valid)
                if success:
                    decoded_frames.append(bytes_to_bits(decoded_rs.decoded_bytes))
                codeword_diagnostics.append({
                    "index": codeword_index,
                    "input_length_bits": expected_codeword_bits,
                    "input_length_bytes": len(codeword_bytes),
                    "decode_success": decoded_rs.decode_success,
                    "valid": decoded_rs.valid,
                    "corrected_error_count": decoded_rs.corrected_error_count,
                    "error_message": decoded_rs.error_message,
                })
            result.successful_fec_decodes = sum(
                bool(item["decode_success"] and item["valid"])
                for item in codeword_diagnostics
            )
            result.failed_fec_decodes = codeword_count - result.successful_fec_decodes
            result.valid = result.failed_fec_decodes == 0
            frame_bits = (
                np.concatenate(decoded_frames).astype(np.uint8, copy=False)
                if result.valid else np.empty(0, dtype=np.uint8)
            )
            result.diagnostics["rs_batch"] = {
                "code": fec_type,
                "codeword_count": codeword_count,
                "codeword_length_bits": expected_codeword_bits,
                "codeword_length_bytes": spec.n,
                "total_frame_length": int(received.size),
                "deinterleaved_frame_length": int(deinterleaved.size),
                "interleaver": interleaver_type,
                "successful_fec_decodes": result.successful_fec_decodes,
                "failed_fec_decodes": result.failed_fec_decodes,
                "codewords": codeword_diagnostics,
            }
        elif fec_type == CONCAT_FEC_NAME:
            if deinterleaved.size == 0 or deinterleaved.size % CONCAT_CODED_BITS:
                result.frame_length_compatible = False
                raise ValueError(
                    f"{CONCAT_FEC_NAME} requires a positive multiple of {CONCAT_CODED_BITS} coded bits; "
                    f"received {deinterleaved.size}"
                )
            codeword_count = int(deinterleaved.size // CONCAT_CODED_BITS)
            result.codeword_count = codeword_count
            result.codeword_length_bits = CONCAT_CODED_BITS
            decoded_frames = []
            codeword_diagnostics = []
            path_metrics: list[int] = []
            single_concatenated_diagnostics: dict[str, object] | None = None
            for codeword_index in range(codeword_count):
                start = codeword_index * CONCAT_CODED_BITS
                stop = start + CONCAT_CODED_BITS
                decoded_concat = decode_concatenated(deinterleaved[start:stop])
                if decoded_concat.final_valid:
                    decoded_frames.append(decoded_concat.decoded_bits.copy())
                if decoded_concat.viterbi_path_metric is not None:
                    path_metrics.append(decoded_concat.viterbi_path_metric)
                codeword_diagnostics.append({
                    "index": codeword_index,
                    "input_length_bits": CONCAT_CODED_BITS,
                    "viterbi_success": decoded_concat.viterbi_success,
                    "rs_success": decoded_concat.rs_success,
                    "final_valid": decoded_concat.final_valid,
                    "viterbi_path_metric": decoded_concat.viterbi_path_metric,
                    "rs_corrected_errors": decoded_concat.rs_corrected_errors,
                    "error_message": decoded_concat.error_message,
                })
                if codeword_count == 1:
                    single_concatenated_diagnostics = {
                        "fec_name": decoded_concat.fec_name,
                        "inner_fec": decoded_concat.inner_fec,
                        "outer_fec": decoded_concat.outer_fec,
                        "viterbi_success": decoded_concat.viterbi_success,
                        "rs_success": decoded_concat.rs_success,
                        "viterbi_path_metric": decoded_concat.viterbi_path_metric,
                        "rs_corrected_errors": decoded_concat.rs_corrected_errors,
                        "final_valid": decoded_concat.final_valid,
                        "decoded_bit_length": decoded_concat.decoded_bit_length,
                        "decoded_byte_length": decoded_concat.decoded_byte_length,
                        "error_message": decoded_concat.error_message,
                        "diagnostics": decoded_concat.diagnostics,
                    }
            result.successful_fec_decodes = sum(
                bool(item["final_valid"]) for item in codeword_diagnostics
            )
            result.failed_fec_decodes = codeword_count - result.successful_fec_decodes
            result.valid = result.failed_fec_decodes == 0
            result.path_metric = (
                sum(path_metrics) if len(path_metrics) == codeword_count else None
            )
            frame_bits = (
                np.concatenate(decoded_frames).astype(np.uint8, copy=False)
                if result.valid else np.empty(0, dtype=np.uint8)
            )
            result.diagnostics["concatenated_batch"] = {
                "fec_name": CONCAT_FEC_NAME,
                "inner_fec": "Conv_R12_K7",
                "outer_fec": "RS_255_223",
                "codeword_count": codeword_count,
                "codeword_length_bits": CONCAT_CODED_BITS,
                "total_frame_length": int(received.size),
                "deinterleaved_frame_length": int(deinterleaved.size),
                "interleaver": interleaver_type,
                "viterbi_successful_codewords": sum(
                    bool(item["viterbi_success"]) for item in codeword_diagnostics
                ),
                "rs_successful_codewords": sum(
                    bool(item["rs_success"]) for item in codeword_diagnostics
                ),
                "successful_fec_decodes": result.successful_fec_decodes,
                "failed_fec_decodes": result.failed_fec_decodes,
                "codewords": codeword_diagnostics,
            }
            if single_concatenated_diagnostics is not None:
                # Preserve the Step 81 single-codeword diagnostics key.
                result.diagnostics["concatenated_diagnostics"] = single_concatenated_diagnostics
        elif fec_type in LDPC_CONFIGS:
            spec = LDPC_CONFIGS[fec_type]
            if expected_frame_length is None:
                raise ValueError(
                    f"{fec_type} requires expected_payload_length to evaluate this LDPC hypothesis"
                )
            if deinterleaved.size == 0 or deinterleaved.size % spec.N:
                result.frame_length_compatible = False
                raise ValueError(
                    f"{fec_type} requires a positive multiple of {spec.N} codeword bits; "
                    f"received {deinterleaved.size}"
                )
            result.codeword_count = int(deinterleaved.size // spec.N)
            result.codeword_length_bits = spec.N
            if deinterleaved_llrs is None:
                # Preserve the legacy hard-bit adapter for callers that do not
                # have calibrated demodulator soft information.
                candidate_llrs = np.where(deinterleaved == 0, 8.0, -8.0)
                result.llr_source = "legacy hard-bit adapter: 0 -> +8.0, 1 -> -8.0"
            else:
                candidate_llrs = deinterleaved_llrs
                result.llr_source = "caller-supplied demodulator soft LLRs"
            decoded_ldpc = decode_ldpc_frame(candidate_llrs, spec.name)
            result.successful_fec_decodes = decoded_ldpc.successful_codewords
            result.failed_fec_decodes = decoded_ldpc.failed_codewords
            result.valid = decoded_ldpc.success
            frame_bits = (
                decoded_ldpc.decoded_bits.copy()
                if decoded_ldpc.success else np.empty(0, dtype=np.uint8)
            )
            codeword_diagnostics = [
                {
                    "index": index,
                    "input_length_bits": spec.N,
                    "syndrome_pass": word.syndrome_pass,
                    "syndrome_weight": word.syndrome_weight,
                    "iterations_used": word.iterations_used,
                    "failure_reason": word.failure_reason,
                }
                for index, word in enumerate(decoded_ldpc.codeword_results)
            ]
            result.diagnostics["ldpc_batch"] = {
                "fec_name": "LDPC",
                "configuration": spec.name,
                "N": spec.N,
                "K": spec.K,
                "rate": spec.rate,
                "Z": spec.Z,
                "matrix_identifier": spec.matrix_identifier,
                "decoder_algorithm": LDPC_DECODER_ALGORITHM,
                "llr_input": result.llr_source,
                "codeword_count": decoded_ldpc.codeword_count,
                "successful_fec_decodes": decoded_ldpc.successful_codewords,
                "failed_fec_decodes": decoded_ldpc.failed_codewords,
                "syndrome_status": [word.syndrome_pass for word in decoded_ldpc.codeword_results],
                "codewords": codeword_diagnostics,
                "total_frame_length": int(received.size),
                "deinterleaved_frame_length": int(deinterleaved.size),
                "interleaver": interleaver_type,
            }
        else:
            frame_bits = deinterleaved
            result.valid = True

        result.frame_bits = frame_bits.copy()
        if crc_present and result.valid:
            result.crc_pass = verify_crc16(frame_bits)
            result.decoded_bits = frame_bits[:-16].copy() if frame_bits.size >= 16 else np.empty(0, dtype=np.uint8)
        elif crc_present:
            result.crc_pass = False
            result.decoded_bits = frame_bits[:-16].copy() if frame_bits.size >= 16 else np.empty(0, dtype=np.uint8)
        else:
            result.crc_pass = None
            result.decoded_bits = frame_bits.copy()

        result.structure_match = (
            expected_frame_length is None or frame_bits.size == expected_frame_length
        )
        result.accepted = bool(
            result.valid
            and result.structure_match
            and (result.crc_pass if crc_present else True)
        )
        result.diagnostics.update({
            "received_bit_count": int(received.size),
            "deinterleaved_bit_count": int(deinterleaved.size),
            "decoded_frame_bit_count": int(frame_bits.size),
            "decoded_payload_bit_count": int(result.decoded_bits.size),
            "expected_payload_length": expected_payload_length,
            "expected_frame_length": expected_frame_length,
            "structure_match": result.structure_match,
            "crc_present": crc_present,
            "crc_pass": result.crc_pass,
            "phase_rotation_rad_supplied": phase_rotation_rad,
            "interleaver_depth": result.interleaver_depth,
            "interleaver_branches": result.interleaver_branches,
            "interleaver_compatible": result.compatible,
            "frame_length_compatible": result.frame_length_compatible,
            "total_frame_length": int(received.size),
            "interleaver": interleaver_type,
            "codeword_count": result.codeword_count,
            "codeword_length_bits": result.codeword_length_bits,
            "successful_fec_decodes": result.successful_fec_decodes,
            "failed_fec_decodes": result.failed_fec_decodes,
            "llr_source": result.llr_source,
            "crc_status": result.crc_pass,
            **pseudo_random_diagnostics,
            **diagonal_diagnostics,
        })
        if interleaver_kind == "diagonal":
            result.diagnostics.update({
                "downstream_fec_valid": result.valid,
                "downstream_fec_result": "valid" if result.valid else "invalid",
                "candidate_status": "accepted" if result.accepted else "failed",
            })
        elif interleaver_kind == "pseudo_random":
            result.diagnostics.update({
                "downstream_fec_valid": result.valid,
                "downstream_fec_result": "valid" if result.valid else "invalid",
                "candidate_status": "accepted" if result.accepted else "failed",
            })
        for batch_key in ("rs_batch", "concatenated_batch", "ldpc_batch"):
            if batch_key in result.diagnostics:
                result.diagnostics[batch_key]["crc_present"] = bool(crc_present)
                result.diagnostics[batch_key]["crc_status"] = result.crc_pass
        if not result.structure_match:
            result.warnings.append("Decoded frame length did not match the expected structure")
        if crc_present and not result.crc_pass:
            result.warnings.append("CRC-16 validation failed for this candidate")
    except (TypeError, ValueError, RuntimeError) as exc:
        result.warnings.append(str(exc))
        result.valid = False
        result.structure_match = False
        result.accepted = False
        incompatible_block_length = (
            interleaver_kind == "block"
            and "multiple of matrix size" in str(exc)
        )
        incompatible_diagonal_length = (
            interleaver_kind == "diagonal"
            and (
                received.size == 0
                or received.size % DIAGONAL_BLOCK_SIZE != 0
            )
        )
        incompatible_pseudo_random_length = (
            interleaver_kind == "pseudo_random"
            and (
                received.size == 0
                or received.size % int(rows or DEFAULT_PSEUDORANDOM_CONFIG.frame_length) != 0
            )
        )
        incompatible_fec_length = (
            result.frame_length_compatible is False
            or (
                fec_type in RS_CODE_SPECS
                and (
                    deinterleaved.size == 0
                    or deinterleaved.size % (RS_CODE_SPECS[fec_type].n * 8) != 0
                )
            )
            or (
                fec_type == CONCAT_FEC_NAME
                and (deinterleaved.size == 0 or deinterleaved.size % CONCAT_CODED_BITS != 0)
            )
            or (
                fec_type in LDPC_CONFIGS
                and (
                    deinterleaved.size == 0
                    or deinterleaved.size % LDPC_CONFIGS[fec_type].N != 0
                )
            )
        )
        if (
            incompatible_block_length
            or incompatible_diagonal_length
            or incompatible_pseudo_random_length
            or incompatible_fec_length
        ):
            result.compatible = False
        if incompatible_block_length:
            result.frame_length_compatible = False
        if incompatible_diagonal_length:
            result.frame_length_compatible = False
        if incompatible_pseudo_random_length:
            result.frame_length_compatible = False
        if incompatible_fec_length:
            result.frame_length_compatible = False
        result.crc_pass = False if crc_present else None
        result.diagnostics.update({
            "received_bit_count": int(received.size),
            "expected_payload_length": expected_payload_length,
            "crc_present": crc_present,
            "phase_rotation_rad_supplied": phase_rotation_rad,
            "interleaver_depth": result.interleaver_depth,
            "interleaver_branches": result.interleaver_branches,
            "candidate_error": str(exc),
            "interleaver_compatible": result.compatible,
            "frame_length_compatible": result.frame_length_compatible,
            "total_frame_length": int(received.size),
            "interleaver": interleaver_type,
            **pseudo_random_diagnostics,
            "llr_source": result.llr_source,
            "codeword_count": result.codeword_count,
            "codeword_length_bits": result.codeword_length_bits,
            "successful_fec_decodes": result.successful_fec_decodes,
            "failed_fec_decodes": result.failed_fec_decodes,
            "crc_status": result.crc_pass,
            **diagonal_diagnostics,
            **({
                "downstream_fec_valid": False,
                "downstream_fec_result": "not_run",
            } if interleaver_kind in ("diagonal", "pseudo_random") else {}),
            "candidate_status": (
                "incompatible"
                if (
                    incompatible_block_length
                    or incompatible_diagonal_length
                    or incompatible_pseudo_random_length
                    or incompatible_fec_length
                )
                else "failed"
            ),
        })
    return result


def run_fec_trials(
    hard_bits: np.ndarray,
    *,
    expected_payload_length: int | None = None,
    crc_present: bool = True,
    soft_llrs: np.ndarray | None = None,
    phase_rotation_rad: float | None = None,
    convolutional_branches: int = 4,
) -> TrialSearchResult:
    """Evaluate the current FEC/interleaver product and report its exact size.

    Input is the hard-bit stream after demodulation. The optional phase value
    records a correction selected by an upstream synchronizer; phase is not
    changed here because hard bits no longer contain carrier-phase information.
    Convolutional interleaver trials use the explicit
    ``convolutional_branches`` argument (default 4 for compatibility with
    Step 79 calls); the selected value is recorded in each result.
    Candidates deinterleave first, decode second, compare frame length, and then
    validate CRC. Incompatible block lengths are retained as failed trial-log
    entries; they are never truncated or padded. CRC-pass candidates rank ahead
    of every CRC-fail candidate.
    With ``crc_present=False``, ranking is based only on validity, structure,
    and path metric, so the result cannot provide CRC-level acceptance evidence.
    """
    received = validate_binary_bits(hard_bits)
    if soft_llrs is not None:
        soft_values = np.asarray(soft_llrs, dtype=np.float64)
        if soft_values.ndim != 1 or soft_values.size != received.size:
            raise ValueError("soft_llrs must be a one-dimensional array matching hard_bits length")
        if not np.all(np.isfinite(soft_values)):
            raise ValueError("soft_llrs must contain only finite values")
        soft_llrs = soft_values
    if not isinstance(crc_present, (bool, np.bool_)):
        raise TypeError("crc_present must be a bool")
    if (
        not isinstance(convolutional_branches, (int, np.integer))
        or isinstance(convolutional_branches, bool)
        or convolutional_branches < 2
    ):
        raise ValueError("convolutional_branches must be an explicit integer of at least two")
    if expected_payload_length is not None and (
        not isinstance(expected_payload_length, (int, np.integer))
        or isinstance(expected_payload_length, bool)
        or expected_payload_length < 0
    ):
        raise ValueError("expected_payload_length must be a nonnegative integer or None")
    if phase_rotation_rad is not None and not np.isfinite(phase_rotation_rad):
        raise ValueError("phase_rotation_rad must be finite or None")

    trials: list[TrialResult] = []
    structurally_skipped_pseudo_random: dict[str, int] = {}
    for fec_type in _FEC_TYPES:
        for interleaver_type, interleaver_kind, first, second in _INTERLEAVERS:
            # The current LDPC project slice specifies the nine Annex R
            # configurations only, without an LDPC interleaver hypothesis.
            # Keep the bounded search at one no-interleaver candidate per code.
            if fec_type in LDPC_CONFIGS and interleaver_type is not None:
                continue
            if interleaver_kind == "pseudo_random":
                frame_length = int(first or DEFAULT_PSEUDORANDOM_CONFIG.frame_length)
                coded_length_ok = received.size > 0 and received.size % frame_length == 0
                expected_frame_length = (
                    int(expected_payload_length) + (16 if crc_present else 0)
                    if expected_payload_length is not None else None
                )
                if coded_length_ok and fec_type is None and expected_frame_length is not None:
                    coded_length_ok = received.size == expected_frame_length
                elif coded_length_ok and fec_type in CONVOLUTIONAL_CODE_SPECS:
                    if received.size % 2:
                        coded_length_ok = False
                    if expected_frame_length is not None:
                        spec = CONVOLUTIONAL_CODE_SPECS[fec_type]
                        coded_length_ok = received.size == 2 * (
                            expected_frame_length + spec.termination_tail_bits
                        )
                elif coded_length_ok and fec_type in RS_CODE_SPECS:
                    spec = RS_CODE_SPECS[fec_type]
                    codeword_bits = spec.n * 8
                    coded_length_ok = received.size % codeword_bits == 0
                    if coded_length_ok and expected_frame_length is not None:
                        coded_length_ok = (
                            received.size // codeword_bits * spec.k * 8
                            == expected_frame_length
                        )
                elif coded_length_ok and fec_type == CONCAT_FEC_NAME:
                    coded_length_ok = received.size % CONCAT_CODED_BITS == 0
                    if coded_length_ok and expected_frame_length is not None:
                        coded_length_ok = (
                            received.size // CONCAT_CODED_BITS * CONCAT_PAYLOAD_BITS
                            == expected_frame_length
                        )
                elif coded_length_ok and fec_type in LDPC_CONFIGS:
                    spec = LDPC_CONFIGS[fec_type]
                    coded_length_ok = received.size % spec.N == 0
                    if coded_length_ok and expected_frame_length is not None:
                        coded_length_ok = received.size // spec.N * spec.K == expected_frame_length
                if not coded_length_ok:
                    key = str(fec_type or "none")
                    structurally_skipped_pseudo_random[key] = (
                        structurally_skipped_pseudo_random.get(key, 0) + 1
                    )
                    continue
            trials.append(_run_candidate(
                received,
                fec_type=fec_type,
                interleaver_type=interleaver_type,
                interleaver_kind=interleaver_kind,
                rows=first if interleaver_kind in ("block", "pseudo_random") else None,
                cols=second if interleaver_kind in ("block", "pseudo_random") else None,
                depth=first if interleaver_kind == "convolutional" else None,
                convolutional_branches=int(convolutional_branches),
                expected_payload_length=(
                    int(expected_payload_length) if expected_payload_length is not None else None
                ),
                crc_present=bool(crc_present),
                phase_rotation_rad=(float(phase_rotation_rad) if phase_rotation_rad is not None else None),
                soft_llrs=soft_llrs,
            ))

    def rank_key(item: tuple[int, TrialResult]) -> tuple[int, int, int, int, float, int]:
        index, trial = item
        normalized_metric = (
            trial.path_metric / max(received.size, 1)
            if trial.path_metric is not None else 0.0
        )
        return (
            0 if trial.accepted else 1,
            0 if trial.crc_pass is True else 1,
            0 if trial.structure_match else 1,
            0 if trial.valid else 1,
            normalized_metric,
            index,
        )

    ranked = [trial for _, trial in sorted(enumerate(trials), key=rank_key)]
    best = ranked[0] if ranked else None
    total_candidate_count = (
        len(_FEC_TYPES) * len(_INTERLEAVERS)
        - len(LDPC_CONFIGS) * (len(_INTERLEAVERS) - 1)
    )
    candidates_after_structural_pruning = sum(
        trial.diagnostics.get("candidate_status") != "incompatible"
        for trial in ranked
    )
    trial_log = [
        {
            "fec_type": trial.fec_type,
            "interleaver_type": trial.interleaver_type,
            "interleaver_depth": trial.interleaver_depth,
            "interleaver_branches": trial.interleaver_branches,
            "interleaver_rows": trial.diagnostics.get("interleaver_rows"),
            "interleaver_columns": trial.diagnostics.get("interleaver_columns"),
            "interleaver_step": trial.diagnostics.get("interleaver_step"),
            "interleaver_seed": trial.diagnostics.get("interleaver_seed"),
            "interleaver_configuration": trial.diagnostics.get("interleaver_configuration"),
            "interleaver_frame_size": trial.diagnostics.get("interleaver_frame_size"),
            "interleaver_processed_frames": trial.diagnostics.get("interleaver_processed_frames"),
            "interleaver_frame_length": trial.diagnostics.get("interleaver_frame_length"),
            "interleaver_complete_blocks": trial.diagnostics.get("interleaver_complete_blocks"),
            "interleaver_block_remainder": trial.diagnostics.get("interleaver_block_remainder"),
            "downstream_fec": trial.diagnostics.get("downstream_fec"),
            "downstream_fec_result": trial.diagnostics.get("downstream_fec_result"),
            "downstream_fec_valid": trial.diagnostics.get("downstream_fec_valid"),
            "compatible": trial.compatible,
            "frame_length_compatible": trial.frame_length_compatible,
            "total_frame_length": trial.diagnostics.get("total_frame_length"),
            "interleaver": trial.interleaver_type,
            "codeword_count": trial.codeword_count,
            "codeword_length_bits": trial.codeword_length_bits,
            "successful_fec_decodes": trial.successful_fec_decodes,
            "failed_fec_decodes": trial.failed_fec_decodes,
            "crc_status": trial.crc_pass,
            "candidate_status": trial.diagnostics.get("candidate_status", "evaluated"),
            "accepted": trial.accepted,
            "valid": trial.valid,
            "structure_match": trial.structure_match,
            "crc_pass": trial.crc_pass,
            "llr_source": trial.llr_source,
            "path_metric": trial.path_metric,
            "warnings": list(trial.warnings),
        }
        for trial in ranked
    ]
    warnings: list[str] = []
    if best is None or not best.accepted:
        warnings.append("No candidate passed the configured validity, structure, and CRC checks")
    if not crc_present:
        warnings.append("CRC is disabled; candidate ranking is not protected by a CRC check")
    return TrialSearchResult(
        ranked_trials=ranked,
        best_result=best,
        fec_type=best.fec_type if best is not None else None,
        interleaver_type=best.interleaver_type if best is not None else None,
        decoded_bits=best.decoded_bits.copy() if best is not None else np.empty(0, dtype=np.uint8),
        crc_pass=best.crc_pass if best is not None else None,
        diagnostics={
            "candidate_count": len(ranked),
            "total_candidate_count": total_candidate_count,
            "candidate_count_after_structural_pruning": candidates_after_structural_pruning,
            "structurally_pruned_candidate_count": (
                total_candidate_count - candidates_after_structural_pruning
            ),
            "fec_hypothesis_count": len(_FEC_TYPES),
            "interleaver_hypothesis_count": len(_INTERLEAVERS),
            "accepted_candidate_count": sum(candidate.accepted for candidate in ranked),
            "crc_present": bool(crc_present),
            "expected_payload_length": expected_payload_length,
            "phase_rotation_rad_supplied": phase_rotation_rad,
            "convolutional_interleaver_branches": int(convolutional_branches),
            "best_candidate_accepted": best.accepted if best is not None else False,
            "best_codeword_count": best.codeword_count if best is not None else None,
            "best_codeword_length_bits": best.codeword_length_bits if best is not None else None,
            "best_total_frame_length": (
                best.diagnostics.get("total_frame_length") if best is not None else None
            ),
            "best_successful_fec_decodes": (
                best.successful_fec_decodes if best is not None else None
            ),
            "best_failed_fec_decodes": (
                best.failed_fec_decodes if best is not None else None
            ),
            "soft_llrs_supplied": soft_llrs is not None,
            "pseudo_random_structurally_skipped_candidates": structurally_skipped_pseudo_random,
        },
        trial_log=trial_log,
        warnings=warnings,
    )
