"""The fixed RS(255,223) outer + terminated Conv_R12_K7 inner code.

The on-air order is payload bits -> 223 payload bytes -> RS(255,223) ->
2040 MSB-first codeword bits -> terminated Conv_R12_K7 -> coded bits. Decoding
performs those operations in reverse. The exact payload, RS-word, and coded
stream lengths are enforced; this module does not pad, truncate, or shorten.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.bitstream.extractor import bits_to_bytes, bytes_to_bits, validate_binary_bits
from core.fec.convolutional import decode_viterbi, encode_convolutional
from core.fec.reed_solomon import ReedSolomonDecodeResult, rs_decode, rs_encode


CONCAT_FEC_NAME = "Concat_RS223_Conv7"
CONCAT_INNER_FEC = "Conv_R12_K7"
CONCAT_OUTER_FEC = "RS_255_223"
CONCAT_PAYLOAD_BITS = 223 * 8
CONCAT_RS_CODEWORD_BITS = 255 * 8
CONCAT_CODED_BITS = 2 * (CONCAT_RS_CODEWORD_BITS + 6)


@dataclass
class ConcatenatedDecodeResult:
    """Per-layer diagnostics for one concatenated-code frame."""

    fec_name: str = CONCAT_FEC_NAME
    inner_fec: str = CONCAT_INNER_FEC
    outer_fec: str = CONCAT_OUTER_FEC
    viterbi_success: bool = False
    rs_success: bool = False
    viterbi_path_metric: int | None = None
    rs_corrected_errors: int | None = None
    final_valid: bool = False
    decoded_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    decoded_bytes: bytes = b""
    input_length: int = 0
    decoded_bit_length: int = 0
    decoded_byte_length: int = 0
    diagnostics: dict[str, object] = field(default_factory=dict)
    error_message: str | None = None


def encode_concatenated(payload_bits: np.ndarray) -> np.ndarray:
    """Encode exactly 223*8 payload bits through RS(255,223), then Conv K7."""
    payload = validate_binary_bits(payload_bits)
    if payload.size != CONCAT_PAYLOAD_BITS:
        raise ValueError(
            f"{CONCAT_FEC_NAME} requires exactly {CONCAT_PAYLOAD_BITS} payload bits; "
            f"received {payload.size}"
        )
    payload_bytes = bits_to_bytes(payload)
    rs_codeword = rs_encode(payload_bytes, CONCAT_OUTER_FEC)
    rs_bits = bytes_to_bits(rs_codeword)
    if rs_bits.size != CONCAT_RS_CODEWORD_BITS:
        raise RuntimeError("RS encoder did not produce exactly 2040 codeword bits")
    coded = encode_convolutional(rs_bits, CONCAT_INNER_FEC, terminated=True)
    if coded.size != CONCAT_CODED_BITS:
        raise RuntimeError("Conv_R12_K7 encoder returned an unexpected concatenated frame length")
    return coded


def decode_concatenated(received_coded_bits: np.ndarray) -> ConcatenatedDecodeResult:
    """Decode exactly one 4092-bit Conv K7 stream followed by the RS layer."""
    received = validate_binary_bits(received_coded_bits)
    if received.size != CONCAT_CODED_BITS:
        raise ValueError(
            f"{CONCAT_FEC_NAME} requires exactly {CONCAT_CODED_BITS} coded bits; "
            f"received {received.size}"
        )

    result = ConcatenatedDecodeResult(input_length=int(received.size))
    try:
        viterbi = decode_viterbi(received, CONCAT_INNER_FEC, terminated=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        result.error_message = str(exc)
        result.diagnostics.update({"viterbi_success": False, "viterbi_error": str(exc)})
        return result

    result.viterbi_path_metric = int(viterbi.path_metric)
    result.viterbi_success = bool(viterbi.valid and viterbi.terminated)
    result.diagnostics["viterbi"] = dict(viterbi.diagnostics)
    result.diagnostics["viterbi_warnings"] = list(viterbi.warnings)
    if not result.viterbi_success:
        result.error_message = "Conv_R12_K7 Viterbi decode did not produce a valid terminated path"
        result.diagnostics["viterbi_success"] = False
        return result

    decoded_rs_bits = validate_binary_bits(viterbi.decoded_bits)
    result.diagnostics["viterbi_decoded_bit_count"] = int(decoded_rs_bits.size)
    if decoded_rs_bits.size != CONCAT_RS_CODEWORD_BITS:
        result.error_message = (
            f"Viterbi output must contain exactly {CONCAT_RS_CODEWORD_BITS} RS codeword bits; "
            f"received {decoded_rs_bits.size}"
        )
        result.diagnostics["rs_success"] = False
        return result

    # The shared conversion is MSB-first and rejects non-byte-aligned input.
    rs_codeword_bytes = bits_to_bytes(decoded_rs_bits)
    rs_result: ReedSolomonDecodeResult = rs_decode(rs_codeword_bytes, CONCAT_OUTER_FEC)
    result.rs_success = bool(rs_result.decode_success and rs_result.valid)
    result.rs_corrected_errors = rs_result.corrected_error_count
    result.diagnostics["rs"] = {
        "code": rs_result.code,
        "n": rs_result.n,
        "k": rs_result.k,
        "parity_symbols": rs_result.parity_symbols,
        "correction_capability": rs_result.correction_capability,
        "decode_success": rs_result.decode_success,
        "valid": rs_result.valid,
        "corrected_error_count": rs_result.corrected_error_count,
        "input_length": rs_result.input_length,
        "output_length": rs_result.output_length,
        "error_message": rs_result.error_message,
        "details": rs_result.diagnostics,
    }
    if rs_result.decode_success:
        result.decoded_bytes = rs_result.decoded_bytes
        result.decoded_byte_length = len(result.decoded_bytes)
        result.decoded_bits = bytes_to_bits(result.decoded_bytes)
        result.decoded_bit_length = int(result.decoded_bits.size)
    result.final_valid = bool(result.viterbi_success and result.rs_success)
    if not result.rs_success:
        result.error_message = rs_result.error_message or "RS(255,223) decode failed structural validity checks"
    result.diagnostics.update({
        "viterbi_success": result.viterbi_success,
        "rs_success": result.rs_success,
        "final_valid": result.final_valid,
        "rs_codeword_bit_count": int(decoded_rs_bits.size),
        "rs_codeword_byte_count": len(rs_codeword_bytes),
    })
    return result
