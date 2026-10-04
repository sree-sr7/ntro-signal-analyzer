import numpy as np
import pytest

from core.bitstream.crc import append_crc16
from core.bitstream.extractor import bits_to_bytes, bytes_to_bits
from core.fec.concatenated import CONCAT_CODED_BITS, CONCAT_PAYLOAD_BITS, encode_concatenated
from core.fec.convolutional import CONVOLUTIONAL_CODE_SPECS, encode_convolutional
from core.fec.reed_solomon import RS_CODE_SPECS, rs_encode
from core.fec.trial_engine import run_fec_trials
from core.interleaver.block import deinterleave_block, interleave_block
from core.interleaver.convolutional import interleave_convolutional
from core.interleaver.diagonal import interleave as interleave_diagonal
from core.interleaver.pseudorandom import (
    DEFAULT_PSEUDORANDOM_CONFIG,
    PseudoRandomInterleaver,
)


def _encoded_frame(
    payload: np.ndarray,
    *,
    code: str = "Conv_R12_K7",
    interleaver: tuple[str, int, int | None] | None = None,
) -> np.ndarray:
    frame = append_crc16(payload)
    encoded = encode_convolutional(frame, code)
    if interleaver is None:
        return encoded
    kind, first, second = interleaver
    if kind == "block":
        assert second is not None
        return interleave_block(encoded, first, second)
    assert kind == "convolutional" and second is not None
    return interleave_convolutional(encoded, depth=first, branches=second)


def _rs_encoded_frame(payload: np.ndarray, code: str) -> np.ndarray:
    frame = append_crc16(payload)
    assert frame.size == RS_CODE_SPECS[code].k * 8
    return bytes_to_bits(rs_encode(bits_to_bytes(frame), code))


def _rs_multi_encoded_frame(payload: np.ndarray, code: str, codeword_count: int) -> np.ndarray:
    frame = append_crc16(payload)
    codeword_payload_length = RS_CODE_SPECS[code].k * 8
    assert frame.size == codeword_count * codeword_payload_length
    codewords = [
        bytes_to_bits(rs_encode(
            bits_to_bytes(frame[start:start + codeword_payload_length]), code
        ))
        for start in range(0, frame.size, codeword_payload_length)
    ]
    return np.concatenate(codewords).astype(np.uint8, copy=False)


def _concat_multi_encoded_frame(payload: np.ndarray, codeword_count: int) -> np.ndarray:
    frame = append_crc16(payload)
    assert frame.size == codeword_count * CONCAT_PAYLOAD_BITS
    codewords = [
        encode_concatenated(frame[start:start + CONCAT_PAYLOAD_BITS])
        for start in range(0, frame.size, CONCAT_PAYLOAD_BITS)
    ]
    return np.concatenate(codewords).astype(np.uint8, copy=False)


@pytest.mark.parametrize("code", CONVOLUTIONAL_CODE_SPECS)
def test_trial_engine_identifies_each_supported_code_without_interleaving(code):
    rng = np.random.default_rng(7001 + len(code))
    payload = rng.integers(0, 2, size=100, dtype=np.uint8)
    transmitted = _encoded_frame(payload, code=code)

    search = run_fec_trials(transmitted, expected_payload_length=payload.size)

    assert search.best_result is not None and search.best_result.accepted
    assert search.fec_type == code
    assert search.interleaver_type is None
    assert search.crc_pass is True
    assert len(search.ranked_trials) == search.diagnostics["candidate_count"] == 81
    assert len(search.trial_log) == 81
    np.testing.assert_array_equal(search.decoded_bits, payload)


def test_trial_engine_identifies_k7_with_block_interleaver():
    rng = np.random.default_rng(7103)
    payload = rng.integers(0, 2, size=106, dtype=np.uint8)
    transmitted = _encoded_frame(payload, interleaver=("block", 16, 16))

    search = run_fec_trials(transmitted, expected_payload_length=payload.size)

    assert search.fec_type == "Conv_R12_K7"
    assert search.interleaver_type == "Block_16x16"
    assert search.crc_pass is True
    np.testing.assert_array_equal(search.decoded_bits, payload)


def test_trial_engine_identifies_k7_with_convolutional_interleaver():
    rng = np.random.default_rng(7207)
    payload = rng.integers(0, 2, size=106, dtype=np.uint8)
    transmitted = _encoded_frame(payload, interleaver=("convolutional", 12, 5))

    search = run_fec_trials(
        transmitted, expected_payload_length=payload.size, convolutional_branches=5
    )

    assert search.fec_type == "Conv_R12_K7"
    assert search.interleaver_type == "Convolutional_Depth_12"
    assert search.best_result.interleaver_depth == 12
    assert search.best_result.interleaver_branches == 5
    assert search.crc_pass is True
    np.testing.assert_array_equal(search.decoded_bits, payload)


def test_trial_engine_decodes_k9_after_diagonal_interleaving():
    rng = np.random.default_rng(8339)
    payload = rng.integers(0, 2, size=104, dtype=np.uint8)
    coded = encode_convolutional(append_crc16(payload), "Conv_R12_K9")
    assert coded.size == 256
    transmitted = interleave_diagonal(coded)

    search = run_fec_trials(transmitted, expected_payload_length=payload.size)
    candidate = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "Conv_R12_K9"
        and trial.interleaver_type == "Diag_16x16_S1"
    )

    assert candidate.accepted and candidate.crc_pass is True
    np.testing.assert_array_equal(candidate.decoded_bits, payload)
    assert candidate.diagnostics["interleaver_rows"] == 16
    assert candidate.diagnostics["interleaver_columns"] == 16
    assert candidate.diagnostics["interleaver_step"] == 1
    assert candidate.diagnostics["interleaver_frame_length"] == 256
    assert candidate.diagnostics["interleaver_complete_blocks"] == 1
    assert candidate.diagnostics["interleaver_block_remainder"] == 0
    assert candidate.diagnostics["downstream_fec"] == "Conv_R12_K9"
    assert candidate.diagnostics["downstream_fec_valid"] is True
    assert candidate.diagnostics["crc_status"] is True


def test_trial_engine_decodes_pseudo_random_interleaved_complete_k9_frame():
    rng = np.random.default_rng(8417)
    payload = rng.integers(0, 2, size=104, dtype=np.uint8)
    coded = encode_convolutional(append_crc16(payload), "Conv_R12_K9")
    interleaver = PseudoRandomInterleaver(DEFAULT_PSEUDORANDOM_CONFIG)
    transmitted = interleaver.interleave(coded)

    search = run_fec_trials(transmitted, expected_payload_length=payload.size)
    candidate = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "Conv_R12_K9"
        and trial.interleaver_type == interleaver.name
    )

    assert coded.size == transmitted.size == 256
    assert candidate.compatible and candidate.frame_length_compatible
    assert candidate.accepted and candidate.crc_pass is True
    assert candidate.diagnostics["interleaver_seed"] == 26147
    assert candidate.diagnostics["interleaver_configuration"]["frame_length"] == 256
    assert candidate.diagnostics["interleaver_processed_frames"] == 1
    assert candidate.diagnostics["downstream_fec"] == "Conv_R12_K9"
    assert candidate.diagnostics["downstream_fec_result"] == "valid"
    assert candidate.diagnostics["downstream_fec_valid"] is True
    assert candidate.diagnostics["crc_status"] is True
    np.testing.assert_array_equal(candidate.decoded_bits, payload)


def test_trial_engine_skips_pseudo_random_interleaver_without_complete_frames():
    search = run_fec_trials(np.zeros(255, dtype=np.uint8), crc_present=False)
    assert not any(
        trial.interleaver_type == "PseudoRandom_256_Seed_26147"
        for trial in search.ranked_trials
    )
    assert search.diagnostics["pseudo_random_structurally_skipped_candidates"]


def test_trial_engine_reports_diagonal_length_incompatibility_without_padding():
    payload = np.zeros(223 * 8 - 16, dtype=np.uint8)
    transmitted = _rs_encoded_frame(payload, "RS_255_223")
    assert transmitted.size == 255 * 8 == 2040

    search = run_fec_trials(transmitted, expected_payload_length=payload.size)
    candidate = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "RS_255_223"
        and trial.interleaver_type == "Diag_16x16_S1"
    )

    assert not candidate.compatible
    assert not candidate.frame_length_compatible
    assert not candidate.accepted
    assert candidate.diagnostics["candidate_status"] == "incompatible"
    assert candidate.diagnostics["interleaver_frame_length"] == 2040
    assert candidate.diagnostics["interleaver_complete_blocks"] == 7
    assert candidate.diagnostics["interleaver_block_remainder"] == 248
    assert candidate.diagnostics["downstream_fec_result"] == "not_run"
    assert candidate.diagnostics["total_frame_length"] == transmitted.size


def test_trial_engine_reports_rs_incompatibility_for_a_complete_diagonal_frame():
    transmitted = np.zeros(256, dtype=np.uint8)

    search = run_fec_trials(
        transmitted, expected_payload_length=transmitted.size, crc_present=False
    )
    candidate = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "RS_255_223"
        and trial.interleaver_type == "Diag_16x16_S1"
    )

    assert not candidate.compatible
    assert not candidate.frame_length_compatible
    assert not candidate.accepted
    assert candidate.diagnostics["candidate_status"] == "incompatible"
    assert "multiple of 2040 codeword bits" in candidate.diagnostics["candidate_error"]
    assert candidate.diagnostics["interleaver_frame_length"] == 256
    assert candidate.diagnostics["interleaver_complete_blocks"] == 1
    assert candidate.diagnostics["interleaver_block_remainder"] == 0
    assert candidate.diagnostics["total_frame_length"] == transmitted.size


def test_trial_engine_finds_crc_frame_with_no_fec():
    rng = np.random.default_rng(7309)
    payload = rng.integers(0, 2, size=100, dtype=np.uint8)
    transmitted = append_crc16(payload)

    search = run_fec_trials(transmitted, expected_payload_length=payload.size)

    assert search.fec_type is None
    assert search.interleaver_type is None
    assert search.crc_pass is True
    np.testing.assert_array_equal(search.decoded_bits, payload)


def test_wrong_fec_and_wrong_interleaver_candidates_fail_crc():
    rng = np.random.default_rng(7401)
    payload = rng.integers(0, 2, size=106, dtype=np.uint8)
    transmitted = _encoded_frame(payload, interleaver=("block", 16, 16))

    search = run_fec_trials(transmitted, expected_payload_length=payload.size)
    wrong_fec = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "Conv_R12_K5" and trial.interleaver_type == "Block_16x16"
    )
    wrong_interleaver = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "Conv_R12_K7" and trial.interleaver_type == "Block_8x8"
    )

    assert search.crc_pass is True
    assert wrong_fec.crc_pass is False and not wrong_fec.accepted
    assert wrong_interleaver.crc_pass is False and not wrong_interleaver.accepted
    assert search.ranked_trials.index(search.best_result) < search.ranked_trials.index(wrong_fec)


def test_trial_engine_recovers_low_count_coded_bit_corruption():
    rng = np.random.default_rng(7507)
    payload = rng.integers(0, 2, size=106, dtype=np.uint8)
    transmitted = _encoded_frame(payload)
    corrupted = transmitted.copy()
    corrupted[[29, 144]] ^= np.uint8(1)

    search = run_fec_trials(corrupted, expected_payload_length=payload.size)

    assert search.best_result is not None and search.best_result.accepted
    assert search.crc_pass is True
    np.testing.assert_array_equal(search.decoded_bits, payload)


def test_excessive_corruption_does_not_receive_crc_acceptance():
    rng = np.random.default_rng(7603)
    payload = rng.integers(0, 2, size=106, dtype=np.uint8)
    transmitted = _encoded_frame(payload)
    corrupted = transmitted.copy()
    flip_indices = rng.choice(corrupted.size, size=96, replace=False)
    corrupted[flip_indices] ^= np.uint8(1)

    search = run_fec_trials(corrupted, expected_payload_length=payload.size)

    assert search.crc_pass is False
    assert search.best_result is not None and not search.best_result.accepted
    assert not any(trial.crc_pass is True and trial.accepted for trial in search.ranked_trials)


def test_trial_engine_preserves_supplied_phase_metadata_and_validates_branches():
    bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    search = run_fec_trials(bits, crc_present=False, phase_rotation_rad=0.25, convolutional_branches=3)
    assert len(search.ranked_trials) == search.diagnostics["candidate_count"] == 81
    assert search.best_result.phase_rotation_rad == 0.25
    assert search.diagnostics["convolutional_interleaver_branches"] == 3
    with pytest.raises(ValueError, match="at least two"):
        run_fec_trials(bits, convolutional_branches=1)


@pytest.mark.parametrize(
    "code,payload_length",
    [("RS_255_223", 223 * 8 - 16), ("RS_255_239", 239 * 8 - 16)],
)
def test_trial_engine_identifies_each_rs_code_with_crc(code, payload_length):
    rng = np.random.default_rng(8101 + payload_length)
    payload = rng.integers(0, 2, size=payload_length, dtype=np.uint8)
    transmitted = _rs_encoded_frame(payload, code)

    search = run_fec_trials(transmitted, expected_payload_length=payload.size)

    assert search.best_result is not None and search.best_result.accepted
    assert search.fec_type == code
    assert search.interleaver_type is None
    assert search.crc_pass is True
    np.testing.assert_array_equal(search.decoded_bits, payload)
    assert {
        (trial.fec_type, trial.interleaver_type)
        for trial in search.ranked_trials
        if trial.crc_pass is True
    } == {(code, None)}


def test_trial_engine_identifies_concatenated_rs223_conv7():
    rng = np.random.default_rng(8209)
    payload = rng.integers(0, 2, size=223 * 8 - 16, dtype=np.uint8)
    transmitted = encode_concatenated(append_crc16(payload))

    search = run_fec_trials(transmitted, expected_payload_length=payload.size)

    assert search.best_result is not None and search.best_result.accepted
    assert search.fec_type == "Concat_RS223_Conv7"
    assert search.interleaver_type is None
    assert search.crc_pass is True
    np.testing.assert_array_equal(search.decoded_bits, payload)
    candidate = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "Concat_RS223_Conv7" and trial.interleaver_type is None
    )
    assert candidate.diagnostics["concatenated_diagnostics"]["viterbi_success"]
    assert candidate.diagnostics["concatenated_diagnostics"]["rs_success"]
    assert {
        (trial.fec_type, trial.interleaver_type)
        for trial in search.ranked_trials
        if trial.crc_pass is True
    } == {("Concat_RS223_Conv7", None)}


def test_wrong_rs_and_convolutional_codes_do_not_pass_frame_crc():
    rng = np.random.default_rng(8303)
    payload = rng.integers(0, 2, size=223 * 8 - 16, dtype=np.uint8)
    transmitted = encode_concatenated(append_crc16(payload))

    search = run_fec_trials(transmitted, expected_payload_length=payload.size)
    wrong_rs = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "RS_255_239" and trial.interleaver_type is None
    )
    wrong_conv = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "Conv_R12_K5" and trial.interleaver_type is None
    )

    assert search.fec_type == "Concat_RS223_Conv7"
    assert wrong_rs.crc_pass is False and not wrong_rs.accepted
    assert wrong_conv.crc_pass is False and not wrong_conv.accepted


def test_trial_engine_recovers_rs_frame_after_correctable_byte_corruption():
    rng = np.random.default_rng(8401)
    payload = rng.integers(0, 2, size=223 * 8 - 16, dtype=np.uint8)
    transmitted = _rs_encoded_frame(payload, "RS_255_223")
    codeword = bytearray(np.packbits(transmitted, bitorder="big").tobytes())
    for position in [3, 47, 119, 222, 254]:
        codeword[position] ^= position + 1
    corrupted = bytes_to_bits(bytes(codeword))

    search = run_fec_trials(corrupted, expected_payload_length=payload.size)

    assert search.fec_type == "RS_255_223"
    assert search.crc_pass is True and search.best_result.accepted
    np.testing.assert_array_equal(search.decoded_bits, payload)


@pytest.mark.parametrize(
    "fec_type,interleaver_type,frame_length,rows,cols",
    [
        ("RS_255_223", "Block_8x8", 255 * 8, 8, 8),
        ("Concat_RS223_Conv7", "Block_16x16", 4092, 16, 16),
    ],
)
def test_block_interleavers_with_incompatible_rs_frame_lengths_are_reported_without_padding(
    fec_type, interleaver_type, frame_length, rows, cols
):
    rng = np.random.default_rng(8501 + frame_length)
    payload = rng.integers(0, 2, size=223 * 8 - 16, dtype=np.uint8)
    if fec_type == "RS_255_223":
        transmitted = _rs_encoded_frame(payload, fec_type)
    else:
        transmitted = encode_concatenated(append_crc16(payload))
    assert transmitted.size == frame_length
    with pytest.raises(ValueError, match="multiple of matrix size"):
        interleave_block(transmitted, rows, cols)

    search = run_fec_trials(transmitted, expected_payload_length=payload.size)
    candidate = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == fec_type and trial.interleaver_type == interleaver_type
    )
    assert not candidate.compatible
    assert not candidate.valid and not candidate.accepted
    assert candidate.diagnostics["candidate_status"] == "incompatible"
    assert candidate.crc_pass is False


@pytest.mark.parametrize(
    "fec_type,frame_length",
    [("RS_255_223", 2040 - 1), ("RS_255_239", 2040 + 1)],
)
def test_rs_candidates_reject_incompatible_stream_lengths(fec_type, frame_length):
    search = run_fec_trials(
        np.zeros(frame_length, dtype=np.uint8), crc_present=False
    )
    candidate = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == fec_type and trial.interleaver_type is None
    )

    assert not candidate.compatible
    assert not candidate.frame_length_compatible
    assert not candidate.valid
    assert candidate.diagnostics["candidate_status"] == "incompatible"


def test_trial_engine_decodes_eight_rs223_codewords_with_block_8x8():
    codeword_count = 8
    frame_length = codeword_count * 255 * 8
    payload_length = codeword_count * 223 * 8 - 16
    rng = np.random.default_rng(8609)
    payload = rng.integers(0, 2, size=payload_length, dtype=np.uint8)
    coded = _rs_multi_encoded_frame(payload, "RS_255_223", codeword_count)

    # Corrupt only one independent RS word within its correction capability.
    corrupted_words = []
    for codeword_index, start in enumerate(range(0, coded.size, 255 * 8)):
        word = bytearray(bits_to_bytes(coded[start:start + 255 * 8]))
        if codeword_index == 3:
            for byte_index in [2, 31, 99, 173, 249]:
                word[byte_index] ^= byte_index + 1
        corrupted_words.append(bytes_to_bits(bytes(word)))
    transmitted = interleave_block(np.concatenate(corrupted_words), 8, 8)

    search = run_fec_trials(transmitted, expected_payload_length=payload.size)

    assert transmitted.size == coded.size == frame_length == 16320
    np.testing.assert_array_equal(deinterleave_block(transmitted, 8, 8), np.concatenate(corrupted_words))
    assert search.fec_type == "RS_255_223"
    assert search.interleaver_type == "Block_8x8"
    assert search.best_result is not None and search.best_result.accepted
    np.testing.assert_array_equal(search.decoded_bits, payload)
    batch = search.best_result.diagnostics["rs_batch"]
    assert batch["codeword_count"] == 8
    assert batch["codeword_length_bits"] == 2040
    assert batch["total_frame_length"] == 16320
    assert batch["interleaver"] == "Block_8x8"
    assert batch["successful_fec_decodes"] == 8
    assert batch["failed_fec_decodes"] == 0
    assert batch["crc_present"] and batch["crc_status"] is True
    assert batch["codewords"][3]["corrected_error_count"] == 5
    assert search.best_result.diagnostics["decoded_frame_bit_count"] == 8 * 223 * 8
    assert not next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "RS_255_223" and trial.interleaver_type is None
    ).accepted
    assert not next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "RS_255_239" and trial.interleaver_type == "Block_8x8"
    ).accepted


def test_one_uncorrectable_rs_codeword_does_not_contaminate_neighboring_words():
    codeword_count = 8
    payload_rng = np.random.default_rng(8801)
    codeword_payloads = [
        payload_rng.integers(0, 256, size=223, dtype=np.uint8).tobytes()
        for _ in range(codeword_count)
    ]
    coded_words = [rs_encode(data, "RS_255_223") for data in codeword_payloads]
    corrupt_rng = np.random.default_rng(881)
    # Keep the known deterministic >t error pattern that reedsolo reports as failure.
    error_payload = corrupt_rng.integers(0, 256, size=223, dtype=np.uint8).tobytes()
    error_positions = corrupt_rng.choice(255, size=17, replace=False)
    error_values = [int(corrupt_rng.integers(1, 256)) for _ in error_positions]
    coded_words[0] = rs_encode(error_payload, "RS_255_223")
    corrupted_word = bytearray(coded_words[0])
    for position, error_value in zip(error_positions, error_values):
        corrupted_word[int(position)] ^= error_value
    coded_words[0] = bytes(corrupted_word)
    coded = np.concatenate([bytes_to_bits(word) for word in coded_words])

    search = run_fec_trials(
        coded, expected_payload_length=codeword_count * 223 * 8, crc_present=False
    )
    candidate = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "RS_255_223" and trial.interleaver_type is None
    )

    batch = candidate.diagnostics["rs_batch"]
    assert not candidate.valid and not candidate.accepted
    assert batch["codeword_count"] == 8
    assert batch["successful_fec_decodes"] == 7
    assert batch["failed_fec_decodes"] == 1
    assert batch["codewords"][0]["valid"] is False
    assert all(word["valid"] for word in batch["codewords"][1:])


def test_trial_engine_decodes_64_concatenated_codewords_with_block_16x16():
    codeword_count = 64
    frame_length = codeword_count * CONCAT_CODED_BITS
    payload_length = codeword_count * CONCAT_PAYLOAD_BITS - 16
    rng = np.random.default_rng(8707)
    payload = rng.integers(0, 2, size=payload_length, dtype=np.uint8)
    coded = _concat_multi_encoded_frame(payload, codeword_count)
    transmitted = interleave_block(coded, 16, 16)

    search = run_fec_trials(transmitted, expected_payload_length=payload.size)

    assert coded.size == transmitted.size == frame_length == 261888
    np.testing.assert_array_equal(deinterleave_block(transmitted, 16, 16), coded)
    assert search.fec_type == "Concat_RS223_Conv7"
    assert search.interleaver_type == "Block_16x16"
    assert search.best_result is not None and search.best_result.accepted
    np.testing.assert_array_equal(search.decoded_bits, payload)
    batch = search.best_result.diagnostics["concatenated_batch"]
    assert batch["codeword_count"] == 64
    assert batch["codeword_length_bits"] == 4092
    assert batch["total_frame_length"] == 261888
    assert batch["interleaver"] == "Block_16x16"
    assert batch["viterbi_successful_codewords"] == 64
    assert batch["rs_successful_codewords"] == 64
    assert batch["successful_fec_decodes"] == 64
    assert batch["failed_fec_decodes"] == 0
    assert batch["crc_present"] and batch["crc_status"] is True
    assert not next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "Concat_RS223_Conv7" and trial.interleaver_type is None
    ).accepted
    assert not next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "Conv_R12_K7" and trial.interleaver_type == "Block_16x16"
    ).accepted
