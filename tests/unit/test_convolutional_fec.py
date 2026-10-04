import numpy as np
import pytest

from core.fec.convolutional import (
    CONVOLUTIONAL_CODE_SPECS,
    GENERATOR_POLYNOMIALS,
    convolutional_encode,
    decode_viterbi,
    encode_convolutional,
    viterbi_decode,
)


KNOWN_VECTORS = {
    "Conv_R12_K3": "111000010111",
    "Conv_R12_K5": "1110100000001011",
    "Conv_R12_K7": "11011101100111010111",
    "Conv_R12_K9": "110111111110000011101011",
}


@pytest.mark.parametrize("code", CONVOLUTIONAL_CODE_SPECS)
def test_each_supported_code_matches_known_deterministic_vector(code):
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)

    encoded = encode_convolutional(bits, code)

    assert "".join(str(int(bit)) for bit in encoded) == KNOWN_VECTORS[code]


@pytest.mark.parametrize("code", CONVOLUTIONAL_CODE_SPECS)
@pytest.mark.parametrize("payload_length", [1, 7, 64, 193])
def test_each_code_round_trips_multiple_terminated_payload_lengths(code, payload_length):
    rng = np.random.default_rng(3001 + payload_length + len(code))
    payload = rng.integers(0, 2, size=payload_length, dtype=np.uint8)
    spec = CONVOLUTIONAL_CODE_SPECS[code]

    encoded = encode_convolutional(payload, spec)
    decoded = decode_viterbi(encoded, spec, terminated=True)

    assert encoded.size == 2 * (payload_length + spec.constraint_length - 1)
    assert decoded.valid and decoded.terminated
    assert decoded.path_metric == 0
    assert decoded.decoded_length == payload_length
    assert decoded.diagnostics["tail_bits_removed"] == spec.constraint_length - 1
    assert decoded.diagnostics["fec_type"] == code
    np.testing.assert_array_equal(decoded.decoded_bits, payload)


@pytest.mark.parametrize("code", CONVOLUTIONAL_CODE_SPECS)
def test_each_code_corrects_a_small_number_of_hard_bit_errors(code):
    rng = np.random.default_rng(4001 + len(code))
    payload = rng.integers(0, 2, size=128, dtype=np.uint8)
    encoded = encode_convolutional(payload, code)
    corrupted = encoded.copy()
    corrupted[[5, encoded.size // 2 + 3]] ^= np.uint8(1)

    decoded = decode_viterbi(corrupted, code, terminated=True)

    np.testing.assert_array_equal(decoded.decoded_bits, payload)
    assert decoded.valid and decoded.terminated
    assert decoded.path_metric <= 2


@pytest.mark.parametrize("code", CONVOLUTIONAL_CODE_SPECS)
def test_unterminated_mode_preserves_all_source_bits(code):
    payload = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    encoded = encode_convolutional(payload, code, terminated=False)

    decoded = decode_viterbi(encoded, code, terminated=False)

    np.testing.assert_array_equal(decoded.decoded_bits, payload)
    assert not decoded.terminated
    assert decoded.diagnostics["tail_bits_removed"] == 0


def test_step79_k7_constants_and_legacy_apis_remain_compatible():
    rng = np.random.default_rng(5003)
    payload = rng.integers(0, 2, size=97, dtype=np.uint8)
    legacy_encoded = convolutional_encode(payload)
    new_encoded = encode_convolutional(payload, "Conv_R12_K7")

    assert GENERATOR_POLYNOMIALS == (0o171, 0o133)
    np.testing.assert_array_equal(legacy_encoded, new_encoded)
    np.testing.assert_array_equal(viterbi_decode(legacy_encoded).decoded_bits, payload)
    assert CONVOLUTIONAL_CODE_SPECS["Conv_R12_K7"].termination_tail_bits == 6


def test_many_independent_frames_have_no_decoder_state_contamination():
    rng = np.random.default_rng(5101)
    for frame_index in range(40):
        code = tuple(CONVOLUTIONAL_CODE_SPECS)[frame_index % 4]
        payload = rng.integers(0, 2, size=frame_index + 1, dtype=np.uint8)
        decoded = decode_viterbi(encode_convolutional(payload, code), code)
        np.testing.assert_array_equal(decoded.decoded_bits, payload)
        assert decoded.path_metric == 0


@pytest.mark.parametrize("code", CONVOLUTIONAL_CODE_SPECS)
def test_invalid_code_and_input_lengths_are_rejected(code):
    spec = CONVOLUTIONAL_CODE_SPECS[code]
    with pytest.raises(ValueError, match="Unsupported convolutional code"):
        encode_convolutional(np.array([1], dtype=np.uint8), "Conv_R12_K4")
    with pytest.raises(ValueError, match="divisible"):
        decode_viterbi(np.array([0, 1, 1], dtype=np.uint8), code)
    with pytest.raises(ValueError, match="at least"):
        decode_viterbi(np.zeros(2 * (spec.constraint_length - 2), dtype=np.uint8), code)
    with pytest.raises(ValueError, match="zero and one"):
        encode_convolutional(np.array([0, 2], dtype=np.uint8), code)


def test_legacy_entry_points_reject_unsupported_rates_and_constraint_lengths():
    bits = np.array([1, 0, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="Only rate"):
        convolutional_encode(bits, rate="R2/3")
    with pytest.raises(ValueError, match="Supported constraint lengths"):
        convolutional_encode(bits, constraint_length=4)
