import numpy as np
import pytest

from core.fec.ldpc_decoder import (
    IEEE_REFERENCE_ID,
    LDPC_CONFIGS,
    LDPC_DECODER_ALGORITHM,
    TAVILDAR_REFERENCE_ID,
    build_parity_check_matrix,
    decode_ldpc,
    decode_ldpc_frame,
    encode_ldpc,
    encode_ldpc_frame,
)


CONFIGS = tuple(LDPC_CONFIGS)


@pytest.mark.parametrize("name", CONFIGS)
def test_configuration_provenance_prototype_and_generated_matrix(name):
    config = LDPC_CONFIGS[name]
    H = build_parity_check_matrix(config)

    assert config.source_reference_id == IEEE_REFERENCE_ID
    assert config.cross_check_reference_id == TAVILDAR_REFERENCE_ID
    assert config.matrix_identifier.startswith("IEEE80211N_D11_0_AnnexR_TableR.")
    assert (config.N, config.K, config.M) == {
        "LDPC_648_R1_2": (648, 324, 324),
        "LDPC_648_R2_3": (648, 432, 216),
        "LDPC_648_R3_4": (648, 486, 162),
        "LDPC_1296_R1_2": (1296, 648, 648),
        "LDPC_1296_R2_3": (1296, 864, 432),
        "LDPC_1296_R3_4": (1296, 972, 324),
        "LDPC_1944_R1_2": (1944, 972, 972),
        "LDPC_1944_R2_3": (1944, 1296, 648),
        "LDPC_1944_R3_4": (1944, 1458, 486),
    }[name]
    assert config.rate == name.split("_R", 1)[1].replace("_", "/")
    assert config.Z == config.N // 24
    assert config.generated_h_dimensions == (config.M, config.N)
    assert H.shape == config.generated_h_dimensions
    assert H.dtype == np.uint8
    assert np.all((H == 0) | (H == 1))
    assert len(config.prototype) == config.M // config.Z
    assert all(len(row) == 24 for row in config.prototype)
    assert all(value == -1 or 0 <= value < config.Z for row in config.prototype for value in row)


@pytest.mark.parametrize("name", CONFIGS)
def test_deterministic_systematic_round_trip_and_zero_syndrome(name):
    config = LDPC_CONFIGS[name]
    rng = np.random.default_rng(82000 + config.N + config.K)
    payload = rng.integers(0, 2, size=config.K, dtype=np.uint8)

    codeword = encode_ldpc(payload, config)
    H = build_parity_check_matrix(config)
    decoded = decode_ldpc(np.where(codeword == 0, 32.0, -32.0), config)

    assert codeword.shape == (config.N,)
    np.testing.assert_array_equal(codeword[:config.K], payload)
    assert not np.any((H @ codeword) & np.uint8(1))
    assert decoded.success and decoded.syndrome_pass
    assert decoded.iterations_used == 0
    assert decoded.decoder_algorithm == LDPC_DECODER_ALGORITHM
    np.testing.assert_array_equal(decoded.decoded_bits, payload)


@pytest.mark.parametrize("name", CONFIGS)
def test_low_confidence_single_bit_error_is_corrected(name):
    config = LDPC_CONFIGS[name]
    rng = np.random.default_rng(82100 + config.N + config.K)
    payload = rng.integers(0, 2, size=config.K, dtype=np.uint8)
    codeword = encode_ldpc(payload, config)
    llrs = np.where(codeword == 0, 5.0, -5.0)
    error_index = (config.N // 3) % config.N
    llrs[error_index] = -np.sign(llrs[error_index]) * 0.2

    decoded = decode_ldpc(llrs, config, max_iterations=30)

    assert decoded.success and decoded.syndrome_pass
    assert decoded.iterations_used > 0
    np.testing.assert_array_equal(decoded.decoded_bits, payload)


def test_soft_llr_magnitude_changes_the_decoding_outcome_for_same_hard_decisions():
    config = LDPC_CONFIGS["LDPC_648_R1_2"]
    rng = np.random.default_rng(82201)
    payload = rng.integers(0, 2, size=config.K, dtype=np.uint8)
    codeword = encode_ldpc(payload, config)
    low_confidence = np.where(codeword == 0, 4.0, -4.0)
    high_confidence = low_confidence.copy()
    error_index = 0
    low_confidence[error_index] = -np.sign(low_confidence[error_index]) * 0.1
    high_confidence[error_index] = -np.sign(high_confidence[error_index]) * 100.0
    np.testing.assert_array_equal(low_confidence < 0, high_confidence < 0)

    corrected = decode_ldpc(low_confidence, config, max_iterations=20)
    held_by_confidence = decode_ldpc(high_confidence, config, max_iterations=20)

    assert corrected.success
    np.testing.assert_array_equal(corrected.decoded_bits, payload)
    assert not held_by_confidence.success


def test_harder_three_error_case_recovers_with_weak_wrong_sign_llrs():
    config = LDPC_CONFIGS["LDPC_648_R1_2"]
    rng = np.random.default_rng(82601)
    payload = rng.integers(0, 2, size=config.K, dtype=np.uint8)
    codeword = encode_ldpc(payload, config)
    llrs = np.where(codeword == 0, 5.0, -5.0)
    error_indices = (2, 87, 271)
    for index in error_indices:
        llrs[index] = -np.sign(llrs[index]) * 0.2

    decoded = decode_ldpc(llrs, config, max_iterations=30)

    assert decoded.success
    assert decoded.iterations_used > 0
    np.testing.assert_array_equal(decoded.decoded_bits, payload)


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("length_kind", ["N-1", "N+1", "2N-1", "arbitrary"])
def test_single_and_frame_decoder_reject_non_codeword_lengths(name, length_kind):
    config = LDPC_CONFIGS[name]
    length = {
        "N-1": config.N - 1,
        "N+1": config.N + 1,
        "2N-1": (2 * config.N) - 1,
        "arbitrary": 17,
    }[length_kind]
    llrs = np.zeros(length, dtype=np.float64)

    with pytest.raises(ValueError):
        decode_ldpc(llrs, config)
    with pytest.raises(ValueError):
        decode_ldpc_frame(llrs, config)


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("length", ["K-1", "K+1", "2K-1"])
def test_encoder_rejects_partial_payloads(name, length):
    config = LDPC_CONFIGS[name]
    size = {"K-1": config.K - 1, "K+1": config.K + 1, "2K-1": 2 * config.K - 1}[length]
    with pytest.raises(ValueError, match="positive multiple"):
        encode_ldpc_frame(np.zeros(size, dtype=np.uint8), config)


@pytest.mark.parametrize("name", CONFIGS)
def test_three_codeword_frame_round_trip(name):
    config = LDPC_CONFIGS[name]
    rng = np.random.default_rng(82300 + config.N + config.K)
    payload = rng.integers(0, 2, size=3 * config.K, dtype=np.uint8)
    codewords = encode_ldpc_frame(payload, config)
    llrs = np.where(codewords == 0, 24.0, -24.0)

    decoded = decode_ldpc_frame(llrs, config)

    assert codewords.size == 3 * config.N
    assert decoded.success
    assert decoded.codeword_count == 3
    assert decoded.successful_codewords == 3
    assert decoded.failed_codewords == 0
    assert decoded.diagnostics["syndrome_status"] == [True, True, True]
    np.testing.assert_array_equal(decoded.decoded_bits, payload)


def test_one_failed_codeword_does_not_prevent_neighboring_words_from_decoding():
    config = LDPC_CONFIGS["LDPC_648_R1_2"]
    rng = np.random.default_rng(82401)
    payload = rng.integers(0, 2, size=3 * config.K, dtype=np.uint8)
    codewords = encode_ldpc_frame(payload, config)
    llrs = np.where(codewords == 0, 12.0, -12.0)
    bad_hard_word = rng.integers(0, 2, size=config.N, dtype=np.uint8)
    while not np.any((build_parity_check_matrix(config) @ bad_hard_word) & np.uint8(1)):
        bad_hard_word = rng.integers(0, 2, size=config.N, dtype=np.uint8)
    llrs[:config.N] = np.where(bad_hard_word == 0, 1_000_000.0, -1_000_000.0)

    decoded = decode_ldpc_frame(llrs, config, max_iterations=1, normalization=0.01)

    assert not decoded.success
    assert decoded.codeword_count == 3
    assert decoded.successful_codewords == 2
    assert decoded.failed_codewords == 1
    assert not decoded.codeword_results[0].syndrome_pass
    assert all(word.syndrome_pass for word in decoded.codeword_results[1:])
    assert decoded.decoded_bits.size == 0


@pytest.mark.parametrize("bad_llrs", [np.array([1.0 + 1.0j]), np.array([np.nan]), np.array([np.inf])])
def test_decoder_rejects_complex_or_nonfinite_llrs(bad_llrs):
    config = LDPC_CONFIGS["LDPC_648_R1_2"]
    llrs = np.resize(bad_llrs, config.N)
    with pytest.raises((TypeError, ValueError)):
        decode_ldpc(llrs, config)
