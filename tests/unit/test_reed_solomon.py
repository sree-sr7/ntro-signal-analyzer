import numpy as np
import pytest

from core.fec.reed_solomon import RS_CODE_SPECS, rs_decode, rs_encode


@pytest.mark.parametrize("code", tuple(RS_CODE_SPECS))
def test_clean_round_trip_uses_exact_full_length_codeword(code):
    spec = RS_CODE_SPECS[code]
    payload = bytes((index * 37 + 11) & 0xFF for index in range(spec.k))

    codeword = rs_encode(payload, code)
    decoded = rs_decode(codeword, code)

    assert len(codeword) == 255
    assert decoded.decode_success and decoded.valid
    assert decoded.code == code
    assert (decoded.n, decoded.k, decoded.parity_symbols, decoded.correction_capability) == (
        255, spec.k, spec.parity_symbols, spec.correction_capability
    )
    assert decoded.corrected_error_count == 0
    assert decoded.output_length == spec.k
    assert decoded.decoded_bytes == payload


@pytest.mark.parametrize("code", tuple(RS_CODE_SPECS))
@pytest.mark.parametrize("seed", [13, 37, 91])
def test_several_random_payloads_round_trip_independently(code, seed):
    spec = RS_CODE_SPECS[code]
    rng = np.random.default_rng(seed)
    for _ in range(4):
        payload = rng.integers(0, 256, size=spec.k, dtype=np.uint8).tobytes()
        result = rs_decode(rs_encode(payload, code), code)
        assert result.valid
        assert result.decoded_bytes == payload


@pytest.mark.parametrize("code", tuple(RS_CODE_SPECS))
def test_corrects_random_byte_errors_within_capability(code):
    spec = RS_CODE_SPECS[code]
    rng = np.random.default_rng(1100 + spec.k)
    payload = rng.integers(0, 256, size=spec.k, dtype=np.uint8).tobytes()
    codeword = bytearray(rs_encode(payload, code))
    error_count = spec.correction_capability - 2
    positions = rng.choice(spec.n, size=error_count, replace=False)
    for position in positions:
        codeword[int(position)] ^= int(rng.integers(1, 256))

    result = rs_decode(codeword, code)

    assert result.valid
    assert result.corrected_error_count == error_count
    assert result.decoded_bytes == payload


@pytest.mark.parametrize("code", tuple(RS_CODE_SPECS))
def test_corrects_exactly_t_byte_errors(code):
    spec = RS_CODE_SPECS[code]
    rng = np.random.default_rng(2200 + spec.k)
    payload = rng.integers(0, 256, size=spec.k, dtype=np.uint8).tobytes()
    codeword = bytearray(rs_encode(payload, code))
    positions = rng.choice(spec.n, size=spec.correction_capability, replace=False)
    for position in positions:
        codeword[int(position)] ^= int(rng.integers(1, 256))

    result = rs_decode(codeword, code)

    assert result.valid
    assert result.corrected_error_count == spec.correction_capability
    assert result.decoded_bytes == payload


@pytest.mark.parametrize("code", tuple(RS_CODE_SPECS))
def test_more_than_t_errors_is_never_claimed_as_exact_original_recovery(code):
    spec = RS_CODE_SPECS[code]
    rng = np.random.default_rng(3300 + spec.k)
    payload = rng.integers(0, 256, size=spec.k, dtype=np.uint8).tobytes()
    codeword = bytearray(rs_encode(payload, code))
    positions = rng.choice(spec.n, size=spec.correction_capability + 1, replace=False)
    for position in positions:
        codeword[int(position)] ^= int(rng.integers(1, 256))

    result = rs_decode(codeword, code)

    assert not (result.valid and result.decoded_bytes == payload)


@pytest.mark.parametrize("code", tuple(RS_CODE_SPECS))
def test_unstructured_random_codewords_do_not_automatically_validate(code):
    rng = np.random.default_rng(9907 + RS_CODE_SPECS[code].k)
    results = [
        rs_decode(rng.integers(0, 256, size=255, dtype=np.uint8).tobytes(), code)
        for _ in range(5)
    ]

    assert all(not result.valid for result in results)


@pytest.mark.parametrize("code", tuple(RS_CODE_SPECS))
@pytest.mark.parametrize("length", [0, 254, 256])
def test_decoder_rejects_non_255_byte_codewords(code, length):
    result = rs_decode(bytes(length), code)

    assert not result.decode_success
    assert not result.valid
    assert result.input_length == length
    assert result.error_message is not None
    with pytest.raises(ValueError, match="exactly"):
        rs_encode(bytes(RS_CODE_SPECS[code].k - 1), code)


@pytest.mark.parametrize("code", tuple(RS_CODE_SPECS))
def test_repeated_independent_frames_do_not_share_decoder_state(code):
    spec = RS_CODE_SPECS[code]
    rng = np.random.default_rng(4400 + spec.k)
    for frame_index in range(32):
        payload = rng.integers(0, 256, size=spec.k, dtype=np.uint8).tobytes()
        codeword = bytearray(rs_encode(payload, code))
        error_count = frame_index % (spec.correction_capability + 1)
        if error_count:
            positions = rng.choice(spec.n, size=error_count, replace=False)
            for position in positions:
                codeword[int(position)] ^= int(rng.integers(1, 256))
        result = rs_decode(codeword, code)
        assert result.valid
        assert result.decoded_bytes == payload
