from types import SimpleNamespace

import numpy as np
import pytest

from core.bitstream.extractor import bytes_to_bits
from core.fec import concatenated
from core.fec.concatenated import (
    CONCAT_CODED_BITS,
    CONCAT_PAYLOAD_BITS,
    CONCAT_RS_CODEWORD_BITS,
    decode_concatenated,
    encode_concatenated,
)
from core.fec.convolutional import encode_convolutional
from core.fec.reed_solomon import RS_CODE_SPECS, ReedSolomonDecodeResult, rs_encode


def test_noiseless_round_trip_reports_both_layers_successful():
    rng = np.random.default_rng(5101)
    payload = rng.integers(0, 2, size=CONCAT_PAYLOAD_BITS, dtype=np.uint8)

    coded = encode_concatenated(payload)
    decoded = decode_concatenated(coded)

    assert coded.size == CONCAT_CODED_BITS == 4092
    assert decoded.viterbi_success and decoded.rs_success and decoded.final_valid
    assert decoded.viterbi_path_metric == 0
    assert decoded.rs_corrected_errors == 0
    assert decoded.decoded_bit_length == CONCAT_PAYLOAD_BITS
    np.testing.assert_array_equal(decoded.decoded_bits, payload)


@pytest.mark.parametrize("seed", [5201, 5203, 5209, 5227])
def test_random_payloads_round_trip_with_explicit_lengths(seed):
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 2, size=CONCAT_PAYLOAD_BITS, dtype=np.uint8)
    coded = encode_concatenated(payload)

    assert coded.size == 4092
    decoded = decode_concatenated(coded)
    assert decoded.final_valid
    assert decoded.decoded_bits.size == CONCAT_PAYLOAD_BITS
    np.testing.assert_array_equal(decoded.decoded_bits, payload)


def test_coded_stream_corruption_correctable_by_inner_viterbi_layer():
    rng = np.random.default_rng(5303)
    payload = rng.integers(0, 2, size=CONCAT_PAYLOAD_BITS, dtype=np.uint8)
    coded = encode_concatenated(payload)
    corrupted = coded.copy()
    corrupted[[20, 1500, 2901, 3907]] ^= np.uint8(1)

    decoded = decode_concatenated(corrupted)

    assert decoded.viterbi_success and decoded.rs_success and decoded.final_valid
    assert decoded.viterbi_path_metric == 4
    np.testing.assert_array_equal(decoded.decoded_bits, payload)


def test_viterbi_failure_prevents_outer_rs_decode(monkeypatch):
    monkeypatch.setattr(
        concatenated,
        "decode_viterbi",
        lambda *args, **kwargs: SimpleNamespace(
            decoded_bits=np.zeros(CONCAT_RS_CODEWORD_BITS, dtype=np.uint8),
            valid=False,
            terminated=False,
            path_metric=123,
            diagnostics={"forced_failure": True},
            warnings=["forced failure"],
        ),
    )
    monkeypatch.setattr(
        concatenated,
        "rs_decode",
        lambda *args, **kwargs: pytest.fail("RS layer must not run after Viterbi failure"),
    )

    decoded = decode_concatenated(np.zeros(CONCAT_CODED_BITS, dtype=np.uint8))

    assert not decoded.viterbi_success
    assert not decoded.rs_success
    assert not decoded.final_valid


def test_outer_rs_failure_keeps_concatenated_frame_invalid():
    rng = np.random.default_rng(881)
    payload_bytes = rng.integers(0, 256, size=RS_CODE_SPECS["RS_255_223"].k, dtype=np.uint8).tobytes()
    codeword = bytearray(rs_encode(payload_bytes, "RS_255_223"))
    positions = rng.choice(255, size=17, replace=False)
    for position in positions:
        codeword[int(position)] ^= int(rng.integers(1, 256))
    received = encode_convolutional(bytes_to_bits(bytes(codeword)), "Conv_R12_K7")

    decoded = decode_concatenated(received)

    assert decoded.viterbi_success
    assert not decoded.rs_success
    assert not decoded.final_valid
    assert decoded.error_message


@pytest.mark.parametrize("length", [0, CONCAT_CODED_BITS - 1, CONCAT_CODED_BITS + 1])
def test_decoder_rejects_non_exact_coded_lengths(length):
    with pytest.raises(ValueError, match="exactly 4092"):
        decode_concatenated(np.zeros(length, dtype=np.uint8))


@pytest.mark.parametrize("length", [CONCAT_PAYLOAD_BITS - 1, CONCAT_PAYLOAD_BITS + 1])
def test_encoder_rejects_non_exact_payload_lengths(length):
    with pytest.raises(ValueError, match="exactly 1784"):
        encode_concatenated(np.zeros(length, dtype=np.uint8))


def test_repeated_independent_concatenated_frames_do_not_share_state():
    rng = np.random.default_rng(5407)
    for frame_index in range(24):
        payload = rng.integers(0, 2, size=CONCAT_PAYLOAD_BITS, dtype=np.uint8)
        decoded = decode_concatenated(encode_concatenated(payload))
        assert decoded.final_valid
        np.testing.assert_array_equal(decoded.decoded_bits, payload)
