import numpy as np
import pytest

from core.bitstream.correlator import correlate_sync_word
from core.bitstream.extractor import (
    bits_to_bytes,
    bytes_to_bits,
    format_bits_binary,
    format_bits_hex,
    pad_bits,
    trim_bits,
    validate_binary_bits,
)
from core.bitstream.frame_analyzer import analyze_frames, estimate_periodicity


def test_bit_byte_conversions_are_msb_first():
    bits = np.array([1, 0, 1, 0, 0, 1, 1, 0], dtype=np.uint8)

    assert bits_to_bytes(bits) == b"\xA6"
    np.testing.assert_array_equal(bytes_to_bits(b"\xA6"), bits)
    assert format_bits_binary(bits, group_size=4) == "1010 0110"
    assert format_bits_hex(bits) == "A6"


def test_bitstream_padding_and_trimming_are_explicit():
    bits = np.array([1, 0, 1], dtype=np.uint8)
    assert bits_to_bytes(bits, pad_to_byte=True) == b"\xA0"
    assert format_bits_hex(bits, pad_to_nibble=True) == "A"
    np.testing.assert_array_equal(pad_bits(bits, 5), np.array([1, 0, 1, 0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(trim_bits(np.array([1, 0, 1, 1], dtype=np.uint8), 3), bits)
    with pytest.raises(ValueError, match="multiple of eight"):
        bits_to_bytes(bits)
    with pytest.raises(ValueError, match="multiple of four"):
        format_bits_hex(bits)


@pytest.mark.parametrize(
    "values",
    [np.array([0, 2], dtype=np.uint8), np.array([0, 1], dtype=np.int64), np.zeros((2, 2), dtype=np.uint8)],
)
def test_binary_uint8_validation_is_strict(values):
    with pytest.raises((TypeError, ValueError)):
        validate_binary_bits(values)


def test_binary_sync_correlation_finds_exact_word_offset():
    sync = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    received = np.r_[np.array([0, 0, 0], dtype=np.uint8), sync, np.array([1, 1], dtype=np.uint8)]

    result = correlate_sync_word(received, sync)

    assert result.peak_index == 3
    assert result.peak_score == pytest.approx(1.0)


def test_frame_analysis_splits_header_and_payload_after_detected_sync():
    rng = np.random.default_rng(1901)
    sync = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    frame_body = np.r_[
        sync,
        rng.integers(0, 2, size=5, dtype=np.uint8),
        rng.integers(0, 2, size=19, dtype=np.uint8),
    ]
    stream = np.r_[np.zeros(7, dtype=np.uint8), frame_body, frame_body, frame_body]

    result = analyze_frames(stream, sync_word=sync, header_length=5, payload_length=19)

    assert result.sync_index == 7
    assert result.frame_length_estimate == frame_body.size
    np.testing.assert_array_equal(result.header_bits, frame_body[8:13])
    np.testing.assert_array_equal(result.payload_bits, frame_body[13:])


def test_periodicity_returns_strongest_repeated_bit_lag():
    pattern = np.array([1, 1, 0, 1, 0, 0, 1, 0], dtype=np.uint8)
    result = estimate_periodicity(np.tile(pattern, 5), min_period=2, max_period=16)

    assert result.period_bits == pattern.size
    assert result.peak_score == pytest.approx(1.0)


def test_frame_analysis_reports_missing_sync_and_constant_periodicity():
    constant = np.zeros(64, dtype=np.uint8)
    result = analyze_frames(constant, sync_word=np.ones(8, dtype=np.uint8))

    assert result.sync_index is None
    assert result.warnings
    assert estimate_periodicity(constant).period_bits is None
