import numpy as np
import pytest

from core.interleaver.pseudorandom import (
    DEFAULT_PSEUDORANDOM_CONFIG,
    PseudoRandomInterleaver,
    PseudoRandomInterleaverConfig,
    permutation_for,
)


def test_seed_26147_frame_256_matches_literal_reference_permutation():
    # Independently calculated from the documented SplitMix64 and
    # Durstenfeld rules. This literal is deliberately not built by the SUT.
    expected = np.array([
        225, 111, 103, 57, 16, 76, 1, 213, 125, 164, 144, 98, 110, 106, 109, 61,         82, 43, 42, 152, 238, 118, 211, 219, 232, 184, 25, 53, 75, 87, 190, 167,         5, 138, 216, 62, 209, 99, 222, 94, 140, 71, 83, 252, 189, 185, 89, 33,         91, 46, 231, 159, 150, 40, 18, 176, 127, 212, 160, 230, 156, 198, 112, 22,         52, 29, 114, 172, 142, 37, 228, 247, 116, 128, 236, 23, 173, 166, 200, 196,         32, 206, 134, 100, 102, 123, 139, 50, 226, 11, 253, 47, 15, 175, 215, 8,         205, 67, 65, 157, 92, 220, 70, 38, 180, 235, 72, 20, 254, 6, 242, 227,         119, 56, 68, 104, 218, 143, 105, 245, 24, 141, 251, 124, 84, 0, 85, 30,         214, 187, 78, 79, 21, 163, 14, 19, 229, 73, 202, 151, 97, 80, 69, 31,         170, 186, 147, 107, 204, 208, 45, 177, 9, 121, 58, 234, 191, 192, 224, 3,         168, 120, 93, 74, 149, 174, 241, 115, 137, 130, 12, 41, 51, 244, 181, 135,         182, 27, 199, 148, 131, 59, 39, 86, 48, 155, 250, 126, 203, 194, 179, 10,         101, 249, 17, 162, 153, 193, 223, 255, 26, 154, 95, 217, 113, 146, 221, 145,         183, 54, 77, 210, 34, 4, 237, 64, 169, 197, 243, 195, 36, 171, 117, 132,         66, 246, 165, 2, 248, 158, 129, 108, 90, 88, 13, 178, 240, 161, 55, 28,         49, 122, 96, 133, 136, 81, 7, 60, 233, 44, 201, 188, 35, 239, 63, 207
    ], dtype=np.intp)
    np.testing.assert_array_equal(permutation_for(DEFAULT_PSEUDORANDOM_CONFIG), expected)


def test_permutation_is_bijective_inspectable_and_repeatable():
    first = PseudoRandomInterleaver(DEFAULT_PSEUDORANDOM_CONFIG)
    second = PseudoRandomInterleaver(PseudoRandomInterleaverConfig(frame_length=256, seed=26147))
    assert first.name == "PseudoRandom_256_Seed_26147"
    assert not first.permutation.flags.writeable
    np.testing.assert_array_equal(np.sort(first.permutation), np.arange(256))
    np.testing.assert_array_equal(first.permutation, second.permutation)


@pytest.mark.parametrize("fill", [0, 1])
def test_constant_frames_are_preserved(fill):
    bits = np.full(256, fill, dtype=np.uint8)
    interleaver = PseudoRandomInterleaver()
    np.testing.assert_array_equal(interleaver.interleave(bits), bits)
    np.testing.assert_array_equal(interleaver.deinterleave(bits), bits)


def test_alternating_and_deterministic_random_bit_frames_round_trip():
    interleaver = PseudoRandomInterleaver()
    alternating = np.arange(256, dtype=np.uint8) % 2
    rng = np.random.default_rng(78403)
    random_bits = rng.integers(0, 2, size=256, dtype=np.uint8)
    for bits in (alternating, random_bits):
        scrambled = interleaver.interleave(bits)
        np.testing.assert_array_equal(interleaver.deinterleave(scrambled), bits)
        assert not np.array_equal(scrambled, bits)


def test_multiple_complete_frames_are_permuted_independently():
    interleaver = PseudoRandomInterleaver()
    bits = np.arange(512, dtype=np.uint16).astype(np.uint8) % 2
    result = interleaver.interleave(bits)
    np.testing.assert_array_equal(result[:256], interleaver.interleave(bits[:256]))
    np.testing.assert_array_equal(result[256:], interleaver.interleave(bits[256:]))
    np.testing.assert_array_equal(interleaver.deinterleave(result), bits)


def test_numeric_soft_values_use_the_same_invertible_mapping():
    interleaver = PseudoRandomInterleaver()
    values = np.linspace(-4.0, 4.0, 512)
    np.testing.assert_array_equal(
        interleaver.deinterleave_values(interleaver.interleave_values(values)), values
    )


@pytest.mark.parametrize("length", [0, 1, 255, 257, 511])
def test_partial_or_empty_frames_are_rejected(length):
    interleaver = PseudoRandomInterleaver()
    with pytest.raises(ValueError, match="positive multiple"):
        interleaver.interleave(np.zeros(length, dtype=np.uint8))


def test_invalid_bits_values_seed_and_frame_length_are_rejected():
    interleaver = PseudoRandomInterleaver()
    with pytest.raises(ValueError, match="zero and one"):
        interleaver.interleave(np.full(256, 2, dtype=np.uint8))
    with pytest.raises(ValueError, match="positive integer"):
        PseudoRandomInterleaverConfig(frame_length=0)
    with pytest.raises(ValueError, match="64"):
        PseudoRandomInterleaverConfig(seed=-1)
    with pytest.raises(ValueError, match="64"):
        PseudoRandomInterleaverConfig(seed=1 << 64)
    with pytest.raises(ValueError, match="finite"):
        interleaver.interleave_values(np.full(256, np.nan))
