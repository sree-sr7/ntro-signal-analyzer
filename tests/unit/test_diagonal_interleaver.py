import numpy as np
import pytest

from core.interleaver.diagonal import (
    DIAGONAL_BLOCK_SIZE,
    INPUT_TO_OUTPUT,
    OUTPUT_TO_INPUT,
    deinterleave,
    interleave,
)


# Independent reference: the manual input matrix contains row-major indexes
# 0..255. Each literal row below lists one MathWorks step-1 diagonal in output
# order, with the row wrapping modulo 16. This fixture is intentionally not
# generated from the implementation's permutation.
_REFERENCE_OUTPUT_TO_INPUT = np.asarray([
    [0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255],
    [16, 33, 50, 67, 84, 101, 118, 135, 152, 169, 186, 203, 220, 237, 254, 15],
    [32, 49, 66, 83, 100, 117, 134, 151, 168, 185, 202, 219, 236, 253, 14, 31],
    [48, 65, 82, 99, 116, 133, 150, 167, 184, 201, 218, 235, 252, 13, 30, 47],
    [64, 81, 98, 115, 132, 149, 166, 183, 200, 217, 234, 251, 12, 29, 46, 63],
    [80, 97, 114, 131, 148, 165, 182, 199, 216, 233, 250, 11, 28, 45, 62, 79],
    [96, 113, 130, 147, 164, 181, 198, 215, 232, 249, 10, 27, 44, 61, 78, 95],
    [112, 129, 146, 163, 180, 197, 214, 231, 248, 9, 26, 43, 60, 77, 94, 111],
    [128, 145, 162, 179, 196, 213, 230, 247, 8, 25, 42, 59, 76, 93, 110, 127],
    [144, 161, 178, 195, 212, 229, 246, 7, 24, 41, 58, 75, 92, 109, 126, 143],
    [160, 177, 194, 211, 228, 245, 6, 23, 40, 57, 74, 91, 108, 125, 142, 159],
    [176, 193, 210, 227, 244, 5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175],
    [192, 209, 226, 243, 4, 21, 38, 55, 72, 89, 106, 123, 140, 157, 174, 191],
    [208, 225, 242, 3, 20, 37, 54, 71, 88, 105, 122, 139, 156, 173, 190, 207],
    [224, 241, 2, 19, 36, 53, 70, 87, 104, 121, 138, 155, 172, 189, 206, 223],
    [240, 1, 18, 35, 52, 69, 86, 103, 120, 137, 154, 171, 188, 205, 222, 239],
], dtype=np.intp)


def test_interleave_all_zero_blocks():
    bits = np.zeros(3 * DIAGONAL_BLOCK_SIZE, dtype=np.uint8)
    np.testing.assert_array_equal(interleave(bits), bits)


def test_interleave_all_one_block():
    bits = np.ones(DIAGONAL_BLOCK_SIZE, dtype=np.uint8)
    np.testing.assert_array_equal(interleave(bits), bits)


def test_interleave_alternating_bits():
    bits = np.arange(DIAGONAL_BLOCK_SIZE, dtype=np.uint8) % 2
    np.testing.assert_array_equal(interleave(bits), bits[OUTPUT_TO_INPUT])


def test_interleave_deterministic_random_bits():
    bits = np.random.default_rng(8301).integers(
        0, 2, size=DIAGONAL_BLOCK_SIZE, dtype=np.uint8
    )
    np.testing.assert_array_equal(deinterleave(interleave(bits)), bits)


def test_single_complete_block_has_expected_length_and_order():
    bits = (np.arange(DIAGONAL_BLOCK_SIZE, dtype=np.uint16) % 2).astype(np.uint8)
    result = interleave(bits)
    assert result.shape == (DIAGONAL_BLOCK_SIZE,)
    np.testing.assert_array_equal(result, bits[OUTPUT_TO_INPUT])


def test_multiple_blocks_are_permuted_independently_and_keep_block_order():
    alternating = np.arange(DIAGONAL_BLOCK_SIZE, dtype=np.uint8) % 2
    bits = np.concatenate([
        np.zeros(DIAGONAL_BLOCK_SIZE, dtype=np.uint8),
        np.ones(DIAGONAL_BLOCK_SIZE, dtype=np.uint8),
        alternating,
    ])
    result = interleave(bits)
    assert result.size == bits.size
    np.testing.assert_array_equal(result[:256], bits[:256])
    np.testing.assert_array_equal(result[256:512], bits[256:512])
    np.testing.assert_array_equal(result[512:], alternating[OUTPUT_TO_INPUT])


def test_precomputed_permutation_is_bijective_and_inverse():
    expected = np.arange(DIAGONAL_BLOCK_SIZE, dtype=np.intp)
    np.testing.assert_array_equal(np.sort(OUTPUT_TO_INPUT), expected)
    np.testing.assert_array_equal(np.sort(INPUT_TO_OUTPUT), expected)
    np.testing.assert_array_equal(INPUT_TO_OUTPUT[OUTPUT_TO_INPUT], expected)
    np.testing.assert_array_equal(OUTPUT_TO_INPUT[INPUT_TO_OUTPUT], expected)


def test_exact_inverse_for_single_and_multiple_blocks():
    bits = np.random.default_rng(8317).integers(
        0, 2, size=5 * DIAGONAL_BLOCK_SIZE, dtype=np.uint8
    )
    np.testing.assert_array_equal(deinterleave(interleave(bits)), bits)
    np.testing.assert_array_equal(interleave(deinterleave(bits)), bits)


def test_permutation_matches_independently_derived_reference_vector():
    # A manually numbered 16x16 input matrix: each entry is its row-major
    # input index. The literal reference rows above give the expected output.
    manual_index_matrix = np.arange(256, dtype=np.intp).reshape(16, 16)
    expected_output_to_input = _REFERENCE_OUTPUT_TO_INPUT.reshape(-1)
    manual_output = np.concatenate([
        manual_index_matrix[(start_row + np.arange(16)) % 16, np.arange(16)]
        for start_row in range(16)
    ])
    np.testing.assert_array_equal(manual_output, expected_output_to_input)
    np.testing.assert_array_equal(OUTPUT_TO_INPUT, expected_output_to_input)

    expected_input_to_output = np.empty(256, dtype=np.intp)
    expected_input_to_output[expected_output_to_input] = np.arange(256, dtype=np.intp)
    np.testing.assert_array_equal(INPUT_TO_OUTPUT, expected_input_to_output)

    basis_blocks = np.eye(256, dtype=np.uint8)
    interleaved_basis = interleave(basis_blocks.reshape(-1)).reshape(256, 256)
    actual_input_to_output = np.argmax(interleaved_basis, axis=1)
    np.testing.assert_array_equal(actual_input_to_output, expected_input_to_output)


@pytest.mark.parametrize("length", [0, 1, 255, 257, 511, 513])
def test_invalid_non_multiple_lengths_are_rejected_without_repair(length):
    bits = np.zeros(length, dtype=np.uint8)
    with pytest.raises(ValueError, match="positive multiple of diagonal block size 256"):
        interleave(bits)
    with pytest.raises(ValueError, match="positive multiple of diagonal block size 256"):
        deinterleave(bits)


def test_repeated_execution_is_deterministic_and_does_not_mutate_input():
    bits = np.random.default_rng(8323).integers(
        0, 2, size=2 * DIAGONAL_BLOCK_SIZE, dtype=np.uint8
    )
    original = bits.copy()
    first = interleave(bits)
    second = interleave(bits)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(bits, original)
