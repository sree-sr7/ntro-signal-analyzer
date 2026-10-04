import numpy as np
import pytest

from core.interleaver.convolutional import (
    SUPPORTED_CANDIDATE_DEPTHS,
    deinterleave_convolutional,
    interleave_convolutional,
)


@pytest.mark.parametrize("depth", SUPPORTED_CANDIDATE_DEPTHS)
@pytest.mark.parametrize("branches", [2, 3, 4, 7])
@pytest.mark.parametrize("length", [0, 1, 2, 7, 32, 257])
def test_convolutional_interleaver_is_exact_inverse_for_short_and_long_frames(
    depth, branches, length
):
    rng = np.random.default_rng(6001 + depth * 3 + branches + length)
    bits = rng.integers(0, 2, size=length, dtype=np.uint8)

    interleaved = interleave_convolutional(bits, depth=depth, branches=branches)

    assert interleaved.size == bits.size
    np.testing.assert_array_equal(
        deinterleave_convolutional(interleaved, depth=depth, branches=branches), bits
    )


@pytest.mark.parametrize("fill", [0, 1])
@pytest.mark.parametrize("depth", SUPPORTED_CANDIDATE_DEPTHS)
def test_convolutional_interleaver_round_trips_constant_data(fill, depth):
    bits = np.full(113, fill, dtype=np.uint8)
    interleaved = interleave_convolutional(bits, depth=depth, branches=5)
    np.testing.assert_array_equal(
        deinterleave_convolutional(interleaved, depth=depth, branches=5), bits
    )


def test_interleaver_uses_caller_selected_branch_count_and_resets_each_call():
    bits = np.random.default_rng(6101).integers(0, 2, size=91, dtype=np.uint8)
    for branches in (2, 4, 6):
        first = interleave_convolutional(bits, depth=8, branches=branches)
        second = interleave_convolutional(bits, depth=8, branches=branches)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(
            deinterleave_convolutional(first, depth=8, branches=branches), bits
        )


@pytest.mark.parametrize("depth,branches", [(0, 4), (-1, 4), (4, 0), (4, 1), (True, 4), (4, True)])
def test_invalid_interleaver_parameters_raise(depth, branches):
    with pytest.raises(ValueError):
        interleave_convolutional(np.array([0, 1], dtype=np.uint8), depth, branches)


def test_interleaver_rejects_non_uint8_and_nonbinary_inputs():
    with pytest.raises(TypeError):
        interleave_convolutional(np.array([0, 1], dtype=np.int64), 4, 2)
    with pytest.raises(ValueError, match="zero and one"):
        deinterleave_convolutional(np.array([0, 2], dtype=np.uint8), 4, 2)
