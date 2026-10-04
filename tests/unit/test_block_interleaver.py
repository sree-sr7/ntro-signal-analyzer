import numpy as np
import pytest

from core.interleaver.block import deinterleave_block, interleave_block


@pytest.mark.parametrize("rows,cols,length", [(8, 8, 64), (16, 16, 256), (32, 32, 1024)])
def test_block_interleaver_round_trips_random_bits(rows, cols, length):
    rng = np.random.default_rng(rows * 100 + cols)
    bits = rng.integers(0, 2, size=length, dtype=np.uint8)

    interleaved = interleave_block(bits, rows, cols)

    assert not np.array_equal(interleaved, bits)
    np.testing.assert_array_equal(deinterleave_block(interleaved, rows, cols), bits)


@pytest.mark.parametrize("fill", [0, 1])
@pytest.mark.parametrize("rows,cols", [(8, 8), (16, 16), (32, 32)])
def test_block_interleaver_round_trips_constant_blocks(fill, rows, cols):
    bits = np.full(rows * cols, fill, dtype=np.uint8)
    np.testing.assert_array_equal(deinterleave_block(interleave_block(bits, rows, cols), rows, cols), bits)


def test_block_interleaver_processes_multiple_complete_matrices():
    bits = np.arange(256, dtype=np.uint16).astype(np.uint8) % 2
    interleaved = interleave_block(bits, 8, 8)
    assert interleaved.size == bits.size
    np.testing.assert_array_equal(deinterleave_block(interleaved, 8, 8), bits)


@pytest.mark.parametrize("function", [interleave_block, deinterleave_block])
def test_block_interleaver_rejects_incompatible_length(function):
    with pytest.raises(ValueError, match="multiple"):
        function(np.zeros(63, dtype=np.uint8), 8, 8)


def test_block_interleaver_rejects_nonbinary_and_invalid_dimensions():
    with pytest.raises(ValueError, match="zero and one"):
        interleave_block(np.full(64, 2, dtype=np.uint8), 8, 8)
    with pytest.raises(ValueError, match="positive integer"):
        interleave_block(np.zeros(64, dtype=np.uint8), 0, 8)
