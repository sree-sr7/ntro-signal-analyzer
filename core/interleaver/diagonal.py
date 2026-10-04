"""Project-defined 16x16, step-1 matrix helical-scan bit interleaver.

For each 256-bit block, input bits fill a 16x16 matrix row-major. Output
diagonal ``d`` starts at matrix position ``(d, 0)``. For column ``c`` from
0 through 15, it emits row ``(d + c) % 16``, column ``c``. Thus each next
diagonal starts one row below the previous one, both indices advance within a
diagonal, and the row wraps at 16. Output blocks stay in input order.

This fixed mapping is the project's adoption of the documented MathWorks
Matrix Helical Scan Interleaver construction with Array Step Size 1. It is not
claimed to be a historical repository definition or an IEEE-standard mode.
"""

from __future__ import annotations

import numpy as np

from core.bitstream.extractor import validate_binary_bits


DIAGONAL_INTERLEAVER_NAME = "Diag_16x16_S1"
DIAGONAL_ROWS = 16
DIAGONAL_COLUMNS = 16
DIAGONAL_STEP = 1
DIAGONAL_BLOCK_SIZE = DIAGONAL_ROWS * DIAGONAL_COLUMNS

# Input index i=16*r+c maps to output index
# 16*((r-c) mod 16)+c. Build its inverse once so forward output can be formed
# by indexing input in output order. Both arrays are immutable module constants.
INPUT_TO_OUTPUT = np.fromiter(
    (
        ((row - column) % DIAGONAL_ROWS) * DIAGONAL_COLUMNS + column
        for row in range(DIAGONAL_ROWS)
        for column in range(DIAGONAL_COLUMNS)
    ),
    dtype=np.intp,
    count=DIAGONAL_BLOCK_SIZE,
)
OUTPUT_TO_INPUT = np.argsort(INPUT_TO_OUTPUT)
INPUT_TO_OUTPUT.flags.writeable = False
OUTPUT_TO_INPUT.flags.writeable = False


def _validate_complete_blocks(bits: np.ndarray) -> np.ndarray:
    values = validate_binary_bits(bits)
    if values.size == 0 or values.size % DIAGONAL_BLOCK_SIZE:
        raise ValueError(
            "bit count must be a positive multiple of diagonal block size "
            f"{DIAGONAL_BLOCK_SIZE}"
        )
    return values


def interleave(bits: np.ndarray) -> np.ndarray:
    """Interleave complete 256-bit blocks using the fixed S=1 permutation."""
    values = _validate_complete_blocks(bits)
    blocks = values.reshape(-1, DIAGONAL_BLOCK_SIZE)
    return blocks[:, OUTPUT_TO_INPUT].reshape(-1).copy()


def deinterleave(bits: np.ndarray) -> np.ndarray:
    """Invert :func:`interleave` for complete 256-bit blocks."""
    values = _validate_complete_blocks(bits)
    blocks = values.reshape(-1, DIAGONAL_BLOCK_SIZE)
    return blocks[:, INPUT_TO_OUTPUT].reshape(-1).copy()
