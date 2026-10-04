"""Deterministic rectangular matrix block interleaver."""

from __future__ import annotations

import numpy as np

from core.bitstream.extractor import validate_binary_bits


def _matrix_dimensions(rows: int, cols: int) -> tuple[int, int]:
    if not isinstance(rows, (int, np.integer)) or isinstance(rows, bool) or rows <= 0:
        raise ValueError("rows must be a positive integer")
    if not isinstance(cols, (int, np.integer)) or isinstance(cols, bool) or cols <= 0:
        raise ValueError("cols must be a positive integer")
    return int(rows), int(cols)


def interleave_block(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Interleave one or more row-major matrices by reading each column.

    For each block, input bits fill a ``rows x cols`` matrix row by row. Output
    bits are read column by column, top to bottom, then left to right. The input
    length must be a positive multiple of ``rows * cols``; multiple complete
    matrices are processed independently in sequence.
    """
    values = validate_binary_bits(bits)
    rows, cols = _matrix_dimensions(rows, cols)
    block_size = rows * cols
    if values.size == 0 or values.size % block_size:
        raise ValueError(f"bit count must be a positive multiple of matrix size {block_size}")
    matrices = values.reshape(-1, rows, cols)
    return matrices.transpose(0, 2, 1).reshape(-1).copy()


def deinterleave_block(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Invert :func:`interleave_block` using its column-read convention."""
    values = validate_binary_bits(bits)
    rows, cols = _matrix_dimensions(rows, cols)
    block_size = rows * cols
    if values.size == 0 or values.size % block_size:
        raise ValueError(f"bit count must be a positive multiple of matrix size {block_size}")
    matrices = values.reshape(-1, cols, rows)
    return matrices.transpose(0, 2, 1).reshape(-1).copy()
