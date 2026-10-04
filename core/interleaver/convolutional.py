"""Frame-local circular convolutional interleaver using shift-register banks.

For ``branches=B`` the input is demultiplexed round-robin into branch 0, branch
1, ..., branch B-1. Branch ``i`` has a shift-register delay of ``i * depth``
bits. Its bank is initialized with the final ``delay`` branch symbols from the
same frame (circular startup), so interleaving is a same-length permutation:
there are no synthetic startup bits and no flush bits. Delayed branch outputs
are multiplexed round-robin in branch order. Each call starts a fresh bank;
state never crosses frame/function calls. Deinterleaving applies the inverse
circular bank rotation with the same explicit parameters.
"""

from __future__ import annotations

import numpy as np

from core.bitstream.extractor import validate_binary_bits


SUPPORTED_CANDIDATE_DEPTHS = (4, 8, 12, 16)


def _validate_parameters(depth: int, branches: int) -> tuple[int, int]:
    if not isinstance(depth, (int, np.integer)) or isinstance(depth, bool) or depth <= 0:
        raise ValueError("depth must be a positive integer")
    if not isinstance(branches, (int, np.integer)) or isinstance(branches, bool) or branches < 2:
        raise ValueError("branches must be an explicit integer of at least two")
    return int(depth), int(branches)


def _split_branches(bits: np.ndarray, branches: int) -> list[np.ndarray]:
    return [bits[branch::branches].copy() for branch in range(branches)]


def _merge_branches(branch_data: list[np.ndarray], bit_count: int, branches: int) -> np.ndarray:
    output = np.empty(bit_count, dtype=np.uint8)
    for index in range(bit_count):
        branch = index % branches
        branch_position = index // branches
        output[index] = branch_data[branch][branch_position]
    return output


def interleave_convolutional(bits: np.ndarray, depth: int, branches: int) -> np.ndarray:
    """Apply circular shift-register delays ``0, depth, ..., (B-1)*depth``.

    Length need not be divisible by the branch count. Branches receive either
    ``floor(N/B)`` or ``ceil(N/B)`` bits, and circular delays are reduced modulo
    each nonempty branch length. Empty streams are returned empty.
    """
    values = validate_binary_bits(bits)
    depth, branches = _validate_parameters(depth, branches)
    banks = _split_branches(values, branches)
    for branch, bank in enumerate(banks):
        if bank.size:
            delay = (branch * depth) % bank.size
            banks[branch] = np.roll(bank, delay).astype(np.uint8, copy=False)
    return _merge_branches(banks, int(values.size), branches)


def deinterleave_convolutional(bits: np.ndarray, depth: int, branches: int) -> np.ndarray:
    """Invert :func:`interleave_convolutional` with identical frame parameters."""
    values = validate_binary_bits(bits)
    depth, branches = _validate_parameters(depth, branches)
    banks = _split_branches(values, branches)
    for branch, bank in enumerate(banks):
        if bank.size:
            delay = (branch * depth) % bank.size
            banks[branch] = np.roll(bank, -delay).astype(np.uint8, copy=False)
    return _merge_branches(banks, int(values.size), branches)
