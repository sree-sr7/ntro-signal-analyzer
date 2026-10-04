"""Deterministic binary sync-word correlation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .extractor import validate_binary_bits


@dataclass
class SyncCorrelationResult:
    """Normalized bipolar correlation scores for every valid sync offset."""

    peak_index: int | None = None
    peak_score: float = 0.0
    scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    sync_length: int = 0
    warnings: list[str] = field(default_factory=list)


def correlate_sync_word(received_bits: np.ndarray, sync_word: np.ndarray) -> SyncCorrelationResult:
    """Cross-correlate binary data with a binary sync word.

    Bits map to bipolar values ``0 -> -1`` and ``1 -> +1``. The correlation is
    divided by sync-word length, so a perfect match scores +1 and its exact
    complement scores -1. ``peak_index`` is the zero-based start offset.
    Ties select the earliest offset.
    """
    received = validate_binary_bits(received_bits)
    sync = validate_binary_bits(sync_word)
    if sync.size == 0:
        raise ValueError("sync_word must not be empty")
    result = SyncCorrelationResult(sync_length=int(sync.size))
    if received.size < sync.size:
        result.warnings.append("Received bit stream is shorter than the sync word")
        return result
    data_bipolar = 2.0 * received.astype(np.float64) - 1.0
    sync_bipolar = 2.0 * sync.astype(np.float64) - 1.0
    scores = np.correlate(data_bipolar, sync_bipolar, mode="valid") / sync.size
    peak = int(np.argmax(scores))
    result.peak_index = peak
    result.peak_score = float(scores[peak])
    result.scores = scores.astype(np.float64, copy=False)
    return result
