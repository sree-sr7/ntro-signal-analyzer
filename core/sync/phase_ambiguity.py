"""Enumerate rotational phase alternatives for PSK signals."""

from __future__ import annotations

import numpy as np

from core.common.models import PhaseAmbiguityResult
from core.dsp._validation import as_samples


def phase_ambiguity_candidates(
    samples: np.ndarray,
    modulation_order: int,
) -> PhaseAmbiguityResult:
    """Return every rotationally equivalent M-PSK phase correction.

    Candidate corrections are ``2πk/M`` for ``k=0..M-1``. The function does
    not rank or select a candidate; later framing, known structure, or CRC
    validation must choose among the returned sample arrays.
    """
    values = as_samples(samples)
    if not isinstance(modulation_order, (int, np.integer)) or modulation_order < 2:
        raise ValueError("modulation_order must be an integer of at least 2")
    phases = [float(2.0 * np.pi * k / int(modulation_order)) for k in range(int(modulation_order))]
    candidates = [np.asarray(values * np.exp(-1j * phase)) for phase in phases]
    return PhaseAmbiguityResult(
        modulation_order=int(modulation_order),
        candidate_phases_rad=phases,
        candidate_samples=candidates,
    )
