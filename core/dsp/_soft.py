"""Measured max-log LLR calculation shared by coherent symbol demodulators."""

from __future__ import annotations

import numpy as np

from core.common.enums import ModulationType
from libs.modulation_library import symbol_indices_to_bits


def max_log_llrs(
    received: np.ndarray,
    constellation: np.ndarray,
    modulation: ModulationType,
) -> tuple[np.ndarray, float]:
    """Estimate bit LLRs from symbol distances and observed decision residuals.

    Positive LLR favors bit zero. The noise-power estimate is the measured
    mean squared complex residual to the nearest ideal symbol. It is an
    engineering estimate, not a calibrated channel-noise measurement.
    """
    values = np.asarray(received, dtype=np.complex128)
    points = np.asarray(constellation, dtype=np.complex128)
    if values.ndim != 1 or points.ndim != 1 or points.size < 2:
        raise ValueError("received symbols and constellation must be one-dimensional")
    if values.size == 0 or not np.all(np.isfinite(values)) or not np.all(np.isfinite(points)):
        raise ValueError("received symbols and constellation must be finite and nonempty")

    labels = symbol_indices_to_bits(
        np.arange(points.size, dtype=np.int64), modulation
    ).reshape(points.size, -1)
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("modulation library returned non-binary symbol labels")
    distances_squared = np.abs(values[:, None] - points[None, :]) ** 2
    decisions = np.argmin(distances_squared, axis=1)
    residual_power = float(np.mean(distances_squared[np.arange(values.size), decisions]))
    # Keep noiseless observations finite and preserve useful soft confidence.
    noise_power = max(residual_power, np.finfo(np.float64).eps)
    width = labels.shape[1]
    llrs = np.empty((values.size, width), dtype=np.float64)
    for bit_position in range(width):
        zero = labels[:, bit_position] == 0
        one = ~zero
        best_zero = np.min(distances_squared[:, zero], axis=1)
        best_one = np.min(distances_squared[:, one], axis=1)
        llrs[:, bit_position] = (best_one - best_zero) / noise_power
    np.clip(llrs, -80.0, 80.0, out=llrs)
    return llrs.reshape(-1), noise_power
