"""Gardner timing recovery for oversampled pulse-shaped signals."""

from __future__ import annotations

import numpy as np

from core.common.models import TimingResult
from core.dsp._validation import as_samples


def _interpolate(values: np.ndarray, position: float) -> complex:
    """Linear interpolation at a fractional sample position."""
    left = int(np.floor(position))
    fraction = position - left
    if left < 0 or left + 1 >= values.size:
        return 0.0j
    return complex(values[left] * (1.0 - fraction) + values[left + 1] * fraction)


def recover_timing(
    samples: np.ndarray,
    samples_per_symbol: float,
    *,
    timing_offset: float = 0.0,
    loop_bandwidth: float = 0.01,
    damping: float = 0.707,
) -> TimingResult:
    """Recover symbol-time samples with a second-order Gardner timing loop.

    The loop assumes at least two input samples per symbol and a pulse shape
    with a usable Gardner S-curve (commonly raised-cosine PSK). Interpolation
    is linear, so this is a practical initial synchronizer, not an optimum
    fractional-delay filter. ``timing_offset`` is the initial sample position.
    """
    values = as_samples(samples)
    sps = float(samples_per_symbol)
    if not np.isfinite(sps) or sps < 2.0:
        raise ValueError("samples_per_symbol must be finite and at least 2")
    if not np.isfinite(timing_offset) or timing_offset < 0 or timing_offset >= sps:
        raise ValueError("timing_offset must be in [0, samples_per_symbol)")
    if not np.isfinite(loop_bandwidth) or loop_bandwidth <= 0 or loop_bandwidth >= 1:
        raise ValueError("loop_bandwidth must be in (0, 1)")
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be finite and positive")
    result = TimingResult(samples_per_symbol=sps)
    if values.size < int(np.ceil(3 * sps)) + 2:
        result.warnings.append("Insufficient samples for Gardner timing recovery")
        return result
    if not np.any(np.abs(values) > 0):
        result.warnings.append("Zero-power input; timing is undefined")
        return result
    lag_coherence = float(
        np.abs(np.vdot(values[:-1], values[1:]))
        / max(np.linalg.norm(values[:-1]) * np.linalg.norm(values[1:]), np.finfo(float).tiny)
    )
    if lag_coherence < 0.1:
        result.warnings.append("Input lacks adjacent-sample coherence for Gardner timing")
        return result

    # Standard second-order loop coefficients. Normalize the detector by local
    # symbol energy so these gains do not imply a fixed input amplitude.
    theta = loop_bandwidth / (damping + 0.25 / damping)
    denominator = 1.0 + 2.0 * damping * theta + theta * theta
    alpha = 4.0 * damping * theta / denominator
    beta = 4.0 * theta * theta / denominator
    omega = sps
    position = float(timing_offset)
    output: list[complex] = []
    errors: list[float] = []
    positions: list[float] = []
    while position + omega / 2.0 + 1.0 < values.size:
        early = _interpolate(values, position - omega / 2.0)
        middle = _interpolate(values, position)
        late = _interpolate(values, position + omega / 2.0)
        error = float(np.real((late - early) * np.conjugate(middle)))
        norm = float(abs(middle) ** 2 + 0.5 * (abs(early) ** 2 + abs(late) ** 2))
        if norm > np.finfo(float).tiny:
            error /= norm
        else:
            error = 0.0
        output.append(middle)
        errors.append(error)
        positions.append(position)
        omega = float(np.clip(omega + beta * error, 0.95 * sps, 1.05 * sps))
        position += omega + alpha * error

    if len(output) < 3:
        result.warnings.append("Timing loop produced too few symbol samples")
        return result
    out_dtype = np.complex64 if values.dtype in (np.dtype(np.complex64), np.dtype(np.float32)) else np.complex128
    result.synchronized_samples = np.asarray(output, dtype=out_dtype)
    # Report the settled sampling phase modulo the nominal symbol period.
    # A circular mean handles a phase estimate that straddles the period edge.
    tail_positions = np.asarray(positions[max(0, len(positions) // 2):])
    phase_angles = 2.0 * np.pi * np.mod(tail_positions, sps) / sps
    result.timing_offset = float(
        np.mod(np.angle(np.mean(np.exp(1j * phase_angles))), 2.0 * np.pi)
        * sps / (2.0 * np.pi)
    )
    # Error convergence and adequate output length provide diagnostics, not a
    # calibrated probability of correct timing.
    tail = np.asarray(errors[max(0, len(errors) // 2):], dtype=np.float64)
    jitter = float(np.median(np.abs(tail))) if tail.size else float("inf")
    result.confidence = float(np.clip(1.0 / (1.0 + 4.0 * jitter), 0.0, 0.95))
    return result
