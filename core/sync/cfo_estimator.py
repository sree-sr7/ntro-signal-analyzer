"""M-th-power coarse CFO estimation for PSK-like complex baseband."""

from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len

from core.common.models import CFOResult
from core.dsp._validation import as_samples, validate_sample_rate


def estimate_cfo(
    samples: np.ndarray,
    sample_rate: float,
    modulation_order: int = 4,
    *,
    min_confidence: float = 0.12,
) -> CFOResult:
    """Estimate coarse CFO using the M-th power of a PSK-like signal.

    Raising ideal M-PSK samples to power M removes their data phase. The mean
    adjacent phase increment estimates frequency in the unambiguous interval
    ``[-sample_rate/(2M), sample_rate/(2M))``. Confidence is phase-step
    coherence; low-coherence/noise-only input returns no numeric estimate.
    Pulse-shaped transitions and non-PSK constellations can lower accuracy.
    """
    values = as_samples(samples)
    fs = validate_sample_rate(sample_rate)
    if not isinstance(modulation_order, (int, np.integer)) or modulation_order < 2:
        raise ValueError("modulation_order must be an integer of at least 2")
    if not np.isfinite(min_confidence) or min_confidence < 0 or min_confidence > 1:
        raise ValueError("min_confidence must be in [0, 1]")
    result = CFOResult(method=f"{int(modulation_order)}th-power spectral peak")
    if not np.iscomplexobj(values):
        result.warnings.append("CFO estimator requires complex PSK-like baseband samples")
        return result
    if values.size < 4:
        result.warnings.append("Too few samples for CFO estimation")
        return result
    amplitude = np.abs(values)
    scale = float(np.max(amplitude))
    if scale == 0.0:
        result.warnings.append("Zero-power input; CFO is undefined")
        return result
    normalized = values.astype(np.complex128, copy=False) / scale
    powered = normalized ** int(modulation_order)
    magnitude = np.abs(powered)
    valid = (magnitude[1:] > 1e-8) & (magnitude[:-1] > 1e-8)
    if np.count_nonzero(valid) < max(3, values.size // 8):
        result.warnings.append("Insufficient nonzero samples after M-th power")
        return result
    index = np.arange(values.size, dtype=np.float64)
    window = np.hanning(values.size)
    nfft = next_fast_len(max(4096, 2 * values.size))
    transformed = np.fft.fft(powered * window, n=nfft)
    power_spectrum = np.abs(transformed) ** 2
    peak_index = int(np.argmax(power_spectrum))
    bin_width = fs / nfft
    peak_frequency = float(
        peak_index * bin_width if peak_index <= nfft // 2 else (peak_index - nfft) * bin_width
    )
    # Three-bin parabolic interpolation on log power improves on FFT-bin precision.
    left = np.log(max(power_spectrum[(peak_index - 1) % nfft], np.finfo(float).tiny))
    center = np.log(max(power_spectrum[peak_index], np.finfo(float).tiny))
    right = np.log(max(power_spectrum[(peak_index + 1) % nfft], np.finfo(float).tiny))
    curvature = left - 2.0 * center + right
    fractional_bin = 0.5 * (left - right) / curvature if curvature != 0 else 0.0
    peak_frequency += float(np.clip(fractional_bin, -0.5, 0.5)) * bin_width
    corrected = powered * np.exp(-2j * np.pi * peak_frequency * index / fs)
    coherence = float(
        np.abs(np.sum(corrected))
        / max(np.sum(np.abs(powered)), np.finfo(float).tiny)
    )
    result.confidence = float(np.clip(coherence, 0.0, 1.0))
    if coherence < min_confidence:
        result.warnings.append("M-th-power phase coherence is too low; CFO rejected")
        return result
    result.estimated_offset_hz = float(peak_frequency / int(modulation_order))
    return result
