"""Welch spectral analysis for real and complex sampled signals."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from core.common.models import SpectralResult
from ._validation import as_samples, validate_sample_rate


def analyze_spectrum(
    samples: np.ndarray,
    sample_rate: float,
    nfft: int = 1024,
    overlap: float = 0.5,
) -> SpectralResult:
    """Estimate a Welch PSD and basic spectrum measurements.

    ``overlap`` is the fraction of each Welch segment overlapped by the next.
    Complex input is returned on a centered two-sided frequency axis; real
    input uses SciPy's one-sided axis. Occupied bandwidth is the frequency
    interval between the 0.5% and 99.5% integrated-power quantiles (99% OBW).
    The noise floor is the 20th percentile of the PSD bins, a robust descriptive
    estimate rather than a universal noise estimator.

    Empty, very short, and all-zero inputs return diagnostics instead of a
    fabricated spectral peak. Non-finite input and invalid configuration raise
    ``ValueError``.
    """
    values = as_samples(samples)
    fs = validate_sample_rate(sample_rate)
    if not isinstance(nfft, (int, np.integer)) or nfft < 4:
        raise ValueError("nfft must be an integer of at least 4")
    if not np.isfinite(overlap) or overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be in the interval [0, 1)")
    result = SpectralResult(sample_rate=fs)
    if values.size == 0:
        result.warnings.append("Empty input; spectrum was not estimated")
        return result
    if values.size < 4:
        result.warnings.append("Too few samples for a useful Welch estimate")
        return result

    segment_length = min(int(nfft), int(values.size))
    noverlap = min(segment_length - 1, int(round(segment_length * overlap)))
    frequencies, power = welch(
        values,
        fs=fs,
        window="hann",
        nperseg=segment_length,
        noverlap=noverlap,
        nfft=max(int(nfft), segment_length),
        detrend=False,
        return_onesided=not np.iscomplexobj(values),
        scaling="density",
        average="mean",
    )
    if np.iscomplexobj(values):
        frequencies = np.fft.fftshift(frequencies)
        power = np.fft.fftshift(power)
    result.frequencies = np.asarray(frequencies, dtype=np.float64)
    result.power = np.asarray(np.maximum(power, 0.0), dtype=np.float64)

    if not np.any(result.power > 0):
        result.noise_floor = 0.0
        result.occupied_bandwidth = 0.0
        result.warnings.append("Input has zero power; peak frequency is undefined")
        return result

    peak_index = int(np.argmax(result.power))
    result.peak_frequency = float(result.frequencies[peak_index])
    result.noise_floor = float(np.percentile(result.power, 20.0))

    # For uniform Welch bins, bin width is constant so density cancels in the
    # cumulative quantiles; retain the bin-center width for a single-bin tone.
    cumulative = np.cumsum(result.power, dtype=np.float64)
    cumulative /= cumulative[-1]
    low_index = int(np.searchsorted(cumulative, 0.005, side="left"))
    high_index = int(np.searchsorted(cumulative, 0.995, side="left"))
    bin_width = fs / max(int(nfft), segment_length)
    result.occupied_bandwidth = float(
        max(0, high_index - low_index) * bin_width
    )
    if values.size < nfft:
        result.warnings.append(
            f"Input has {values.size} samples; effective segment length is {segment_length}"
        )
    return result
