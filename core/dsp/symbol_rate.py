"""Bounded classical symbol-rate estimation for modulated baseband samples."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from core.common.models import SymbolRateResult
from ._validation import as_samples, validate_sample_rate


def _cyclic_coherence(feature: np.ndarray, rate: float, sample_rate: float) -> float:
    """Measure whether a periodic feature's phase remains stable across blocks."""
    values = np.asarray(feature, dtype=np.float64)
    block_count = 16
    block_length = values.size // block_count
    if block_length < 32:
        return 0.0
    coefficients: list[complex] = []
    for block in range(block_count):
        start = block * block_length
        stop = start + block_length
        index = np.arange(start, stop, dtype=np.float64)
        reference = np.exp(-2j * np.pi * rate * index / sample_rate)
        coefficients.append(complex(np.sum(values[start:stop] * reference)))
    denominator = sum(abs(value) for value in coefficients)
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.clip(abs(sum(coefficients)) / denominator, 0.0, 1.0))


def estimate_symbol_rate(
    samples: np.ndarray,
    sample_rate: float,
    min_rate: float,
    max_rate: float,
    *,
    max_candidates: int = 5,
) -> SymbolRateResult:
    """Estimate a symbol rate inside an explicit search interval.

    The estimator combines peaks in the power-envelope spectrum with a
    nonlinearity autocorrelation check. When the envelope is constant (as for
    ideal FSK), it also examines the phase-increment/frequency-discriminator
    sequence. The strongest periodicity is returned with alternatives and a
    conservative confidence; this bounded estimate is not ground truth and is
    intended for pulse-shaped PSK and simple FSK-like signals. Confidence is
    blockwise cyclic-feature coherence, not a calibrated probability.
    """
    values = as_samples(samples)
    fs = validate_sample_rate(sample_rate)
    low = float(min_rate)
    high = float(max_rate)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= low:
        raise ValueError("rates must be finite, positive, and min_rate < max_rate")
    if not isinstance(max_candidates, (int, np.integer)) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if values.size < 32:
        return SymbolRateResult(warnings=["Too few samples for bounded rate estimation"])
    low = max(low, fs / max(2, values.size // 4))
    high = min(high, fs / 2.0)
    if low >= high:
        return SymbolRateResult(warnings=["Rate bounds exceed the resolution of this record"])

    # Welch-like periodogram of nonlinearly detected envelope power. Remove DC,
    # apply a Hann taper, and search only within the caller's baud interval.
    power = np.abs(values) ** 2
    power = power - np.mean(power)
    window = np.hanning(values.size)
    envelope_spec = np.abs(np.fft.rfft(power * window)) ** 2
    frequency_axis = np.fft.rfftfreq(values.size, d=1.0 / fs)
    in_band = (frequency_axis >= low) & (frequency_axis <= high)
    periodicity: dict[float, float] = {}
    methods: dict[float, str] = {}
    cyclic_features: dict[str, np.ndarray] = {"power-envelope periodicity": power}
    if np.any(in_band) and np.any(envelope_spec[in_band] > 0):
        band_indices = np.flatnonzero(in_band)
        local_peaks, _ = find_peaks(envelope_spec)
        peak_indices = local_peaks[in_band[local_peaks]]
        if peak_indices.size == 0:
            peak_indices = np.array([band_indices[np.argmax(envelope_spec[in_band])]])
        for index in peak_indices:
            f = float(frequency_axis[index])
            if low <= f <= high:
                periodicity[f] = max(periodicity.get(f, 0.0), float(envelope_spec[index]))
                methods[f] = "power-envelope periodicity"

    # Search the real autocorrelation of envelope power and discriminator.
    # A first-lobe peak/decay supplies useful evidence when a spectral line is
    # weak. For FSK, the discriminator's held-symbol structure is informative.
    min_lag = max(2, int(np.floor(fs / high)))
    max_lag = min(values.size // 3, int(np.ceil(fs / low)))
    features: list[tuple[str, np.ndarray]] = [("power-envelope", power)]
    phase_step: np.ndarray | None = None
    if np.iscomplexobj(values):
        phase_step = np.angle(values[1:] * np.conjugate(values[:-1]))
        if phase_step.size >= 32:
            features.append(("phase-increment", phase_step - np.mean(phase_step)))

            # FSK carries symbol changes in instantaneous frequency while its
            # magnitude can be constant. Frequency-jump events remain aligned
            # to the symbol clock, producing a line at the symbol rate.
            jumps = np.abs(np.diff(phase_step))
            jump_scale = float(np.max(jumps)) if jumps.size else 0.0
            if jump_scale > 0:
                from scipy.signal import find_peaks as _find_peaks
                events, _ = _find_peaks(
                    jumps,
                    height=0.25 * jump_scale,
                    distance=max(1, min_lag // 2),
                )
                if events.size >= 4:
                    event_train = np.zeros(jumps.size, dtype=np.float64)
                    event_train[events] = 1.0
                    cyclic_features["frequency-transition cyclostationarity"] = event_train
                    event_spectrum = np.abs(np.fft.rfft(event_train * np.hanning(event_train.size))) ** 2
                    event_axis = np.fft.rfftfreq(event_train.size, d=1.0 / fs)
                    event_band = (event_axis >= low) & (event_axis <= high)
                    event_indices, _ = find_peaks(event_spectrum)
                    event_indices = event_indices[event_band[event_indices]]
                    if event_indices.size:
                        for index in event_indices:
                            rate = float(event_axis[index])
                            periodicity[rate] = max(
                                periodicity.get(rate, 0.0), float(event_spectrum[index])
                            )
                            methods[rate] = "frequency-transition cyclostationarity"

    rate_scores = dict(periodicity)
    for name, feature in features:
        feature = np.asarray(feature, dtype=np.float64)
        feature -= np.mean(feature)
        energy = float(np.dot(feature, feature))
        if energy <= np.finfo(np.float64).tiny:
            continue
        spectrum = np.fft.rfft(feature, n=2 * feature.size)
        corr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=2 * feature.size)[:feature.size]
        # Unbiased normalization preserves the triangular first lobe of a
        # rectangular FSK discriminator and reduces finite-record bias.
        lag = np.arange(feature.size, dtype=np.float64)
        corr = corr / np.maximum(feature.size - lag, 1.0)
        corr /= max(float(corr[0]), np.finfo(np.float64).tiny)
        search_end = min(max_lag, corr.size - 2)
        if min_lag >= search_end:
            continue
        region = corr[min_lag:search_end + 1]
        peaks, _ = find_peaks(region)
        possible = (peaks + min_lag).astype(int)
        # Add the largest point when a broad/monotonic first lobe has no local
        # maxima; it remains a candidate but gets appropriately modest score.
        if possible.size == 0 and region.size:
            possible = np.array([int(np.argmax(region)) + min_lag])
        for sample_lag in possible:
            rate = fs / float(sample_lag)
            if low <= rate <= high:
                score = max(0.0, float(corr[sample_lag]))
                # Normalize feature evidence to a bounded score and let a
                # strong spectral line dominate coincident autocorrelation.
                score = min(1.0, score)
                rate_scores[rate] = max(rate_scores.get(rate, 0.0), score)
                methods.setdefault(rate, f"{name} autocorrelation")

    if not rate_scores:
        return SymbolRateResult(
            method="bounded periodicity search",
            warnings=["No periodicity peak fell inside the requested rate range"],
        )

    ranked = sorted(rate_scores.items(), key=lambda pair: pair[1], reverse=True)
    selected_rate, best_score = ranked[0]
    # Normalize line strength relative to the median in-band envelope spectrum;
    # autocorrelation scores are already in [0,1].
    selected_method = methods.get(selected_rate, "")
    if selected_method in cyclic_features:
        confidence = _cyclic_coherence(cyclic_features[selected_method], selected_rate, fs)
    else:
        confidence = float(np.clip(best_score, 0.0, 0.75))
    warnings: list[str] = []
    if confidence < 0.8:
        warnings.append("Periodicity evidence is weak; treat the estimate as unreliable")
    return SymbolRateResult(
        estimated_rate=float(selected_rate),
        confidence=confidence,
        method=methods.get(selected_rate, "bounded periodicity search"),
        candidates=[float(rate) for rate, _ in ranked[:max_candidates]],
        warnings=warnings,
    )
