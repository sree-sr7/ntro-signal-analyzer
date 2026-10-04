"""Scale-normalized features for bounded modulation classification."""

from __future__ import annotations

import numpy as np

from core.common.enums import ModulationType
from core.common.models import ModulationFeatureResult
from core.dsp._validation import as_samples
from libs.modulation_library import constellation_for


FEATURE_NAMES = (
    "amplitude_coefficient_of_variation",
    "amplitude_kurtosis",
    "normalized_moment_2",
    "normalized_moment_4",
    "normalized_moment_8",
    "psk2_residual",
    "psk2_phase_entropy",
    "psk4_residual",
    "psk4_phase_entropy",
    "psk8_residual",
    "psk8_phase_entropy",
    "qam16_residual",
    "fsk_quality",
    "fsk_tone_separation_norm",
    "fsk_frequency_repeat_fraction",
    "spectral_entropy",
    "spectral_peak_fraction",
)


def _entropy(labels: np.ndarray, count: int) -> float:
    """Return normalized discrete entropy for integer cluster labels."""
    probabilities = np.bincount(labels, minlength=count).astype(np.float64)
    probabilities /= max(float(np.sum(probabilities)), 1.0)
    nonzero = probabilities > 0
    return float(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])) / np.log(count))


def _psk_metrics(samples: np.ndarray, order: int) -> tuple[float, float]:
    """Return phase-insensitive normalized decision residual and label entropy."""
    constellation = constellation_for({2: ModulationType.BPSK, 4: ModulationType.QPSK,
                                        8: ModulationType.PSK8}[order])
    magnitude = np.abs(samples)
    usable = magnitude > np.finfo(np.float64).eps * max(float(np.max(magnitude)), 1.0)
    if not np.any(usable):
        return 1.0, 0.0
    values = samples[usable]
    unit = values / np.abs(values)
    anchor = np.angle(np.mean(constellation ** order)) / order
    observed = np.angle(np.mean(unit ** order)) / order
    phase_rotation = np.angle(np.exp(1j * order * (observed - anchor))) / order
    normalized = values / np.sqrt(np.mean(np.abs(samples) ** 2))
    corrected = normalized * np.exp(-1j * phase_rotation)
    distances = np.abs(corrected[:, None] - constellation[None, :])
    indices = np.argmin(distances, axis=1)
    residual = float(np.sqrt(np.mean(distances[np.arange(indices.size), indices] ** 2)))
    return residual, _entropy(indices, order)


def _qam16_residual(samples: np.ndarray) -> float:
    """Return nearest 16-QAM RMS residual after a fourth-moment phase fit."""
    constellation = constellation_for(ModulationType.QAM16)
    normalized = samples / np.sqrt(np.mean(np.abs(samples) ** 2))
    # The fourth moment of square 16-QAM has a nonzero reference phase. Its
    # ratio to the observed moment estimates carrier rotation modulo π/2,
    # exactly the symmetry of this constellation. Unlike an angle grid, this
    # keeps the residual continuous and invariant under arbitrary phase/gain.
    reference_moment = np.mean(constellation**4)
    observed_moment = np.mean(normalized**4)
    if abs(observed_moment) <= np.finfo(float).eps:
        return 1.0
    rotation = float(np.angle(observed_moment / reference_moment) / 4.0)
    corrected = normalized * np.exp(-1j * rotation)
    distances = np.abs(corrected[:, None] - constellation[None, :])
    nearest = np.min(distances, axis=1)
    return float(np.sqrt(np.mean(nearest**2)))


def _fsk_metrics(
    samples: np.ndarray,
    samples_per_symbol: float,
) -> tuple[float, float, float]:
    """Estimate two-tone phase-increment clustering and symbol dwell quality."""
    if samples_per_symbol < 2 or samples.size < 4:
        return 0.0, 0.0, 0.0
    increments = np.angle(samples[1:] * np.conjugate(samples[:-1]))
    finite = increments[np.isfinite(increments)]
    if finite.size < 4 or float(np.std(finite)) <= np.finfo(float).eps:
        return 0.0, 0.0, 0.0
    centers = np.percentile(finite, [25.0, 75.0]).astype(np.float64)
    labels = np.zeros(finite.size, dtype=np.int8)
    for _ in range(8):
        labels = (np.abs(finite - centers[1]) < np.abs(finite - centers[0])).astype(np.int8)
        counts = np.bincount(labels, minlength=2)
        if np.any(counts == 0):
            return 0.0, 0.0, 0.0
        centers = np.array([np.mean(finite[labels == index]) for index in range(2)])
    separation = float(abs(centers[1] - centers[0]))
    total_variance = float(np.var(finite))
    within_variance = float(np.mean((finite - centers[labels]) ** 2))
    balance = float(2.0 * np.min(np.bincount(labels, minlength=2)) / finite.size)
    repeat_fraction = float(np.mean(np.abs(np.diff(finite)) < max(separation * 0.15, 1e-8)))
    quality = float(np.clip(balance * repeat_fraction * (1.0 - within_variance / max(total_variance, 1e-12)), 0, 1))
    return quality, separation / np.sqrt(total_variance), repeat_fraction


def extract_modulation_features(
    samples: np.ndarray,
    *,
    samples_per_symbol: float = 1.0,
) -> ModulationFeatureResult:
    """Extract finite, scale-normalized features from synchronized samples.

    PSK/QAM metrics assume symbol-rate samples. For FSK, provide the waveform's
    samples per symbol (at least two) so the two instantaneous-frequency states
    and their dwell structure can be measured. Absolute input amplitude is not
    used as a classifier feature. Non-finite values raise ``ValueError``.
    """
    values = as_samples(samples)
    sps = float(samples_per_symbol)
    if not np.isfinite(sps) or sps < 1:
        raise ValueError("samples_per_symbol must be finite and at least 1")
    result = ModulationFeatureResult(feature_names=FEATURE_NAMES, sample_count=values.size)
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    if values.size < 16:
        result.values = features
        result.warnings.append("At least 16 synchronized samples are required for useful features")
        return result
    amplitude = np.abs(values).astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(amplitude**2)))
    if rms <= np.finfo(np.float64).tiny:
        result.values = features
        result.warnings.append("Zero-power input; modulation features are undefined")
        return result

    normalized = values.astype(np.complex128, copy=False) / rms
    mean_amplitude = float(np.mean(amplitude))
    amplitude_variance = float(np.var(amplitude))
    amplitude_cv = float(np.sqrt(amplitude_variance) / max(mean_amplitude, np.finfo(float).tiny))
    amplitude_kurtosis = (
        float(np.mean((amplitude - mean_amplitude) ** 4) / amplitude_variance**2)
        if amplitude_variance > np.finfo(float).eps * rms**4
        else 0.0
    )
    unit = normalized / np.maximum(np.abs(normalized), np.finfo(float).tiny)
    moments = [float(abs(np.mean(unit**order))) for order in (2, 4, 8)]
    psk_metrics = [_psk_metrics(normalized, order) for order in (2, 4, 8)]
    qam_residual = _qam16_residual(normalized)
    fsk_quality, fsk_separation, fsk_repeat = _fsk_metrics(normalized, sps)

    spectrum = np.fft.fft(normalized)
    spectral_power = np.abs(spectrum) ** 2
    spectral_total = float(np.sum(spectral_power))
    probabilities = spectral_power / max(spectral_total, np.finfo(float).tiny)
    positive = probabilities > 0
    spectral_entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])) / np.log(values.size))
    spectral_peak_fraction = float(np.max(probabilities))

    features[:] = [
        amplitude_cv,
        amplitude_kurtosis,
        *moments,
        psk_metrics[0][0], psk_metrics[0][1],
        psk_metrics[1][0], psk_metrics[1][1],
        psk_metrics[2][0], psk_metrics[2][1],
        qam_residual,
        fsk_quality,
        fsk_separation,
        fsk_repeat,
        spectral_entropy,
        spectral_peak_fraction,
    ]
    result.values = features
    result.diagnostics = {
        "rms_amplitude": rms,
        "mean_amplitude": mean_amplitude,
        "phase_reference_rad": float(np.angle(np.mean(unit**4)) / 4.0),
    }
    if not np.all(np.isfinite(features)):
        raise ArithmeticError("Feature extraction produced a non-finite value")
    return result
