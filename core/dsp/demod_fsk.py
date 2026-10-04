"""Noncoherent two-tone correlation demodulator for configured 2FSK."""

from __future__ import annotations

import numpy as np

from core.common.enums import ModulationType
from core.common.models import DemodulationResult
from core.dsp._validation import as_samples, validate_sample_rate
from libs.modulation_library import symbol_indices_to_bits


def demodulate_fsk(
    samples: np.ndarray,
    *,
    sample_rate: float,
    samples_per_symbol: int,
    tone_frequencies_hz: tuple[float, float],
) -> DemodulationResult:
    """Demodulate bit-0/bit-1 tones by per-symbol noncoherent correlation energy.

    Samples are complex baseband at the configured sample rate. The tone pair
    is ordered as ``(bit-0 frequency, bit-1 frequency)`` and may use arbitrary
    positive amplitude scaling. The final incomplete symbol is discarded with
    a warning; no carrier phase estimate is needed.
    """
    values = as_samples(samples)
    fs = validate_sample_rate(sample_rate)
    if not isinstance(samples_per_symbol, (int, np.integer)) or samples_per_symbol < 2:
        raise ValueError("samples_per_symbol must be an integer of at least 2")
    if len(tone_frequencies_hz) != 2:
        raise ValueError("tone_frequencies_hz must contain (bit-0, bit-1) frequencies")
    tones = np.asarray(tone_frequencies_hz, dtype=np.float64)
    if not np.all(np.isfinite(tones)) or np.isclose(tones[0], tones[1]):
        raise ValueError("FSK tones must be finite and distinct")
    if np.any(np.abs(tones) >= fs / 2.0):
        raise ValueError("FSK tone frequencies must lie inside the Nyquist interval")
    result = DemodulationResult(
        modulation=ModulationType.FSK2,
        samples_per_symbol=float(samples_per_symbol),
    )
    if values.size < 2 * samples_per_symbol:
        result.warnings.append("At least two complete FSK symbols are required")
        return result
    symbol_count = values.size // samples_per_symbol
    used_count = symbol_count * samples_per_symbol
    segments = values[:used_count].reshape(symbol_count, samples_per_symbol)
    sample_index = np.arange(samples_per_symbol, dtype=np.float64)
    references = np.exp(-2j * np.pi * tones[:, None] * sample_index[None, :] / fs)
    correlations = segments @ references.T
    energies = np.abs(correlations) ** 2
    indices = np.argmax(energies, axis=1).astype(np.int64)
    margins = np.abs(energies[:, 1] - energies[:, 0]) / np.maximum(
        np.sum(energies, axis=1), np.finfo(float).tiny
    )
    result.symbol_indices = indices
    result.symbol_values = tones[indices]
    result.bits = symbol_indices_to_bits(indices, ModulationType.FSK2)
    result.confidence = float(np.clip(np.mean(margins), 0.0, 1.0))
    result.diagnostics = {
        "symbol_count": symbol_count,
        "mean_bit0_correlation_energy": float(np.mean(energies[:, 0])),
        "mean_bit1_correlation_energy": float(np.mean(energies[:, 1])),
        "mean_energy_margin": float(np.mean(margins)),
        "input_rms_amplitude": float(np.sqrt(np.mean(np.abs(segments) ** 2))),
        "ambiguous_symbol_count": int(np.count_nonzero(margins < 0.05)),
        "invalid_symbol_count": 0,
    }
    if used_count != values.size:
        result.warnings.append(f"Discarded {values.size - used_count} trailing samples from an incomplete symbol")
    return result
