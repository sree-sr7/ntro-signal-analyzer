"""Hard coherent BPSK, QPSK, and 8PSK symbol decisions."""

from __future__ import annotations

import numpy as np

from core.common.enums import ModulationType
from core.common.models import DemodulationResult, SymbolResult
from core.dsp._validation import as_samples
from core.dsp._soft import max_log_llrs
from libs.modulation_library import constellation_for, symbol_indices_to_bits


def decide_psk_symbols(
    samples: np.ndarray,
    modulation: ModulationType,
    *,
    phase_reference: float = 0.0,
    normalize_amplitude: bool = True,
) -> SymbolResult:
    """Return nearest constellation symbols for synchronized PSK samples.

    ``phase_reference`` is the carrier phase to remove, including a selected
    Phase 2 ambiguity candidate when one has been validated. Amplitude
    normalization is per symbol and does not infer a modulation label.
    """
    values = as_samples(samples)
    if modulation not in (ModulationType.BPSK, ModulationType.QPSK, ModulationType.PSK8):
        raise ValueError("PSK demodulator supports BPSK, QPSK, and 8PSK only")
    if not np.isfinite(phase_reference):
        raise ValueError("phase_reference must be finite")
    result = SymbolResult(modulation=modulation)
    if values.size < 4:
        result.warnings.append("At least four synchronized samples are required for PSK decisions")
        return result
    constellation = constellation_for(modulation)
    corrected = values.astype(np.complex128, copy=False) * np.exp(-1j * phase_reference)
    magnitude = np.abs(corrected)
    peak = float(np.max(magnitude))
    valid = magnitude > np.finfo(float).eps * max(peak, 1.0)
    if not np.any(valid):
        result.warnings.append("Zero-power input; PSK decisions are undefined")
        return result
    decision_input = corrected.copy()
    if normalize_amplitude:
        decision_input[valid] /= magnitude[valid]
    distances = np.abs(decision_input[:, None] - constellation[None, :])
    indices = np.argmin(distances, axis=1).astype(np.int64)
    nearest = distances[np.arange(values.size), indices]
    sorted_distances = np.partition(distances, 1, axis=1)
    margins = (sorted_distances[:, 1] - sorted_distances[:, 0]) / np.maximum(
        sorted_distances[:, 1] + sorted_distances[:, 0], np.finfo(float).tiny
    )
    normalized_rms = float(np.sqrt(np.mean(nearest[valid] ** 2)))
    valid_fraction = float(np.mean(valid))
    result.symbol_indices = indices
    result.symbols = constellation[indices]
    result.symbol_count = int(values.size)
    result.confidence = float(np.clip(
        valid_fraction * np.mean(margins[valid]) * np.exp(-normalized_rms), 0.0, 1.0
    ))
    if np.any(~valid):
        result.warnings.append(f"{int(np.count_nonzero(~valid))} zero-amplitude symbols are ambiguous")
    return result


def demodulate_psk(
    samples: np.ndarray,
    modulation: ModulationType,
    *,
    phase_reference: float = 0.0,
    normalize_amplitude: bool = True,
) -> DemodulationResult:
    """Hard-decision demodulate BPSK, QPSK, or 8PSK into Gray-mapped bits."""
    decided = decide_psk_symbols(
        samples,
        modulation,
        phase_reference=phase_reference,
        normalize_amplitude=normalize_amplitude,
    )
    result = DemodulationResult(
        modulation=modulation,
        confidence=decided.confidence,
        warnings=list(decided.warnings),
        symbol_values=decided.symbols,
        symbol_indices=decided.symbol_indices,
        diagnostics={"symbol_count": decided.symbol_count},
    )
    if decided.symbol_count == 0:
        return result
    result.bits = symbol_indices_to_bits(decided.symbol_indices, modulation)
    values = as_samples(samples)
    corrected_for_soft = values.astype(np.complex128, copy=False) * np.exp(-1j * phase_reference)
    magnitudes = np.abs(corrected_for_soft)
    valid_soft = magnitudes > np.finfo(float).eps * max(float(np.max(magnitudes)), 1.0)
    if normalize_amplitude:
        soft_input = corrected_for_soft.copy()
        soft_input[valid_soft] /= magnitudes[valid_soft]
    else:
        soft_input = corrected_for_soft
    constellation = constellation_for(modulation)
    result.llrs, estimated_noise_power = max_log_llrs(
        soft_input[valid_soft], constellation, modulation
    )
    rms = float(np.sqrt(np.mean(np.abs(values) ** 2)))
    if np.any(np.abs(values) > 0):
        corrected = values.astype(np.complex128, copy=False) * np.exp(-1j * phase_reference)
        normalized = corrected / np.maximum(np.abs(corrected), np.finfo(float).tiny) if normalize_amplitude else corrected
        residual = np.abs(normalized - decided.symbols)
        result.diagnostics.update({
            "mean_decision_distance": float(np.mean(residual)),
            "normalized_rms": float(np.sqrt(np.mean(residual**2))),
            "input_rms_amplitude": rms,
            "soft_llr_sign": "positive favors bit 0",
            "soft_llr_method": "max-log nearest-constellation distances",
            "soft_noise_power_estimate": estimated_noise_power,
            "soft_llr_clipping": [-80.0, 80.0],
            "invalid_symbol_count": int(np.count_nonzero(np.abs(values) == 0)),
            "ambiguous_symbol_count": int(np.count_nonzero(
                np.abs(np.partition(np.abs(normalized[:, None] - constellation_for(modulation)[None, :]), 1, axis=1)[:, 1]
                    - np.partition(np.abs(normalized[:, None] - constellation_for(modulation)[None, :]), 1, axis=1)[:, 0]) < 1e-3
            )),
        })
    return result
