"""Nearest-neighbor 16-QAM coherent demodulator."""

from __future__ import annotations

import numpy as np

from core.common.enums import ModulationType
from core.common.models import DemodulationResult
from core.dsp._validation import as_samples
from core.dsp._soft import max_log_llrs
from libs.modulation_library import constellation_for, symbol_indices_to_bits


def demodulate_qam(
    samples: np.ndarray,
    modulation: ModulationType = ModulationType.QAM16,
    *,
    phase_reference: float = 0.0,
    normalize_amplitude: bool = True,
) -> DemodulationResult:
    """Demodulate rectangular Gray-coded 16-QAM synchronized symbol samples.

    When enabled, global RMS normalization handles arbitrary positive gain;
    phase correction is explicit and must come from synchronization. The
    central modulation library defines the ±1/±3 levels and bit labels.
    """
    values = as_samples(samples)
    if modulation is not ModulationType.QAM16:
        raise ValueError("QAM demodulator supports 16QAM only")
    if not np.isfinite(phase_reference):
        raise ValueError("phase_reference must be finite")
    result = DemodulationResult(modulation=modulation)
    if values.size < 4:
        result.warnings.append("At least four synchronized samples are required for 16-QAM decisions")
        return result
    rms = float(np.sqrt(np.mean(np.abs(values) ** 2)))
    if rms <= np.finfo(float).tiny:
        result.warnings.append("Zero-power input; 16-QAM decisions are undefined")
        return result
    constellation = constellation_for(modulation)
    corrected = values.astype(np.complex128, copy=False) * np.exp(-1j * phase_reference)
    normalized = corrected / rms if normalize_amplitude else corrected
    distances = np.abs(normalized[:, None] - constellation[None, :])
    indices = np.argmin(distances, axis=1).astype(np.int64)
    nearest = distances[np.arange(values.size), indices]
    second = np.partition(distances, 1, axis=1)[:, 1]
    margin = (second - nearest) / np.maximum(second + nearest, np.finfo(float).tiny)
    normalized_rms = float(np.sqrt(np.mean(nearest**2)))
    result.symbol_indices = indices
    result.symbol_values = constellation[indices]
    result.bits = symbol_indices_to_bits(indices, modulation)
    result.llrs, estimated_noise_power = max_log_llrs(normalized, constellation, modulation)
    result.confidence = float(np.clip(np.mean(margin) * np.exp(-normalized_rms), 0.0, 1.0))
    result.diagnostics = {
        "symbol_count": int(values.size),
        "mean_decision_distance": float(np.mean(nearest)),
        "normalized_rms": normalized_rms,
        "input_rms_amplitude": rms,
        "soft_llr_sign": "positive favors bit 0",
        "soft_llr_method": "max-log nearest-constellation distances",
        "soft_noise_power_estimate": estimated_noise_power,
        "soft_llr_clipping": [-80.0, 80.0],
        "amplitude_scale_applied": bool(normalize_amplitude),
        "invalid_symbol_count": 0,
        "ambiguous_symbol_count": int(np.count_nonzero((second - nearest) < 1e-3)),
    }
    return result
