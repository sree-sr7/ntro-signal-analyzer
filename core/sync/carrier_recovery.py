"""Feed-forward M-th-power carrier recovery for PSK symbol streams."""

from __future__ import annotations

import numpy as np

from core.common.models import CarrierRecoveryResult
from core.dsp._validation import as_samples, validate_sample_rate


DEFAULT_CARRIER_MIN_CONFIDENCE = 0.15


def recover_carrier(
    samples: np.ndarray,
    sample_rate: float,
    modulation_order: int = 4,
    *,
    constellation_phase_rad: float = 0.0,
    min_confidence: float = DEFAULT_CARRIER_MIN_CONFIDENCE,
) -> CarrierRecoveryResult:
    """Estimate and remove residual PSK phase and frequency by M-th power.

    A least-squares phase slope is fit to unwrapped ``samples ** M`` phase;
    the intercept provides phase modulo ``2π/M`` after removing the optional
    constellation phase anchor. The frequency estimate is likewise ambiguous
    outside ``±sample_rate/(2M)``. The output therefore
    resolves frequency and one phase representative; phase rotations remain
    explicit alternatives from :func:`core.sync.phase_ambiguity`.
    """
    values = as_samples(samples)
    fs = validate_sample_rate(sample_rate)
    if not isinstance(modulation_order, (int, np.integer)) or modulation_order < 2:
        raise ValueError("modulation_order must be an integer of at least 2")
    if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    if not np.isfinite(constellation_phase_rad):
        raise ValueError("constellation_phase_rad must be finite")
    result = CarrierRecoveryResult()
    if not np.iscomplexobj(values):
        result.warnings.append("Carrier recovery requires complex PSK-like baseband samples")
        return result
    if values.size < 8:
        result.warnings.append("Too few samples for carrier recovery")
        return result
    maximum = float(np.max(np.abs(values)))
    if maximum == 0.0:
        result.warnings.append("Zero-power input; carrier recovery is undefined")
        return result
    normalized = values.astype(np.complex128, copy=False) / maximum
    powered = normalized ** int(modulation_order)
    phase = np.unwrap(np.angle(powered))
    index = np.arange(values.size, dtype=np.float64)
    slope, intercept = np.polyfit(index, phase, 1)
    fitted = slope * index + intercept
    residual = phase - fitted
    confidence = float(np.abs(np.mean(np.exp(1j * residual))))
    result.confidence = confidence
    if confidence < min_confidence:
        result.warnings.append("PSK carrier phase is incoherent; recovery rejected")
        return result
    offset_hz = float(slope * fs / (2.0 * np.pi * int(modulation_order)))
    phase_offset = float(np.angle(np.exp(
        1j * (intercept / int(modulation_order) - constellation_phase_rad)
    )))
    rotation = np.exp(-1j * (2.0 * np.pi * offset_hz * index / fs + phase_offset))
    if values.dtype == np.dtype(np.complex64):
        output_dtype = np.complex64
    else:
        output_dtype = np.complex128
    result.frequency_offset = offset_hz
    result.phase_offset = phase_offset
    result.recovered_samples = np.asarray(values * rotation, dtype=output_dtype)
    return result
