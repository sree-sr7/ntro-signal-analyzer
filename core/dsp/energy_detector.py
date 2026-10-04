"""Adaptive windowed-energy activity detection."""

from __future__ import annotations

import numpy as np

from core.common.models import DetectionResult
from ._validation import as_samples


def detect_activity(
    samples: np.ndarray,
    *,
    threshold_db: float = 6.0,
    window_size: int = 128,
    hop_size: int | None = None,
) -> DetectionResult:
    """Detect regions whose mean power exceeds an estimated background level.

    The background is the median of frame powers at or below their median,
    which tolerates a minority of active frames without assuming an absolute
    signal amplitude. ``estimated_snr`` is reported in dB. Bounds are half-open
    sample indices; regions closer than one window can merge at this resolution.
    """
    values = as_samples(samples)
    if not np.isfinite(threshold_db) or threshold_db <= 0:
        raise ValueError("threshold_db must be finite and positive")
    if not isinstance(window_size, (int, np.integer)) or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    hop = int(hop_size) if hop_size is not None else max(1, int(window_size) // 4)
    if hop < 1:
        raise ValueError("hop_size must be a positive integer")
    if values.size == 0:
        return DetectionResult(False, warnings=["Empty input; no activity detected"])

    window = min(int(window_size), int(values.size))
    if window < window_size:
        hop = min(hop, window)
    power = np.abs(values) ** 2
    # Cumulative sums avoid materializing an overlapping-window matrix.
    cumulative = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    starts = np.arange(0, values.size - window + 1, hop, dtype=np.int64)
    if starts.size == 0 or starts[-1] != values.size - window:
        starts = np.append(starts, values.size - window)
    frame_power = (cumulative[starts + window] - cumulative[starts]) / window
    median = float(np.median(frame_power))
    background_frames = frame_power[frame_power <= median]
    noise_floor = float(np.median(background_frames)) if background_frames.size else median

    if not np.any(power > 0):
        return DetectionResult(
            False, noise_floor=0.0, confidence=1.0,
            warnings=["Input has zero power; no activity detected"],
        )

    ratio = 10.0 ** (threshold_db / 10.0)
    # Scale to the data itself when the background estimate is exactly zero,
    # while keeping exact silence inactive. This is relative, not an amplitude
    # threshold, and handles noiseless synthetic bursts.
    reference = noise_floor if noise_floor > 0 else float(np.max(frame_power)) * 1e-12
    active = frame_power > reference * ratio
    if not np.any(active):
        return DetectionResult(
            False, noise_floor=noise_floor, confidence=0.8,
            warnings=["No frames exceeded the adaptive energy threshold"],
        )

    edge = np.diff(np.r_[False, active, False].astype(np.int8))
    region_starts = np.flatnonzero(edge == 1)
    region_ends = np.flatnonzero(edge == -1)
    regions = [
        (int(starts[first]), int(min(values.size, starts[last - 1] + window)))
        for first, last in zip(region_starts, region_ends)
    ]
    active_power = frame_power[active]
    signal_level = float(np.median(active_power))
    snr_db = (
        float(10.0 * np.log10(signal_level / noise_floor))
        if noise_floor > 0 and signal_level > 0
        else None
    )
    contrast_db = max(0.0, snr_db - threshold_db) if snr_db is not None else 40.0
    confidence = float(np.clip(0.55 + contrast_db / 40.0, 0.0, 0.99))
    warnings: list[str] = []
    if len(regions) > 1:
        warnings.append(f"Detected {len(regions)} separated activity regions")
    if window < window_size:
        warnings.append("Input shorter than configured window; used the full input")
    return DetectionResult(
        True,
        start_index=regions[0][0],
        end_index=regions[-1][1],
        estimated_snr=snr_db,
        noise_floor=noise_floor,
        confidence=confidence,
        warnings=warnings,
        activity_regions=regions,
    )
