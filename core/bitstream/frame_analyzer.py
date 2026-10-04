"""Simple sync-based bit framing and periodicity measurements."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .correlator import SyncCorrelationResult, correlate_sync_word
from .extractor import validate_binary_bits


@dataclass
class PeriodicityResult:
    """Strongest normalized binary autocorrelation lag in a bounded range."""

    period_bits: int | None = None
    peak_score: float = 0.0
    candidate_scores: list[tuple[int, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FrameAnalysisResult:
    """Sync location and caller-sized header/payload views for one frame."""

    sync_result: SyncCorrelationResult | None = None
    sync_index: int | None = None
    frame_length_estimate: int | None = None
    header_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    payload_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    warnings: list[str] = field(default_factory=list)


def estimate_periodicity(
    bits: np.ndarray,
    *,
    min_period: int = 1,
    max_period: int | None = None,
    candidate_count: int = 3,
) -> PeriodicityResult:
    """Estimate a bit repetition period using normalized bipolar autocorrelation.

    This reports the strongest measured lag, not an inferred protocol or a
    guaranteed frame boundary.
    """
    values = validate_binary_bits(bits)
    if not isinstance(min_period, (int, np.integer)) or isinstance(min_period, bool) or min_period < 1:
        raise ValueError("min_period must be a positive integer")
    if max_period is not None and (
        not isinstance(max_period, (int, np.integer)) or isinstance(max_period, bool) or max_period < min_period
    ):
        raise ValueError("max_period must be an integer at least min_period")
    if not isinstance(candidate_count, (int, np.integer)) or isinstance(candidate_count, bool) or candidate_count < 1:
        raise ValueError("candidate_count must be a positive integer")
    result = PeriodicityResult()
    if values.size < 4:
        result.warnings.append("At least four bits are required to estimate periodicity")
        return result
    maximum = min(values.size // 2, int(max_period) if max_period is not None else values.size // 2)
    if min_period > maximum:
        result.warnings.append("No candidate period fits within the available bit stream")
        return result

    bipolar = 2.0 * values.astype(np.float64) - 1.0
    bipolar -= np.mean(bipolar)
    energy = float(np.dot(bipolar, bipolar))
    if energy <= np.finfo(float).eps:
        result.warnings.append("A constant bit stream has no measurable periodicity")
        return result
    fft_size = 1 << (2 * values.size - 1).bit_length()
    spectrum = np.fft.rfft(bipolar, n=fft_size)
    autocorrelation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:values.size]
    cumulative_energy = np.concatenate(([0.0], np.cumsum(bipolar * bipolar)))
    lags: list[tuple[int, float]] = []
    for lag in range(int(min_period), maximum + 1):
        left_energy = float(cumulative_energy[values.size - lag])
        right_energy = float(cumulative_energy[values.size] - cumulative_energy[lag])
        denominator = np.sqrt(left_energy * right_energy)
        score = float(autocorrelation[lag] / denominator) if denominator > 0 else 0.0
        lags.append((lag, float(np.clip(score, -1.0, 1.0))))
    # Scores within 1e-9 are ties; prefer the shortest (fundamental) period so
    # FFT rounding cannot make a multiple of the true period win.
    ranked = sorted(lags, key=lambda item: (-round(item[1], 9), item[0]))
    result.period_bits, result.peak_score = ranked[0]
    result.candidate_scores = ranked[:int(candidate_count)]
    return result


def analyze_frames(
    bits: np.ndarray,
    *,
    sync_word: np.ndarray | None = None,
    header_length: int = 0,
    payload_length: int | None = None,
    sync_threshold: float = 0.9,
    periodicity_max_lag: int | None = None,
) -> FrameAnalysisResult:
    """Analyze a bit stream and optionally split fields after its sync word.

    ``header_length`` counts header bits after the sync word. The returned
    ``header_bits`` contain those header bits only; ``payload_bits`` begin after
    the sync word and header. A supplied ``payload_length`` limits the payload;
    otherwise all remaining bits are returned. Repeated above-threshold sync
    hits yield a median spacing as a frame-length estimate. With fewer than two
    hits, the bounded autocorrelation estimate is reported instead.
    """
    values = validate_binary_bits(bits)
    if not isinstance(header_length, (int, np.integer)) or isinstance(header_length, bool) or header_length < 0:
        raise ValueError("header_length must be a nonnegative integer")
    if payload_length is not None and (
        not isinstance(payload_length, (int, np.integer)) or isinstance(payload_length, bool) or payload_length < 0
    ):
        raise ValueError("payload_length must be a nonnegative integer or None")
    if not np.isfinite(sync_threshold) or not -1.0 <= sync_threshold <= 1.0:
        raise ValueError("sync_threshold must be between -1 and 1")

    periodicity = estimate_periodicity(values, max_period=periodicity_max_lag)
    result = FrameAnalysisResult(frame_length_estimate=periodicity.period_bits)
    result.warnings.extend(periodicity.warnings)
    if sync_word is None:
        result.warnings.append("No sync word supplied; header/payload split is unavailable")
        return result

    sync = validate_binary_bits(sync_word)
    correlation = correlate_sync_word(values, sync)
    result.sync_result = correlation
    result.warnings.extend(correlation.warnings)
    if correlation.peak_index is None or correlation.peak_score < sync_threshold:
        result.warnings.append("No sync-word correlation peak met the configured threshold")
        return result

    hit_indices = np.flatnonzero(correlation.scores >= sync_threshold)
    if hit_indices.size >= 2:
        # False sync-like substrings may repeat at a fixed offset inside every
        # frame. Compare adjacent as well as short multi-hit spacings so the
        # frame period can win over a recurring intra-frame alias.
        spacing_values: list[int] = []
        maximum_hit_stride = min(8, int(hit_indices.size) - 1)
        for stride in range(1, maximum_hit_stride + 1):
            spacing_values.extend(
                (hit_indices[stride:] - hit_indices[:-stride]).astype(int).tolist()
            )
        counts = Counter(spacing_values)
        result.frame_length_estimate = min(
            (spacing for spacing, count in counts.items() if count == max(counts.values()))
        )
    result.sync_index = correlation.peak_index
    fields_start = result.sync_index + sync.size
    header_end = fields_start + int(header_length)
    if header_end > values.size:
        result.warnings.append("Stream ended before the requested header length")
        return result
    result.header_bits = values[fields_start:header_end].copy()
    payload_end = values.size if payload_length is None else header_end + int(payload_length)
    if payload_end > values.size:
        result.warnings.append("Stream ended before the requested payload length")
        result.payload_bits = values[header_end:].copy()
        return result
    result.payload_bits = values[header_end:payload_end].copy()
    return result
