"""Bit conversion, CRC, sync correlation, and simple frame analysis."""

from .correlator import SyncCorrelationResult, correlate_sync_word
from .crc import append_crc16, compute_crc16, verify_crc16
from .extractor import (
    bits_to_bytes,
    bytes_to_bits,
    format_bits_binary,
    format_bits_hex,
    pad_bits,
    trim_bits,
    validate_binary_bits,
)
from .frame_analyzer import FrameAnalysisResult, PeriodicityResult, analyze_frames, estimate_periodicity

__all__ = [
    "FrameAnalysisResult",
    "PeriodicityResult",
    "SyncCorrelationResult",
    "analyze_frames",
    "append_crc16",
    "bits_to_bytes",
    "bytes_to_bits",
    "compute_crc16",
    "correlate_sync_word",
    "estimate_periodicity",
    "format_bits_binary",
    "format_bits_hex",
    "pad_bits",
    "trim_bits",
    "validate_binary_bits",
    "verify_crc16",
]
