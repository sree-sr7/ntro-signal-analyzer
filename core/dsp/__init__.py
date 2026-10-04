"""Digital signal-processing foundations."""

from .energy_detector import detect_activity
from .demod_fsk import demodulate_fsk
from .demod_psk import decide_psk_symbols, demodulate_psk
from .demod_qam import demodulate_qam
from .demodulation import demodulate
from .modulation_features import extract_modulation_features
from .pulse_shaping import matched_filter, root_raised_cosine
from .spectral import analyze_spectrum
from .symbol_rate import estimate_symbol_rate

__all__ = [
    "analyze_spectrum",
    "decide_psk_symbols",
    "demodulate",
    "demodulate_fsk",
    "demodulate_psk",
    "demodulate_qam",
    "detect_activity",
    "estimate_symbol_rate",
    "extract_modulation_features",
    "matched_filter",
    "root_raised_cosine",
]
