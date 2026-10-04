"""Deterministic Phase 3 modulation waveform helpers."""

import numpy as np

from core.common.enums import ModulationType
from libs.modulation_library import bits_to_symbols


def random_bits(count: int, rng: np.random.Generator) -> np.ndarray:
    """Return a deterministic flat uint8 bit vector from the provided RNG."""
    return rng.integers(0, 2, size=count, dtype=np.uint8)


def fsk_waveform(
    bits: np.ndarray,
    *,
    sample_rate: float,
    samples_per_symbol: int,
    tones_hz: tuple[float, float],
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate continuous-phase rectangular 2FSK with configured tones."""
    states = bits_to_symbols(bits, ModulationType.FSK2)
    frequencies = np.repeat(np.where(states < 0, tones_hz[0], tones_hz[1]), samples_per_symbol)
    phase = np.r_[0.0, np.cumsum(2.0 * np.pi * frequencies[:-1] / sample_rate)]
    return amplitude * np.exp(1j * phase)


def add_awgn(
    samples: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add complex AWGN at the requested average sample SNR."""
    power = float(np.mean(np.abs(samples) ** 2))
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(samples.size) + 1j * rng.standard_normal(samples.size)
    )
    return samples + noise
