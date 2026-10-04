"""Deterministic synthetic waveform helpers shared by Phase 2 tests."""

import numpy as np
from scipy.signal import fftconvolve

from core.dsp.pulse_shaping import root_raised_cosine


def psk_symbols(order: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Generate unit-energy M-PSK symbols from an explicit RNG."""
    indices = rng.integers(0, order, count)
    return np.exp(2j * np.pi * indices / order)


def psk_waveform(
    symbols: np.ndarray,
    samples_per_symbol: int,
    *,
    rolloff: float = 0.25,
    span: int = 8,
) -> np.ndarray:
    """Upsample and root-raised-cosine shape a symbol sequence."""
    impulses = np.zeros(symbols.size * samples_per_symbol, dtype=np.complex128)
    impulses[::samples_per_symbol] = symbols
    taps = root_raised_cosine(rolloff, samples_per_symbol, span)
    return fftconvolve(impulses, taps, mode="same")


def add_awgn(
    samples: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add deterministic complex AWGN at the requested average SNR."""
    signal_power = float(np.mean(np.abs(samples) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(samples.size) + 1j * rng.standard_normal(samples.size)
    )
    return samples + noise


def fractional_delay(samples: np.ndarray, delay: float) -> np.ndarray:
    """Apply a linear-interpolated fractional sample delay."""
    index = np.arange(samples.size, dtype=np.float64)
    positions = index - delay
    return (
        np.interp(positions, index, samples.real, left=0.0, right=0.0)
        + 1j * np.interp(positions, index, samples.imag, left=0.0, right=0.0)
    )
