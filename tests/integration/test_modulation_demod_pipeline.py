import numpy as np
import pytest

from core.common.enums import ModulationType
from core.dsp.demodulation import demodulate
from core.ml.classifier import classify_classical
from libs.modulation_library import bits_to_symbols
from tests.modulation_helpers import add_awgn, fsk_waveform, random_bits


SUPPORTED = [
    (ModulationType.BPSK, 1),
    (ModulationType.QPSK, 2),
    (ModulationType.PSK8, 3),
    (ModulationType.QAM16, 4),
]


@pytest.mark.parametrize("modulation,width", SUPPORTED)
def test_synchronized_symbol_rate_pipeline_classifies_and_demodulates(modulation, width):
    rng = np.random.default_rng(1201 + width)
    bits = random_bits(1024 * width, rng)
    samples = bits_to_symbols(bits, modulation)

    classification = classify_classical(samples)
    recovered = demodulate(samples, modulation)

    assert classification.modulation is modulation
    np.testing.assert_array_equal(recovered.bits, bits)


def test_synchronized_2fsk_pipeline_classifies_and_demodulates():
    rng = np.random.default_rng(1301)
    bits = random_bits(1024, rng)
    samples = fsk_waveform(
        bits, sample_rate=48_000, samples_per_symbol=16, tones_hz=(-3_000, 3_000)
    )

    classification = classify_classical(samples, samples_per_symbol=16)
    recovered = demodulate(
        samples, ModulationType.FSK2, sample_rate=48_000, samples_per_symbol=16,
        tone_frequencies_hz=(-3_000, 3_000),
    )

    assert classification.modulation is ModulationType.FSK2
    np.testing.assert_array_equal(recovered.bits, bits)


@pytest.mark.parametrize("modulation,width", SUPPORTED)
def test_pipeline_demodulates_noisy_constellation_samples(modulation, width):
    rng = np.random.default_rng(1409 + width)
    bits = random_bits(2048 * width, rng)
    samples = bits_to_symbols(bits, modulation)
    snr = 21.0 if modulation is ModulationType.QAM16 else 17.0
    noisy = add_awgn(samples, snr, rng)

    recovered = demodulate(noisy, modulation)

    ber = float(np.mean(recovered.bits != bits))
    assert ber < 0.03, f"{modulation.value} pipeline BER={ber:.6f} at {snr:.1f} dB SNR"


def test_pipeline_demodulates_noisy_fsk_samples():
    rng = np.random.default_rng(1511)
    bits = random_bits(1024, rng)
    samples = fsk_waveform(
        bits, sample_rate=48_000, samples_per_symbol=16, tones_hz=(-3_000, 3_000)
    )
    noisy = add_awgn(samples, 10.0, rng)

    recovered = demodulate(
        noisy, ModulationType.FSK2, sample_rate=48_000, samples_per_symbol=16,
        tone_frequencies_hz=(-3_000, 3_000),
    )

    ber = float(np.mean(recovered.bits != bits))
    assert ber < 0.01, f"2FSK pipeline BER={ber:.6f} at 10 dB SNR"


def test_pipeline_returns_unknown_for_noise_and_rejects_unsupported_modulation():
    rng = np.random.default_rng(1601)
    noise = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)) / np.sqrt(2.0)

    classification = classify_classical(noise)

    assert classification.modulation is ModulationType.UNKNOWN
    assert classification.warnings
    with pytest.raises(ValueError, match="unsupported"):
        demodulate(np.ones(32), ModulationType.UNKNOWN)
