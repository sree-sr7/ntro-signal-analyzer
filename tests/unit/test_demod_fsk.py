import numpy as np
import pytest

from core.dsp.demod_fsk import demodulate_fsk
from tests.modulation_helpers import add_awgn, fsk_waveform, random_bits


FS = 48_000
SPS = 16
TONES = (-3_000.0, 3_000.0)


def test_2fsk_demodulation_round_trips_bits_and_amplitude_scale():
    rng = np.random.default_rng(907)
    bits = random_bits(1024, rng)
    waveform = fsk_waveform(bits, sample_rate=FS, samples_per_symbol=SPS, tones_hz=TONES)

    result = demodulate_fsk(
        2.3 * waveform, sample_rate=FS, samples_per_symbol=SPS, tone_frequencies_hz=TONES
    )

    np.testing.assert_array_equal(result.bits, bits)
    assert result.samples_per_symbol == SPS
    assert result.confidence > 0.9


def test_2fsk_awgn_bit_error_rate_is_bounded():
    rng = np.random.default_rng(1009)
    bits = random_bits(2048, rng)
    waveform = fsk_waveform(bits, sample_rate=FS, samples_per_symbol=SPS, tones_hz=TONES)
    received = add_awgn(waveform, 10.0, rng)

    result = demodulate_fsk(
        received, sample_rate=FS, samples_per_symbol=SPS, tone_frequencies_hz=TONES
    )

    ber = float(np.mean(result.bits != bits))
    assert ber < 0.01, f"2FSK BER={ber:.6f} at 10 dB SNR"


def test_2fsk_demodulation_discards_incomplete_tail_with_warning():
    rng = np.random.default_rng(1103)
    bits = random_bits(32, rng)
    waveform = fsk_waveform(bits, sample_rate=FS, samples_per_symbol=SPS, tones_hz=TONES)

    result = demodulate_fsk(
        np.r_[waveform, 1 + 1j], sample_rate=FS, samples_per_symbol=SPS,
        tone_frequencies_hz=TONES,
    )

    np.testing.assert_array_equal(result.bits, bits)
    assert result.warnings and "trailing samples" in result.warnings[0]


@pytest.mark.parametrize(
    "configuration",
    [
        {"sample_rate": 48_000, "samples_per_symbol": 1, "tone_frequencies_hz": TONES},
        {"sample_rate": 48_000, "samples_per_symbol": 16, "tone_frequencies_hz": (1000, 1000)},
        {"sample_rate": 48_000, "samples_per_symbol": 16, "tone_frequencies_hz": (1000, 24_000)},
    ],
)
def test_2fsk_invalid_configuration_is_rejected(configuration):
    with pytest.raises(ValueError):
        demodulate_fsk(np.ones(128), **configuration)
