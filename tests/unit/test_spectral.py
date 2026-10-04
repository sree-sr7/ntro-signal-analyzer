import numpy as np
import pytest

from core.common.models import SpectralResult
from core.dsp.spectral import analyze_spectrum


def test_single_complex_tone_peak_is_near_generated_frequency():
    sample_rate = 12_000.0
    expected_hz = 1_337.0
    index = np.arange(16_384)
    samples = np.exp(2j * np.pi * expected_hz * index / sample_rate)
    result = analyze_spectrum(samples, sample_rate, nfft=4096, overlap=0.5)

    assert isinstance(result, SpectralResult)
    # A 4096 point transform has 2.93 Hz bin spacing; 6 Hz allows interpolation/binning error.
    assert abs(result.peak_frequency - expected_hz) < 6.0
    assert result.frequencies.size == 4096
    assert result.power.size == 4096
    assert result.occupied_bandwidth < 20.0


def test_two_tone_complex_spectrum_covers_both_tones():
    sample_rate = 16_000.0
    index = np.arange(16_384)
    samples = np.exp(2j * np.pi * 1_200 * index / sample_rate) + 0.7 * np.exp(
        -2j * np.pi * 2_400 * index / sample_rate
    )
    result = analyze_spectrum(samples, sample_rate, nfft=4096)

    assert abs(result.peak_frequency - 1_200) < 5.0
    # The 99% interval should include both separated carriers and their Hann skirts.
    assert 3_300.0 < result.occupied_bandwidth < 3_900.0


def test_tone_plus_noise_has_finite_spectrum_and_nonzero_noise_floor():
    rng = np.random.default_rng(401)
    sample_rate = 8_000.0
    index = np.arange(8192)
    tone = np.exp(2j * np.pi * 800 * index / sample_rate)
    noise = 0.2 * (rng.standard_normal(index.size) + 1j * rng.standard_normal(index.size))
    result = analyze_spectrum(tone + noise, sample_rate, nfft=2048)

    assert result.peak_frequency == pytest.approx(800.0, abs=5.0)
    assert result.noise_floor > 0.0
    assert np.all(np.isfinite(result.power))
    assert result.occupied_bandwidth > 0.0


def test_real_input_uses_one_sided_frequency_axis():
    sample_rate = 10_000.0
    index = np.arange(4096)
    samples = np.cos(2 * np.pi * 1_250 * index / sample_rate)
    result = analyze_spectrum(samples, sample_rate, nfft=1024)

    assert result.frequencies[0] == 0.0
    assert result.frequencies[-1] == sample_rate / 2
    assert result.peak_frequency == pytest.approx(1_250.0, abs=10.0)


def test_empty_short_zero_and_nonfinite_inputs_are_explicit():
    empty = analyze_spectrum(np.array([], dtype=np.complex64), 1_000.0)
    short = analyze_spectrum(np.array([1.0, 0.0, -1.0]), 1_000.0)
    zero = analyze_spectrum(np.zeros(128, dtype=np.complex64), 1_000.0)

    assert empty.peak_frequency is None and empty.warnings
    assert short.peak_frequency is None and short.warnings
    assert zero.peak_frequency is None and zero.noise_floor == 0.0
    with pytest.raises(ValueError, match="finite"):
        analyze_spectrum(np.array([1 + 0j, np.nan + 0j]), 1_000.0)
