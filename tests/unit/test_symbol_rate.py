import numpy as np

from core.common.models import SymbolRateResult
from core.dsp.symbol_rate import estimate_symbol_rate
from tests.signal_helpers import psk_symbols, psk_waveform


def test_bpsk_and_qpsk_symbol_rate_candidates_are_near_generated_rate():
    sample_rate = 48_000.0
    expected_rate = 6_000.0
    for order, seed in ((2, 601), (4, 602)):
        rng = np.random.default_rng(seed)
        symbols = psk_symbols(order, 1800, rng)
        waveform = psk_waveform(symbols, 8, rolloff=0.3)
        result = estimate_symbol_rate(waveform, sample_rate, 3_000.0, 9_000.0)

        assert isinstance(result, SymbolRateResult)
        assert result.estimated_rate is not None
        # A 1/T record gives 6.7 Hz spectral bins; 15% covers baud harmonics and pulse-shape bias.
        assert abs(result.estimated_rate - expected_rate) < 900.0
        assert any(abs(candidate - expected_rate) < 900.0 for candidate in result.candidates)
        assert 0.0 <= result.confidence <= 1.0


def test_fsk_like_frequency_hops_return_bounded_clock_candidate():
    rng = np.random.default_rng(603)
    sample_rate = 32_000.0
    samples_per_symbol = 8
    symbols = 2 * rng.integers(0, 2, 2500) - 1
    discriminator = np.repeat(symbols, samples_per_symbol) * 0.22
    phase = np.cumsum(np.r_[0.0, discriminator[:-1]])
    waveform = np.exp(1j * phase)
    result = estimate_symbol_rate(waveform, sample_rate, 2_000.0, 8_000.0)

    assert result.estimated_rate is not None
    assert abs(result.estimated_rate - 4_000.0) < 800.0
    assert result.method != "unavailable"


def test_rate_search_is_bounded_and_short_input_is_reported():
    short = estimate_symbol_rate(np.ones(10, dtype=np.complex64), 10_000.0, 100.0, 2_000.0)
    result = estimate_symbol_rate(np.ones(2048), 10_000.0, 100.0, 2_000.0)
    rng = np.random.default_rng(604)
    noise = rng.standard_normal(16_000) + 1j * rng.standard_normal(16_000)
    noise_result = estimate_symbol_rate(noise, 48_000.0, 3_000.0, 9_000.0)

    assert short.estimated_rate is None and short.warnings
    assert result.estimated_rate is None or 100.0 <= result.estimated_rate <= 2_000.0
    assert noise_result.warnings and noise_result.confidence < 0.8
