import numpy as np
import pytest

from core.common.models import TimingResult
from core.sync.timing_recovery import recover_timing
from tests.signal_helpers import fractional_delay, psk_symbols, psk_waveform


@pytest.mark.parametrize("sps", [4, 6, 8])
def test_gardner_recovers_fractional_timing_phase(sps):
    rng = np.random.default_rng(801 + sps)
    symbols = psk_symbols(4, 2500, rng)
    waveform = psk_waveform(symbols, sps, rolloff=0.35)
    expected_offset = 0.37 * sps
    delayed = fractional_delay(waveform, expected_offset)
    result = recover_timing(delayed, sps, loop_bandwidth=0.008)

    assert isinstance(result, TimingResult)
    assert result.synchronized_samples.size > 200
    assert result.timing_offset is not None
    phase_error = (result.timing_offset - expected_offset + sps / 2) % sps - sps / 2
    # Linear interpolation and Gardner loop settling leave sub-sample error for these oversampled RRC signals.
    assert abs(phase_error) < 1.5


def test_gardner_handles_awgn_and_rejects_insufficient_data():
    rng = np.random.default_rng(809)
    symbols = psk_symbols(2, 1200, rng)
    waveform = psk_waveform(symbols, 8, rolloff=0.25)
    waveform += 0.08 * (rng.standard_normal(waveform.size) + 1j * rng.standard_normal(waveform.size))
    noisy = recover_timing(waveform, 8)
    short = recover_timing(np.ones(12, dtype=np.complex64), 8)

    assert noisy.synchronized_samples.size > 100
    assert noisy.confidence > 0.0
    assert noisy.timing_offset is not None
    assert min(noisy.timing_offset, 8.0 - noisy.timing_offset) < 1.0
    assert short.synchronized_samples.size == 0 and short.warnings


def test_gardner_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        recover_timing(np.ones(100), 1.5)
    with pytest.raises(ValueError):
        recover_timing(np.ones(100), 4, timing_offset=4)
