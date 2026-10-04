import numpy as np
import pytest

from core.common.models import CFOResult
from core.sync.cfo_estimator import estimate_cfo
from tests.signal_helpers import add_awgn, psk_symbols


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("cfo_hz", [0.0, 730.0, -1_150.0])
@pytest.mark.parametrize("snr_db", [5.0, 15.0])
def test_mth_power_cfo_estimation_error(order, cfo_hz, snr_db):
    rng = np.random.default_rng(701 + order + int(abs(cfo_hz)) + int(snr_db))
    sample_rate = 64_000.0
    sps = 8
    symbols = psk_symbols(order, 1500, rng)
    samples = np.repeat(symbols, sps)
    index = np.arange(samples.size)
    samples *= np.exp(2j * np.pi * cfo_hz * index / sample_rate)
    samples = add_awgn(samples, snr_db, rng)
    result = estimate_cfo(samples, sample_rate, order)

    assert isinstance(result, CFOResult)
    assert result.estimated_offset_hz is not None
    # At 5 dB the averaged M-th-power phase increment remains well inside the 100 Hz tolerance.
    assert abs(result.estimated_offset_hz - cfo_hz) < 100.0
    assert result.confidence > 0.12


def test_cfo_rejects_zero_noise_only_and_nonfinite_inputs():
    zero = estimate_cfo(np.zeros(64, dtype=np.complex64), 8_000.0)
    rng = np.random.default_rng(799)
    noise = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    noise_result = estimate_cfo(noise, 8_000.0)

    assert zero.estimated_offset_hz is None and zero.warnings
    assert noise_result.estimated_offset_hz is None and noise_result.warnings
    with pytest.raises(ValueError, match="finite"):
        estimate_cfo(np.array([1 + 0j, np.inf + 0j]), 8_000.0)
