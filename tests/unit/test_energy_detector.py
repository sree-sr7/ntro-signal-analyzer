import numpy as np
import pytest

from core.common.models import DetectionResult
from core.dsp.energy_detector import detect_activity


def test_detects_signal_burst_over_complex_noise():
    rng = np.random.default_rng(19)
    samples = 0.15 * (rng.standard_normal(3000) + 1j * rng.standard_normal(3000))
    index = np.arange(900)
    samples[1000:1900] += np.exp(2j * np.pi * 0.07 * index)
    result = detect_activity(samples, window_size=128, threshold_db=6.0)

    assert isinstance(result, DetectionResult)
    assert result.detected
    assert result.start_index is not None and 850 <= result.start_index <= 1050
    assert result.end_index is not None and 1850 <= result.end_index <= 2050
    assert result.estimated_snr is not None and result.estimated_snr > 6.0


def test_noise_only_and_zero_input_do_not_trigger():
    rng = np.random.default_rng(20)
    noise = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
    noise_result = detect_activity(noise, window_size=128)
    zero_result = detect_activity(np.zeros(256, dtype=np.complex64))

    assert not noise_result.detected
    assert noise_result.noise_floor > 0.0
    assert not zero_result.detected
    assert zero_result.noise_floor == 0.0


def test_weak_relative_burst_and_separated_regions():
    rng = np.random.default_rng(21)
    noise = 0.3 * (rng.standard_normal(4000) + 1j * rng.standard_normal(4000))
    noise[500:850] += 0.9 * (rng.standard_normal(350) + 1j * rng.standard_normal(350))
    noise[2600:2950] += 0.9 * (rng.standard_normal(350) + 1j * rng.standard_normal(350))
    result = detect_activity(noise, window_size=64, threshold_db=4.0)

    assert result.detected
    assert len(result.activity_regions) == 2
    assert abs(result.activity_regions[0][0] - 500) < 80
    assert abs(result.activity_regions[1][0] - 2600) < 80
    assert result.warnings


def test_empty_short_and_invalid_inputs():
    empty = detect_activity(np.array([], dtype=np.float32))
    short = detect_activity(np.ones(3), window_size=128)
    assert not empty.detected and empty.warnings
    # A constant three-sample record has no background interval from which to infer activity.
    assert short.warnings
    with pytest.raises(ValueError, match="finite"):
        detect_activity(np.array([1.0, np.inf]))
