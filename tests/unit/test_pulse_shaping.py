import numpy as np
import pytest

from core.dsp.pulse_shaping import matched_filter, root_raised_cosine


def test_rrc_coefficients_are_symmetric_and_energy_normalized():
    taps = root_raised_cosine(0.35, 8, 10)
    assert taps.size == 81
    np.testing.assert_allclose(taps, taps[::-1], rtol=0, atol=1e-14)
    assert np.sum(taps**2) == pytest.approx(1.0, abs=1e-13)


def test_matched_filter_recovers_center_of_synthetic_shaped_pulse():
    sps = 6
    taps = root_raised_cosine(0.25, sps, 8)
    impulses = np.zeros(2048, dtype=np.complex64)
    impulses[1024] = 1.0 + 0.5j
    shaped = np.convolve(impulses, taps.astype(np.float32), mode="same").astype(np.complex64)
    filtered = matched_filter(shaped, taps.astype(np.float32))

    assert filtered.dtype == np.complex64
    assert filtered.size == shaped.size
    assert abs(filtered[1024]) > abs(shaped[1024])
    assert abs(filtered[1024] - (1.0 + 0.5j)) < 0.02


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rolloff": -0.1, "samples_per_symbol": 8, "span": 8},
        {"rolloff": 1.1, "samples_per_symbol": 8, "span": 8},
        {"rolloff": 0.25, "samples_per_symbol": 1, "span": 8},
        {"rolloff": 0.25, "samples_per_symbol": 8, "span": 0},
    ],
)
def test_rrc_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        root_raised_cosine(**kwargs)


def test_matched_filter_rejects_empty_coefficients_and_nonfinite_data():
    with pytest.raises(ValueError, match="empty"):
        matched_filter(np.ones(10), np.array([]))
    with pytest.raises(ValueError, match="finite"):
        matched_filter(np.array([1.0, np.nan]), np.ones(3))
