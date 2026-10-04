import numpy as np
import pytest

from core.common.enums import ModulationType
from core.dsp.modulation_features import FEATURE_NAMES, extract_modulation_features
from libs.modulation_library import bits_to_symbols
from tests.modulation_helpers import fsk_waveform, random_bits


@pytest.mark.parametrize(
    "modulation,width",
    [
        (ModulationType.BPSK, 1),
        (ModulationType.QPSK, 2),
        (ModulationType.PSK8, 3),
        (ModulationType.QAM16, 4),
    ],
)
def test_features_are_scale_invariant_for_symbol_rate_constellations(modulation, width):
    rng = np.random.default_rng(41 + width)
    symbols = bits_to_symbols(random_bits(600 * width, rng), modulation)

    reference = extract_modulation_features(symbols)
    scaled = extract_modulation_features(symbols * (2.75 * np.exp(0.31j)))

    assert reference.feature_names == FEATURE_NAMES
    assert np.all(np.isfinite(reference.values))
    # On a noiseless QPSK constellation the BPSK (psk2) decision labels sit on
    # exact ties, so rounding at ~1e-16 decides them: that one feature is only
    # stable to ~3e-3 there. Every other feature must match strictly.
    tie_feature = FEATURE_NAMES.index("psk2_phase_entropy")
    strict = np.ones(len(FEATURE_NAMES), dtype=bool)
    if modulation is ModulationType.QPSK:
        strict[tie_feature] = False
        np.testing.assert_allclose(
            scaled.values[tie_feature], reference.values[tie_feature], atol=5e-3
        )
    np.testing.assert_allclose(
        scaled.values[strict], reference.values[strict], rtol=1e-7, atol=1e-8
    )


def test_features_capture_configured_two_tone_fsk_quality_and_scale():
    rng = np.random.default_rng(53)
    bits = random_bits(512, rng)
    samples = fsk_waveform(
        bits, sample_rate=48_000, samples_per_symbol=16, tones_hz=(-3_000, 3_000)
    )

    result = extract_modulation_features(samples, samples_per_symbol=16)
    scaled = extract_modulation_features(0.2 * samples, samples_per_symbol=16)
    features = result.as_dict()

    assert features["fsk_quality"] > 0.8
    assert features["fsk_tone_separation_norm"] > 1.5
    # Instantaneous phase increments involve floating point angle operations;
    # allow the resulting roundoff while requiring scale stability.
    np.testing.assert_allclose(scaled.values, result.values, rtol=0.003, atol=0.003)


def test_short_and_silent_inputs_return_warnings_and_finite_zero_features():
    short = extract_modulation_features(np.ones(15, dtype=np.complex64))
    silent = extract_modulation_features(np.zeros(64, dtype=np.complex64))

    assert short.sample_count == 15 and short.warnings
    assert silent.sample_count == 64 and silent.warnings
    assert np.all(np.isfinite(short.values)) and np.all(short.values == 0)
    assert np.all(np.isfinite(silent.values)) and np.all(silent.values == 0)


@pytest.mark.parametrize("samples", [np.array([1 + 0j] * 32), np.array([1 + 0j, np.nan + 0j] * 16)])
def test_nonfinite_samples_are_rejected(samples):
    if samples.size == 32 and np.all(np.isfinite(samples)):
        samples[0] = np.inf + 0j
    with pytest.raises(ValueError, match="finite"):
        extract_modulation_features(samples)


def test_invalid_samples_per_symbol_is_rejected():
    with pytest.raises(ValueError, match="samples_per_symbol"):
        extract_modulation_features(np.ones(32), samples_per_symbol=0)
