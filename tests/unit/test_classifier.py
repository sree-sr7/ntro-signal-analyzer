import numpy as np
import pytest

from core.common.enums import ClassifierStatus, ModulationType
from core.ml.classifier import OnnxModulationClassifier, classify_classical
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
def test_classical_classifier_identifies_supported_constellations(modulation, width):
    rng = np.random.default_rng(101 + width)
    samples = bits_to_symbols(random_bits(1024 * width, rng), modulation)

    result = classify_classical(samples)

    assert result.status is ClassifierStatus.SUCCESS
    assert result.method == "classical_baseline"
    assert result.modulation is modulation
    assert result.top_candidates[0][0] is modulation
    assert 0.0 < result.confidence <= 1.0
    assert np.isclose(sum(result.probabilities.values()), 1.0)


def test_classical_classifier_identifies_two_tone_fsk():
    rng = np.random.default_rng(211)
    bits = random_bits(1024, rng)
    samples = fsk_waveform(
        bits, sample_rate=48_000, samples_per_symbol=16, tones_hz=(-3_000, 3_000)
    )

    result = classify_classical(samples, samples_per_symbol=16)

    assert result.status is ClassifierStatus.SUCCESS
    assert result.modulation is ModulationType.FSK2
    assert result.top_candidates[0][0] is ModulationType.FSK2


def test_classical_classifier_returns_unknown_for_noise_and_insufficient_data():
    rng = np.random.default_rng(307)
    noise = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)) / np.sqrt(2)
    noisy = classify_classical(noise)
    short = classify_classical(np.ones(20, dtype=np.complex64))

    assert noisy.modulation is ModulationType.UNKNOWN
    assert noisy.warnings
    assert short.modulation is ModulationType.UNKNOWN
    assert short.status is ClassifierStatus.INSUFFICIENT_DATA


def test_missing_onnx_model_is_reported_without_classical_fallback(tmp_path):
    classifier = OnnxModulationClassifier(tmp_path / "absent.onnx")

    result = classifier.classify(np.ones(128, dtype=np.complex64))

    assert not classifier.available
    assert result.modulation is ModulationType.UNKNOWN
    assert result.method == "onnx"
    assert result.status is ClassifierStatus.MODEL_UNAVAILABLE
    assert result.errors
    assert "no baseline fallback" in result.warnings[0]
