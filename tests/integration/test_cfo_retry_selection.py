import numpy as np
import pytest

from core.common.enums import ClassifierStatus, ModulationType, SampleDtype
from core.common.models import CFOResult, ClassificationResult, IQConfig
from core.pipeline import analyzer as analyzer_module
from core.pipeline.analyzer import Analyzer


class SequenceClassifier:
    method = "sequence-test-classifier"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def classify(self, samples, *, samples_per_symbol=1.0):
        result = self.responses[self.calls]
        self.calls += 1
        return result


def _classification(modulation, confidence, status=ClassifierStatus.SUCCESS):
    return ClassificationResult(
        modulation=modulation,
        confidence=confidence,
        method="test",
        probabilities={modulation: 1.0},
        status=status,
    )


def _analyze_retry(
    tmp_path,
    monkeypatch,
    corrected_confidence,
    corrected_fit_score,
    corrected_status=ClassifierStatus.SUCCESS,
    corrected_modulation=ModulationType.BPSK,
):
    sample_rate = 1_000_000.0
    indices = np.arange(1024, dtype=np.float64)
    symbols = np.where(indices.astype(np.int64) % 2, -1.0, 1.0)
    samples = (symbols * np.exp(2j * np.pi * 150.0 * indices / sample_rate)).astype(np.complex64)
    path = tmp_path / "retry.iq"
    samples.tofile(path)

    classifier = SequenceClassifier([
        _classification(ModulationType.PSK8, 0.8),
        _classification(corrected_modulation, corrected_confidence, corrected_status),
    ])
    fit_calls = iter([
        (ModulationType.PSK8, 0.9),
        (ModulationType.BPSK, corrected_fit_score),
    ])
    monkeypatch.setattr(analyzer_module, "strongest_modulation_evidence", lambda _: next(fit_calls))
    monkeypatch.setattr(
        analyzer_module,
        "estimate_cfo",
        lambda *args, **kwargs: CFOResult(estimated_offset_hz=150.0, confidence=0.99),
    )

    result = Analyzer(classifier=classifier).analyze_file(
        path,
        iq_config=IQConfig(dtype=SampleDtype.COMPLEX64, sample_rate=sample_rate),
        run_fec=False,
    )
    return result, classifier


def test_cfo_retry_selects_supported_result_within_confidence_tie_band(tmp_path, monkeypatch):
    result, classifier = _analyze_retry(
        tmp_path,
        monkeypatch,
        corrected_confidence=0.799995,
        corrected_fit_score=0.9,
    )

    assert classifier.calls == 2
    assert result.synchronization["classifier_cfo_correction"]["status"] == "applied"
    assert result.classification["modulation"] == "bpsk"


@pytest.mark.parametrize(
    ("corrected_confidence", "corrected_fit_score", "corrected_status"),
    [
        (0.79998, 0.9, ClassifierStatus.SUCCESS),  # confidence loss exceeds the tie band
        (0.799999, 0.34, ClassifierStatus.SUCCESS),  # corrected features fail signal support
        (0.799999, 0.9, ClassifierStatus.ERROR),  # invalid classifier status
    ],
)
def test_cfo_retry_retains_first_result_when_corrected_result_is_not_eligible(
    tmp_path, monkeypatch, corrected_confidence, corrected_fit_score, corrected_status
):
    result, classifier = _analyze_retry(
        tmp_path,
        monkeypatch,
        corrected_confidence=corrected_confidence,
        corrected_fit_score=corrected_fit_score,
        corrected_status=corrected_status,
    )
    correction = result.synchronization["classifier_cfo_correction"]
    assert classifier.calls == 2
    assert correction["status"] == "not_applied"
    assert result.classification["modulation"] == "8psk"


def test_cfo_retry_does_not_select_corrected_unknown_class(tmp_path, monkeypatch):
    result, classifier = _analyze_retry(
        tmp_path,
        monkeypatch,
        corrected_confidence=0.799999,
        corrected_fit_score=0.9,
        corrected_modulation=ModulationType.UNKNOWN,
    )

    assert classifier.calls == 2
    assert result.synchronization["classifier_cfo_correction"]["status"] == "not_applied"
    assert result.classification["modulation"] == "8psk"
