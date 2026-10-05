import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication
from scipy.io import wavfile

from core.common.enums import ClassifierStatus, ModulationType
from core.common.models import ClassificationResult
from core.pipeline.analyzer import AnalysisResult, Analyzer
from gui.main_window import MainWindow


class FixedClassifier:
    def classify(self, samples, *, samples_per_symbol=1.0):
        return ClassificationResult(
            modulation=ModulationType.BPSK,
            confidence=0.95,
            method="gui_test_classifier",
            probabilities={ModulationType.BPSK: 0.95},
            status=ClassifierStatus.SUCCESS,
        )


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application


def _wav(path: Path) -> Path:
    samples = np.tile(np.array([-12000, 12000], dtype=np.int16), 64)
    wavfile.write(path, 48_000, samples)
    return path


def _run_demo_and_wait(window, qapp):
    loop = QEventLoop()
    window.analysis_completed.connect(lambda _result: loop.quit())
    QTimer.singleShot(10_000, loop.quit)
    window.run_demo_button.click()
    assert not window.run_demo_button.isEnabled()
    loop.exec()
    qapp.processEvents()
    if window._thread is not None:
        idle_loop = QEventLoop()
        window._thread.finished.connect(idle_loop.quit)
        QTimer.singleShot(2_000, idle_loop.quit)
        idle_loop.exec()
        qapp.processEvents()
    return window._last_result


def test_application_starts_and_main_window_constructs(qapp):
    window = MainWindow(analyzer=Analyzer(classifier=FixedClassifier()))
    assert window.windowTitle() == "NTRO Signal Analyzer"
    assert window.analyze_button.isEnabled()
    assert window.run_demo_button.text() == "Run Demo"
    assert window.tabs.count() == 6
    window.close()


def test_file_selection_and_missing_file_state(qapp, tmp_path):
    window = MainWindow(analyzer=Analyzer(classifier=FixedClassifier()))
    path = _wav(tmp_path / "selected.wav")

    window.select_file(path)

    assert window.selected_path == str(path)
    assert "selected.wav" in window.file_label.text()
    assert "WAV metadata" in window.status_label.text()
    window.select_file(tmp_path / "missing.iq")
    assert "does not exist" in window.status_label.text()
    window.close()


def test_analyzer_worker_completion_populates_plots_and_result(qapp, tmp_path):
    path = _wav(tmp_path / "worker.wav")
    window = MainWindow(analyzer=Analyzer(classifier=FixedClassifier()))
    window.run_fec.setChecked(False)
    window.select_file(path)
    loop = QEventLoop()
    window.analysis_completed.connect(lambda _result: loop.quit())
    QTimer.singleShot(10_000, loop.quit)

    window.start_analysis()
    loop.exec()
    qapp.processEvents()
    if window._thread is not None:
        idle_loop = QEventLoop()
        window._thread.finished.connect(idle_loop.quit)
        QTimer.singleShot(2_000, idle_loop.quit)
        idle_loop.exec()
        qapp.processEvents()

    assert window._last_result is not None
    assert window._last_result.status == "partial"
    assert window.status_label.text() == "Status: PARTIAL / phase orientation unresolved"
    assert "Phase orientation unresolved" in window.overview.toPlainText()
    assert "Prediction: bpsk" in window.overview.toPlainText()
    assert window.progress.value() == 100
    assert window.waveform_plot.listDataItems()
    assert window.spectrum_plot.listDataItems()
    assert window.constellation_plot.listDataItems()
    assert window.waterfall_image.image is not None
    assert window.analyze_button.isEnabled()
    window.close()


def test_result_display_shows_errors_and_reset_clears_state(qapp, tmp_path):
    window = MainWindow(analyzer=Analyzer(classifier=FixedClassifier()))
    result = AnalysisResult(
        status="error",
        filename="broken.wav",
        errors=[{"stage": "file", "code": "load_failed", "message": "Malformed WAV header"}],
    )

    window.select_file(tmp_path / "broken.wav")
    window.populate_result(result)

    assert "Malformed WAV header" in window.overview.toPlainText()
    assert window.status_label.text() == "Status: ERROR"
    window.reset()
    assert window.selected_path is None
    assert window.overview.toPlainText() == ""
    assert window.progress.value() == 0
    assert not window.waveform_plot.listDataItems()
    window.close()


def test_plot_widgets_accept_public_result_arrays(qapp):
    window = MainWindow(analyzer=Analyzer(classifier=FixedClassifier()))
    arrays = {
        "waveform_sample_index": np.arange(8),
        "waveform_i": np.linspace(-1, 1, 8),
        "waveform_q": np.linspace(1, -1, 8),
        "spectrum_frequency_hz": np.linspace(-4, 4, 8),
        "spectrum_power": np.ones(8),
        "waterfall_magnitude_db": np.ones((8, 4)),
        "waterfall_frequency_unit": "Hz",
        "constellation_i": np.array([-1.0, 1.0]),
        "constellation_q": np.array([0.0, 0.0]),
    }

    window._populate_plots(arrays)

    assert len(window.waveform_plot.listDataItems()) == 2
    assert len(window.spectrum_plot.listDataItems()) == 1
    assert window.waterfall_image.image.shape == (4, 8)
    assert len(window.constellation_plot.listDataItems()) == 1
    window.close()


def test_run_demo_uses_result_views_for_generated_pipeline_data(qapp):
    window = MainWindow(analyzer=Analyzer())

    result = _run_demo_and_wait(window, qapp)

    assert result is not None
    assert result.status == "complete"
    assert result.source == "DEMO / SYNTHETIC"
    assert result.demo["payload_match"] is True
    assert "Source: DEMO / SYNTHETIC" in window.overview.toPlainText()
    assert "Verification: MATCHED" in window.overview.toPlainText()
    assert window.waveform_plot.listDataItems()
    assert window.spectrum_plot.listDataItems()
    assert window.waterfall_image.image is not None
    assert window.constellation_plot.listDataItems()
    assert "CRC=True" in window.details.toPlainText()
    assert window.run_demo_button.isEnabled()
    window.close()


def test_run_demo_shows_model_unavailable_without_fake_classification(qapp, tmp_path):
    window = MainWindow(analyzer=Analyzer(model_path=tmp_path / "missing.onnx"))

    result = _run_demo_and_wait(window, qapp)

    assert result is not None
    assert result.classification["status"] == "model_unavailable"
    assert result.classification["modulation"] == "unknown"
    assert "Classifier state: MODEL_UNAVAILABLE" in window.overview.toPlainText()
    assert "Prediction: unknown" in window.overview.toPlainText()
    assert window.waveform_plot.listDataItems()
    assert window.spectrum_plot.listDataItems()
    assert window.waterfall_image.image is not None
    assert not window.constellation_plot.listDataItems()
    assert window.run_demo_button.isEnabled()
    window.close()
