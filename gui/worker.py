"""Background analyzer worker for the desktop interface."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from core.pipeline.analyzer import Analyzer
from core.pipeline.demo import DemoConfig, analyze_demo


class AnalysisWorker(QObject):
    """Run one analyzer request outside the GUI thread."""

    progress = pyqtSignal(str, int)
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        analyzer: Analyzer,
        path: str | None,
        configuration: dict[str, Any] | None = None,
        *,
        demo_configuration: DemoConfig | None = None,
    ):
        super().__init__()
        self.analyzer = analyzer
        self.path = path
        self.configuration = dict(configuration or {})
        self.demo_configuration = demo_configuration

    @pyqtSlot()
    def run(self) -> None:
        try:
            if self.demo_configuration is not None:
                result = analyze_demo(
                    self.analyzer,
                    configuration=self.demo_configuration,
                    progress_callback=self.progress.emit,
                )
            else:
                if self.path is None:
                    raise ValueError("A signal file path is required outside Demo Mode")
                result = self.analyzer.analyze_file(
                    self.path,
                    progress_callback=self.progress.emit,
                    **self.configuration,
                )
        except Exception as exc:
            # Analyzer converts expected input/configuration failures into
            # AnalysisResult objects; an exception here is unexpected by design.
            self.error.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.result_ready.emit(result)
