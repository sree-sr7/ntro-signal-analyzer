"""Desktop application entry point."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main(argv: Sequence[str] | None = None) -> int:
    """Start the offline signal-analysis window."""
    app = QApplication(list(argv) if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
