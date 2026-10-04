"""Offline PyQt6/pyqtgraph front end for the public analyzer result API."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QTextBrowser,
    QWidget,
)

from core.common.enums import Endianness, IQLayout, SampleDtype
from core.common.models import IQConfig
from core.pipeline.analyzer import AnalysisResult, Analyzer
from core.pipeline.demo import DEFAULT_DEMO_CONFIG
from gui.worker import AnalysisWorker


_DASHBOARD_STYLE = """
QMainWindow, QWidget {
    background: #101820;
    color: #dce5ec;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QLabel { background: transparent; }
QFrame#appHeader, QFrame#actionBar {
    background: #17232d;
    border: 1px solid #2c3d4b;
    border-radius: 8px;
}
QLabel#appTitle { font-size: 21pt; font-weight: 700; color: #f1f5f8; }
QLabel#appSubtitle, QLabel#tabCaption { color: #9badba; font-size: 9.5pt; }
QLabel#headerState {
    color: #96d7b1;
    background: #20352e;
    border: 1px solid #35594a;
    border-radius: 10px;
    min-width: 88px;
    padding: 6px 12px;
    font-weight: 700;
    letter-spacing: 1px;
}
QGroupBox {
    border: 1px solid #2b3c49;
    border-radius: 7px;
    margin-top: 12px;
    padding: 10px 9px 9px 9px;
    font-weight: 650;
    color: #b9c9d4;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLabel#fileInfo { color: #dce5ec; font-weight: 600; }
QLabel#demoNote {
    color: #bdc9d1;
    background: #1b2932;
    border-left: 3px solid #7092a8;
    border-radius: 4px;
    padding: 9px;
    font-size: 9pt;
}
QLabel#tabHeading { color: #eef3f6; font-size: 15pt; font-weight: 650; }
QLabel#emptyHeading { color: #eef3f6; font-size: 19pt; font-weight: 650; padding: 6px; }
QLabel#emptyDescription { color: #a6b5c0; font-size: 11pt; padding: 4px 20px; }
QLabel#emptyBadge {
    color: #a9c6d9;
    background: #1b2a35;
    border: 1px solid #314653;
    border-radius: 12px;
    padding: 7px 13px;
    margin-top: 14px;
    font-size: 9pt;
    font-weight: 650;
}
QToolButton#sectionToggle {
    border: 1px solid #2b3c49;
    border-radius: 6px;
    background: #17232d;
    color: #c2d0d9;
    text-align: left;
    padding: 9px;
    font-weight: 700;
}
QToolButton#sectionToggle:hover { background: #20303c; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #0d151c;
    border: 1px solid #344755;
    border-radius: 5px;
    padding: 5px 7px;
    min-height: 22px;
    selection-background-color: #3c718d;
}
QCheckBox { spacing: 7px; color: #c8d3da; }
QPushButton {
    color: #e4ebf0;
    background: #263744;
    border: 1px solid #3b5262;
    border-radius: 6px;
    padding: 8px 13px;
    font-weight: 600;
}
QPushButton:hover { background: #304553; }
QPushButton:disabled { color: #778792; background: #202b33; border-color: #2c3840; }
QPushButton#openButton { background: #315e78; border-color: #477b98; }
QPushButton#openButton:hover { background: #3a6d89; }
QPushButton#demoButton { background: #273c4a; border-color: #456273; }
QPushButton#demoButton:hover { background: #304b5c; }
QPushButton#analyzeButton { background: #277665; border-color: #378f7c; color: #f0f8f5; }
QPushButton#analyzeButton:hover { background: #318a76; }
QPushButton#resetButton { background: transparent; border-color: transparent; color: #9eafb9; }
QPushButton#resetButton:hover { background: #202c35; border-color: #34434d; }
QTabWidget::pane { border: 1px solid #2b3c49; border-radius: 7px; background: #111a23; }
QTabBar::tab {
    background: #17232d;
    color: #9fb0bc;
    border: 1px solid #293a47;
    padding: 9px 13px;
    margin-right: 3px;
}
QTabBar::tab:selected { background: #263947; color: #f0f4f6; border-bottom-color: #6d9ab1; }
QTextBrowser#analysisBrowser, QPlainTextEdit {
    background: #111a23;
    color: #dce5ec;
    border: 1px solid #2b3c49;
    border-radius: 7px;
    padding: 6px;
    selection-background-color: #3c718d;
}
QTableWidget {
    background: #111a23;
    alternate-background-color: #16232d;
    color: #dce5ec;
    gridline-color: #2b3c49;
    border: 1px solid #2b3c49;
    border-radius: 6px;
}
QHeaderView::section {
    background: #1c2a34;
    color: #b9c9d4;
    border: none;
    border-bottom: 1px solid #344755;
    padding: 7px;
    font-weight: 650;
}
QStatusBar { background: #0d151c; color: #acbac4; border-top: 1px solid #283943; }
QProgressBar { color: #e4ebf0; background: #18252e; border: 1px solid #30434f; border-radius: 4px; min-height: 16px; text-align: center; }
QProgressBar::chunk { background: #5c9a83; border-radius: 3px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #111a23; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #3a4d5a; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


class MainWindow(QMainWindow):
    """Single-window offline workflow driven only by Analyzer.analyze_file."""

    analysis_completed = pyqtSignal(object)
    analysis_failed = pyqtSignal(str)

    def __init__(self, analyzer: Analyzer | None = None) -> None:
        super().__init__()
        self.analyzer = analyzer or Analyzer()
        self.selected_path: str | None = None
        self._thread: QThread | None = None
        self._worker: AnalysisWorker | None = None
        self._last_result: AnalysisResult | None = None
        self._demo_running = False
        self._demo_mode = False
        self._normal_demo_control_values: tuple[float, int] | None = None
        self.setWindowTitle("NTRO Signal Analyzer")
        self.setMinimumSize(1080, 680)
        self.resize(1440, 900)
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet(_DASHBOARD_STYLE)
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 14, 16, 8)
        root_layout.setSpacing(12)

        header = QFrame(root)
        header.setObjectName("appHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 12, 18, 12)
        title_stack = QVBoxLayout()
        title = QLabel("NTRO Signal Analyzer")
        title.setObjectName("appTitle")
        subtitle = QLabel("Automated IQ/WAV Signal Analysis")
        subtitle.setObjectName("appSubtitle")
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header_layout.addLayout(title_stack)
        header_layout.addStretch(1)
        self.header_state_label = QLabel("READY")
        self.header_state_label.setObjectName("headerState")
        self.header_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.header_state_label, 0, Qt.AlignmentFlag.AlignVCenter)
        root_layout.addWidget(header)

        action_bar = QFrame(root)
        action_bar.setObjectName("actionBar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(12, 8, 12, 8)
        action_layout.setSpacing(9)
        self.open_button = QPushButton("Open IQ/WAV")
        self.open_button.setObjectName("openButton")
        self.open_button.clicked.connect(self.open_file_dialog)
        self.run_demo_button = QPushButton("Run Demo")
        self.run_demo_button.setObjectName("demoButton")
        self.run_demo_button.clicked.connect(self.start_demo)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setObjectName("analyzeButton")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("resetButton")
        self.reset_button.clicked.connect(self.reset)
        action_layout.addWidget(self.open_button)
        action_layout.addWidget(self.run_demo_button)
        action_layout.addWidget(self.analyze_button)
        action_layout.addStretch(1)
        action_layout.addWidget(self.reset_button)
        root_layout.addWidget(action_bar)

        content_splitter = QSplitter(Qt.Orientation.Horizontal, root)
        content_splitter.setChildrenCollapsible(False)
        controls_scroll = QScrollArea(content_splitter)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(340)
        controls_scroll.setMaximumWidth(460)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(4, 4, 8, 4)
        controls_layout.setSpacing(10)

        input_group = QGroupBox("INPUT")
        input_form = QFormLayout(input_group)
        self.source_label = QLabel("NOT SELECTED")
        self.source_label.setObjectName("fileInfo")
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        self.file_label.setObjectName("fileInfo")
        input_form.addRow("Source", self.source_label)
        input_form.addRow("File", self.file_label)

        basic_group = QGroupBox("BASIC SETTINGS")
        basic_form = QFormLayout(basic_group)
        self.sample_rate = QDoubleSpinBox()
        self.sample_rate.setRange(0.0, 1.0e12)
        self.sample_rate.setDecimals(3)
        self.sample_rate.setValue(0.0)
        self.sample_rate.setSuffix(" Hz")
        self.sample_rate.setSpecialValueText("Required for raw IQ")
        self.samples_per_symbol = QSpinBox()
        self.samples_per_symbol.setRange(1, 65536)
        self.samples_per_symbol.setValue(1)
        basic_form.addRow("Sample rate", self.sample_rate)
        basic_form.addRow("Samples per symbol", self.samples_per_symbol)

        advanced_body = QWidget()
        advanced_layout = QVBoxLayout(advanced_body)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(8)
        raw_group = QGroupBox("Raw IQ / WAV options")
        raw_form = QFormLayout(raw_group)
        self.dtype = QComboBox()
        for value in SampleDtype:
            self.dtype.addItem(value.value, value)
        self.dtype.setCurrentIndex(self.dtype.findData(SampleDtype.COMPLEX64))
        self.iq_layout = QComboBox()
        for value in IQLayout:
            if value is not IQLayout.UNKNOWN:
                self.iq_layout.addItem(value.value, value)
        self.iq_layout.setCurrentIndex(self.iq_layout.findData(IQLayout.COMPLEX_NATIVE))
        self.endianness = QComboBox()
        for value in Endianness:
            self.endianness.addItem(value.value, value)
        self.stereo_iq = QCheckBox("Interpret stereo WAV as I/Q")
        raw_form.addRow("Sample dtype", self.dtype)
        raw_form.addRow("IQ layout", self.iq_layout)
        raw_form.addRow("Byte order", self.endianness)
        raw_form.addRow(self.stereo_iq)
        advanced_layout.addWidget(raw_group)

        sync_group = QGroupBox("Synchronization and demodulation")
        sync_form = QFormLayout(sync_group)
        self.timing_recovery = QCheckBox("Run Gardner timing recovery")
        self.rrc_enabled = QCheckBox("Apply RRC matched filter")
        self.rrc_rolloff = QDoubleSpinBox()
        self.rrc_rolloff.setRange(0.0, 1.0)
        self.rrc_rolloff.setSingleStep(0.05)
        self.rrc_rolloff.setValue(0.35)
        self.tone0 = QLineEdit()
        self.tone0.setPlaceholderText("Hz; required for 2FSK")
        self.tone1 = QLineEdit()
        self.tone1.setPlaceholderText("Hz; required for 2FSK")
        self.sync_word = QLineEdit()
        self.sync_word.setPlaceholderText("Optional hex bytes, MSB first")
        self.header_length = QSpinBox()
        self.header_length.setRange(0, 1_000_000)
        self.payload_length = QSpinBox()
        self.payload_length.setRange(0, 10_000_000)
        self.payload_length.setSpecialValueText("Unspecified")
        sync_form.addRow(self.timing_recovery)
        sync_form.addRow(self.rrc_enabled)
        sync_form.addRow("RRC rolloff", self.rrc_rolloff)
        sync_form.addRow("2FSK bit-0 tone", self.tone0)
        sync_form.addRow("2FSK bit-1 tone", self.tone1)
        sync_form.addRow("Sync word (hex)", self.sync_word)
        sync_form.addRow("Header bits after sync", self.header_length)
        sync_form.addRow("Payload bits (0 = all)", self.payload_length)
        advanced_layout.addWidget(sync_group)
        self.advanced_section = self._collapsible_section(
            "ADVANCED SETTINGS", advanced_body, expanded=False
        )

        fec_group = QGroupBox("FEC / FRAME SETTINGS")
        fec_form = QFormLayout(fec_group)
        self.run_fec = QCheckBox("Run FEC / interleaver search")
        self.run_fec.setChecked(True)
        self.crc_present = QCheckBox("Input includes appended CRC-16")
        self.expected_payload = QSpinBox()
        self.expected_payload.setRange(0, 10_000_000)
        self.expected_payload.setSpecialValueText("Unspecified")
        fec_form.addRow(self.run_fec)
        fec_form.addRow(self.crc_present)
        fec_form.addRow("Expected payload (bits)", self.expected_payload)
        self.demo_note_label = QLabel(
            "Demo: deterministic local signal.\nNot an independent validation dataset."
        )
        self.demo_note_label.setObjectName("demoNote")
        self.demo_note_label.setWordWrap(True)
        controls_layout.addWidget(input_group)
        controls_layout.addWidget(basic_group)
        controls_layout.addWidget(self.advanced_section)
        controls_layout.addWidget(fec_group)
        controls_layout.addWidget(self.demo_note_label)
        controls_layout.addStretch(1)
        controls_scroll.setWidget(controls)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMinimumWidth(600)
        self.analysis_stack = QStackedWidget()
        self.analysis_empty_state = self._empty_state_widget(
            "No signal loaded",
            "Open an IQ/WAV file or run the built-in demo to begin analysis.",
            "SUPPORTED  .IQ     .WAV",
        )
        self.overview = QTextBrowser()
        self.overview.setReadOnly(True)
        self.overview.setOpenLinks(False)
        self.overview.setOpenExternalLinks(False)
        self.overview.setObjectName("analysisBrowser")
        self.analysis_stack.addWidget(self.analysis_empty_state)
        self.analysis_stack.addWidget(self.overview)
        self.analysis_stack.setCurrentWidget(self.analysis_empty_state)
        self.tabs.addTab(self.analysis_stack, "Analysis")

        self.waveform_plot = self._new_plot()
        self.waveform_plot.setLabel("bottom", "Sample index")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.22)
        waveform_tab, self.waveform_empty_state, self.waveform_stack = self._plot_tab(
            "Time-domain waveform", "Analyzer output · in-phase and quadrature amplitude", self.waveform_plot,
            "No waveform data yet", "Open a signal or run Demo to view the time-domain samples.",
        )
        self.tabs.addTab(waveform_tab, "Waveform")

        self.spectrum_plot = self._new_plot()
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setLabel("left", "Power", units="dB")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.22)
        spectrum_tab, self.spectrum_empty_state, self.spectrum_stack = self._plot_tab(
            "Signal spectrum", "Analyzer-computed power spectrum · frequency in hertz", self.spectrum_plot,
            "No spectrum data yet", "A sample rate is required to display the computed frequency axis.",
        )
        self.tabs.addTab(spectrum_tab, "Spectrum")

        self.waterfall_plot = self._new_plot()
        self.waterfall_plot.setLabel("bottom", "Time", units="seconds")
        self.waterfall_plot.setLabel("left", "Frequency", units="Hz")
        self.waterfall_image = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_image)
        waterfall_tab, self.waterfall_empty_state, self.waterfall_stack = self._plot_tab(
            "Spectrum over time", "STFT magnitude from the analyzer · brighter regions indicate stronger energy", self.waterfall_plot,
            "No waterfall data yet", "The analyzer needs at least eight samples to form a waterfall view.",
        )
        self.tabs.addTab(waterfall_tab, "Waterfall")

        self.constellation_plot = self._new_plot()
        self.constellation_plot.setLabel("bottom", "In-phase")
        self.constellation_plot.setLabel("left", "Quadrature")
        self.constellation_plot.showGrid(x=True, y=True, alpha=0.22)
        self.constellation_plot.setAspectLocked(True, ratio=1.0)
        constellation_tab, self.constellation_empty_state, self.constellation_stack = self._plot_tab(
            "Demodulated constellation", "Synchronized symbol points returned by the analyzer", self.constellation_plot,
            "No constellation available", "Constellation points appear after an accepted modulation result.",
        )
        self.tabs.addTab(constellation_tab, "Constellation")

        fec_tab = QWidget()
        fec_layout = QVBoxLayout(fec_tab)
        fec_layout.setContentsMargins(14, 14, 14, 14)
        fec_layout.setSpacing(8)
        fec_heading = QLabel("FEC and frame recovery")
        fec_heading.setObjectName("tabHeading")
        fec_caption = QLabel("Candidate trial results and CRC validation from the analyzer.")
        fec_caption.setObjectName("tabCaption")
        self.fec_table = QTableWidget(0, 5)
        self.fec_table.setHorizontalHeaderLabels(["FEC", "Interleaver", "Status", "CRC", "Details"])
        self.fec_table.setAlternatingRowColors(True)
        self.fec_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.fec_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.fec_table.setWordWrap(True)
        self.fec_table.verticalHeader().setVisible(False)
        fec_header = self.fec_table.horizontalHeader()
        fec_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        fec_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        fec_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        fec_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        fec_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("Detailed analyzer diagnostics will appear here.")
        technical_group = QGroupBox("Technical diagnostics")
        technical_layout = QVBoxLayout(technical_group)
        technical_layout.addWidget(self.details)
        fec_layout.addWidget(fec_heading)
        fec_layout.addWidget(fec_caption)
        fec_layout.addWidget(self.fec_table, 1)
        fec_layout.addWidget(technical_group, 1)
        self.tabs.addTab(fec_tab, "FEC / Frame Details")

        content_splitter.addWidget(controls_scroll)
        content_splitter.addWidget(self.tabs)
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([390, 990])
        root_layout.addWidget(content_splitter, 1)
        self.setCentralWidget(root)

        self.status_label = QLabel("Status: READY")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.progress.setFixedWidth(128)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)

    @staticmethod
    def _empty_state_widget(title: str, description: str, badge: str) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.addStretch(2)
        heading = QLabel(title)
        heading.setObjectName("emptyHeading")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        explanation = QLabel(description)
        explanation.setObjectName("emptyDescription")
        explanation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        explanation.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(explanation)
        if badge:
            supported = QLabel(badge)
            supported.setObjectName("emptyBadge")
            supported.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(supported)
        layout.addStretch(3)
        return panel

    @staticmethod
    def _new_plot() -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setBackground("#111a23")
        plot.getPlotItem().layout.setContentsMargins(12, 10, 18, 12)
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen("#607486")
            axis.setTextPen("#b8c7d4")
        return plot

    @classmethod
    def _plot_tab(
        cls,
        title: str,
        description: str,
        plot: pg.PlotWidget,
        empty_title: str,
        empty_description: str,
    ) -> tuple[QWidget, QWidget, QStackedWidget]:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("tabHeading")
        caption = QLabel(description)
        caption.setObjectName("tabCaption")
        stack = QStackedWidget()
        empty_state = cls._empty_state_widget(empty_title, empty_description, "")
        stack.addWidget(empty_state)
        stack.addWidget(plot)
        stack.setCurrentWidget(empty_state)
        layout.addWidget(heading)
        layout.addWidget(caption)
        layout.addWidget(stack, 1)
        return tab, empty_state, stack

    @staticmethod
    def _collapsible_section(title: str, content: QWidget, *, expanded: bool) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        toggle = QToolButton()
        toggle.setText(title)
        toggle.setCheckable(True)
        toggle.setChecked(expanded)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        toggle.setObjectName("sectionToggle")
        content.setVisible(expanded)

        def set_expanded(checked: bool) -> None:
            content.setVisible(checked)
            toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

        toggle.toggled.connect(set_expanded)
        layout.addWidget(toggle)
        layout.addWidget(content)
        return section

    def open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open signal file",
            "",
            "Signal files (*.iq *.IQ *.wav);;All files (*)",
        )
        if path:
            self.select_file(path)

    def select_file(self, path: str | Path) -> None:
        """Select a local file and show its basic filesystem metadata."""
        self._set_demo_mode(False)
        selected = Path(path).expanduser()
        self.selected_path = str(selected)
        if selected.is_file():
            self.source_label.setText("IQ/WAV FILE")
            self.file_label.setText(
                f"{selected.name}\n{selected.stat().st_size:,} bytes"
            )
            self.status_label.setText(
                "Raw IQ has no embedded sample rate or dtype; review the explicit settings."
                if selected.suffix.lower() == ".iq"
                else "WAV metadata will be read from its RIFF header."
            )
        else:
            self.source_label.setText("FILE PATH")
            self.file_label.setText(str(selected))
            self.status_label.setText("Selected path does not exist or is not a file.")

    def _set_demo_mode(self, enabled: bool) -> None:
        """Update GUI-only source and fixed demo configuration presentation."""
        if enabled == self._demo_mode:
            return

        if enabled:
            self._normal_demo_control_values = (
                float(self.sample_rate.value()),
                int(self.samples_per_symbol.value()),
            )
            self.sample_rate.setDecimals(1)
            self.sample_rate.setValue(DEFAULT_DEMO_CONFIG.sample_rate_hz / 1000.0)
            self.sample_rate.setSuffix(" kHz · Demo")
            self.samples_per_symbol.setValue(DEFAULT_DEMO_CONFIG.samples_per_symbol)
            self.samples_per_symbol.setSuffix(" · Demo")
            self.sample_rate.setEnabled(False)
            self.samples_per_symbol.setEnabled(False)
            self.source_label.setText("DEMO / SYNTHETIC")
            self.file_label.setText("deterministic_demo.iq")
        else:
            if self._normal_demo_control_values is not None:
                sample_rate, samples_per_symbol = self._normal_demo_control_values
                self.sample_rate.setSuffix(" Hz")
                self.sample_rate.setDecimals(3)
                self.sample_rate.setValue(sample_rate)
                self.samples_per_symbol.setSuffix("")
                self.samples_per_symbol.setValue(samples_per_symbol)
                self.sample_rate.setEnabled(True)
                self.samples_per_symbol.setEnabled(True)
                self._normal_demo_control_values = None
            self.source_label.setText("IQ/WAV FILE" if self.selected_path else "NOT SELECTED")
            if self.selected_path is None:
                self.file_label.setText("No file selected")

        self._demo_mode = enabled

    @staticmethod
    def _parse_sync_word(value: str) -> np.ndarray | None:
        stripped = value.strip()
        if not stripped:
            return None
        try:
            raw = bytes.fromhex(stripped)
        except ValueError as exc:
            raise ValueError("Sync word must be even-length hexadecimal bytes.") from exc
        if not raw:
            raise ValueError("Sync word cannot be empty.")
        return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")

    def _analysis_configuration(self) -> dict[str, Any]:
        iq_config = IQConfig(
            dtype=self.dtype.currentData(),
            sample_rate=(float(self.sample_rate.value()) or None),
            iq_layout=self.iq_layout.currentData(),
            endianness=self.endianness.currentData(),
        )
        tones = None
        tone0, tone1 = self.tone0.text().strip(), self.tone1.text().strip()
        if tone0 or tone1:
            if not tone0 or not tone1:
                raise ValueError("Enter both 2FSK tones or leave both empty.")
            try:
                tones = (float(tone0), float(tone1))
            except ValueError as exc:
                raise ValueError("2FSK tones must be numeric frequencies in Hz.") from exc
        return {
            "iq_config": iq_config,
            "treat_stereo_as_iq": self.stereo_iq.isChecked(),
            "samples_per_symbol": float(self.samples_per_symbol.value()),
            "matched_filter_rolloff": (
                float(self.rrc_rolloff.value()) if self.rrc_enabled.isChecked() else None
            ),
            "recover_symbol_timing": self.timing_recovery.isChecked(),
            "tone_frequencies_hz": tones,
            "expected_payload_length": (
                int(self.expected_payload.value()) or None
            ),
            "crc_present": self.crc_present.isChecked(),
            "run_fec": self.run_fec.isChecked(),
            "sync_word": self._parse_sync_word(self.sync_word.text()),
            "header_length": int(self.header_length.value()),
            "payload_length": int(self.payload_length.value()) or None,
        }

    def start_analysis(self) -> None:
        if self._thread is not None:
            return
        self._set_demo_mode(False)
        if self.selected_path is None:
            self.status_label.setText("Choose an .IQ or .wav file before analysis.")
            return
        path = Path(self.selected_path)
        if not path.is_file():
            self.status_label.setText("Selected path does not exist or is not a file.")
            return
        try:
            configuration = self._analysis_configuration()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._demo_running = False
        self._start_worker(
            AnalysisWorker(self.analyzer, str(path), configuration),
            initial_status="Analysis running…",
        )

    def start_demo(self) -> None:
        """Run the built-in deterministic source through the regular analyzer worker."""
        if self._thread is not None:
            return
        self._set_demo_mode(True)
        self._demo_running = True
        self._start_worker(
            AnalysisWorker(
                self.analyzer,
                None,
                demo_configuration=DEFAULT_DEMO_CONFIG,
            ),
            initial_status="Generating demo signal...",
        )

    def _start_worker(self, worker: AnalysisWorker, *, initial_status: str) -> None:
        self.open_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.run_demo_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText(initial_status)
        self.header_state_label.setText("ANALYZING")
        self.header_state_label.setStyleSheet("color: #e2c17b; background: #382f20; border-color: #665634;")
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.progress.connect(self._on_progress)
        worker.result_ready.connect(self._on_result)
        worker.error.connect(self._on_worker_error)
        worker.result_ready.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.started.connect(worker.run)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_progress(self, message: str, percent: int) -> None:
        if self._demo_running:
            normalized = message.lower()
            if normalized.startswith("generating demo"):
                display = "GENERATING DEMO..."
                state = "ANALYZING"
            elif "classif" in normalized:
                display = "CLASSIFYING..."
                state = "ANALYZING"
            elif "demodulat" in normalized:
                display = "DEMODULATING..."
                state = "ANALYZING"
            elif "fec" in normalized or "crc" in normalized or "frame" in normalized or "result" in normalized:
                display = "RECOVERING..."
                state = "ANALYZING"
            elif normalized in {"analysis finished", "complete"}:
                display = "COMPLETE"
                state = "COMPLETE"
            else:
                display = "ANALYZING..."
                state = "ANALYZING"
            self.status_label.setText(display)
            self.header_state_label.setText(state)
        else:
            self.status_label.setText(message)
            self.header_state_label.setText("ANALYZING")
        self.progress.setValue(max(0, min(100, int(percent))))

    def _on_result(self, result: AnalysisResult) -> None:
        self.populate_result(result)
        self.analysis_completed.emit(result)

    def _on_worker_error(self, message: str) -> None:
        self.header_state_label.setText("ERROR")
        self.header_state_label.setStyleSheet("color: #e6a09a; background: #382322; border-color: #71413e;")
        self.status_label.setText(f"Analysis error: {message}")
        self.status_label.setStyleSheet("color: #b00020; font-weight: bold;")
        self.overview.setPlainText(
            "ANALYSIS ERROR\n\n"
            + message
            + "\n\nThe analyzer did not return a result object."
        )
        self.analysis_stack.setCurrentWidget(self.overview)
        self.progress.setValue(100)
        self.analysis_failed.emit(message)

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._demo_running = False
        self.open_button.setEnabled(True)
        self.analyze_button.setEnabled(True)
        self.run_demo_button.setEnabled(True)
        self.reset_button.setEnabled(True)

    @staticmethod
    def _bits_preview(value: Any, limit: int = 256) -> str:
        if value is None:
            return "unavailable"
        bits = np.asarray(value, dtype=np.uint8).reshape(-1)
        prefix = "".join(str(int(bit)) for bit in bits[:limit])
        suffix = "…" if bits.size > limit else ""
        return f"{prefix}{suffix} ({bits.size} bits)"

    def populate_result(self, result: AnalysisResult) -> None:
        """Render the public AnalysisResult; no internal DSP/FEC APIs are used."""
        self._last_result = result
        phase_resolution = result.synchronization.get("phase_ambiguity_resolution", {})
        phase_unresolved = (
            isinstance(phase_resolution, dict)
            and phase_resolution.get("status") == "unresolved"
        )
        display_status = "partial" if phase_unresolved else result.status
        status_color = {
            "complete": ("#96d7b1", "#20352e", "#35594a"),
            "partial": ("#e2c17b", "#382f20", "#665634"),
            "rejected": ("#e6a09a", "#382322", "#71413e"),
            "error": ("#e6a09a", "#382322", "#71413e"),
        }.get(display_status, ("#b8c8d2", "#202b33", "#34434d"))
        status_label = display_status.upper()
        if result.classification.get("status") == "model_unavailable":
            status_label = "MODEL_UNAVAILABLE"
        self.header_state_label.setText(status_label)
        self.header_state_label.setStyleSheet(
            "color: %s; background: %s; border-color: %s;"
            % status_color
        )
        self.status_label.setStyleSheet(
            f"color: {status_color[0]}; font-weight: 600;"
        )
        self.status_label.setText(
            "Status: PARTIAL / phase orientation unresolved"
            if phase_unresolved
            else f"Status: {display_status.upper()}"
        )
        self.progress.setValue(100)
        self.analysis_stack.setCurrentWidget(self.overview)

        file_info = result.file_metadata
        classification = result.classification
        demodulation = result.demodulation
        synchronization = result.synchronization
        fec = result.fec
        rate = self._format_hz(file_info.get("sample_rate_hz"))
        duration = file_info.get("duration_seconds")
        duration_text = f"{float(duration) * 1000.0:.2f} ms" if duration is not None else "Unavailable"
        sample_count = file_info.get("sample_count", "Unavailable")
        modulation = str(classification.get("modulation", "unavailable"))
        confidence = classification.get("confidence")
        confidence_text = f"{float(confidence) * 100.0:.1f}% confidence" if confidence is not None else "Confidence unavailable"
        classifier_status = str(classification.get("status", "unavailable")).upper()
        cfo_result = synchronization.get("coarse_cfo")
        cfo_value = cfo_result.get("estimated_offset_hz") if isinstance(cfo_result, dict) else None
        cfo_text = "Not estimated" if cfo_value is None else f"{float(cfo_value):+.2f} Hz"
        preprocessing = synchronization.get("preprocessing", {})
        timing_text = (
            "Timing recovery applied"
            if preprocessing.get("timing_recovery_applied")
            else f"{preprocessing.get('effective_samples_per_symbol', 'Unknown')} sample/symbol input"
        )
        phase_text = "<br>Phase orientation unresolved" if phase_unresolved else ""
        fec_status = str(fec.get("status", "unavailable"))
        if fec_status == "accepted":
            recovery_text, recovery_tone = "SUCCESS", "success"
        elif fec_status == "crc_accepted_no_fec":
            recovery_text, recovery_tone = "CRC ACCEPTED · NO FEC", "warning"
        elif fec_status in {"no_accepted_candidate", "failed"}:
            recovery_text, recovery_tone = "NOT ACCEPTED", "error"
        elif fec_status == "not_requested":
            recovery_text, recovery_tone = "NOT REQUESTED", "neutral"
        else:
            recovery_text, recovery_tone = "UNAVAILABLE", "neutral"
        crc_value = fec.get("crc_pass")
        crc_text = "PASS" if crc_value is True else "FAIL" if crc_value is False else "NOT CONFIGURED"
        crc_tone = "success" if crc_value is True else "error" if crc_value is False else "neutral"
        if result.demo is not None:
            frame_text = "Not configured for demo profile"
            payload_length = result.demo.get("configuration", {}).get("payload_length_bits")
            payload_verification = str(result.demo.get("payload_recovery", "not_available")).replace("_", " ").upper()
            payload_tone = "success" if payload_verification == "MATCHED" else "error" if payload_verification == "MISMATCH" else "neutral"
            frame_value = f"{payload_length} bits recovered" if payload_length is not None else "Payload unavailable"
        else:
            frame_text = str(result.frames.get("status", "unavailable")).replace("_", " ").title()
            payload_length = result.frames.get("payload_bit_count")
            payload_verification = "Not available"
            payload_tone = "neutral"
            frame_value = frame_text
        payload_text = f"{payload_length} bits" if payload_length is not None else "Length unavailable"
        fec_name = fec.get("fec_type") or "No accepted FEC"
        interleaver_name = fec.get("interleaver_type") or "None"
        demod_status = str(demodulation.get("status", "unavailable")).upper()
        bit_count = demodulation.get("bit_count", 0)
        llr_count = demodulation.get("llr_count", 0)
        demod_tone = "success" if demod_status == "COMPLETE" else "error" if demod_status == "FAILED" else "neutral"

        cards = [
            ("SIGNAL", rate, f"Duration: {duration_text} · Samples: {sample_count}<br>Validation: {result.validation_status.replace('_', ' ').upper()}", "neutral"),
            ("MODULATION", modulation.upper(), f"Prediction: {html.escape(modulation)} · {confidence_text}<br>Classifier state: {classifier_status} · {html.escape(str(classification.get('method', 'unknown')).upper())}", "success" if classifier_status == "SUCCESS" else "error" if classifier_status in {"ERROR", "MODEL_UNAVAILABLE"} else "warning"),
            ("SYNCHRONIZATION", cfo_text, f"Timing: {timing_text}<br>Sample rate: {rate}{phase_text}", "neutral"),
            ("DEMODULATION", demod_status, f"Recovered bits: {bit_count}<br>LLR availability: {'Available (' + str(llr_count) + ')' if llr_count else 'Unavailable'}", demod_tone),
            ("FEC RECOVERY", recovery_text, f"{html.escape(str(fec_name))} · {html.escape(str(interleaver_name))}<br>CRC: {crc_text}", recovery_tone),
            ("FRAME / PAYLOAD", frame_value, f"Frame / header: {frame_text}<br>Payload: {payload_text} · Payload verification: {payload_verification}", payload_tone),
        ]
        rows = []
        for offset in range(0, len(cards), 2):
            pair = cards[offset:offset + 2]
            row_cells = "".join(self._dashboard_card(*card) for card in pair)
            if len(pair) == 1:
                row_cells += "<td width='50%'></td>"
            rows.append(f"<tr>{row_cells}</tr>")
        status_text = "COMPLETE" if display_status == "complete" else display_status.upper()
        source = html.escape(result.source)
        alert_html = ""
        if result.errors:
            message = html.escape(str(result.errors[0].get("message", "Analysis reported an error")))
            alert_html = f"<p style='color:#e6a09a; background:#382322; padding:9px;'><b>Attention</b> · {message}</p>"
        elif result.warnings:
            message = html.escape(str(result.warnings[0]))
            alert_html = f"<p style='color:#e2c17b; background:#382f20; padding:9px;'><b>Note</b> · {message}</p>"
        demo_html = ""
        if result.demo is not None:
            note = html.escape(str(result.demo.get("note", "")))
            demo_html = (
                "<p style='color:#aebdc6; background:#1a2730; border-left:3px solid #668397; padding:9px;'>"
                f"DEMO / SYNTHETIC · {note}</p>"
            )
        dashboard_html = f"""
        <html><body style="background-color:#111a23; color:#dce5ec; font-family:'Segoe UI';">
        <table width="100%" cellspacing="0" cellpadding="4"><tr>
          <td><span style="font-size:20pt; font-weight:bold; color:#f1f5f8;">Analysis result</span><br>
          <span style="color:#9badba;">{html.escape(result.filename)} · {html.escape(result.file_format.upper())}</span></td>
          <td align="right"><span style="font-weight:bold; color:{status_color[0]};">{status_text}</span><br>
          <span style="color:#aebdc6;">Source: {source}</span></td>
        </tr></table>
        {demo_html}
        <table width="100%" cellspacing="8" cellpadding="0">{''.join(rows)}</table>
        {alert_html}
        </body></html>
        """
        self.overview.setHtml(dashboard_html)
        self._populate_plots(
            result.visualization,
            duration_seconds=(float(duration) if duration is not None else None),
        )
        self._populate_fec_table(fec)
        summary = (
            f"FEC={fec.get('fec_type')}; Interleaver={fec.get('interleaver_type')}; "
            f"Status={fec.get('status')}; CRC={fec.get('crc_pass')}"
        )
        diagnostic_json = json.dumps(result.to_dict(include_arrays=False), indent=2, ensure_ascii=False)
        self.details.setPlainText(
            f"{summary}\n\nComplete AnalysisResult diagnostics:\n{diagnostic_json}"
        )

    @staticmethod
    def _dashboard_card(title: str, value: str, detail: str, tone: str) -> str:
        accents = {
            "success": "#76b894",
            "warning": "#c7a45d",
            "error": "#c77e78",
            "neutral": "#7194a9",
        }
        accent = accents.get(tone, accents["neutral"])
        return (
            "<td width='50%' valign='top' style='background-color:#17242e; border:1px solid #2d404e; padding:12px;'>"
            f"<span style='color:#9badba; font-size:9pt; font-weight:bold;'>{html.escape(title)}</span><br>"
            f"<span style='color:{accent}; font-size:17pt; font-weight:bold;'>{html.escape(str(value))}</span><br>"
            f"<span style='color:#bdc9d1; font-size:9.5pt;'>{detail}</span></td>"
        )

    @staticmethod
    def _format_hz(value: Any) -> str:
        if value is None:
            return "Unavailable"
        number = float(value)
        absolute = abs(number)
        if absolute >= 1_000_000:
            return f"{number / 1_000_000:.3f} MHz"
        if absolute >= 1_000:
            return f"{number / 1_000:.3f} kHz"
        return f"{number:.3f} Hz"

    def _populate_fec_table(self, fec: dict[str, Any]) -> None:
        candidates = fec.get("candidates", []) or []
        if not candidates:
            candidates = [{
                "fec_type": fec.get("fec_type"),
                "interleaver_type": fec.get("interleaver_type"),
                "candidate_status": fec.get("status", "unavailable"),
                "accepted": fec.get("status") == "accepted",
                "valid": fec.get("best_candidate_valid"),
                "crc_pass": fec.get("crc_pass"),
                "codeword_count": fec.get("codeword_count"),
            }]
        self.fec_table.setRowCount(len(candidates))
        for row, candidate in enumerate(candidates):
            status = "ACCEPTED" if candidate.get("accepted") else str(candidate.get("candidate_status", "evaluated")).replace("_", " ").upper()
            crc = candidate.get("crc_pass")
            crc_label = "PASS" if crc is True else "FAIL" if crc is False else "—"
            details = f"Valid: {candidate.get('valid', '—')} · Codewords: {candidate.get('codeword_count', '—')}"
            values = (
                candidate.get("fec_type") or "None",
                candidate.get("interleaver_type") or "None",
                status,
                crc_label,
                details,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.fec_table.setItem(row, column, item)
        self.fec_table.resizeRowsToContents()

    def _populate_plots(
        self,
        arrays: dict[str, Any],
        *,
        duration_seconds: float | None = None,
    ) -> None:
        self.waveform_plot.clear()
        self.spectrum_plot.clear()
        self.constellation_plot.clear()
        x = arrays.get("waveform_sample_index")
        i_values = arrays.get("waveform_i")
        q_values = arrays.get("waveform_q")
        waveform_ready = x is not None and i_values is not None and np.size(x) > 0
        self.waveform_stack.setCurrentWidget(
            self.waveform_plot if waveform_ready else self.waveform_empty_state
        )
        if waveform_ready:
            self.waveform_plot.plot(x, i_values, pen=pg.mkPen("#70b9e7", width=1.4), name="I")
            if q_values is not None and np.any(np.asarray(q_values) != 0):
                self.waveform_plot.plot(x, q_values, pen=pg.mkPen("#e3a16c", width=1.2), name="Q")
        spectrum_x = arrays.get("spectrum_frequency_hz")
        spectrum_y = arrays.get("spectrum_power")
        spectrum_ready = (
            spectrum_x is not None
            and spectrum_y is not None
            and len(spectrum_x) > 0
            and len(spectrum_y) > 0
        )
        self.spectrum_stack.setCurrentWidget(
            self.spectrum_plot if spectrum_ready else self.spectrum_empty_state
        )
        if spectrum_ready:
            self.spectrum_plot.plot(
                spectrum_x,
                10.0 * np.log10(np.maximum(spectrum_y, np.finfo(float).tiny)),
                pen=pg.mkPen("#70b9e7", width=1.5),
            )
        self.waterfall_image.clear()
        waterfall = arrays.get("waterfall_magnitude_db")
        waterfall_ready = waterfall is not None and np.size(waterfall) > 0
        self.waterfall_stack.setCurrentWidget(
            self.waterfall_plot if waterfall_ready else self.waterfall_empty_state
        )
        if waterfall_ready:
            self.waterfall_image.setImage(np.asarray(waterfall).T, autoLevels=True)
            waterfall_colormap = pg.ColorMap(
                pos=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                color=np.array(
                    [
                        [31, 25, 66],
                        [55, 85, 141],
                        [31, 145, 155],
                        [89, 190, 128],
                        [249, 221, 111],
                    ],
                    dtype=np.ubyte,
                ),
            )
            self.waterfall_image.setLookupTable(
                waterfall_colormap.getLookupTable(0.0, 1.0, 256)
            )
            waterfall_times = arrays.get("waterfall_time_seconds")
            waterfall_frequencies = arrays.get("waterfall_frequency")
            if waterfall_times is not None and waterfall_frequencies is not None:
                time_values = np.asarray(waterfall_times, dtype=np.float64)
                frequency_values = np.asarray(waterfall_frequencies, dtype=np.float64)
                if time_values.size and frequency_values.size:
                    time_start = float(time_values[0])
                    frequency_start = float(frequency_values[0])
                    time_width = (
                        float(time_values[-1] - time_values[0])
                        + float(time_values[-1] - time_values[-2])
                        if time_values.size > 1
                        else float(duration_seconds or 1.0)
                    )
                    frequency_height = (
                        float(frequency_values[-1] - frequency_values[0])
                        + float(np.mean(np.diff(frequency_values)))
                        if frequency_values.size > 1
                        else 1.0
                    )
                    self.waterfall_image.setRect(QRectF(
                        time_start,
                        frequency_start,
                        time_width if time_width != 0 else 1.0,
                        frequency_height if frequency_height != 0 else 1.0,
                    ))
        frequency_unit = arrays.get("waterfall_frequency_unit", "Hz")
        self.waterfall_plot.setLabel("left", "Frequency", units=frequency_unit)
        self.waterfall_plot.setLabel(
            "bottom", "Time", units=arrays.get("waterfall_time_unit", "samples")
        )
        constellation_i = arrays.get("constellation_i")
        constellation_q = arrays.get("constellation_q")
        constellation_ready = (
            constellation_i is not None
            and constellation_q is not None
            and np.size(constellation_i) > 0
            and np.size(constellation_q) > 0
        )
        self.constellation_stack.setCurrentWidget(
            self.constellation_plot if constellation_ready else self.constellation_empty_state
        )
        if constellation_ready:
            self.constellation_plot.plot(
                constellation_i,
                constellation_q,
                pen=None,
                symbol="o",
                symbolSize=7,
                symbolPen=pg.mkPen("#d9edf7", width=0.8),
                symbolBrush=pg.mkBrush("#70b9e7"),
            )
            combined = np.concatenate((
                np.asarray(constellation_i, dtype=np.float64).reshape(-1),
                np.asarray(constellation_q, dtype=np.float64).reshape(-1),
            ))
            finite = combined[np.isfinite(combined)]
            limit = max(float(np.max(np.abs(finite))) * 1.18, 1.0) if finite.size else 1.0
            self.constellation_plot.setXRange(-limit, limit, padding=0)
            self.constellation_plot.setYRange(-limit, limit, padding=0)

    def reset(self) -> None:
        """Clear selected file, analysis text, plots, and progress state."""
        if self._thread is not None:
            return
        self._set_demo_mode(False)
        self.selected_path = None
        self._last_result = None
        self.source_label.setText("NOT SELECTED")
        self.file_label.setText("No file selected")
        self.overview.clear()
        self.analysis_stack.setCurrentWidget(self.analysis_empty_state)
        self.details.clear()
        self.fec_table.setRowCount(0)
        self._populate_plots({})
        self.progress.setValue(0)
        self.status_label.setStyleSheet("")
        self.status_label.setText("Status: READY")
        self.header_state_label.setText("READY")
        self.header_state_label.setStyleSheet("")
