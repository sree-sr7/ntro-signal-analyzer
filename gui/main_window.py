"""Offline PyQt6/pyqtgraph front end for the public analyzer result API."""

from __future__ import annotations

import html
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject, QRectF, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFocusEvent, QFont, QKeySequence, QShortcut, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
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


_WORKSTATION_STYLE = """
QMainWindow, QWidget {
    background: #17191c; color: #eceff1; font-family: "Segoe UI";
    font-size: @BODY_FONT@;
}
QLabel { background: transparent; }
QFrame#appHeader { background: #17191c; border-bottom: 1px solid #363b41; }
QFrame#inputPanel { background: #202328; border: 1px solid #363b41; border-radius: 5px; }
QLabel#appTitle { font-size: @TITLE_FONT@; font-weight: 600; color: #eceff1; }
QLabel#appSubtitle, QLabel#tabCaption, QLabel#contextText { color: #a7afb7; }
QLabel#sectionEyebrow { color: #a7afb7; font-size: @SMALL_FONT@; font-weight: 600; }
QLabel#metadataLabel { color: #a7afb7; font-size: @SMALL_FONT@; }
QLabel#scoreLabel { color: #a7afb7; }
QPlainTextEdit#diagnosticsText { font-family: "Consolas"; }
QLabel#headerState {
    color: #eceff1; background: #202328; border: 1px solid #363b41;
    border-radius: 5px; min-width: 76px; padding: 5px 9px; font-weight: 600;
}
QLabel#headerState[tone="success"] { color: #a8c9b2; background: #202824; border-color: #43564a; }
QLabel#headerState[tone="warning"] { color: #d6bf8a; background: #28251f; border-color: #5a503a; }
QLabel#headerState[tone="error"] { color: #d2a19d; background: #2b2222; border-color: #60403e; }
QGroupBox {
    border: none; border-top: 1px solid #363b41; margin-top: 15px;
    padding: 13px 0 4px; font-weight: 600; color: #eceff1;
}
QGroupBox::title { subcontrol-origin: margin; left: 0; padding: 0 8px 0 0; }
QLabel#fileInfo { color: #eceff1; font-weight: 600; }
QLabel#demoNote { color: #c1c8ce; border-left: 2px solid #5b87a8; padding: 4px 8px; }
QLabel#tabHeading { color: #eceff1; font-size: @HEADING_FONT@; font-weight: 600; }
QLabel#emptyHeading { color: #eceff1; font-size: @EMPTY_FONT@; font-weight: 600; padding: 4px; }
QLabel#emptyDescription { color: #a7afb7; padding: 4px 20px; }
QLabel#emptyBadge { color: #b5c0c8; border-top: 1px solid #363b41; padding: 8px; margin-top: 10px; }
QToolButton#sectionToggle {
    border: none; border-top: 1px solid #363b41; border-radius: 4px;
    background: transparent; color: #eceff1; text-align: left;
    padding: 9px 4px; font-weight: 600;
}
QToolButton#sectionToggle:hover { background: #282c31; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #17191c; color: #eceff1; border: 1px solid #363b41;
    border-radius: 4px; padding: 5px 7px; min-height: 23px;
    selection-background-color: #476d89;
}
QDoubleSpinBox[required="true"] { border-color: #5b87a8; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPushButton:focus, QCheckBox:focus, QToolButton:focus, QTabBar::tab:focus {
    border: 1px solid #5b87a8;
}
QCheckBox { spacing: 7px; color: #c9d0d5; }
QPushButton {
    color: #eceff1; background: #282c31; border: 1px solid #3b4148;
    border-radius: 5px; padding: 6px 11px; font-weight: 500;
}
QPushButton:hover { background: #32373d; }
QPushButton:disabled { color: #77818a; background: #202328; border-color: #30353b; }
QPushButton#openButton, QPushButton#analyzeButton { background: #5b87a8; border-color: #5b87a8; color: #f5f7f8; }
QPushButton#openButton:hover, QPushButton#analyzeButton:hover { background: #6a95b5; border-color: #6a95b5; }
QPushButton#demoButton { background: transparent; border-color: #3b4148; color: #c8d0d5; }
QPushButton#resetButton, QPushButton#copyButton { background: transparent; border-color: transparent; color: #a7afb7; }
QPushButton#resetButton:hover, QPushButton#copyButton:hover { background: #282c31; border-color: #363b41; color: #eceff1; }
QPushButton#detailLevelButton { background: transparent; border-color: transparent; padding: 4px 8px; color: #a7afb7; }
QPushButton#detailLevelButton:hover { background: #282c31; border-color: transparent; color: #eceff1; }
QPushButton#detailLevelButton:checked { background: #282c31; border-color: #3b4148; color: #eceff1; }
QTabWidget::pane { border: none; border-top: 1px solid #363b41; background: #17191c; }
QTabBar::tab {
    background: transparent; color: #a7afb7; border: none;
    border-bottom: 2px solid transparent; padding: 9px 11px; margin-right: 3px;
}
QTabBar::tab:hover { color: #eceff1; background: #202328; }
QTabBar::tab:selected { color: #eceff1; border-bottom-color: #5b87a8; background: #202328; }
QTextBrowser#analysisBrowser, QPlainTextEdit {
    background: #17191c; color: #eceff1; border: none; padding: 4px;
    selection-background-color: #476d89;
}
QTableWidget { background: #17191c; alternate-background-color: #202328; color: #eceff1; gridline-color: #363b41; border: none; }
QHeaderView::section { background: #202328; color: #c5cbd0; border: none; border-bottom: 1px solid #363b41; padding: 7px; font-weight: 600; }
QStatusBar { background: #17191c; color: #a7afb7; border-top: 1px solid #363b41; }
QProgressBar { color: #a7afb7; background: #202328; border: 1px solid #363b41; border-radius: 3px; min-height: 12px; text-align: center; }
QProgressBar::chunk { background: #5b87a8; border-radius: 2px; }
QScrollArea { border: none; background: transparent; }
QScrollArea#controlsScroll { background: #202328; border-right: 1px solid #363b41; }
QWidget#settingsPane { background: #202328; }
QScrollBar:vertical { background: #202328; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #454c54; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


class _PassiveWheelFilter(QObject):
    """Keep unfocused numeric selectors from consuming scroll-container input."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if not isinstance(watched, QWidget):
            return False
        if event.type() == QEvent.Type.FocusIn and isinstance(event, QFocusEvent):
            keyboard_reasons = {
                Qt.FocusReason.TabFocusReason,
                Qt.FocusReason.BacktabFocusReason,
                Qt.FocusReason.ShortcutFocusReason,
            }
            watched.setProperty(
                "_wheelKeyboardFocused", event.reason() in keyboard_reasons
            )
            return False
        if event.type() == QEvent.Type.FocusOut:
            watched.setProperty("_wheelKeyboardFocused", False)
            return False
        if event.type() != QEvent.Type.Wheel:
            return False
        if self._belongs_to_plot(watched) or self._control_has_keyboard_focus(watched):
            return False

        scroll_area = self._ancestor_scroll_area(watched)
        if scroll_area is not None and isinstance(event, QWheelEvent):
            delta = event.pixelDelta().y()
            if delta == 0:
                angle = event.angleDelta().y()
                steps = angle / 120
                if angle and not steps:
                    steps = 1 if angle > 0 else -1
                delta = int(round(steps * scroll_area.verticalScrollBar().singleStep() * 3))
            bar = scroll_area.verticalScrollBar()
            bar.setValue(bar.value() - delta)
        event.accept()
        return True

    @staticmethod
    def _control_has_keyboard_focus(control: QWidget) -> bool:
        focused = QApplication.focusWidget()
        owns_focus = focused is control or (
            focused is not None and control.isAncestorOf(focused)
        )
        return owns_focus and (
            bool(control.property("_wheelKeyboardFocused"))
            or (
                focused is not None
                and bool(focused.property("_wheelKeyboardFocused"))
            )
        )

    @staticmethod
    def _ancestor_scroll_area(widget: QWidget) -> QScrollArea | None:
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                return parent
            parent = parent.parentWidget()
        return None

    @staticmethod
    def _belongs_to_plot(widget: QWidget) -> bool:
        parent: QWidget | None = widget
        plot_widget_types = (pg.PlotWidget, pg.GraphicsView)
        while parent is not None:
            if isinstance(parent, plot_widget_types) or type(parent).__name__ in {
                "ViewBox", "PlotItem", "GraphicsView", "GraphicsLayoutWidget",
            }:
                return True
            parent = parent.parentWidget()
        return False


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
        self._ui_scale = 1.0
        self._summary_text = ""
        self._active_result_level = "short"
        self._overview_variants: dict[str, str] = {}
        self._wav_channel_count: int | None = None
        self.setWindowTitle("NTRO Signal Analyzer")
        self.setMinimumSize(960, 640)
        self.resize(1440, 900)
        self._build_ui()
        self._install_wheel_filters()
        self._setup_shortcuts()
        self._apply_ui_scale()

    def _build_ui(self) -> None:
        self.setStyleSheet(_WORKSTATION_STYLE)
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 10, 16, 6)
        root_layout.setSpacing(8)

        header = QFrame(root)
        header.setObjectName("appHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(2, 7, 2, 9)
        title_stack = QVBoxLayout()
        title_stack.setSpacing(1)
        title = QLabel("NTRO Signal Analyzer")
        title.setObjectName("appTitle")
        subtitle = QLabel("Offline signal analysis")
        subtitle.setObjectName("appSubtitle")
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header_layout.addLayout(title_stack)
        header_layout.addStretch(1)
        self.header_state_label = QLabel("READY")
        self.header_state_label.setObjectName("headerState")
        self.header_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header_state_label.setProperty("tone", "neutral")
        header_layout.addWidget(self.header_state_label, 0, Qt.AlignmentFlag.AlignVCenter)
        root_layout.addWidget(header)

        input_panel = QFrame(root)
        input_panel.setObjectName("inputPanel")
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(12, 9, 12, 9)
        input_layout.setSpacing(7)

        input_row = QHBoxLayout()
        input_row.setSpacing(9)
        input_heading = QLabel("Input")
        input_heading.setObjectName("sectionEyebrow")
        input_row.addWidget(input_heading)
        self.open_button = QPushButton("Open Signal")
        self.open_button.setObjectName("openButton")
        self.open_button.setDefault(True)
        self.open_button.clicked.connect(self.open_file_dialog)
        input_row.addWidget(self.open_button)
        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("fileInfo")
        self.file_label.setWordWrap(True)
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        input_row.addWidget(self.file_label, 1)
        self.source_label = QLabel("NOT SELECTED")
        self.source_label.setObjectName("metadataLabel")
        input_row.addWidget(self.source_label)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setObjectName("analyzeButton")
        self.analyze_button.clicked.connect(self.start_analysis)
        input_row.addWidget(self.analyze_button)
        input_layout.addLayout(input_row)

        context_row = QHBoxLayout()
        context_row.setSpacing(10)
        self.input_metadata_label = QLabel("Supported inputs: raw .IQ and .WAV")
        self.input_metadata_label.setObjectName("metadataLabel")
        self.input_metadata_label.setWordWrap(True)
        context_row.addWidget(self.input_metadata_label, 1)
        self.input_guidance_label = QLabel("Open a signal to review its input assumptions.")
        self.input_guidance_label.setObjectName("contextText")
        self.input_guidance_label.setWordWrap(True)
        context_row.addWidget(self.input_guidance_label, 2)
        self.stereo_iq = QCheckBox("Interpret channels as I/Q")
        self.stereo_iq.setVisible(False)
        self.stereo_iq.setToolTip(
            "For a two-channel WAV, use channel 1 as I and channel 2 as Q. "
            "Enable this only when the stereo channels contain quadrature components."
        )
        self.stereo_iq.toggled.connect(self._update_stereo_guidance)
        context_row.addWidget(self.stereo_iq)
        self.demo_note_label = QLabel(
            "DEMO / SYNTHETIC · deterministic local signal; not an independent validation dataset."
        )
        self.demo_note_label.setObjectName("demoNote")
        self.demo_note_label.setWordWrap(True)
        self.demo_note_label.setVisible(False)
        context_row.addWidget(self.demo_note_label, 1)
        self.run_demo_button = QPushButton("Run Demo")
        self.run_demo_button.setObjectName("demoButton")
        self.run_demo_button.clicked.connect(self.start_demo)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("resetButton")
        self.reset_button.clicked.connect(self.reset)
        context_row.addWidget(self.run_demo_button)
        context_row.addWidget(self.reset_button)
        input_layout.addLayout(context_row)
        root_layout.addWidget(input_panel)

        content_splitter = QSplitter(Qt.Orientation.Horizontal, root)
        content_splitter.setChildrenCollapsible(False)
        self.controls_scroll = QScrollArea(content_splitter)
        self.controls_scroll.setObjectName("controlsScroll")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.controls_scroll.setMinimumWidth(250)
        self.controls_scroll.setMaximumWidth(390)
        controls = QWidget()
        controls.setObjectName("settingsPane")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 2, 12, 8)
        controls_layout.setSpacing(8)

        basic_group = QGroupBox("Basic settings")
        basic_form = QFormLayout(basic_group)
        basic_form.setContentsMargins(0, 14, 0, 6)
        basic_form.setHorizontalSpacing(9)
        basic_form.setVerticalSpacing(8)
        basic_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.sample_rate = QDoubleSpinBox()
        self.sample_rate.setRange(0.0, 1.0e12)
        self.sample_rate.setDecimals(3)
        self.sample_rate.setValue(0.0)
        self.sample_rate.setSuffix(" Hz")
        self.sample_rate.setSpecialValueText("Enter rate")
        self.sample_rate.setMinimumWidth(110)
        self.sample_rate.valueChanged.connect(self._on_sample_rate_changed)
        self.sample_rate.setToolTip(
            "Raw .IQ files contain no sample-rate metadata. Enter the known rate for "
            "frequency-axis analysis and rate-dependent stages; do not estimate it from samples."
        )
        self.samples_per_symbol = QSpinBox()
        self.samples_per_symbol.setRange(1, 65536)
        self.samples_per_symbol.setValue(1)
        self.samples_per_symbol.setToolTip(
            "This is the input samples-per-symbol value used by the modulation-specific "
            "pipeline. Change it for an oversampled capture when the actual value is known."
        )
        self.run_fec = QCheckBox("Search FEC / interleaver")
        self.run_fec.setChecked(True)
        self.run_fec.setToolTip(
            "Run the FEC and interleaver candidate search after demodulation. Disable it "
            "to skip recovery trials when only signal classification is needed."
        )
        self.sample_rate_note = QLabel("Raw IQ needs a rate; WAV uses its header rate.")
        self.sample_rate_note.setObjectName("contextText")
        self.sample_rate_note.setWordWrap(True)
        basic_form.addRow("Sample rate", self.sample_rate)
        basic_form.addRow("Samples per symbol", self.samples_per_symbol)
        basic_form.addRow(self.run_fec)
        basic_form.addRow(self.sample_rate_note)

        advanced_body = QWidget()
        advanced_layout = QVBoxLayout(advanced_body)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(7)

        raw_group = QGroupBox("Raw IQ format assumptions")
        raw_form = QFormLayout(raw_group)
        raw_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
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
        self.dtype.setToolTip(
            "Select the numeric sample encoding used by this raw IQ file. Raw files carry "
            "no dtype metadata, so change this only when the capture format is known."
        )
        self.iq_layout.setToolTip(
            "Choose how I and Q are stored in the raw samples. Match the documented capture "
            "layout; it is not inferred from the file."
        )
        self.endianness.setToolTip(
            "Choose the byte order of multi-byte raw samples. Change it only when the capture "
            "format specifies a different byte order."
        )
        raw_form.addRow("Sample dtype", self.dtype)
        raw_form.addRow("IQ layout", self.iq_layout)
        raw_form.addRow("Byte order", self.endianness)
        advanced_layout.addWidget(raw_group)

        sync_group = QGroupBox("Timing, synchronization and frame")
        sync_form = QFormLayout(sync_group)
        sync_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
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
        self.timing_recovery.setToolTip(
            "Estimate and recover symbol timing for oversampled captures. Enable when the "
            "capture contains multiple samples per symbol and timing recovery is intended."
        )
        self.rrc_enabled.setToolTip(
            "Apply a root-raised-cosine matched filter before demodulation. Use it when the "
            "transmit pulse shaping is known to be root-raised-cosine."
        )
        self.rrc_rolloff.setToolTip(
            "Set the matched-filter roll-off factor. Change it to the known pulse-shaping "
            "value; it is used only while the filter is enabled."
        )
        self.tone0.setToolTip(
            "Known 2FSK frequency for bit 0, in hertz. Supply both tones when the input is "
            "2FSK and the tone pair is known; otherwise leave both empty."
        )
        self.tone1.setToolTip(
            "Known 2FSK frequency for bit 1, in hertz. Supply both tones when the input is "
            "2FSK and the tone pair is known; otherwise leave both empty."
        )
        self.sync_word.setToolTip(
            "Optional frame synchronization word as hexadecimal bytes, most-significant bit first. "
            "Enter it only when the expected sync word is known."
        )
        self.header_length.setToolTip(
            "Number of header bits after a detected sync word. Set this only when the frame "
            "format specifies a header length."
        )
        self.payload_length.setToolTip(
            "Optional number of payload bits to recover after the header. Leave at zero to "
            "use all remaining bits."
        )
        sync_form.addRow(self.timing_recovery)
        sync_form.addRow(self.rrc_enabled)
        sync_form.addRow("RRC rolloff", self.rrc_rolloff)
        sync_form.addRow("2FSK bit-0 tone", self.tone0)
        sync_form.addRow("2FSK bit-1 tone", self.tone1)
        sync_form.addRow("Sync word (hex)", self.sync_word)
        sync_form.addRow("Header bits after sync", self.header_length)
        sync_form.addRow("Payload bits (0 = all)", self.payload_length)
        advanced_layout.addWidget(sync_group)

        recovery_group = QGroupBox("Recovery overrides")
        fec_form = QFormLayout(recovery_group)
        fec_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.crc_present = QCheckBox("Input includes appended CRC-16")
        self.expected_payload = QSpinBox()
        self.expected_payload.setRange(0, 10_000_000)
        self.expected_payload.setSpecialValueText("Unspecified")
        self.crc_present.setToolTip(
            "Tell the candidate search that the input includes an appended CRC-16. "
            "An unconfigured CRC is not treated as a pass."
        )
        self.expected_payload.setToolTip(
            "Optional expected payload length in bits. Set it only when the frame length "
            "is known so recovery can apply that explicit constraint."
        )
        fec_form.addRow(self.crc_present)
        fec_form.addRow("Expected payload (bits)", self.expected_payload)
        advanced_layout.addWidget(recovery_group)
        self.advanced_section = self._collapsible_section(
            "Advanced settings", advanced_body, expanded=False
        )
        controls_layout.addWidget(basic_group)
        controls_layout.addWidget(self.advanced_section)
        controls_layout.addStretch(1)
        self.controls_scroll.setWidget(controls)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMinimumWidth(500)
        self.analysis_stack = QStackedWidget()
        self.analysis_empty_state = self._empty_state_widget(
            "No signal loaded",
            "Open a raw IQ or WAV capture. The input strip shows known metadata and assumptions.",
            "SUPPORTED INPUTS  .IQ  ·  .WAV",
        )
        self.overview = QTextBrowser()
        self.overview.setReadOnly(True)
        self.overview.setOpenLinks(False)
        self.overview.setOpenExternalLinks(False)
        self.overview.setObjectName("analysisBrowser")
        self.overview.setToolTip(
            "Classifier model scores are uncalibrated ranking scores, not probabilities of correctness."
        )
        result_page = QWidget()
        result_layout = QVBoxLayout(result_page)
        result_layout.setContentsMargins(10, 8, 10, 8)
        result_layout.setSpacing(5)
        result_toolbar = QHBoxLayout()
        result_title = QLabel("Result")
        result_title.setObjectName("sectionEyebrow")
        result_toolbar.addWidget(result_title)
        self.result_level_group = QButtonGroup(self)
        self.result_level_group.setExclusive(True)
        self.result_level_buttons: dict[str, QPushButton] = {}
        for level in ("Short", "Detail", "Technical"):
            button = QPushButton(level)
            button.setObjectName("detailLevelButton")
            button.setCheckable(True)
            button.setToolTip({
                "Short": "Show the verdict and essential result fields.",
                "Detail": "Show analyst-facing interpretation and additional metadata.",
                "Technical": "Open the full raw AnalysisResult diagnostics.",
            }[level])
            self.result_level_group.addButton(button)
            self.result_level_buttons[level.lower()] = button
            result_toolbar.addWidget(button)
        self.result_level_buttons["detail"].setEnabled(False)
        self.result_level_buttons["technical"].setEnabled(False)
        self.result_level_buttons["short"].setChecked(True)
        self.result_level_buttons["short"].clicked.connect(
            lambda: self._show_result_level("short")
        )
        self.result_level_buttons["detail"].clicked.connect(
            lambda: self._show_result_level("detail")
        )
        self.result_level_buttons["technical"].clicked.connect(
            lambda: self._show_result_level("technical")
        )
        result_toolbar.addStretch(1)
        self.model_score_label = QLabel("Model score: unavailable")
        self.model_score_label.setObjectName("scoreLabel")
        self.model_score_label.setToolTip(
            "Model score is the classifier's raw output for its predicted modulation. "
            "It is not a calibrated probability of correctness. Fit evidence, when available, "
            "is a geometric support score and is also not probabilistic."
        )
        result_toolbar.addWidget(self.model_score_label)
        self.copy_summary_button = QPushButton("Copy summary")
        self.copy_summary_button.setObjectName("copyButton")
        self.copy_summary_button.setEnabled(False)
        self.copy_summary_button.clicked.connect(self.copy_summary)
        result_toolbar.addWidget(self.copy_summary_button)
        result_layout.addLayout(result_toolbar)
        result_layout.addWidget(self.overview, 1)
        self.analysis_stack.addWidget(self.analysis_empty_state)
        self.analysis_stack.addWidget(result_page)
        self.analysis_stack.setCurrentWidget(self.analysis_empty_state)
        self.tabs.addTab(self.analysis_stack, "Overview")
        self.tabs.setTabToolTip(0, "Verdict, essential metadata and analysis outcome")

        self.waveform_plot = self._new_plot()
        self.waveform_plot.setLabel("bottom", "Sample index")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.18)
        waveform_tab, self.waveform_empty_state, self.waveform_stack = self._plot_tab(
            "Time-domain waveform", "Analyzer output · I and Q amplitude by sample index",
            self.waveform_plot, "Waveform unavailable",
            "The analyzer returned no waveform samples for this result.",
        )
        self.tabs.addTab(waveform_tab, "Waveform")
        self.tabs.setTabToolTip(1, "Evidence · time-domain waveform")

        self.spectrum_plot = self._new_plot()
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setLabel("left", "Power", units="dB")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.18)
        spectrum_tab, self.spectrum_empty_state, self.spectrum_stack = self._plot_tab(
            "Signal spectrum", "Analyzer-computed power spectrum · frequency in hertz",
            self.spectrum_plot, "Spectrum unavailable",
            "No frequency-domain series is available for this result.",
        )
        self.tabs.addTab(spectrum_tab, "Spectrum")
        self.tabs.setTabToolTip(2, "Evidence · frequency-domain spectrum")

        self.waterfall_plot = self._new_plot()
        self.waterfall_plot.setLabel("bottom", "Time", units="seconds")
        self.waterfall_plot.setLabel("left", "Frequency", units="Hz")
        self.waterfall_image = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_image)
        waterfall_tab, self.waterfall_empty_state, self.waterfall_stack = self._plot_tab(
            "Spectrum over time", "Analyzer time-frequency magnitude · brighter regions show stronger energy",
            self.waterfall_plot, "Waterfall unavailable",
            "The analyzer returned no time-frequency matrix for this result.",
        )
        self.tabs.addTab(waterfall_tab, "Waterfall")
        self.tabs.setTabToolTip(3, "Evidence · time-frequency view")

        self.constellation_plot = self._new_plot()
        self.constellation_plot.setLabel("bottom", "In-phase")
        self.constellation_plot.setLabel("left", "Quadrature")
        self.constellation_plot.showGrid(x=True, y=True, alpha=0.18)
        self.constellation_plot.setAspectLocked(True, ratio=1.0)
        constellation_tab, self.constellation_empty_state, self.constellation_stack = self._plot_tab(
            "Demodulated constellation", "Synchronized symbol points returned by the analyzer",
            self.constellation_plot, "Constellation unavailable",
            "No synchronized constellation points are available in this result.",
        )
        self.tabs.addTab(constellation_tab, "Constellation")
        self.tabs.setTabToolTip(4, "Evidence · synchronized symbol points")

        fec_tab = QWidget()
        fec_layout = QVBoxLayout(fec_tab)
        fec_layout.setContentsMargins(8, 8, 8, 8)
        self.recovery_tabs = QTabWidget()
        self.recovery_tabs.setDocumentMode(True)

        recovery_page = QWidget()
        recovery_layout = QVBoxLayout(recovery_page)
        recovery_layout.setContentsMargins(8, 7, 8, 7)
        recovery_heading = QLabel("FEC and frame recovery")
        recovery_heading.setObjectName("tabHeading")
        fec_caption = QLabel(
            "Accepted outcome is summarized in Overview. Candidate trials are listed here."
        )
        fec_caption.setObjectName("tabCaption")
        fec_actions = QHBoxLayout()
        fec_actions.addStretch(1)
        self.copy_bits_button = QPushButton("Copy recovered bits")
        self.copy_bits_button.setObjectName("copyButton")
        self.copy_bits_button.setEnabled(False)
        self.copy_bits_button.clicked.connect(lambda: self.copy_recovered("binary"))
        self.copy_hex_button = QPushButton("Copy recovered HEX")
        self.copy_hex_button.setObjectName("copyButton")
        self.copy_hex_button.setEnabled(False)
        self.copy_hex_button.clicked.connect(lambda: self.copy_recovered("hex"))
        fec_actions.addWidget(self.copy_bits_button)
        fec_actions.addWidget(self.copy_hex_button)

        self.fec_table = QTableWidget(0, 5)
        self.fec_table.setHorizontalHeaderLabels(
            ["FEC", "Interleaver", "Status", "CRC", "Details"]
        )
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
        recovery_layout.addWidget(recovery_heading)
        recovery_layout.addWidget(fec_caption)
        recovery_layout.addLayout(fec_actions)
        recovery_layout.addWidget(self.fec_table, 1)
        self.recovery_tabs.addTab(recovery_page, "Candidates")

        diagnostics_page = QWidget()
        diagnostics_layout = QVBoxLayout(diagnostics_page)
        diagnostics_layout.setContentsMargins(8, 7, 8, 7)
        diagnostics_header = QHBoxLayout()
        diagnostics_title = QLabel("AnalysisResult diagnostics")
        diagnostics_title.setObjectName("tabHeading")
        diagnostics_header.addWidget(diagnostics_title)
        diagnostics_header.addStretch(1)
        self.copy_diagnostics_button = QPushButton("Copy diagnostics")
        self.copy_diagnostics_button.setObjectName("copyButton")
        self.copy_diagnostics_button.setEnabled(False)
        self.copy_diagnostics_button.clicked.connect(self.copy_diagnostics)
        diagnostics_header.addWidget(self.copy_diagnostics_button)
        diagnostics_note = QLabel("Raw technical output · complete result structure")
        diagnostics_note.setObjectName("tabCaption")
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setObjectName("diagnosticsText")
        self.details.setFont(QFont("Consolas", 9))
        self.details.setPlaceholderText(
            "Technical diagnostics are unavailable until an analysis result is returned."
        )
        self.details.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        diagnostics_layout.addLayout(diagnostics_header)
        diagnostics_layout.addWidget(diagnostics_note)
        diagnostics_layout.addWidget(self.details, 1)
        self.recovery_tabs.addTab(diagnostics_page, "Diagnostics")
        fec_layout.addWidget(self.recovery_tabs)
        self.tabs.addTab(fec_tab, "Recovery")
        self.tabs.setTabToolTip(5, "Accepted recovery, candidate trials and diagnostics")
        self.tabs.currentChanged.connect(self._sync_result_level)
        self.recovery_tabs.currentChanged.connect(self._sync_result_level)

        content_splitter.addWidget(self.controls_scroll)
        content_splitter.addWidget(self.tabs)
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([320, 1060])
        root_layout.addWidget(content_splitter, 1)
        self.setCentralWidget(root)

        self.status_label = QLabel("Status: READY")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.progress.setMinimumWidth(90)
        self.progress.setMaximumWidth(180)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)
        self._update_input_context()

    def _install_wheel_filters(self) -> None:
        self._wheel_filter = _PassiveWheelFilter(self)
        for control in (
            *self.findChildren(QComboBox),
            *self.findChildren(QSpinBox),
            *self.findChildren(QDoubleSpinBox),
        ):
            control.installEventFilter(self._wheel_filter)
            for child in control.findChildren(QWidget):
                child.installEventFilter(self._wheel_filter)

    def _setup_shortcuts(self) -> None:
        self._shortcuts: list[QShortcut] = []
        bindings = (
            ("Ctrl++", 0.1),
            ("Ctrl+=", 0.1),
            ("Ctrl+Shift+=", 0.1),
            ("Ctrl+-", -0.1),
        )
        for sequence, delta in bindings:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda delta=delta: self._change_ui_scale(delta))
            self._shortcuts.append(shortcut)
        reset_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        reset_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        reset_shortcut.activated.connect(lambda: self._change_ui_scale(reset=True))
        self._shortcuts.append(reset_shortcut)

    def _change_ui_scale(self, delta: float = 0.0, *, reset: bool = False) -> None:
        next_scale = 1.0 if reset else min(1.4, max(0.85, self._ui_scale + delta))
        if abs(next_scale - self._ui_scale) < 0.001:
            return
        self._ui_scale = next_scale
        self._apply_ui_scale()

    def _apply_ui_scale(self) -> None:
        scale = self._ui_scale
        stylesheet = (
            _WORKSTATION_STYLE
            .replace("@BODY_FONT@", f"{10 * scale:.1f}pt")
            .replace("@SMALL_FONT@", f"{9 * scale:.1f}pt")
            .replace("@HEADING_FONT@", f"{14 * scale:.1f}pt")
            .replace("@TITLE_FONT@", f"{18 * scale:.1f}pt")
            .replace("@EMPTY_FONT@", f"{19 * scale:.1f}pt")
        )
        self.setStyleSheet(stylesheet)

    def _show_result_level(self, level: str) -> None:
        """Show the selected human or technical depth without duplicating the view."""
        self._active_result_level = level
        if level == "technical":
            self.tabs.setCurrentIndex(5)
            self.recovery_tabs.setCurrentIndex(1)
            return
        self.tabs.setCurrentIndex(0)
        if self._last_result is not None:
            self.analysis_stack.setCurrentIndex(1)
        rendered = self._overview_variants.get(level)
        if rendered:
            self.overview.setHtml(rendered)

    def _sync_result_level(self, *_args: Any) -> None:
        if self.tabs.currentIndex() == 5 and self.recovery_tabs.currentIndex() == 1:
            self.result_level_buttons["technical"].setChecked(True)
        elif self.tabs.currentIndex() == 0:
            selected = (
                "detail"
                if getattr(self, "_active_result_level", "short") == "detail"
                else "short"
            )
            self.result_level_buttons[selected].setChecked(True)

    def _set_header_state(self, text: str, tone: str = "neutral") -> None:
        self.header_state_label.setText(text)
        self.header_state_label.setProperty("tone", tone)
        self.header_state_label.style().unpolish(self.header_state_label)
        self.header_state_label.style().polish(self.header_state_label)
        self.header_state_label.adjustSize()
        text_width = self.header_state_label.sizeHint().width()
        self.header_state_label.setMinimumWidth(max(76, text_width))

    def _on_sample_rate_changed(self, _value: float) -> None:
        self._update_input_context()
        if (
            self._demo_mode
            or self._thread is not None
            or self._last_result is not None
            or self.selected_path is None
            or Path(self.selected_path).suffix.lower() != ".iq"
        ):
            return
        if self.sample_rate.value() > 0:
            self._set_header_state("FILE READY", "neutral")
            self.status_label.setText(
                f"Raw IQ sample rate {self._format_hz(self.sample_rate.value())} is user supplied; review format assumptions."
            )
        else:
            self._set_header_state("NEEDS INPUT", "warning")
            self.status_label.setText(
                "Raw IQ has no embedded sample rate or dtype; review the explicit settings."
            )

    def _clear_result_presentation(self) -> None:
        self._last_result = None
        self._summary_text = ""
        self._overview_variants.clear()
        self._active_result_level = "short"
        self.overview.clear()
        self.details.clear()
        self.fec_table.setRowCount(0)
        self.model_score_label.setText("Model score: unavailable")
        self._update_copy_controls()
        self._populate_plots({})
        self.result_level_buttons["short"].setChecked(True)
        self.result_level_buttons["detail"].setEnabled(False)
        self.result_level_buttons["technical"].setEnabled(False)
        self.recovery_tabs.setCurrentIndex(0)
        self.analysis_stack.setCurrentWidget(self.analysis_empty_state)
        self.tabs.setCurrentIndex(0)

    def _set_copy_acknowledgement(self, button: QPushButton) -> None:
        original = button.property("copy-label")
        if not original:
            original = button.text()
            button.setProperty("copy-label", original)
        button.setText("Copied")

        def restore_label() -> None:
            if button is not None:
                button.setText(str(original))

        QTimer.singleShot(1300, restore_label)

    def _copy_text(self, text: str, button: QPushButton) -> None:
        if not text:
            return
        QApplication.clipboard().setText(text)
        self._set_copy_acknowledgement(button)

    def copy_summary(self) -> None:
        self._copy_text(self._summary_text, self.copy_summary_button)

    def copy_diagnostics(self) -> None:
        self._copy_text(self.details.toPlainText(), self.copy_diagnostics_button)

    def _recovered_bits(self) -> np.ndarray | None:
        if self._last_result is None:
            return None
        arrays = self._last_result.visualization
        for key in (
            "fec_recovered_bits", "crc_verified_bits", "frame_payload_bits", "recovered_bits",
        ):
            values = arrays.get(key)
            if values is not None and np.size(values):
                return np.asarray(values, dtype=np.uint8).reshape(-1)
        return None

    def copy_recovered(self, representation: str) -> None:
        bits = self._recovered_bits()
        if bits is None:
            return
        if representation == "binary":
            self._copy_text("".join(str(int(bit)) for bit in bits), self.copy_bits_button)
        else:
            packed = np.packbits(bits, bitorder="big").tobytes().hex().upper()
            self._copy_text(packed, self.copy_hex_button)

    def _update_copy_controls(self) -> None:
        has_result = self._last_result is not None
        has_bits = self._recovered_bits() is not None
        self.copy_summary_button.setEnabled(has_result)
        self.copy_diagnostics_button.setEnabled(has_result)
        self.copy_bits_button.setEnabled(has_bits)
        self.copy_hex_button.setEnabled(has_bits)
        if has_bits:
            bit_count = int(self._recovered_bits().size)
            self.copy_hex_button.setToolTip(
                "Copies recovered bits packed MSB-first into hexadecimal. "
                + ("The final byte is zero-padded." if bit_count % 8 else "")
            )

    @staticmethod
    def _inspect_wav_header(path: Path) -> dict[str, Any]:
        """Read only RIFF metadata through Python's wave header reader."""
        with wave.open(str(path), "rb") as wav_file:
            channels = int(wav_file.getnchannels())
            sample_rate = int(wav_file.getframerate())
            frame_count = int(wav_file.getnframes())
            sample_width = int(wav_file.getsampwidth())
            compression = str(wav_file.getcomptype())
        return {
            "channels": channels,
            "sample_rate_hz": sample_rate,
            "frame_count": frame_count,
            "sample_width_bits": sample_width * 8,
            "compression": compression,
            "duration_seconds": frame_count / sample_rate if sample_rate > 0 else None,
        }

    def _update_input_context(self) -> None:
        self.stereo_iq.setVisible(False)
        self._wav_channel_count = None
        self.sample_rate.setProperty("required", False)
        self.sample_rate.style().unpolish(self.sample_rate)
        self.sample_rate.style().polish(self.sample_rate)

        if self._demo_mode:
            self.analyze_button.setVisible(False)
            self.source_label.setText("DEMO / SYNTHETIC")
            self.file_label.setText("deterministic_demo.iq")
            self.input_metadata_label.setText(
                f"{self._format_hz(DEFAULT_DEMO_CONFIG.sample_rate_hz)} · demo configuration"
            )
            self.input_guidance_label.setText(
                "Generated locally for workflow demonstration; this is not an independent validation dataset."
            )
            self.sample_rate_note.setText("Demo sample rate is fixed for this generated signal.")
            self.sample_rate.setEnabled(False)
            self.demo_note_label.setVisible(True)
            return

        self.demo_note_label.setVisible(False)
        if self.selected_path is None:
            self.source_label.setText("No input selected")
            self.file_label.setText("No file selected")
            self.analyze_button.setVisible(False)
            self.input_metadata_label.setText("Supported inputs: raw .IQ and .WAV")
            self.input_guidance_label.setText(
                "Raw IQ needs known format assumptions. WAV sample rate and channels come from its header."
            )
            self.sample_rate_note.setText("Raw IQ needs a rate; WAV uses its header rate.")
            self.sample_rate.setEnabled(True)
            self.sample_rate.setProperty("required", False)
            return

        selected = Path(self.selected_path)
        self.analyze_button.setVisible(True)
        if not selected.is_file():
            self.source_label.setText("File path")
            self.file_label.setText(str(selected))
            self.input_metadata_label.setText("File unavailable")
            self.input_guidance_label.setText("The selected path does not exist or is not a file.")
            self.sample_rate.setEnabled(selected.suffix.lower() == ".iq")
            self.sample_rate_note.setText("No file metadata available.")
            return

        suffix = selected.suffix.lower()
        try:
            size_text = f"{selected.stat().st_size:,} bytes"
        except OSError:
            size_text = "size unavailable"
        if suffix == ".iq":
            self.source_label.setText("Raw IQ")
            self.file_label.setText(f"{selected.name} · {size_text}")
            rate = float(self.sample_rate.value())
            if rate > 0:
                self.input_metadata_label.setText(
                    f"RAW IQ · {self._format_hz(rate)} · user supplied"
                )
                self.input_guidance_label.setText(
                    "Sample rate is user supplied. Confirm dtype, I/Q layout and byte order under Advanced Settings."
                )
            else:
                self.input_metadata_label.setText("RAW IQ · sample rate unknown")
                self.input_guidance_label.setText(
                    "Enter the known sample rate. Confirm dtype, I/Q layout and byte order under Advanced Settings."
                )
                self.sample_rate_note.setText(
                    "Sample rate is unknown; add the known value for frequency-axis analysis."
                )
            if rate > 0:
                self.sample_rate_note.setText(
                    "Sample rate is user supplied; format fields are explicit assumptions."
                )
            self.sample_rate.setEnabled(True)
            self.sample_rate.setProperty("required", True)
            self.sample_rate.style().unpolish(self.sample_rate)
            self.sample_rate.style().polish(self.sample_rate)
        elif suffix == ".wav":
            self.source_label.setText("WAV")
            self.file_label.setText(f"{selected.name} · {size_text}")
            self.sample_rate.setEnabled(False)
            self.sample_rate.setProperty("required", False)
            try:
                header = self._inspect_wav_header(selected)
            except Exception as exc:
                self.input_metadata_label.setText("WAV · header inspection unavailable")
                self.input_guidance_label.setText(
                    f"Read-only WAV header inspection failed ({exc}); analysis will report its own file status."
                )
                self.sample_rate_note.setText("WAV sample-rate metadata could not be read.")
                self.status_label.setText("WAV metadata header inspection failed.")
                self._set_header_state("HEADER ERROR", "error")
                return

            rate_text = self._format_hz(header["sample_rate_hz"])
            channels = header["channels"]
            self._wav_channel_count = int(channels)
            channel_text = f"{channels} channel" if channels == 1 else f"{channels} channels"
            duration = header["duration_seconds"]
            duration_text = f" · {duration:.3f} s" if duration is not None else ""
            self.input_metadata_label.setText(
                f"{rate_text} · WAV header · {channel_text} · {header['sample_width_bits']}-bit{duration_text}"
            )
            self.sample_rate.setValue(float(header["sample_rate_hz"]))
            self.sample_rate_note.setText("Sample rate is read from the WAV header.")
            if channels == 2:
                self.stereo_iq.setVisible(True)
                self.stereo_iq.setChecked(False)
                self.input_guidance_label.setText(
                    "Stereo WAV detected. By default the analyzer uses channel 1 only; enable I/Q interpretation when appropriate."
                )
            elif channels > 2:
                self.input_guidance_label.setText(
                    "This WAV has more than two channels; the analyzer reports multichannel input as unsupported."
                )
            else:
                self.input_guidance_label.setText(
                    "Mono WAV detected. The analyzer uses its single channel."
                )
        else:
            self.source_label.setText("Unsupported file")
            self.file_label.setText(f"{selected.name} · {size_text}")
            self.input_metadata_label.setText("Format not accepted by this analyzer")
            self.input_guidance_label.setText(
                "This frozen analyzer accepts raw .IQ and .WAV only; SigMF metadata is not consumed."
                if suffix in {".sigmf-meta", ".sigmf-data"}
                else "This frozen analyzer accepts raw .IQ and .WAV only."
            )
            self.sample_rate_note.setText("No supported input metadata available.")
            self.sample_rate.setEnabled(False)

    def _update_stereo_guidance(self, enabled: bool) -> None:
        if enabled:
            self.input_guidance_label.setText(
                "Stereo WAV I/Q interpretation enabled · channel 1 is I and channel 2 is Q."
            )
        else:
            self.input_guidance_label.setText(
                "Stereo WAV detected. By default the analyzer uses channel 1 only; enable I/Q interpretation when appropriate."
            )
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
        plot.setBackground("#17191c")
        plot.getPlotItem().layout.setContentsMargins(10, 8, 14, 10)
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen("#454c54")
            axis.setTextPen("#a7afb7")
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
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
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
            "Supported signals (*.iq *.IQ *.wav *.WAV);;All files (*)",
        )
        if path:
            self.select_file(path)

    def select_file(self, path: str | Path) -> None:
        """Select a file and inspect only its path and supported header metadata."""
        self._set_demo_mode(False)
        selected = Path(path).expanduser()
        self.selected_path = str(selected)
        self._clear_result_presentation()
        self.progress.setValue(0)
        self.sample_rate.setValue(0.0)
        self.stereo_iq.setChecked(False)
        self._update_input_context()
        if selected.is_file():
            suffix = selected.suffix.lower()
            if suffix == ".iq":
                if self.sample_rate.value() > 0:
                    self._set_header_state("FILE READY", "neutral")
                    self.status_label.setText(
                        f"Raw IQ sample rate {self._format_hz(self.sample_rate.value())} is user supplied; review format assumptions."
                    )
                else:
                    self._set_header_state("NEEDS INPUT", "warning")
                    self.status_label.setText(
                        "Raw IQ has no embedded sample rate or dtype; review the explicit settings."
                    )
            elif suffix == ".wav":
                if "inspection unavailable" in self.input_metadata_label.text():
                    self._set_header_state("HEADER ERROR", "error")
                elif self._wav_channel_count is not None and self._wav_channel_count > 2:
                    self._set_header_state("UNSUPPORTED", "warning")
                    self.status_label.setText(
                        "WAV input with more than two channels is not supported by this analyzer."
                    )
                else:
                    self._set_header_state("FILE SELECTED", "neutral")
                    self.status_label.setText("WAV metadata read from its RIFF header.")
            else:
                self._set_header_state("UNSUPPORTED", "warning")
                self.status_label.setText(
                    "Unsupported format. The analyzer accepts raw .IQ and .WAV files."
                )
        else:
            self._set_header_state("FILE MISSING", "error")
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
            self.sample_rate.setSuffix(" kHz · DEMO")
            self.samples_per_symbol.setValue(DEFAULT_DEMO_CONFIG.samples_per_symbol)
            self.samples_per_symbol.setSuffix(" · DEMO")
            self.sample_rate.setEnabled(False)
            self.samples_per_symbol.setEnabled(False)
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

        self._demo_mode = enabled
        self._update_input_context()

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
        self._clear_result_presentation()
        if self.selected_path is None:
            self.status_label.setText("Choose an .IQ or .wav file before analysis.")
            self._set_header_state("NEEDS INPUT", "warning")
            return
        path = Path(self.selected_path)
        if not path.is_file():
            self.status_label.setText("Selected path does not exist or is not a file.")
            self._set_header_state("FILE MISSING", "error")
            return
        if path.suffix.lower() not in {".iq", ".wav"}:
            self.status_label.setText(
                "Unsupported format. The analyzer accepts raw .IQ and .WAV files."
            )
            self._set_header_state("UNSUPPORTED", "warning")
            return
        try:
            configuration = self._analysis_configuration()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            self._set_header_state("NEEDS INPUT", "warning")
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
        self._clear_result_presentation()
        self.overview.setHtml(
            "<html><body style='background:#17191c;color:#eceff1;font-family:Segoe UI;'>"
            "<div style='color:#a7afb7;'>Analysis in progress</div>"
            "<div style='font-size:19pt;font-weight:600;margin-top:8px;'>Analyzing</div>"
            "<p style='color:#a7afb7;'>The analyzer is running in the background. Progress appears below.</p>"
            "</body></html>"
        )
        self.analysis_stack.setCurrentIndex(1)
        self.open_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.run_demo_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.controls_scroll.setEnabled(False)
        self.stereo_iq.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText(initial_status)
        self._set_header_state("ANALYZING", "neutral")
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
            self._set_header_state(state, "neutral")
        else:
            self.status_label.setText(message)
            self._set_header_state("ANALYZING", "neutral")
        self.progress.setValue(max(0, min(100, int(percent))))

    def _on_result(self, result: AnalysisResult) -> None:
        self.populate_result(result)
        self.analysis_completed.emit(result)

    def _on_worker_error(self, message: str) -> None:
        self._set_header_state("ERROR", "error")
        self.status_label.setText(f"Analysis error: {message}")
        self.status_label.setStyleSheet("")
        self.overview.setHtml(
            "<html><body style='background:#17191c;color:#eceff1;font-family:Segoe UI;'>"
            "<div style='color:#d2a19d;font-size:18pt;font-weight:600;'>Error</div>"
            "<p>Worker stopped before returning an AnalysisResult.</p>"
            f"<p style='color:#a7afb7;'>{html.escape(message)}</p>"
            "</body></html>"
        )
        self._summary_text = f"NTRO Signal Analyzer\nERROR\n{message}"
        self.analysis_stack.setCurrentIndex(1)
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
        self.controls_scroll.setEnabled(True)
        self.stereo_iq.setEnabled(self.stereo_iq.isVisible())

    @staticmethod
    def _bits_preview(value: Any, limit: int = 256) -> str:
        if value is None:
            return "unavailable"
        bits = np.asarray(value, dtype=np.uint8).reshape(-1)
        prefix = "".join(str(int(bit)) for bit in bits[:limit])
        suffix = "…" if bits.size > limit else ""
        return f"{prefix}{suffix} ({bits.size} bits)"

    def populate_result(self, result: AnalysisResult) -> None:
        """Render the public AnalysisResult without changing backend semantics."""
        self._last_result = result
        phase_resolution = result.synchronization.get("phase_ambiguity_resolution", {})
        phase_unresolved = (
            isinstance(phase_resolution, dict)
            and phase_resolution.get("status") == "unresolved"
        )
        raw_status = str(result.status or "unknown").lower()
        display_status = "partial" if phase_unresolved else raw_status
        tone = {
            "complete": "success",
            "partial": "warning",
            "rejected": "error",
            "error": "error",
        }.get(display_status, "neutral")
        self._set_header_state(display_status.upper(), tone)
        self.status_label.setStyleSheet("")
        self.status_label.setText(
            "Status: PARTIAL / phase orientation unresolved"
            if phase_unresolved
            else f"Status: {display_status.upper()}"
        )
        self.progress.setValue(100)
        self.tabs.setCurrentIndex(0)
        self.analysis_stack.setCurrentIndex(1)

        file_info = result.file_metadata
        classification = result.classification
        demodulation = result.demodulation
        synchronization = result.synchronization
        fec = result.fec
        modulation = str(classification.get("modulation", "unknown")).lower()
        if modulation in {"", "none", "unavailable"}:
            modulation = "unknown"
        score = classification.get("confidence")
        score_text = f"{float(score):.3f}" if score is not None else "unavailable"
        classifier_status = str(classification.get("status", "unavailable")).upper()
        method = str(classification.get("method", "unknown"))
        self.model_score_label.setText(
            f"Model score: {score_text}"
            + (" · uncalibrated" if score is not None else "")
        )
        self.result_level_buttons["detail"].setEnabled(True)
        self.result_level_buttons["technical"].setEnabled(True)

        sample_rate = file_info.get("sample_rate_hz")
        sample_rate_text = self._format_hz(sample_rate)
        if result.demo is not None:
            sample_rate_origin = "demo configuration"
        elif sample_rate is None:
            sample_rate_origin = "unknown"
        elif str(result.file_format).lower() == "wav":
            sample_rate_origin = "metadata"
        else:
            sample_rate_origin = "user supplied"
        if sample_rate is not None:
            sample_rate_text = f"{sample_rate_text} · {sample_rate_origin}"
        elif str(result.file_format).lower() == "iq":
            sample_rate_text = "Unknown · required for frequency-axis analysis"
        duration = file_info.get("duration_seconds")
        duration_text = (
            f"{float(duration):.3f} s" if duration is not None else "Unavailable"
        )
        sample_count = file_info.get("sample_count", "Unavailable")

        cfo_result = synchronization.get("coarse_cfo")
        cfo_value = (
            cfo_result.get("estimated_offset_hz")
            if isinstance(cfo_result, dict)
            else None
        )
        cfo_text = "Unavailable" if cfo_value is None else f"{float(cfo_value):+.2f} Hz"
        preprocessing = synchronization.get("preprocessing", {})
        if preprocessing.get("timing_recovery_applied"):
            timing_text = "Applied"
        else:
            effective_sps = preprocessing.get("effective_samples_per_symbol")
            timing_text = (
                f"Not applied · {effective_sps} samples/symbol"
                if effective_sps is not None
                else "Not applied · input setting unavailable"
            )

        fec_status = str(fec.get("status", "unavailable"))
        if fec_status == "accepted":
            recovery_text = "Accepted"
        elif fec_status == "crc_accepted_no_fec":
            recovery_text = "CRC accepted · no FEC"
        elif fec_status in {"no_accepted_candidate", "no_fec_candidate"}:
            recovery_text = "No accepted FEC candidate"
        elif fec_status == "failed":
            recovery_text = "Failed"
        elif fec_status == "not_requested":
            recovery_text = "Not requested"
        else:
            recovery_text = "Unavailable"
        crc_value = fec.get("crc_pass")
        if crc_value is True:
            crc_text = "PASS"
        elif crc_value is False:
            crc_text = "FAIL"
        elif fec.get("crc_present") is False:
            crc_text = "NOT CONFIGURED"
        elif fec_status == "not_requested":
            crc_text = "NOT CHECKED"
        else:
            crc_text = "UNAVAILABLE"

        if result.demo is not None:
            frame_text = "Not configured for demo profile"
            payload_length = result.demo.get("configuration", {}).get("payload_length_bits")
            payload_verification = str(
                result.demo.get("payload_recovery", "not_available")
            ).replace("_", " ").upper()
            frame_value = (
                f"{payload_length} bits recovered"
                if payload_length is not None
                else "Payload unavailable"
            )
        else:
            frame_text = str(result.frames.get("status", "unavailable")).replace(
                "_", " "
            ).title()
            payload_length = result.frames.get("payload_bit_count")
            payload_verification = "Not available"
            frame_value = frame_text
        payload_text = (
            f"{payload_length} bits" if payload_length is not None else "Unavailable"
        )
        fec_name = fec.get("fec_type") or "None accepted"
        interleaver_name = fec.get("interleaver_type") or "None"
        fec_detail = recovery_text
        if fec.get("fec_type"):
            fec_detail += f" · {fec_name}"
        if fec.get("interleaver_type"):
            fec_detail += f" · {interleaver_name}"
        demod_status = str(demodulation.get("status", "unavailable")).upper()
        bit_count = demodulation.get("bit_count", 0)
        llr_count = demodulation.get("llr_count", 0)
        llr_text = f"Available ({llr_count})" if llr_count else "Unavailable"

        def escaped(value: Any) -> str:
            return html.escape(str(value))

        def detail_row(label: str, value: str, *, value_tone: str = "normal") -> str:
            value_color = {
                "success": "#a8c9b2",
                "warning": "#d6bf8a",
                "error": "#d2a19d",
                "normal": "#eceff1",
            }.get(value_tone, "#eceff1")
            return (
                "<tr style='border-bottom:1px solid #363b41;'>"
                f"<td width='30%' valign='top' style='color:#a7afb7;padding:8px 10px 8px 2px;'>{escaped(label)}</td>"
                f"<td valign='top' style='color:{value_color};padding:8px 2px;'>{value}</td></tr>"
            )

        source = str(result.source or "UNKNOWN")
        filename = result.filename or Path(result.input_path).name or "Unknown file"
        status_color = {
            "complete": "#a8c9b2",
            "partial": "#d6bf8a",
            "rejected": "#d2a19d",
            "error": "#d2a19d",
        }.get(display_status, "#a7afb7")
        status_description = {
            "complete": "Analyzer completed the reported stages.",
            "partial": "The result remains partial; review the listed limitations and recovery state.",
            "rejected": (
                "The analyzer did not accept this as a supported result. Review validation "
                "and input-format assumptions before retrying."
            ),
            "error": "Analysis stopped with an error reported by the analyzer.",
            "model_unavailable": (
                "The classifier model was unavailable; no modulation hypothesis was established."
            ),
        }.get(display_status, "The analyzer status is unknown.")
        error_detail = None
        error_stage = None
        if result.errors:
            error_detail = str(
                result.errors[0].get("message", "Analyzer reported an error.")
            )
            error_stage = str(result.errors[0].get("stage", "unknown stage"))
            stage_descriptions = {
                "file": "Input could not be loaded or interpreted.",
                "validation": "Input validation stopped the analysis.",
                "configuration": "An analysis setting could not be applied.",
                "synchronization": "Synchronization stopped with an error.",
                "demodulation": "Demodulation stopped with an error.",
                "fec": "Recovery processing stopped with an error.",
                "frame": "Frame analysis stopped with an error.",
            }
            status_description = stage_descriptions.get(
                error_stage.lower(), "The analyzer reported an error."
            )
        elif phase_unresolved:
            status_description = (
                "Phase orientation could not be resolved. The result remains PARTIAL; "
                "recovery values below are shown exactly as reported."
            )

        fit_score = classification.get("geometric_fit_score")
        fit_text = (
            f"{float(fit_score):.3f}"
            if fit_score is not None
            else "Unavailable"
        )
        classification_line = (
            f"Prediction: {escaped(modulation)}"
            f" · Classifier state: {escaped(classifier_status)}"
            f" · {escaped(method)}"
        )
        modulation_heading = escaped(modulation.upper())
        if phase_unresolved:
            status_description = (
                "Phase orientation unresolved. The analyzer result remains PARTIAL; "
                "inspect demodulation, frame and recovery rows before interpreting payload data."
            )

        alert_html = ""
        if result.errors:
            alert_html = (
                "<p style='margin:9px 0 4px;color:#d2a19d;'>"
                f"Analyzer detail · {escaped(error_detail)}</p>"
            )
        elif result.warnings:
            warning_text = escaped(result.warnings[0])
            alert_html = (
                "<p style='margin:9px 0 4px;color:#d6bf8a;'>"
                f"Analyzer note · {warning_text}</p>"
            )
        demo_html = ""
        if result.demo is not None:
            note = result.demo.get("note")
            demo_html = (
                "<p style='margin:8px 0;color:#a7afb7;border-left:2px solid #5b87a8;padding:4px 8px;'>"
                "DEMO / SYNTHETIC"
                + (f" · {escaped(note)}" if note else "")
                + "</p>"
            )

        font_size = 10 * self._ui_scale
        primary_size = 24 * self._ui_scale
        modulation_size = (
            primary_size
            if len(modulation_heading) <= 5
            else primary_size * max(0.65, 5 / len(modulation_heading))
        )
        sync_value = (
            "Phase orientation unresolved · " if phase_unresolved else ""
        ) + f"CFO {escaped(cfo_text)} · Timing recovery {escaped(timing_text)}"
        fec_tone = (
            "success" if fec_status == "accepted" else
            "warning" if fec_status == "crc_accepted_no_fec" else
            "error" if fec_status in {"failed", "no_accepted_candidate", "no_fec_candidate"} else
            "normal"
        )
        fec_value = fec_detail
        if fec.get("fec_type"):
            fec_value = f"{escaped(recovery_text)} · {escaped(fec_name)}"
        interleaver_value = (
            escaped(interleaver_name) if fec.get("interleaver_type") else
            "None selected" if fec_status == "accepted" else
            "None accepted" if fec_status in {"no_accepted_candidate", "no_fec_candidate"} else
            "Unavailable"
        )
        validation_value = escaped(
            str(result.validation_status).replace("_", " ").upper()
        )
        core_rows = "".join((
            detail_row(
                "Signal",
                f"{escaped(sample_rate_text)} · {escaped(duration_text)} · {escaped(sample_count)} samples",
            ),
            detail_row("Validation", validation_value),
            detail_row(
                "File / source",
                f"{escaped(filename)} · {escaped(result.file_format.upper())} · Source: {escaped(source)}",
            ),
            detail_row(
                "Synchronization", sync_value,
                value_tone="warning" if phase_unresolved else "normal",
            ),
            detail_row(
                "Demodulation",
                f"{escaped(demod_status)} · {escaped(bit_count)} recovered bits",
                value_tone="success" if demod_status == "COMPLETE" else "error" if demod_status == "FAILED" else "normal",
            ),
            detail_row("FEC", fec_value, value_tone=fec_tone),
            detail_row("Interleaver", interleaver_value),
            detail_row(
                "CRC", escaped(crc_text),
                value_tone="success" if crc_text == "PASS" else "error" if crc_text == "FAIL" else "normal",
            ),
            detail_row(
                "Frame / payload",
                f"{escaped(frame_value)} · {escaped(payload_text)} · Verification: {escaped(payload_verification)}",
            ),
        ))
        detailed_rows = core_rows + "".join((
            detail_row("Classifier", f"{escaped(classifier_status)} · {escaped(method)}"),
            detail_row("Fit evidence", escaped(fit_text)),
            detail_row("LLR", escaped(llr_text)),
        ))

        def overview_html(rows: str) -> str:
            return f"""
            <html><body style="background-color:#17191c;color:#eceff1;font-family:'Segoe UI';font-size:{font_size:.1f}pt;margin:8px;">
              <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                <tr>
                  <td width="34%" valign="top" style="padding:8px 22px 8px 2px;">
                    <div style="color:{status_color};font-size:{10 * self._ui_scale:.1f}pt;font-weight:600;">{escaped(display_status.upper())}</div>
                    <div style="color:#5b87a8;font-size:{modulation_size:.1f}pt;font-weight:600;margin:5px 0 2px;">{modulation_heading}</div>
                    <div style="color:#a7afb7;margin:2px 0 10px;">Prediction: {escaped(modulation)} · Classifier state: {escaped(classifier_status)}</div>
                    <div style="color:#c5cbd0;line-height:1.35;">{escaped(status_description)}</div>
                    {demo_html}
                    {alert_html}
                  </td>
                  <td valign="top" style="border-left:1px solid #363b41;padding:2px 2px 2px 22px;">
                    <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                      {rows}
                    </table>
                  </td>
                </tr>
              </table>
            </body></html>
            """

        self._overview_variants = {
            "short": overview_html(core_rows),
            "detail": overview_html(detailed_rows),
        }
        self._active_result_level = "short"
        self.result_level_buttons["short"].setChecked(True)
        self.overview.setHtml(self._overview_variants["short"])

        summary_parts = [
            "NTRO Signal Analyzer",
            display_status.upper(),
            f"Prediction: {modulation}",
            f"Model score: {score_text}"
            + (" (uncalibrated score)" if score is not None else ""),
            f"Classifier state: {classifier_status}",
            f"Source: {source}",
            f"File: {filename}",
            f"Sample rate: {sample_rate_text}",
            f"Validation: {result.validation_status}",
            f"Synchronization: CFO {cfo_text}; timing recovery {timing_text}",
            f"Demodulation: {demod_status}; recovered bits {bit_count}; LLR {llr_text}",
            f"FEC={fec.get('fec_type')}; Interleaver={fec.get('interleaver_type')}; "
            f"Status={fec.get('status')}; CRC={crc_value}",
            f"CRC state: {crc_text}",
            f"Frame / payload: {frame_value}; {payload_text}; verification {payload_verification}",
        ]
        if phase_unresolved:
            summary_parts.append(status_description)
        if result.errors:
            summary_parts.append(
                f"Analyzer error at {error_stage}: {error_detail}"
            )
        elif result.warnings:
            summary_parts.append(f"Analyzer note: {result.warnings[0]}")
        if result.demo is not None and result.demo.get("note"):
            summary_parts.append(f"DEMO / SYNTHETIC: {result.demo['note']}")
        self._summary_text = "\n".join(str(part) for part in summary_parts)
        self._populate_plots(
            result.visualization,
            duration_seconds=float(duration) if duration is not None else None,
        )
        self._populate_fec_table(fec)
        diagnostic_json = json.dumps(
            result.to_dict(include_arrays=False), indent=2, ensure_ascii=False
        )
        self.details.setPlainText(
            f"FEC={fec.get('fec_type')}; Interleaver={fec.get('interleaver_type')}; "
            f"Status={fec.get('status')}; CRC={crc_value}\n\n"
            f"Complete AnalysisResult diagnostics:\n{diagnostic_json}"
        )
        self._update_copy_controls()
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
            status = (
                "ACCEPTED"
                if candidate.get("accepted") is True
                else str(candidate.get("candidate_status", "evaluated")).replace("_", " ").upper()
            )
            crc = candidate.get("crc_pass")
            if crc is True:
                crc_label = "PASS"
            elif crc is False:
                crc_label = "FAIL"
            elif candidate.get("crc_present", fec.get("crc_present")) is False:
                crc_label = "NOT CONFIGURED"
            elif fec.get("status") == "not_requested":
                crc_label = "NOT CHECKED"
            else:
                crc_label = "UNAVAILABLE"
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
        has_result = self._last_result is not None

        def set_empty_text(state_widget: QWidget, title: str, description: str) -> None:
            heading = state_widget.findChild(QLabel, "emptyHeading")
            explanation = state_widget.findChild(QLabel, "emptyDescription")
            if heading is not None:
                heading.setText(title)
            if explanation is not None:
                explanation.setText(description)

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
        if not waveform_ready:
            set_empty_text(
                self.waveform_empty_state,
                "No waveform data" if has_result else "No waveform yet",
                "The analyzer returned no waveform samples for this result."
                if has_result else "Run an analysis to view the time-domain samples.",
            )
        if waveform_ready:
            self.waveform_plot.plot(x, i_values, pen=pg.mkPen("#5b87a8", width=1.4), name="I")
            if q_values is not None and np.any(np.asarray(q_values) != 0):
                self.waveform_plot.plot(x, q_values, pen=pg.mkPen("#a7afb7", width=1.1), name="Q")
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
        if not spectrum_ready:
            if has_result and self._last_result and self._last_result.file_format == "iq" and self._last_result.file_metadata.get("sample_rate_hz") is None:
                spectrum_reason = "No sample rate is available for this raw IQ capture; the frequency axis is unavailable."
            else:
                spectrum_reason = "The analyzer returned no frequency-domain series for this result."
            set_empty_text(
                self.spectrum_empty_state,
                "No spectrum data" if has_result else "No spectrum yet",
                spectrum_reason if has_result else "Run an analysis to view the computed spectrum.",
            )
        if spectrum_ready:
            self.spectrum_plot.plot(
                spectrum_x,
                10.0 * np.log10(np.maximum(spectrum_y, np.finfo(float).tiny)),
                pen=pg.mkPen("#5b87a8", width=1.5),
            )
        self.waterfall_image.clear()
        waterfall = arrays.get("waterfall_magnitude_db")
        waterfall_ready = waterfall is not None and np.size(waterfall) > 0
        self.waterfall_stack.setCurrentWidget(
            self.waterfall_plot if waterfall_ready else self.waterfall_empty_state
        )
        if not waterfall_ready:
            set_empty_text(
                self.waterfall_empty_state,
                "No waterfall data" if has_result else "No waterfall yet",
                "The analyzer returned no time-frequency matrix for this result."
                if has_result else "Run an analysis to view signal energy over time.",
            )
        if waterfall_ready:
            self.waterfall_image.setImage(np.asarray(waterfall).T, autoLevels=True)
            waterfall_colormap = pg.ColorMap(
                pos=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                color=np.array(
                    [
                        [23, 25, 28],
                        [40, 47, 55],
                        [58, 73, 85],
                        [94, 121, 143],
                        [198, 209, 216],
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
        if not constellation_ready:
            set_empty_text(
                self.constellation_empty_state,
                "No constellation data" if has_result else "No constellation yet",
                "No synchronized constellation points are available in this result."
                if has_result else "Run an analysis to view synchronized symbol points when available.",
            )
        if constellation_ready:
            self.constellation_plot.plot(
                constellation_i,
                constellation_q,
                pen=None,
                symbol="o",
                symbolSize=7,
                symbolPen=pg.mkPen("#c5cbd0", width=0.8),
                symbolBrush=pg.mkBrush("#5b87a8"),
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
        self.sample_rate.setValue(0.0)
        self.stereo_iq.setChecked(False)
        self._clear_result_presentation()
        self.progress.setValue(0)
        self.status_label.setStyleSheet("")
        self.status_label.setText("Status: READY")
        self._set_header_state("READY", "neutral")
        self._update_input_context()
