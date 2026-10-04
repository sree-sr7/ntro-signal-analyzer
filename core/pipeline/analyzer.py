"""Public file-to-analysis orchestration over the existing DSP/FEC modules."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.signal import find_peaks, stft

from core.bitstream.frame_analyzer import analyze_frames
from core.common.enums import ClassifierStatus, ModulationType, SignalFormat
from core.common.models import (
    ClassificationResult,
    IQConfig,
    SignalData,
    TimingResult,
)
from core.common.validation import validate_signal_data
from core.dsp.demodulation import demodulate
from core.dsp.modulation_features import extract_modulation_features
from core.dsp.pulse_shaping import matched_filter, root_raised_cosine
from core.dsp.spectral import analyze_spectrum
from core.dsp.symbol_rate import estimate_symbol_rate
from core.fec.trial_engine import TrialSearchResult, run_fec_trials
from core.file_io.format_detector import detect_format
from core.file_io.iq_loader import load_iq
from core.file_io.wav_loader import load_wav
from core.ml.classifier import (
    MIN_CLASS_EVIDENCE_SCORE,
    OnnxModulationClassifier,
    strongest_modulation_evidence,
)
from core.sync.carrier_recovery import (
    DEFAULT_CARRIER_MIN_CONFIDENCE,
    recover_carrier,
)
from core.sync.cfo_estimator import estimate_cfo
from core.sync.timing_recovery import recover_timing


ProgressCallback = Callable[[str, int], None]


def _safe_value(value: Any, *, include_arrays: bool) -> Any:
    """Convert project result values into JSON-compatible values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if not include_arrays:
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if np.iscomplexobj(value):
            return [[float(item.real), float(item.imag)] for item in value.reshape(-1)]
        return value.tolist()
    if isinstance(value, np.generic):
        return _safe_value(value.item(), include_arrays=include_arrays)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if is_dataclass(value):
        return {
            item.name: _safe_value(getattr(value, item.name), include_arrays=include_arrays)
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(_safe_value(key, include_arrays=include_arrays)): _safe_value(
                item, include_arrays=include_arrays
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, include_arrays=include_arrays) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


@dataclass
class AnalysisResult:
    """Stable, serializable output consumed by the GUI and external callers."""

    status: str = "error"
    filename: str = ""
    input_path: str = ""
    file_format: str = "unknown"
    validation_status: str = "not_run"
    file_metadata: dict[str, Any] | None = None
    signal_statistics: dict[str, Any] | None = None
    synchronization: dict[str, Any] | None = None
    features: dict[str, Any] | None = None
    classification: dict[str, Any] | None = None
    demodulation: dict[str, Any] | None = None
    fec: dict[str, Any] | None = None
    frames: dict[str, Any] | None = None
    visualization: dict[str, Any] | None = None
    stages: list[dict[str, Any]] | None = None
    errors: list[dict[str, str]] | None = None
    warnings: list[str] | None = None
    source: str = "FILE"
    demo: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.file_metadata = self.file_metadata or {}
        self.signal_statistics = self.signal_statistics or {}
        self.synchronization = self.synchronization or {}
        self.features = self.features or {}
        self.classification = self.classification or {}
        self.demodulation = self.demodulation or {}
        self.fec = self.fec or {}
        self.frames = self.frames or {}
        self.visualization = self.visualization or {}
        self.stages = self.stages or []
        self.errors = self.errors or []
        self.warnings = self.warnings or []

    @property
    def successful(self) -> bool:
        """True only when classification and demodulation produced usable data."""
        return self.status == "complete"

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Return JSON-safe result metadata; plotting arrays are opt-in."""
        return {
            item.name: _safe_value(
                getattr(self, item.name),
                include_arrays=include_arrays,
            )
            for item in fields(self)
        }

    def to_json(self, *, include_arrays: bool = False, indent: int | None = 2) -> str:
        """Serialize this result without requiring DSP/FEC class imports."""
        return json.dumps(self.to_dict(include_arrays=include_arrays), indent=indent)


class Analyzer:
    """Run explicit file loading, synchronization, classification, demodulation,
    FEC/interleaver trials, and optional frame correlation.

    Raw IQ input requires an explicit IQConfig because raw files contain no
    sample metadata. Unknown symbol timing, tone pairs, CRC presence, and
    protocol frame fields are never silently guessed.
    """

    def __init__(
        self,
        classifier: Any | None = None,
        *,
        model_path: str | Path | None = None,
    ) -> None:
        if classifier is not None and model_path is not None:
            raise ValueError("provide classifier or model_path, not both")
        if classifier is None:
            selected_model = (
                Path(model_path)
                if model_path is not None
                else Path(__file__).resolve().parents[2] / "models" / "production_mlp_final.onnx"
            )
            classifier = OnnxModulationClassifier(selected_model)
        self.classifier = classifier

    @staticmethod
    def _progress(callback: ProgressCallback | None, message: str, percent: int) -> None:
        if callback is not None:
            callback(message, percent)

    @staticmethod
    def _error(result: AnalysisResult, stage: str, code: str, message: str) -> None:
        result.errors.append({"stage": stage, "code": code, "message": message})
        result.stages.append({"name": stage, "status": "failed", "message": message})
        result.status = "error"

    def analyze_file(
        self,
        path: str | Path,
        *,
        iq_config: IQConfig | None = None,
        treat_stereo_as_iq: bool = False,
        samples_per_symbol: float | None = None,
        symbol_rate_bounds: tuple[float, float] | None = None,
        matched_filter_rolloff: float | None = None,
        matched_filter_span: int = 6,
        recover_symbol_timing: bool = False,
        timing_offset: float = 0.0,
        tone_frequencies_hz: tuple[float, float] | None = None,
        expected_payload_length: int | None = None,
        crc_present: bool = False,
        run_fec: bool = True,
        sync_word: np.ndarray | None = None,
        sync_threshold: float = 0.9,
        header_length: int = 0,
        payload_length: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AnalysisResult:
        """Analyze one WAV or explicitly configured raw IQ file.

        PSK/QAM classification features expect symbol-rate samples. For an
        oversampled PSK/QAM capture, request timing recovery and supply its
        samples-per-symbol value. The FSK feature path instead receives the
        waveform and uses samples_per_symbol. These inputs are recorded in the
        result because the current classifier has modulation-specific sample
        assumptions.
        """
        input_path = Path(path).expanduser()
        result = AnalysisResult(
            filename=input_path.name,
            input_path=str(input_path.resolve(strict=False)),
        )
        detected = detect_format(input_path)
        result.file_format = detected.value
        if not input_path.exists() or not input_path.is_file():
            self._error(result, "file", "file_unavailable", f"File does not exist or is not a file: {input_path}")
            return result
        if (
            detected not in (SignalFormat.IQ, SignalFormat.WAV)
            or input_path.suffix.lower() not in (".iq", ".wav")
        ):
            self._error(
                result,
                "file",
                "unsupported_format",
                f"Only .IQ and .wav files are supported by this analyzer, detected {detected.value}",
            )
            return result
        if detected is SignalFormat.IQ and iq_config is None:
            self._error(
                result,
                "file",
                "iq_configuration_required",
                "Raw IQ files have no embedded sample metadata; provide an explicit IQConfig.",
            )
            return result
        if not isinstance(treat_stereo_as_iq, bool) or not isinstance(crc_present, bool):
            self._error(result, "configuration", "invalid_boolean", "Boolean options must be bool values.")
            return result
        try:
            if detected is SignalFormat.IQ:
                loaded = load_iq(input_path, iq_config)
            else:
                loaded = load_wav(input_path, treat_stereo_as_iq=treat_stereo_as_iq)
        except (OSError, TypeError, ValueError) as exc:
            self._error(result, "file", "load_error", str(exc))
            return result
        if not loaded.success or loaded.signal is None:
            self._error(
                result,
                "file",
                "load_failed",
                "; ".join(loaded.errors) or "Input loader returned no signal.",
            )
            result.warnings.extend(loaded.warnings)
            return result
        if any("truncat" in warning.lower() for warning in loaded.warnings):
            self._error(
                result,
                "file",
                "incomplete_sample",
                "The input contains trailing bytes outside a complete IQ sample; strict analysis stopped.",
            )
            result.warnings.extend(loaded.warnings)
            return result
        signal: SignalData = loaded.signal
        signal_validation = validate_signal_data(signal)
        if not signal_validation.valid:
            self._error(
                result,
                "validation",
                "invalid_signal",
                "; ".join(signal_validation.errors),
            )
            result.warnings.extend(signal_validation.warnings)
            return result
        if not np.all(np.isfinite(signal.samples)):
            self._error(
                result,
                "validation",
                "non_finite_samples",
                "Signal contains non-finite values that the DSP stages cannot process.",
            )
            return result
        if signal.samples.size == 0:
            self._error(result, "validation", "empty_signal", "Signal contains no samples.")
            return result

        metadata = signal.metadata
        fs = metadata.sample_rate
        try:
            sample_rate_valid = fs is not None and bool(np.isfinite(fs)) and float(fs) > 0
        except (TypeError, ValueError):
            sample_rate_valid = False
        if fs is not None and not sample_rate_valid:
            result.errors.append({
                "stage": "validation",
                "code": "invalid_sample_rate",
                "message": f"Sample rate must be a positive finite number, got {fs!r}.",
            })
        if not sample_rate_valid:
            result.warnings.append(
                "Positive sample rate unavailable; frequency-axis DSP and rate-dependent stages are skipped."
            )
        if loaded.warnings:
            result.warnings.extend(loaded.warnings)
        if signal_validation.warnings:
            result.warnings.extend(signal_validation.warnings)
        result.validation_status = "valid_with_warnings" if result.warnings else "valid"
        result.file_metadata = {
            "filename": input_path.name,
            "format": metadata.format.value,
            "sample_count": int(signal.samples.size),
            "sample_rate_hz": float(fs) if sample_rate_valid else None,
            "duration_seconds": (
                float(signal.samples.size / fs) if sample_rate_valid else None
            ),
            "center_frequency_hz": metadata.center_frequency,
            "channel_count": int(metadata.num_channels),
            "dtype": metadata.dtype.value if metadata.dtype is not None else None,
            "iq_layout": metadata.iq_layout.value,
            "endianness": metadata.endianness.value,
            "source_metadata": metadata.source_metadata,
        }
        samples = np.asarray(signal.samples)
        result.signal_statistics = {
            "sample_count": int(samples.size),
            "rms_amplitude": float(np.sqrt(np.mean(np.abs(samples) ** 2))),
            "peak_amplitude": float(np.max(np.abs(samples))),
            "mean_i": float(np.mean(samples.real)),
            "mean_q": float(np.mean(samples.imag)),
            "standard_deviation_i": float(np.std(samples.real)),
            "standard_deviation_q": float(np.std(samples.imag)),
        }
        result.stages.append({"name": "file_load_and_validation", "status": "complete"})
        self._progress(progress_callback, "Input loaded and validated", 12)
        self._progress(progress_callback, "Synchronizing signal...", 18)

        if samples_per_symbol is not None and (
            not np.isfinite(samples_per_symbol) or samples_per_symbol < 1
        ):
            self._error(
                result, "configuration", "invalid_samples_per_symbol",
                "samples_per_symbol must be finite and at least one.",
            )
            return result
        if not isinstance(recover_symbol_timing, bool) or not isinstance(run_fec, bool):
            self._error(result, "configuration", "invalid_boolean", "Boolean options must be bool values.")
            return result

        working = np.asarray(samples)
        timing_result: TimingResult | None = None
        preprocessing: dict[str, Any] = {
            "input_sample_count": int(samples.size),
            "cfo_correction_applied": False,
            "matched_filter_applied": False,
            "timing_recovery_requested": recover_symbol_timing,
            "timing_recovery_applied": False,
            "samples_per_symbol_supplied": samples_per_symbol,
        }
        effective_sps = float(samples_per_symbol) if samples_per_symbol is not None else None
        if symbol_rate_bounds is not None and (
            not isinstance(symbol_rate_bounds, (tuple, list))
            or len(symbol_rate_bounds) != 2
        ):
            self._error(
                result, "configuration", "invalid_symbol_rate_bounds",
                "symbol_rate_bounds must contain exactly (minimum_hz, maximum_hz).",
            )
            return result
        if symbol_rate_bounds is not None:
            if not sample_rate_valid:
                result.warnings.append("Symbol-rate estimation skipped because sample rate is unavailable.")
            else:
                try:
                    symbol_rate = estimate_symbol_rate(
                        working, float(fs), float(symbol_rate_bounds[0]), float(symbol_rate_bounds[1])
                    )
                    result.synchronization["symbol_rate"] = _safe_value(
                        symbol_rate, include_arrays=False
                    )
                    if effective_sps is None and symbol_rate.estimated_rate is not None:
                        effective_sps = float(fs) / symbol_rate.estimated_rate
                except (IndexError, TypeError, ValueError) as exc:
                    result.errors.append({
                        "stage": "symbol_rate",
                        "code": "symbol_rate_error",
                        "message": str(exc),
                    })
        if matched_filter_rolloff is not None:
            if effective_sps is None or not float(effective_sps).is_integer() or effective_sps < 2:
                self._error(
                    result, "preprocessing", "rrc_requires_integer_sps",
                    "Matched filtering requires an explicit or estimated integer samples_per_symbol >= 2.",
                )
                return result
            try:
                taps = root_raised_cosine(
                    matched_filter_rolloff, int(effective_sps), matched_filter_span,
                    dtype=np.float64,
                )
                working = matched_filter(working, taps)
                preprocessing.update({
                    "matched_filter_applied": True,
                    "rrc_rolloff": float(matched_filter_rolloff),
                    "rrc_span_symbols": int(matched_filter_span),
                    "rrc_tap_count": int(taps.size),
                })
            except (TypeError, ValueError) as exc:
                self._error(result, "preprocessing", "matched_filter_error", str(exc))
                return result
        if recover_symbol_timing:
            if effective_sps is None or effective_sps < 2:
                self._error(
                    result, "timing", "timing_requires_sps",
                    "Gardner timing recovery requires samples_per_symbol >= 2.",
                )
                return result
            try:
                timing_result = recover_timing(
                    working, effective_sps, timing_offset=timing_offset
                )
                result.synchronization["timing"] = _safe_value(
                    timing_result, include_arrays=False
                )
                if timing_result.synchronized_samples.size:
                    working = timing_result.synchronized_samples
                    preprocessing["timing_recovery_applied"] = True
                else:
                    result.warnings.extend(timing_result.warnings)
            except (TypeError, ValueError) as exc:
                result.errors.append({"stage": "timing", "code": "timing_error", "message": str(exc)})
        preprocessing.update({
            "output_sample_count": int(working.size),
            "effective_samples_per_symbol": effective_sps,
            "classifier_input": (
                "recovered symbol-rate samples"
                if preprocessing["timing_recovery_applied"]
                else "waveform samples"
            ),
            "classifier_feature_assumptions": (
                "PSK/QAM features assume symbol-rate input; FSK features use waveform samples and samples_per_symbol."
            ),
        })
        result.synchronization["preprocessing"] = preprocessing
        if sample_rate_valid and working.size >= 4:
            try:
                spectral = analyze_spectrum(working, float(fs))
                result.synchronization["spectrum_measurements"] = {
                    "peak_frequency_hz": spectral.peak_frequency,
                    "occupied_bandwidth_hz": spectral.occupied_bandwidth,
                    "noise_floor": spectral.noise_floor,
                    "warnings": list(spectral.warnings),
                }
                peaks: list[dict[str, float]] = []
                if spectral.power.size:
                    indices, _ = find_peaks(spectral.power)
                    if indices.size == 0:
                        indices = np.array([int(np.argmax(spectral.power))])
                    selected = sorted(
                        indices.tolist(), key=lambda idx: spectral.power[idx], reverse=True
                    )[:10]
                    peaks = [
                        {
                            "frequency_hz": float(spectral.frequencies[index]),
                            "power": float(spectral.power[index]),
                        }
                        for index in selected
                    ]
                result.visualization["spectrum_frequency_hz"] = spectral.frequencies
                result.visualization["spectrum_power"] = spectral.power
                result.synchronization["dominant_spectral_peaks"] = peaks
            except (TypeError, ValueError) as exc:
                result.warnings.append(f"Spectrum analysis unavailable: {exc}")
        elif sample_rate_valid:
            result.warnings.append("Spectrum skipped: at least four samples are required.")

        plot_sample_limit = 4096
        preview_step = max(1, int(math.ceil(working.size / plot_sample_limit)))
        result.visualization["waveform_sample_index"] = np.arange(
            0, working.size, preview_step, dtype=np.int64
        )
        result.visualization["waveform_i"] = working[::preview_step].real
        result.visualization["waveform_q"] = working[::preview_step].imag
        waterfall_fs = float(fs) if sample_rate_valid else 1.0
        if working.size >= 8:
            segment_length = min(256, int(working.size))
            frequencies, times, values = stft(
                working,
                fs=waterfall_fs,
                window="hann",
                nperseg=segment_length,
                noverlap=segment_length // 2,
                nfft=segment_length,
                detrend=False,
                return_onesided=False,
                boundary=None,
            )
            frequencies = np.fft.fftshift(frequencies)
            power_db = 20.0 * np.log10(
                np.maximum(np.abs(np.fft.fftshift(values, axes=0)), np.finfo(float).tiny)
            )
            result.visualization.update({
                "waterfall_frequency": frequencies,
                "waterfall_time_seconds": times,
                "waterfall_magnitude_db": power_db,
                "waterfall_frequency_unit": "Hz" if sample_rate_valid else "cycles/sample",
                "waterfall_time_unit": "seconds" if sample_rate_valid else "samples",
            })
        result.stages.append({"name": "preprocessing_and_spectrum", "status": "complete"})
        self._progress(progress_callback, "Preprocessing and signal measurements complete", 35)

        classifier_samples = np.asarray(working)
        classifier_transient_discarded = 0
        if preprocessing["timing_recovery_applied"]:
            # The Gardner loop needs time to acquire symbol timing; its early
            # output has arbitrary sampling phase and corrupts the constellation
            # features (observed at >= 6 samples/symbol). Classify on the settled
            # portion only. Demodulation below still uses all of `working`.
            discard = int(0.3 * classifier_samples.size)
            if classifier_samples.size - discard >= 256:
                classifier_samples = classifier_samples[discard:]
                classifier_transient_discarded = discard
        preprocessing["classifier_timing_transient_discarded_samples"] = (
            classifier_transient_discarded
        )
        classifier_sps = (
            1.0 if preprocessing["timing_recovery_applied"]
            else (effective_sps if effective_sps is not None else 1.0)
        )
        feature_result = None
        self._progress(progress_callback, "Extracting modulation features...", 40)
        try:
            feature_result = extract_modulation_features(
                classifier_samples, samples_per_symbol=classifier_sps
            )
            result.features = {
                "names": list(feature_result.feature_names),
                "values": [float(value) for value in feature_result.values],
                "by_name": feature_result.as_dict(),
                "sample_count": feature_result.sample_count,
                "diagnostics": dict(feature_result.diagnostics),
                "warnings": list(feature_result.warnings),
                "input_samples_per_symbol": classifier_sps,
            }
        except (TypeError, ValueError, ArithmeticError) as exc:
            result.features = {"status": "error", "errors": [str(exc)]}
            result.errors.append({
                "stage": "features", "code": "feature_extraction_error", "message": str(exc)
            })
        self._progress(progress_callback, "Classifying modulation...", 47)
        try:
            classification: ClassificationResult = self.classifier.classify(
                classifier_samples, samples_per_symbol=classifier_sps
            )
        except (TypeError, ValueError, OSError) as exc:
            result.classification = {
                "modulation": ModulationType.UNKNOWN.value,
                "status": ClassifierStatus.ERROR.value,
                "method": getattr(self.classifier, "method", "classifier"),
                "errors": [str(exc)],
                "warnings": [],
                "confidence": 0.0,
                "probabilities": {},
                "top_candidates": [],
            }
            result.errors.append({
                "stage": "classification", "code": "classifier_error", "message": str(exc)
            })
            result.status = "rejected"
            result.stages.append({"name": "classification", "status": "failed"})
            return result
        model_modulation = classification.modulation
        if feature_result is not None:
            strongest_fit, fit_score = strongest_modulation_evidence(
                feature_result.as_dict()
            )
            signal_gate_rejected = fit_score < MIN_CLASS_EVIDENCE_SCORE
            gate_reason = (
                "No supported modulation has a sufficient geometric signal fit; "
                "the classifier label is treated as unknown/OOD."
                if signal_gate_rejected else
                "At least one supported modulation has sufficient geometric signal fit."
            )
            signal_gate = {
                "status": "rejected" if signal_gate_rejected else "passed",
                "decision": "unknown_ood" if signal_gate_rejected else "signal_candidate",
                "reason": gate_reason,
                "stage": "signal_detection",
                "strongest_supported_modulation": strongest_fit.value,
                "geometric_fit_score": fit_score,
                "minimum_geometric_fit_score": MIN_CLASS_EVIDENCE_SCORE,
                "score_interpretation": (
                    "bounded geometric fit score; not a probability or OOD probability"
                ),
            }
            if signal_gate_rejected:
                classification.modulation = ModulationType.UNKNOWN
                classification.warnings.append(
                    f"OOD: {gate_reason}"
                )
        else:
            signal_gate_rejected = False
            signal_gate = {
                "status": "unavailable",
                "decision": "not_evaluated",
                "reason": "Modulation features were unavailable, so signal support could not be evaluated.",
                "stage": "signal_detection",
                "score_interpretation": "no score produced",
            }
        classifier_confidence_before_gate = float(classification.confidence)
        classifier_probabilities_before_gate = dict(classification.probabilities)

        # The current ONNX feature contract is phase-sensitive.  For an
        # accepted PSK candidate, estimate/remove CFO for the classifier input,
        # then keep the correction only when the classifier still returns a
        # supported result with no confidence loss.  Demodulation continues
        # from `working` and performs its own synchronization pass.
        cfo_classification = {
            "status": "not_applied",
            "reason": "No accepted PSK classification required classifier-side CFO correction.",
            "modulation_before": model_modulation.value,
            "confidence_before": float(classification.confidence),
        }
        if (
            not signal_gate_rejected
            and model_modulation in (
                ModulationType.BPSK,
                ModulationType.QPSK,
                ModulationType.PSK8,
            )
            and classification.status is ClassifierStatus.SUCCESS
            and sample_rate_valid
            and classifier_samples.size >= 8
        ):
            order = {
                ModulationType.BPSK: 2,
                ModulationType.QPSK: 4,
                ModulationType.PSK8: 8,
            }[model_modulation]
            classifier_sample_rate = (
                float(fs) / effective_sps
                if preprocessing["timing_recovery_applied"] and effective_sps
                else float(fs)
            )
            try:
                classifier_cfo = estimate_cfo(
                    classifier_samples, classifier_sample_rate, order
                )
                cfo_classification.update({
                    "estimate_hz": classifier_cfo.estimated_offset_hz,
                    "estimator_confidence": classifier_cfo.confidence,
                    "modulation_order": order,
                })
                if classifier_cfo.estimated_offset_hz is not None:
                    indices = np.arange(classifier_samples.size, dtype=np.float64)
                    corrected_samples = classifier_samples * np.exp(
                        -2j * np.pi * classifier_cfo.estimated_offset_hz
                        * indices / classifier_sample_rate
                    )
                    corrected_features = extract_modulation_features(
                        corrected_samples, samples_per_symbol=classifier_sps
                    )
                    corrected_classification = self.classifier.classify(
                        corrected_samples, samples_per_symbol=classifier_sps
                    )
                    if (
                        corrected_classification.status is ClassifierStatus.SUCCESS
                        and corrected_classification.modulation is not ModulationType.UNKNOWN
                        and corrected_classification.confidence >= classification.confidence
                    ):
                        classification = corrected_classification
                        feature_result = corrected_features
                        classifier_samples = corrected_samples
                        result.features = {
                            "names": list(feature_result.feature_names),
                            "values": [float(value) for value in feature_result.values],
                            "by_name": feature_result.as_dict(),
                            "sample_count": feature_result.sample_count,
                            "diagnostics": dict(feature_result.diagnostics),
                            "warnings": list(feature_result.warnings),
                            "input_samples_per_symbol": classifier_sps,
                        }
                        cfo_classification.update({
                            "status": "applied",
                            "reason": "The CFO-corrected classifier result remained supported and did not lose confidence.",
                            "modulation_after": classification.modulation.value,
                            "confidence_before": float(cfo_classification["confidence_before"]),
                            "confidence_after": float(classification.confidence),
                        })
                    else:
                        cfo_classification.update({
                            "status": "not_applied",
                            "reason": "The corrected classifier result was unknown, unavailable, or less confident.",
                            "modulation_after": corrected_classification.modulation.value,
                            "confidence_after": float(corrected_classification.confidence),
                        })
                else:
                    cfo_classification["reason"] = (
                        "The PSK CFO estimator did not return a usable offset."
                    )
            except (TypeError, ValueError, ArithmeticError, OSError) as exc:
                cfo_classification.update({
                    "status": "error",
                    "reason": f"Classifier-side CFO correction failed safely: {exc}",
                })
        result.synchronization["classifier_cfo_correction"] = cfo_classification
        result.classification = _safe_value(classification, include_arrays=False)
        result.classification["signal_detection_gate"] = signal_gate
        if signal_gate_rejected:
            result.classification["model_prediction_before_signal_gate"] = {
                "modulation": model_modulation.value,
                "confidence": classifier_confidence_before_gate,
                "probabilities": _safe_value(
                    classifier_probabilities_before_gate, include_arrays=False
                ),
            }
            # The classifier confidence belongs to its rejected class, not to
            # the resulting UNKNOWN/OOD decision.
            result.classification["confidence"] = None
            result.classification["probabilities"] = {}
            result.classification["top_candidates"] = []
        result.stages.append({
            "name": "classification",
            "status": "complete" if classification.status is ClassifierStatus.SUCCESS else "failed",
            "modulation": classification.modulation.value,
            "model_modulation_before_signal_gate": model_modulation.value,
            "classifier_status": classification.status.value,
        })
        result.stages.append({
            "name": "signal_detection_gate",
            "status": "rejected" if signal_gate_rejected else signal_gate["status"],
            "decision": signal_gate["decision"],
            "reason": signal_gate["reason"],
        })
        self._progress(progress_callback, "Modulation classification complete", 52)
        if (
            classification.modulation is ModulationType.UNKNOWN
            or classification.status is not ClassifierStatus.SUCCESS
        ):
            if signal_gate_rejected:
                result.classification["rejection"] = {
                    "decision": "signal_rejected_unknown_ood",
                    "reason": signal_gate["reason"],
                    "stage": "signal_detection",
                }
            else:
                result.classification["rejection"] = {
                    "decision": "unknown",
                    "reason": (
                        "The configured classifier returned no supported modulation."
                    ),
                    "stage": "classification",
                }
            result.status = "rejected"
            result.warnings.extend(classification.warnings)
            result.errors.extend(
                {"stage": "classification", "code": "model_rejected", "message": error}
                for error in classification.errors
            )
            if classification.status is ClassifierStatus.MODEL_UNAVAILABLE:
                result.warnings.append(
                    "Production classifier unavailable; no classical fallback or forced modulation was used."
                )
            result.demodulation = {
                "status": "unavailable",
                "reason": "No supported modulation was accepted by the configured classifier.",
                "bit_count": 0,
                "llr_count": 0,
            }
            result.fec = {"status": "unavailable", "reason": "Demodulation did not produce bits."}
            result.frames = {
                "status": "unavailable",
                "reason": "Demodulation did not produce a bitstream.",
            }
            self._progress(progress_callback, "Signal rejected or classifier unavailable", 100)
            return result

        modulation = classification.modulation
        representation_error: str | None = None
        if modulation in (
            ModulationType.BPSK,
            ModulationType.QPSK,
            ModulationType.PSK8,
            ModulationType.QAM16,
        ):
            if (
                effective_sps is not None
                and effective_sps > 1
                and not preprocessing["timing_recovery_applied"]
            ):
                representation_error = (
                    "The classifier accepted a PSK/QAM label from waveform-rate samples, "
                    "but the feature contract requires synchronized symbol-rate samples. "
                    "Request successful timing recovery or provide symbol-rate input."
                )
        elif modulation is ModulationType.FSK2:
            if preprocessing["timing_recovery_applied"]:
                representation_error = (
                    "The classifier used timing-recovered symbol samples, while the FSK "
                    "feature and demodulator contract requires waveform samples."
                )
            elif effective_sps is None or effective_sps < 2:
                representation_error = (
                    "2FSK classification requires waveform samples and "
                    "samples_per_symbol >= 2."
                )
        if representation_error is not None:
            result.classification["analysis_input_contract"] = "rejected"
            result.errors.append({
                "stage": "classification",
                "code": "sample_representation_mismatch",
                "message": representation_error,
            })
            result.status = "rejected"
            result.demodulation = {
                "status": "unavailable",
                "selected_modulation": modulation.value,
                "reason": representation_error,
                "bit_count": 0,
                "llr_count": 0,
            }
            result.fec = {
                "status": "unavailable",
                "reason": "Classifier sample representation did not satisfy the selected modulation contract.",
            }
            result.frames = {
                "status": "unavailable",
                "reason": "Demodulation did not run after sample-representation rejection.",
            }
            self._progress(progress_callback, "Classifier input representation was incompatible", 100)
            return result

        self._progress(progress_callback, "Demodulating...", 57)
        demod_samples = np.asarray(working)
        synchronization_details: dict[str, Any] = {
            "coarse_cfo": None,
            "carrier_recovery": None,
            "phase_ambiguity": "not resolved by this pipeline",
        }
        phase_reference = 0.0
        phase_rotation_order: int | None = None
        effective_sample_rate = (
            float(fs) / effective_sps
            if sample_rate_valid and preprocessing["timing_recovery_applied"]
            and effective_sps is not None
            else (float(fs) if sample_rate_valid else None)
        )
        if (
            modulation in (ModulationType.BPSK, ModulationType.QPSK, ModulationType.PSK8)
            and effective_sample_rate is not None
            and demod_samples.size >= 8
        ):
            modulation_order = {
                ModulationType.BPSK: 2,
                ModulationType.QPSK: 4,
                ModulationType.PSK8: 8,
            }[modulation]
            cfo = estimate_cfo(demod_samples, effective_sample_rate, modulation_order)
            synchronization_details["coarse_cfo"] = _safe_value(cfo, include_arrays=False)
            if cfo.estimated_offset_hz is not None:
                indices = np.arange(demod_samples.size, dtype=np.float64)
                demod_samples = demod_samples * np.exp(
                    -2j * np.pi * cfo.estimated_offset_hz * indices / effective_sample_rate
                )
                synchronization_details["coarse_cfo_applied_hz"] = cfo.estimated_offset_hz
            carrier = recover_carrier(
                demod_samples,
                effective_sample_rate,
                modulation_order,
                constellation_phase_rad=(
                    np.pi / 4.0 if modulation is ModulationType.QPSK else 0.0
                ),
            )
            synchronization_details["carrier_recovery"] = _safe_value(
                carrier, include_arrays=False
            )
            if carrier.recovered_samples.size:
                demod_samples = carrier.recovered_samples
                phase_reference = 0.0
            elif "coarse_cfo_applied_hz" in synchronization_details:
                maximum = float(np.max(np.abs(demod_samples)))
                if maximum > 0.0:
                    unit = demod_samples.astype(np.complex128, copy=False) / np.maximum(
                        np.abs(demod_samples), np.finfo(float).tiny
                    )
                    phase_vector = np.mean(unit ** modulation_order)
                    phase_coherence = float(np.abs(phase_vector))
                    if phase_coherence >= DEFAULT_CARRIER_MIN_CONFIDENCE:
                        phase_reference = float(np.angle(phase_vector) / modulation_order)
                        if modulation_order == 4:
                            # QPSK's shared constellation starts at pi/4; the
                            # fourth-power phase includes that fixed anchor.
                            phase_reference -= np.pi / 4.0
                        synchronization_details["phase_only_recovery"] = {
                            "status": "applied",
                            "method": "circular mean after accepted coarse CFO correction",
                            "coherence": phase_coherence,
                            "minimum_coherence": DEFAULT_CARRIER_MIN_CONFIDENCE,
                            "phase_reference_rad": phase_reference,
                        }
                    else:
                        synchronization_details["phase_only_recovery"] = {
                            "status": "rejected",
                            "method": "circular mean after accepted coarse CFO correction",
                            "coherence": phase_coherence,
                            "minimum_coherence": DEFAULT_CARRIER_MIN_CONFIDENCE,
                            "reason": "Circular phase evidence was below the carrier recovery minimum.",
                        }
            if modulation in (
                ModulationType.BPSK,
                ModulationType.QPSK,
                ModulationType.PSK8,
            ):
                phase_rotation_order = modulation_order
        if phase_rotation_order is not None:
            phase_count = phase_rotation_order
            synchronization_details["phase_ambiguity_resolution"] = {
                "status": "unresolved",
                "method": "rotational alternatives require sync or CRC evidence",
                "candidate_count": phase_count,
                "reason": (
                    "No phase candidate has been selected yet; a CRC-validated FEC search "
                    "can resolve it when the CRC is declared present."
                ),
            }
        result.synchronization.update(synchronization_details)
        if modulation in (ModulationType.BPSK, ModulationType.QPSK, ModulationType.PSK8, ModulationType.QAM16):
            demod_sps = 1.0 if preprocessing["timing_recovery_applied"] else effective_sps
        else:
            demod_sps = effective_sps
        try:
            if modulation is ModulationType.FSK2:
                if not sample_rate_valid or demod_sps is None or not float(demod_sps).is_integer():
                    raise ValueError(
                        "2FSK demodulation requires a valid sample rate and integer samples_per_symbol."
                    )
                if tone_frequencies_hz is None:
                    raise ValueError(
                        "2FSK demodulation requires explicit (bit-0, bit-1) tone_frequencies_hz."
                    )
                demod_result = demodulate(
                    demod_samples,
                    modulation,
                    sample_rate=float(fs),
                    samples_per_symbol=int(demod_sps),
                    tone_frequencies_hz=tone_frequencies_hz,
                )
            else:
                demod_result = demodulate(
                    demod_samples,
                    modulation,
                    phase_reference=phase_reference,
                )
        except (TypeError, ValueError) as exc:
            result.demodulation = {
                "status": "failed",
                "selected_modulation": modulation.value,
                "reason": str(exc),
                "bit_count": 0,
                "llr_count": 0,
            }
            result.errors.append({"stage": "demodulation", "code": "demodulation_error", "message": str(exc)})
            result.status = "partial"
            result.stages.append({"name": "demodulation", "status": "failed"})
            result.fec = {"status": "unavailable", "reason": "Demodulator configuration was invalid."}
            result.frames = {
                "status": "unavailable",
                "reason": "Demodulation did not produce a bitstream.",
            }
            self._progress(progress_callback, "Demodulation configuration failed", 100)
            return result
        result.demodulation = {
            "status": "complete" if demod_result.bits.size else "failed",
            "selected_modulation": modulation.value,
            "demodulator": (
                "2FSK noncoherent correlation-energy"
                if modulation is ModulationType.FSK2
                else "coherent nearest-constellation"
            ),
            "input_sample_count": int(demod_samples.size),
            "samples_per_symbol": demod_sps,
            "bit_count": int(demod_result.bits.size),
            "llr_count": int(demod_result.llrs.size),
            "confidence": float(demod_result.confidence),
            "warnings": list(demod_result.warnings),
            "diagnostics": dict(demod_result.diagnostics),
            "llr_source": (
                "measured max-log constellation metrics"
                if demod_result.llrs.size else "not provided by this demodulator"
            ),
            "synchronization_assumptions": [
                "Caller-provided sample rate and timing parameters are used.",
                "Rotational phase ambiguity is not resolved without a known frame/sync reference.",
            ],
            "phase_reference_rad": phase_reference,
        }
        if demod_result.bits.size:
            result.visualization["constellation_i"] = demod_result.symbol_values.real
            result.visualization["constellation_q"] = demod_result.symbol_values.imag
            result.visualization["recovered_bits"] = demod_result.bits
        result.stages.append({
            "name": "demodulation",
            "status": "complete" if demod_result.bits.size else "failed",
        })
        self._progress(progress_callback, "Demodulation complete", 66)
        if not demod_result.bits.size:
            result.status = "partial"
            result.warnings.extend(demod_result.warnings)
            result.fec = {"status": "unavailable", "reason": "Demodulator returned no bits."}
            result.frames = {
                "status": "unavailable",
                "reason": "Demodulation did not produce a bitstream.",
            }
            self._progress(progress_callback, "No bitstream recovered", 100)
            return result

        trial_search: TrialSearchResult | None = None
        if run_fec:
            self._progress(progress_callback, "Testing FEC/interleavers and CRC...", 70)
            try:
                phase_search_enabled = bool(
                    phase_rotation_order is not None
                    and crc_present
                )
                phase_references = (
                    [
                        phase_reference + 2.0 * np.pi * k / phase_rotation_order
                        for k in range(phase_rotation_order)
                    ]
                    if phase_search_enabled and phase_rotation_order is not None
                    else [phase_reference]
                )
                selected_phase_index = 0
                phase_search_attempts = 0
                for phase_index, candidate_phase in enumerate(phase_references):
                    candidate_demod = (
                        demod_result
                        if phase_index == 0
                        else demodulate(
                            demod_samples,
                            modulation,
                            phase_reference=candidate_phase,
                        )
                    )
                    candidate_soft_values = (
                        candidate_demod.llrs
                        if candidate_demod.llrs.size == candidate_demod.bits.size
                        else None
                    )
                    candidate_search = run_fec_trials(
                        candidate_demod.bits,
                        expected_payload_length=expected_payload_length,
                        crc_present=crc_present,
                        soft_llrs=candidate_soft_values,
                    )
                    phase_search_attempts += 1
                    if trial_search is None:
                        trial_search = candidate_search
                    if (
                        candidate_search.best_result is not None
                        and candidate_search.best_result.accepted
                    ):
                        trial_search = candidate_search
                        demod_result = candidate_demod
                        selected_phase_index = phase_index
                        break
                if phase_search_enabled:
                    phase_resolution = {
                        "status": (
                            "crc_selected"
                            if trial_search is not None
                            and trial_search.best_result is not None
                            and trial_search.best_result.accepted
                            else "unresolved"
                        ),
                        "method": "CRC-validated FEC candidate across PSK rotations",
                        "candidate_count": len(phase_references),
                        "candidates_tested": phase_search_attempts,
                        "selected_candidate_index": (
                            selected_phase_index
                            if trial_search is not None
                            and trial_search.best_result is not None
                            and trial_search.best_result.accepted
                            else None
                        ),
                        "selected_phase_reference_rad": (
                            phase_references[selected_phase_index]
                            if trial_search is not None
                            and trial_search.best_result is not None
                            and trial_search.best_result.accepted
                            else None
                        ),
                        "reason": (
                            "A CRC-accepted candidate selected the phase rotation."
                            if trial_search is not None
                            and trial_search.best_result is not None
                            and trial_search.best_result.accepted
                            else "No tested phase rotation produced a CRC-accepted candidate."
                        ),
                    }
                    result.synchronization["phase_ambiguity_resolution"] = phase_resolution
                    if selected_phase_index:
                        result.demodulation.update({
                            "phase_reference_rad": phase_references[selected_phase_index],
                            "phase_ambiguity_candidates_tested": phase_search_attempts,
                            "confidence": float(demod_result.confidence),
                            "warnings": list(demod_result.warnings),
                            "diagnostics": dict(demod_result.diagnostics),
                        })
                        result.visualization["constellation_i"] = demod_result.symbol_values.real
                        result.visualization["constellation_q"] = demod_result.symbol_values.imag
                        result.visualization["recovered_bits"] = demod_result.bits
                assert trial_search is not None
                soft_values = (
                    demod_result.llrs
                    if demod_result.llrs.size == demod_result.bits.size
                    else None
                )
                best = trial_search.best_result
                verified_uncoded_crc = bool(
                    best is not None
                    and best.fec_type is None
                    and crc_present
                    and best.accepted
                    and best.crc_pass is True
                )
                accepted_fec = bool(
                    best is not None and best.fec_type is not None and best.accepted
                )
                result.fec = {
                    "status": (
                        "accepted" if accepted_fec
                        else "crc_accepted_no_fec" if verified_uncoded_crc
                        else "no_fec_candidate" if best is not None and best.fec_type is None
                        else "no_accepted_candidate"
                    ),
                    "candidate_count": len(trial_search.ranked_trials),
                    "fec_type": trial_search.fec_type,
                    "interleaver_type": trial_search.interleaver_type,
                    "crc_pass": trial_search.crc_pass,
                    "crc_present": crc_present,
                    "expected_payload_length": expected_payload_length,
                    "soft_llrs_supplied": soft_values is not None,
                    "best_candidate_accepted": bool(best.accepted) if best is not None else False,
                    "best_candidate_valid": bool(best.valid) if best is not None else False,
                    "decoded_bit_count": int(trial_search.decoded_bits.size),
                    "codeword_count": best.codeword_count if best is not None else None,
                    "codeword_length_bits": best.codeword_length_bits if best is not None else None,
                    "successful_fec_decodes": best.successful_fec_decodes if best is not None else None,
                    "failed_fec_decodes": best.failed_fec_decodes if best is not None else None,
                    "diagnostics": dict(trial_search.diagnostics),
                    "best_diagnostics": dict(best.diagnostics) if best is not None else {},
                    "candidates": trial_search.trial_log,
                    "warnings": list(trial_search.warnings),
                }
                if (
                    best is not None
                    and best.fec_type is not None
                    and best.accepted
                    and trial_search.decoded_bits.size
                ):
                    result.visualization["fec_recovered_bits"] = trial_search.decoded_bits
                elif verified_uncoded_crc and trial_search.decoded_bits.size:
                    result.visualization["crc_verified_bits"] = trial_search.decoded_bits
            except (TypeError, ValueError, RuntimeError) as exc:
                result.fec = {
                    "status": "failed",
                    "reason": str(exc),
                    "candidate_count": 0,
                    "soft_llrs_supplied": bool(demod_result.llrs.size),
                }
                result.errors.append({"stage": "fec", "code": "fec_trial_error", "message": str(exc)})
        else:
            result.fec = {"status": "not_requested", "candidate_count": 0}
        result.stages.append({
            "name": "fec_interleaver_trials",
            "status": result.fec.get("status", "unavailable"),
        })
        self._progress(progress_callback, "FEC/interleaver search complete", 82)

        selected_candidate = trial_search.best_result if trial_search is not None else None
        selected_candidate_verified = bool(
            selected_candidate is not None
            and selected_candidate.accepted
            and (
                selected_candidate.fec_type is not None
                or (crc_present and selected_candidate.crc_pass is True)
            )
        )
        frame_bits = (
            trial_search.decoded_bits
            if trial_search is not None
            and selected_candidate_verified
            and trial_search.decoded_bits.size
            else demod_result.bits
        )
        self._progress(progress_callback, "Building analysis result...", 88)
        try:
            frame_result = analyze_frames(
                frame_bits,
                sync_word=sync_word,
                header_length=header_length,
                payload_length=payload_length,
                sync_threshold=sync_threshold,
            )
            sync = frame_result.sync_result
            result.frames = {
                "status": (
                    "correlated" if frame_result.sync_index is not None
                    else "unavailable" if sync_word is None
                    else "not_found"
                ),
                "input_source": (
                    "accepted FEC candidate"
                    if selected_candidate_verified and selected_candidate.fec_type is not None
                    else "CRC-accepted uncoded candidate"
                    if selected_candidate_verified
                    else "demodulator hard decisions"
                ),
                "sync_word_supplied": sync_word is not None,
                "sync_index": frame_result.sync_index,
                "sync_peak_score": sync.peak_score if sync is not None else None,
                "frame_length_estimate_bits": frame_result.frame_length_estimate,
                "header_bit_count": int(frame_result.header_bits.size),
                "payload_bit_count": int(frame_result.payload_bits.size),
                "header_bits": frame_result.header_bits.copy(),
                "payload_bits": frame_result.payload_bits.copy(),
                "warnings": list(frame_result.warnings),
            }
            if sync_word is not None:
                result.visualization["sync_correlation_scores"] = (
                    sync.scores if sync is not None else np.empty(0, dtype=np.float64)
                )
            if frame_result.header_bits.size:
                result.visualization["header_bits"] = frame_result.header_bits
            if frame_result.payload_bits.size:
                result.visualization["frame_payload_bits"] = frame_result.payload_bits
        except (TypeError, ValueError) as exc:
            result.frames = {"status": "failed", "reason": str(exc)}
            result.errors.append({"stage": "frame_analysis", "code": "frame_analysis_error", "message": str(exc)})
        result.stages.append({"name": "frame_and_payload_analysis", "status": result.frames.get("status", "unavailable")})
        self._progress(progress_callback, "Frame analysis complete", 94)

        class_accepted = classification.modulation is not ModulationType.UNKNOWN
        demod_ok = bool(demod_result.bits.size)
        fec_accepted = selected_candidate_verified
        phase_resolution = result.synchronization.get("phase_ambiguity_resolution", {})
        phase_unresolved = (
            isinstance(phase_resolution, dict)
            and phase_resolution.get("status") == "unresolved"
        )
        if (
            class_accepted
            and demod_ok
            and (fec_accepted or not run_fec)
            and not result.errors
            and not phase_unresolved
        ):
            result.status = "complete"
        else:
            result.status = "partial"
        self._progress(progress_callback, "Analysis finished", 100)
        return result


def analyze_file(path: str | Path, **configuration: Any) -> AnalysisResult:
    """Convenience public entry point using the configured production model path."""
    return Analyzer().analyze_file(path, **configuration)
