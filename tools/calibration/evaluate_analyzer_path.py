"""Evaluate frozen-model calibration on independent raw signals via Analyzer.

The script writes only new report artifacts. It never changes model weights,
the embedded scaler, feature/class order, production thresholds, or inference
code. Run ``prepare`` for fit/assessment, ``fit`` to freeze a candidate (or
freeze the no-calibration decision), ``final`` only after that freeze, then
``report``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.common.enums import ModulationType, SampleDtype
from core.common.models import IQConfig
from core.dsp.modulation_features import FEATURE_NAMES
from core.ml.classifier import OnnxModulationClassifier
from core.ml.model_def import DEFAULT_MODEL_CLASS_ORDER, FEATURE_SCHEMA_VERSION
from core.pipeline.analyzer import Analyzer
from libs.modulation_library import constellation_for
from tools.calibration.evaluate_temperature import (
    CLASS_NAMES,
    SNR_BANDS,
    _ece,
    _negative_log_likelihood,
    fit_temperature,
    score_dataset,
    stable_softmax,
)
from tools.tier2.generate_signals import _apply_channel, _make_fsk


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "production_mlp_final.onnx"
CONTRACT_PATH = ROOT / "production_ml_contract.json"
FEATURES_PER_SIGNAL = 2048
FIT_SEED = 2026100601
ASSESSMENT_SEED = 2026100602
FINAL_SEED = 2026100603
SPLIT_SEEDS = {
    "calibration_fit": FIT_SEED,
    "calibration_assessment": ASSESSMENT_SEED,
    "final_calibration_test": FINAL_SEED,
}
CLASS_ENUMS = tuple(DEFAULT_MODEL_CLASS_ORDER)
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SAMPLES_PER_CLASS_DEFAULT = 400
SNR_RANGE_DB = (8.0, 20.0)
SYMBOL_SAMPLE_RATE_HZ = 1_000_000.0
FSK_SAMPLE_RATE_HZ = 48_000.0
FSK_SAMPLES_PER_SYMBOL = 8
FSK_TONES_HZ = (-3_000.0, 3_000.0)
CFO_RANGE_HZ = (-400.0, 400.0)
ACCURACY_GATE_OVERALL = 0.90
ACCURACY_GATE_PER_CLASS = 0.80
ECE_REGRESSION_TOLERANCE = 0.02


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_contract() -> tuple[dict[str, Any], str]:
    contract = _read_json(CONTRACT_PATH)
    model_hash = sha256_file(MODEL_PATH)
    if model_hash != contract["artifact"]["onnx_sha256"]:
        raise RuntimeError("production ONNX SHA-256 does not match the frozen contract")
    if tuple(contract["features"]) != tuple(FEATURE_NAMES):
        raise RuntimeError("production feature order does not match implementation")
    if tuple(contract["class_order"]) != CLASS_NAMES:
        raise RuntimeError("production class order does not match evaluation order")
    if tuple(item.value for item in DEFAULT_MODEL_CLASS_ORDER) != CLASS_NAMES:
        raise RuntimeError("classifier fallback class order differs from frozen contract")
    if contract["input"]["feature_count"] != len(FEATURE_NAMES):
        raise RuntimeError("frozen model feature count differs from feature implementation")
    if contract["inference"]["wrapper_applies_temperature"]:
        raise RuntimeError("production inference unexpectedly applies a temperature")
    if not contract["preprocessing"]["embedded_in_onnx"]:
        raise RuntimeError("the expected embedded scaler is not recorded in the contract")
    return contract, model_hash


class _SessionTap:
    """Observe raw model outputs while delegating every inference unchanged."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self.outputs: list[np.ndarray] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def run(self, output_names: Any, input_feed: Any, run_options: Any = None) -> Any:
        if run_options is None:
            result = self._session.run(output_names, input_feed)
        else:
            result = self._session.run(output_names, input_feed, run_options)
        if result:
            self.outputs.append(np.asarray(result[0], dtype=np.float64).reshape(-1).copy())
        return result


class _RecordingClassifier:
    """Call the frozen production wrapper and retain features/logits per pass."""

    def __init__(self, model_path: Path) -> None:
        self.base = OnnxModulationClassifier(model_path)
        if not self.base.available:
            raise RuntimeError("frozen ONNX model failed to load")
        self.tap = _SessionTap(self.base._session)
        self.base._session = self.tap
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def classify(self, samples: np.ndarray, *, samples_per_symbol: float = 1.0) -> Any:
        output_start = len(self.tap.outputs)
        result = self.base.classify(samples, samples_per_symbol=samples_per_symbol)
        if len(self.tap.outputs) == output_start:
            raise RuntimeError("production classifier did not produce raw ONNX logits")
        logits = self.tap.outputs[-1]
        feature_map = result.features
        try:
            feature_row = np.asarray([feature_map[name] for name in FEATURE_NAMES], dtype=np.float64)
        except (KeyError, TypeError) as exc:
            raise RuntimeError("production classifier result omitted required feature values") from exc
        if logits.shape != (len(CLASS_NAMES),) or feature_row.shape != (len(FEATURE_NAMES),):
            raise RuntimeError("unexpected production classifier output dimensions")
        self.calls.append({
            "features": feature_row,
            "logits": logits,
            "classifier_label": result.modulation.value,
            "classifier_confidence": float(result.confidence),
            "classifier_status": result.status.value,
        })
        return result


def _seeded_rng(split_seed: int, class_index: int, sample_index: int) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([split_seed, class_index, sample_index]))
    )


def _make_raw_signal(class_index: int, rng: np.random.Generator, snr_db: float) -> tuple[np.ndarray, dict[str, Any]]:
    modulation = CLASS_ENUMS[class_index]
    phase = float(rng.uniform(-np.pi, np.pi))
    gain = float(rng.uniform(0.65, 1.45))
    if modulation is ModulationType.FSK2:
        states = rng.integers(0, 2, FEATURES_PER_SIGNAL, dtype=np.uint8)
        waveform = _make_fsk(
            states, FSK_SAMPLE_RATE_HZ, FSK_SAMPLES_PER_SYMBOL, FSK_TONES_HZ
        )
        sample_rate = FSK_SAMPLE_RATE_HZ
        samples_per_symbol = FSK_SAMPLES_PER_SYMBOL
        cfo_hz = 0.0
        pulse_shaping = "continuous-phase rectangular 2FSK symbol intervals"
    else:
        constellation = constellation_for(modulation)
        symbol_indices = rng.integers(0, constellation.size, FEATURES_PER_SIGNAL)
        waveform = constellation[symbol_indices]
        sample_rate = SYMBOL_SAMPLE_RATE_HZ
        samples_per_symbol = None
        cfo_hz = float(rng.uniform(*CFO_RANGE_HZ))
        pulse_shaping = "none; symbol-rate samples, matching the documented PSK/QAM classifier input contract"
    received = _apply_channel(
        waveform,
        sample_rate=sample_rate,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        phase_rad=phase,
        gain=gain,
        rng=rng,
    ).astype(np.complex64, copy=False)
    metadata = {
        "class_name": CLASS_NAMES[class_index],
        "sample_rate_hz": sample_rate,
        "samples_per_symbol": samples_per_symbol,
        "waveform_samples_per_symbol": (
            FSK_SAMPLES_PER_SYMBOL if modulation is ModulationType.FSK2 else 1
        ),
        "symbol_or_state_count": FEATURES_PER_SIGNAL,
        "sample_count": int(received.size),
        "snr_db": float(snr_db),
        "cfo_hz": cfo_hz,
        "phase_rad": phase,
        "gain": gain,
        "pulse_shaping": pulse_shaping,
        "matched_filter_rolloff": None,
        "timing_recovery_requested": False,
        "timing_recovery_applied": False,
        "raw_format": "complex64 native-endian .iq",
    }
    return received, metadata


def _class_index(label: Any) -> int:
    return CLASS_INDEX.get(str(label).lower(), -1)


def _analyze_one(
    analyzer: Analyzer,
    recorder: _RecordingClassifier,
    raw_path: Path,
    signal: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    signal.tofile(raw_path)
    recorder.calls.clear()
    config = IQConfig(dtype=SampleDtype.COMPLEX64, sample_rate=metadata["sample_rate_hz"])
    analysis = analyzer.analyze_file(
        raw_path,
        iq_config=config,
        samples_per_symbol=metadata["samples_per_symbol"],
        recover_symbol_timing=False,
        matched_filter_rolloff=None,
        run_fec=False,
    )
    if not recorder.calls:
        raise RuntimeError(f"Analyzer did not call production ONNX classifier: {analysis.errors}")
    if analysis.file_format != "iq" or not analysis.features.get("names"):
        raise RuntimeError(f"Analyzer failed to load/extract production IQ features: {analysis.errors}")
    if tuple(analysis.features["names"]) != tuple(FEATURE_NAMES):
        raise RuntimeError("Analyzer-reported feature schema differs from frozen contract")
    call_count = len(recorder.calls)
    if call_count > 2:
        raise RuntimeError(f"unexpected analyzer classifier call count {call_count}")
    cfo = analysis.synchronization.get("classifier_cfo_correction", {})
    selected_index = 1 if cfo.get("status") == "applied" and call_count == 2 else 0
    selected = recorder.calls[selected_index]
    class_result = analysis.classification
    analyzer_label = str(class_result.get("modulation", "unknown")).lower()
    gate = class_result.get("signal_detection_gate", {})
    return {
        "features": selected["features"],
        "logits": selected["logits"],
        "all_call_features": [call["features"] for call in recorder.calls],
        "all_call_logits": [call["logits"] for call in recorder.calls],
        "all_call_labels": [call["classifier_label"] for call in recorder.calls],
        "all_call_confidence": [call["classifier_confidence"] for call in recorder.calls],
        "all_call_status": [call["classifier_status"] for call in recorder.calls],
        "call_count": call_count,
        "selected_call_index": selected_index,
        "selected_classifier_confidence": selected["classifier_confidence"],
        "selected_classifier_label": selected["classifier_label"],
        "analyzer_label": analyzer_label,
        "analyzer_confidence": class_result.get("confidence"),
        "signal_gate_decision": str(gate.get("decision", "unavailable")),
        "signal_gate_score": gate.get("geometric_fit_score"),
        "cfo_status": str(cfo.get("status", "not_applied")),
        "cfo_estimate_hz": cfo.get("estimate_hz"),
        "analysis_status": analysis.status,
        "analysis_errors": analysis.errors,
    }


def _summarize_split(
    split_name: str,
    split_seed: int,
    samples_per_class: int,
    output_path: Path,
    seen_hashes: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    recorder = _RecordingClassifier(MODEL_PATH)
    analyzer = Analyzer(classifier=recorder)
    feature_rows: list[np.ndarray] = []
    selected_logits: list[np.ndarray] = []
    all_call_features: list[np.ndarray] = []
    all_call_logits: list[np.ndarray] = []
    all_call_labels: list[list[int]] = []
    all_call_confidence: list[list[float]] = []
    call_counts: list[int] = []
    selected_calls: list[int] = []
    analyzer_labels: list[int] = []
    selected_classifier_labels: list[int] = []
    selected_confidence: list[float] = []
    analyzer_confidence: list[float] = []
    gate_decisions: list[str] = []
    gate_scores: list[float] = []
    cfo_statuses: list[str] = []
    cfo_estimates: list[float] = []
    class_labels: list[int] = []
    snr_values: list[float] = []
    cfo_values: list[float] = []
    sps_values: list[float] = []
    sample_counts: list[int] = []
    phase_values: list[float] = []
    gain_values: list[float] = []
    waveform_hashes: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    local_hashes: set[str] = set()

    with tempfile.TemporaryDirectory(prefix="ntro-analyzer-raw-cal-") as temp_dir:
        raw_path = Path(temp_dir) / "current_signal.iq"
        total = len(CLASS_NAMES) * samples_per_class
        completed = 0
        for class_index, class_name in enumerate(CLASS_NAMES):
            for sample_index in range(samples_per_class):
                rng = _seeded_rng(split_seed, class_index, sample_index)
                snr_db = float(rng.uniform(*SNR_RANGE_DB))
                signal, meta = _make_raw_signal(class_index, rng, snr_db)
                digest = hashlib.sha256(signal.tobytes()).hexdigest()
                if digest in local_hashes or digest in seen_hashes:
                    raise RuntimeError(f"duplicate raw waveform detected in {split_name}: {digest}")
                local_hashes.add(digest)
                meta.update({
                    "split": split_name,
                    "split_seed": split_seed,
                    "class_index": class_index,
                    "sample_index": sample_index,
                    "seed_components": [split_seed, class_index, sample_index],
                    "raw_waveform_sha256": digest,
                })
                analysis = _analyze_one(analyzer, recorder, raw_path, signal, meta)
                meta.update({
                    key: analysis[key]
                    for key in (
                        "call_count", "selected_call_index", "selected_classifier_label",
                        "selected_classifier_confidence", "analyzer_label", "analyzer_confidence",
                        "signal_gate_decision", "signal_gate_score", "cfo_status", "cfo_estimate_hz",
                        "analysis_status", "analysis_errors",
                    )
                })
                feature_rows.append(analysis["features"])
                selected_logits.append(analysis["logits"])
                # Zero-padding is disambiguated by call_count and keeps persisted
                # numeric arrays finite for downstream checks.
                padded_features = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float64)
                padded_logits = np.zeros((2, len(CLASS_NAMES)), dtype=np.float64)
                padded_call_labels = [-1, -1]
                padded_call_confidence = [0.0, 0.0]
                for call_index, (features, logits) in enumerate(zip(
                    analysis["all_call_features"], analysis["all_call_logits"]
                )):
                    padded_features[call_index] = features
                    padded_logits[call_index] = logits
                    padded_call_labels[call_index] = _class_index(analysis["all_call_labels"][call_index])
                    padded_call_confidence[call_index] = analysis["all_call_confidence"][call_index]
                all_call_features.append(padded_features)
                all_call_logits.append(padded_logits)
                all_call_labels.append(padded_call_labels)
                all_call_confidence.append(padded_call_confidence)
                call_counts.append(analysis["call_count"])
                selected_calls.append(analysis["selected_call_index"])
                analyzer_labels.append(_class_index(analysis["analyzer_label"]))
                selected_classifier_labels.append(_class_index(analysis["selected_classifier_label"]))
                selected_confidence.append(analysis["selected_classifier_confidence"])
                analyzer_confidence.append(
                    float(analysis["analyzer_confidence"])
                    if analysis["analyzer_confidence"] is not None else float("nan")
                )
                gate_decisions.append(analysis["signal_gate_decision"])
                gate_scores.append(
                    float(analysis["signal_gate_score"])
                    if analysis["signal_gate_score"] is not None else float("nan")
                )
                cfo_statuses.append(analysis["cfo_status"])
                cfo_estimates.append(
                    float(analysis["cfo_estimate_hz"])
                    if analysis["cfo_estimate_hz"] is not None else float("nan")
                )
                class_labels.append(class_index)
                snr_values.append(snr_db)
                cfo_values.append(meta["cfo_hz"])
                sps_values.append(meta["waveform_samples_per_symbol"])
                sample_counts.append(meta["sample_count"])
                phase_values.append(meta["phase_rad"])
                gain_values.append(meta["gain"])
                waveform_hashes.append(digest)
                metadata_rows.append(meta)
                seen_hashes.add(digest)
                completed += 1
                if completed % 100 == 0 or completed == total:
                    print(f"{split_name}: {completed}/{total} raw signals passed through Analyzer", flush=True)

    arrays: dict[str, np.ndarray] = {
        "features": np.asarray(feature_rows, dtype=np.float64),
        "logits": np.asarray(selected_logits, dtype=np.float64),
        "all_call_features": np.asarray(all_call_features, dtype=np.float64),
        "all_call_logits": np.asarray(all_call_logits, dtype=np.float64),
        "all_call_labels": np.asarray(all_call_labels, dtype=np.int64),
        "all_call_confidence": np.asarray(all_call_confidence, dtype=np.float64),
        "call_count": np.asarray(call_counts, dtype=np.int64),
        "selected_call_index": np.asarray(selected_calls, dtype=np.int64),
        "selected_classifier_labels": np.asarray(selected_classifier_labels, dtype=np.int64),
        "selected_classifier_confidence": np.asarray(selected_confidence, dtype=np.float64),
        "analyzer_labels": np.asarray(analyzer_labels, dtype=np.int64),
        "analyzer_confidence": np.asarray(analyzer_confidence, dtype=np.float64),
        "signal_gate_decision": np.asarray(gate_decisions, dtype="U32"),
        "signal_gate_score": np.asarray(gate_scores, dtype=np.float64),
        "cfo_status": np.asarray(cfo_statuses, dtype="U32"),
        "cfo_estimate_hz": np.asarray(cfo_estimates, dtype=np.float64),
        "labels": np.asarray(class_labels, dtype=np.int64),
        "snr_db": np.asarray(snr_values, dtype=np.float64),
        "cfo_hz": np.asarray(cfo_values, dtype=np.float64),
        "samples_per_symbol": np.asarray(sps_values, dtype=np.float64),
        "sample_count": np.asarray(sample_counts, dtype=np.int64),
        "phase_rad": np.asarray(phase_values, dtype=np.float64),
        "gain": np.asarray(gain_values, dtype=np.float64),
        "waveform_sha256": np.asarray(waveform_hashes, dtype="U64"),
    }
    for key, values in arrays.items():
        if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
            # NaNs are only intentional padding for absent call slots and undefined
            # confidence/gate/CFO metadata; primary feature/logit arrays must be finite.
            if key in {"features", "logits", "all_call_features", "all_call_logits"}:
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(f"non-finite values in primary evaluation array {key}")
    np.savez_compressed(output_path, **arrays)
    artifact_hash = sha256_file(output_path)
    class_counts = {name: int(np.sum(arrays["labels"] == index)) for index, name in enumerate(CLASS_NAMES)}
    info = {
        "split": split_name,
        "seed": split_seed,
        "samples_per_class": samples_per_class,
        "sample_count": int(arrays["labels"].size),
        "class_counts": class_counts,
        "unique_raw_waveform_count": len(local_hashes),
        "artifact": output_path.name,
        "artifact_sha256": artifact_hash,
        "raw_waveform_sha256_list": waveform_hashes,
        "generation": {
            "method": "deterministic random symbols/states + existing independent Tier-2 AWGN/channel and FSK waveform helpers",
            "seed_rule": "PCG64(SeedSequence([split_seed, class_index, sample_index]))",
            "symbols_or_states_per_signal": FEATURES_PER_SIGNAL,
            "snr_range_db": list(SNR_RANGE_DB),
            "psk_qam_cfo_range_hz": list(CFO_RANGE_HZ),
            "2fsk_cfo_hz": 0.0,
            "2fsk_tones_hz": list(FSK_TONES_HZ),
            "2fsk_samples_per_symbol": FSK_SAMPLES_PER_SYMBOL,
            "psk_qam_representation": "symbol-rate complex64; no matched filter/timing recovery/pulse shaping because production contract expects synchronized symbol-rate samples",
            "fsk_representation": "continuous-phase 8-sample/symbol complex64 waveform",
            "raw_format": "native complex64 .iq; loaded by Analyzer.analyze_file with explicit IQConfig",
            "phase_and_gain": "independent uniform phase [-pi, pi], gain [0.65, 1.45]",
            "signal_normalization": "none before file load; project feature extractor applies its existing RMS normalization",
        },
        "samples_per_symbol_observed": {
            "minimum": float(np.min(arrays["samples_per_symbol"])),
            "maximum": float(np.max(arrays["samples_per_symbol"])),
        },
        "snr_db_observed": [float(np.min(arrays["snr_db"])), float(np.max(arrays["snr_db"]))],
        "cfo_hz_observed": [float(np.min(arrays["cfo_hz"])), float(np.max(arrays["cfo_hz"]))],
        "cfo_retry_counts": {
            key: int(np.sum(arrays["cfo_status"] == key))
            for key in sorted(set(cfo_statuses))
        },
        "analyzer_gate_counts": {
            key: int(np.sum(arrays["signal_gate_decision"] == key))
            for key in sorted(set(gate_decisions))
        },
    }
    return arrays, info


def _metrics_by_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    counts = np.bincount(truth, minlength=len(CLASS_NAMES))
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES) + 1), dtype=np.int64)
    unknown_column = len(CLASS_NAMES)
    for actual, estimate in zip(truth, predicted):
        confusion[actual, estimate if 0 <= estimate < len(CLASS_NAMES) else unknown_column] += 1
    per_class: dict[str, Any] = {}
    f1_values = []
    for class_index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[class_index, class_index])
        fp = int(np.sum(confusion[:, class_index]) - tp)
        fn = int(counts[class_index] - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / counts[class_index] if counts[class_index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "count": int(counts[class_index]),
            "accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "sample_count": int(truth.size),
        "accuracy": float(np.mean(truth == predicted)) if truth.size else None,
        "macro_precision": float(np.mean([row["precision"] for row in per_class.values()])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class.values()])),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_columns": list(CLASS_NAMES) + ["unknown/rejected"],
    }


def _baseline_split_metrics(data: dict[str, np.ndarray]) -> dict[str, Any]:
    labels = data["labels"]
    model_predictions = np.argmax(data["logits"], axis=1)
    analyzer_predictions = data["analyzer_labels"]
    result = {
        "sample_count": int(labels.size),
        "class_counts": {name: int(np.sum(labels == index)) for index, name in enumerate(CLASS_NAMES)},
        "selected_model_invocation": _metrics_by_predictions(labels, model_predictions),
        "analyzer_end_to_end": _metrics_by_predictions(labels, analyzer_predictions),
        "mean_raw_top_class_probability": float(np.mean(np.max(stable_softmax(data["logits"]), axis=1))),
        "mean_margin_weighted_classifier_confidence": float(np.nanmean(data["selected_classifier_confidence"])),
        "analyzer_unknown_or_rejected_count": int(np.sum(analyzer_predictions < 0)),
        "mean_classifier_calls_per_signal": float(np.mean(data["call_count"])),
        "mean_cfo_retry_applied": bool(np.any(data["cfo_status"] == "applied")),
        "snr_bands": {},
    }
    for low, high in SNR_BANDS:
        mask = (data["snr_db"] >= low) & (
            (data["snr_db"] < high) if high < SNR_RANGE_DB[1] else (data["snr_db"] <= high)
        )
        key = f"{low:g}-{high:g} dB"
        result["snr_bands"][key] = {
            "sample_count": int(np.sum(mask)),
            "selected_model_invocation": _metrics_by_predictions(labels[mask], model_predictions[mask]),
            "analyzer_end_to_end": _metrics_by_predictions(labels[mask], analyzer_predictions[mask]),
        }
    return result


def _feature_distribution(data: dict[str, np.ndarray], contract: dict[str, Any]) -> dict[str, Any]:
    observed = np.asarray(data["features"], dtype=np.float64)
    scaler_mean = np.asarray(contract["preprocessing"]["mean"], dtype=np.float64)
    scaler_scale = np.asarray(contract["preprocessing"]["scale"], dtype=np.float64)
    summary: dict[str, Any] = {
        "scaler_statistics_source": "aggregate StandardScaler mean/scale in frozen production_ml_contract.json; raw training rows and class-conditional training feature distributions were unavailable",
        "overall": {},
        "by_class": {},
        "class_pair_standardized_mean_separation": {},
        "features_over_two_training_scales_from_aggregate_mean": [],
    }
    for index, name in enumerate(FEATURE_NAMES):
        values = observed[:, index]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        shift = (mean - float(scaler_mean[index])) / float(scaler_scale[index])
        summary["overall"][name] = {
            "production_path_mean": mean,
            "production_path_std": std,
            "production_path_min": float(np.min(values)),
            "production_path_p05": float(np.quantile(values, 0.05)),
            "production_path_p95": float(np.quantile(values, 0.95)),
            "production_path_max": float(np.max(values)),
            "training_scaler_aggregate_mean": float(scaler_mean[index]),
            "training_scaler_aggregate_scale": float(scaler_scale[index]),
            "mean_shift_in_training_scales": float(shift),
            "observed_std_over_training_scale": float(std / scaler_scale[index]),
        }
        if abs(shift) > 2.0:
            summary["features_over_two_training_scales_from_aggregate_mean"].append(name)
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_rows = observed[data["labels"] == class_index]
        if class_rows.shape[0] == 0:
            summary["by_class"][class_name] = {
                name: {
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "p05": None,
                    "p95": None,
                    "max": None,
                    "mean_shift_in_aggregate_training_scales": None,
                }
                for name in FEATURE_NAMES
            }
            continue
        summary["by_class"][class_name] = {
            name: {
                "count": int(class_rows.shape[0]),
                "mean": float(np.mean(class_rows[:, feature_index])),
                "std": float(np.std(class_rows[:, feature_index], ddof=1)),
                "min": float(np.min(class_rows[:, feature_index])),
                "p05": float(np.quantile(class_rows[:, feature_index], 0.05)),
                "p95": float(np.quantile(class_rows[:, feature_index], 0.95)),
                "max": float(np.max(class_rows[:, feature_index])),
                "mean_shift_in_aggregate_training_scales": float(
                    (np.mean(class_rows[:, feature_index]) - scaler_mean[feature_index])
                    / scaler_scale[feature_index]
                ),
            }
            for feature_index, name in enumerate(FEATURE_NAMES)
        }
    for left, right in (("bpsk", "qpsk"), ("bpsk", "8psk"), ("qpsk", "8psk")):
        i, j = CLASS_INDEX[left], CLASS_INDEX[right]
        left_rows = observed[data["labels"] == i]
        right_rows = observed[data["labels"] == j]
        pooled = np.sqrt((np.var(left_rows, axis=0, ddof=1) + np.var(right_rows, axis=0, ddof=1)) / 2.0)
        d = (np.mean(left_rows, axis=0) - np.mean(right_rows, axis=0)) / np.maximum(pooled, 1e-12)
        summary["class_pair_standardized_mean_separation"][f"{left}_vs_{right}"] = {
            name: float(value) for name, value in zip(FEATURE_NAMES, d)
        }
    return summary


def _combine_feature_data(*datasets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "features": np.concatenate([np.asarray(item["features"]) for item in datasets], axis=0),
        "labels": np.concatenate([np.asarray(item["labels"]) for item in datasets], axis=0),
    }


def _combine_raw_datasets(*datasets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([np.asarray(item[key]) for item in datasets], axis=0)
        for key in datasets[0]
    }


def _path_diagnostics(data: dict[str, np.ndarray]) -> dict[str, Any]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    first_raw = np.argmax(data["all_call_logits"][:, 0, :], axis=1)
    selected_raw = np.argmax(data["logits"], axis=1)
    first_classification = data["all_call_labels"][:, 0]
    retry_classification = data["all_call_labels"][:, 1]
    call_count = data["call_count"]
    applied = data["cfo_status"] == "applied"
    output: dict[str, Any] = {
        "per_class": {},
        "bpsk_abs_cfo_bands": {},
        "interpretation": "First-pass predictions use the first ONNX invocation. Retry outputs include every second production classifier call. Analyzer outcomes include the existing confidence selection and signal-support gate.",
    }
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_mask = labels == class_index
        retry_mask = class_mask & (call_count == 2)
        rejected_retry_mask = retry_mask & ~applied
        output["per_class"][class_name] = {
            "sample_count": int(np.sum(class_mask)),
            "first_pass_raw_argmax_accuracy": float(np.mean(first_raw[class_mask] == class_index)),
            "first_pass_classifier_accepted_recall": float(np.mean(first_classification[class_mask] == class_index)),
            "second_classifier_call_count": int(np.sum(retry_mask)),
            "cfo_retry_applied_count": int(np.sum(class_mask & applied)),
            "retry_not_selected_count": int(np.sum(rejected_retry_mask)),
            "retry_not_selected_but_second_pass_label_correct": int(
                np.sum(rejected_retry_mask & (retry_classification == class_index))
            ),
            "selected_model_raw_argmax_accuracy": float(np.mean(selected_raw[class_mask] == class_index)),
            "analyzer_end_to_end_recall": float(np.mean(data["analyzer_labels"][class_mask] == class_index)),
            "analyzer_output_counts": {
                (CLASS_NAMES[i] if i >= 0 else "unknown/rejected"): int(
                    np.sum(data["analyzer_labels"][class_mask] == i)
                )
                for i in [-1, *range(len(CLASS_NAMES))]
                if np.any(data["analyzer_labels"][class_mask] == i)
            },
            "first_raw_argmax_counts": {
                CLASS_NAMES[i]: int(np.sum(first_raw[class_mask] == i))
                for i in range(len(CLASS_NAMES))
                if np.any(first_raw[class_mask] == i)
            },
        }
    for low, high in ((0.0, 100.0), (100.0, 200.0), (200.0, 300.0), (300.0, 400.000001)):
        mask = (
            (labels == CLASS_INDEX["bpsk"])
            & (np.abs(data["cfo_hz"]) >= low)
            & (np.abs(data["cfo_hz"]) < high)
        )
        output["bpsk_abs_cfo_bands"][f"{low:g}-{min(high, 400.0):g} Hz"] = {
            "sample_count": int(np.sum(mask)),
            "selected_model_raw_argmax_accuracy": float(np.mean(selected_raw[mask] == 0)) if np.any(mask) else None,
            "analyzer_end_to_end_accuracy": float(np.mean(data["analyzer_labels"][mask] == 0)) if np.any(mask) else None,
            "cfo_retry_applied_fraction": float(np.mean(applied[mask])) if np.any(mask) else None,
        }
    rejected_bpsk = (
        (labels == CLASS_INDEX["bpsk"])
        & (call_count == 2)
        & ~applied
    )
    if np.any(rejected_bpsk):
        gaps = data["all_call_confidence"][rejected_bpsk, 0] - data["all_call_confidence"][rejected_bpsk, 1]
        output["bpsk_rejected_retry_detail"] = {
            "retry_count": int(np.sum(rejected_bpsk)),
            "second_pass_returned_bpsk": int(np.sum(rejected_bpsk & (retry_classification == CLASS_INDEX["bpsk"]))),
            "mean_first_confidence": float(np.mean(data["all_call_confidence"][rejected_bpsk, 0])),
            "mean_second_confidence": float(np.mean(data["all_call_confidence"][rejected_bpsk, 1])),
            "mean_first_minus_second_confidence": float(np.mean(gaps)),
            "note": "Analyzer retains the first result unless corrected confidence is at least the initial confidence; no threshold was changed.",
        }
    return output


def _verify_onnx_parity(
    root: Path, model_hash: str, datasets: list[dict[str, np.ndarray]]
) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    compared = 0
    mismatches = 0
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for data in datasets:
        counts = np.asarray(data["call_count"], dtype=np.int64)
        example_rows, call_slots = np.where(np.arange(2)[None, :] < counts[:, None])
        features = np.asarray(data["all_call_features"][example_rows, call_slots], dtype=np.float32)
        expected = np.asarray(data["all_call_logits"][example_rows, call_slots], dtype=np.float64)
        for start in range(0, features.shape[0], 512):
            end = min(start + 512, features.shape[0])
            actual = np.asarray(
                session.run(["logits"], {model_input.name: features[start:end]})[0],
                dtype=np.float64,
            )
            target = expected[start:end]
            difference = np.abs(actual - target)
            max_abs_diff = max(max_abs_diff, float(np.max(difference)))
            denominator = np.maximum(np.abs(target), np.finfo(np.float64).tiny)
            max_rel_diff = max(max_rel_diff, float(np.max(difference / denominator)))
            mismatches += int(np.sum(~np.isclose(actual, target, rtol=0.0, atol=0.0)))
            compared += int(target.shape[0])
    return {
        "model_sha256": model_hash,
        "execution_provider": "CPUExecutionProvider",
        "compared_production_classifier_calls": compared,
        "exact_float32_logit_match_count": compared - mismatches,
        "mismatch_count": mismatches,
        "max_absolute_logit_difference": max_abs_diff,
        "max_relative_logit_difference": max_rel_diff,
        "all_calls_match_bitwise": mismatches == 0,
        "feature_values_cast_to_float32_as_in_production_wrapper": True,
        "external_scaling_or_temperature_applied": False,
    }


def _write_sample_metadata(
    root: Path,
    manifest: dict[str, Any],
    datasets: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    metadata_root = root / "sample_metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {}
    for split_name, data in datasets.items():
        split_seed = int(manifest["seeds"][split_name])
        class_offsets = {index: 0 for index in range(len(CLASS_NAMES))}
        path = metadata_root / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row_index, class_index_raw in enumerate(data["labels"]):
                class_index = int(class_index_raw)
                sample_index = class_offsets[class_index]
                class_offsets[class_index] += 1
                rng = _seeded_rng(split_seed, class_index, sample_index)
                snr_db = float(rng.uniform(*SNR_RANGE_DB))
                signal, generation = _make_raw_signal(class_index, rng, snr_db)
                waveform_hash = hashlib.sha256(signal.tobytes()).hexdigest()
                recorded_hash = str(data["waveform_sha256"][row_index])
                if waveform_hash != recorded_hash:
                    raise RuntimeError(
                        f"seeded waveform regeneration mismatch for {split_name} row {row_index}"
                    )
                if snr_db != float(data["snr_db"][row_index]):
                    raise RuntimeError(f"seeded SNR regeneration mismatch for {split_name} row {row_index}")
                selected_label_index = int(data["selected_classifier_labels"][row_index])
                analyzer_label_index = int(data["analyzer_labels"][row_index])
                item = {
                    **generation,
                    "split": split_name,
                    "seed": split_seed,
                    "seed_components": [split_seed, class_index, sample_index],
                    "class_index": class_index,
                    "sample_index": sample_index,
                    "model_sha256": manifest["model_sha256"],
                    "feature_schema_version": manifest["feature_schema_version"],
                    "feature_order": manifest["feature_order"],
                    "class_order": manifest["class_order"],
                    "raw_waveform_sha256": waveform_hash,
                    "matched_filter_applied": False,
                    "analyzer_classifier_call_count": int(data["call_count"][row_index]),
                    "selected_call_index": int(data["selected_call_index"][row_index]),
                    "selected_classifier_label": (
                        CLASS_NAMES[selected_label_index] if selected_label_index >= 0 else "unknown"
                    ),
                    "analyzer_final_label": (
                        CLASS_NAMES[analyzer_label_index] if analyzer_label_index >= 0 else "unknown/rejected"
                    ),
                    "signal_gate_decision": str(data["signal_gate_decision"][row_index]),
                    "cfo_retry_status": str(data["cfo_status"][row_index]),
                    "cfo_retry_estimate_hz": (
                        float(data["cfo_estimate_hz"][row_index])
                        if np.isfinite(data["cfo_estimate_hz"][row_index]) else None
                    ),
                }
                stream.write(json.dumps(item, allow_nan=False) + "\n")
        output[split_name] = {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "record_count": int(sum(class_offsets.values())),
            "sha256": sha256_file(path),
            "waveforms_regenerated_and_verified": True,
        }
    return output


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _load_split(root: Path, name: str, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    info = manifest["splits"].get(name)
    if info is None:
        raise RuntimeError(f"split is not present in manifest: {name}")
    artifact = root / info["artifact"]
    if sha256_file(artifact) != info["artifact_sha256"]:
        raise RuntimeError(f"artifact hash mismatch for {name}")
    with np.load(artifact, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _prepare(output: Path, samples_per_class: int) -> None:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    contract, model_hash = _check_contract()
    seen_hashes: set[str] = set()
    manifest: dict[str, Any] = {
        "model_sha256": model_hash,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_order": list(FEATURE_NAMES),
        "class_order": list(CLASS_NAMES),
        "dataset_type": "independent synthetic random-symbol raw IQ; no payloads or stored training/validation/Tier-2 samples",
        "samples_per_class": samples_per_class,
        "scaler_embedded_in_onnx": True,
        "seeds": {name: int(seed) for name, seed in SPLIT_SEEDS.items()},
        "splits": {},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "workflow": {
            "final_split_generated_or_evaluated_before_fit_freeze": False,
            "temperature_fit_input": "selected production-path ONNX raw logits from calibration_fit only",
            "final_set_is_only_generated_after_fit_or_no-fit decision is frozen": True,
        },
    }
    # Preserve the actual contract scaler for transparent distribution comparisons.
    manifest["training_scaler_aggregate"] = {
        "source": "production_ml_contract.json",
        "mean": contract["preprocessing"]["mean"],
        "scale": contract["preprocessing"]["scale"],
        "class_conditional_training_rows_available": False,
    }
    for split_name in ("calibration_fit", "calibration_assessment"):
        artifact = output / f"{split_name}.npz"
        data, info = _summarize_split(
            split_name, SPLIT_SEEDS[split_name], samples_per_class, artifact, seen_hashes
        )
        info["baseline_metrics"] = _baseline_split_metrics(data)
        manifest["splits"][split_name] = info
        _write_json(output / "dataset_manifest.json", manifest)
        _write_json(output / f"{split_name}_baseline_metrics.json", info["baseline_metrics"])
    manifest["duplicate_waveform_check"] = {
        "checked_across_all_evaluated_splits": True,
        "unique_waveform_count": len(seen_hashes),
        "duplicates_found": 0,
    }
    fit_data = _load_split(output, "calibration_fit", manifest)
    assessment_data = _load_split(output, "calibration_assessment", manifest)
    manifest["feature_distribution"] = _feature_distribution(
        _combine_feature_data(fit_data, assessment_data), contract
    )
    _write_json(output / "dataset_manifest.json", manifest)
    _write_json(output / "feature_distribution.json", manifest["feature_distribution"])
    baseline = {
        name: manifest["splits"][name]["baseline_metrics"]
        for name in ("calibration_fit", "calibration_assessment")
    }
    _write_json(output / "baseline_metrics.json", baseline)
    print(f"Prepared fit and assessment only; unique raw IQ waveforms={len(seen_hashes)}; final split remains unopened.", flush=True)


def _calibration_quality(metrics: dict[str, Any]) -> dict[str, Any]:
    aggregate = {
        metric: metrics[f"{metric}_after"] < metrics[f"{metric}_before"]
        for metric in ("nll", "ece", "brier")
    }
    per_class = {
        name: {
            "ece_before": values["ece_one_vs_rest_before"],
            "ece_after": values["ece_one_vs_rest_after"],
            "no_material_regression": (
                values["ece_one_vs_rest_after"]
                <= values["ece_one_vs_rest_before"] + ECE_REGRESSION_TOLERANCE
            ),
        }
        for name, values in metrics["per_class"].items()
    }
    snr_bands = {
        name: {
            "ece_before": values.get("ece_before"),
            "ece_after": values.get("ece_after"),
            "no_material_regression": (
                values.get("sample_count", 0) == 0
                or values["ece_after"] <= values["ece_before"] + ECE_REGRESSION_TOLERANCE
            ),
        }
        for name, values in metrics["snr_bands"].items()
    }
    passed = (
        all(aggregate.values())
        and all(row["no_material_regression"] for row in per_class.values())
        and all(row["no_material_regression"] for row in snr_bands.values())
    )
    return {
        "aggregate_nll_ece_brier_all_improved": all(aggregate.values()),
        "aggregate_metrics": aggregate,
        "per_class": per_class,
        "snr_bands": snr_bands,
        "no_class_or_snr_ece_regression_over_tolerance": (
            all(row["no_material_regression"] for row in per_class.values())
            and all(row["no_material_regression"] for row in snr_bands.values())
        ),
        "candidate_quality_passed": passed,
        "ece_regression_tolerance": ECE_REGRESSION_TOLERANCE,
    }


def _fit(root: Path) -> None:
    manifest = _read_json(root / "dataset_manifest.json")
    contract, model_hash = _check_contract()
    if manifest["model_sha256"] != model_hash:
        raise RuntimeError("dataset was generated for a different frozen ONNX model")
    fit = _load_split(root, "calibration_fit", manifest)
    assessment = _load_split(root, "calibration_assessment", manifest)
    if (root / "final_calibration_test.npz").exists():
        raise RuntimeError("final split exists before calibration freeze; refusing to fit")
    baseline = _read_json(root / "baseline_metrics.json")
    accuracy_checks: dict[str, Any] = {}
    eligible = True
    for name in ("calibration_fit", "calibration_assessment"):
        metrics = baseline[name]["analyzer_end_to_end"]
        class_recall = {key: value["recall"] for key, value in metrics["per_class"].items()}
        split_pass = (
            metrics["accuracy"] >= ACCURACY_GATE_OVERALL
            and all(value >= ACCURACY_GATE_PER_CLASS for value in class_recall.values())
        )
        accuracy_checks[name] = {
            "overall_accuracy": metrics["accuracy"],
            "per_class_recall": class_recall,
            "passes_predeclared_calibration_eligibility": split_pass,
        }
        eligible = eligible and split_pass
    decision: dict[str, Any] = {
        "model_sha256": model_hash,
        "accuracy_eligibility_thresholds": {
            "overall_analyzer_accuracy_minimum": ACCURACY_GATE_OVERALL,
            "each_class_recall_minimum": ACCURACY_GATE_PER_CLASS,
        },
        "accuracy_checks": accuracy_checks,
        "calibration_candidate_fitted": False,
        "candidate_deployable_after_assessment": False,
    }
    if not eligible:
        decision.update({
            "decision": "case_c_accuracy_not_healthy_for_probability_calibration",
            "reason": "A predeclared calibration eligibility gate failed on fit or assessment Analyzer end-to-end accuracy; no temperature was fitted.",
            "temperature": None,
            "final_temperature_for_metrics": None,
        })
        _write_json(root / "calibration_decision.json", decision)
        print("Accuracy gate failed; calibration was not fitted. The no-fit decision is frozen before final-set generation.", flush=True)
        return
    temperature, optimizer = fit_temperature(fit["logits"], fit["labels"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("temperature optimizer did not return a positive finite value")
    fit_metrics = score_dataset(fit, temperature)
    assessment_metrics = score_dataset(assessment, temperature)
    quality = _calibration_quality(assessment_metrics)
    # The candidate T is frozen here; the final dataset has not been generated/read.
    frozen = {
        "temperature": temperature,
        "method": "temperature scaling: stable softmax(selected production-path logits / T)",
        "fitted_on": "calibration_fit selected logits only",
        "assessment_used_for_fit_or_selection": False,
        "final_test_generated_or_used_for_fit_or_selection": False,
        "model_sha256": model_hash,
        "fit_artifact_sha256": manifest["splits"]["calibration_fit"]["artifact_sha256"],
        "fit_nll_before": fit_metrics["nll_before"],
        "fit_nll_after": fit_metrics["nll_after"],
        "optimizer": optimizer,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(root / "frozen_temperature.json", frozen)
    decision.update({
        "decision": "candidate_frozen_for_independent_assessment_and_final_evaluation",
        "calibration_candidate_fitted": True,
        "temperature": temperature,
        "final_temperature_for_metrics": temperature,
        "optimizer": optimizer,
        "fit_metrics": fit_metrics,
        "independent_assessment_metrics": assessment_metrics,
        "independent_assessment_quality": quality,
        "candidate_deployable_after_assessment": quality["candidate_quality_passed"],
        "reason": (
            "Candidate is eligible for final evaluation because fit and assessment Analyzer accuracy passed the predeclared gate."
            if quality["candidate_quality_passed"] else
            "Candidate remains diagnostic only because independent assessment did not pass all aggregate, class-wise, and SNR-band calibration checks."
        ),
    })
    _write_json(root / "calibration_decision.json", decision)
    _write_json(root / "fit_and_assessment_metrics.json", {
        "temperature": temperature,
        "optimizer": optimizer,
        "calibration_fit": fit_metrics,
        "independent_calibration_assessment": assessment_metrics,
        "assessment_quality": quality,
        "final_set_generated_or_opened": False,
    })
    print(f"Frozen candidate T={temperature:.8g}; fit/assessment complete; final split has not been generated or opened.", flush=True)


def _final(root: Path, samples_per_class: int) -> None:
    decision = _read_json(root / "calibration_decision.json")
    manifest = _read_json(root / "dataset_manifest.json")
    contract, model_hash = _check_contract()
    if decision["model_sha256"] != model_hash or manifest["model_sha256"] != model_hash:
        raise RuntimeError("calibration decision/dataset model hash differs from current ONNX model")
    if "final_calibration_test" in manifest["splits"] or (root / "final_calibration_test.npz").exists():
        raise RuntimeError("final split has already been generated; refusing to overwrite it")
    if samples_per_class != manifest["samples_per_class"]:
        raise RuntimeError("final split sample count must match the frozen fit/assessment count")
    seen_hashes: set[str] = set()
    for split in ("calibration_fit", "calibration_assessment"):
        seen_hashes.update(manifest["splits"][split]["raw_waveform_sha256_list"])
    path = root / "final_calibration_test.npz"
    data, info = _summarize_split(
        "final_calibration_test", FINAL_SEED, samples_per_class, path, seen_hashes
    )
    info["baseline_metrics"] = _baseline_split_metrics(data)
    info["calibration_metrics"] = None
    temperature = decision.get("final_temperature_for_metrics")
    if temperature is not None:
        metrics = score_dataset(data, float(temperature))
        quality = _calibration_quality(metrics)
        info["calibration_metrics"] = metrics
        info["final_calibration_quality"] = quality
        assessment_pass = bool(decision.get("candidate_deployable_after_assessment"))
        decision["final_evaluation_quality"] = quality
        decision["candidate_deployable_after_final_test"] = bool(
            assessment_pass and quality["candidate_quality_passed"]
        )
        if not decision["candidate_deployable_after_final_test"]:
            decision["reason_after_final_test"] = (
                "Candidate is not suitable for deployment because the frozen temperature failed at least one final aggregate, class-wise, or SNR-band check, or assessment had already failed."
            )
        else:
            decision["reason_after_final_test"] = (
                "Frozen temperature improved aggregate metrics and passed the assessment and final class-wise/SNR-band checks; this supports calibrated-score consideration within this synthetic scope only."
            )
        _write_json(root / "calibration_decision.json", decision)
        _write_json(root / "final_test_metrics.json", {
            "model_sha256": model_hash,
            "temperature": float(temperature),
            "temperature_record_sha256": sha256_file(root / "frozen_temperature.json"),
            "final_test_artifact_sha256": info["artifact_sha256"],
            "metrics": metrics,
            "quality": quality,
            "temperature_refit_after_freeze": False,
            "temperature_changed_after_freeze": False,
        })
    manifest["splits"]["final_calibration_test"] = info
    manifest["duplicate_waveform_check"] = {
        "checked_across_all_three_disjoint_splits": True,
        "unique_waveform_count": len(seen_hashes),
        "duplicates_found": 0,
    }
    manifest["final_test_opened_after_decision_freeze"] = True
    fit_data = _load_split(root, "calibration_fit", manifest)
    assessment_data = _load_split(root, "calibration_assessment", manifest)
    all_data = _combine_raw_datasets(fit_data, assessment_data, data)
    feature_distribution = {
        "initial_classifier_input_before_any_CFO_retry": _feature_distribution({
            "features": all_data["all_call_features"][:, 0, :],
            "labels": all_data["labels"],
        }, contract),
        "CFO_retry_classifier_input_when_second_call_occurs": _feature_distribution({
            "features": all_data["all_call_features"][all_data["call_count"] == 2, 1, :],
            "labels": all_data["labels"][all_data["call_count"] == 2],
        }, contract),
        "selected_classifier_input_for_final_Analyzer_classification": _feature_distribution({
            "features": all_data["features"],
            "labels": all_data["labels"],
        }, contract),
        "scope_note": "Training class-conditional feature rows are unavailable. Frozen scaler mean/scale are aggregate training statistics only.",
    }
    manifest["feature_distribution"] = feature_distribution
    _write_json(root / "dataset_manifest.json", manifest)
    _write_json(root / "final_calibration_test_baseline_metrics.json", info["baseline_metrics"])
    _write_json(root / "feature_distribution.json", feature_distribution)
    path_diagnostics = _path_diagnostics(all_data)
    _write_json(root / "classifier_path_diagnostics.json", path_diagnostics)
    print(
        f"Final holdout evaluated after decision freeze; accuracy={info['baseline_metrics']['analyzer_end_to_end']['accuracy']:.4f}; unique raw waveforms={len(seen_hashes)}.",
        flush=True,
    )


def _format_matrix(matrix: list[list[int]], columns: list[str]) -> list[str]:
    rows = ["| Truth \\ Prediction | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
    rows.extend(
        f"| {class_name} | " + " | ".join(str(value) for value in row) + " |"
        for class_name, row in zip(CLASS_NAMES, matrix)
    )
    return rows


def _write_reliability_diagram(root: Path, metrics_by_split: list[tuple[str, dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metrics_by_split), figsize=(5.2 * len(metrics_by_split), 4.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (name, metrics) in zip(axes, metrics_by_split):
        axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
        for key, label, marker in (("reliability_before", "Raw", "o"), ("reliability_after", "Temperature scaled", "s")):
            rows = [item for item in metrics[key] if item["count"] > 0]
            axis.plot(
                [item["mean_confidence"] for item in rows],
                [item["accuracy"] for item in rows],
                marker=marker,
                linewidth=1.4,
                label=label,
            )
        axis.set_title(name)
        axis.set_xlabel("Mean maximum softmax probability")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Accuracy of selected production-path model invocation")
    fig.suptitle("Reliability for logits captured from the Analyzer path")
    fig.tight_layout()
    fig.savefig(root / "reliability_diagram.png", dpi=180)
    plt.close(fig)


def _report(root: Path) -> None:
    manifest = _read_json(root / "dataset_manifest.json")
    baseline = _read_json(root / "baseline_metrics.json")
    decision = _read_json(root / "calibration_decision.json")
    final_base = _read_json(root / "final_calibration_test_baseline_metrics.json")
    contract, model_hash = _check_contract()
    if manifest["model_sha256"] != model_hash:
        raise RuntimeError("report model hash differs from the frozen ONNX artifact")
    manifest["evaluation_script_sha256"] = sha256_file(Path(__file__).resolve())
    _write_json(root / "dataset_manifest.json", manifest)
    fit_data = _load_split(root, "calibration_fit", manifest)
    assessment_data = _load_split(root, "calibration_assessment", manifest)
    final_data = _load_split(root, "final_calibration_test", manifest)
    sample_metadata = _write_sample_metadata(root, manifest, {
        "calibration_fit": fit_data,
        "calibration_assessment": assessment_data,
        "final_calibration_test": final_data,
    })
    manifest["per_signal_metadata"] = sample_metadata
    _write_json(root / "dataset_manifest.json", manifest)
    all_data = _combine_raw_datasets(fit_data, assessment_data, final_data)
    parity = _verify_onnx_parity(root, model_hash, [fit_data, assessment_data, final_data])
    _write_json(root / "production_parity.json", parity)
    if "selected_classifier_input_for_final_Analyzer_classification" not in manifest.get("feature_distribution", {}):
        feature_distribution = {
            "initial_classifier_input_before_any_CFO_retry": _feature_distribution({
                "features": all_data["all_call_features"][:, 0, :], "labels": all_data["labels"]
            }, contract),
            "CFO_retry_classifier_input_when_second_call_occurs": _feature_distribution({
                "features": all_data["all_call_features"][all_data["call_count"] == 2, 1, :],
                "labels": all_data["labels"][all_data["call_count"] == 2],
            }, contract),
            "selected_classifier_input_for_final_Analyzer_classification": _feature_distribution({
                "features": all_data["features"], "labels": all_data["labels"]
            }, contract),
            "scope_note": "Training class-conditional feature rows are unavailable. Frozen scaler mean/scale are aggregate training statistics only.",
        }
        manifest["feature_distribution"] = feature_distribution
        _write_json(root / "dataset_manifest.json", manifest)
        _write_json(root / "feature_distribution.json", feature_distribution)
    if not (root / "classifier_path_diagnostics.json").is_file():
        _write_json(root / "classifier_path_diagnostics.json", _path_diagnostics(all_data))
    calibration_outputs: list[tuple[str, dict[str, Any]]] = []
    if decision["calibration_candidate_fitted"]:
        fit_assessment = _read_json(root / "fit_and_assessment_metrics.json")
        final_metrics = _read_json(root / "final_test_metrics.json")["metrics"]
        calibration_outputs = [
            ("Fit", fit_assessment["calibration_fit"]),
            ("Assessment", fit_assessment["independent_calibration_assessment"]),
            ("Final test", final_metrics),
        ]
    else:
        # T=1 makes an explicit no-calibration reference; the report labels it as such.
        for name in ("calibration_fit", "calibration_assessment"):
            data = _load_split(root, name, manifest)
            calibration_outputs.append((name.replace("calibration_", "").title(), score_dataset(data, 1.0)))
        data = _load_split(root, "final_calibration_test", manifest)
        calibration_outputs.append(("Final test", score_dataset(data, 1.0)))
    _write_reliability_diagram(root, calibration_outputs)

    rows = []
    for name, metrics in calibration_outputs:
        rows.append(
            f"| {name} | {metrics['sample_count']} | {metrics['accuracy_before']:.4f} / {metrics['accuracy_after']:.4f} | "
            f"{metrics['nll_before']:.4f} / {metrics['nll_after']:.4f} | "
            f"{metrics['ece_before']:.4f} / {metrics['ece_after']:.4f} | "
            f"{metrics['brier_before']:.4f} / {metrics['brier_after']:.4f} | "
            f"{metrics['mean_max_probability_before']:.4f} / {metrics['mean_max_probability_after']:.4f} |"
        )
    split_baselines = {
        "calibration_fit": baseline["calibration_fit"],
        "calibration_assessment": baseline["calibration_assessment"],
        "final_calibration_test": final_base,
    }
    acc_rows = []
    per_class_rows = []
    snr_rows = []
    confusion_sections: list[str] = []
    for split, split_metrics in split_baselines.items():
        analyzer_metrics = split_metrics["analyzer_end_to_end"]
        model_metrics = split_metrics["selected_model_invocation"]
        acc_rows.append(
            f"| {split} | {analyzer_metrics['sample_count']} | {model_metrics['accuracy']:.4f} | "
            f"{analyzer_metrics['accuracy']:.4f} | {analyzer_metrics['macro_precision']:.4f} | "
            f"{analyzer_metrics['macro_recall']:.4f} | {analyzer_metrics['macro_f1']:.4f} | "
            f"{split_metrics['mean_raw_top_class_probability']:.4f} |"
        )
        for class_name in CLASS_NAMES:
            values = analyzer_metrics["per_class"][class_name]
            per_class_rows.append(
                f"| {split} | {class_name} | {values['count']} | {values['accuracy']:.4f} | "
                f"{values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} |"
            )
        for band, values in split_metrics["snr_bands"].items():
            model = values["selected_model_invocation"]
            analyzer = values["analyzer_end_to_end"]
            snr_rows.append(
                f"| {split} | {band} | {values['sample_count']} | {model['accuracy']:.4f} | {analyzer['accuracy']:.4f} |"
            )
        confusion_sections.extend([
            f"{split} (rows = truth; columns = prediction, with UNKNOWN/rejected retained):",
            "",
            *_format_matrix(analyzer_metrics["confusion_matrix"], analyzer_metrics["confusion_columns"]),
            "",
        ])
    final_analyzer = final_base["analyzer_end_to_end"]
    final_model = final_base["selected_model_invocation"]
    cfo_counts = manifest["splits"]["final_calibration_test"]["cfo_retry_counts"]
    feature = manifest["feature_distribution"]["selected_classifier_input_for_final_Analyzer_classification"]
    first_features = manifest["feature_distribution"]["initial_classifier_input_before_any_CFO_retry"]
    path_diagnostics = _read_json(root / "classifier_path_diagnostics.json")
    shifted = feature["features_over_two_training_scales_from_aggregate_mean"]
    feature_rows = []
    scaler_mean = manifest["training_scaler_aggregate"]["mean"]
    scaler_scale = manifest["training_scaler_aggregate"]["scale"]
    feature_names_for_table = (
        "psk2_residual", "psk2_phase_entropy", "psk4_residual", "psk4_phase_entropy",
        "psk8_residual", "psk8_phase_entropy", "normalized_moment_2",
        "normalized_moment_4", "normalized_moment_8", "spectral_entropy",
        "spectral_peak_fraction",
    )
    for feature_name in feature_names_for_table:
        feature_index = manifest["feature_order"].index(feature_name)
        for class_name in ("bpsk", "qpsk", "8psk"):
            initial = first_features["by_class"][class_name][feature_name]
            selected = feature["by_class"][class_name][feature_name]
            feature_rows.append(
                f"| {feature_name} | {class_name} | "
                f"{initial['mean']:.3f} ± {initial['std']:.3f} "
                f"[{initial['p05']:.3f}, {initial['p95']:.3f}] | "
                f"{selected['mean']:.3f} ± {selected['std']:.3f} "
                f"[{selected['p05']:.3f}, {selected['p95']:.3f}] | "
                f"{float(scaler_mean[feature_index]):.3f} ± {float(scaler_scale[feature_index]):.3f} |"
            )
    bpsk_cfo_rows = [
        f"| {band} | {value['sample_count']} | {value['analyzer_end_to_end_accuracy']:.4f} | "
        f"{value['selected_model_raw_argmax_accuracy']:.4f} | {value['cfo_retry_applied_fraction']:.4f} |"
        for band, value in path_diagnostics["bpsk_abs_cfo_bands"].items()
    ]
    deployment = (
        "CALIBRATED candidate passed this synthetic assessment/final protocol; no production integration is performed by this diagnostic tool."
        if decision.get("candidate_deployable_after_final_test") else
        (
            "UNCALIBRATED. No temperature was fitted because BPSK Analyzer recall failed the predeclared calibration-eligibility gate."
            if not decision["calibration_candidate_fitted"] else
            "UNCALIBRATED. The frozen candidate failed at least one independent aggregate, class-wise, or SNR-band check."
        )
    )
    temperature = decision.get("temperature")
    temp_text = f"{float(temperature):.8g}" if temperature is not None else "not fitted"
    body = [
        "# Frozen production classifier: raw Analyzer-path evaluation",
        "",
        "This diagnostic uses new raw complex64 IQ files passed to `Analyzer.analyze_file`, the production `OnnxModulationClassifier`, the ONNX model, and the existing analyzer signal-support/CFO retry path. It does not modify model weights, scaler, feature/class order, production thresholds, or runtime code.",
        "",
        "## Frozen contract and raw-input path",
        "",
        f"- Branch/commit at report generation: `{manifest.get('git_branch', 'recorded by caller')}` / `{manifest.get('git_commit', 'recorded by caller')}`.",
        f"- Evaluation script SHA-256: `{manifest['evaluation_script_sha256']}`.",
        f"- ONNX SHA-256: `{manifest['model_sha256']}`.",
        f"- Feature schema: `{manifest['feature_schema_version']}` ({len(manifest['feature_order'])} features).",
        f"- Feature order: `{', '.join(manifest['feature_order'])}`.",
        f"- Class order: `{', '.join(manifest['class_order'])}`.",
        "- `Analyzer.analyze_file` detects IQ/WAV format; raw `.iq` has no embedded sample metadata and requires an explicit `IQConfig`. This evaluation used native complex64 raw IQ with an explicit sample rate. The WAV loader/stereo handling path was not part of these measurements.",
        "- With no symbol-rate bounds, matched filter, or timing recovery requested, PSK/QAM samples reached feature extraction unchanged at symbol rate with classifier SPS=1. The FSK call received the waveform with SPS=8. The feature extractor RMS-normalizes internally; the file samples themselves were not normalized before Analyzer.",
        "- The Analyzer extracts features once for its signal-support gate, then `OnnxModulationClassifier.classify` extracts the same 17 features again for inference. The wrapper casts the raw feature vector to float32 and applies no external scaler; StandardScaler is embedded in ONNX before the MLP.",
        "- The model emits raw `logits`; the wrapper computes stable softmax probabilities. Its accepted label also uses the existing probability/margin/geometric-evidence rules, and Analyzer applies a separate 0.35 geometric support gate. Classifier confidence is margin/support-weighted and is not the raw top softmax probability.",
        "- Optional preprocessing is explicit: symbol-rate estimation requires bounds; matched filtering requires integer SPS; timing recovery requires SPS≥2. If timing recovery succeeds, PSK/QAM classifier input drops the first 30% when at least 256 samples remain, then classifier SPS becomes 1. This evaluation did not request any of those operations.",
        "- No CFO correction precedes first classification. After an accepted supported BPSK/QPSK/8PSK result, the Analyzer estimates CFO using that predicted order, corrects classifier samples and calls the classifier again; it keeps the corrected result only if status is successful, the class remains supported, and corrected margin-weighted confidence is at least the initial confidence. FSK/QAM do not use this classifier-side retry unless QAM was initially misidentified as PSK.",
        "- PSK/QAM: 2048 symbol-rate complex64 samples, random symbols, AWGN, independent phase/gain and CFO in ±400 Hz; Analyzer receives no timing recovery or matched filter because its documented classifier contract expects synchronized symbol-rate PSK/QAM samples.",
        "- 2FSK: 2048 random states, continuous phase, 8 samples/symbol, tones ±3 kHz at 48 kHz, AWGN and random phase/gain; Analyzer receives the waveform plus SPS=8.",
        "- SNR: continuous uniform 8–20 dB. Feature extractor performs its existing RMS normalization. No added pulse shaping or synchronization preprocessing was invented.",
        "- For accepted BPSK/QPSK/8PSK candidates, Analyzer's existing CFO estimate/correct/reclassify pass runs unchanged. Raw logits/features from each actual classifier call were recorded; the selected logits follow the Analyzer's `applied`/`not_applied` decision. The signal-support gate's UNKNOWN/rejected outcomes count as incorrect in end-to-end accuracy.",
        "- Training rows and class-conditional training feature distributions were not present in the checkout, data folders, or source archives. The only available training-distribution reference is the frozen contract's aggregate StandardScaler mean/scale; it is not a substitute for per-class training samples.",
        f"- Independent ONNX parity check: `{parity['exact_float32_logit_match_count']}/{parity['compared_production_classifier_calls']}` captured call outputs match direct CPU ONNX replay bit-for-bit; maximum absolute difference {parity['max_absolute_logit_difference']:.3g}.",
        "",
        "## Independent raw datasets",
        "",
        "| Split | Seed | Samples | Per class | Unique waveforms | SNR range | Raw IQ artifact SHA-256 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("calibration_fit", "calibration_assessment", "final_calibration_test"):
        info = manifest["splits"][name]
        body.append(
            f"| {name} | {info['seed']} | {info['sample_count']} | {info['samples_per_class']} | "
            f"{info['unique_raw_waveform_count']} | {info['snr_db_observed'][0]:.2f}–{info['snr_db_observed'][1]:.2f} dB | `{info['artifact_sha256']}` |"
        )
    body.extend([
        "",
        f"No waveform hashes repeated across the three splits ({manifest['duplicate_waveform_check']['unique_waveform_count']} unique raw waveforms). Fit/assessment and the candidate/no-fit decision were frozen before the final split was generated. The existing final production test set and Tier-2 payloads were not used.",
        "",
        "Per-signal generation and inference metadata, including seed components, class, waveform hash, sample/sample-rate/SPS counts, SNR, CFO, phase, gain, pulse-shaping/matched-filter/timing settings, and selected Analyzer result, are in the three `sample_metadata/*.jsonl` manifests. The script regenerates each waveform from its seed and verifies its hash against the evaluated NPZ before writing these records.",
        "",
        "## Baseline accuracy",
        "",
        "Selected-model accuracy is the argmax of the last classifier invocation chosen by the existing Analyzer CFO retry decision. Analyzer end-to-end accuracy uses its final class label after the signal-support gate; UNKNOWN/rejected labels are errors. Precision/recall/F1 below are macro averages across the five supported classes.",
        "",
        "The earlier feature-level 64.4% aggregate is not reproduced on this Analyzer-path envelope: selected model invocation accuracy is about 94.4% and Analyzer end-to-end accuracy about 92.8%. That aggregate hides the BPSK weakness shown below.",
        "",
        "| Split | N | Selected model accuracy | Analyzer accuracy | Macro precision | Macro recall | Macro F1 | Mean raw top score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *acc_rows,
        "",
        "Analyzer per-class metrics for all splits:",
        "",
        "| Split | Class | N | Accuracy | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
        *per_class_rows,
        "",
        f"Final holdout mean maximum raw softmax score: {final_base['mean_raw_top_class_probability']:.4f}. Mean margin/support-weighted Analyzer classifier confidence: {final_base['mean_margin_weighted_classifier_confidence']:.4f}. Mean classifier calls per signal: {final_base['mean_classifier_calls_per_signal']:.3f}; CFO retry statuses: `{json.dumps(cfo_counts, sort_keys=True)}`.",
        "",
        "Analyzer confusion matrices by split:",
        "",
        *confusion_sections,
        "",
        "SNR-band accuracy for all splits:",
        "",
        "| Split | SNR | N | Selected model accuracy | Analyzer end-to-end accuracy |",
        "|---|---|---:|---:|---:|",
        *snr_rows,
        "",
        "## Feature and training-distribution diagnosis",
        "",
        "All feature rows below came from the production classifier calls captured inside the raw-file Analyzer execution. The scaler comparison uses frozen aggregate training scaler statistics only; class-conditional training distributions and original training arrays were unavailable.",
        "",
        f"Features whose fresh overall mean differs from the aggregate scaler mean by more than two scaler scales: {', '.join(shifted) if shifted else 'none'}.",
        "",
        "No evidence of a broad selected-feature mean shift appears against the aggregate scaler reference. This does not establish per-class training alignment: the contract lacks class-wise training rows and channel-generation metadata. Overall feature-distribution comparisons cannot show whether training covered the evaluated CFO/SNR combinations.",
        "",
        "`feature_distribution.json` reports all 17 features separately for the initial raw classifier input, the CFO retry input, and the path-selected feature set. It includes per-class means, standard deviations, min/p05/p95/max ranges, shifts in aggregate scaler units, and pooled standardized mean separations for BPSK/QPSK/8PSK. The 8PSK phase-entropy and normalized-moment features separate corrected BPSK/QPSK from 8PSK; spectral entropy and peak fraction are nearly common across symbol-rate PSK classes in this dataset.",
        "",
        "Selected PSK/moment/spectral features by class. Values are mean ± standard deviation [5th, 95th percentile]; scaler column is the pooled training mean ± scale, not a class-specific reference:",
        "",
        "| Feature | Class | First-pass mean ± SD [p05, p95] | Analyzer-selected mean ± SD [p05, p95] | Aggregate training scaler mean ± scale |",
        "|---|---|---:|---:|---:|",
        *feature_rows,
        "",
        "### PSK path finding",
        "",
        f"Across all three splits after the final holdout was opened, the initial model argmax labeled BPSK as 8PSK in {path_diagnostics['per_class']['bpsk']['first_raw_argmax_counts'].get('8psk', 0)}/{path_diagnostics['per_class']['bpsk']['sample_count']} signals and QPSK as 8PSK in {path_diagnostics['per_class']['qpsk']['first_raw_argmax_counts'].get('8psk', 0)}/{path_diagnostics['per_class']['qpsk']['sample_count']}. The accepted CFO retry substantially reduces that first-pass bias. This post-freeze aggregation was descriptive only; no model, temperature, or threshold choice used the holdout.",
        f"For BPSK, {path_diagnostics['bpsk_rejected_retry_detail']['retry_count']} attempted second passes were not selected; in {path_diagnostics['bpsk_rejected_retry_detail']['second_pass_returned_bpsk']} of them the corrected classifier returned BPSK, but its margin-weighted confidence averaged {path_diagnostics['bpsk_rejected_retry_detail']['mean_second_confidence']:.8f} versus {path_diagnostics['bpsk_rejected_retry_detail']['mean_first_confidence']:.8f} initially, so the unchanged Analyzer rule retained the initial result. The mean confidence difference was only {path_diagnostics['bpsk_rejected_retry_detail']['mean_first_minus_second_confidence']:.2e}.",
        "",
        "BPSK Analyzer accuracy by absolute CFO magnitude:",
        "",
        "| |CFO| band | N | Analyzer accuracy | Selected raw-logit accuracy | CFO retry applied fraction |",
        "|---|---:|---:|---:|---:|",
        *bpsk_cfo_rows,
        "",
        "The cleanly CFO-corrected classifier outputs are usually BPSK-correct, while many are rejected because the corrected margin-weighted confidence is lower by only about 1.5×10⁻⁶. The dominant observed issue is therefore the interaction between CFO-conditioned first-pass features and Analyzer's confidence-based retry selection; the available evidence does not isolate a model-weight failure.",
        "",
        "## Temperature calibration",
        "",
        f"Temperature fit status: `{decision['calibration_candidate_fitted']}`; frozen candidate T: `{temp_text}`. Eligibility required both fit and assessment Analyzer accuracy ≥{ACCURACY_GATE_OVERALL:.0%} overall and ≥{ACCURACY_GATE_PER_CLASS:.0%} recall for every supported class. BPSK recall failed that gate in both pre-final splits, so no temperature was fitted. The no-fit decision was frozen before the final set was generated.",
        "",
        "No calibrated before/after comparison exists. The T=1 rows below are raw-inference baseline metrics shown with identical before/after columns only to retain the same NLL/ECE/Brier summary format; they are not evidence of calibration improvement. The reliability diagram plots these raw model-invocation scores. Signal-support fit evidence is not a probability, and OOD rejection is not evaluated here.",
        "",
        "| Split | N | Accuracy before / after | NLL before / after | ECE before / after | Brier before / after | Mean max probability before / after |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"Deployment decision: **{deployment}**",
        "",
        "`baseline_metrics.json`, the per-split baseline metric files, and `classifier_path_diagnostics.json` contain overall, per-class, SNR-band, confusion and retry-path evidence. No temperature is integrated into the Analyzer or GUI by this report.",
        "",
        "## Retraining feasibility and decision",
        "",
        "**RETRAINING NOT INDICATED**",
        "",
        "The BPSK weakness is reproducible and is material at |CFO| ≥200 Hz, but the evidence points first to Analyzer retry selection: corrected second-pass classifications frequently return BPSK correctly and are then discarded by the confidence comparison. Retraining could improve first-pass BPSK predictions for high-CFO inputs, but it would not reliably resolve a rule that selects the wrong first pass over a correct second pass. No evidence shows that the network architecture lacks capacity on CFO-corrected BPSK inputs.",
        "",
        "If later training is authorized after the retry-path behavior is diagnosed, the most relevant new training coverage is BPSK at 200–400 Hz CFO across the evaluated 8–20 dB SNR, with varied phase/gain and paired raw/CFO-corrected feature inputs; include QPSK high-CFO cases as a secondary check. The original training arrays and CFO/SNR generation metadata should be recovered first to test whether this is true training-distribution mismatch. If retrained from new training data, fit/validate a scaler only on training data and embed it consistently; keep the 17-feature schema and five-class order if possible. The ONNX hash would change, and any scaler change must be reflected in the contract. Revalidate leakage-separated model metrics, raw Analyzer path, class order/feature parity, GUI, full suite, and Tier-2/OOD separately before release. No retraining was run.",
        "",
        "## Scope and limitations",
        "",
        "The conclusions apply to this deterministic synthetic symbol-rate/FSK signal envelope and this exact model hash. They do not establish OTA/real-world accuracy, universal confidence calibration, performance for arbitrary pulse shaping or timing recovery, or universal OOD rejection. Per-class training-distribution comparisons cannot be made without the original training rows. OOD/Tier-2 results are separate and are not folded into these in-scope accuracy/calibration metrics.",
        "",
        "The model, embedded scaler, feature order, class order and production inference code remain frozen. Existing UI wording remains `Model score: <value> · uncalibrated` unless a separately validated integration is made.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe -m tools.calibration.evaluate_analyzer_path prepare --output reports\\calibration\\<new-run>",
        ".\\.venv\\Scripts\\python.exe -m tools.calibration.evaluate_analyzer_path fit --data-dir reports\\calibration\\<new-run>",
        ".\\.venv\\Scripts\\python.exe -m tools.calibration.evaluate_analyzer_path final --data-dir reports\\calibration\\<new-run>",
        ".\\.venv\\Scripts\\python.exe -m tools.calibration.evaluate_analyzer_path report --data-dir reports\\calibration\\<new-run>",
        "```",
        "",
    ])
    (root / "analyzer_path_calibration_report.md").write_text("\n".join(body), encoding="utf-8")
    print(f"Wrote {root / 'analyzer_path_calibration_report.md'}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="generate and evaluate fit + assessment raw signals only")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS_DEFAULT)
    fit = sub.add_parser("fit", help="freeze fit-only temperature or no-fit decision")
    fit.add_argument("--data-dir", type=Path, required=True)
    final = sub.add_parser("final", help="evaluate final holdout only after the temperature/no-fit decision is frozen")
    final.add_argument("--data-dir", type=Path, required=True)
    final.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS_DEFAULT)
    report = sub.add_parser("report", help="write figures and final markdown report")
    report.add_argument("--data-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "prepare":
        if args.samples_per_class < 1:
            raise SystemExit("--samples-per-class must be positive")
        _prepare(args.output.resolve(), args.samples_per_class)
    elif args.command == "fit":
        _fit(args.data_dir.resolve())
    elif args.command == "final":
        _final(args.data_dir.resolve(), args.samples_per_class)
    else:
        _report(args.data_dir.resolve())


if __name__ == "__main__":
    main()
