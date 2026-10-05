"""Evaluate frozen-model temperature scaling on independent synthetic splits.

This tool never updates model weights, feature scaling, class order, or ONNX.
The fit command reads only the fit and independent calibration-assessment
arrays. The final-test command requires a previously written frozen
temperature record and does not call the optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from scipy.optimize import minimize_scalar

from core.common.enums import ModulationType
from core.dsp.modulation_features import FEATURE_NAMES, extract_modulation_features
from core.ml.model_def import DEFAULT_MODEL_CLASS_ORDER, FEATURE_SCHEMA_VERSION
from libs.modulation_library import constellation_for
from tools.tier2.generate_signals import _apply_channel, _make_fsk


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "production_mlp_final.onnx"
CONTRACT_PATH = ROOT / "production_ml_contract.json"
CLASS_NAMES = ("bpsk", "qpsk", "8psk", "16qam", "2fsk")
MODULATIONS = (
    ModulationType.BPSK,
    ModulationType.QPSK,
    ModulationType.PSK8,
    ModulationType.QAM16,
    ModulationType.FSK2,
)
SPLITS = {
    "calibration_fit": 2026100501,
    "calibration_assessment": 2026100502,
    "final_test": 2026100503,
}
SAMPLES_PER_CLASS = 400
SYMBOLS_PER_EXAMPLE = 2048
SNR_RANGE_DB = (8.0, 20.0)
SNR_BANDS = ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0))
SAMPLE_RATE_HZ = 1_000_000.0
FSK_SAMPLE_RATE_HZ = 48_000.0
FSK_SAMPLES_PER_SYMBOL = 8
FSK_TONES_HZ = (-3_000.0, 3_000.0)
CFO_RANGE_HZ = (-400.0, 400.0)
RELIABILITY_BINS = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Return stable softmax probabilities for a positive temperature."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"logits must have shape [batch, {len(CLASS_NAMES)}]")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    scaled = values / float(temperature)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _negative_log_likelihood(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    target = np.asarray(labels, dtype=np.int64)
    normalizer = np.logaddexp.reduce(values, axis=1)
    return float(np.mean(normalizer - values[np.arange(target.size), target]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Minimize calibration-fit NLL over log(T); assessment/final are not inputs."""
    values = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_NAMES):
        raise ValueError("expected five-class logits")
    if target.shape != (values.shape[0],) or target.size == 0:
        raise ValueError("labels must contain one class index per logit row")
    if np.any(target < 0) or np.any(target >= len(CLASS_NAMES)):
        raise ValueError("labels contain an out-of-range class index")
    bounds = (-8.0, 8.0)
    result = minimize_scalar(
        lambda log_t: _negative_log_likelihood(values, target, math.exp(log_t)),
        method="bounded",
        bounds=bounds,
        options={"xatol": 1e-10, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    if result.x <= bounds[0] + 1e-5 or result.x >= bounds[1] - 1e-5:
        raise RuntimeError("temperature optimum reached the configured log(T) search boundary")
    temperature = float(math.exp(float(result.x)))
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("optimizer produced an invalid temperature")
    return temperature, {
        "method": "bounded_scalar_minimization_of_calibration_fit_NLL_over_log_temperature",
        "success": bool(result.success),
        "iterations": int(result.nfev),
        "fit_nll": float(result.fun),
        "log_temperature_bounds": list(bounds),
    }


def _ece(confidence: np.ndarray, correctness: np.ndarray, bins: int = RELIABILITY_BINS) -> tuple[float, list[dict[str, Any]]]:
    confidence = np.asarray(confidence, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    total = max(confidence.size, 1)
    error = 0.0
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        count = int(np.sum(mask))
        if count:
            mean_confidence = float(np.mean(confidence[mask]))
            accuracy = float(np.mean(correctness[mask]))
            error += count / total * abs(mean_confidence - accuracy)
        else:
            mean_confidence = None
            accuracy = None
        rows.append({
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
        })
    return float(error), rows


def _subset_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float) -> dict[str, Any]:
    raw = stable_softmax(logits, 1.0)
    calibrated = stable_softmax(logits, temperature)
    predicted = np.argmax(logits, axis=1)
    calibrated_predicted = np.argmax(calibrated, axis=1)
    truth = np.asarray(labels, dtype=np.int64)
    correctness = predicted == truth
    confidence = np.max(raw, axis=1)
    calibrated_confidence = np.max(calibrated, axis=1)
    raw_ece, raw_bins = _ece(confidence, correctness)
    calibrated_ece, calibrated_bins = _ece(calibrated_confidence, correctness)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[truth]
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    class_rows: dict[str, Any] = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_truth = truth == class_index
        class_conf_raw, _ = _ece(raw[:, class_index], class_truth)
        class_conf_cal, _ = _ece(calibrated[:, class_index], class_truth)
        count = int(np.sum(class_truth))
        class_rows[class_name] = {
            "count": count,
            "accuracy": float(np.mean(predicted[class_truth] == class_index)) if count else None,
            "ece_one_vs_rest_before": class_conf_raw,
            "ece_one_vs_rest_after": class_conf_cal,
            "mean_probability_before": float(np.mean(raw[:, class_index])) if truth.size else None,
            "mean_probability_after": float(np.mean(calibrated[:, class_index])) if truth.size else None,
        }
    return {
        "sample_count": int(truth.size),
        "accuracy_before": float(np.mean(predicted == truth)),
        "accuracy_after": float(np.mean(calibrated_predicted == truth)),
        "nll_before": _negative_log_likelihood(logits, truth, 1.0),
        "nll_after": _negative_log_likelihood(logits, truth, temperature),
        "ece_before": raw_ece,
        "ece_after": calibrated_ece,
        "brier_before": float(np.mean(np.sum((raw - one_hot) ** 2, axis=1))),
        "brier_after": float(np.mean(np.sum((calibrated - one_hot) ** 2, axis=1))),
        "mean_max_probability_before": float(np.mean(confidence)),
        "mean_max_probability_after": float(np.mean(calibrated_confidence)),
        "argmax_unchanged": bool(np.array_equal(predicted, calibrated_predicted)),
        "reliability_before": raw_bins,
        "reliability_after": calibrated_bins,
        "confusion_matrix": confusion.tolist(),
        "per_class": class_rows,
    }


def score_dataset(data: dict[str, np.ndarray], temperature: float) -> dict[str, Any]:
    """Compute overall, class, reliability, and predeclared SNR-band metrics."""
    logits = np.asarray(data["logits"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    snr = np.asarray(data["snr_db"], dtype=np.float64)
    result = _subset_metrics(logits, labels, temperature)
    result["temperature"] = float(temperature)
    bands: dict[str, Any] = {}
    for low, high in SNR_BANDS:
        mask = (snr >= low) & ((snr < high) if high < SNR_RANGE_DB[1] else (snr <= high))
        label = f"{low:g}-{high:g} dB"
        if not np.any(mask):
            bands[label] = {"sample_count": 0}
            continue
        band_metrics = _subset_metrics(logits[mask], labels[mask], temperature)
        band_metrics["snr_min_db"] = float(np.min(snr[mask]))
        band_metrics["snr_max_db"] = float(np.max(snr[mask]))
        bands[label] = band_metrics
    result["snr_bands"] = bands
    return result


def _check_contract() -> tuple[str, ort.InferenceSession, dict[str, Any]]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    actual_hash = sha256_file(MODEL_PATH)
    if actual_hash != contract["artifact"]["onnx_sha256"]:
        raise RuntimeError("production ONNX SHA-256 does not match production_ml_contract.json")
    if tuple(contract["features"]) != tuple(FEATURE_NAMES):
        raise RuntimeError("feature implementation/order differs from production_ml_contract.json")
    if tuple(contract["class_order"]) != CLASS_NAMES:
        raise RuntimeError("contract class order differs from calibration class order")
    if tuple(item.value for item in DEFAULT_MODEL_CLASS_ORDER) != CLASS_NAMES:
        raise RuntimeError("classifier fallback class order differs from contract")
    if contract["input"]["shape"] != ["batch", 17] or contract["output"]["shape"] != ["batch", 5]:
        raise RuntimeError("production tensor shape differs from the frozen contract")
    if not contract["preprocessing"]["embedded_in_onnx"] or contract["preprocessing"]["apply_external_scaling"]:
        raise RuntimeError("contract does not describe an embedded scaler with raw feature inputs")
    if contract["inference"]["wrapper_applies_temperature"]:
        raise RuntimeError("current wrapper contract unexpectedly applies temperature")
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise RuntimeError("expected one ONNX feature input and one logits output")
    if model_input.shape != ["batch", 17] or model_output.shape != ["batch", 5]:
        raise RuntimeError("ONNX Runtime tensor shape differs from the production contract")
    if model_input.type != "tensor(float)" or model_output.type != "tensor(float)":
        raise RuntimeError("ONNX Runtime input/output must both be float32")
    if model_output.name != "logits":
        raise RuntimeError(f"expected raw-logit output named 'logits', got {model_output.name!r}")
    return actual_hash, session, contract


def _make_example(class_index: int, rng: np.random.Generator, snr_db: float) -> tuple[np.ndarray, float]:
    modulation = MODULATIONS[class_index]
    if modulation is ModulationType.FSK2:
        random_states = rng.integers(0, 2, SYMBOLS_PER_EXAMPLE, dtype=np.uint8)
        waveform = _make_fsk(random_states, FSK_SAMPLE_RATE_HZ, FSK_SAMPLES_PER_SYMBOL, FSK_TONES_HZ)
        sample_rate = FSK_SAMPLE_RATE_HZ
        samples_per_symbol = FSK_SAMPLES_PER_SYMBOL
        cfo_hz = 0.0
    else:
        constellation = constellation_for(modulation)
        indices = rng.integers(0, constellation.size, SYMBOLS_PER_EXAMPLE)
        waveform = constellation[indices]
        sample_rate = SAMPLE_RATE_HZ
        samples_per_symbol = 1.0
        cfo_hz = float(rng.uniform(*CFO_RANGE_HZ))
    received = _apply_channel(
        waveform,
        sample_rate=sample_rate,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        phase_rad=float(rng.uniform(-np.pi, np.pi)),
        gain=float(rng.uniform(0.65, 1.45)),
        rng=rng,
    )
    feature_result = extract_modulation_features(received, samples_per_symbol=samples_per_symbol)
    if feature_result.warnings:
        raise RuntimeError(f"feature extraction failed: {feature_result.warnings}")
    if tuple(feature_result.feature_names) != tuple(FEATURE_NAMES):
        raise RuntimeError("feature extraction order changed during dataset generation")
    return np.asarray(feature_result.values, dtype=np.float32), cfo_hz


def _generate_split(
    name: str,
    seed: int,
    model_hash: str,
    session: ort.InferenceSession,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    snrs: list[float] = []
    cfos: list[float] = []
    for class_index in range(len(CLASS_NAMES)):
        for sample_index in range(SAMPLES_PER_CLASS):
            rng = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([seed, class_index, sample_index]))
            )
            snr_db = float(rng.uniform(*SNR_RANGE_DB))
            feature_row, cfo_hz = _make_example(class_index, rng, snr_db)
            feature_rows.append(feature_row)
            labels.append(class_index)
            snrs.append(snr_db)
            cfos.append(cfo_hz)
    features = np.stack(feature_rows).astype(np.float32, copy=False)
    logits = np.asarray(
        session.run(["logits"], {session.get_inputs()[0].name: features})[0], dtype=np.float64
    )
    if logits.shape != (len(labels), len(CLASS_NAMES)) or not np.all(np.isfinite(logits)):
        raise RuntimeError(f"ONNX inference returned invalid logits for split {name}")
    if np.allclose(np.sum(logits, axis=1), 1.0, rtol=0.0, atol=1e-3):
        raise RuntimeError("production ONNX output appears normalized rather than raw logits")
    feature_row_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in features]
    if len(set(feature_row_hashes)) != len(feature_row_hashes):
        raise RuntimeError(f"duplicate feature rows were generated in split {name}")
    data = {
        "features": features,
        "logits": logits,
        "labels": np.asarray(labels, dtype=np.int64),
        "snr_db": np.asarray(snrs, dtype=np.float64),
        "cfo_hz": np.asarray(cfos, dtype=np.float64),
    }
    info = {
        "split": name,
        "seed": int(seed),
        "samples_per_class": SAMPLES_PER_CLASS,
        "sample_count": int(len(labels)),
        "unique_feature_rows": len(set(feature_row_hashes)),
        "class_counts": {name: int(np.sum(data["labels"] == index)) for index, name in enumerate(CLASS_NAMES)},
        "snr_range_db_configured": list(SNR_RANGE_DB),
        "snr_range_db_observed": [float(np.min(data["snr_db"])), float(np.max(data["snr_db"]))],
        "snr_distribution": "continuous uniform per independent example",
        "cfo_range_hz_configured": {"BPSK/QPSK/8PSK/16QAM": list(CFO_RANGE_HZ), "2FSK": [0.0, 0.0]},
        "cfo_range_hz_observed": [float(np.min(data["cfo_hz"])), float(np.max(data["cfo_hz"]))],
        "signal_generation": {
            "symbols_or_fsk_states_per_example": SYMBOLS_PER_EXAMPLE,
            "psk_qam_sample_rate_hz": SAMPLE_RATE_HZ,
            "fsk_sample_rate_hz": FSK_SAMPLE_RATE_HZ,
            "fsk_samples_per_symbol": FSK_SAMPLES_PER_SYMBOL,
            "fsk_tones_hz": list(FSK_TONES_HZ),
            "random_phase_rad": "uniform[-pi, pi]",
            "amplitude_gain": "uniform[0.65, 1.45]",
            "channel": "independent Tier-2 NumPy channel helper; AWGN at configured SNR",
            "payload": "none; independent equiprobable symbols/states only",
            "rng": "NumPy PCG64 SeedSequence([split_seed, class_index, sample_index])",
        },
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_order": list(FEATURE_NAMES),
        "class_order": list(CLASS_NAMES),
        "model_sha256": model_hash,
        "elapsed_seconds_generation_and_inference": float(time.perf_counter() - started),
    }
    return data, info


def _save_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **data)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def verify_dataset(root: Path) -> None:
    """Verify saved split hashes, balanced labels, and no repeated feature rows."""
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seen_rows: set[str] = set()
    cross_split_duplicates = 0
    for split_name, info in manifest["splits"].items():
        path = root / info["artifact"]
        if sha256_file(path) != info["artifact_sha256"]:
            raise RuntimeError(f"artifact SHA-256 mismatch for {split_name}")
        data = _load_npz(path)
        row_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in data["features"]]
        if len(set(row_hashes)) != len(row_hashes):
            raise RuntimeError(f"duplicate features found within {split_name}")
        if data["features"].shape != (info["sample_count"], len(FEATURE_NAMES)):
            raise RuntimeError(f"feature shape mismatch for {split_name}")
        if data["logits"].shape != (info["sample_count"], len(CLASS_NAMES)):
            raise RuntimeError(f"logit shape mismatch for {split_name}")
        if not np.array_equal(np.bincount(data["labels"], minlength=len(CLASS_NAMES)), np.full(len(CLASS_NAMES), SAMPLES_PER_CLASS)):
            raise RuntimeError(f"class counts are not balanced for {split_name}")
        current = set(row_hashes)
        cross_split_duplicates += len(seen_rows.intersection(current))
        seen_rows.update(current)
        info["unique_feature_rows"] = len(current)
    if cross_split_duplicates:
        raise RuntimeError(f"found {cross_split_duplicates} duplicate feature vectors across splits")
    manifest["cross_split_duplicate_feature_rows"] = 0
    manifest["no_duplicate_feature_rows_verified"] = True
    manifest["historical_calibration_workflow_found"] = False
    manifest["feature_pipeline_scope"] = (
        "raw ONNX classifier feature inputs; does not model the analyzer's optional second PSK/CFO classification pass"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Verified artifact hashes, balanced class counts, and unique feature rows across all splits")


def prepare(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    model_hash, session, contract = _check_contract()
    split_info: dict[str, Any] = {}
    seen_feature_rows: set[str] = set()
    cross_split_duplicates = 0
    for name, seed in SPLITS.items():
        data, info = _generate_split(name, seed, model_hash, session)
        row_hashes = {hashlib.sha256(row.tobytes()).hexdigest() for row in data["features"]}
        cross_split_duplicates += len(seen_feature_rows.intersection(row_hashes))
        seen_feature_rows.update(row_hashes)
        path = root / f"{name}.npz"
        _save_npz(path, data)
        info["artifact"] = path.name
        info["artifact_sha256"] = sha256_file(path)
        split_info[name] = info
        print(f"Prepared {name}: {info['sample_count']} samples in {info['elapsed_seconds_generation_and_inference']:.2f}s")
    manifest = {
        "evaluation": "post-hoc temperature scaling; frozen production ONNX and scaler",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository_revision": _git_revision(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "model_sha256": model_hash,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "class_order": list(CLASS_NAMES),
        "contract_model_architecture": contract["model"]["architecture"],
        "scaler_embedded_in_onnx": contract["preprocessing"]["embedded_in_onnx"],
        "external_scaler_applied": False,
        "previously_recorded_temperature": contract["calibration"].get("temperature"),
        "previous_temperature_used": contract["calibration"].get("used_in_current_inference"),
        "splits": split_info,
        "cross_split_duplicate_feature_rows": cross_split_duplicates,
        "leakage_controls": {
            "split_seeds_distinct": len(set(SPLITS.values())) == len(SPLITS),
            "no_train_samples_used": True,
            "no_final_test_values_used_to_fit_temperature": True,
            "fit_command_reads_only": ["calibration_fit.npz", "calibration_assessment.npz"],
            "final_test_command_requires_frozen_temperature": True,
        },
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote deterministic split manifest for model {model_hash}")


def _git_revision() -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fit_and_assess(root: Path) -> None:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    model_hash, _, _ = _check_contract()
    if manifest["model_sha256"] != model_hash:
        raise RuntimeError("dataset/model SHA-256 mismatch")
    fit_info = manifest["splits"]["calibration_fit"]
    assessment_info = manifest["splits"]["calibration_assessment"]
    fit_path = root / fit_info["artifact"]
    assessment_path = root / assessment_info["artifact"]
    if sha256_file(fit_path) != fit_info["artifact_sha256"] or sha256_file(assessment_path) != assessment_info["artifact_sha256"]:
        raise RuntimeError("calibration split artifact hash mismatch")
    # Deliberately do not open final_test.npz in this command.
    fit_data = _load_npz(fit_path)
    assessment_data = _load_npz(assessment_path)
    temperature, optimizer = fit_temperature(fit_data["logits"], fit_data["labels"])
    fit_metrics = score_dataset(fit_data, temperature)
    assessment_metrics = score_dataset(assessment_data, temperature)
    if not fit_metrics["argmax_unchanged"] or not assessment_metrics["argmax_unchanged"]:
        raise RuntimeError("positive temperature unexpectedly changed predicted class ordering")
    frozen = {
        "temperature": temperature,
        "method": "temperature scaling: stable softmax(logits / T)",
        "fitted_on": "calibration_fit only",
        "assessment_used_for_fit_or_selection": False,
        "final_test_used_for_fit_or_selection": False,
        "model_sha256": model_hash,
        "calibration_fit_artifact_sha256": fit_info["artifact_sha256"],
        "calibration_assessment_artifact_sha256": assessment_info["artifact_sha256"],
        "optimizer": optimizer,
        "frozen_before_final_test_evaluation": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (root / "frozen_temperature.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    results = {
        "model_sha256": model_hash,
        "temperature": temperature,
        "optimizer": optimizer,
        "calibration_fit": fit_metrics,
        "independent_calibration_assessment": assessment_metrics,
        "note": "Final-test arrays were not opened or evaluated by this command.",
    }
    (root / "fit_and_assessment_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Frozen T={temperature:.10g}; fit NLL {fit_metrics['nll_before']:.5f} -> {fit_metrics['nll_after']:.5f}")
    print(f"Independent assessment NLL {assessment_metrics['nll_before']:.5f} -> {assessment_metrics['nll_after']:.5f}")


def assess_final(root: Path) -> None:
    frozen_path = root / "frozen_temperature.json"
    if not frozen_path.is_file():
        raise RuntimeError("fit and freeze temperature before final-test assessment")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    model_hash, _, _ = _check_contract()
    if frozen["model_sha256"] != model_hash:
        raise RuntimeError("frozen temperature is for a different ONNX model")
    temperature = float(frozen["temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("frozen temperature must be finite and greater than zero")
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    info = manifest["splits"]["final_test"]
    final_path = root / info["artifact"]
    if sha256_file(final_path) != info["artifact_sha256"]:
        raise RuntimeError("final-test artifact hash mismatch")
    final_data = _load_npz(final_path)
    metrics = score_dataset(final_data, temperature)
    if not metrics["argmax_unchanged"]:
        raise RuntimeError("positive temperature unexpectedly changed final predicted class ordering")
    result = {
        "model_sha256": model_hash,
        "temperature": temperature,
        "temperature_record_sha256": sha256_file(frozen_path),
        "final_test_artifact_sha256": info["artifact_sha256"],
        "metrics": metrics,
        "fit_repeated": False,
        "temperature_changed_after_freeze": False,
        "assessment_used_for_fit_or_selection": False,
    }
    (root / "final_test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_figures(root)
    _write_report(root)
    print(f"Final-test accuracy={metrics['accuracy_before']:.4f}, NLL={metrics['nll_before']:.5f}->{metrics['nll_after']:.5f}, ECE={metrics['ece_before']:.5f}->{metrics['ece_after']:.5f}")


def _write_figures(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit_assessment = json.loads((root / "fit_and_assessment_metrics.json").read_text(encoding="utf-8"))
    final = json.loads((root / "final_test_metrics.json").read_text(encoding="utf-8"))
    groups = (
        ("Calibration fit", fit_assessment["calibration_fit"]),
        ("Independent assessment", fit_assessment["independent_calibration_assessment"]),
        ("Final test", final["metrics"]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    for axis, (title, metrics) in zip(axes, groups):
        axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
        for key, label, marker in (("reliability_before", "Before", "o"), ("reliability_after", "After", "s")):
            rows = [row for row in metrics[key] if row["count"] > 0]
            axis.plot(
                [row["mean_confidence"] for row in rows],
                [row["accuracy"] for row in rows],
                marker=marker,
                linewidth=1.5,
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Mean top-class probability")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Empirical accuracy")
    fig.suptitle("Reliability on disjoint synthetic in-scope splits")
    fig.tight_layout()
    fig.savefig(root / "reliability_diagram.png", dpi=180)
    plt.close(fig)

    matrix = np.asarray(final["metrics"]["confusion_matrix"], dtype=np.int64)
    fig, axis = plt.subplots(figsize=(6, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Ground-truth class")
    axis.set_title("Final-test confusion matrix (raw; identical after temperature scaling)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Sample count")
    fig.tight_layout()
    fig.savefig(root / "confusion_matrix_final.png", dpi=180)
    plt.close(fig)


def _write_report(root: Path) -> None:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    fit = json.loads((root / "fit_and_assessment_metrics.json").read_text(encoding="utf-8"))
    final = json.loads((root / "final_test_metrics.json").read_text(encoding="utf-8"))
    rows = []
    for split_name, metrics in (
        ("Fit", fit["calibration_fit"]),
        ("Independent assessment", fit["independent_calibration_assessment"]),
        ("Final test", final["metrics"]),
    ):
        rows.append(
            f"| {split_name} | {metrics['sample_count']} | {metrics['accuracy_before']:.4f} / {metrics['accuracy_after']:.4f} | "
            f"{metrics['nll_before']:.5f} / {metrics['nll_after']:.5f} | {metrics['ece_before']:.5f} / {metrics['ece_after']:.5f} | "
            f"{metrics['brier_before']:.5f} / {metrics['brier_after']:.5f} | "
            f"{metrics['mean_max_probability_before']:.4f} / {metrics['mean_max_probability_after']:.4f} |"
        )
    split_rows = [
        f"| {name} | {value['seed']} | {value['sample_count']} | {value['snr_range_db_observed'][0]:.2f}–{value['snr_range_db_observed'][1]:.2f} dB | `{value['artifact_sha256']}` |"
        for name, value in manifest["splits"].items()
    ]
    class_rows = []
    for class_name in CLASS_NAMES:
        assessment = fit["independent_calibration_assessment"]["per_class"][class_name]
        final_class = final["metrics"]["per_class"][class_name]
        class_rows.append(
            f"| {class_name} | {assessment['accuracy']:.4f} | "
            f"{assessment['ece_one_vs_rest_before']:.5f} / {assessment['ece_one_vs_rest_after']:.5f} | "
            f"{final_class['accuracy']:.4f} | "
            f"{final_class['ece_one_vs_rest_before']:.5f} / {final_class['ece_one_vs_rest_after']:.5f} |"
        )
    snr_rows = []
    for split_name, metrics in (
        ("Assessment", fit["independent_calibration_assessment"]),
        ("Final test", final["metrics"]),
    ):
        for band, band_metrics in metrics["snr_bands"].items():
            snr_rows.append(
                f"| {split_name} | {band} | {band_metrics['sample_count']} | {band_metrics['accuracy_before']:.4f} | "
                f"{band_metrics['nll_before']:.5f} / {band_metrics['nll_after']:.5f} | "
                f"{band_metrics['ece_before']:.5f} / {band_metrics['ece_after']:.5f} |"
            )
    confusion_rows = [
        f"| {class_name} | " + " | ".join(str(value) for value in final["metrics"]["confusion_matrix"][index]) + " |"
        for index, class_name in enumerate(CLASS_NAMES)
    ]
    body = [
        "# Temperature scaling evaluation",
        "",
        "This is a controlled synthetic evaluation of post-hoc probability scaling for the frozen ONNX classifier. The temperature is fitted only on `calibration_fit`; `calibration_assessment` is independent, and the separate `final_test` split was evaluated only after the temperature was frozen.",
        "",
        "## Contract and data",
        "",
        f"- Model SHA-256: `{manifest['model_sha256']}`",
        f"- Feature schema: `{manifest['feature_schema_version']}`; {len(manifest['splits']['calibration_fit']['feature_order'])} raw features; external scaling: no; scaler embedded in ONNX: {manifest['scaler_embedded_in_onnx']}",
        f"- Class order: `{', '.join(manifest['class_order'])}`",
        f"- Selected temperature: `{fit['temperature']:.10g}` from calibration-fit NLL minimization; calibrated output is `softmax(logits / T)`.",
        "- Random symbols/states only: no payload, train samples, or repeated waveforms were used. Each split has its own PCG64 SeedSequence root.",
        "- No calibration/assessment arrays or temperature-fitting workflow were present in the tracked project or supplied source archives. This new run re-estimates T and does not reuse the historical `3.824104` value.",
        "- SNR was drawn continuously and uniformly from 8–20 dB, the span represented by the current Tier-2 in-scope scenarios. CFO was uniform in ±400 Hz for PSK/QAM and 0 Hz for the T2-06-style FSK setup.",
        "",
        "| Split | Seed | Rows | Observed SNR | Artifact SHA-256 |",
        "|---|---:|---:|---:|---|",
        *split_rows,
        "",
        "## Before / after metrics",
        "",
        "Accuracy and predicted class are unchanged by positive temperature scaling because argmax is unchanged; probability sharpness and mean maximum probability do change. ECE is top-label, 10 equal-width bins; Brier is the mean per-row sum of five-class squared errors.",
        "",
        "| Split | N | Accuracy before / after | NLL before / after | ECE before / after | Brier before / after | Mean max probability before / after |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Class and SNR breakdown",
        "",
        "Per-class accuracy is unchanged by a positive temperature. One-vs-rest ECE is shown before/after; a class-level regression matters even when aggregate ECE improves.",
        "",
        "| Class | Assessment accuracy | Assessment class ECE before / after | Final accuracy | Final class ECE before / after |",
        "|---|---:|---:|---:|---:|",
        *class_rows,
        "",
        "| Split | SNR band | N | Accuracy | NLL before / after | Top-label ECE before / after |",
        "|---|---|---:|---:|---:|---:|",
        *snr_rows,
        "",
        "Final-test confusion matrix (rows = truth; columns = prediction; class order `bpsk, qpsk, 8psk, 16qam, 2fsk`):",
        "",
        "| Truth \\ Prediction | BPSK | QPSK | 8PSK | 16QAM | 2FSK |",
        "|---|---:|---:|---:|---:|---:|",
        *confusion_rows,
        "",
        "## Deployment decision",
        "",
        "**Temperature scaling was evaluated but not sufficiently validated for deployment.** Aggregate NLL, ECE, and Brier score improved on independent assessment and final synthetic splits, and accuracy/argmax did not change. However, calibration was inconsistent by class: one-vs-rest ECE worsened for 16-QAM and 2FSK, while BPSK/QPSK had low feature-level accuracy in this generated inference-input distribution. The current analyzer can perform an additional PSK/CFO classifier pass, which this direct feature-level calibration set does not model. These results do not justify changing production inference or describing the GUI score as calibrated.",
        "",
        "## Files",
        "",
        "- `dataset_manifest.json`: generation parameters, class/SNR counts, versions, seeds, and model hash.",
        "- `frozen_temperature.json`: fit-only temperature and optimizer record; frozen before final-test evaluation.",
        "- `fit_and_assessment_metrics.json`, `final_test_metrics.json`: full metrics, per-class scores, SNR-band metrics, reliability-bin values, and confusion counts.",
        "- `reliability_diagram.png`, `confusion_matrix_final.png`: reproducible figures.",
        "- `calibration_fit.npz`, `calibration_assessment.npz`, `final_test.npz`: derived raw features/logits, class labels, SNR, and CFO; no waveforms or payloads.",
        "",
        "## Reproduction",
        "",
        "Use a new output directory for each run. `fit` reads only fit and assessment arrays; run `final` only after `frozen_temperature.json` exists.",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe -m tools.calibration.evaluate_temperature prepare --output reports\\calibration\\<new-run>",
        ".\\.venv\\Scripts\\python.exe -m tools.calibration.evaluate_temperature fit --data-dir reports\\calibration\\<new-run>",
        ".\\.venv\\Scripts\\python.exe -m tools.calibration.evaluate_temperature final --data-dir reports\\calibration\\<new-run>",
        "```",
        "",
        "## Scope limits",
        "",
        "These synthetic symbol-rate signals do not establish real-world/OTA probability calibration, end-to-end analyzer calibration, universal confidence, or OOD behavior. The project checkout contained no historical calibration/assessment arrays or script, so this run uses independent seeds and the current Tier-2 SNR span. OOD behavior is reported separately by the fresh Tier-2 run; temperature fitting does not include OOD samples.",
        "",
        "The production temperature remains unapplied. Preserve the current UI wording: `Model score: <value> · uncalibrated`.",
        "",
    ]
    (root / "calibration_report.md").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="generate disjoint fit, assessment, and final splits")
    prepare_parser.add_argument("--output", type=Path, required=True)
    fit_parser = subparsers.add_parser("fit", help="fit on calibration-fit only; assess on independent assessment")
    fit_parser.add_argument("--data-dir", type=Path, required=True)
    final_parser = subparsers.add_parser("final", help="evaluate final test with a frozen temperature")
    final_parser.add_argument("--data-dir", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify", help="verify saved split artifacts and duplicate checks")
    verify_parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output.resolve())
    elif args.command == "fit":
        fit_and_assess(args.data_dir.resolve())
    elif args.command == "verify":
        verify_dataset(args.data_dir.resolve())
    else:
        assess_final(args.data_dir.resolve())


if __name__ == "__main__":
    main()
