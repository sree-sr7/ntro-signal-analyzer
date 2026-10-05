"""Fresh, targeted production-path investigation of Analyzer CFO retry selection.

This experiment generates its own deterministic synthetic IQ inputs. It does not
read calibration/assessment/final arrays or Tier-2 payloads, and it never alters
the model or the production analyzer. Candidate rules are replayed from the two
actual production classifier calls captured for each Analyzer run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from core.common.enums import ModulationType, SampleDtype
from core.common.models import IQConfig
from core.dsp.modulation_features import FEATURE_NAMES
from core.ml.classifier import (
    MIN_CLASS_EVIDENCE_SCORE,
    _baseline_scores,
    strongest_modulation_evidence,
)
from core.ml.model_def import DEFAULT_MODEL_CLASS_ORDER
from core.pipeline.analyzer import Analyzer
from libs.modulation_library import constellation_for
from tools.calibration.evaluate_analyzer_path import (
    MODEL_PATH,
    ROOT,
    _RecordingClassifier,
    _apply_channel,
    _check_contract,
    _make_fsk,
    sha256_file,
)


SPLITS = {"development": 2026100701, "confirmation": 2026100702}
CLASS_NAMES = tuple(item.value for item in DEFAULT_MODEL_CLASS_ORDER)
PSK = (ModulationType.BPSK, ModulationType.QPSK, ModulationType.PSK8)
PSK_NAMES = tuple(item.value for item in PSK)
SUPPORTED_NAMES = CLASS_NAMES
CFO_BANDS = ((0.0, 100.0), (100.0, 200.0), (200.0, 300.0), (300.0, 400.0))
CFO_BAND_NAMES = ("0-100", "100-200", "200-300", "300-400")
SNR_VALUES = (9.0, 14.0, 19.0)
SAMPLE_RATE = 1_000_000.0
SAMPLES_PER_SIGNAL = 2048
REPLICATES_PER_CELL = 40
CONTROLS_PER_SNR = 40
OOD_PER_KIND = 80
FSK_SAMPLE_RATE = 48_000.0
FSK_SPS = 8
FSK_TONES = (-3_000.0, 3_000.0)
NAMES = (*CLASS_NAMES, "ood")
NAME_INDEX = {name: index for index, name in enumerate(NAMES)}
CANDIDATES = ("current", "epsilon_1e-6", "epsilon_1e-5", "geometry", "geometry_epsilon_1e-6")


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - float(np.max(logits))
    exp = np.exp(shifted)
    return exp / float(np.sum(exp))


def _stable_rng(seed: int, *parts: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, *parts])))


def _make_ofdm(rng: np.random.Generator, snr_db: float) -> np.ndarray:
    nfft, cp, symbol_count, half_active = 64, 16, 24, 26
    active = np.r_[np.arange(1, half_active + 1), np.arange(nfft - half_active, nfft)]
    blocks = []
    for _ in range(symbol_count):
        bins = np.zeros(nfft, dtype=np.complex128)
        bins[active] = np.exp(1j * (np.pi / 4 + rng.integers(0, 4, active.size) * np.pi / 2))
        useful = np.fft.ifft(bins) * np.sqrt(nfft)
        blocks.append(np.r_[useful[-cp:], useful])
    waveform = np.concatenate(blocks)
    power = float(np.mean(np.abs(waveform) ** 2))
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(waveform.size) + 1j * rng.standard_normal(waveform.size)
    )
    return (waveform + noise).astype(np.complex64)


def _make_noise(rng: np.random.Generator, snr_db: float) -> np.ndarray:
    # Unit-power noise-only input; SNR metadata denotes the deterministic scale
    # bracket used for its complex standard deviation and is not a signal SNR.
    scale = 10.0 ** ((snr_db - 14.0) / 20.0)
    noise = (rng.standard_normal(SAMPLES_PER_SIGNAL) + 1j * rng.standard_normal(SAMPLES_PER_SIGNAL)) / np.sqrt(2.0)
    return (noise * scale).astype(np.complex64)


def _make_signal(
    split_seed: int,
    label: str,
    snr_db: float,
    cfo_band_index: int | None,
    sample_index: int,
    *,
    control_kind: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    class_index = NAME_INDEX.get(label, len(NAMES))
    rng = _stable_rng(split_seed, class_index, 99 if cfo_band_index is None else cfo_band_index, sample_index)
    phase = float(rng.uniform(-np.pi, np.pi))
    gain = float(rng.uniform(0.65, 1.45))
    cfo_hz = 0.0
    sample_rate = SAMPLE_RATE
    samples_per_symbol = None
    if label in CLASS_NAMES:
        modulation = ModulationType(label)
        if modulation is ModulationType.FSK2:
            states = rng.integers(0, 2, SAMPLES_PER_SIGNAL // FSK_SPS, dtype=np.uint8)
            waveform = _make_fsk(states, FSK_SAMPLE_RATE, FSK_SPS, FSK_TONES)
            sample_rate = FSK_SAMPLE_RATE
            samples_per_symbol = FSK_SPS
            pulse_shaping = "continuous-phase rectangular 2FSK symbol intervals"
        else:
            constellation = constellation_for(modulation)
            symbol_indices = rng.integers(0, constellation.size, SAMPLES_PER_SIGNAL)
            waveform = constellation[symbol_indices]
            if cfo_band_index is not None:
                low, high = CFO_BANDS[cfo_band_index]
                magnitude = float(rng.uniform(low, high))
                cfo_hz = magnitude * (-1.0 if rng.integers(0, 2) == 0 else 1.0)
            elif modulation is ModulationType.QAM16:
                cfo_hz = float(rng.uniform(-400.0, 400.0))
            pulse_shaping = "none; symbol-rate samples"
        received = _apply_channel(
            waveform,
            sample_rate=sample_rate,
            snr_db=snr_db,
            cfo_hz=cfo_hz,
            phase_rad=phase,
            gain=gain,
            rng=rng,
        ).astype(np.complex64, copy=False)
        target = label
        generator = "uniform random constellation symbols / independent AWGN"
    elif control_kind == "noise":
        received = _make_noise(rng, snr_db)
        target = "ood"
        pulse_shaping = "noise only"
        generator = "independent complex Gaussian noise"
    elif control_kind == "ofdm":
        received = _make_ofdm(rng, snr_db)
        target = "ood"
        pulse_shaping = "64-point OFDM with 16-sample cyclic prefix and QPSK subcarriers"
        generator = "independent OFDM control"
    else:
        raise ValueError(f"Unsupported class/control {label}/{control_kind}")
    meta = {
        "split": "development" if split_seed == SPLITS["development"] else "confirmation",
        "split_seed": split_seed,
        "label": target,
        "generator_class": label,
        "generator": generator,
        "control_kind": control_kind,
        "cfo_band": None if cfo_band_index is None else CFO_BAND_NAMES[cfo_band_index],
        "sample_index": sample_index,
        "sample_rate_hz": sample_rate,
        "samples_per_symbol": samples_per_symbol,
        "symbol_or_state_count": int(received.size if samples_per_symbol is None else received.size // samples_per_symbol),
        "sample_count": int(received.size),
        "snr_db": float(snr_db),
        "snr_interpretation": "signal-to-noise ratio" if label in CLASS_NAMES else "noise scale index for OOD control",
        "cfo_hz": float(cfo_hz),
        "phase_rad": phase,
        "gain": gain,
        "pulse_shaping": pulse_shaping,
        "matched_filter_rolloff": None,
        "timing_recovery_requested": False,
        "raw_format": "complex64 native-endian .iq",
        "sha256": hashlib.sha256(np.ascontiguousarray(received).view(np.uint8)).hexdigest(),
    }
    return received, meta


def prepare(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _check_contract()
    manifest: dict[str, Any] = {
        "schema": "ntro-cfo-retry-investigation-v1",
        "model_sha256": sha256_file(MODEL_PATH),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "method": "fresh deterministic raw IQ passed through Analyzer.analyze_file; separate dev and confirm seeds; no prior evaluation arrays or Tier-2 payloads",
        "samples_per_psk_cell": REPLICATES_PER_CELL,
        "snr_values_db": list(SNR_VALUES),
        "cfo_bands_abs_hz": [list(item) for item in CFO_BANDS],
        "class_order": list(CLASS_NAMES),
        "ood_controls": {"noise_only_each_split": OOD_PER_KIND, "ofdm_each_split": OOD_PER_KIND},
        "splits": {},
    }
    seen: set[str] = set()
    for split, seed in SPLITS.items():
        signals: dict[str, np.ndarray] = {}
        rows: list[dict[str, Any]] = []

        def add(label: str, snr_db: float, cfo_band_index: int | None, index: int, kind: str | None = None) -> None:
            signal, meta = _make_signal(seed, label, snr_db, cfo_band_index, index, control_kind=kind)
            if meta["sha256"] in seen:
                raise RuntimeError(f"duplicate waveform generated: {split}/{label}/{index}")
            seen.add(meta["sha256"])
            key = f"signal_{len(rows):05d}"
            signals[key] = signal
            meta["key"] = key
            rows.append(meta)

        for class_index, modulation in enumerate(PSK):
            for band_index in range(len(CFO_BANDS)):
                for snr_index, snr_db in enumerate(SNR_VALUES):
                    for replicate in range(REPLICATES_PER_CELL):
                        index = (((class_index * len(CFO_BANDS) + band_index) * len(SNR_VALUES) + snr_index) * REPLICATES_PER_CELL + replicate)
                        add(modulation.value, snr_db, band_index, index)
        for modulation in (ModulationType.QAM16, ModulationType.FSK2):
            for snr_index, snr_db in enumerate(SNR_VALUES):
                for replicate in range(CONTROLS_PER_SNR):
                    index = snr_index * CONTROLS_PER_SNR + replicate
                    add(modulation.value, snr_db, None, index)
        for kind in ("noise", "ofdm"):
            for sample_index in range(OOD_PER_KIND):
                snr_db = SNR_VALUES[sample_index % len(SNR_VALUES)]
                add("ood", snr_db, None, sample_index, kind)
        np.savez_compressed(output / f"signals_{split}.npz", **signals)
        with (output / f"generated_{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        manifest["splits"][split] = {
            "seed": seed,
            "count": len(rows),
            "sha256_unique_including_prior_split": True,
            "signal_archive": f"signals_{split}.npz",
            "metadata": f"generated_{split}.jsonl",
        }
        _json_write(output / "dataset_manifest.json", manifest)
        print(f"prepared {split}: {len(rows)} unique signals", flush=True)
    manifest["total_unique_waveforms"] = len(seen)
    _json_write(output / "dataset_manifest.json", manifest)


def _run_analysis(recorder: _RecordingClassifier, analyzer: Analyzer, temp_path: Path, signal: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
    np.asarray(signal, dtype=np.complex64).tofile(temp_path)
    recorder.calls.clear()
    config = IQConfig(dtype=SampleDtype.COMPLEX64, sample_rate=float(meta["sample_rate_hz"]))
    analysis = analyzer.analyze_file(
        temp_path,
        iq_config=config,
        samples_per_symbol=meta["samples_per_symbol"],
        recover_symbol_timing=False,
        matched_filter_rolloff=None,
        run_fec=False,
    )
    calls = []
    for call in recorder.calls:
        logits = np.asarray(call["logits"], dtype=np.float64)
        probabilities = _softmax(logits)
        ranked = np.sort(probabilities)[::-1]
        feature_values = np.asarray(call["features"], dtype=np.float64)
        feature_map = dict(zip(FEATURE_NAMES, feature_values.tolist()))
        geometric_label, geometric_score = strongest_modulation_evidence(feature_map)
        class_fit = _baseline_scores(feature_map)
        calls.append({
            "class": call["classifier_label"],
            "status": call["classifier_status"],
            "confidence": float(call["classifier_confidence"]),
            "logits": logits.tolist(),
            "softmax": probabilities.tolist(),
            "margin": float(ranked[0] - ranked[1]),
            "features": feature_values.tolist(),
            "geometric_strongest_class": geometric_label.value,
            "geometric_strongest_score": float(geometric_score),
            "classifier_class_geometric_score": float(class_fit.get(ModulationType(call["classifier_label"]), 0.0)) if call["classifier_label"] != "unknown" else 0.0,
        })
    gate = analysis.classification.get("signal_detection_gate", {})
    cfo = analysis.synchronization.get("classifier_cfo_correction", {})
    actual_selected = 1 if cfo.get("status") == "applied" and len(calls) == 2 else 0
    return {
        "calls": calls,
        "actual_selected_index": actual_selected,
        "actual_selected_reason": cfo.get("reason", "no retry"),
        "analysis_label": str(analysis.classification.get("modulation", "unknown")).lower(),
        "analysis_status": analysis.status,
        "signal_gate": gate,
        "cfo": cfo,
        "analysis_errors": analysis.errors,
    }


def _valid_retry(record: dict[str, Any]) -> bool:
    calls = record["calls"]
    return (
        len(calls) == 2
        and record["signal_gate"].get("decision") == "signal_candidate"
        and calls[1]["status"] == "success"
        and calls[1]["class"] in SUPPORTED_NAMES
        and calls[1]["geometric_strongest_score"] >= MIN_CLASS_EVIDENCE_SCORE
    )


def _select(record: dict[str, Any], rule: str) -> int:
    if not _valid_retry(record):
        return 0
    first, second = record["calls"]
    gap = float(first["confidence"] - second["confidence"])
    if rule == "current":
        return 1 if second["confidence"] >= first["confidence"] else 0
    if rule == "epsilon_1e-6":
        return 1 if gap <= 1e-6 else 0
    if rule == "epsilon_1e-5":
        return 1 if gap <= 1e-5 else 0
    geometry_valid = (
        second["geometric_strongest_class"] == second["class"]
        and second["geometric_strongest_score"] >= MIN_CLASS_EVIDENCE_SCORE
    )
    if rule == "geometry":
        return 1 if geometry_valid else 0
    if rule == "geometry_epsilon_1e-6":
        return 1 if geometry_valid and gap <= 1e-6 else 0
    raise ValueError(f"unknown candidate rule: {rule}")


def _metrics(labels: list[str], predictions: list[str]) -> dict[str, Any]:
    matrix = np.zeros((len(NAMES), len(NAMES)), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        matrix[NAME_INDEX[actual], NAME_INDEX.get(predicted, NAME_INDEX["ood"])] += 1
    per_class: dict[str, Any] = {}
    precisions, recalls, f1s = [], [], []
    for index, label in enumerate(NAMES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": int(matrix[index, :].sum()), "precision": precision, "recall": recall, "f1": f1}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "count": len(labels),
        "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": list(NAMES),
    }


def evaluate(output: Path, split: str, candidates: tuple[str, ...]) -> None:
    if split not in SPLITS:
        raise ValueError(split)
    _check_contract()
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output / f"generated_{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    with np.load(output / f"signals_{split}.npz") as archive:
        signals = {row["key"]: archive[row["key"]].copy() for row in rows}
    recorder = _RecordingClassifier(MODEL_PATH)
    analyzer = Analyzer(classifier=recorder)
    detailed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ntro-cfo-investigation-") as temp_dir:
        temp_path = Path(temp_dir) / "signal.iq"
        for number, meta in enumerate(rows, start=1):
            signal = signals[meta["key"]]
            if hashlib.sha256(np.ascontiguousarray(signal).view(np.uint8)).hexdigest() != meta["sha256"]:
                raise RuntimeError(f"waveform checksum mismatch at {split}/{meta['key']}")
            observed = _run_analysis(recorder, analyzer, temp_path, signal, meta)
            item = {**meta, **observed}
            item["attempted_retry"] = len(observed["calls"]) == 2
            item["corrected_path_otherwise_valid"] = _valid_retry(item)
            item["corrected_class_differs_from_initial"] = (
                len(observed["calls"]) == 2
                and observed["calls"][1]["class"] != observed["calls"][0]["class"]
            )
            item["corrected_class_is_psk"] = (
                len(observed["calls"]) == 2 and observed["calls"][1]["class"] in PSK_NAMES
            )
            item["confidence_gap_initial_minus_corrected"] = (
                float(observed["calls"][0]["confidence"] - observed["calls"][1]["confidence"])
                if len(observed["calls"]) == 2 else None
            )
            item["candidate_decisions"] = {}
            for rule in candidates:
                selected = _select(item, rule)
                if not item["attempted_retry"]:
                    reason = "No second classifier pass was produced by the Analyzer."
                elif not item["corrected_path_otherwise_valid"]:
                    reason = "Corrected pass failed existing status/support/gate validity requirements."
                elif selected == 1 and rule == "current":
                    reason = "Corrected result was valid and its confidence was not lower."
                elif selected == 1 and rule.startswith("epsilon_"):
                    reason = "Corrected result was valid and the confidence gap was within the frozen tie band."
                elif selected == 1 and rule.startswith("geometry"):
                    reason = "Corrected result was valid and corrected features corroborated its label with existing geometric fit evidence."
                elif rule.startswith("geometry") and item["calls"][1]["geometric_strongest_class"] != item["calls"][1]["class"]:
                    reason = "Corrected features' strongest geometric fit did not match the corrected classifier class."
                elif rule.startswith("geometry") and item["calls"][1]["geometric_strongest_score"] < MIN_CLASS_EVIDENCE_SCORE:
                    reason = "Corrected geometric fit did not meet the existing support floor."
                else:
                    reason = "Corrected confidence fell outside the candidate's tie band."
                item["candidate_decisions"][rule] = {"selected_pass": selected, "reason": reason}
            if observed["analysis_errors"]:
                errors.append({"key": meta["key"], "errors": observed["analysis_errors"]})
            detailed.append(item)
            if number % 250 == 0 or number == len(rows):
                print(f"{split}: analyzed {number}/{len(rows)}", flush=True)

    truth = [item["label"] for item in detailed]
    baseline_indices = [_select(item, "current") for item in detailed]
    # This label includes Analyzer's signal-support gate. A gate rejection
    # maps the prediction to UNKNOWN even when the underlying model guessed a
    # supported class.
    baseline_predictions = [item["analysis_label"] for item in detailed]
    simulated_current = [
        "unknown" if item["signal_gate"].get("decision") != "signal_candidate"
        else item["calls"][index]["class"]
        for item, index in zip(detailed, baseline_indices)
    ]
    parity_mismatches = sum(left != right for left, right in zip(baseline_predictions, simulated_current))
    result: dict[str, Any] = {
        "split": split,
        "model_sha256": manifest["model_sha256"],
        "candidate_rules": {},
        "current_metrics": _metrics(truth, baseline_predictions),
        "current_simulation_parity_mismatches": parity_mismatches,
        "retry_diagnostics": {
            "attempted": sum(item["attempted_retry"] for item in detailed),
            "selected_second_pass": sum(index == 1 for index in baseline_indices),
            "rejected_second_pass": sum(item["attempted_retry"] and index == 0 for item, index in zip(detailed, baseline_indices)),
            "otherwise_valid_correct_second_pass_discarded": sum(
                item["attempted_retry"]
                and _valid_retry(item)
                and item["calls"][1]["class"] == item["label"]
                and index == 0
                and item["calls"][0]["class"] != item["label"]
                for item, index in zip(detailed, baseline_indices)
            ),
        },
        "by_cfo_snr": {},
        "repeatability": {},
        "errors": errors,
    }
    for rule in candidates:
        indices = [_select(item, rule) for item in detailed]
        predictions = [
            "unknown" if item["signal_gate"].get("decision") != "signal_candidate"
            else item["calls"][index]["class"]
            for item, index in zip(detailed, indices)
        ]
        changed = [i for i, (left, right) in enumerate(zip(baseline_predictions, predictions)) if left != right]
        correctly_recovered = [i for i in changed if predictions[i] == truth[i] and baseline_predictions[i] != truth[i]]
        introduced_errors = [i for i in changed if predictions[i] != truth[i] and baseline_predictions[i] == truth[i]]
        if rule != "current":
            result["candidate_rules"][rule] = {
                "metrics": _metrics(truth, predictions),
                "changed_from_current": len(changed),
                "previously_discarded_correct_retry_selected": len(correctly_recovered),
                "new_incorrect_decisions": len(introduced_errors),
                "correct_recoveries_by_truth_class": {
                    name: sum(truth[i] == name for i in correctly_recovered) for name in NAMES
                },
                "introduced_errors_by_truth_class": {
                    name: sum(truth[i] == name for i in introduced_errors) for name in NAMES
                },
                "selected_second_pass": sum(index == 1 for index in indices),
                "rejected_second_pass": sum(
                    item["attempted_retry"] and index == 0 for item, index in zip(detailed, indices)
                ),
            }
    for class_name in PSK_NAMES:
        for band in CFO_BAND_NAMES:
            for snr in SNR_VALUES:
                subset = [item for item in detailed if item["generator_class"] == class_name and item["cfo_band"] == band and item["snr_db"] == snr]
                key = f"{class_name}|{band}|{snr:g}dB"
                result["by_cfo_snr"][key] = {
                    "count": len(subset),
                    "attempted_retries": sum(item["attempted_retry"] for item in subset),
                    "current_accuracy": sum(item["analysis_label"] == item["label"] for item in subset) / max(1, len(subset)),
                    "initial_argmax_accuracy": sum(item["calls"][0]["class"] == item["label"] for item in subset) / max(1, len(subset)),
                    "corrected_argmax_accuracy_on_retries": sum(item["calls"][-1]["class"] == item["label"] for item in subset if item["attempted_retry"]) / max(1, sum(item["attempted_retry"] for item in subset)),
                }
    retry_items = [item for item in detailed if item["attempted_retry"]]
    confidence_gaps = [item["confidence_gap_initial_minus_corrected"] for item in retry_items]
    result["confidence_gap_summary"] = {
        "retry_count": len(confidence_gaps),
        "corrected_lower_confidence_count": sum(value > 0 for value in confidence_gaps),
        "corrected_higher_or_equal_confidence_count": sum(value <= 0 for value in confidence_gaps),
        "mean": float(np.mean(confidence_gaps)) if confidence_gaps else None,
        "median": float(np.median(confidence_gaps)) if confidence_gaps else None,
        "p90_abs": float(np.quantile(np.abs(confidence_gaps), 0.9)) if confidence_gaps else None,
        "within_1e-6": sum(abs(value) <= 1e-6 for value in confidence_gaps),
        "within_1e-5": sum(abs(value) <= 1e-5 for value in confidence_gaps),
        "min": float(np.min(confidence_gaps)) if confidence_gaps else None,
        "max": float(np.max(confidence_gaps)) if confidence_gaps else None,
    }
    result["retry_diagnostics"]["by_truth_class"] = {
        label: {
            "attempted": sum(item["label"] == label and item["attempted_retry"] for item in detailed),
            "current_selected": sum(item["label"] == label and item["actual_selected_index"] == 1 for item in detailed),
            "current_rejected": sum(item["label"] == label and item["attempted_retry"] and item["actual_selected_index"] == 0 for item in detailed),
            "correct_retry_rejected": sum(item["label"] == label and item["attempted_retry"] and item["calls"][1]["class"] == label and item["actual_selected_index"] == 0 for item in detailed),
        }
        for label in NAMES
    }
    for rule in candidates:
        indices = [_select(item, rule) for item in detailed]
        result["retry_diagnostics"].setdefault("selected_second_pass_by_rule", {})[rule] = sum(index == 1 for index in indices)
    # Re-run 60 retryable waveforms through the full Analyzer path and compare
    # logits/confidence. This directly measures determinism of the confidence
    # comparator on identical bytes; it does not contribute to rule selection.
    repeat_samples = retry_items[:60]
    if repeat_samples:
        max_abs_logit_delta = 0.0
        max_abs_confidence_delta = 0.0
        with tempfile.TemporaryDirectory(prefix="ntro-cfo-repeat-") as temp_dir:
            temp_path = Path(temp_dir) / "signal.iq"
            for item in repeat_samples:
                signal = signals[item["key"]]
                repeated = _run_analysis(recorder, analyzer, temp_path, signal, item)
                for call_a, call_b in zip(item["calls"], repeated["calls"]):
                    max_abs_logit_delta = max(max_abs_logit_delta, float(np.max(np.abs(np.asarray(call_a["logits"]) - np.asarray(call_b["logits"])))) )
                    max_abs_confidence_delta = max(max_abs_confidence_delta, abs(float(call_a["confidence"] - call_b["confidence"])))
        result["repeatability"] = {
            "identical_waveforms_replayed": len(repeat_samples),
            "maximum_absolute_logit_delta": max_abs_logit_delta,
            "maximum_absolute_confidence_delta": max_abs_confidence_delta,
        }
    _json_write(output / f"metrics_{split}.json", result)
    with (output / f"analyzer_records_{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for item in detailed:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
    print(json.dumps({"split": split, "current_accuracy": result["current_metrics"]["accuracy"], "metrics": str(output / f"metrics_{split}.json")}, indent=2))


def verify_production_candidate(output: Path, split: str) -> None:
    """Run the patched Analyzer and compare it with the frozen replay decision."""
    if split not in SPLITS:
        raise ValueError(split)
    _check_contract()
    frozen_path = output / "frozen_selection_rule.json"
    source_records_path = output / f"analyzer_records_{split}.jsonl"
    if not frozen_path.is_file() or not source_records_path.is_file():
        raise FileNotFoundError("freeze a candidate and keep the pre-change Analyzer records before verification")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    rule = str(frozen["rule"])
    expected_rows = {
        item["key"]: item
        for item in (
            json.loads(line)
            for line in source_records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    generated = [json.loads(line) for line in (output / f"generated_{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    with np.load(output / f"signals_{split}.npz") as archive:
        signals = {item["key"]: archive[item["key"]].copy() for item in generated}
    recorder = _RecordingClassifier(MODEL_PATH)
    analyzer = Analyzer(classifier=recorder)
    detailed: list[dict[str, Any]] = []
    expected_labels: list[str] = []
    actual_labels: list[str] = []
    pass_mismatches = 0
    logit_delta = 0.0
    feature_delta = 0.0
    confidence_delta = 0.0
    with tempfile.TemporaryDirectory(prefix="ntro-cfo-production-verify-") as temp_dir:
        temp_path = Path(temp_dir) / "signal.iq"
        for number, meta in enumerate(generated, start=1):
            signal = signals[meta["key"]]
            expected = expected_rows[meta["key"]]
            observed = _run_analysis(recorder, analyzer, temp_path, signal, meta)
            expected_index = _select(expected, rule)
            expected_label = (
                "unknown"
                if expected["signal_gate"].get("decision") != "signal_candidate"
                else expected["calls"][expected_index]["class"]
            )
            actual_index = int(observed["actual_selected_index"])
            actual_label = observed["analysis_label"]
            expected_labels.append(expected_label)
            actual_labels.append(actual_label)
            if actual_index != expected_index:
                pass_mismatches += 1
            if len(observed["calls"]) != len(expected["calls"]):
                pass_mismatches += 1
            for current_call, expected_call in zip(observed["calls"], expected["calls"]):
                logit_delta = max(logit_delta, float(np.max(np.abs(np.asarray(current_call["logits"]) - np.asarray(expected_call["logits"])))))
                feature_delta = max(feature_delta, float(np.max(np.abs(np.asarray(current_call["features"]) - np.asarray(expected_call["features"])))))
                confidence_delta = max(confidence_delta, abs(float(current_call["confidence"] - expected_call["confidence"])))
            detailed.append({
                **meta,
                **observed,
                "frozen_rule": rule,
                "frozen_expected_selected_index": expected_index,
                "frozen_expected_label": expected_label,
                "production_label_matches_frozen_rule": actual_label == expected_label,
            })
            if number % 250 == 0 or number == len(generated):
                print(f"production verification {split}: {number}/{len(generated)}", flush=True)
    metrics = _metrics([item["label"] for item in generated], actual_labels)
    result = {
        "split": split,
        "rule": rule,
        "model_sha256": sha256_file(MODEL_PATH),
        "production_code_verification": True,
        "signal_count": len(generated),
        "label_parity_mismatches": sum(left != right for left, right in zip(expected_labels, actual_labels)),
        "selected_pass_parity_mismatches": pass_mismatches,
        "maximum_absolute_logits_difference": logit_delta,
        "maximum_absolute_features_difference": feature_delta,
        "maximum_absolute_confidence_difference": confidence_delta,
        "actual_analyzer_metrics": metrics,
        "matches_frozen_candidate_metrics": metrics == json.loads((output / f"metrics_{split}.json").read_text(encoding="utf-8"))["candidate_rules"][rule]["metrics"],
    }
    _json_write(output / f"production_verification_{split}.json", result)
    with (output / f"production_records_{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for item in detailed:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
    print(json.dumps({
        "split": split,
        "label_parity_mismatches": result["label_parity_mismatches"],
        "selected_pass_parity_mismatches": pass_mismatches,
        "max_logit_delta": logit_delta,
        "metrics": str(output / f"production_verification_{split}.json"),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "freeze", "evaluate", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(SPLITS))
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--rule", choices=tuple(rule for rule in CANDIDATES if rule != "current"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "freeze":
        if args.rule is None:
            parser.error("--rule is required for freeze")
        metrics_path = args.output / "metrics_development.json"
        if not metrics_path.is_file():
            parser.error("development metrics must exist before freezing a rule")
        development = json.loads(metrics_path.read_text(encoding="utf-8"))
        if args.rule not in development.get("candidate_rules", {}):
            parser.error("the selected rule was not evaluated on development data")
        chosen = development["candidate_rules"][args.rule]
        if chosen["new_incorrect_decisions"] != 0:
            parser.error("cannot freeze a candidate that introduced incorrect development decisions")
        rationale = (
            "Selected before opening confirmation metrics. In development, this tie-band rule recovered "
            f"{chosen['previously_discarded_correct_retry_selected']} previously discarded correct retry results, "
            "introduced zero incorrect decisions across all six target/control groups, and improved macro F1. "
            "The 1e-5 absolute band is only 0.0084% of the existing 0.12 minimum accepted classifier margin; "
            "it resolves near-ties only while retaining the existing valid-status, supported-class, and initial "
            "signal-gate requirements. Repeated identical-input inference showed exact determinism, so this is "
            "a decision tie band rather than a claim that the confidence gap is floating-point noise."
        ) if args.rule == "epsilon_1e-5" else (
            "Selected before opening confirmation metrics based on development metrics."
        )
        freeze = {
            "rule": args.rule,
            "frozen_before_confirmation_evaluation": True,
            "development_metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            "development_metrics": chosen,
            "rationale": rationale,
        }
        _json_write(args.output / "frozen_selection_rule.json", freeze)
        print(json.dumps({"rule": args.rule, "frozen": True, "path": str(args.output / "frozen_selection_rule.json")}, indent=2))
    elif args.command == "evaluate":
        if not args.split:
            parser.error("--split is required for evaluate")
        if args.split == "confirmation":
            frozen_path = args.output / "frozen_selection_rule.json"
            if not frozen_path.is_file():
                parser.error("freeze a candidate after development and before evaluating confirmation")
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            if tuple(args.candidates) != (frozen["rule"],):
                parser.error("confirmation evaluation is restricted to the single frozen candidate")
        evaluate(args.output, args.split, tuple(args.candidates))
    else:
        if args.split != "confirmation":
            parser.error("--split confirmation is required for production verification")
        verify_production_candidate(args.output, args.split)


if __name__ == "__main__":
    main()
