"""Sequential public-API Tier-2 baseline runner."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime
import scipy

from core.common.models import IQConfig
from core.pipeline.analyzer import Analyzer
from tools.tier2.generate_signals import GENERATOR_VERSION, generate_scenario, load_config


DEFAULT_CONFIG = ROOT / "tools" / "tier2" / "config" / "baseline_scenarios.json"
FORBIDDEN_GENERATOR_IMPORTS = (
    "core.pipeline.demo",
    "core.fec",
    "core.interleaver",
    "tests",
    "scripts",
    "generators",
    "data",
    "models",
)


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_log(path: Path, line: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    output = f"{stamp} {line}"
    print(output, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(output + "\n")


def _audit_generator_imports() -> dict[str, Any]:
    generator = ROOT / "tools" / "tier2" / "generate_signals.py"
    source = generator.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(generator))
    imports: list[str] = []
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        else:
            continue
        imports.extend(imported)
        for module in imported:
            if any(module == blocked or module.startswith(blocked + ".") for blocked in FORBIDDEN_GENERATOR_IMPORTS):
                violations.append(module)
    return {
        "generator_file": str(generator.relative_to(ROOT)),
        "direct_imports": sorted(set(imports)),
        "forbidden_imports": sorted(set(violations)),
        "audit_pass": not violations,
        "scope": "static AST audit of independent generator only; analyzer imports are expected in runner",
    }


def _environment() -> dict[str, Any]:
    model_path = ROOT / "models" / "production_mlp_final.onnx"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path.exists() else None
    try:
        import PyQt6
        pyqt_version = getattr(PyQt6, "__version__", "installed; version recorded by baseline manifest")
    except ImportError:
        pyqt_version = None
    try:
        import gnuradio
        gnuradio_version = getattr(gnuradio, "__version__", "installed; version unavailable")
    except ImportError:
        gnuradio_version = None
    try:
        import komm
        komm_version = getattr(komm, "__version__", "unknown")
    except ImportError:
        komm_version = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_revision": None,
        "repository_revision_note": "No .git metadata is present in the supplied repository checkout.",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "onnxruntime": onnxruntime.__version__,
        "pyqt6": pyqt_version,
        "gnuradio": gnuradio_version,
        "komm": komm_version,
        "model_path": str(model_path.relative_to(ROOT)),
        "model_sha256": model_hash,
        "model_previous_validated_sha256": None,
        "model_continuity": "Unavailable: no prior validated model hash was found in the repository or supplied records.",
        "harness_version": "1.0.0",
        "generator_version": GENERATOR_VERSION,
    }


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _expected_interleaver(value: Any) -> Any:
    """Normalize the scenario spelling to the API's ``None`` representation."""
    return None if value in (None, "None") else value


def _payload_result(result: Any, *, direct_demodulation: bool = False) -> tuple[bool | None, str | None]:
    candidates = [
        result.visualization.get("fec_recovered_bits"),
        result.visualization.get("crc_verified_bits"),
    ]
    if direct_demodulation:
        candidates.append(result.visualization.get("recovered_bits"))
    for candidate in candidates:
        if candidate is not None:
            bits = np.asarray(candidate, dtype=np.uint8).reshape(-1)
            packed = np.packbits(bits, bitorder="big").tobytes()
            return True, hashlib.sha256(packed).hexdigest()
    return None, None


def _compare(scenario: dict[str, Any], gt: dict[str, Any], result: Any) -> dict[str, Any]:
    classification = result.classification or {}
    demod = result.demodulation or {}
    fec = result.fec or {}
    fec_diag = fec.get("diagnostics", {}) if isinstance(fec, dict) else {}
    accepted_candidates = int(fec_diag.get("accepted_candidate_count", 0) or 0)
    predicted_modulation = _enum_value(classification.get("modulation"))
    expected_modulation = {
        "BPSK": "bpsk", "QPSK": "qpsk", "8PSK": "8psk", "16QAM": "16qam", "2FSK": "2fsk"
    }.get(gt["modulation"], "unknown")
    input_gate = bool(result.file_metadata and result.file_metadata.get("sample_count") == gt["sample_count"])
    modulation_gate = bool(
        not gt.get("ood")
        and predicted_modulation == expected_modulation
        and classification.get("status") == "success"
    )
    demod_count = int(demod.get("bit_count", 0) or 0)
    demod_gate = bool(
        not gt.get("ood")
        and demod.get("status") == "complete"
        and demod_count > 0
    )
    if gt.get("fec") == "None":
        fec_gate = True
        fec_status = "NOT_REQUIRED"
        accepted_fec = fec.get("fec_type")
        accepted_interleaver = fec.get("interleaver_type")
    else:
        accepted_fec = fec.get("fec_type")
        accepted_interleaver = fec.get("interleaver_type")
        fec_gate = bool(
            fec.get("status") == "accepted"
            and accepted_fec == gt.get("fec")
            and accepted_interleaver == _expected_interleaver(gt.get("interleaver"))
            and fec.get("crc_pass") is True
        )
        fec_status = str(fec.get("status", "unavailable"))
    payload_match: bool | None = None
    recovered_payload_sha256 = None
    if gt.get("payload_sha256") is not None:
        recovered, recovered_payload_sha256 = _payload_result(
            result, direct_demodulation=gt.get("fec") == "None"
        )
        if recovered is True:
            payload_match = recovered_payload_sha256 == gt["payload_sha256"]
    if gt.get("ood"):
        accepted_recovery = bool(accepted_candidates or fec.get("status") == "accepted")
        explicitly_rejected = bool(
            result.status == "rejected"
            or predicted_modulation in ("unknown", None)
            or classification.get("status") != "success"
        )
        ood_gate = explicitly_rejected and not accepted_recovery and fec.get("crc_pass") is not True
        status = "OOD_PASS" if ood_gate else "OOD_FAIL"
    else:
        required_gates = [input_gate, modulation_gate, demod_gate, fec_gate]
        if payload_match is not None:
            required_gates.append(payload_match)
        if all(required_gates):
            status = "PASS"
        elif result.status == "error" or not input_gate:
            status = "ERROR"
        elif result.status == "rejected":
            status = "REJECTED"
        elif input_gate and modulation_gate and demod_gate:
            status = "PARTIAL"
        else:
            status = "FAIL"
    cfo = (result.synchronization or {}).get("coarse_cfo", {}) or {}
    errors = list(result.errors or [])
    reason = "; ".join(str(item.get("message", item)) for item in errors)
    if gt.get("ood") and status == "OOD_FAIL":
        ood_reasons = []
        if not explicitly_rejected:
            ood_reasons.append(
                f"classifier accepted {predicted_modulation} with confidence {classification.get('confidence')}"
            )
        if accepted_candidates:
            ood_reasons.append(f"analyzer reported {accepted_candidates} accepted trial candidates")
        if fec.get("crc_pass") is True:
            ood_reasons.append("CRC-valid recovery was reported")
        reason = "; ".join(ood_reasons) or "OOD rejection gate failed"
    if not reason and status not in ("PASS", "OOD_PASS"):
        failed = [
            name for name, passed in (
                ("input", input_gate), ("modulation", modulation_gate), ("demodulation", demod_gate),
                ("fec_interleaver_crc", fec_gate), ("payload", payload_match if payload_match is not None else True)
            ) if not passed
        ]
        reason = "failed gates: " + ", ".join(failed)
    return {
        "scenario_id": gt["scenario_id"],
        "scenario_type": "OOD" if gt.get("ood") else "in_scope",
        "status": status,
        "gates": {
            "input": input_gate,
            "modulation": modulation_gate if not gt.get("ood") else None,
            "demodulation": demod_gate if not gt.get("ood") else None,
            "fec_interleaver_crc": fec_gate if not gt.get("ood") else None,
            "payload": payload_match,
            "ood_rejection": ood_gate if gt.get("ood") else None,
        },
        "ground_truth_modulation": gt["modulation"],
        "predicted_modulation": predicted_modulation,
        "modulation_confidence": classification.get("confidence"),
        "classifier_status": classification.get("status"),
        "analyzer_status": result.status,
        "ood_state": {
            "expected_ood": bool(gt.get("ood")),
            "explicitly_rejected": bool(gt.get("ood") and (result.status == "rejected" or predicted_modulation in ("unknown", None) or classification.get("status") != "success")),
            "accepted_candidate_count": accepted_candidates,
            "best_candidate_accepted": fec_diag.get("best_candidate_accepted"),
            "fec_search_status": fec.get("status"),
            "crc_valid_recovery": fec.get("crc_pass") is True,
        },
        "injected_cfo_hz": gt.get("carrier_offset_hz"),
        "estimated_cfo_hz": cfo.get("estimated_offset_hz") if isinstance(cfo, dict) else None,
        "fec_ground_truth": gt.get("fec"),
        "interleaver_ground_truth": gt.get("interleaver"),
        "accepted_fec": accepted_fec,
        "accepted_interleaver": accepted_interleaver,
        "fec_status": fec_status,
        "crc": fec.get("crc_pass"),
        "payload_match": payload_match,
        "payload_sha256_expected": gt.get("payload_sha256"),
        "payload_sha256_recovered": recovered_payload_sha256,
        "demodulated_bit_count": demod_count,
        "failure_reason": reason,
    }


def _summary_rows(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "scenario_id", "ground_truth_modulation", "predicted_modulation", "modulation_confidence",
        "ood_state", "injected_cfo_hz", "estimated_cfo_hz", "fec_ground_truth",
        "interleaver_ground_truth", "accepted_fec", "accepted_interleaver", "crc",
        "payload_match", "status", "failure_reason",
    )
    rows = []
    for item in comparisons:
        row = dict(item)
        row["ood_state"] = json.dumps(item.get("ood_state", {}), sort_keys=True)
        rows.append({key: row.get(key) for key in fields})
    return rows


def _write_summary(reports_root: Path, run_dir: Path, comparisons: list[dict[str, Any]], config: dict[str, Any], environment: dict[str, Any]) -> None:
    counts = {status: sum(row["status"] == status for row in comparisons) for status in ("PASS", "PARTIAL", "FAIL", "REJECTED", "BLOCKED", "ERROR", "OOD_PASS", "OOD_FAIL")}
    in_scope = [row for row in comparisons if row["scenario_type"] == "in_scope"]
    ood = [row for row in comparisons if row["scenario_type"] == "OOD"]
    summary = {
        "run_id": run_dir.name,
        "environment": environment,
        "scenarios_attempted": len(comparisons),
        "counts": counts,
        "in_scope_successful": sum(row["status"] == "PASS" for row in in_scope),
        "in_scope_failed": sum(row["status"] in ("FAIL", "REJECTED", "ERROR", "PARTIAL") for row in in_scope),
        "ood_successful": sum(row["status"] == "OOD_PASS" for row in ood),
        "ood_failed": sum(row["status"] == "OOD_FAIL" for row in ood),
        "scenarios": comparisons,
        "ldpc_extension": {"scenario_id": "T2-11-LDPC", "status": "BLOCKED", "reason": "No independent, verified IEEE 802.11n QC-LDPC encoder/reference implementation was available in the environment; the project decoder/encoder was not used to generate a Tier-2 capture."},
    }
    _json_write(reports_root / "tier2_summary.json", summary)
    rows = _summary_rows(comparisons)
    with (reports_root / "tier2_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else ["scenario_id", "status"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Tier-2 Independent Validation Baseline",
        "",
        f"Run: `{run_dir.name}`",
        f"Scenarios attempted: {len(comparisons)}",
        "",
        "No single accuracy percentage is reported; this is a fixed, small scenario set.",
        "",
        "| Scenario | Ground truth | Predicted | Confidence | CFO in / estimated | FEC / interleaver expected | Accepted | CRC | Payload | Status | Failure reason |",
        "|---|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['scenario_id']} | {item['ground_truth_modulation']} | {item['predicted_modulation']} | {item['modulation_confidence']} | {item['injected_cfo_hz']} / {item['estimated_cfo_hz']} | {item['fec_ground_truth']} / {item['interleaver_ground_truth']} | {item['accepted_fec']} / {item['accepted_interleaver']} | {item['crc']} | {item['payload_match']} | {item['status']} | {item['failure_reason']} |"
        )
    lines += ["", "## Aggregate counts", "", f"- Scenarios attempted: {len(comparisons)}"]
    lines += [f"- {name}: {value}" for name, value in counts.items()]
    lines += [
        f"- In-scope successful: {summary['in_scope_successful']}",
        f"- In-scope failed: {summary['in_scope_failed']}",
        f"- OOD successful: {summary['ood_successful']}",
        f"- OOD failed: {summary['ood_failed']}",
        "",
        "## Optional LDPC extension",
        "",
        "T2-11-LDPC: BLOCKED. No independent verified encoder was available; no project LDPC generation code was used.",
        "",
    ]
    (reports_root / "tier2_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, default=ROOT / "reports" / "tier2" / "baseline")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    reports_root = args.reports_root.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty Tier-2 run directory: {run_dir}")
    existing_reports = [
        reports_root / name
        for name in ("tier2_summary.json", "tier2_summary.csv", "tier2_report.md")
        if (reports_root / name).exists()
    ]
    if existing_reports:
        raise SystemExit(
            "Refusing to overwrite existing Tier-2 aggregate reports: "
            + ", ".join(str(path) for path in existing_reports)
        )
    reports_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    log_path = run_dir / "run.log"
    environment = _environment()
    audit = _audit_generator_imports()
    frozen_manifest = ROOT / "reports" / "tier2" / "manifests" / f"{run_dir.name}_environment.json"
    baseline_manifest = json.loads(frozen_manifest.read_text(encoding="utf-8")) if frozen_manifest.exists() else None
    run_manifest = {
        "run_id": run_dir.name,
        "environment": environment,
        "pre_harness_snapshot": baseline_manifest,
        "independence_audit": audit,
        "scenario_config": str(args.config.resolve()),
        "generator_import_list": audit["direct_imports"],
        "invocation": "Analyzer.analyze_file(...) for each capture; sequential execution",
        "gnu_radio_available": bool(environment["gnuradio"]),
    }
    _json_write(run_dir / "run_manifest.json", run_manifest)
    _write_log(log_path, f"Tier-2 run {run_dir.name} started; scenario_count={len(config['scenarios'])}")
    _write_log(log_path, f"generator_import_audit={json.dumps(audit, sort_keys=True)}")
    if not audit["audit_pass"]:
        _write_log(log_path, "LEAKAGE AUDIT FAILED; continuing captures while marking the overall validation compromised")
    analyzer = Analyzer()
    comparisons: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        scenario_dir = run_dir / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        _write_log(log_path, f"{scenario_id} generation started")
        try:
            ground_truth = generate_scenario(scenario, config["sample_rates_hz"], scenario_dir)
            _write_log(scenario_dir / "console.log", f"Generated {ground_truth['file_path']} ({ground_truth['sample_count']} samples, {ground_truth['file_format']})")
            input_path = scenario_dir / ground_truth["file_path"]
            analyzer_config: dict[str, Any] = {
                "crc_present": bool(ground_truth["analyzer_configuration_allowed"]["crc_present"]),
                "run_fec": bool(ground_truth["analyzer_configuration_allowed"]["run_fec"]),
            }
            if ground_truth["file_format"] == "complex64_le_iq":
                analyzer_config["iq_config"] = IQConfig(sample_rate=float(ground_truth["sample_rate_hz"]))
            else:
                analyzer_config["treat_stereo_as_iq"] = True
            if ground_truth.get("samples_per_symbol") is not None:
                analyzer_config["samples_per_symbol"] = float(ground_truth["samples_per_symbol"])
            if ground_truth.get("tone_frequencies_hz"):
                analyzer_config["tone_frequencies_hz"] = tuple(ground_truth["tone_frequencies_hz"])
            if scenario_id in {"T2-09", "T2-10"}:
                # Keep OOD captures bounded while still exercising the default downstream search.
                analyzer_config["run_fec"] = True
            _write_log(log_path, f"{scenario_id} Analyzer.analyze_file started with keys={sorted(analyzer_config)}")
            start = time.perf_counter()
            def progress(message: str, percent: int) -> None:
                _write_log(scenario_dir / "console.log", f"progress {percent}% {message}")
            result = analyzer.analyze_file(input_path, progress_callback=progress, **analyzer_config)
            elapsed = time.perf_counter() - start
            result_dict = result.to_dict(include_arrays=True)
            _json_write(scenario_dir / "analysis_result.json", result_dict)
            comparison = _compare(scenario, ground_truth, result)
            comparison["elapsed_seconds"] = elapsed
            comparison["analysis_result_file"] = "analysis_result.json"
            _json_write(scenario_dir / "comparison.json", comparison)
            comparisons.append(comparison)
            _write_log(log_path, f"{scenario_id} complete status={comparison['status']} analyzer_status={result.status} elapsed_seconds={elapsed:.3f}")
        except BaseException as exc:
            error = {
                "scenario_id": scenario_id,
                "scenario_type": "OOD" if scenario.get("ood") else "in_scope",
                "status": "ERROR",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "gates": {},
                "ground_truth_modulation": scenario.get("modulation"),
                "predicted_modulation": None,
                "modulation_confidence": None,
                "ood_state": {"expected_ood": bool(scenario.get("ood")), "accepted_candidate_count": None, "crc_valid_recovery": None},
                "injected_cfo_hz": scenario.get("cfo_hz"),
                "estimated_cfo_hz": None,
                "fec_ground_truth": scenario.get("fec"),
                "interleaver_ground_truth": scenario.get("interleaver"),
                "accepted_fec": None,
                "accepted_interleaver": None,
                "crc": None,
                "payload_match": None,
            }
            _json_write(scenario_dir / "comparison.json", error)
            with (scenario_dir / "console.log").open("a", encoding="utf-8") as stream:
                stream.write(error["traceback"])
            comparisons.append(error)
            _write_log(log_path, f"{scenario_id} ERROR {error['failure_reason']}; continuing")
    if audit["audit_pass"] is False:
        for row in comparisons:
            row.setdefault("validation_notes", []).append("generator leakage audit failed")
    _write_summary(reports_root, run_dir, comparisons, config, environment)
    _json_write(run_dir / "run_manifest.json", {**run_manifest, "finished_utc": datetime.now(timezone.utc).isoformat(), "scenario_results": len(comparisons)})
    _write_log(log_path, f"Tier-2 run finished scenarios={len(comparisons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
