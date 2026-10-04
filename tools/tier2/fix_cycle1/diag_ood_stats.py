"""Evaluate FROZEN OOD gate statistics on the saved Tier-2 captures (read-only; no tuning)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from core.dsp.modulation_features import FEATURE_NAMES, extract_modulation_features
from core.sync.cfo_estimator import estimate_cfo
from tools.tier2.fix_cycle1.diag_common import load_capture, load_gt

BACKUP = Path(os.environ["USERPROFILE"]) / "Downloads" / "NTRO_26147_COMPLETE_BACKUP_20260929_201029"
contract = json.loads((BACKUP / "final_ml_model" / "production_ml_contract.json").read_text())
mean = np.array(contract["preprocessing"]["mean"])
scale = np.array(contract["preprocessing"]["scale"])
tr = np.load(BACKUP / "final_ml_dataset" / "train_features.npz", allow_pickle=True)
Z = (tr["X"].astype(np.float64) - mean) / scale
MU = Z.mean(0)
PREC = np.linalg.pinv(np.cov(Z, rowvar=False))
G_THR = contract["ood_gate"]["global_mahalanobis_threshold"]
Q_THR = contract["ood_gate"]["qam16_psk4_residual_threshold"]


def d2(raw_features: np.ndarray) -> float:
    c = (raw_features - mean) / scale - MU
    return float(c @ PREC @ c)


def stats(x):
    f = extract_modulation_features(x)
    v = f.values
    c = (v - mean) / scale
    z = np.abs(c)
    top = np.argsort(-z)[:3]
    return {
        "d2": round(d2(v), 2),
        "gate_global_reject": bool(d2(v) > G_THR),
        "psk4_residual": round(float(v[FEATURE_NAMES.index("psk4_residual")]), 3),
        "worst_z_features": {FEATURE_NAMES[i]: round(float(c[i]), 1) for i in top},
    }


def main(ids):
    out = {}
    for sid in ids:
        gt = load_gt(sid)
        x = load_capture(sid).astype(np.complex128)
        fs = gt["sample_rate_hz"]
        n = np.arange(x.size)
        rec = {"gt": gt["modulation"], "as_analyzed(raw)": stats(x)}
        if gt["samples_per_symbol"] == 1:
            cands = {}
            for order in (2, 4, 8):
                e = estimate_cfo(x, fs, order)
                if e.estimated_offset_hz is not None:
                    cands[order] = (e.estimated_offset_hz, e.confidence)
            if cands:
                order = min(cands)
                rec["cfo_corrected(lowest coherent order %d)" % order] = stats(x * np.exp(-2j * np.pi * cands[order][0] * n / fs))
            rec["cfo_candidates"] = {k: [round(v[0], 2), round(v[1], 3)] for k, v in cands.items()}
        out[sid] = rec
    return out


if __name__ == "__main__":
    ids = sys.argv[1:] or [f"T2-{i:02d}" for i in range(1, 11) if i != 6]
    print(json.dumps(main(ids), indent=1))
    print(f"frozen thresholds: global d2 > {G_THR} ; qam16/psk4 residual {Q_THR}")
