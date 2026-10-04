"""Diagnose classifier inputs/outputs on the saved Tier-2 captures (read-only)."""

from __future__ import annotations

import json
import sys

import numpy as np

from tools.tier2.fix_cycle1.diag_common import CYCLE1, MODEL, load_capture, load_gt
from core.dsp.modulation_features import FEATURE_NAMES, extract_modulation_features
from core.dsp.energy_detector import detect_activity
from core.ml.classifier import OnnxModulationClassifier
from core.sync.cfo_estimator import estimate_cfo

CLS = OnnxModulationClassifier(MODEL)


def run(samples: np.ndarray, sps: float = 1.0) -> dict:
    res = CLS.classify(samples, samples_per_symbol=sps)
    feats = extract_modulation_features(samples, samples_per_symbol=sps)
    return {
        "mod": res.modulation.value,
        "conf": round(float(res.confidence), 4),
        "probs": {k.value: round(float(v), 4) for k, v in res.probabilities.items()},
        "features": dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in feats.values])),
    }


def main(ids):
    out = {}
    for sid in ids:
        gt = load_gt(sid)
        x = load_capture(sid).astype(np.complex128)
        fs = gt["sample_rate_hz"]
        cfo = gt["carrier_offset_hz"]
        n = np.arange(x.size)
        rec = {
            "gt_mod": gt["modulation"], "snr_db": gt["snr_db"], "cfo_hz": cfo, "n": int(x.size),
            "raw_rms": float(np.sqrt(np.mean(abs(x) ** 2))),
            "raw_amp_cv": float(np.std(abs(x)) / np.mean(abs(x))),
            "cycles_of_cfo_rotation_over_capture": float(cfo * x.size / fs),
        }
        rec["as_analyzed(raw, CFO uncorrected)"] = run(x)
        rec["DIAG_oracle_cfo_removed"] = run(x * np.exp(-2j * np.pi * cfo * n / fs))
        for order in (2, 4, 8):
            e = estimate_cfo(x, fs, order)
            rec[f"est_cfo_order{order}"] = {"hz": e.estimated_offset_hz, "conf": round(float(e.confidence), 3)}
            if e.estimated_offset_hz is not None:
                rec[f"DIAG_est_cfo_order{order}_removed"] = run(x * np.exp(-2j * np.pi * e.estimated_offset_hz * n / fs))
        out[sid] = rec
    return out


if __name__ == "__main__":
    ids = sys.argv[1:] or ["T2-01", "T2-02", "T2-03", "T2-05", "T2-07", "T2-08"]
    result = main(ids)
    print(json.dumps(result, indent=1, default=str))
