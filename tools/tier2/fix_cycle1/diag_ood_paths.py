"""Local historical diagnostics; not part of production inference or its OOD gate."""

from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort

from core.dsp.energy_detector import detect_activity
from core.dsp.modulation_features import FEATURE_NAMES, extract_modulation_features
from tools.tier2.fix_cycle1.diag_common import ROOT, MODEL, load_capture, load_gt, scenario_dir

contract = json.loads((ROOT / "production_ml_contract.json").read_text(encoding="utf-8"))
TEMP = contract["calibration"]["temperature"]
Q_THR = 0.618044223  # Historical diagnostic-only threshold; production does not use it.
CLASSES = contract["class_order"]
sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
IDX_PSK4 = FEATURE_NAMES.index("psk4_residual")


def logits(x):
    f = extract_modulation_features(x).values.astype(np.float32)[None, :]
    return sess.run(None, {"features": f})[0][0].astype(np.float64), f[0]


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


out = {"energy_detector": {}, "softmax_entropy": {}}
for sid in [f"T2-{i:02d}" for i in range(1, 11) if i != 6]:
    x = load_capture(sid).astype(np.complex128)
    det = detect_activity(x)
    out["energy_detector"][sid] = {
        "activity_detected": bool(det.detected),
        "warnings": list(det.warnings),
        "estimated_snr_db": det.estimated_snr,
    }
for sid in ("T2-09", "T2-10", "T2-04", "T2-05"):
    x = load_capture(sid).astype(np.complex128)
    z, f = logits(x)
    p1, pT = softmax(z), softmax(z / TEMP)
    ent = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(p)))
    out["softmax_entropy"][sid] = {
        "logits": dict(zip(CLASSES, np.round(z, 2).tolist())),
        "softmax_T1": dict(zip(CLASSES, np.round(p1, 6).tolist())),
        f"softmax_T{TEMP}": dict(zip(CLASSES, np.round(pT, 6).tolist())),
        "normalized_entropy_T1": round(ent(p1), 6),
        "normalized_entropy_Tcal": round(ent(pT), 6),
        "psk4_residual": round(float(f[IDX_PSK4]), 4),
    }

# Fresh-seed stability of the QAM16/PSK4 gate on Gaussian noise and generic OFDM (NOT Tier-2 captures).
rng = np.random.default_rng(20261003)
rows = {"noise": [], "ofdm": []}
for _ in range(300):
    x = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)) / np.sqrt(2)
    z, f = logits(x)
    rows["noise"].append((CLASSES[int(np.argmax(z))], float(f[IDX_PSK4])))
for _ in range(100):
    nfft, cp, act, nsym = int(rng.choice([64, 128, 256])), int(rng.choice([8, 16, 32])), 0, int(rng.integers(10, 40))
    act = int(nfft * rng.choice([0.5, 0.75, 0.8]))
    pos = act // 2
    idx = np.r_[np.arange(1, pos + 1), np.arange(nfft - pos, nfft)]
    sig = []
    for _s in range(nsym):
        b = np.zeros(nfft, complex)
        b[idx] = np.exp(1j * (np.pi / 4 + rng.integers(0, 4, idx.size) * np.pi / 2))
        t = np.fft.ifft(b) * np.sqrt(nfft)
        sig.append(np.r_[t[-cp:], t])
    x = np.concatenate(sig)
    snr = rng.uniform(5, 30)
    x = x + np.sqrt(np.mean(abs(x) ** 2) / 10 ** (snr / 10) / 2) * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    z, f = logits(x)
    rows["ofdm"].append((CLASSES[int(np.argmax(z))], float(f[IDX_PSK4])))
for k, v in rows.items():
    classes = [c for c, _ in v]
    psk4 = np.array([p for _, p in v])
    pred_q16 = np.array([c == "QAM16" for c in classes])
    out[f"fresh_{k}"] = {
        "n": len(v),
        "predicted_class_counts": {c: int(classes.count(c)) for c in sorted(set(classes))},
        "psk4_residual_mean_sd_min_max": [round(float(psk4.mean()), 4), round(float(psk4.std()), 4), round(float(psk4.min()), 4), round(float(psk4.max()), 4)],
        "predicted_QAM16_and_gate_rejects": int(np.sum(pred_q16 & (psk4 > Q_THR))),
        "predicted_QAM16_and_gate_passes": int(np.sum(pred_q16 & (psk4 <= Q_THR))),
        "predicted_non_QAM16_unaffected_by_gate": int(np.sum(~pred_q16)),
    }

# T2-05 WAV clipping check
gt = load_gt("T2-05")
from scipy.io import wavfile

sr, pcm = wavfile.read(str(scenario_dir("T2-05") / "capture.wav"))
out["T2-05_wav"] = {
    "gain": gt["gain"], "sample_rate": int(sr), "dtype": str(pcm.dtype), "shape": list(pcm.shape),
    "pcm_min": int(pcm.min()), "pcm_max": int(pcm.max()),
    "fraction_at_positive_full_scale": float(np.mean(pcm == 32767)),
    "fraction_at_negative_full_scale": float(np.mean(pcm == -32768)),
}
print(json.dumps(out, indent=1))
