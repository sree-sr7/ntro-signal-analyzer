"""Probe which train/validation statistic reproduces the frozen global Mahalanobis threshold.

Uses only pre-existing training/validation features (never the reserved final test split).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

BACKUP = Path(os.environ["USERPROFILE"]) / "Downloads" / "NTRO_26147_COMPLETE_BACKUP_20260929_201029"
contract = json.loads((BACKUP / "final_ml_model" / "production_ml_contract.json").read_text())
mean = np.array(contract["preprocessing"]["mean"])
scale = np.array(contract["preprocessing"]["scale"])
THR = contract["ood_gate"]["global_mahalanobis_threshold"]
classes = contract["model"]["class_order"]
tr = np.load(BACKUP / "final_ml_dataset" / "train_features.npz", allow_pickle=True)
va = np.load(BACKUP / "final_ml_dataset" / "validation_features.npz", allow_pickle=True)
Xtr, ytr = tr["X"].astype(np.float64), tr["y"]
Xva, yva = va["X"].astype(np.float64), va["y"]


def d2(x, mu, prec):
    c = x - mu
    return np.einsum("ij,jk,ik->i", c, prec, c)


def inv_pct(values, thr):
    return 100.0 * float(np.mean(values <= thr))


Ztr, Zva = (Xtr - mean) / scale, (Xva - mean) / scale
variants = {}
mu = Ztr.mean(0)
cov = np.cov(Ztr, rowvar=False)
for tag, reg in (("pinv", 0.0), ("ridge1e-6", 1e-6), ("ridge1e-3", 1e-3), ("ridge1e-2", 1e-2)):
    prec = np.linalg.pinv(cov + reg * np.eye(17)) if reg else np.linalg.pinv(cov)
    variants[f"global/train-fit/{tag}"] = (d2(Ztr, mu, prec), d2(Zva, mu, prec))
# fit on train+val
Zall = np.vstack([Ztr, Zva])
mu_a, prec_a = Zall.mean(0), np.linalg.pinv(np.cov(Zall, rowvar=False))
variants["global/train+val-fit/pinv"] = (d2(Ztr, mu_a, prec_a), d2(Zva, mu_a, prec_a))
# diag-only
var = Ztr.var(0)
variants["global/diag-cov"] = (np.sum((Ztr - mu) ** 2 / var, 1), np.sum((Zva - mu) ** 2 / var, 1))
# class-conditional min with ridge
for reg in (1e-6, 1e-3, 1e-2):
    dt, dv = [], []
    for c in classes:
        Zc = Ztr[ytr == c]
        pc = np.linalg.pinv(np.cov(Zc, rowvar=False) + reg * np.eye(17))
        dt.append(d2(Ztr, Zc.mean(0), pc))
        dv.append(d2(Zva, Zc.mean(0), pc))
    variants[f"min-class/ridge{reg}"] = (np.min(dt, 0), np.min(dv, 0))
# own-class distance (distance to the true class Gaussian)
dt = np.zeros(len(Ztr)); dv = np.zeros(len(Zva))
for c in classes:
    Zc = Ztr[ytr == c]
    pc = np.linalg.pinv(np.cov(Zc, rowvar=False))
    dt[ytr == c] = d2(Ztr[ytr == c], Zc.mean(0), pc)
    dv[yva == c] = d2(Zva[yva == c], Zc.mean(0), pc)
variants["own-class/pinv"] = (dt, dv)

print(f"threshold = {THR}")
for name, (a, b) in variants.items():
    print(f"{name:34s} inverse-percentile: train={inv_pct(a, THR):8.4f}  val={inv_pct(b, THR):8.4f}   "
          f"max train={a.max():9.2f} val={b.max():9.2f}   p99.9 val={np.percentile(b, 99.9):8.2f}")
