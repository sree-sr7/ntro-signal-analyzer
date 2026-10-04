"""Identify the frozen OOD-gate definitions from pre-existing train/validation material only.

The final test split is deliberately never loaded (the contract records that it is untouched).
No Tier-2 data is used here.
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
G_THR = contract["ood_gate"]["global_mahalanobis_threshold"]
Q_THR = contract["ood_gate"]["qam16_psk4_residual_threshold"]
names = contract["features"]
classes = contract["model"]["class_order"]

tr = np.load(BACKUP / "final_ml_dataset" / "train_features.npz", allow_pickle=True)
va = np.load(BACKUP / "final_ml_dataset" / "validation_features.npz", allow_pickle=True)
Xtr, ytr = tr["X"].astype(np.float64), tr["y"]
Xva, yva = va["X"].astype(np.float64), va["y"]
print("label set", sorted(set(ytr)))


def d2(x, mu, prec):
    c = x - mu
    return np.einsum("ij,jk,ik->i", c, prec, c)


def report(tag, dtr, dva):
    qs = [50, 90, 99, 99.9, 100]
    print(f"{tag:46s} train pct {[round(float(np.percentile(dtr, q)), 2) for q in qs]}  val pct {[round(float(np.percentile(dva, q)), 2) for q in qs]}")


for space in ("raw", "scaled"):
    A = Xtr if space == "raw" else (Xtr - mean) / scale
    B = Xva if space == "raw" else (Xva - mean) / scale
    mu = A.mean(0)
    cov = np.cov(A, rowvar=False)
    prec = np.linalg.pinv(cov)
    report(f"GLOBAL pooled fit, {space}, d^2", d2(A, mu, prec), d2(B, mu, prec))
    report(f"GLOBAL pooled fit, {space}, d (sqrt)", np.sqrt(d2(A, mu, prec)), np.sqrt(d2(B, mu, prec)))
    # class-conditional: minimum over class Gaussians
    mus, precs = {}, {}
    for c in classes:
        sel = ytr == c
        if sel.sum() == 0:
            sel = ytr == c.replace("PSK8", "8PSK")
        Ac = A[sel]
        mus[c] = Ac.mean(0)
        precs[c] = np.linalg.pinv(np.cov(Ac, rowvar=False))
    dmin_tr = np.min([d2(A, mus[c], precs[c]) for c in classes], axis=0)
    dmin_va = np.min([d2(B, mus[c], precs[c]) for c in classes], axis=0)
    report(f"MIN class-conditional, {space}, d^2", dmin_tr, dmin_va)
    report(f"MIN class-conditional, {space}, d (sqrt)", np.sqrt(dmin_tr), np.sqrt(dmin_va))
    # shared (pooled within-class) covariance, min over class means
    within = sum(np.cov(A[ytr == c], rowvar=False) * (np.sum(ytr == c) - 1) for c in classes) / (len(A) - len(classes))
    pw = np.linalg.pinv(within)
    dsh_tr = np.min([d2(A, mus[c], pw) for c in classes], axis=0)
    dsh_va = np.min([d2(B, mus[c], pw) for c in classes], axis=0)
    report(f"MIN shared-within-cov, {space}, d^2", dsh_tr, dsh_va)

print(f"\nFrozen thresholds: global_mahalanobis={G_THR}, qam16_psk4_residual={Q_THR}")
i_ps4 = names.index("psk4_residual")
i_q16 = names.index("qam16_residual")
for c in classes:
    for src, X, y in (("train", Xtr, ytr), ("val", Xva, yva)):
        v = X[y == c]
        if v.size:
            print(f"{c:6s} {src:5s} n={len(v):5d} psk4_residual pct[min,50,99,99.9,max]={[round(float(np.percentile(v[:, i_ps4], q)), 3) for q in (0, 50, 99, 99.9, 100)]}"
                  f" qam16_residual max={v[:, i_q16].max():.3f}")
