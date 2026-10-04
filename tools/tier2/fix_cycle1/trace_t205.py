"""T2-05 trace: independent IQ -> WAV -> analyzer IQ -> symbols -> bits (read-only on the baseline)."""

from __future__ import annotations

import json

import numpy as np

from core.file_io.wav_loader import load_wav
from tools.tier2.fix_cycle1.diag_common import BASELINE, load_gt, payload_bits, scenario_dir

sid = "T2-05"
gt = load_gt(sid)
bits = payload_bits(sid)
res = json.loads((BASELINE / sid / "analysis_result.json").read_text(encoding="utf-8"))

# --- independent re-creation of the *noiseless* transmitted symbols from the saved payload ---
groups = bits.reshape(-1, 2)
idx_by_label = {0: 0, 1: 1, 3: 2, 2: 3}
sym_idx = np.array([idx_by_label[int(a) * 2 + int(b)] for a, b in groups])
tx = np.exp(1j * (np.pi / 4 + sym_idx * np.pi / 2))
n = np.arange(tx.size)
ideal = tx * gt["gain"] * np.exp(1j * (gt["initial_phase_rad"] + 2 * np.pi * gt["carrier_offset_hz"] * n / gt["sample_rate_hz"]))

# --- WAV representation vs analyzer-loaded IQ ---
loaded = load_wav(scenario_dir(sid) / "capture.wav", treat_stereo_as_iq=True)
x = np.asarray(loaded.signal.samples)
from scipy.io import wavfile

_, pcm = wavfile.read(str(scenario_dir(sid) / "capture.wav"))
manual = pcm[:, 0].astype(np.float64) / 32768 + 1j * pcm[:, 1].astype(np.float64) / 32768
out = {
    "loader": {
        "sample_count": int(x.size), "sample_rate": loaded.signal.metadata.sample_rate,
        "warnings": list(loaded.warnings),
        "max_abs_diff_vs_manual_I=ch0,Q=ch1": float(np.max(np.abs(x - manual))),
        "max_abs_diff_vs_swapped_IQ": float(np.max(np.abs(x - (manual.imag + 1j * manual.real)))),
    }
}

# --- quantify WAV conversion distortion: generator's pre-clip signal is unknown (noise), but clip loss is measurable ---
noise_free_peak = gt["gain"] * np.maximum(np.abs(ideal.real), np.abs(ideal.imag))
out["wav_conversion"] = {
    "gain": gt["gain"], "initial_phase_rad": gt["initial_phase_rad"],
    "noise_free_peak_component_amplitude_max": float(noise_free_peak.max() / gt["gain"] * 1.0),
    "noise_free_component_abs_levels": sorted({round(float(v), 3) for v in np.abs(np.r_[ideal.real, ideal.imag])})[:8],
    "fraction_noise_free_components_beyond_PCM_range": float(np.mean(np.abs(np.r_[ideal.real, ideal.imag]) > 32767 / 32768)),
    "fraction_pcm_samples_at_full_scale": float(np.mean(np.abs(pcm) >= 32767)),
}
clipped = np.clip(np.c_[ideal.real, ideal.imag], -1, 32767 / 32768)
out["wav_conversion"]["noise_free_clip_error_rms_vs_unclipped"] = float(np.sqrt(np.mean((clipped - np.c_[ideal.real, ideal.imag]) ** 2)))
out["wav_conversion"]["noise_free_clip_error_rms_relative_to_signal_rms"] = float(
    out["wav_conversion"]["noise_free_clip_error_rms_vs_unclipped"] / np.sqrt(np.mean(np.abs(ideal) ** 2) / 2))
# angular error of clipped constellation points relative to the unclipped ones
cl = clipped[:, 0] + 1j * clipped[:, 1]
ang_err = np.angle(cl * np.conj(ideal))
out["wav_conversion"]["clip_angle_error_deg_max_noise_free"] = float(np.degrees(np.max(np.abs(ang_err))))

# --- analyzer demod bits vs ground truth: all 4 phase rotations + mapping variants ---
rec_bits = np.asarray(res["visualization"]["recovered_bits"], dtype=np.uint8)
out["analyzer"] = {
    "demod_bit_count": int(rec_bits.size), "payload_bit_count": int(bits.size),
    "carrier_recovery": res["synchronization"]["carrier_recovery"],
    "coarse_cfo": res["synchronization"]["coarse_cfo"],
    "direct_bit_error_rate": float(np.mean(rec_bits[: bits.size] != bits)),
}
rec_syms = np.asarray(res["visualization"]["constellation_i"]) + 1j * np.asarray(res["visualization"]["constellation_q"])
rec_idx = np.rint(((np.angle(rec_syms) - np.pi / 4) / (np.pi / 2))).astype(int) % 4
inv = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
ber_by_rot = {}
sym_ber_by_rot = {}
for r in range(4):
    shifted = (rec_idx - r) % 4
    rb = np.array([inv[int(i)] for i in shifted], dtype=np.uint8).reshape(-1)
    ber_by_rot[f"rotate_back_{r}x90deg"] = float(np.mean(rb != bits))
    sym_ber_by_rot[f"rotate_back_{r}x90deg"] = float(np.mean(shifted != sym_idx))
out["analyzer"]["bit_error_rate_after_undoing_k_quarter_turns"] = ber_by_rot
out["analyzer"]["symbol_error_rate_after_undoing_k_quarter_turns"] = sym_ber_by_rot
best = min(sym_ber_by_rot, key=sym_ber_by_rot.get)
out["analyzer"]["best_rotation"] = best
# where do the residual errors sit? (clipped samples?)
r = int(best.split("_")[2][0])
shifted = (rec_idx - r) % 4
err = shifted != sym_idx
comp_clipped = (np.abs(pcm[:, 0]) >= 32767) | (np.abs(pcm[:, 1]) >= 32767)
out["analyzer"]["residual_errors_at_best_rotation"] = {
    "count": int(err.sum()),
    "errors_among_symbols_with_a_clipped_component": int((err & comp_clipped).sum()),
    "symbols_with_a_clipped_component": int(comp_clipped.sum()),
    "error_rate_when_clipped": float(err[comp_clipped].mean()) if comp_clipped.any() else None,
    "error_rate_when_not_clipped": float(err[~comp_clipped].mean()),
}
print(json.dumps(out, indent=1, default=str))
