"""T2-03 trace: independent bits -> waveform -> analyzer demod -> deinterleave -> Viterbi -> CRC.

Read-only against the frozen baseline. Uses the independent generator's *functions* only to
re-derive the true intermediate arrays from the saved payload bits.
"""

from __future__ import annotations

import json

import numpy as np

from core.bitstream.crc import verify_crc16
from core.fec.convolutional import decode_viterbi
from core.interleaver.convolutional import deinterleave_convolutional
from tools.tier2.fix_cycle1.diag_common import BASELINE, load_gt, payload_bits
from tools.tier2 import generate_signals as g

sid = "T2-03"
gt = load_gt(sid)
payload = payload_bits(sid)
res = json.loads((BASELINE / sid / "analysis_result.json").read_text(encoding="utf-8"))

frame = g._append_crc(payload)  # independent CRC-16/CCITT-FALSE
coded = g._conv_encode(frame, 7)  # komm K7
pre_mod = g._conv_interleave(coded, 12)  # independent conv interleaver, depth 12, 4 branches
out = {
    "sizes": {"payload": int(payload.size), "frame+crc": int(frame.size), "coded": int(coded.size),
              "encoded_modulation_input": int(pre_mod.size), "gt_encoded_bits": gt["encoded_modulation_input_bit_count"]},
    "project_crc_accepts_independent_frame": bool(verify_crc16(frame)),
}

# --- project's receive chain on the TRUE transmitted bits (isolates deinterleaver / Viterbi / CRC) ---
di = deinterleave_convolutional(pre_mod, depth=12, branches=4)
out["project_deinterleaver_inverts_independent_interleaver"] = bool(np.array_equal(di, coded))
vit = decode_viterbi(di, "Conv_R12_K7", terminated=True)
out["project_viterbi_on_true_coded_bits"] = {
    "valid": bool(vit.valid), "decoded_equals_frame": bool(np.array_equal(vit.decoded_bits, frame)),
    "path_metric": int(vit.path_metric),
}
out["project_crc_on_decoded_frame"] = bool(verify_crc16(vit.decoded_bits))

# --- analyzer's demodulated bits vs truth, for all 8 PSK phase rotations ---
rec_bits = np.asarray(res["visualization"]["recovered_bits"], dtype=np.uint8)
syms = np.asarray(res["visualization"]["constellation_i"]) + 1j * np.asarray(res["visualization"]["constellation_q"])
labels = ("000", "001", "011", "010", "110", "111", "101", "100")
lab_bits = np.array([[int(c) for c in s] for s in labels], dtype=np.uint8)
rec_idx = np.rint(np.angle(syms) / (2 * np.pi / 8)).astype(int) % 8
out["analyzer_demod"] = {"bit_count": int(rec_bits.size), "carrier_recovery": res["synchronization"]["carrier_recovery"],
                         "direct_BER_vs_true_pre_mod_bits": float(np.mean(rec_bits != pre_mod[: rec_bits.size]))}
by_rot = {}
best = None
for k in range(8):
    b = lab_bits[(rec_idx - k) % 8].reshape(-1)
    ber = float(np.mean(b != pre_mod[: b.size]))
    by_rot[f"undo_{k}x45deg"] = round(ber, 5)
    if best is None or ber < best[1]:
        best = (k, ber, b)
out["analyzer_demod"]["BER_after_undoing_k_eighth_turns"] = by_rot
out["analyzer_demod"]["best_rotation_k"] = best[0]

# --- full project chain from the best-rotation demod bits ---
b = best[2]
di2 = deinterleave_convolutional(b, depth=12, branches=4)
vit2 = decode_viterbi(di2, "Conv_R12_K7", terminated=True)
out["chain_with_best_rotation"] = {
    "deinterleaved_BER_vs_coded": float(np.mean(di2 != coded)),
    "viterbi_decoded_equals_frame": bool(np.array_equal(vit2.decoded_bits, frame)),
    "crc_pass": bool(verify_crc16(vit2.decoded_bits)),
    "payload_equals_ground_truth": bool(np.array_equal(vit2.decoded_bits[:-16], payload)),
}
# search log from the baseline: what did the trial engine see?
cands = res["fec"]["candidates"]
out["baseline_trial_summary"] = {
    "n_candidates": len(cands),
    "accepted": sum(1 for c in cands if c["accepted"]),
    "gt_candidate_entry": next(({k: c[k] for k in ("fec_type", "interleaver_type", "valid", "crc_pass", "path_metric", "candidate_status")}
                                for c in cands if c["fec_type"] == "Conv_R12_K7" and c["interleaver_type"] == "Convolutional_Depth_12"), None),
}
print(json.dumps(out, indent=1))
