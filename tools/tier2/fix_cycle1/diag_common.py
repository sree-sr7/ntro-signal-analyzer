"""Shared helpers for Tier-2 Fix Cycle 1 diagnostics (read-only on the frozen baseline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE = ROOT / "reports" / "tier2" / "baseline" / "20260930_201445_269"
CYCLE1 = ROOT / "reports" / "tier2" / "fixes" / "cycle1"
MODEL = ROOT / "models" / "production_mlp_final.onnx"


def scenario_dir(sid: str) -> Path:
    return BASELINE / sid


def load_gt(sid: str) -> dict:
    return json.loads((scenario_dir(sid) / "ground_truth.json").read_text(encoding="utf-8"))


def load_capture(sid: str) -> np.ndarray:
    """Load the saved capture exactly as written by the independent generator."""
    gt = load_gt(sid)
    path = scenario_dir(sid) / gt["file_path"]
    if gt["file_format"] == "complex64_le_iq":
        return np.fromfile(path, dtype="<c8")
    from scipy.io import wavfile

    _, pcm = wavfile.read(str(path))
    return pcm[:, 0].astype(np.float32) / 32768.0 + 1j * (pcm[:, 1].astype(np.float32) / 32768.0)


def payload_bits(sid: str) -> np.ndarray | None:
    gt = load_gt(sid)
    if not gt.get("payload_bits_file"):
        return None
    raw = (scenario_dir(sid) / gt["payload_bits_file"]).read_bytes()
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
    return bits[: gt["payload_bit_count"]]
