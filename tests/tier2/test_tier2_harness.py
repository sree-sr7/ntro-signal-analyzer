from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from tools.tier2.generate_signals import (
    _block_interleave,
    _conv_encode,
    _diagonal_interleave,
    _rs_encode,
    generate_scenario,
    load_config,
)
from tools.tier2.run_tier2 import _expected_interleaver


ROOT = Path(__file__).resolve().parents[2]


def test_generator_has_no_application_or_dataset_imports() -> None:
    path = ROOT / "tools" / "tier2" / "generate_signals.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = ("core", "tests", "scripts", "generators", "data", "models", "reedsolo")
    assert not [module for module in imports if any(module == item or module.startswith(item + ".") for item in forbidden)]
    assert "numpy" in imports
    assert "komm" in imports


def test_baseline_fec_lengths_are_whole_interleaver_frames() -> None:
    config = load_config(ROOT / "tools" / "tier2" / "config" / "baseline_scenarios.json")
    cases = {item["scenario_id"]: item for item in config["scenarios"]}
    t201 = cases["T2-01"]
    encoded_01 = 2 * (t201["payload_bits"] + 16 + 4)
    assert encoded_01 == 3904
    assert encoded_01 % 64 == 0
    t208 = cases["T2-08"]
    encoded_08 = 2 * (t208["payload_bits"] + 16 + 6)
    assert encoded_08 == 8192
    assert encoded_08 % 256 == 0
    assert cases["T2-04"]["payload_bytes"] + 2 == 223 * cases["T2-04"]["rs_codewords"]
    assert cases["T2-07"]["payload_bytes"] + 2 == 223 * cases["T2-07"]["concatenated_codewords"]


def test_independent_encoder_and_interleaver_outputs_have_exact_lengths() -> None:
    payload = bytes(range(223))
    rs_word = _rs_encode(payload)
    assert len(rs_word) == 255
    assert rs_word[:223] == payload
    bits = np.tile(np.array([0, 1], dtype=np.uint8), 1024)
    coded = _conv_encode(bits, 7)
    assert coded.size == (bits.size + 6) * 2
    assert _block_interleave(coded[: (coded.size // 64) * 64], 8, 8).size == (coded.size // 64) * 64
    diag = _diagonal_interleave(np.tile(np.array([0, 1], dtype=np.uint8), 128))
    assert diag.size == 256


def test_ground_truth_promotes_required_tones_and_ood_label(tmp_path: Path) -> None:
    config = load_config(ROOT / "tools" / "tier2" / "config" / "baseline_scenarios.json")
    cases = {item["scenario_id"]: item for item in config["scenarios"]}
    fsk_gt = generate_scenario(cases["T2-06"], config["sample_rates_hz"], tmp_path / "fsk")
    noise_gt = generate_scenario(cases["T2-09"], config["sample_rates_hz"], tmp_path / "noise")
    assert fsk_gt["tone_frequencies_hz"] == [-3000.0, 3000.0]
    assert fsk_gt["analyzer_configuration_allowed"]["tone_frequencies_hz"] == [-3000.0, 3000.0]
    assert noise_gt["ood"] is True


def test_none_interleaver_matches_public_api_representation() -> None:
    assert _expected_interleaver("None") is None
    assert _expected_interleaver(None) is None
    assert _expected_interleaver("Block_8x8") == "Block_8x8"
