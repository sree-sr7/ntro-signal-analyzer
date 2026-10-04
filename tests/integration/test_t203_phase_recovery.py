import json
from pathlib import Path

import numpy as np

from core.common.enums import SampleDtype
from core.common.models import IQConfig
from core.pipeline.analyzer import Analyzer


def test_t203_crc_selects_correct_8psk_phase_and_recovers_payload():
    root = Path(__file__).resolve().parents[2]
    scenario = root / "reports" / "tier2" / "baseline" / "20260930_201445_269" / "T2-03"
    ground_truth = json.loads((scenario / "ground_truth.json").read_text(encoding="utf-8"))
    payload_bytes = np.frombuffer((scenario / ground_truth["payload_bits_file"]).read_bytes(), dtype=np.uint8)
    expected_payload = np.unpackbits(payload_bytes, bitorder="big")[: ground_truth["payload_bit_count"]]

    result = Analyzer().analyze_file(
        scenario / ground_truth["file_path"],
        iq_config=IQConfig(
            dtype=SampleDtype.COMPLEX64,
            sample_rate=float(ground_truth["sample_rate_hz"]),
        ),
        expected_payload_length=int(ground_truth["payload_bit_count"]),
        crc_present=True,
    )

    assert result.status == "complete"
    assert result.fec["fec_type"] == "Conv_R12_K7"
    assert result.fec["interleaver_type"] == "Convolutional_Depth_12"
    assert result.fec["crc_pass"] is True
    assert result.synchronization["phase_ambiguity_resolution"]["status"] == "crc_selected"
    assert result.synchronization["phase_ambiguity_resolution"]["selected_candidate_index"] == 1
    np.testing.assert_array_equal(result.visualization["fec_recovered_bits"], expected_payload)
