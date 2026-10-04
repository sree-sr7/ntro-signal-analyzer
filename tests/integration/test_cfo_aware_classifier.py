from pathlib import Path

from core.common.enums import SampleDtype
from core.common.models import IQConfig
from core.pipeline.analyzer import Analyzer


def test_saved_qpsk_with_cfo_is_classified_after_cfo_correction():
    root = Path(__file__).resolve().parents[2]
    capture = (
        root / "reports" / "tier2" / "baseline" / "20260930_201445_269"
        / "T2-01" / "capture.iq"
    )
    config = IQConfig(dtype=SampleDtype.COMPLEX64, sample_rate=1_000_000.0)

    result = Analyzer().analyze_file(capture, iq_config=config, run_fec=False)

    correction = result.synchronization["classifier_cfo_correction"]
    assert correction["status"] == "applied"
    assert correction["modulation_before"] == "8psk"
    assert correction["modulation_after"] == "qpsk"
    assert result.classification["modulation"] == "qpsk"
