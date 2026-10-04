import socket

import numpy as np
import pytest

from core.common.enums import ClassifierStatus, ModulationType
from core.pipeline.analyzer import Analyzer
from core.pipeline.demo import (
    DEFAULT_DEMO_CONFIG,
    DEMO_PAYLOAD_HEX,
    DEMO_SOURCE,
    DemoConfig,
    analyze_demo,
    generate_demo_signal,
)


def test_demo_signal_is_exactly_reproducible_for_the_same_configuration():
    first = generate_demo_signal(DEFAULT_DEMO_CONFIG)
    second = generate_demo_signal(DEFAULT_DEMO_CONFIG)

    assert first.samples.dtype == np.complex64
    assert first.samples.shape == (DEFAULT_DEMO_CONFIG.interleaver_frame_length,)
    np.testing.assert_array_equal(first.samples, second.samples)
    np.testing.assert_array_equal(first.expected_payload_bits, second.expected_payload_bits)


def test_demo_configuration_and_fixed_payload_are_valid_and_reproducible():
    configuration = DemoConfig()

    assert configuration.modulation is ModulationType.BPSK
    assert configuration.samples_per_symbol == 1
    assert configuration.payload_length_bits == 104
    assert configuration.payload_hex == DEMO_PAYLOAD_HEX
    np.testing.assert_array_equal(configuration.payload_bits, DemoConfig().payload_bits)
    with pytest.raises(ValueError, match="supports BPSK only"):
        DemoConfig(modulation=ModulationType.QPSK)
    with pytest.raises(ValueError, match="form exactly one frame"):
        DemoConfig(interleaver_frame_length=512)
    with pytest.raises(ValueError, match="between -100 and 100"):
        DemoConfig(snr_db=float("inf"))


def test_demo_runs_real_analyzer_and_local_production_onnx_without_network(monkeypatch):
    def deny_network(*_args, **_kwargs):
        raise AssertionError("Demo Mode attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    stages = []
    result = analyze_demo(
        Analyzer(),
        progress_callback=lambda message, percent: stages.append((message, percent)),
    )

    assert result.source == DEMO_SOURCE
    assert result.file_metadata["source"] == DEMO_SOURCE
    assert result.status == "complete"
    assert result.file_format == "iq"
    assert result.classification["method"] == "onnx"
    assert result.classification["status"] == ClassifierStatus.SUCCESS.value
    assert result.classification["modulation"] == ModulationType.BPSK.value
    assert result.demo["production_classifier_wrapper_invoked"] is True
    assert result.demo["production_onnx_session_loaded"] is True
    assert result.demo["production_onnx_inference_succeeded"] is True
    assert result.fec["fec_type"] == DEFAULT_DEMO_CONFIG.fec
    assert result.fec["interleaver_type"] == "PseudoRandom_256_Seed_26147"
    assert result.fec["crc_pass"] is True
    assert result.demo["payload_recovery"] == "matched"
    assert result.demo["payload_match"] is True
    assert result.demo["recovered_payload_hex"] == DEMO_PAYLOAD_HEX
    assert result.visualization["waveform_i"].size > 0
    assert result.visualization["spectrum_power"].size > 0
    assert result.visualization["waterfall_magnitude_db"].size > 0
    assert result.visualization["constellation_i"].size > 0
    assert result.visualization["fec_recovered_bits"].size == 104
    assert [percent for _, percent in stages] == sorted(percent for _, percent in stages)
    assert stages[0][0] == "Generating demo signal..."
    assert stages[-1][1] == 100


def test_demo_reports_missing_production_model_without_fabricating_a_class(tmp_path):
    result = analyze_demo(Analyzer(model_path=tmp_path / "missing.onnx"))

    assert result.source == DEMO_SOURCE
    assert result.status == "rejected"
    assert result.classification["method"] == "onnx"
    assert result.classification["status"] == ClassifierStatus.MODEL_UNAVAILABLE.value
    assert result.classification["modulation"] == ModulationType.UNKNOWN.value
    assert result.demo["production_classifier_wrapper_invoked"] is True
    assert result.demo["production_onnx_session_loaded"] is False
    assert result.demo["production_onnx_inference_succeeded"] is False
    assert result.demo["payload_recovery"] == "not_available"
    assert result.demodulation["status"] == "unavailable"
    assert result.visualization["waveform_i"].size > 0
    assert result.visualization["spectrum_power"].size > 0
    assert result.visualization["waterfall_magnitude_db"].size > 0
    assert "constellation_i" not in result.visualization
