import numpy as np
import pytest
from scipy.io import wavfile

from core.bitstream.crc import append_crc16
from core.common.enums import ClassifierStatus, ModulationType, SampleDtype
from core.common.models import ClassificationResult, IQConfig
from core.fec.convolutional import encode_convolutional
from core.fec.ldpc_decoder import LDPC_CONFIGS, encode_ldpc
from core.interleaver.pseudorandom import PseudoRandomInterleaver
from core.pipeline.analyzer import AnalysisResult, Analyzer
from libs.modulation_library import bits_to_symbols


class FixedClassifier:
    def __init__(self, modulation=ModulationType.BPSK, status=ClassifierStatus.SUCCESS):
        self.modulation = modulation
        self.status = status
        self.calls = []

    def classify(self, samples, *, samples_per_symbol=1.0):
        values = np.asarray(samples).copy()
        self.calls.append((values, samples_per_symbol))
        return ClassificationResult(
            modulation=self.modulation,
            confidence=0.98 if self.modulation is not ModulationType.UNKNOWN else 0.0,
            method="injected_test_classifier",
            probabilities=(
                {self.modulation: 0.98}
                if self.modulation is not ModulationType.UNKNOWN else {}
            ),
            status=self.status,
            warnings=["test OOD rejection"] if self.modulation is ModulationType.UNKNOWN else [],
        )


def _write_iq(path, samples):
    np.asarray(samples, dtype=np.complex64).tofile(path)
    return IQConfig(dtype=SampleDtype.COMPLEX64, sample_rate=1_000_000.0)


def test_iq_pipeline_runs_classify_demod_fec_interleaver_crc_and_frame_analysis(tmp_path):
    rng = np.random.default_rng(34001)
    payload = rng.integers(0, 2, size=104, dtype=np.uint8)
    coded = encode_convolutional(append_crc16(payload), "Conv_R12_K9")
    transmitted = PseudoRandomInterleaver().interleave(coded)
    samples = bits_to_symbols(transmitted, ModulationType.BPSK)
    iq_config = _write_iq(tmp_path / "signal.IQ", samples)
    classifier = FixedClassifier()

    result = Analyzer(classifier=classifier).analyze_file(
        tmp_path / "signal.IQ",
        iq_config=iq_config,
        expected_payload_length=payload.size,
        crc_present=True,
        sync_word=np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8),
    )

    assert result.status == "complete"
    assert result.synchronization["phase_ambiguity_resolution"]["status"] == "crc_selected"
    assert result.validation_status == "valid_with_warnings"
    assert result.classification["modulation"] == "bpsk"
    assert result.demodulation["bit_count"] == transmitted.size
    assert result.demodulation["llr_count"] == transmitted.size
    assert result.fec["fec_type"] == "Conv_R12_K9"
    assert result.fec["interleaver_type"] == "PseudoRandom_256_Seed_26147"
    assert result.fec["crc_pass"] is True
    assert result.frames["sync_word_supplied"] is True
    assert result.frames["status"] in {"correlated", "not_found"}
    np.testing.assert_array_equal(result.visualization["fec_recovered_bits"], payload)
    np.testing.assert_array_equal(classifier.calls[0][0], samples)
    assert classifier.calls[0][1] == 1.0


def test_wav_pipeline_reads_metadata_and_produces_visualization_arrays(tmp_path):
    samples = bits_to_symbols(
        np.tile(np.array([0, 1], dtype=np.uint8), 128), ModulationType.BPSK
    )
    wav_path = tmp_path / "mono.wav"
    pcm = np.clip(np.rint(samples.real * 10000), -32768, 32767).astype(np.int16)
    wavfile.write(wav_path, 48_000, pcm)
    analyzer = Analyzer(classifier=FixedClassifier())

    result = analyzer.analyze_file(wav_path, run_fec=False)

    assert result.file_metadata["format"] == "wav"
    assert result.file_metadata["sample_rate_hz"] == 48_000.0
    assert result.file_metadata["channel_count"] == 1
    assert result.status == "partial"
    assert result.synchronization["phase_ambiguity_resolution"]["status"] == "unresolved"
    assert result.visualization["spectrum_frequency_hz"].size > 0
    assert result.visualization["waterfall_magnitude_db"].ndim == 2
    assert result.visualization["constellation_i"].size == result.demodulation["bit_count"]


def test_configured_matched_filter_and_gardner_timing_reach_classifier_as_symbols(tmp_path):
    symbol_values = bits_to_symbols(
        np.tile(np.array([0, 1, 1, 0], dtype=np.uint8), 64), ModulationType.BPSK
    )
    samples = np.repeat(symbol_values, 4).astype(np.complex64)
    iq_config = _write_iq(tmp_path / "oversampled.iq", samples)
    classifier = FixedClassifier()

    result = Analyzer(classifier=classifier).analyze_file(
        tmp_path / "oversampled.iq",
        iq_config=iq_config,
        samples_per_symbol=4,
        matched_filter_rolloff=0.35,
        recover_symbol_timing=True,
        run_fec=False,
    )

    assert result.synchronization["preprocessing"]["matched_filter_applied"]
    assert result.synchronization["preprocessing"]["timing_recovery_applied"]
    assert classifier.calls[0][1] == 1.0
    assert classifier.calls[0][0].size < samples.size
    assert result.synchronization["timing"]["samples_per_symbol"] == 4.0


def test_classifier_input_is_not_rescaled_by_the_pipeline(tmp_path):
    samples = np.array([0.25 + 0.5j, -0.75 + 0.125j] * 64, dtype=np.complex64)
    iq_config = _write_iq(tmp_path / "scaled.iq", samples)
    classifier = FixedClassifier()

    result = Analyzer(classifier=classifier).analyze_file(
        tmp_path / "scaled.iq", iq_config=iq_config, run_fec=False
    )

    np.testing.assert_array_equal(classifier.calls[0][0], samples)
    assert classifier.calls[0][0].dtype == np.complex64
    assert classifier.calls[0][1] == 1.0
    assert result.features["names"] and len(result.features["values"]) == 17
    assert result.classification["method"] == "injected_test_classifier"


def test_ood_or_unknown_classification_stops_before_demodulation(tmp_path):
    samples = np.ones(64, dtype=np.complex64)
    iq_config = _write_iq(tmp_path / "unknown.iq", samples)
    classifier = FixedClassifier(
        modulation=ModulationType.UNKNOWN,
        status=ClassifierStatus.SUCCESS,
    )

    result = Analyzer(classifier=classifier).analyze_file(
        tmp_path / "unknown.iq", iq_config=iq_config
    )

    assert result.status == "rejected"
    assert result.demodulation["status"] == "unavailable"
    assert result.fec["status"] == "unavailable"
    assert any("OOD" in warning for warning in result.warnings)


def test_missing_production_model_is_explicit_and_has_no_classical_fallback(tmp_path):
    iq_path = tmp_path / "no_model.iq"
    iq_config = _write_iq(iq_path, np.ones(64, dtype=np.complex64))
    analyzer = Analyzer(model_path=tmp_path / "missing.onnx")

    result = analyzer.analyze_file(iq_path, iq_config=iq_config)

    assert result.status == "rejected"
    assert result.classification["status"] == "model_unavailable"
    assert result.classification["method"] == "onnx"
    assert "no baseline fallback" in result.classification["warnings"][0]
    assert result.demodulation["status"] == "unavailable"


def test_file_and_configuration_failures_are_structured(tmp_path):
    analyzer = Analyzer(classifier=FixedClassifier())
    unsupported = tmp_path / "capture.txt"
    unsupported.write_text("not a signal")
    missing_iq_config = tmp_path / "unconfigured.iq"
    missing_iq_config.write_bytes(np.zeros(64, dtype=np.complex64).tobytes())
    malformed_wav = tmp_path / "broken.wav"
    malformed_wav.write_bytes(b"RIFFbroken")
    empty = tmp_path / "empty.iq"
    empty.write_bytes(b"")

    assert analyzer.analyze_file(unsupported).errors[0]["code"] == "unsupported_format"
    assert analyzer.analyze_file(missing_iq_config).errors[0]["code"] == "iq_configuration_required"
    assert analyzer.analyze_file(malformed_wav).errors[0]["code"] == "load_failed"
    assert analyzer.analyze_file(empty, iq_config=IQConfig()).errors[0]["code"] == "load_failed"


def test_invalid_iq_sample_rate_is_reported_and_skipped(tmp_path):
    # A valid BPSK stream (a constant DC fixture is correctly rejected by the
    # signal-support gate, which would end the run as "rejected").
    bpsk = bits_to_symbols(
        np.random.default_rng(7).integers(0, 2, 256, dtype=np.uint8), ModulationType.BPSK
    )
    iq_config = _write_iq(tmp_path / "bad_rate.iq", bpsk)
    iq_config.sample_rate = 0.0

    result = Analyzer(classifier=FixedClassifier()).analyze_file(
        tmp_path / "bad_rate.iq", iq_config=iq_config, run_fec=False
    )

    assert any(error["code"] == "invalid_sample_rate" for error in result.errors)
    assert result.file_metadata["sample_rate_hz"] is None
    assert result.visualization.get("spectrum_frequency_hz") is None
    assert result.status == "partial"


def test_fsk_without_explicit_tones_reports_demodulator_configuration_error(tmp_path):
    # A real two-tone 2FSK waveform (a constant DC fixture is correctly rejected
    # by the signal-support gate before the demodulator configuration is checked).
    from tests.modulation_helpers import fsk_waveform, random_bits

    samples = fsk_waveform(
        random_bits(256, np.random.default_rng(8)),
        sample_rate=48_000, samples_per_symbol=4, tones_hz=(-6_000, 6_000),
    ).astype(np.complex64)
    iq_config = _write_iq(tmp_path / "fsk.iq", samples)
    iq_config.sample_rate = 48_000.0
    classifier = FixedClassifier(modulation=ModulationType.FSK2)

    result = Analyzer(classifier=classifier).analyze_file(
        tmp_path / "fsk.iq",
        iq_config=iq_config,
        samples_per_symbol=4,
    )

    assert result.status == "partial"
    assert result.demodulation["status"] == "failed"
    assert "explicit" in result.demodulation["reason"]
    assert result.fec["status"] == "unavailable"


def test_oversampled_psk_without_timing_recovery_is_rejected_by_feature_contract(tmp_path):
    samples = np.tile(np.array([1.0, -1.0], dtype=np.float32), 128).astype(np.complex64)
    iq_config = _write_iq(tmp_path / "oversampled_unrecovered.iq", samples)

    result = Analyzer(classifier=FixedClassifier()).analyze_file(
        tmp_path / "oversampled_unrecovered.iq",
        iq_config=iq_config,
        samples_per_symbol=4,
        run_fec=False,
    )

    assert result.status == "rejected"
    assert any(
        error["code"] == "sample_representation_mismatch"
        for error in result.errors
    )
    assert result.demodulation["status"] == "unavailable"


def test_ldpc_trials_receive_demodulator_generated_soft_llrs(tmp_path):
    configuration = "LDPC_648_R1_2"
    spec = LDPC_CONFIGS[configuration]
    payload = np.zeros(spec.K - 16, dtype=np.uint8)
    coded = encode_ldpc(append_crc16(payload), configuration)
    samples = bits_to_symbols(coded, ModulationType.BPSK)
    iq_config = _write_iq(tmp_path / "ldpc.iq", samples)

    result = Analyzer(classifier=FixedClassifier()).analyze_file(
        tmp_path / "ldpc.iq",
        iq_config=iq_config,
        expected_payload_length=payload.size,
        crc_present=True,
    )

    assert result.fec["fec_type"] == configuration
    assert result.fec["soft_llrs_supplied"] is True
    assert result.fec["crc_pass"] is True
    assert result.fec["best_diagnostics"]["ldpc_batch"]["llr_input"] == (
        "caller-supplied demodulator soft LLRs"
    )
    np.testing.assert_array_equal(result.visualization["fec_recovered_bits"], payload)


def test_serialization_excludes_large_arrays_unless_requested(tmp_path):
    samples = np.ones(64, dtype=np.complex64)
    iq_config = _write_iq(tmp_path / "serialize.iq", samples)
    result = Analyzer(classifier=FixedClassifier()).analyze_file(
        tmp_path / "serialize.iq", iq_config=iq_config, run_fec=False
    )

    compact = result.to_dict()
    expanded = result.to_dict(include_arrays=True)
    assert compact["visualization"]["waveform_i"]["shape"] == [64]
    assert isinstance(expanded["visualization"]["waveform_i"], list)
    assert isinstance(result.to_json(), str)
    assert isinstance(result, AnalysisResult)
