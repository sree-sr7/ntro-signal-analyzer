import numpy as np
import pytest
from pathlib import Path
from core.common.validation import (
    validate_file_path, validate_signal_data, validate_sample_rate,
    validate_iq_config_for_loading,
)
from core.common.models import SignalData, SignalMetadata, IQConfig
from core.common.enums import SampleDtype


class TestValidateFilePath:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.iq"
        f.write_bytes(b"\x00" * 100)
        result = validate_file_path(f)
        assert result.valid is True

    def test_nonexistent(self, tmp_path):
        result = validate_file_path(tmp_path / "nope.iq")
        assert result.valid is False
        assert any("does not exist" in e for e in result.errors)

    def test_directory_not_file(self, tmp_path):
        result = validate_file_path(tmp_path)
        assert result.valid is False
        assert any("not a file" in e for e in result.errors)

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.iq"
        f.write_bytes(b"")
        result = validate_file_path(f)
        assert result.valid is False
        assert any("empty" in e.lower() for e in result.errors)


class TestValidateSignalData:
    def test_valid_signal(self):
        samples = np.array([1+2j, 3+4j], dtype=np.complex64)
        meta = SignalMetadata(sample_count=2)
        sd = SignalData(samples=samples, metadata=meta)
        result = validate_signal_data(sd)
        assert result.valid is True

    def test_none_samples(self):
        sd = SignalData(samples=None, metadata=SignalMetadata())
        result = validate_signal_data(sd)
        assert result.valid is False

    def test_empty_samples(self):
        sd = SignalData(samples=np.array([], dtype=np.complex64), metadata=SignalMetadata())
        result = validate_signal_data(sd)
        assert result.valid is False

    def test_nan_samples_warning(self):
        samples = np.array([1+2j, np.nan+0j, 3+4j], dtype=np.complex128)
        meta = SignalMetadata(sample_count=3)
        sd = SignalData(samples=samples, metadata=meta)
        result = validate_signal_data(sd)
        # Small ratio of NaN => warning not error
        assert result.valid is True
        assert len(result.warnings) > 0

    def test_all_nan_error(self):
        samples = np.full(10, np.nan + 0j, dtype=np.complex128)
        meta = SignalMetadata(sample_count=10)
        sd = SignalData(samples=samples, metadata=meta)
        result = validate_signal_data(sd)
        assert result.valid is False

    def test_sample_count_mismatch_warning(self):
        samples = np.zeros(50, dtype=np.complex64)
        meta = SignalMetadata(sample_count=100)  # mismatch
        sd = SignalData(samples=samples, metadata=meta)
        result = validate_signal_data(sd)
        assert len(result.warnings) > 0


class TestValidateSampleRate:
    def test_valid(self):
        result = validate_sample_rate(1e6)
        assert result.valid is True

    def test_none(self):
        result = validate_sample_rate(None)
        assert result.valid is False

    def test_negative(self):
        result = validate_sample_rate(-1000)
        assert result.valid is False

    def test_very_high_warning(self):
        result = validate_sample_rate(1e13)
        assert result.valid is True  # valid but warned
        assert len(result.warnings) > 0


class TestValidateIQConfig:
    def test_valid_config(self):
        cfg = IQConfig(dtype=SampleDtype.COMPLEX64)
        result = validate_iq_config_for_loading(cfg)
        assert result.valid is True

    def test_negative_offset(self):
        cfg = IQConfig(dtype=SampleDtype.COMPLEX64, header_offset_bytes=-1)
        result = validate_iq_config_for_loading(cfg)
        assert result.valid is False
