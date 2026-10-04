import numpy as np
import pytest
from core.common.models import (
    IQConfig, SignalMetadata, SignalData, LoadResult, ValidationResult,
)
from core.common.enums import (
    SignalFormat, IQLayout, Endianness, SampleDtype, LoadStatus,
)


class TestSignalMetadata:
    def test_default_values(self):
        meta = SignalMetadata()
        assert meta.format == SignalFormat.UNKNOWN
        assert meta.sample_rate is None
        assert meta.center_frequency is None
        assert meta.sample_count == 0
        assert meta.iq_layout == IQLayout.UNKNOWN
        assert meta.warnings == []

    def test_has_sample_rate_true(self):
        meta = SignalMetadata(sample_rate=1e6)
        assert meta.has_sample_rate is True

    def test_has_sample_rate_false_none(self):
        meta = SignalMetadata(sample_rate=None)
        assert meta.has_sample_rate is False

    def test_has_sample_rate_false_zero(self):
        meta = SignalMetadata(sample_rate=0.0)
        assert meta.has_sample_rate is False

    def test_has_sample_rate_false_negative(self):
        meta = SignalMetadata(sample_rate=-1.0)
        assert meta.has_sample_rate is False

    def test_has_center_frequency(self):
        meta = SignalMetadata(center_frequency=433.92e6)
        assert meta.has_center_frequency is True

    def test_source_metadata_independent(self):
        m1 = SignalMetadata()
        m2 = SignalMetadata()
        m1.source_metadata["key"] = "value"
        assert "key" not in m2.source_metadata


class TestSignalData:
    def test_sample_count(self):
        samples = np.zeros(100, dtype=np.complex64)
        sd = SignalData(samples=samples, metadata=SignalMetadata())
        assert sd.sample_count == 100

    def test_is_complex(self):
        sd = SignalData(samples=np.zeros(10, dtype=np.complex64), metadata=SignalMetadata())
        assert sd.is_complex is True

    def test_is_not_complex_real(self):
        sd = SignalData(samples=np.zeros(10, dtype=np.float32), metadata=SignalMetadata())
        assert sd.is_complex is False


class TestValidationResult:
    def test_initially_valid(self):
        vr = ValidationResult(valid=True)
        assert vr.valid is True
        assert vr.errors == []

    def test_add_error_invalidates(self):
        vr = ValidationResult(valid=True)
        vr.add_error("bad thing")
        assert vr.valid is False
        assert "bad thing" in vr.errors

    def test_add_warning_keeps_valid(self):
        vr = ValidationResult(valid=True)
        vr.add_warning("maybe odd")
        assert vr.valid is True
        assert "maybe odd" in vr.warnings

    def test_merge(self):
        vr1 = ValidationResult(valid=True, warnings=["w1"])
        vr2 = ValidationResult(valid=False, errors=["e1"])
        vr1.merge(vr2)
        assert vr1.valid is False
        assert "e1" in vr1.errors
        assert "w1" in vr1.warnings


class TestLoadResult:
    def test_success(self):
        sd = SignalData(samples=np.zeros(10, dtype=np.complex64), metadata=SignalMetadata())
        lr = LoadResult.ok(sd)
        assert lr.success is True
        assert lr.status == LoadStatus.SUCCESS
        assert lr.signal is not None

    def test_partial_with_warnings(self):
        sd = SignalData(samples=np.zeros(10, dtype=np.complex64), metadata=SignalMetadata())
        lr = LoadResult.ok(sd, warnings=["something"])
        assert lr.success is True
        assert lr.status == LoadStatus.PARTIAL

    def test_failure(self):
        lr = LoadResult.fail(["file not found"])
        assert lr.success is False
        assert lr.status == LoadStatus.FAILED
        assert lr.signal is None
        assert "file not found" in lr.errors


class TestIQConfig:
    def test_defaults(self):
        cfg = IQConfig()
        assert cfg.dtype == SampleDtype.COMPLEX64
        assert cfg.sample_rate is None
        assert cfg.header_offset_bytes == 0
        assert cfg.endianness == Endianness.LITTLE
