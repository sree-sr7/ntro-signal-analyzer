import numpy as np
import pytest
from pathlib import Path
from core.common.models import IQConfig
from core.common.enums import SampleDtype, IQLayout, Endianness
from core.file_io.iq_loader import load_iq


class TestIQLoaderComplex64:
    def test_load_complex64(self, tmp_path):
        """Load a tiny complex64 IQ file."""
        original = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
        path = tmp_path / "test.iq"
        original.tofile(str(path))
        config = IQConfig(dtype=SampleDtype.COMPLEX64, sample_rate=1e6)
        result = load_iq(path, config)
        assert result.success
        assert result.signal is not None
        assert len(result.signal.samples) == 3
        np.testing.assert_allclose(
            result.signal.samples, original, rtol=1e-5
        )
        assert result.signal.metadata.sample_rate == 1e6

    def test_load_complex64_no_sample_rate(self, tmp_path):
        """Loading without sample_rate should succeed with warning."""
        data = np.zeros(10, dtype=np.complex64)
        path = tmp_path / "test.iq"
        data.tofile(str(path))
        config = IQConfig(dtype=SampleDtype.COMPLEX64)  # no sample_rate
        result = load_iq(path, config)
        assert result.success
        assert result.signal.metadata.sample_rate is None
        assert any("sample rate" in w.lower() for w in result.signal.metadata.warnings)


class TestIQLoaderFloat32Interleaved:
    def test_load_float32_iq(self, tmp_path):
        """Load float32 interleaved I,Q,I,Q,..."""
        # 3 complex samples: (1+2j), (3+4j), (5+6j)
        interleaved = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        path = tmp_path / "test.iq"
        interleaved.tofile(str(path))
        config = IQConfig(
            dtype=SampleDtype.FLOAT32,
            iq_layout=IQLayout.IQ_INTERLEAVED,
            sample_rate=2e6,
        )
        result = load_iq(path, config)
        assert result.success
        assert len(result.signal.samples) == 3
        np.testing.assert_allclose(
            result.signal.samples,
            np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64),
            rtol=1e-5,
        )

    def test_load_float32_qi(self, tmp_path):
        """Load float32 QI interleaved: Q,I,Q,I,..."""
        # QI order: Q0=2, I0=1, Q1=4, I1=3
        interleaved = np.array([2, 1, 4, 3], dtype=np.float32)
        path = tmp_path / "test.iq"
        interleaved.tofile(str(path))
        config = IQConfig(
            dtype=SampleDtype.FLOAT32,
            iq_layout=IQLayout.QI_INTERLEAVED,
            sample_rate=1e6,
        )
        result = load_iq(path, config)
        assert result.success
        assert len(result.signal.samples) == 2
        np.testing.assert_allclose(
            result.signal.samples,
            np.array([1+2j, 3+4j], dtype=np.complex64),
            rtol=1e-5,
        )


class TestIQLoaderInt16:
    def test_load_int16(self, tmp_path):
        """Load int16 interleaved IQ."""
        # Two samples: I=16384, Q=-16384 and I=0, Q=32767
        raw = np.array([16384, -16384, 0, 32767], dtype=np.int16)
        path = tmp_path / "test.cs16"
        raw.tofile(str(path))
        config = IQConfig(
            dtype=SampleDtype.INT16,
            iq_layout=IQLayout.IQ_INTERLEAVED,
            sample_rate=1e6,
        )
        result = load_iq(path, config)
        assert result.success
        assert len(result.signal.samples) == 2
        # Check normalization: 16384/32768 = 0.5
        assert abs(result.signal.samples[0].real - 0.5) < 0.01
        assert abs(result.signal.samples[0].imag - (-0.5)) < 0.01


class TestIQLoaderUint8:
    def test_load_uint8_rtlsdr(self, tmp_path):
        """Load uint8 RTL-SDR format (centered at 127.5)."""
        raw = np.array([128, 128, 255, 0], dtype=np.uint8)
        path = tmp_path / "test.cu8"
        raw.tofile(str(path))
        config = IQConfig(
            dtype=SampleDtype.UINT8,
            iq_layout=IQLayout.IQ_INTERLEAVED,
            sample_rate=2.4e6,
        )
        result = load_iq(path, config)
        assert result.success
        assert len(result.signal.samples) == 2
        # 128 - 127.5 = 0.5, / 128 ≈ 0.0039
        assert abs(result.signal.samples[0].real) < 0.01


class TestIQLoaderInt8:
    def test_load_int8(self, tmp_path):
        raw = np.array([64, -64, 0, 127], dtype=np.int8)
        path = tmp_path / "test.cs8"
        raw.tofile(str(path))
        config = IQConfig(
            dtype=SampleDtype.INT8,
            iq_layout=IQLayout.IQ_INTERLEAVED,
            sample_rate=1e6,
        )
        result = load_iq(path, config)
        assert result.success
        assert len(result.signal.samples) == 2
        # 64/128 = 0.5
        assert abs(result.signal.samples[0].real - 0.5) < 0.01


class TestIQLoaderEdgeCases:
    def test_nonexistent_file(self, tmp_path):
        config = IQConfig(dtype=SampleDtype.COMPLEX64)
        result = load_iq(tmp_path / "nope.iq", config)
        assert not result.success
        assert len(result.errors) > 0

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.iq"
        path.write_bytes(b"")
        config = IQConfig(dtype=SampleDtype.COMPLEX64)
        result = load_iq(path, config)
        assert not result.success

    def test_truncated_file_warning(self, tmp_path):
        """File with extra bytes that don't form a complete sample."""
        # 3 complete complex64 samples = 24 bytes, add 3 extra
        data = np.zeros(3, dtype=np.complex64)
        path = tmp_path / "test.iq"
        with open(path, 'wb') as f:
            data.tofile(f)
            f.write(b"\x00\x00\x00")  # 3 extra bytes
        config = IQConfig(dtype=SampleDtype.COMPLEX64, sample_rate=1e6)
        result = load_iq(path, config)
        assert result.success
        assert len(result.signal.samples) == 3  # only complete samples
        # Should have a warning about truncation
        all_warnings = result.warnings + result.signal.metadata.warnings
        assert any("truncat" in w.lower() for w in all_warnings)

    def test_header_offset(self, tmp_path):
        """Skip header bytes."""
        header = b"HEADER12"  # 8 bytes
        data = np.array([1+2j, 3+4j], dtype=np.complex64)
        path = tmp_path / "test.iq"
        with open(path, 'wb') as f:
            f.write(header)
            data.tofile(f)
        config = IQConfig(
            dtype=SampleDtype.COMPLEX64,
            header_offset_bytes=8,
            sample_rate=1e6,
        )
        result = load_iq(path, config)
        assert result.success
        assert len(result.signal.samples) == 2
        np.testing.assert_allclose(result.signal.samples, data, rtol=1e-5)
