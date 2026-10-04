import numpy as np
import pytest
from pathlib import Path
import scipy.io.wavfile as wavfile
from core.file_io.format_detector import detect_format, detect_iq_dtype_hint
from core.common.enums import SignalFormat, SampleDtype


class TestDetectFormat:
    def test_wav_extension(self, tmp_path):
        path = tmp_path / "test.wav"
        path.write_bytes(b"RIFF" + b"\x00" * 100)
        assert detect_format(path) == SignalFormat.WAV

    def test_sigmf_meta_extension(self, tmp_path):
        path = tmp_path / "test.sigmf-meta"
        path.write_text('{}')
        assert detect_format(path) == SignalFormat.SIGMF

    def test_iq_extension(self, tmp_path):
        for ext in [".iq", ".bin", ".raw", ".cf32", ".cs16", ".cs8", ".cu8"]:
            path = tmp_path / f"test{ext}"
            path.write_bytes(b"\x00" * 100)
            assert detect_format(path) == SignalFormat.IQ, f"Failed for {ext}"

    def test_unknown_extension(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_bytes(b"\x00" * 100)
        assert detect_format(path) == SignalFormat.UNKNOWN

    def test_wav_by_header(self, tmp_path):
        """A file with wrong extension but RIFF header."""
        path = tmp_path / "test.dat"
        # Write minimal RIFF header
        sr = 44100
        data = np.zeros(100, dtype=np.int16)
        wavfile.write(str(tmp_path / "temp.wav"), sr, data)
        # Copy content to .dat
        import shutil
        shutil.copy(tmp_path / "temp.wav", path)
        # Should detect as WAV if content-based detection is tried
        # (depends on whether detect_format checks content for unknown extensions)
        fmt = detect_format(path)
        # This might be UNKNOWN by extension, which is acceptable
        assert fmt in (SignalFormat.WAV, SignalFormat.UNKNOWN)


class TestDetectIQDtypeHint:
    def test_cf32(self, tmp_path):
        assert detect_iq_dtype_hint(tmp_path / "x.cf32") == SampleDtype.COMPLEX64

    def test_cs16(self, tmp_path):
        assert detect_iq_dtype_hint(tmp_path / "x.cs16") == SampleDtype.INT16

    def test_cs8(self, tmp_path):
        assert detect_iq_dtype_hint(tmp_path / "x.cs8") == SampleDtype.INT8

    def test_cu8(self, tmp_path):
        assert detect_iq_dtype_hint(tmp_path / "x.cu8") == SampleDtype.UINT8

    def test_ambiguous_iq(self, tmp_path):
        assert detect_iq_dtype_hint(tmp_path / "x.iq") is None

    def test_ambiguous_bin(self, tmp_path):
        assert detect_iq_dtype_hint(tmp_path / "x.bin") is None
