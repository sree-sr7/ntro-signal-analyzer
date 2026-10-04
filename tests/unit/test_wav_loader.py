import numpy as np
import pytest
from pathlib import Path
import scipy.io.wavfile as wavfile
from core.file_io.wav_loader import load_wav
from core.common.enums import SignalFormat, IQLayout


class TestWavLoaderMono:
    def test_load_mono_int16(self, tmp_path):
        """Load a mono 16-bit WAV."""
        sr = 44100
        data = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        path = tmp_path / "mono.wav"
        wavfile.write(str(path), sr, data)
        result = load_wav(path)
        assert result.success
        assert result.signal.metadata.sample_rate == sr
        assert result.signal.metadata.format == SignalFormat.WAV
        assert result.signal.metadata.num_channels == 1
        assert result.signal.metadata.iq_layout == IQLayout.REAL_ONLY
        assert len(result.signal.samples) == 5

    def test_load_mono_float32(self, tmp_path):
        """Load a mono 32-bit float WAV."""
        sr = 48000
        data = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        path = tmp_path / "mono_float.wav"
        wavfile.write(str(path), sr, data)
        result = load_wav(path)
        assert result.success
        assert result.signal.metadata.sample_rate == sr
        assert len(result.signal.samples) == 5


class TestWavLoaderStereo:
    def test_stereo_default_uses_ch0(self, tmp_path):
        """Stereo WAV without IQ flag uses channel 0 only."""
        sr = 22050
        data = np.column_stack([
            np.array([1000, 2000, 3000], dtype=np.int16),
            np.array([4000, 5000, 6000], dtype=np.int16),
        ])
        path = tmp_path / "stereo.wav"
        wavfile.write(str(path), sr, data)
        result = load_wav(path, treat_stereo_as_iq=False)
        assert result.success
        assert result.signal.metadata.iq_layout == IQLayout.REAL_ONLY
        assert len(result.signal.samples) == 3
        # Should warn about stereo
        all_warnings = result.warnings + result.signal.metadata.warnings
        assert any("stereo" in w.lower() or "channel" in w.lower() for w in all_warnings)

    def test_stereo_as_iq(self, tmp_path):
        """Stereo WAV treated as IQ: ch0=I, ch1=Q."""
        sr = 48000
        i_data = np.array([16384, 0, -16384], dtype=np.int16)
        q_data = np.array([0, 16384, 0], dtype=np.int16)
        data = np.column_stack([i_data, q_data])
        path = tmp_path / "iq.wav"
        wavfile.write(str(path), sr, data)
        result = load_wav(path, treat_stereo_as_iq=True)
        assert result.success
        assert result.signal.metadata.iq_layout == IQLayout.IQ_INTERLEAVED
        assert result.signal.is_complex
        assert len(result.signal.samples) == 3
        # Check I/Q values: 16384/32768 = 0.5
        assert abs(result.signal.samples[0].real - 0.5) < 0.01
        assert abs(result.signal.samples[0].imag) < 0.01


class TestWavLoaderEdgeCases:
    def test_nonexistent_file(self, tmp_path):
        result = load_wav(tmp_path / "nope.wav")
        assert not result.success

    def test_not_a_wav_file(self, tmp_path):
        """A file that's not actually WAV."""
        path = tmp_path / "fake.wav"
        path.write_bytes(b"NOT A WAV FILE AT ALL")
        result = load_wav(path)
        assert not result.success

    def test_metadata_extraction(self, tmp_path):
        sr = 96000
        data = np.zeros(1000, dtype=np.int16)
        path = tmp_path / "meta.wav"
        wavfile.write(str(path), sr, data)
        result = load_wav(path)
        assert result.success
        meta = result.signal.metadata
        assert meta.sample_rate == 96000
        assert meta.bits_per_sample == 16
        assert meta.duration_seconds is not None
        assert abs(meta.duration_seconds - 1000 / 96000) < 0.001
