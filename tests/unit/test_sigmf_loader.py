import numpy as np
import json
import pytest
from pathlib import Path
from core.file_io.sigmf_loader import load_sigmf
from core.common.enums import SignalFormat


def _create_sigmf_pair(tmp_path, basename, global_meta, data_array):
    """Helper to create a .sigmf-meta + .sigmf-data pair."""
    meta_path = tmp_path / f"{basename}.sigmf-meta"
    data_path = tmp_path / f"{basename}.sigmf-data"
    
    full_meta = {
        "global": global_meta,
        "captures": [],
        "annotations": [],
    }
    meta_path.write_text(json.dumps(full_meta, indent=2))
    data_array.tofile(str(data_path))
    return meta_path


class TestSigMFLoader:
    def test_load_cf32_le(self, tmp_path):
        """Load a cf32_le SigMF recording."""
        data = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
        meta_path = _create_sigmf_pair(
            tmp_path, "test",
            {
                "core:datatype": "cf32_le",
                "core:sample_rate": 1000000,
                "core:version": "1.0.0",
            },
            data,
        )
        result = load_sigmf(meta_path)
        assert result.success
        assert result.signal is not None
        assert len(result.signal.samples) == 3
        assert result.signal.metadata.sample_rate == 1e6
        assert result.signal.metadata.format == SignalFormat.SIGMF

    def test_load_ri16_le(self, tmp_path):
        """Load a ri16_le (real int16) SigMF recording."""
        data = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        meta_path = _create_sigmf_pair(
            tmp_path, "real16",
            {
                "core:datatype": "ri16_le",
                "core:sample_rate": 48000,
                "core:version": "1.0.0",
            },
            data,
        )
        result = load_sigmf(meta_path)
        assert result.success
        assert len(result.signal.samples) == 4

    def test_load_with_captures_frequency(self, tmp_path):
        """Load SigMF with center frequency in captures."""
        data = np.zeros(10, dtype=np.complex64)
        meta_path = tmp_path / "freq.sigmf-meta"
        data_path = tmp_path / "freq.sigmf-data"
        meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": 2400000,
                "core:version": "1.0.0",
            },
            "captures": [
                {"core:sample_start": 0, "core:frequency": 433920000}
            ],
            "annotations": [],
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        data.tofile(str(data_path))
        result = load_sigmf(meta_path)
        assert result.success
        assert result.signal.metadata.center_frequency == 433920000

    def test_missing_data_file(self, tmp_path):
        """Meta file exists but data file is missing."""
        meta_path = tmp_path / "orphan.sigmf-meta"
        meta = {
            "global": {"core:datatype": "cf32_le", "core:version": "1.0.0"},
            "captures": [],
            "annotations": [],
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        result = load_sigmf(meta_path)
        assert not result.success

    def test_missing_datatype(self, tmp_path):
        """Meta file has no core:datatype."""
        data = np.zeros(10, dtype=np.complex64)
        meta_path = _create_sigmf_pair(
            tmp_path, "notype",
            {"core:version": "1.0.0"},  # no datatype
            data,
        )
        result = load_sigmf(meta_path)
        assert not result.success

    def test_missing_sample_rate_warning(self, tmp_path):
        """No sample_rate should produce a warning, not failure."""
        data = np.zeros(10, dtype=np.complex64)
        meta_path = _create_sigmf_pair(
            tmp_path, "nosr",
            {
                "core:datatype": "cf32_le",
                "core:version": "1.0.0",
                # no core:sample_rate
            },
            data,
        )
        result = load_sigmf(meta_path)
        assert result.success
        all_warnings = result.warnings + result.signal.metadata.warnings
        assert any("sample rate" in w.lower() for w in all_warnings)

    def test_nonexistent_meta_file(self, tmp_path):
        result = load_sigmf(tmp_path / "ghost.sigmf-meta")
        assert not result.success
