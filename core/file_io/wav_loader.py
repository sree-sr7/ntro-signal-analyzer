"""WAV file loader.

Loads WAV files using scipy.io.wavfile, extracting metadata from
the RIFF container header. Supports mono and stereo PCM formats.

Stereo WAV files are NOT automatically interpreted as IQ data.
The caller must explicitly set treat_stereo_as_iq=True.
"""

import numpy as np
import scipy.io.wavfile
from pathlib import Path

from core.common.enums import SignalFormat, IQLayout, SampleDtype
from core.common.models import SignalMetadata, SignalData, LoadResult
from core.common.validation import validate_file_path


# Maximum integer values for normalization to [-1.0, 1.0) range.
_INT_NORM: dict[type, float] = {
    np.dtype("int8"):  128.0,
    np.dtype("int16"): 32768.0,
    np.dtype("int32"): 2147483648.0,
}


def load_wav(path: Path, treat_stereo_as_iq: bool = False) -> LoadResult:
    """Load a WAV audio file.

    Args:
        path: Path to the .wav file.
        treat_stereo_as_iq: If True, interpret stereo channels as I (ch0)
            and Q (ch1). If False (default), use channel 0 only and warn.

    Returns:
        LoadResult with loaded signal or structured errors.
    """
    try:
        # --- validate path ---
        file_val = validate_file_path(path)
        if not file_val.valid:
            return LoadResult.fail(file_val.errors, file_val.warnings)

        warnings: list[str] = []

        # --- read WAV via scipy ---
        try:
            sample_rate, data = scipy.io.wavfile.read(str(path))
        except Exception as exc:
            return LoadResult.fail([f"Failed to read WAV file: {exc}"])

        # --- extract shape info ---
        if data.ndim == 1:
            num_channels = 1
            sample_count = len(data)
        elif data.ndim == 2:
            sample_count, num_channels = data.shape
        else:
            return LoadResult.fail(
                [f"Unexpected WAV data dimensions: {data.ndim}"]
            )

        if num_channels > 2:
            return LoadResult.fail(
                ["WAV files with more than 2 channels are not supported"]
            )

        if sample_count == 0:
            return LoadResult.fail(["WAV file contains no samples"])

        # --- record source dtype info ---
        wav_dtype = data.dtype
        bits_per_sample = wav_dtype.itemsize * 8

        # Map to SampleDtype enum
        dtype_enum_map = {
            np.dtype("int8"):    SampleDtype.INT8,
            np.dtype("int16"):   SampleDtype.INT16,
            np.dtype("int32"):   SampleDtype.FLOAT32,  # normalized to float
            np.dtype("float32"): SampleDtype.FLOAT32,
            np.dtype("float64"): SampleDtype.FLOAT64,
        }
        sample_dtype = dtype_enum_map.get(wav_dtype, SampleDtype.FLOAT32)

        # --- normalize to float ---
        norm_divisor = _INT_NORM.get(wav_dtype)
        if norm_divisor is not None:
            data_norm = data.astype(np.float64) / norm_divisor
        else:
            # float32 or float64 — use as-is
            data_norm = data.astype(np.float64)

        # --- build complex samples ---
        if num_channels == 1:
            samples = data_norm.astype(np.float32) + 0j
            iq_layout = IQLayout.REAL_ONLY
        elif treat_stereo_as_iq:
            i_ch = data_norm[:, 0].astype(np.float32)
            q_ch = data_norm[:, 1].astype(np.float32)
            samples = (i_ch + 1j * q_ch).astype(np.complex64)
            iq_layout = IQLayout.IQ_INTERLEAVED
        else:
            samples = data_norm[:, 0].astype(np.float32) + 0j
            iq_layout = IQLayout.REAL_ONLY
            warnings.append(
                "Stereo WAV loaded — using channel 0 only. "
                "Set treat_stereo_as_iq=True to interpret as IQ data."
            )

        # --- build metadata ---
        duration = float(sample_count) / float(sample_rate)

        metadata = SignalMetadata(
            source_path=path,
            format=SignalFormat.WAV,
            sample_rate=float(sample_rate),
            center_frequency=None,
            sample_count=sample_count,
            duration_seconds=duration,
            dtype=sample_dtype,
            iq_layout=iq_layout,
            num_channels=num_channels,
            bits_per_sample=bits_per_sample,
            source_metadata={
                "wav_sample_rate": int(sample_rate),
                "wav_channels": num_channels,
                "wav_dtype": str(wav_dtype),
            },
            warnings=list(warnings),
        )

        return LoadResult.ok(
            SignalData(samples=samples, metadata=metadata),
            warnings=warnings if warnings else None,
        )

    except Exception as exc:
        return LoadResult.fail([f"Unexpected error loading WAV file: {exc}"])
