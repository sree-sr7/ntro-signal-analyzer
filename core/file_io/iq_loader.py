"""Raw IQ file loader.

Loads raw binary IQ files given an explicit IQConfig that describes
the data type, layout, endianness, and optional header offset.

Raw IQ files contain NO embedded metadata — all interpretation depends
on the caller-provided configuration.
"""

import numpy as np
from pathlib import Path

from core.common.enums import (
    SignalFormat, IQLayout, Endianness, SampleDtype,
)
from core.common.models import IQConfig, SignalMetadata, SignalData, LoadResult
from core.common.validation import validate_file_path, validate_iq_config_for_loading


def load_iq(path: Path, config: IQConfig) -> LoadResult:
    """Load a raw IQ binary file.

    Args:
        path: Path to the raw binary file.
        config: Explicit configuration describing the binary format.

    Returns:
        LoadResult with loaded signal or structured errors.
    """
    try:
        # --- validate inputs ---
        file_val = validate_file_path(path)
        if not file_val.valid:
            return LoadResult.fail(file_val.errors, file_val.warnings)

        config_val = validate_iq_config_for_loading(config)
        if not config_val.valid:
            return LoadResult.fail(config_val.errors, config_val.warnings)

        warnings: list[str] = []

        # --- resolve numpy dtype and byte order ---
        byte_order = ">" if config.endianness == Endianness.BIG else "<"

        _DTYPE_MAP: dict[SampleDtype, type] = {
            SampleDtype.COMPLEX64:  np.complex64,
            SampleDtype.COMPLEX128: np.complex128,
            SampleDtype.FLOAT32:    np.float32,
            SampleDtype.FLOAT64:    np.float64,
            SampleDtype.INT16:      np.int16,
            SampleDtype.INT8:       np.int8,
            SampleDtype.UINT8:      np.uint8,
        }

        raw_np_dtype = np.dtype(_DTYPE_MAP[config.dtype]).newbyteorder(byte_order)

        # --- compute how many samples fit ---
        file_size = path.stat().st_size
        data_size = file_size - config.header_offset_bytes
        if data_size <= 0:
            return LoadResult.fail(
                [f"No data after header offset ({config.header_offset_bytes} bytes "
                 f"in a {file_size}-byte file)"]
            )

        is_native_complex = config.dtype in (
            SampleDtype.COMPLEX64, SampleDtype.COMPLEX128,
        )

        # For native complex types each element IS one sample.
        # For interleaved types, two consecutive elements form one sample.
        bytes_per_element = raw_np_dtype.itemsize
        elements_per_sample = 1 if is_native_complex else 2
        bytes_per_sample = bytes_per_element * elements_per_sample

        num_complete_samples = data_size // bytes_per_sample
        remainder = data_size % bytes_per_sample

        if remainder != 0:
            warnings.append(
                f"File size does not divide evenly into samples — "
                f"truncating {remainder} trailing byte(s)."
            )

        if num_complete_samples == 0:
            return LoadResult.fail(
                ["File contains no complete samples after header offset."]
            )

        # --- read raw binary ---
        elements_to_read = num_complete_samples * elements_per_sample
        with open(path, "rb") as fh:
            fh.seek(config.header_offset_bytes)
            raw = np.fromfile(fh, dtype=raw_np_dtype, count=elements_to_read)

        # --- convert to complex samples ---
        if is_native_complex:
            samples = raw.astype(
                np.complex64 if config.dtype == SampleDtype.COMPLEX64
                else np.complex128
            )
        else:
            # Separate I and Q channels
            if config.iq_layout == IQLayout.QI_INTERLEAVED:
                q_raw = raw[0::2]
                i_raw = raw[1::2]
            else:
                # IQ_INTERLEAVED is the default for non-complex types
                i_raw = raw[0::2]
                q_raw = raw[1::2]

            # Normalize integer types to [-1, 1) float range
            if config.dtype == SampleDtype.INT16:
                i_f = i_raw.astype(np.float32) / 32768.0
                q_f = q_raw.astype(np.float32) / 32768.0
            elif config.dtype == SampleDtype.INT8:
                i_f = i_raw.astype(np.float32) / 128.0
                q_f = q_raw.astype(np.float32) / 128.0
            elif config.dtype == SampleDtype.UINT8:
                i_f = (i_raw.astype(np.float32) - 127.5) / 128.0
                q_f = (q_raw.astype(np.float32) - 127.5) / 128.0
            elif config.dtype == SampleDtype.FLOAT64:
                i_f = i_raw.astype(np.float64)
                q_f = q_raw.astype(np.float64)
            else:
                # FLOAT32 — no normalization needed
                i_f = i_raw.astype(np.float32)
                q_f = q_raw.astype(np.float32)

            if config.dtype == SampleDtype.FLOAT64:
                samples = (i_f + 1j * q_f).astype(np.complex128)
            else:
                samples = (i_f + 1j * q_f).astype(np.complex64)

        # --- build metadata ---
        if config.sample_rate is None:
            warnings.append(
                "Sample rate not provided for raw IQ file — "
                "required for frequency-domain analysis"
            )
        if config.center_frequency is None:
            warnings.append("Center frequency not provided")

        duration: float | None = None
        if config.sample_rate is not None and config.sample_rate > 0:
            duration = num_complete_samples / config.sample_rate

        metadata = SignalMetadata(
            source_path=path,
            format=SignalFormat.IQ,
            sample_rate=config.sample_rate,
            center_frequency=config.center_frequency,
            sample_count=num_complete_samples,
            duration_seconds=duration,
            dtype=config.dtype,
            iq_layout=config.iq_layout,
            endianness=config.endianness,
            num_channels=1,
            bits_per_sample=raw_np_dtype.itemsize * 8,
            warnings=list(warnings),
        )

        return LoadResult.ok(
            SignalData(samples=samples, metadata=metadata),
            warnings=warnings if warnings else None,
        )

    except Exception as exc:
        return LoadResult.fail([f"Unexpected error loading IQ file: {exc}"])
