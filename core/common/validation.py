import numpy as np
from pathlib import Path
from typing import Optional
from .models import SignalData, SignalMetadata, ValidationResult
from .enums import SampleDtype

def validate_file_path(path: Path) -> ValidationResult:
    """Validate that a file path exists and is readable."""
    result = ValidationResult(valid=True)
    if not path.exists():
        result.add_error(f"File does not exist: {path}")
    elif not path.is_file():
        result.add_error(f"Path is not a file: {path}")
    elif path.stat().st_size == 0:
        result.add_error(f"File is empty: {path}")
    return result

def validate_signal_data(signal: SignalData) -> ValidationResult:
    """Validate loaded signal data for basic integrity."""
    result = ValidationResult(valid=True)
    
    # Check samples exist and are non-empty
    if signal.samples is None:
        result.add_error("Signal samples array is None")
        return result
    if signal.samples.size == 0:
        result.add_error("Signal samples array is empty")
        return result
    
    # Check for non-finite values
    if not np.all(np.isfinite(signal.samples)):
        non_finite_count = int(np.sum(~np.isfinite(signal.samples)))
        total = signal.samples.size
        ratio = non_finite_count / total
        if ratio > 0.5:
            result.add_error(
                f"Signal contains {non_finite_count}/{total} non-finite values "
                f"({ratio:.1%}) — data is likely corrupt"
            )
        else:
            result.add_warning(
                f"Signal contains {non_finite_count}/{total} non-finite values "
                f"({ratio:.1%})"
            )
    
    # Check metadata consistency
    meta = signal.metadata
    if meta.sample_count != len(signal.samples):
        result.add_warning(
            f"Metadata sample_count ({meta.sample_count}) does not match "
            f"actual sample count ({len(signal.samples)})"
        )
    
    if meta.has_sample_rate:
        if meta.sample_rate <= 0:
            result.add_error(f"Invalid sample rate: {meta.sample_rate}")
        elif meta.duration_seconds is not None:
            expected_duration = len(signal.samples) / meta.sample_rate
            if abs(expected_duration - meta.duration_seconds) > 0.001:
                result.add_warning(
                    f"Duration mismatch: metadata says {meta.duration_seconds:.6f}s, "
                    f"computed {expected_duration:.6f}s"
                )
    
    return result

def validate_sample_rate(sample_rate: Optional[float]) -> ValidationResult:
    """Validate a sample rate value."""
    result = ValidationResult(valid=True)
    if sample_rate is None:
        result.add_error("Sample rate is not specified")
    elif sample_rate <= 0:
        result.add_error(f"Sample rate must be positive, got {sample_rate}")
    elif sample_rate > 1e12:
        result.add_warning(f"Unusually high sample rate: {sample_rate} Hz")
    return result

def validate_iq_config_for_loading(config) -> ValidationResult:
    """Validate IQ configuration has enough info to load a file."""
    from .enums import SampleDtype, IQLayout
    result = ValidationResult(valid=True)
    
    if config.dtype is None:
        result.add_error("dtype must be specified for raw IQ loading")
    
    if config.header_offset_bytes < 0:
        result.add_error(f"header_offset_bytes cannot be negative: {config.header_offset_bytes}")
    
    return result
