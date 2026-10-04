"""Small shared input checks for numerical DSP routines."""

import numpy as np


def as_samples(samples: np.ndarray, *, name: str = "samples") -> np.ndarray:
    """Return a one-dimensional numeric sample view after checking finiteness."""
    array = np.asarray(samples)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.dtype.kind not in "fciu":
        raise TypeError(f"{name} must have a real or complex numeric dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_sample_rate(sample_rate: float) -> float:
    """Validate and return a finite, positive sample rate."""
    rate = float(sample_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    return rate
