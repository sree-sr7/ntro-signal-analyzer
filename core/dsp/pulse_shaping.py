"""Root-raised-cosine pulse generation and matched filtering."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from ._validation import as_samples


def root_raised_cosine(
    rolloff: float,
    samples_per_symbol: int,
    span: int,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Create an energy-normalized, symmetric root-raised-cosine FIR.

    The odd-length filter spans ``span`` symbols, centered at zero. Its tap
    energy is normalized to one, making it suitable for matched-filter use.
    """
    beta = float(rolloff)
    if not np.isfinite(beta) or beta < 0 or beta > 1:
        raise ValueError("rolloff must be finite and in [0, 1]")
    if not isinstance(samples_per_symbol, (int, np.integer)) or samples_per_symbol < 2:
        raise ValueError("samples_per_symbol must be an integer of at least 2")
    if not isinstance(span, (int, np.integer)) or span < 1:
        raise ValueError("span must be a positive integer")
    target_dtype = np.dtype(dtype)
    if target_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("dtype must be float32 or float64")

    count = int(span) * int(samples_per_symbol)
    if count % 2:
        count += 1
    t = np.arange(-count // 2, count // 2 + 1, dtype=np.float64) / samples_per_symbol
    h = np.empty_like(t)
    zero = np.isclose(t, 0.0, atol=1e-14)
    if beta == 0.0:
        h[zero] = 1.0
        h[~zero] = np.sin(np.pi * t[~zero]) / (np.pi * t[~zero])
    else:
        singular = np.isclose(np.abs(t), 1.0 / (4.0 * beta), atol=1e-12)
        general = ~(zero | singular)
        h[zero] = 1.0 + beta * (4.0 / np.pi - 1.0)
        ts = t[singular]
        h[singular] = (beta / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
        )
        tg = t[general]
        h[general] = (
            np.sin(np.pi * tg * (1.0 - beta))
            + 4.0 * beta * tg * np.cos(np.pi * tg * (1.0 + beta))
        ) / (np.pi * tg * (1.0 - (4.0 * beta * tg) ** 2))
    h /= np.sqrt(np.sum(h * h))
    return h.astype(target_dtype, copy=False)


def matched_filter(samples: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply the time-reversed conjugate of a real or complex FIR to samples.

    Output length equals input length (centered convolution). A complex64 input
    remains complex64; complex128 remains complex128. Real input preserves
    float32/float64 precision where SciPy supports it.
    """
    values = as_samples(samples)
    taps = as_samples(coefficients, name="coefficients")
    if taps.size == 0:
        raise ValueError("coefficients must not be empty")
    if values.size == 0:
        return np.empty(0, dtype=values.dtype)
    dtype = np.result_type(values.dtype, taps.dtype)
    filtered = fftconvolve(values, np.conjugate(taps[::-1]), mode="same")
    if np.iscomplexobj(values) or np.iscomplexobj(taps):
        if values.dtype == np.dtype(np.complex64) and taps.dtype in (np.dtype(np.float32), np.dtype(np.complex64)):
            dtype = np.dtype(np.complex64)
        else:
            dtype = np.dtype(np.complex128)
    elif values.dtype == np.dtype(np.float32) and taps.dtype == np.dtype(np.float32):
        dtype = np.dtype(np.float32)
    else:
        dtype = np.dtype(np.float64)
    return np.asarray(filtered, dtype=dtype)
