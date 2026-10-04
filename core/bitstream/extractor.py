"""Validated bit/byte conversions and presentation helpers."""

from __future__ import annotations

import numpy as np


def validate_binary_bits(bits: np.ndarray) -> np.ndarray:
    """Validate and return a one-dimensional ``uint8`` array of zero/one bits.

    The input is not coerced: callers must make any dtype conversion explicit.
    The returned array may share storage with the input.
    """
    values = np.asarray(bits)
    if values.ndim != 1:
        raise ValueError("bits must be a one-dimensional array")
    if values.dtype != np.uint8:
        raise TypeError("bits must have dtype uint8")
    if np.any(values > 1):
        raise ValueError("bits must contain only zero and one")
    return values


def pad_bits(bits: np.ndarray, target_length: int, *, pad_value: int = 0) -> np.ndarray:
    """Explicitly extend a bit array to ``target_length`` with zero or one bits."""
    values = validate_binary_bits(bits)
    if not isinstance(target_length, (int, np.integer)) or isinstance(target_length, bool):
        raise TypeError("target_length must be an integer")
    if target_length < values.size:
        raise ValueError("target_length cannot be shorter than the input")
    if pad_value not in (0, 1):
        raise ValueError("pad_value must be zero or one")
    return np.pad(values, (0, int(target_length) - values.size), constant_values=pad_value).astype(
        np.uint8, copy=False
    )


def trim_bits(bits: np.ndarray, target_length: int) -> np.ndarray:
    """Explicitly retain the first ``target_length`` bits, discarding the tail."""
    values = validate_binary_bits(bits)
    if not isinstance(target_length, (int, np.integer)) or isinstance(target_length, bool):
        raise TypeError("target_length must be an integer")
    if target_length < 0 or target_length > values.size:
        raise ValueError("target_length must be between zero and the input length")
    return values[:int(target_length)].copy()


def bits_to_bytes(bits: np.ndarray, *, pad_to_byte: bool = False) -> bytes:
    """Pack MSB-first bits into bytes.

    A non-byte-aligned input is rejected unless ``pad_to_byte=True`` is
    explicitly supplied; explicit padding appends zero bits on the right.
    """
    values = validate_binary_bits(bits)
    remainder = values.size % 8
    if remainder:
        if not pad_to_byte:
            raise ValueError("bit count must be a multiple of eight unless pad_to_byte=True")
        values = pad_bits(values, values.size + 8 - remainder)
    return np.packbits(values, bitorder="big").tobytes()


def bytes_to_bits(data: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Unpack bytes to MSB-first ``uint8`` bits."""
    if isinstance(data, np.ndarray):
        values = np.asarray(data)
        if values.ndim != 1 or values.dtype != np.uint8:
            raise TypeError("byte array input must be one-dimensional uint8")
        octets = values
    else:
        try:
            octets = np.frombuffer(data, dtype=np.uint8)
        except (TypeError, ValueError) as exc:
            raise TypeError("data must be bytes-like or a one-dimensional uint8 array") from exc
    return np.unpackbits(octets, bitorder="big").astype(np.uint8, copy=False)


def format_bits_binary(bits: np.ndarray, *, group_size: int = 0, separator: str = " ") -> str:
    """Format bits as text, optionally separating fixed-size groups."""
    values = validate_binary_bits(bits)
    if not isinstance(group_size, (int, np.integer)) or isinstance(group_size, bool) or group_size < 0:
        raise ValueError("group_size must be a nonnegative integer")
    text = "".join("1" if bit else "0" for bit in values)
    if group_size == 0 or not text:
        return text
    return separator.join(text[start:start + group_size] for start in range(0, len(text), group_size))


def format_bits_hex(bits: np.ndarray, *, pad_to_nibble: bool = False) -> str:
    """Format MSB-first bits as uppercase hexadecimal without a prefix.

    Non-nibble-aligned data is rejected unless ``pad_to_nibble=True``; when
    enabled, zero bits are appended on the right to complete the last nibble.
    """
    values = validate_binary_bits(bits)
    remainder = values.size % 4
    if remainder:
        if not pad_to_nibble:
            raise ValueError("bit count must be a multiple of four unless pad_to_nibble=True")
        values = pad_bits(values, values.size + 4 - remainder)
    if values.size == 0:
        return ""
    packed = np.packbits(values.reshape(-1, 4), axis=1, bitorder="big")
    nibbles = (packed[:, 0] >> 4).astype(np.uint8)
    return "".join(f"{nibble:X}" for nibble in nibbles)
