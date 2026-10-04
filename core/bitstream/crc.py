"""CRC-16/CCITT-FALSE for byte strings or MSB-first bit streams.

Convention: width 16, polynomial 0x1021, initial value 0xFFFF, no input or
output reflection, and xorout 0x0000. Bytes and bits are processed most
significant bit first. Appended FCS bits/bytes are also most-significant first
(the high CRC byte precedes the low CRC byte).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .extractor import bytes_to_bits, validate_binary_bits


CRC16_POLYNOMIAL = 0x1021
CRC16_INITIAL = 0xFFFF
CRC16_XOROUT = 0x0000


def _as_bits(data: np.ndarray | Sequence[int]) -> np.ndarray:
    """Validate binary integer sequences for CRC bit-mode APIs."""
    values = np.asarray(data)
    if values.ndim == 1 and values.size == 0:
        return np.empty(0, dtype=np.uint8)
    if values.ndim != 1 or values.dtype.kind not in "biu":
        raise TypeError("bit input must be a one-dimensional integer sequence")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("bit input must contain only zero and one")
    return validate_binary_bits(values.astype(np.uint8, copy=False))


def _is_bytes(data: object) -> bool:
    return isinstance(data, (bytes, bytearray, memoryview))


def _crc16_bits(bits: np.ndarray) -> int:
    crc = CRC16_INITIAL
    for bit in bits:
        feedback = ((crc >> 15) & 1) ^ int(bit)
        crc = (crc << 1) & 0xFFFF
        if feedback:
            crc ^= CRC16_POLYNOMIAL
    return crc ^ CRC16_XOROUT


def compute_crc16(data: np.ndarray | Sequence[int] | bytes | bytearray | memoryview) -> int:
    """Compute CRC-16/CCITT-FALSE over bytes or a binary bit sequence.

    Bytes-like inputs are consumed as octets; other inputs are interpreted as
    one-dimensional bits. Bit inputs need not end on a byte boundary.
    """
    bits = bytes_to_bits(data) if _is_bytes(data) else _as_bits(data)
    return _crc16_bits(bits)


def append_crc16(
    data: np.ndarray | Sequence[int] | bytes | bytearray | memoryview,
) -> np.ndarray | bytes:
    """Append the CRC FCS, preserving bit mode as ``uint8`` or bytes mode as bytes."""
    if _is_bytes(data):
        raw = bytes(data)
        crc = compute_crc16(raw)
        return raw + crc.to_bytes(2, byteorder="big")
    bits = _as_bits(data)
    crc = compute_crc16(bits)
    fcs_bits = np.unpackbits(np.frombuffer(crc.to_bytes(2, "big"), dtype=np.uint8), bitorder="big")
    return np.concatenate((bits, fcs_bits)).astype(np.uint8, copy=False)


def verify_crc16(data: np.ndarray | Sequence[int] | bytes | bytearray | memoryview) -> bool:
    """Return whether input data ends with a valid CRC-16/CCITT-FALSE FCS.

    Inputs shorter than the 16-bit FCS return ``False``. Invalid bit values or
    unsupported input types raise a validation exception.
    """
    if _is_bytes(data):
        raw = bytes(data)
        if len(raw) < 2:
            return False
        message, received_fcs = raw[:-2], int.from_bytes(raw[-2:], "big")
        return compute_crc16(message) == received_fcs
    bits = _as_bits(data)
    if bits.size < 16:
        return False
    message = bits[:-16]
    received_fcs = int("".join(str(int(bit)) for bit in bits[-16:]), 2)
    return compute_crc16(message) == received_fcs
