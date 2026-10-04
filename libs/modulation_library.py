"""Shared constellation, bit-label, and normalization definitions.

Bit groups are transmitted most-significant bit first. PSK constellations have
unit energy. Rectangular 16-QAM uses I/Q levels ±1 and ±3 normalized by
``sqrt(10)``. 2FSK's symbol representation is the frequency state ``[-1,+1]``;
the configured tone pair is supplied to its waveform modulator/demodulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.common.enums import ModulationType


@dataclass(frozen=True)
class ModulationDefinition:
    """Immutable mapping metadata for one supported modulation family."""

    modulation: ModulationType
    constellation: np.ndarray
    bit_labels: tuple[str, ...]
    normalization: str

    @property
    def bits_per_symbol(self) -> int:
        """Number of input bits represented by each symbol."""
        return len(self.bit_labels[0])


def _definition(
    modulation: ModulationType,
    constellation: np.ndarray,
    labels: tuple[str, ...],
    normalization: str,
) -> ModulationDefinition:
    values = np.asarray(constellation)
    values.setflags(write=False)
    return ModulationDefinition(modulation, values, labels, normalization)


_GRAY_8 = ("000", "001", "011", "010", "110", "111", "101", "100")
_GRAY_16QAM_AXIS = np.array([-3.0, -1.0, 3.0, 1.0])
_QAM16 = np.array([
    (_GRAY_16QAM_AXIS[index >> 2] + 1j * _GRAY_16QAM_AXIS[index & 3]) / np.sqrt(10.0)
    for index in range(16)
])

MODULATION_DEFINITIONS: dict[ModulationType, ModulationDefinition] = {
    ModulationType.BPSK: _definition(
        ModulationType.BPSK, np.array([1.0 + 0j, -1.0 + 0j]), ("0", "1"), "unit energy"
    ),
    ModulationType.QPSK: _definition(
        ModulationType.QPSK,
        np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2)),
        ("00", "01", "11", "10"),
        "unit energy; Gray phase order beginning at 45 degrees",
    ),
    ModulationType.PSK8: _definition(
        ModulationType.PSK8,
        np.exp(1j * np.arange(8) * (2 * np.pi / 8)),
        _GRAY_8,
        "unit energy; Gray phase order beginning at 0 degrees",
    ),
    ModulationType.QAM16: _definition(
        ModulationType.QAM16,
        _QAM16,
        tuple(f"{index:04b}" for index in range(16)),
        "rectangular Gray-coded I/Q levels ±1 and ±3 divided by sqrt(10)",
    ),
    ModulationType.FSK2: _definition(
        ModulationType.FSK2,
        np.array([-1.0, 1.0]),
        ("0", "1"),
        "frequency state -1 for bit 0 and +1 for bit 1",
    ),
}


def modulation_definition(modulation: ModulationType) -> ModulationDefinition:
    """Return the central mapping definition for a supported modulation."""
    try:
        return MODULATION_DEFINITIONS[ModulationType(modulation)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported modulation: {modulation}") from exc


def constellation_for(modulation: ModulationType) -> np.ndarray:
    """Return a copy of the defined symbol values for a modulation family."""
    return modulation_definition(modulation).constellation.copy()


def indices_to_symbols(indices: np.ndarray, modulation: ModulationType) -> np.ndarray:
    """Map integer symbol indices to the shared constellation values."""
    definition = modulation_definition(modulation)
    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise TypeError("indices must be a one-dimensional integer array")
    if np.any(values < 0) or np.any(values >= len(definition.bit_labels)):
        raise ValueError("symbol index is outside the modulation constellation")
    return definition.constellation[values]


def symbol_indices_to_bits(indices: np.ndarray, modulation: ModulationType) -> np.ndarray:
    """Map integer symbol indices to an MSB-first flat uint8 bit array."""
    definition = modulation_definition(modulation)
    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise TypeError("indices must be a one-dimensional integer array")
    if np.any(values < 0) or np.any(values >= len(definition.bit_labels)):
        raise ValueError("symbol index is outside the modulation constellation")
    labels = np.asarray([[int(bit) for bit in label] for label in definition.bit_labels], dtype=np.uint8)
    return labels[values].reshape(-1)


def bits_to_symbols(bits: np.ndarray, modulation: ModulationType) -> np.ndarray:
    """Map a flat binary array to the shared PSK/QAM symbols or FSK states."""
    definition = modulation_definition(modulation)
    values = np.asarray(bits)
    if values.ndim != 1 or values.dtype.kind not in "biu":
        raise TypeError("bits must be a one-dimensional integer or boolean array")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("bits must contain only zero and one")
    width = definition.bits_per_symbol
    if values.size % width:
        raise ValueError(f"bit count must be a multiple of {width}")
    groups = values.astype(np.uint8, copy=False).reshape(-1, width)
    weights = 1 << np.arange(width - 1, -1, -1, dtype=np.uint8)
    binary_codes = groups @ weights
    lookup = np.full(1 << width, -1, dtype=np.int64)
    for index, label in enumerate(definition.bit_labels):
        lookup[int(label, 2)] = index
    indices = lookup[binary_codes]
    if np.any(indices < 0):
        raise ValueError("bit mapping is incomplete for this modulation")
    return definition.constellation[indices]


def symbols_to_indices(samples: np.ndarray, modulation: ModulationType) -> np.ndarray:
    """Return nearest shared-constellation indices for symbol-rate samples."""
    definition = modulation_definition(modulation)
    values = np.asarray(samples)
    if values.ndim != 1 or values.dtype.kind not in "fci":
        raise TypeError("samples must be a one-dimensional numeric array")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values")
    distances = np.abs(values[:, None] - definition.constellation[None, :])
    return np.argmin(distances, axis=1).astype(np.int64)
