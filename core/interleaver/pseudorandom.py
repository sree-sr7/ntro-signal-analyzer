"""Reproducible project-scoped pseudo-random block interleaver.

The default profile is not a historical repository mode or a communications
standard. It uses a SplitMix64 stream to drive Durstenfeld's descending
Fisher-Yates permutation and applies that one permutation independently to
each complete 256-bit frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.bitstream.extractor import validate_binary_bits


_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15


@dataclass(frozen=True)
class PseudoRandomInterleaverConfig:
    """Explicit parameters for a reproducible pseudo-random block profile."""

    frame_length: int = 256
    seed: int = 26147

    def __post_init__(self) -> None:
        if (
            not isinstance(self.frame_length, (int, np.integer))
            or isinstance(self.frame_length, (bool, np.bool_))
            or self.frame_length <= 0
        ):
            raise ValueError("frame_length must be a positive integer")
        if (
            not isinstance(self.seed, (int, np.integer))
            or isinstance(self.seed, (bool, np.bool_))
            or not 0 <= self.seed <= _MASK64
        ):
            raise ValueError("seed must be an integer in [0, 2**64 - 1]")
        object.__setattr__(self, "frame_length", int(self.frame_length))
        object.__setattr__(self, "seed", int(self.seed))


def _splitmix64_next(state: int) -> tuple[int, int]:
    """Return one SplitMix64 output and the following 64-bit state."""
    state = (state + _GAMMA) & _MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31), state


def _randbelow(bound: int, state: int) -> tuple[int, int]:
    """Draw without modulo bias from the specified SplitMix64 stream."""
    limit = (1 << 64) - ((1 << 64) % bound)
    while True:
        value, state = _splitmix64_next(state)
        if value < limit:
            return value % bound, state


def permutation_for(
    configuration: PseudoRandomInterleaverConfig = PseudoRandomInterleaverConfig(),
) -> np.ndarray:
    """Return output-position to input-position indices for one frame.

    The returned array is immutable and directly inspectable. Forward
    interleaving uses output = input[permutation].
    """
    if not isinstance(configuration, PseudoRandomInterleaverConfig):
        raise TypeError("configuration must be PseudoRandomInterleaverConfig")
    permutation = list(range(configuration.frame_length))
    state = configuration.seed
    # Durstenfeld's descending Fisher-Yates form; bounded draws use rejection
    # sampling so modulo reduction does not bias the choice.
    for position in range(configuration.frame_length - 1, 0, -1):
        selected, state = _randbelow(position + 1, state)
        permutation[position], permutation[selected] = (
            permutation[selected],
            permutation[position],
        )
    result = np.asarray(permutation, dtype=np.intp)
    result.flags.writeable = False
    return result


class PseudoRandomInterleaver:
    """Interleave complete frames with one fixed, seed-derived permutation."""

    def __init__(
        self,
        configuration: PseudoRandomInterleaverConfig = PseudoRandomInterleaverConfig(),
    ) -> None:
        if not isinstance(configuration, PseudoRandomInterleaverConfig):
            raise TypeError("configuration must be PseudoRandomInterleaverConfig")
        self.configuration = configuration
        self.permutation = permutation_for(configuration)
        inverse = np.argsort(self.permutation)
        inverse.flags.writeable = False
        self.inverse_permutation = inverse

    @property
    def name(self) -> str:
        return f"PseudoRandom_{self.configuration.frame_length}_Seed_{self.configuration.seed}"

    def _validate_length(self, count: int) -> None:
        frame_length = self.configuration.frame_length
        if count == 0 or count % frame_length:
            raise ValueError(
                f"bit count must be a positive multiple of pseudo-random frame length {frame_length}"
            )

    def interleave_values(self, values: np.ndarray) -> np.ndarray:
        """Apply this frame permutation to finite numeric values such as LLRs."""
        array = np.asarray(values)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
            raise ValueError("values must be a one-dimensional numeric array")
        if not np.all(np.isfinite(array)):
            raise ValueError("values must contain only finite numbers")
        self._validate_length(array.size)
        return array.reshape(-1, self.configuration.frame_length)[:, self.permutation].reshape(-1).copy()

    def deinterleave_values(self, values: np.ndarray) -> np.ndarray:
        """Invert the numeric-value permutation without changing frame lengths."""
        array = np.asarray(values)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
            raise ValueError("values must be a one-dimensional numeric array")
        if not np.all(np.isfinite(array)):
            raise ValueError("values must contain only finite numbers")
        self._validate_length(array.size)
        return array.reshape(-1, self.configuration.frame_length)[:, self.inverse_permutation].reshape(-1).copy()

    def interleave(self, bits: np.ndarray) -> np.ndarray:
        """Interleave one or more complete binary frames."""
        values = validate_binary_bits(bits)
        self._validate_length(values.size)
        return self.interleave_values(values)

    def deinterleave(self, bits: np.ndarray) -> np.ndarray:
        """Restore one or more complete binary frames."""
        values = validate_binary_bits(bits)
        self._validate_length(values.size)
        return self.deinterleave_values(values)


DEFAULT_PSEUDORANDOM_CONFIG = PseudoRandomInterleaverConfig()


def interleave(
    bits: np.ndarray,
    configuration: PseudoRandomInterleaverConfig = DEFAULT_PSEUDORANDOM_CONFIG,
) -> np.ndarray:
    """Functional API for the project's default pseudo-random family."""
    return PseudoRandomInterleaver(configuration).interleave(bits)


def deinterleave(
    bits: np.ndarray,
    configuration: PseudoRandomInterleaverConfig = DEFAULT_PSEUDORANDOM_CONFIG,
) -> np.ndarray:
    """Functional inverse for the pseudo-random interleave function."""
    return PseudoRandomInterleaver(configuration).deinterleave(bits)
