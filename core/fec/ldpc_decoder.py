"""Strict IEEE 802.11n QC-LDPC encoders and normalized min-sum decoders.

The supported prototype rows are transcribed from IEEE P802.11n/D11.0,
Annex R, Tables R.1(a-c), R.2(a-c), and R.3(a-c).  The same rows were
cross-checked against the corresponding H_* tables in tavildar/LDPC's
LdpcM/LDPCCode.m.  A prototype value of -1 denotes an all-zero Z by Z block;
nonnegative values select the draft's cyclically right-shifted identity block.

This implementation accepts complete codewords only.  It intentionally does
not implement PPDU shortening, puncturing, padding, or partial-codeword rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypeAlias

import numpy as np

from core.bitstream.extractor import validate_binary_bits


IEEE_REFERENCE_ID = "IEEE80211N-D11.0-2009"
TAVILDAR_REFERENCE_ID = "TAVILDAR-LDPC-MATLAB"
LDPC_DECODER_ALGORITHM = "normalized_min_sum_llr"


def _parse_prototype(rows: str) -> tuple[tuple[int, ...], ...]:
    """Parse a compact, human-readable prototype table."""
    parsed = tuple(
        tuple(int(value) for value in line.split())
        for line in rows.strip().splitlines()
        if line.strip()
    )
    if not parsed or any(len(row) != len(parsed[0]) for row in parsed):
        raise ValueError("LDPC prototype must be a nonempty rectangular table")
    return parsed


# Annex R matrix prototypes.  Values and row/column order are preserved from
# the sources named in each LDPCConfig below; -1 is the null submatrix marker.
_PROTOTYPES: dict[str, tuple[tuple[int, ...], ...]] = {
    "LDPC_648_R1_2": _parse_prototype("""
        0 -1 -1 -1 0 0 -1 -1 0 -1 -1 0 1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
        22 0 -1 -1 17 -1 0 0 12 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1
        6 -1 0 -1 10 -1 -1 -1 24 -1 0 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1
        2 -1 -1 0 20 -1 -1 -1 25 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1
        23 -1 -1 -1 3 -1 -1 -1 0 -1 9 11 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1
        24 -1 23 1 17 -1 3 -1 10 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1
        25 -1 -1 -1 8 -1 -1 -1 7 18 -1 -1 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1
        13 24 -1 -1 0 -1 8 -1 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1
        7 20 -1 16 22 10 -1 -1 23 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1
        11 -1 -1 -1 19 -1 -1 -1 13 -1 3 17 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1
        25 -1 8 -1 23 18 -1 14 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0
        3 -1 -1 -1 16 -1 -1 2 25 5 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0
    """),
    "LDPC_648_R2_3": _parse_prototype("""
        25 26 14 -1 20 -1 2 -1 4 -1 -1 8 -1 16 -1 18 1 0 -1 -1 -1 -1 -1 -1
        10 9 15 11 -1 0 -1 1 -1 -1 18 -1 8 -1 10 -1 -1 0 0 -1 -1 -1 -1 -1
        16 2 20 26 21 -1 6 -1 1 26 -1 7 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1
        10 13 5 0 -1 3 -1 7 -1 -1 26 -1 -1 13 -1 16 -1 -1 -1 0 0 -1 -1 -1
        23 14 24 -1 12 -1 19 -1 17 -1 -1 -1 20 -1 21 -1 0 -1 -1 -1 0 0 -1 -1
        6 22 9 20 -1 25 -1 17 -1 8 -1 14 -1 18 -1 -1 -1 -1 -1 -1 -1 0 0 -1
        14 23 21 11 20 -1 24 -1 18 -1 19 -1 -1 -1 -1 22 -1 -1 -1 -1 -1 -1 0 0
        17 11 11 20 -1 21 -1 26 -1 3 -1 -1 18 -1 26 -1 1 -1 -1 -1 -1 -1 -1 0
    """),
    "LDPC_648_R3_4": _parse_prototype("""
        16 17 22 24 9 3 14 -1 4 2 7 -1 26 -1 2 -1 21 -1 1 0 -1 -1 -1 -1
        25 12 12 3 3 26 6 21 -1 15 22 -1 15 -1 4 -1 -1 16 -1 0 0 -1 -1 -1
        25 18 26 16 22 23 9 -1 0 -1 4 -1 4 -1 8 23 11 -1 -1 -1 0 0 -1 -1
        9 7 0 1 17 -1 -1 7 3 -1 3 23 -1 16 -1 -1 21 -1 0 -1 -1 0 0 -1
        24 5 26 7 1 -1 -1 15 24 15 -1 8 -1 13 -1 13 -1 11 -1 -1 -1 -1 0 0
        2 2 19 14 24 1 15 19 -1 21 -1 2 -1 24 -1 3 -1 2 1 -1 -1 -1 -1 0
    """),
    "LDPC_1296_R1_2": _parse_prototype("""
        40 -1 -1 -1 22 -1 49 23 43 -1 -1 -1 1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
        50 1 -1 -1 48 35 -1 -1 13 -1 30 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1
        39 50 -1 -1 4 -1 2 -1 -1 -1 -1 49 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1
        33 -1 -1 38 37 -1 -1 4 1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1
        45 -1 -1 -1 0 22 -1 -1 20 42 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1
        51 -1 -1 48 35 -1 -1 -1 44 -1 18 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1
        47 11 -1 -1 -1 17 -1 -1 51 -1 -1 -1 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1
        5 -1 25 -1 6 -1 45 -1 13 40 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1
        33 -1 -1 34 24 -1 -1 -1 23 -1 -1 46 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1
        1 -1 27 -1 1 -1 -1 -1 38 -1 44 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1
        -1 18 -1 -1 23 -1 -1 8 0 35 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0
        49 -1 17 -1 30 -1 -1 -1 34 -1 -1 19 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0
    """),
    "LDPC_1296_R2_3": _parse_prototype("""
        39 31 22 43 -1 40 4 -1 11 -1 -1 50 -1 -1 -1 6 1 0 -1 -1 -1 -1 -1 -1
        25 52 41 2 6 -1 14 -1 34 -1 -1 -1 24 -1 37 -1 -1 0 0 -1 -1 -1 -1 -1
        43 31 29 0 21 -1 28 -1 -1 2 -1 -1 7 -1 17 -1 -1 -1 0 0 -1 -1 -1 -1
        20 33 48 -1 4 13 -1 26 -1 -1 22 -1 -1 46 42 -1 -1 -1 -1 0 0 -1 -1 -1
        45 7 18 51 12 25 -1 -1 -1 50 -1 -1 5 -1 -1 -1 0 -1 -1 -1 0 0 -1 -1
        35 40 32 16 5 -1 -1 18 -1 -1 43 51 -1 32 -1 -1 -1 -1 -1 -1 -1 0 0 -1
        9 24 13 22 28 -1 -1 37 -1 -1 25 -1 -1 52 -1 13 -1 -1 -1 -1 -1 -1 0 0
        32 22 4 21 16 -1 -1 -1 27 28 -1 38 -1 -1 -1 8 1 -1 -1 -1 -1 -1 -1 0
    """),
    "LDPC_1296_R3_4": _parse_prototype("""
        39 40 51 41 3 29 8 36 -1 14 -1 6 -1 33 -1 11 -1 4 1 0 -1 -1 -1 -1
        48 21 47 9 48 35 51 -1 38 -1 28 -1 34 -1 50 -1 50 -1 -1 0 0 -1 -1 -1
        30 39 28 42 50 39 5 17 -1 6 -1 18 -1 20 -1 15 -1 40 -1 -1 0 0 -1 -1
        29 0 1 43 36 30 47 -1 49 -1 47 -1 3 -1 35 -1 34 -1 0 -1 -1 0 0 -1
        1 32 11 23 10 44 12 7 -1 48 -1 4 -1 9 -1 17 -1 16 -1 -1 -1 -1 0 0
        13 7 15 47 23 16 47 -1 43 -1 29 -1 52 -1 2 -1 53 -1 1 -1 -1 -1 -1 0
    """),
    "LDPC_1944_R1_2": _parse_prototype("""
        57 -1 -1 -1 50 -1 11 -1 50 -1 79 -1 1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
        3 -1 28 -1 0 -1 -1 -1 55 7 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1
        30 -1 -1 -1 24 37 -1 -1 56 14 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1
        62 53 -1 -1 53 -1 -1 3 35 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1
        40 -1 -1 20 66 -1 -1 22 28 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1
        0 -1 -1 -1 8 -1 42 -1 50 -1 -1 8 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1
        69 79 79 -1 -1 -1 56 -1 52 -1 -1 -1 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1
        65 -1 -1 -1 38 57 -1 -1 72 -1 27 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1
        64 -1 -1 -1 14 52 -1 -1 30 -1 -1 32 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1
        -1 45 -1 70 0 -1 -1 -1 77 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1
        2 56 -1 57 35 -1 -1 -1 -1 -1 12 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0
        24 -1 61 -1 60 -1 -1 27 51 -1 -1 16 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0
    """),
    "LDPC_1944_R2_3": _parse_prototype("""
        61 75 4 63 56 -1 -1 -1 -1 -1 -1 8 -1 2 17 25 1 0 -1 -1 -1 -1 -1 -1
        56 74 77 20 -1 -1 -1 64 24 4 67 -1 7 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1
        28 21 68 10 7 14 65 -1 -1 -1 23 -1 -1 -1 75 -1 -1 -1 0 0 -1 -1 -1 -1
        48 38 43 78 76 -1 -1 -1 -1 5 36 -1 15 72 -1 -1 -1 -1 -1 0 0 -1 -1 -1
        40 2 53 25 -1 52 62 -1 20 -1 -1 44 -1 -1 -1 -1 0 -1 -1 -1 0 0 -1 -1
        69 23 64 10 22 -1 21 -1 -1 -1 -1 -1 68 23 29 -1 -1 -1 -1 -1 -1 0 0 -1
        12 0 68 20 55 61 -1 40 -1 -1 -1 52 -1 -1 -1 44 -1 -1 -1 -1 -1 -1 0 0
        58 8 34 64 78 -1 -1 11 78 24 -1 -1 -1 -1 -1 58 1 -1 -1 -1 -1 -1 -1 0
    """),
    "LDPC_1944_R3_4": _parse_prototype("""
        48 29 28 39 9 61 -1 -1 -1 63 45 80 -1 -1 -1 37 32 22 1 0 -1 -1 -1 -1
        4 49 42 48 11 30 -1 -1 -1 49 17 41 37 15 -1 54 -1 -1 -1 0 0 -1 -1 -1
        35 76 78 51 37 35 21 -1 17 64 -1 -1 -1 59 7 -1 -1 32 -1 -1 0 0 -1 -1
        9 65 44 9 54 56 73 34 42 -1 -1 -1 35 -1 -1 -1 46 39 0 -1 -1 0 0 -1
        3 62 7 80 68 26 -1 80 55 -1 36 -1 26 -1 9 -1 72 -1 -1 -1 -1 -1 0 0
        26 75 33 21 69 59 3 38 -1 -1 -1 35 -1 62 36 26 -1 -1 1 -1 -1 -1 -1 0
    """),
}


@dataclass(frozen=True)
class LDPCConfig:
    """A sourced Annex R prototype and its lifted code parameters."""

    name: str
    N: int
    K: int
    rate: str
    Z: int
    prototype: tuple[tuple[int, ...], ...]
    source_reference_id: str
    cross_check_reference_id: str
    matrix_identifier: str
    generated_h_dimensions: tuple[int, int]

    @property
    def M(self) -> int:
        return self.N - self.K


def _make_config(name: str, N: int, K: int, rate: str, Z: int, table: str, part: str) -> LDPCConfig:
    prototype = _PROTOTYPES[name]
    return LDPCConfig(
        name=name,
        N=N,
        K=K,
        rate=rate,
        Z=Z,
        prototype=prototype,
        source_reference_id=IEEE_REFERENCE_ID,
        cross_check_reference_id=TAVILDAR_REFERENCE_ID,
        matrix_identifier=f"IEEE80211N_D11_0_AnnexR_TableR.{table}({part})",
        generated_h_dimensions=(N - K, N),
    )


LDPC_CONFIGS: dict[str, LDPCConfig] = {
    "LDPC_648_R1_2": _make_config("LDPC_648_R1_2", 648, 324, "1/2", 27, "1", "a"),
    "LDPC_648_R2_3": _make_config("LDPC_648_R2_3", 648, 432, "2/3", 27, "1", "b"),
    "LDPC_648_R3_4": _make_config("LDPC_648_R3_4", 648, 486, "3/4", 27, "1", "c"),
    "LDPC_1296_R1_2": _make_config("LDPC_1296_R1_2", 1296, 648, "1/2", 54, "2", "a"),
    "LDPC_1296_R2_3": _make_config("LDPC_1296_R2_3", 1296, 864, "2/3", 54, "2", "b"),
    "LDPC_1296_R3_4": _make_config("LDPC_1296_R3_4", 1296, 972, "3/4", 54, "2", "c"),
    "LDPC_1944_R1_2": _make_config("LDPC_1944_R1_2", 1944, 972, "1/2", 81, "3", "a"),
    "LDPC_1944_R2_3": _make_config("LDPC_1944_R2_3", 1944, 1296, "2/3", 81, "3", "b"),
    "LDPC_1944_R3_4": _make_config("LDPC_1944_R3_4", 1944, 1458, "3/4", 81, "3", "c"),
}

LDPC_CONFIG_NAMES: tuple[str, ...] = tuple(LDPC_CONFIGS)
LDPC_TYPE_NAMES: tuple[str, ...] = LDPC_CONFIG_NAMES
LDPC_FEC_NAME = "IEEE80211n_QC_LDPC"
_ConfigArg: TypeAlias = LDPCConfig | str


def get_ldpc_config(config: _ConfigArg) -> LDPCConfig:
    """Return a supported configuration by name or validate a config object."""
    if isinstance(config, LDPCConfig):
        known = LDPC_CONFIGS.get(config.name)
        if known != config:
            raise ValueError("config must be one of the verified LDPC_CONFIGS")
        return known
    if not isinstance(config, str):
        raise TypeError("config must be a configuration name or LDPCConfig")
    try:
        return LDPC_CONFIGS[config]
    except KeyError as exc:
        raise ValueError(f"unsupported IEEE 802.11n LDPC configuration: {config}") from exc


@lru_cache(maxsize=len(LDPC_CONFIGS))
def _build_h_cached(name: str) -> np.ndarray:
    config = LDPC_CONFIGS[name]
    prototype = config.prototype
    expected_rows = config.M // config.Z
    expected_cols = config.N // config.Z
    if len(prototype) != expected_rows or any(len(row) != expected_cols for row in prototype):
        raise ValueError(f"{name} prototype dimensions do not match N, K, and Z")
    if any(value < -1 or value >= config.Z for row in prototype for value in row):
        raise ValueError(f"{name} contains a shift outside 0..Z-1 or the -1 null marker")

    H = np.zeros((config.M, config.N), dtype=np.uint8)
    local_rows = np.arange(config.Z)
    for block_row, shifts in enumerate(prototype):
        row_start = block_row * config.Z
        for block_col, shift in enumerate(shifts):
            if shift < 0:
                continue
            col_start = block_col * config.Z
            H[row_start + local_rows, col_start + ((local_rows + shift) % config.Z)] = 1

    if H.shape != config.generated_h_dimensions or np.any(H > 1):
        raise ValueError(f"{name} generated an invalid parity-check matrix")
    H.setflags(write=False)
    return H


def build_parity_check_matrix(config: _ConfigArg) -> np.ndarray:
    """Generate and return a copy of the full binary parity-check matrix H."""
    return _build_h_cached(get_ldpc_config(config).name).copy()


def _syndrome(H: np.ndarray, codeword: np.ndarray) -> np.ndarray:
    return np.bitwise_and(H @ codeword, np.uint8(1))


@lru_cache(maxsize=len(LDPC_CONFIGS))
def _parity_inverse_rows(name: str) -> tuple[int, ...]:
    """Invert the verified systematic parity submatrix over GF(2)."""
    config = LDPC_CONFIGS[name]
    parity = _build_h_cached(name)[:, config.K:]
    m = config.M
    augmented: list[int] = []
    for row_index, row in enumerate(parity):
        left = 0
        for col_index in np.flatnonzero(row):
            left |= 1 << int(col_index)
        augmented.append(left | (1 << (m + row_index)))

    for column in range(m):
        pivot = next((row for row in range(column, m) if (augmented[row] >> column) & 1), None)
        if pivot is None:
            raise ValueError(f"{name} rightmost parity submatrix is singular over GF(2)")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_row = augmented[column]
        for row in range(m):
            if row != column and ((augmented[row] >> column) & 1):
                augmented[row] ^= pivot_row

    left_mask = (1 << m) - 1
    if any((row & left_mask) != (1 << index) for index, row in enumerate(augmented)):
        raise ValueError(f"{name} parity submatrix inversion did not produce identity")
    return tuple(row >> m for row in augmented)


def encode_ldpc(payload_bits: np.ndarray, config: _ConfigArg) -> np.ndarray:
    """Systematically encode exactly one K-bit payload into one N-bit word."""
    cfg = get_ldpc_config(config)
    payload = validate_binary_bits(payload_bits)
    if payload.size != cfg.K:
        raise ValueError(f"{cfg.name} encoder requires exactly {cfg.K} payload bits; received {payload.size}")

    H = _build_h_cached(cfg.name)
    information = H[:, :cfg.K]
    rhs = np.bitwise_and(information @ payload, np.uint8(1))
    rhs_integer = 0
    for row_index in np.flatnonzero(rhs):
        rhs_integer |= 1 << int(row_index)
    parity_bits = np.fromiter(
        ((row & rhs_integer).bit_count() & 1 for row in _parity_inverse_rows(cfg.name)),
        dtype=np.uint8,
        count=cfg.M,
    )
    codeword = np.concatenate((payload.copy(), parity_bits))
    if np.any(_syndrome(H, codeword)):
        raise RuntimeError(f"{cfg.name} encoder produced a nonzero syndrome")
    return codeword


@dataclass
class LDPCDecodeResult:
    """One complete codeword's soft-decision decoding outcome."""

    configuration: str
    N: int
    K: int
    rate: str
    decoder_algorithm: str = LDPC_DECODER_ALGORITHM
    decoded_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    codeword_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    success: bool = False
    syndrome_pass: bool = False
    syndrome_weight: int = 0
    iterations_used: int = 0
    max_iterations: int = 0
    failure_reason: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass
class LDPCFrameDecodeResult:
    """Independent decoding outcomes for a complete concatenation of words."""

    configuration: str
    N: int
    K: int
    rate: str
    codeword_count: int
    successful_codewords: int
    failed_codewords: int
    success: bool
    decoded_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    codeword_results: list[LDPCDecodeResult] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)


def _validate_llrs(llrs: np.ndarray, expected_length: int, *, label: str) -> np.ndarray:
    values = np.asarray(llrs)
    if values.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional LLR array")
    if values.size != expected_length:
        raise ValueError(f"{label} requires exactly {expected_length} LLRs; received {values.size}")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.complexfloating):
        raise TypeError(f"{label} must contain real numeric LLR values")
    converted = values.astype(np.float64, copy=True)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{label} LLRs must all be finite")
    return converted


def decode_ldpc(
    llrs: np.ndarray,
    config: _ConfigArg,
    *,
    max_iterations: int = 50,
    normalization: float = 0.8,
) -> LDPCDecodeResult:
    """Decode exactly one N-LLR word with flooding normalized min-sum.

    LLR convention: positive values favor bit 0, negative values favor bit 1.
    Magnitudes are retained throughout variable/check-node message updates.
    """
    cfg = get_ldpc_config(config)
    if not isinstance(max_iterations, (int, np.integer)) or isinstance(max_iterations, (bool, np.bool_)):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one")
    if not np.isfinite(normalization) or not 0 < normalization <= 1:
        raise ValueError("normalization must be finite and in (0, 1]")
    channel = _validate_llrs(llrs, cfg.N, label=cfg.name)
    H = _build_h_cached(cfg.name)
    check_edges = [np.flatnonzero(H[row]) for row in range(cfg.M)]
    edge_variables = np.concatenate(check_edges).astype(np.int64, copy=False)
    edge_starts = np.cumsum([0, *(edges.size for edges in check_edges)])
    check_messages = np.zeros(edge_variables.size, dtype=np.float64)

    hard = (channel < 0).astype(np.uint8)
    syndrome = _syndrome(H, hard)
    iterations_used = 0
    if not np.any(syndrome):
        return LDPCDecodeResult(
            configuration=cfg.name,
            N=cfg.N,
            K=cfg.K,
            rate=cfg.rate,
            decoded_bits=hard[:cfg.K].copy(),
            codeword_bits=hard.copy(),
            success=True,
            syndrome_pass=True,
            syndrome_weight=0,
            iterations_used=0,
            max_iterations=int(max_iterations),
            diagnostics={
                "decoder_algorithm": LDPC_DECODER_ALGORITHM,
                "llr_convention": "positive favors bit 0",
                "normalization": float(normalization),
                "syndrome_weight": 0,
            },
        )

    posterior = channel.copy()
    for iteration in range(1, int(max_iterations) + 1):
        variable_messages = (
            channel[edge_variables]
            + np.bincount(edge_variables, weights=check_messages, minlength=cfg.N)[edge_variables]
            - check_messages
        )
        next_messages = np.empty_like(check_messages)
        for check_index, edge_variables_for_check in enumerate(check_edges):
            start, stop = int(edge_starts[check_index]), int(edge_starts[check_index + 1])
            local = variable_messages[start:stop]
            magnitudes = np.abs(local)
            minimum_index = int(np.argmin(magnitudes))
            first_minimum = float(magnitudes[minimum_index])
            if magnitudes.size > 1:
                second_minimum = float(np.partition(magnitudes, 1)[1])
            else:
                second_minimum = first_minimum
            negative_parity = bool(np.count_nonzero(local < 0) & 1)
            signs_excluding = np.where(np.logical_xor(negative_parity, local < 0), -1.0, 1.0)
            minima_excluding = np.full(magnitudes.size, first_minimum, dtype=np.float64)
            minima_excluding[minimum_index] = second_minimum
            next_messages[start:stop] = float(normalization) * signs_excluding * minima_excluding

        posterior = channel + np.bincount(edge_variables, weights=next_messages, minlength=cfg.N)
        hard = (posterior < 0).astype(np.uint8)
        syndrome = _syndrome(H, hard)
        iterations_used = iteration
        check_messages = next_messages
        if not np.any(syndrome):
            break

    syndrome_weight = int(np.count_nonzero(syndrome))
    passed = syndrome_weight == 0
    return LDPCDecodeResult(
        configuration=cfg.name,
        N=cfg.N,
        K=cfg.K,
        rate=cfg.rate,
        decoded_bits=hard[:cfg.K].copy(),
        codeword_bits=hard.copy(),
        success=passed,
        syndrome_pass=passed,
        syndrome_weight=syndrome_weight,
        iterations_used=iterations_used,
        max_iterations=int(max_iterations),
        failure_reason=None if passed else f"nonzero syndrome after {iterations_used} iterations",
        diagnostics={
            "decoder_algorithm": LDPC_DECODER_ALGORITHM,
            "llr_convention": "positive favors bit 0",
            "normalization": float(normalization),
            "syndrome_weight": syndrome_weight,
        },
    )


def encode_ldpc_frame(payload_bits: np.ndarray, config: _ConfigArg) -> np.ndarray:
    """Encode one or more complete K-bit payloads; reject partial final words."""
    cfg = get_ldpc_config(config)
    payload = validate_binary_bits(payload_bits)
    if payload.size == 0 or payload.size % cfg.K:
        raise ValueError(f"{cfg.name} frame payload length must be a positive multiple of {cfg.K}; received {payload.size}")
    codewords = [encode_ldpc(payload[start:start + cfg.K], cfg) for start in range(0, payload.size, cfg.K)]
    return np.concatenate(codewords).astype(np.uint8, copy=False)


def decode_ldpc_frame(
    llrs: np.ndarray,
    config: _ConfigArg,
    *,
    max_iterations: int = 50,
    normalization: float = 0.8,
) -> LDPCFrameDecodeResult:
    """Decode only a positive, exact multiple of N LLRs, one word at a time."""
    cfg = get_ldpc_config(config)
    values = np.asarray(llrs)
    if values.ndim != 1:
        raise ValueError("LDPC frame LLRs must be a one-dimensional array")
    if values.size == 0 or values.size % cfg.N:
        raise ValueError(f"{cfg.name} frame length must be a positive exact multiple of {cfg.N}; received {values.size}")
    values = _validate_llrs(values, int(values.size), label=f"{cfg.name} frame")
    codeword_results = [
        decode_ldpc(values[start:start + cfg.N], cfg, max_iterations=max_iterations, normalization=normalization)
        for start in range(0, values.size, cfg.N)
    ]
    successful = sum(result.success for result in codeword_results)
    failed = len(codeword_results) - successful
    all_successful = failed == 0
    decoded = (
        np.concatenate([result.decoded_bits for result in codeword_results]).astype(np.uint8, copy=False)
        if all_successful else np.empty(0, dtype=np.uint8)
    )
    return LDPCFrameDecodeResult(
        configuration=cfg.name,
        N=cfg.N,
        K=cfg.K,
        rate=cfg.rate,
        codeword_count=len(codeword_results),
        successful_codewords=successful,
        failed_codewords=failed,
        success=all_successful,
        decoded_bits=decoded,
        codeword_results=codeword_results,
        diagnostics={
            "decoder_algorithm": LDPC_DECODER_ALGORITHM,
            "codeword_count": len(codeword_results),
            "successful_codewords": successful,
            "failed_codewords": failed,
            "syndrome_status": [result.syndrome_pass for result in codeword_results],
            "iterations_used": [result.iterations_used for result in codeword_results],
        },
    )
