"""Hard-decision rate-1/2 convolutional codes and Viterbi decoding.

Supported codes are the explicit ``Conv_R12_K3/K5/K7/K9`` specifications
below. ``K`` is the full shift-register constraint length including the
current input stage; the trellis state therefore contains ``K - 1`` memory
bits. A K-bit register is formed as ``(state << 1) | input_bit`` so the new
input occupies bit 0. Output parity is emitted in the listed generator order,
then the low ``K - 1`` bits become the next state. All codes start at state 0.

Terminated encoding appends ``K - 1`` zero tail bits. Terminated Viterbi
decoding constrains the endpoint to state 0 and strips those decoded tail bits.
Only rate-1/2 hard decisions are supported; no soft-decision or punctured-code
behavior is implied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from core.bitstream.extractor import validate_binary_bits


SUPPORTED_RATE = "R1/2"
SUPPORTED_CONSTRAINT_LENGTH = 7
# Retained for Step 79 imports that used this K=7 constant directly.
GENERATOR_POLYNOMIALS = (0o171, 0o133)


@dataclass(frozen=True)
class ConvolutionalCodeSpec:
    """One fully specified rate-1/2 convolutional code."""

    name: str
    rate: str
    constraint_length: int
    generator_polynomials: tuple[int, int]
    termination_tail_bits: int
    termination_behavior: str


CONVOLUTIONAL_CODE_SPECS: dict[str, ConvolutionalCodeSpec] = {
    "Conv_R12_K3": ConvolutionalCodeSpec(
        name="Conv_R12_K3",
        rate="R1/2",
        constraint_length=3,
        generator_polynomials=(0o7, 0o5),
        termination_tail_bits=2,
        termination_behavior="Append two zero input bits; decoder ends in state zero and removes them.",
    ),
    "Conv_R12_K5": ConvolutionalCodeSpec(
        name="Conv_R12_K5",
        rate="R1/2",
        constraint_length=5,
        generator_polynomials=(0o23, 0o35),
        termination_tail_bits=4,
        termination_behavior="Append four zero input bits; decoder ends in state zero and removes them.",
    ),
    "Conv_R12_K7": ConvolutionalCodeSpec(
        name="Conv_R12_K7",
        rate="R1/2",
        constraint_length=7,
        generator_polynomials=(0o171, 0o133),
        termination_tail_bits=6,
        termination_behavior="Append six zero input bits; decoder ends in state zero and removes them.",
    ),
    "Conv_R12_K9": ConvolutionalCodeSpec(
        name="Conv_R12_K9",
        rate="R1/2",
        constraint_length=9,
        generator_polynomials=(0o561, 0o753),
        termination_tail_bits=8,
        termination_behavior="Append eight zero input bits; decoder ends in state zero and removes them.",
    ),
}


@dataclass
class ConvolutionalDecodeResult:
    """Result of hard-decision Viterbi decoding for a supported code."""

    decoded_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    valid: bool = False
    terminated: bool = False
    path_metric: int = 0
    decoded_length: int = 0
    diagnostics: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _resolve_code(code: str | ConvolutionalCodeSpec) -> ConvolutionalCodeSpec:
    if isinstance(code, ConvolutionalCodeSpec):
        known = CONVOLUTIONAL_CODE_SPECS.get(code.name)
        if known != code:
            raise ValueError("code specification must match one of the supported Conv_R12_K* entries")
        return known
    if not isinstance(code, str) or code not in CONVOLUTIONAL_CODE_SPECS:
        supported = ", ".join(CONVOLUTIONAL_CODE_SPECS)
        raise ValueError(f"Unsupported convolutional code {code!r}; supported codes: {supported}")
    return CONVOLUTIONAL_CODE_SPECS[code]


def _legacy_code(rate: str, constraint_length: int) -> ConvolutionalCodeSpec:
    if rate != SUPPORTED_RATE:
        raise ValueError("Only rate R1/2 is supported")
    if not isinstance(constraint_length, (int, np.integer)) or isinstance(constraint_length, bool):
        raise TypeError("constraint_length must be an integer")
    for spec in CONVOLUTIONAL_CODE_SPECS.values():
        if spec.constraint_length == constraint_length:
            return spec
    raise ValueError("Supported constraint lengths are 3, 5, 7, and 9")


def _parity(value: int) -> int:
    return value.bit_count() & 1


def encode_convolutional(
    bits: np.ndarray,
    code: str | ConvolutionalCodeSpec = "Conv_R12_K7",
    *,
    terminated: bool = True,
) -> np.ndarray:
    """Encode binary bits with one explicit supported rate-1/2 code."""
    source = validate_binary_bits(bits)
    spec = _resolve_code(code)
    if not isinstance(terminated, (bool, np.bool_)):
        raise TypeError("terminated must be a bool")
    tail_count = spec.termination_tail_bits if terminated else 0
    input_bits = np.concatenate((source, np.zeros(tail_count, dtype=np.uint8)))
    outputs = np.empty(input_bits.size * 2, dtype=np.uint8)
    state = 0
    state_mask = (1 << (spec.constraint_length - 1)) - 1
    for index, bit in enumerate(input_bits):
        register = (state << 1) | int(bit)
        outputs[2 * index] = _parity(register & spec.generator_polynomials[0])
        outputs[2 * index + 1] = _parity(register & spec.generator_polynomials[1])
        state = register & state_mask
    return outputs


def _trellis(
    spec: ConvolutionalCodeSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build transition tables for one code; tables carry no decoder state."""
    state_count = 1 << (spec.constraint_length - 1)
    state_mask = state_count - 1
    previous_states = np.repeat(np.arange(state_count, dtype=np.int16), 2)
    input_bits = np.tile(np.array([0, 1], dtype=np.uint8), state_count)
    registers = (previous_states.astype(np.int32) << 1) | input_bits.astype(np.int32)
    next_states = (registers & state_mask).astype(np.int16)
    outputs = np.empty((previous_states.size, 2), dtype=np.uint8)
    for edge, register in enumerate(registers):
        outputs[edge, 0] = _parity(int(register) & spec.generator_polynomials[0])
        outputs[edge, 1] = _parity(int(register) & spec.generator_polynomials[1])
    return previous_states, input_bits, next_states, outputs


@lru_cache(maxsize=None)
def _acs_tables(
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build immutable add-compare-select tables for one supported code.

    Every destination state ``d`` has exactly two incoming edges, from
    ``d >> 1`` and from ``(d >> 1) | 2**(K-2)``, both carrying input bit
    ``d & 1``. The lower predecessor therefore has the lower edge index in
    the original edge-ordered search, so ties still resolve to it.
    """
    spec = CONVOLUTIONAL_CODE_SPECS[name]
    state_count = 1 << (spec.constraint_length - 1)
    destination = np.arange(state_count, dtype=np.intp)
    input_bit = destination & 1
    first_previous = destination >> 1
    second_previous = first_previous | (state_count >> 1)
    observed = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)

    def branch_metrics(previous: np.ndarray) -> np.ndarray:
        register = (previous << 1) | input_bit
        expected = np.empty((state_count, 2), dtype=np.uint8)
        for column, polynomial in enumerate(spec.generator_polynomials):
            expected[:, column] = [_parity(int(value) & polynomial) for value in register]
        # Row q is the Hamming distance for observed pair (q >> 1, q & 1).
        return (expected[None, :, :] != observed[:, None, :]).sum(axis=2).astype(np.float64)

    tables = (
        first_previous,
        second_previous,
        branch_metrics(first_previous),
        branch_metrics(second_previous),
    )
    for table in tables:
        table.setflags(write=False)
    return tables


def decode_viterbi(
    received_bits: np.ndarray,
    code: str | ConvolutionalCodeSpec = "Conv_R12_K7",
    *,
    terminated: bool = True,
) -> ConvolutionalDecodeResult:
    """Hard-decision Viterbi decode without truncating an incomplete symbol.

    The rate-1/2 input must contain complete two-bit output symbols. In
    terminated mode the final trellis state is constrained to zero and
    ``K - 1`` decoded tail bits are removed. In unterminated mode the minimum
    metric endpoint is selected and all decoded bits are returned.
    """
    received = validate_binary_bits(received_bits)
    spec = _resolve_code(code)
    if not isinstance(terminated, (bool, np.bool_)):
        raise TypeError("terminated must be a bool")
    if received.size % 2:
        raise ValueError("received_bits length must be divisible by the rate-1/2 output width of two")
    step_count = received.size // 2
    if step_count == 0:
        raise ValueError("received_bits must contain at least one complete output symbol")
    tail_count = spec.termination_tail_bits if terminated else 0
    if terminated and step_count < tail_count:
        raise ValueError(f"terminated decoding for {spec.name} requires at least {tail_count} trellis steps")

    state_count = 1 << (spec.constraint_length - 1)
    first_previous, second_previous, first_metrics, second_metrics = _acs_tables(spec.name)
    path_metrics = np.full(state_count, np.inf, dtype=np.float64)
    path_metrics[0] = 0.0
    # One bool per (step, state): True when the higher-numbered predecessor won.
    took_second = np.empty((step_count, state_count), dtype=np.bool_)
    observed_pairs = ((received[0::2].astype(np.intp) << 1) | received[1::2]).tolist()

    for step, pair in enumerate(observed_pairs):
        first = path_metrics.take(first_previous) + first_metrics[pair]
        second = path_metrics.take(second_previous) + second_metrics[pair]
        won = second < first  # strict: ties keep the lower predecessor, as before
        took_second[step] = won
        path_metrics = np.where(won, second, first)

    final_state = 0 if terminated else int(np.argmin(path_metrics))
    path_metric = int(path_metrics[final_state])
    decoded_with_tail = np.empty(step_count, dtype=np.uint8)
    state = final_state
    upper_bit_shift = spec.constraint_length - 2
    for step in range(step_count - 1, -1, -1):
        decoded_with_tail[step] = state & 1
        state = (state >> 1) | (int(took_second[step, state]) << upper_bit_shift)
    start_state = state
    if start_state != 0:
        raise RuntimeError("Viterbi survivor path did not return to the initial zero state")

    tail_valid = bool(not terminated or np.all(decoded_with_tail[-tail_count:] == 0))
    decoded = decoded_with_tail[:-tail_count].copy() if terminated else decoded_with_tail
    warnings: list[str] = []
    if not tail_valid:
        warnings.append("Decoded path did not contain the requested zero termination tail")
    return ConvolutionalDecodeResult(
        decoded_bits=decoded,
        valid=tail_valid,
        terminated=bool(terminated and final_state == 0 and tail_valid),
        path_metric=path_metric,
        decoded_length=int(decoded.size),
        diagnostics={
            "fec_type": spec.name,
            "rate": spec.rate,
            "constraint_length": spec.constraint_length,
            "generator_polynomials_octal": tuple(format(poly, "o") for poly in spec.generator_polynomials),
            "termination_behavior": spec.termination_behavior,
            "encoded_bit_count": int(received.size),
            "trellis_step_count": int(step_count),
            "decoded_bit_count_including_tail": int(decoded_with_tail.size),
            "tail_bits_removed": int(tail_count),
            "tail_bits_valid": tail_valid,
            "final_state": int(final_state),
            "initial_state_after_traceback": int(start_state),
        },
        warnings=warnings,
    )


def convolutional_encode(
    bits: np.ndarray,
    rate: str = "R1/2",
    constraint_length: int = 7,
    *,
    terminate: bool = True,
) -> np.ndarray:
    """Backward-compatible encoder API; K=7 defaults are unchanged."""
    spec = _legacy_code(rate, constraint_length)
    return encode_convolutional(bits, spec, terminated=terminate)


def viterbi_decode(
    hard_bits: np.ndarray,
    rate: str = "R1/2",
    constraint_length: int = 7,
    *,
    terminated: bool = True,
) -> ConvolutionalDecodeResult:
    """Backward-compatible decoder API; K=7 defaults are unchanged."""
    spec = _legacy_code(rate, constraint_length)
    return decode_viterbi(hard_bits, spec, terminated=terminated)
