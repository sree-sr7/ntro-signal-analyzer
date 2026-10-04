"""The vectorized Viterbi must be bit-identical to the original reference.

The reference below is the previous edge-by-edge implementation, kept here
verbatim in behavior so that any future change to ``decode_viterbi`` is
checked against it: same decoded bits, same path metric, same final state,
same tie-breaking (lower-numbered predecessor wins).
"""

import hashlib

import numpy as np
import pytest

from core.fec.convolutional import (
    CONVOLUTIONAL_CODE_SPECS,
    decode_viterbi,
    encode_convolutional,
)


def _parity(value: int) -> int:
    return value.bit_count() & 1


def _reference_decode(received: np.ndarray, spec, terminated: bool):
    state_count = 1 << (spec.constraint_length - 1)
    state_mask = state_count - 1
    previous_states = np.repeat(np.arange(state_count, dtype=np.int16), 2)
    input_bits = np.tile(np.array([0, 1], dtype=np.uint8), state_count)
    registers = (previous_states.astype(np.int32) << 1) | input_bits.astype(np.int32)
    next_states = (registers & state_mask).astype(np.int16)
    expected_outputs = np.empty((previous_states.size, 2), dtype=np.uint8)
    for edge, register in enumerate(registers):
        expected_outputs[edge, 0] = _parity(int(register) & spec.generator_polynomials[0])
        expected_outputs[edge, 1] = _parity(int(register) & spec.generator_polynomials[1])

    step_count = received.size // 2
    path_metrics = np.full(state_count, np.inf, dtype=np.float64)
    path_metrics[0] = 0.0
    survivor_previous = np.full((step_count, state_count), -1, dtype=np.int16)
    survivor_input = np.zeros((step_count, state_count), dtype=np.uint8)
    edges_by_destination = [np.flatnonzero(next_states == s) for s in range(state_count)]
    for step in range(step_count):
        observed = received[2 * step:2 * step + 2]
        branch = np.count_nonzero(expected_outputs != observed[None, :], axis=1)
        candidates = path_metrics[previous_states] + branch
        next_metrics = np.full(state_count, np.inf, dtype=np.float64)
        for destination, edges in enumerate(edges_by_destination):
            winner = int(edges[int(np.argmin(candidates[edges]))])
            next_metrics[destination] = candidates[winner]
            survivor_previous[step, destination] = previous_states[winner]
            survivor_input[step, destination] = input_bits[winner]
        path_metrics = next_metrics
    final_state = 0 if terminated else int(np.argmin(path_metrics))
    decoded = np.empty(step_count, dtype=np.uint8)
    state = final_state
    for step in range(step_count - 1, -1, -1):
        decoded[step] = survivor_input[step, state]
        state = int(survivor_previous[step, state])
    return decoded, int(path_metrics[final_state]), final_state, state


def _corrupt(bits: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    flips = rng.random(bits.size) < rate
    return (bits ^ flips.astype(np.uint8)).astype(np.uint8)


@pytest.mark.parametrize("code", list(CONVOLUTIONAL_CODE_SPECS))
@pytest.mark.parametrize("error_rate", [0.0, 0.02, 0.08, 0.25, 0.5])
@pytest.mark.parametrize("terminated", [True, False])
def test_vectorized_viterbi_matches_reference_exactly(code, error_rate, terminated):
    spec = CONVOLUTIONAL_CODE_SPECS[code]
    seed_material = f"{code}|{error_rate:.8f}|{int(terminated)}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    for payload_bits in (24, 101, 300):
        payload = rng.integers(0, 2, payload_bits, dtype=np.uint8)
        coded = encode_convolutional(payload, code, terminated=terminated)
        received = _corrupt(coded, error_rate, rng)
        ref_bits, ref_metric, ref_final, ref_start = _reference_decode(
            received, spec, terminated
        )
        result = decode_viterbi(received, code, terminated=terminated)
        tail = spec.termination_tail_bits if terminated else 0
        expected_decoded = ref_bits[:-tail] if terminated else ref_bits
        np.testing.assert_array_equal(result.decoded_bits, expected_decoded)
        assert result.path_metric == ref_metric
        assert result.diagnostics["final_state"] == ref_final
        assert result.diagnostics["initial_state_after_traceback"] == ref_start == 0


@pytest.mark.parametrize("code", list(CONVOLUTIONAL_CODE_SPECS))
def test_vectorized_viterbi_matches_reference_on_pure_random_and_degenerate_streams(code):
    """Wrong-hypothesis streams (random/constant) are exactly the tie-heavy case."""
    spec = CONVOLUTIONAL_CODE_SPECS[code]
    rng = np.random.default_rng(777)
    streams = [
        rng.integers(0, 2, 2 * 200, dtype=np.uint8),
        np.zeros(2 * 150, dtype=np.uint8),
        np.ones(2 * 150, dtype=np.uint8),
        np.tile(np.array([0, 1], dtype=np.uint8), 150),
        np.tile(np.array([1, 0], dtype=np.uint8), 150),
    ]
    for received in streams:
        for terminated in (True, False):
            ref_bits, ref_metric, ref_final, _ = _reference_decode(received, spec, terminated)
            result = decode_viterbi(received, code, terminated=terminated)
            tail = spec.termination_tail_bits if terminated else 0
            np.testing.assert_array_equal(
                result.decoded_bits, ref_bits[:-tail] if terminated else ref_bits
            )
            assert result.path_metric == ref_metric
            assert result.diagnostics["final_state"] == ref_final
