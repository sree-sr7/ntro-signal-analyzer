import numpy as np
import pytest

from core.common.models import PhaseAmbiguityResult
from core.sync.phase_ambiguity import phase_ambiguity_candidates


@pytest.mark.parametrize("order", [2, 4])
def test_phase_ambiguity_returns_all_rotations_without_ranking(order):
    samples = np.array([1 + 1j, -1 + 0.5j], dtype=np.complex128)
    result = phase_ambiguity_candidates(samples, order)

    assert isinstance(result, PhaseAmbiguityResult)
    assert len(result.candidate_phases_rad) == order
    assert len(result.candidate_samples) == order
    for phase, candidate in zip(result.candidate_phases_rad, result.candidate_samples):
        np.testing.assert_allclose(candidate, samples * np.exp(-1j * phase))
    assert result.candidate_phases_rad == pytest.approx(
        [2 * np.pi * k / order for k in range(order)]
    )


def test_phase_ambiguity_rejects_invalid_order_and_nonfinite_samples():
    with pytest.raises(ValueError):
        phase_ambiguity_candidates(np.ones(4), 1)
    with pytest.raises(ValueError, match="finite"):
        phase_ambiguity_candidates(np.array([np.inf + 0j]), 4)
