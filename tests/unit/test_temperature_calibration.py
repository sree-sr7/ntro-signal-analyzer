from __future__ import annotations

import numpy as np
import pytest

from tools.calibration.evaluate_temperature import (
    CLASS_NAMES,
    SAMPLES_PER_CLASS,
    SPLITS,
    _make_example,
    fit_temperature,
    score_dataset,
    stable_softmax,
)


def test_temperature_must_be_finite_and_positive() -> None:
    logits = np.array([[2.0, 0.0, -1.0, 0.5, -0.5]])
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="greater than zero"):
            stable_softmax(logits, invalid)


def test_temperature_scaling_changes_probabilities_without_changing_argmax() -> None:
    logits = np.array([[5.0, 1.0, 0.0, -1.0, -2.0], [-1.0, 3.0, 2.0, 0.0, -2.0]])
    raw = stable_softmax(logits)
    calibrated = stable_softmax(logits, 2.5)
    assert not np.allclose(raw, calibrated)
    assert np.array_equal(np.argmax(raw, axis=1), np.argmax(calibrated, axis=1))


def test_fit_uses_positive_scalar_temperature_and_reduces_fit_nll() -> None:
    logits = np.array([
        [4.0, 1.0, 0.0, -1.0, -2.0],
        [2.0, 0.0, 1.0, -1.0, -2.0],
        [0.0, 4.0, 1.0, -1.0, -2.0],
        [1.0, 2.0, 0.0, -1.0, -2.0],
        [0.0, 1.0, 4.0, -1.0, -2.0],
    ])
    labels = np.array([0, 1, 1, 0, 2])
    temperature, optimizer = fit_temperature(logits, labels)
    assert temperature > 0
    assert optimizer["success"] is True
    assert optimizer["fit_nll"] <= float(np.mean(-np.log(stable_softmax(logits)[np.arange(labels.size), labels])))
    assert np.array_equal(np.argmax(logits, axis=1), np.argmax(logits / temperature, axis=1))


def test_split_seeds_and_class_counts_are_distinct_and_balanced() -> None:
    assert len(set(SPLITS.values())) == 3
    assert SAMPLES_PER_CLASS > 0
    assert len(CLASS_NAMES) == 5


def test_synthetic_example_generation_is_seed_deterministic() -> None:
    for class_index in range(len(CLASS_NAMES)):
        first, first_cfo = _make_example(
            class_index,
            np.random.Generator(np.random.PCG64(12345)),
            snr_db=14.0,
        )
        second, second_cfo = _make_example(
            class_index,
            np.random.Generator(np.random.PCG64(12345)),
            snr_db=14.0,
        )
        np.testing.assert_array_equal(first, second)
        assert first_cfo == second_cfo


def test_metrics_keep_class_order_confusion_and_snr_breakdown() -> None:
    labels = np.arange(5, dtype=np.int64)
    logits = np.eye(5, dtype=np.float64) * 3.0
    data = {
        "logits": logits,
        "labels": labels,
        "snr_db": np.array([8.5, 11.9, 12.0, 16.0, 20.0]),
    }
    metrics = score_dataset(data, temperature=2.0)
    assert metrics["argmax_unchanged"] is True
    assert metrics["accuracy_before"] == metrics["accuracy_after"] == 1.0
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == len(labels)
    assert tuple(metrics["per_class"]) == CLASS_NAMES
    assert all("sample_count" in row for row in metrics["snr_bands"].values())

