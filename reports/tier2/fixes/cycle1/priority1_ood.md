# Priority 1 — OOD gate

## Finding

The frozen T2-09 and T2-10 results both trusted the ONNX `16qam` label at effectively 1.0 confidence and entered 86 FEC candidates. Entropy is not a useful guard here: the prior diagnostic showed normalized entropy below 0.018 for both captures. The existing adaptive frame-energy detector marked every saved in-scope capture and both OOD captures as inactive, so that implementation cannot be safely reused as a transmit/noise gate.

The repository's existing bounded geometric classifier has an explicit 0.35 class-evidence acceptance floor. Reusing its non-probabilistic fit score as a signal-support gate is defensible without importing historical Mahalanobis or QAM/PSK threshold values. The gate reports the score type and does not call it an OOD probability.

## Change

- `core/ml/classifier.py`: exposed the strongest geometric modulation evidence and named the existing 0.35 floor.
- `core/pipeline/analyzer.py`: applied the fit-support gate after classification and before demodulation/FEC; rejection returns `unknown`, reason, and `signal_detection` stage.
- `tests/integration/test_ood_gate.py`: added noise-only, OFDM unknown, and supported-QPSK checks.

## Focused checks

`pytest tests/integration/test_ood_gate.py -q`: 3 passed.

Same frozen captures, no regeneration:

| Scenario | Baseline | After | Gate score | FEC |
|---|---|---|---:|---|
| T2-09 noise | 16QAM, confidence 1.000, OOD_FAIL | rejected / unknown_ood | 0.1853 | not entered |
| T2-10 OFDM | 16QAM, confidence 1.000, OOD_FAIL | rejected / unknown_ood | 0.1924 | not entered |

Full analyzer outputs, comparisons, runtimes, and logs are in `reports/tier2/fixes/cycle1/T2-09/` and `T2-10/`. The gate is conservative but is not a general unknown-waveform detector: unsupported signals that closely fit an in-scope geometry may still pass.
