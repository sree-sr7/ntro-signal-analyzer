# Priority 2 — BPSK/QPSK to 8PSK classification

## Finding

`diag_classifier.py` showed that T2-01, T2-02, T2-07, and T2-08 were classified on the raw capture before any CFO correction. Applying the estimator's CFO to the same saved samples changed the ONNX labels to QPSK, BPSK, QPSK, and QPSK respectively, with near-1 confidence. The saved samples, IQ scaling, feature order, samples/symbol, and model input dimensions were unchanged. This is a phase-sensitive classifier pipeline compatibility defect, not a frozen-model domain shift.

## Change

`core/pipeline/analyzer.py` now retries classification on CFO-corrected samples for a supported PSK first-pass label. It adopts that result only when it remains supported and confidence does not fall. Demodulation starts from the original working capture and retains its own synchronization path. The signal-support gate uses the uncorrected features, so the classifier cannot use correction to bypass OOD rejection.

## Frozen-capture classifier check

| Scenario | Baseline label | After classifier sync check | CFO estimate | Check runtime |
|---|---|---|---:|---:|
| T2-01 | 8PSK | QPSK | +145.202 Hz | 0.087 s |
| T2-02 | 8PSK | BPSK | −198.853 Hz | 0.035 s |
| T2-03 | 8PSK | 8PSK | +400.973 Hz | 0.024 s |
| T2-07 | 8PSK | QPSK | +249.999 Hz | 0.996 s |
| T2-08 | 8PSK | QPSK | +298.866 Hz | 0.046 s |

The four mislabeled scenarios now match their transmitted modulation. The capture and feature extractor were not altered. Classifier-only results are saved as `classifier_sync_check.json` under each corresponding `reports/tier2/fixes/cycle1/T2-*/` directory; these checks deliberately used `run_fec=false` and are not final end-to-end outcomes.

## Focused test

`pytest tests/integration/test_ood_gate.py tests/integration/test_cfo_aware_classifier.py -q`: 4 passed. No model/domain-shift finding is supported by this evidence.
