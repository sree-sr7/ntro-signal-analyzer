# Analyzer CFO retry selection investigation

## Decision

**A bounded retry-selection change is justified for the tested synthetic operating envelope.** The Analyzer now treats a corrected-pass confidence loss of at most `1e-5` as a tie, while still requiring a successful supported classifier result and requiring the corrected features to pass the existing `0.35` geometric signal-support floor. The original signal-support gate remains in force.

The tolerance is a decision tie band, **not a claim about floating-point noise**. Replaying 60 identical retryable waveforms in each split produced exactly identical logits and confidence values. The observed confidence gaps are deterministic model outputs. The tie band was selected on the fresh development set, frozen before the independent confirmation set was evaluated, and confirmed without a new incorrect decision in that confirmation set.

The tie band is `0.0084%` of the classifier's existing `0.12` minimum accepted probability margin. It applies only after CFO correction and cannot bypass classifier status, supported-class, or signal-support requirements. The frozen decision record is in `frozen_selection_rule.json`.

## Production decision path and cause

The Analyzer first classifies the raw classifier input. If the initial supported class is BPSK, QPSK, or 8PSK and the original geometric signal gate passes, the Analyzer estimates CFO using the initially predicted PSK order, rotates a classifier-only copy of the samples, and classifies it again. Demodulation continues from its existing synchronization path.

Before this change, the second result won only if it had `SUCCESS` status, was not `UNKNOWN`, and had confidence greater than or equal to the first result. A strict confidence comparison therefore retained the first pass when a valid corrected result's confidence was lower by a very small amount.

Classifier confidence is the top probability minus the runner-up probability, multiplied by the sample-support factor. With 2,048 samples the support factor is effectively 1. Under CFO, raw constellation features can lead the model to a very strong but incorrect PSK class. CFO correction can then return the right class with a slightly smaller confidence value. The confidence ordering is reproducible but does not reliably rank those two different signal conditions by correctness.

On the confirmation set, the raw first-pass BPSK accuracy fell to 10.8% in the 100–200 Hz band and 0% in both bands above 200 Hz. The strict Analyzer path recovered much of that with retries, but remained substantially weaker than the corrected classifier result. The new tie band selected 77 correct BPSK retries and 3 correct QPSK retries that the strict comparison had rejected.

## Dataset and capture

The evaluator generated two fresh, non-overlapping deterministic splits. It did not read training, calibration, previous assessment, prior final-test, or Tier-2 payload arrays.

| Split | Seed | BPSK/QPSK/8PSK | QAM/FSK controls | Noise/OFDM controls | Total |
|---|---:|---:|---:|---:|---:|
| Development | 2026100701 | 480 each | 120 each | 80 each | 1,840 |
| Confirmation | 2026100702 | 480 each | 120 each | 80 each | 1,840 |

Each PSK class has 40 independent signals in every combination of four absolute CFO bands (0–100, 100–200, 200–300, 300–400 Hz) and three SNR points (9, 14, 19 dB). PSK/QAM inputs are 2,048 symbol-rate complex64 samples at 1 MHz, with independent random symbols, phase, gain, CFO sign/magnitude, and AWGN. QAM and FSK controls cover all three SNR points. Each split also has 80 noise-only and 80 OFDM OOD controls.

All 3,680 waveform hashes are unique across both splits. Raw IQ archives and per-waveform generation metadata are saved with the report. Analyzer records retain both actual classifier passes: features, logits, raw softmax, class, status, confidence, probability margin, geometric fit evidence, corrected-path validity, CFO estimate, actual selection reason, and each candidate rule's selection reason.

Frozen model SHA-256: `0e0ea216116f8859775b1e55167520db63f95dd93a25a77243bff803ed7fb677`.

## Candidate rules

The following candidates were evaluated on development data; only the selected `1e-5` rule was then evaluated on confirmation data:

| Rule | Development accuracy | Macro F1 | Correct retry results recovered | New incorrect decisions |
|---|---:|---:|---:|---:|
| Current strict comparison | 92.66% | 0.9390 | — | — |
| `1e-6` tie band | 93.80% | 0.9465 | 21 | 0 |
| `1e-5` tie band | 96.47% | 0.9639 | 70 | 0 |
| Geometric strongest-class corroboration | 85.11% | 0.8925 | 65 | 204 |
| Geometric corroboration plus `1e-6` | 82.45% | 0.8749 | 16 | 204 |

The geometric-class agreement alternatives were rejected: their added class-alignment condition caused many regressions. The accepted rule uses the same geometric support floor as the existing gate without requiring the geometric strongest class to equal the ML class. Retrospective verification found that every attempted corrected pass in development and confirmation cleared the `0.35` floor; the minimum scores among attempted retries were `0.4375` and `0.4352`, respectively.

## Confirmation results

The strict baseline below is the original production selector. The candidate is the frozen `1e-5` rule. Metrics cover 1,840 signals, including OOD controls; UNKNOWN/rejected predictions are counted as correct OOD rejection only for OOD rows.

| Metric | Current | Candidate |
|---|---:|---:|
| Accuracy | 91.96% | 96.30% |
| Macro precision | 0.9379 | 0.9590 |
| Macro recall | 0.9413 | 0.9691 |
| Macro F1 | 0.9334 | 0.9618 |

| True class | Current precision / recall / F1 | Candidate precision / recall / F1 |
|---|---:|---:|
| BPSK | 1.000 / 0.758 / 0.863 | 1.000 / 0.919 / 0.958 |
| QPSK | 1.000 / 0.948 / 0.973 | 1.000 / 0.954 / 0.977 |
| 8PSK | 0.811 / 1.000 / 0.896 | 0.938 / 1.000 / 0.968 |
| 16QAM | 1.000 / 0.942 / 0.970 | 1.000 / 0.942 / 0.970 |
| 2FSK | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| OOD rejection | 0.816 / 1.000 / 0.899 | 0.816 / 1.000 / 0.899 |

Of 1,414 confirmation retries, the current rule selected the second pass 1,086 times and rejected it 328 times. The frozen rule selected 1,408 second passes and rejected 6. It changed 80 final labels; all 80 changed from incorrect to correct (77 BPSK and 3 QPSK), and **none changed a correct result to an incorrect one**. It did not change any QAM, FSK, noise-only, or OFDM outcome.

### Accuracy by absolute CFO band

Each entry aggregates 120 signals across the three SNR values.

| Class | Absolute CFO | Raw first pass | Current Analyzer | Candidate |
|---|---|---:|---:|---:|
| BPSK | 0–100 Hz | 100.0% | 91.7% | 91.7% |
| BPSK | 100–200 Hz | 10.8% | 77.5% | 80.0% |
| BPSK | 200–300 Hz | 0.0% | 69.2% | 99.2% |
| BPSK | 300–400 Hz | 0.0% | 65.0% | 96.7% |
| QPSK | 0–100 Hz | 95.0% | 99.2% | 99.2% |
| QPSK | 100–200 Hz | 2.5% | 89.2% | 90.0% |
| QPSK | 200–300 Hz | 0.0% | 95.8% | 95.8% |
| QPSK | 300–400 Hz | 0.0% | 95.0% | 96.7% |
| 8PSK | All four bands | 100.0% | 100.0% | 100.0% |

The QPSK raw first-pass column shows the same CFO-sensitive first pass, while its current retry path was already strong. The candidate's small QPSK lift is a secondary regression check, not its tuning target.

### Accuracy by SNR

| Class | SNR | Current Analyzer | Candidate |
|---|---:|---:|---:|
| BPSK | 9 dB | 87.50% | 93.13% |
| BPSK | 14 dB | 73.75% | 90.63% |
| BPSK | 19 dB | 66.25% | 91.88% |
| QPSK | 9 dB | 88.13% | 88.13% |
| QPSK | 14 dB | 98.13% | 98.13% |
| QPSK | 19 dB | 98.13% | 100.00% |
| 8PSK | 9 / 14 / 19 dB | 100.00% | 100.00% |

### Confusion matrices

Rows are actual class; columns are predicted BPSK, QPSK, 8PSK, 16QAM, 2FSK, and UNKNOWN/rejected. The OOD row counts correct UNKNOWN rejection.

| Actual | Current | Candidate |
|---|---|---|
| BPSK | `[364, 0, 88, 0, 0, 28]` | `[441, 0, 11, 0, 0, 28]` |
| QPSK | `[0, 455, 20, 0, 0, 5]` | `[0, 458, 17, 0, 0, 5]` |
| 8PSK | `[0, 0, 480, 0, 0, 0]` | `[0, 0, 480, 0, 0, 0]` |
| 16QAM | `[0, 0, 4, 113, 0, 3]` | `[0, 0, 4, 113, 0, 3]` |
| 2FSK | `[0, 0, 0, 0, 120, 0]` | `[0, 0, 0, 0, 120, 0]` |
| OOD | `[0, 0, 0, 0, 0, 160]` | `[0, 0, 0, 0, 0, 160]` |

## Numerical and residual-error findings

Sixty retryable signals in each split were replayed through the production Analyzer. The maximum repeated-logit and repeated-confidence differences were both exactly zero. The 80 recovered confirmation cases had an average initial-minus-corrected confidence gap of `1.68e-6` and a maximum of `4.32e-6`. These are stable score differences, not nondeterministic arithmetic noise. The tie band resolves the observed near-tie policy failure without changing any tested result outside that band.

The correction estimator's typical errors were small on PSK retries (median absolute error about 1.3–1.5 Hz; 90th percentile about 8–11 Hz), but a small number of large outliers reached roughly 48–59 kHz. Every tie-band-selected corrected result in this targeted dataset also passed the geometric support floor. The estimator's tail remains a reason to keep CFO assumptions bounded and to monitor real captures.

After the candidate, BPSK still has 39 errors out of 480 (8.1%): 28 are UNKNOWN/rejected and 11 are labeled 8PSK. Five of those residual errors attempted a retry but the corrected result was still wrong. Most residual error is outside the high-CFO retry-discard cases addressed here. **RETRAINING MAY BE JUSTIFIED** for those residual BPSK errors, but the original training arrays and class-conditional provenance are unavailable. No training was started; any training investigation is a separate authorized phase.

## Regression and validation

- Corrected status and supported-class requirements remain mandatory.
- The original signal-support gate and UNKNOWN/OOD handling remain unchanged. The corrected feature path must also clear the same existing geometric score floor before selection.
- Confirmation OOD controls remained rejected: noise 80/80 and OFDM 80/80. No FSK or QAM result changed.
- Full test suite: **644 passed**. Focused retry, CFO, OOD, and classifier tests: **16 passed**.
- Post-change production verification replayed all 1,840 confirmation waveforms through the patched Analyzer. Label parity with the frozen candidate simulation was 1,840/1,840; selected-pass parity was 1,840/1,840; maximum logits, feature, and confidence differences were all zero.
- Tier-2: **10 scenarios**; 6 PASS, 1 PARTIAL, 1 REJECTED, 2 OOD_PASS, 0 FAIL, 0 ERROR. FEC/CRC passed for T2-01, T2-02, T2-03, T2-06, T2-07, and T2-08. Noise and OFDM remained OOD_PASS. T2-04 remains REJECTED and T2-05 remains PARTIAL with the same payload gate outcome as the prior post-GUI Tier-2 run. The optional T2-11 LDPC extension remains blocked because no independent verified encoder was available.
- Calibration remains **UNCALIBRATED**. No temperature, probability calibration, threshold fitting, scaler change, or ONNX change was performed.

Tier-2 artifacts: `reports/tier2/baseline/20261005_cfo_retry_final/` and `reports/tier2/fixes/cfo_retry_final_20261005/`.

## Files and repository state

Production change: `core/pipeline/analyzer.py`. Regression coverage: `tests/integration/test_cfo_retry_selection.py`. Reproduction tool, generated IQs, original-path and production-verification per-call records, frozen-rule record, metrics, and this report are in this directory. Tier-2 outputs are under the paths above.

The investigation and change were performed on `gui-redesign`, based on `4138cd4380ef1d374ace329b48d75672f86a6115`. No retraining, calibration, GUI change, main/freeze-tag change, merge, or push occurred.
