# Temperature scaling evaluation

This is a controlled synthetic evaluation of post-hoc probability scaling for the frozen ONNX classifier. The temperature is fitted only on `calibration_fit`; `calibration_assessment` is independent, and the separate `final_test` split was evaluated only after the temperature was frozen.

## Contract and data

- Model SHA-256: `0e0ea216116f8859775b1e55167520db63f95dd93a25a77243bff803ed7fb677`
- Feature schema: `phase3-modulation-features-v1`; 17 raw features; external scaling: no; scaler embedded in ONNX: True
- Class order: `bpsk, qpsk, 8psk, 16qam, 2fsk`
- Selected temperature: `6.027927189` from calibration-fit NLL minimization; calibrated output is `softmax(logits / T)`.
- Random symbols/states only: no payload, train samples, or repeated waveforms were used. Each split has its own PCG64 SeedSequence root.
- No calibration/assessment arrays or temperature-fitting workflow were present in the tracked project or supplied source archives. This new run re-estimates T and does not reuse the historical `3.824104` value.
- SNR was drawn continuously and uniformly from 8–20 dB, the span represented by the current Tier-2 in-scope scenarios. CFO was uniform in ±400 Hz for PSK/QAM and 0 Hz for the T2-06-style FSK setup.

| Split | Seed | Rows | Observed SNR | Artifact SHA-256 |
|---|---:|---:|---:|---|
| calibration_fit | 2026100501 | 2000 | 8.01–20.00 dB | `1c92b1acffc11ee6e6d387e5ab9b1cced163d9f802a887c760409c2591138105` |
| calibration_assessment | 2026100502 | 2000 | 8.01–19.99 dB | `2f905136fbe0a6b91853eb3e867d2abdefe14fc15598a3d208c6c1377c83a95b` |
| final_test | 2026100503 | 2000 | 8.00–20.00 dB | `a9e536e38fc794afbd5a4de814e841cfca8f3f0a974c55cb5c0e3f7687d133e6` |

## Before / after metrics

Accuracy and predicted class are unchanged by positive temperature scaling because argmax is unchanged; probability sharpness and mean maximum probability do change. ECE is top-label, 10 equal-width bins; Brier is the mean per-row sum of five-class squared errors.

| Split | N | Accuracy before / after | NLL before / after | ECE before / after | Brier before / after | Mean max probability before / after |
|---|---:|---:|---:|---:|---:|---:|
| Fit | 2000 | 0.6580 / 0.6580 | 3.54563 / 0.97904 | 0.31533 / 0.04012 | 0.64346 / 0.48822 | 0.9733 / 0.6869 |
| Independent assessment | 2000 | 0.6590 / 0.6590 | 3.55015 / 0.98301 | 0.31400 / 0.03102 | 0.64553 / 0.49033 | 0.9730 / 0.6848 |
| Final test | 2000 | 0.6440 / 0.6440 | 3.65346 / 1.00102 | 0.32824 / 0.05244 | 0.67035 / 0.50320 | 0.9722 / 0.6843 |

## Class and SNR breakdown

Per-class accuracy is unchanged by a positive temperature. One-vs-rest ECE is shown before/after; a class-level regression matters even when aggregate ECE improves.

| Class | Assessment accuracy | Assessment class ECE before / after | Final accuracy | Final class ECE before / after |
|---|---:|---:|---:|---:|
| bpsk | 0.2800 | 0.14444 / 0.08673 | 0.2375 | 0.15239 / 0.09002 |
| qpsk | 0.2550 | 0.14675 / 0.10464 | 0.2025 | 0.15561 / 0.10973 |
| 8psk | 1.0000 | 0.34183 / 0.23856 | 1.0000 | 0.35494 / 0.24530 |
| 16qam | 0.7600 | 0.05062 / 0.14562 | 0.7800 | 0.04689 / 0.14429 |
| 2fsk | 1.0000 | 0.00009 / 0.10806 | 1.0000 | 0.00010 / 0.10761 |

| Split | SNR band | N | Accuracy | NLL before / after | Top-label ECE before / after |
|---|---|---:|---:|---:|---:|
| Assessment | 8-12 dB | 653 | 0.6401 | 3.34620 / 0.95166 | 0.32465 / 0.08412 |
| Assessment | 12-16 dB | 668 | 0.6347 | 3.81372 / 1.02108 | 0.33232 / 0.05840 |
| Assessment | 16-20 dB | 679 | 0.7010 | 3.48699 / 0.97571 | 0.28679 / 0.08392 |
| Final test | 8-12 dB | 691 | 0.6064 | 3.65973 / 1.00207 | 0.35613 / 0.09052 |
| Final test | 12-16 dB | 659 | 0.6449 | 3.61573 / 0.99474 | 0.32190 / 0.08112 |
| Final test | 16-20 dB | 650 | 0.6831 | 3.68506 / 1.00627 | 0.30501 / 0.08892 |

Final-test confusion matrix (rows = truth; columns = prediction; class order `bpsk, qpsk, 8psk, 16qam, 2fsk`):

| Truth \ Prediction | BPSK | QPSK | 8PSK | 16QAM | 2FSK |
|---|---:|---:|---:|---:|---:|
| bpsk | 95 | 0 | 305 | 0 | 0 |
| qpsk | 0 | 81 | 319 | 0 | 0 |
| 8psk | 0 | 0 | 400 | 0 | 0 |
| 16qam | 0 | 0 | 88 | 312 | 0 |
| 2fsk | 0 | 0 | 0 | 0 | 400 |

## Deployment decision

**Temperature scaling was evaluated but not sufficiently validated for deployment.** Aggregate NLL, ECE, and Brier score improved on independent assessment and final synthetic splits, and accuracy/argmax did not change. However, calibration was inconsistent by class: one-vs-rest ECE worsened for 16-QAM and 2FSK, while BPSK/QPSK had low feature-level accuracy in this generated inference-input distribution. The current analyzer can perform an additional PSK/CFO classifier pass, which this direct feature-level calibration set does not model. These results do not justify changing production inference or describing the GUI score as calibrated.

## Files

- `dataset_manifest.json`: generation parameters, class/SNR counts, versions, seeds, and model hash.
- `frozen_temperature.json`: fit-only temperature and optimizer record; frozen before final-test evaluation.
- `fit_and_assessment_metrics.json`, `final_test_metrics.json`: full metrics, per-class scores, SNR-band metrics, reliability-bin values, and confusion counts.
- `reliability_diagram.png`, `confusion_matrix_final.png`: reproducible figures.
- `calibration_fit.npz`, `calibration_assessment.npz`, `final_test.npz`: derived raw features/logits, class labels, SNR, and CFO; no waveforms or payloads.

## Reproduction

Use a new output directory for each run. `fit` reads only fit and assessment arrays; run `final` only after `frozen_temperature.json` exists.

```powershell
.\.venv\Scripts\python.exe -m tools.calibration.evaluate_temperature prepare --output reports\calibration\<new-run>
.\.venv\Scripts\python.exe -m tools.calibration.evaluate_temperature fit --data-dir reports\calibration\<new-run>
.\.venv\Scripts\python.exe -m tools.calibration.evaluate_temperature final --data-dir reports\calibration\<new-run>
```

## Scope limits

These synthetic symbol-rate signals do not establish real-world/OTA probability calibration, end-to-end analyzer calibration, universal confidence, or OOD behavior. The project checkout contained no historical calibration/assessment arrays or script, so this run uses independent seeds and the current Tier-2 SNR span. OOD behavior is reported separately by the fresh Tier-2 run; temperature fitting does not include OOD samples.

The production temperature remains unapplied. Preserve the current UI wording: `Model score: <value> · uncalibrated`.
