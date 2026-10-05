# Presentation claim sheet — SIH 26147

Use the statements below with their stated scope. The evidence is for the
specific Windows run and frozen captures listed here; it does not establish
general accuracy or real-radio performance.

## Safe slide claims

1. **“On frozen T2-07, the analyzer runtime fell from 427.646 seconds in Cycle
   1 to 32.193 seconds in Cycle 2, a 13.28× speedup (92.47% less runtime),
   with CRC pass and exact payload recovery.”**

   Scope: one unchanged frozen capture, 1 MHz sample-rate input, CRC enabled,
   and the known 114,160-bit expected payload length supplied. The analyzer
   received no payload contents. Describe this as the measured Cycle 2 runtime
   optimization; do not attribute the full gain exclusively to Viterbi or
   generalize it to other signals.

2. **“Five frozen cases—T2-01, T2-02, T2-03, T2-07, and T2-08—passed CRC and
   exact-payload checks without a declared payload length.”**

   Scope: CRC presence and the 1 MHz sample-rate metadata were provided;
   payload contents were not. This is not a no-metadata or unconstrained blind
   recovery claim. The T2-07 run took 136.618 seconds, 4.24× its Cycle 2 run
   with the declared length.

3. **“The current Windows test suite passed 632 of 632 tests, including
   headless GUI tests.”**

   Scope: `QT_QPA_PLATFORM=offscreen`. The developer separately completed
   native Windows manual checks for launch, deterministic demo, T2-01/T2-02,
   T2-09 rejection, T2-05 partial status, and T2-07 responsiveness; these
   checks are recorded in the release note.

4. **“The checked-in ONNX classifier uses 17 ordered input features and five
   output classes, with its StandardScaler embedded in the ONNX graph.”**

   Scope: ONNX Runtime CPU provider. The model digest was
   `0e0ea216116f8859775b1e55167520db63a25a77243bff803ed7fb677` during
   this pass. A new synthetic evaluation fitted `T=6.027927`, improving
   aggregate NLL/ECE on independent assessment and final sets, but class-wise
   calibration was inconsistent and the candidate was not deployed. Current
   inference does not apply a temperature; do not describe outputs as
   temperature-calibrated probabilities.

5. **“The signal-support gate uses a 0.35 bounded geometric fit-evidence
   threshold; the recorded noise and OFDM examples scored 0.1853 and 0.1924
   and were rejected.”**

   Scope: these are two observed examples. The score is not a calibrated
   probability and does not guarantee rejection of every unsupported signal.

6. **“One controlled 8-samples/symbol synthetic run exercised timing recovery;
   the signal was misclassified and did not pass FEC/CRC.”**

   This wording is accurate. Do not shorten it to imply a successful
   oversampled end-to-end decode.

## Claims to avoid

- “The system achieves 100% accuracy,” or any broad accuracy figure inferred
  from the small named capture set.
- “A 13.28× speedup on all signals,” “92.47% faster in general,” or “Viterbi
  alone caused the full speedup.”
- “Five captures recovered with no metadata,” “fully blind recovery,” or
  “6/8 blind recovery.” The no-length check still supplied sample rate and CRC
  presence; other analyzer inputs remain scenario-dependent.
- “CRC guarantees the correct phase” or “zero false accepts.” The randomized
  check tested 28,480 hypotheses and observed one accepted garbage candidate.
  CRC-16 has an approximate `2^-16` collision probability per independent
  random candidate before structural and FEC validation.
- “Universal OOD detection,” “OOD probability,” or a confidence guarantee.
- “Temperature-calibrated confidence” or “calibration temperature is applied.”
- “CFO is always corrected before classification.” Classification can examine
  bounded CFO hypotheses; there is no universal pre-classification correction
  guarantee.
- “1,456 FEC trials,” “768-seed blind search,” or an arbitrary pseudo-random
  seed sweep. The current trial engine prunes structural incompatibilities and
  supports configured project-scoped pseudo-random profiles.
- “Rate-1/3 convolutional decoding,” “soft-decision Viterbi,” or “punctured
  convolutional decoding.” The implemented Viterbi path is hard-decision,
  rate-1/2.
- “GNU Radio validated,” “real OTA validated,” or “validated against live
  radio captures.”
- “T2-04 16-QAM recovery passed,” “T2-05 complete,” or “oversampled QPSK
  payload recovery passed.”
- “End-to-end DQPSK, 64-QAM, or 4FSK verified.” Those cases were not established
  by this validation pass.

## Evidence locations

- Runtime-only result: `reports/tier2/fixes/cycle2_runtime_only/`
- No-declared-length phase search and negative check:
  `reports/tier2/fixes/cycle2_crc_phase_search/`
- T2-04/T2-05 current-code checks and oversampled run:
  `reports/validation/`
- Full freeze-gate assessment:
  `reports/pre_freeze/final_pre_freeze_report.md`
