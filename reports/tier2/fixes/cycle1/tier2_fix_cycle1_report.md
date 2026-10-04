# Tier-2 Fix Cycle 1 Report

**Baseline:** `20260930_201445_269`  
**Scope:** targeted repairs from existing diagnostics, using the same frozen captures. This cycle is complete. No second fix cycle or final presentation work was started.

## Diagnostic findings

- The analyzer had no OOD gate. T2-09 noise and T2-10 OFDM received near-certain 16QAM softmax scores. Entropy was also low, and the existing adaptive energy detector rejected saved in-scope signals, so neither confidence/entropy nor that detector could safely act as a signal-presence check. Historical Mahalanobis thresholds were unavailable and were not restored.
- On T2-01, T2-02, T2-07, and T2-08, the frozen ONNX model changed to the correct class when the saved capture was corrected for measured CFO before classifier feature extraction. This isolates a pipeline compatibility defect; it does not support a model-domain-shift finding.
- T2-03 true encoded bits passed project deinterleaving, K7 Viterbi, and CRC. Analyzer demodulation diverged after carrier recovery rejected phase; CRC and expected payload length supplied enough evidence to select the correct phase.
- T2-05 WAV diagnostics confirmed channel 0=I, channel 1=Q, exact PCM/32768 scaling, and 1 MHz sampling. Carrier recovery lacked the QPSK 45-degree anchor. The corrected analyzer reports four unresolved rotations; a reference-only diagnostic found one exact 8,192-bit rotation, but the analyzer did not select it.
- T2-07 trial pruning now rejects only candidates that cannot match received/frame/codeword/interleaver lengths. The ground-truth candidate remained eligible and passed.

## Before and after by scenario

| Scenario | Baseline | After | Root cause / remaining limitation |
|---|---|---|---|
| T2-01 | FAIL; 8psk; 5856 bits; CRC False | complete; qpsk; 3904 bits; Conv_R12_K5 / Block_8x8; CRC True; payload exact True | The feature extractor saw uncompensated CFO, producing a confident 8PSK label for a QPSK signal. Phase ambiguity is selected only if the supplied frame length and CRC validate a candidate; otherwise the phase remains unresolved. |
| T2-02 | FAIL; 8psk; 12420 bits; CRC False | complete; bpsk; 4140 bits; Conv_R12_K7 / none; CRC True; payload exact True | The feature extractor saw uncompensated CFO, producing a confident 8PSK label for a BPSK signal. Phase ambiguity is selected only if the supplied frame length and CRC validate a candidate; otherwise the phase remains unresolved. |
| T2-03 | PARTIAL; 8psk; 4140 bits; CRC False | complete; 8psk; 4140 bits; Conv_R12_K7 / Convolutional_Depth_12; CRC True; payload exact True | Carrier phase fitting was rejected; a CRC-selected PSK phase rotation resolves the ambiguity. CRC phase search requires caller-supplied expected payload length and CRC presence. |
| T2-04 | REJECTED; unknown; 0 bits | not rerun; support gate projects PASS (0.578); classifier remains unknown | The frozen classifier had near-tied 16QAM and 8PSK scores and rejected the class as ambiguous; the new signal-support gate does not resolve class ambiguity. T2-04 remains an in-scope 16QAM classification failure. The OOD gate does not worsen its path, but this cycle did not change the model or classifier acceptance margin. |
| T2-05 | PARTIAL; qpsk; 8192 bits | PARTIAL; qpsk; 8192 bits; payload exact False | Carrier phase recovery omitted the QPSK constellation phase anchor, biasing its phase estimate. After correcting that estimate, QPSK still has an inherent fourfold rotational ambiguity that cannot be selected without CRC or a sync word. A previous cycle output also used the wrong stereo-WAV option because the ground-truth format string is stereo_wav_iq; the final same-capture run now uses treat_stereo_as_iq=True. The analyzer returns a complete 8,192-bit QPSK demodulation but cannot select the payload rotation without frame evidence; harness payload match remains false. The saved-reference rotation diagnostic found an exact candidate at quarter_turns=2 and did not apply it. |
| T2-07 | partial; 8psk; 392832 bits | complete; qpsk; 261888 bits | Raw CFO phase drift caused the frozen classifier to label QPSK as 8PSK, inflating the bit stream and making the FEC search explore structurally implausible hypotheses. Expected payload length must be supplied by the caller for these exact length prunes; no payload contents were passed into analysis. |
| T2-08 | FAIL; 8psk; 12288 bits; CRC False | complete; qpsk; 8192 bits; Conv_R12_K7 / Diag_16x16_S1; CRC True; payload exact True | The feature extractor saw uncompensated CFO, producing a confident 8PSK label for a QPSK signal. Phase ambiguity is selected only if the supplied frame length and CRC validate a candidate; otherwise the phase remains unresolved. |
| T2-09 | OOD_FAIL; 16qam; 16384 bits | rejected; unknown; 0 bits | The frozen classifier assigned near-certain 16QAM to noise-only input; the geometric signal-support gate rejects it. The structural fit gate is not a calibrated general OOD detector; unknown signals that fit a supported geometry may pass. |
| T2-10 | OOD_FAIL; 16qam; 6400 bits | rejected; unknown; 0 bits | The frozen classifier assigned near-certain 16QAM to OFDM; the geometric signal-support gate rejects it. The structural fit gate is not a calibrated general OOD detector; unknown signals that fit a supported geometry may pass. |

### T2-07 runtime

Runtime fell from **2,618.441 s (43m 38s)** to **427.646 s (7m 8s)**, a **6.12× speedup** (83.7% lower). The search had 89 theoretical hypotheses; after length/modulation correction, 9 were structurally compatible. The ground-truth `Concat_RS223_Conv7 / Block_16x16` trial was evaluated, accepted, passed CRC, and recovered the exact 114,160-bit payload. Exactly one full T2-07 FEC rerun was performed.

### OOD behavior

T2-09 and T2-10 previously proceeded as 16QAM with confidence effectively 1.0 and 86 trial entries; the baseline review recorded 50 and 45 accepted candidates. The new gate rejects them as `unknown_ood` at `signal_detection`, reports its reason and bounded geometric-fit score (0.1853 and 0.1924 versus the existing 0.35 acceptance floor), exposes no fake OOD probability, and stops before demodulation/FEC.

T2-04 was not rerun: its saved feature vector scores 0.578 on the gate, so it would pass signal support; the original classifier still returns unknown because its 16QAM and 8PSK evidence is ambiguous. T2-06 remains the preserved supplemental PASS: 2FSK, Conv_R12_K3, CRC pass, exact payload.

### T2-05 residual

The corrected WAV-as-IQ run takes 0.045 s, classifies QPSK, and demodulates 8,192 bits. Its harness payload match remains false on the primary unresolved rotation. Four-quarter-turn diagnostic: k=0 BER=1.000, k=1 BER=0.500, k=2 BER=0.000, k=3 BER=0.500. The exact candidate is retained only as diagnostic evidence; no hidden reference bits were used to set the analyzer output.

## Files changed

Production code:
- `core/ml/classifier.py` — named existing 0.35 geometric-evidence floor and exposed strongest supported fit evidence.
- `core/pipeline/analyzer.py` — signal-support OOD gate; CFO-corrected classifier retry; phase-only recovery and CRC-validated phase search.
- `core/sync/carrier_recovery.py` — backward-compatible QPSK constellation phase anchor.
- `core/fec/trial_engine.py` — structural length/divisibility preflight and pruning counts.

Targeted test files:
- `tests/integration/test_ood_gate.py`
- `tests/integration/test_cfo_aware_classifier.py`
- `tests/integration/test_t203_phase_recovery.py`
- `tests/integration/test_trial_pruning.py`
- `tests/unit/test_carrier_recovery.py`

Per-scenario analyzer outputs, comparisons, runtimes, and logs are saved under `reports/tier2/fixes/cycle1/T2-*/`. Consolidated files: `tier2_fix_cycle1_summary.json`, `.csv`, and this report.

## Targeted verification

- `pytest tests/integration/test_ood_gate.py tests/integration/test_cfo_aware_classifier.py -q` — **4 passed**.
- `pytest tests/unit/test_carrier_recovery.py -q` — **4 passed**.
- Focused trial-pruning, selected structural FEC, and T2-03 phase regressions — **11 passed in 6.45 s**.
- `pytest tests/integration/test_t203_phase_recovery.py -q` — **1 passed** (also included in the focused group).
- The full 500+ pytest suite was **not run**.

## Preservation and remaining issues

- Production ONNX was not changed. Current SHA-256: `0e0ea216116f8859775b1e55167520db63f95dd93a25a77243bff803ed7fb677` (matches the recorded value).
- Frozen Tier-2 baseline and companion artifacts were preserved: **117/117** manifest-listed files match their pre-cycle hashes.
- No confirmed model-domain shift was found; CFO-corrected same-capture classification supports a pipeline compatibility cause.
- T2-04 remains unclassified (unknown) due class ambiguity. T2-05 remains payload-ambiguous without CRC/sync. The OOD gate is a transparent geometric support rule, not a calibrated detector for every unknown waveform.
- `docs/research_references.md` and `docs/references.bib` were not changed because no external research was needed.
