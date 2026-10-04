# NTRO Signal Analyzer

**SIH 2026 — Problem Statement 26147 (NTRO)**

Offline analysis of WAV, SigMF, and configured raw-IQ captures. The project
contains a PyQt desktop application, DSP and demodulation code, a production
ONNX modulation classifier, FEC/interleaver trial code, and preserved Tier-2
evidence.

## Core freeze status

The SIH 26147 core freeze release is recorded as
`v1.0.0-sih26147-core-freeze`. The previously recorded Windows suite passed
**632/632 tests**; the production model digest, 117 Cycle 1 checksums, and 169
protected files remain unchanged. The developer completed native Windows
manual checks for application launch, deterministic demo, T2-01 and T2-02
analysis, T2-09 rejection, T2-05 stereo-WAV partial status, and T2-07
responsiveness during long-running analysis.

One of 28,480 randomized CRC-phase hypotheses produced an accepted CRC-16
candidate. This is a known limitation, not a zero-false-accept claim. See the
[release note](docs/release/v1.0.0-sih26147-core-freeze.md),
[final pre-freeze report](reports/pre_freeze/final_pre_freeze_report.md), and
[presentation claim sheet](docs/presentation_claims.md).

The core freeze preserves the current GUI. UX redesign is separate post-freeze
work on the `gui-redesign` branch.

## Implemented scope

- File input for WAV, SigMF, and raw IQ. Raw IQ needs explicit sample rate,
  numeric format, byte order, and I/Q layout.
- Bounded classification for BPSK, QPSK, 8PSK, rectangular 16-QAM, and
  configured binary 2FSK, using the included
  [`production_mlp_final.onnx`](models/production_mlp_final.onnx).
- An ONNX interface that runs with ONNX Runtime's CPU provider. The model
  accepts 17 ordered raw features and includes its StandardScaler graph. Do
  not scale features externally. The recorded calibration temperature
  `3.824104` is metadata only; current inference does not apply it, so output
  scores are not temperature-calibrated probabilities.
- Bounded spectral, CFO, timing, carrier-phase, and symbol-rate analysis.
  CFO hypotheses can be considered during classification; the system does not
  universally correct CFO before classification. Timing recovery targets
  supported oversampled pulse-shaped PSK cases and does not imply a validated
  end-to-end decode for every oversampled signal.
- Hard-decision PSK/QAM/2FSK demodulation and hard-decision rate-1/2
  convolutional Viterbi decoding. The Viterbi implementation remains
  hard-decision; it is not a soft-decision or punctured-code decoder. Separate
  FEC modules include Reed-Solomon, concatenated decoding, and configured
  QC-LDPC decoding.
- Block, convolutional, diagonal, and project-scoped pseudo-random
  interleavers. The pseudo-random profile is a local adoption with explicit
  frame length and seed, not a standard-defined SIH radio profile.
- CRC-assisted phase-orientation selection when CRC presence is enabled.
  CRC-16 can collide, and the controlled random check observed one accepted
  candidate; it is not a proof that arbitrary garbage will be rejected.
- A bounded geometric signal-support gate at `0.35`. This is fit evidence,
  not an OOD probability or a universal unknown-signal rejection guarantee.

The FEC trial engine prunes structurally incompatible hypotheses; its work
depends on the capture and supplied metadata. It is not a fixed 1,456-trial
sweep or a 768-seed blind search. Pseudo-random interleaver support is
project-scoped and pinned to configured profiles rather than searched across
an unbounded seed space.

## Validation summary

- Windows test suite with `QT_QPA_PLATFORM=offscreen`: **632 passed, 0 failed,
  0 skipped, 0 errors, 34.04 seconds**.
- Vectorized Viterbi equivalence: 44 regression cases, 160 reference decode
  comparisons, 0 mismatches across decoded bits, path metrics, and termination
  behavior.
- Frozen T2-07, with declared payload length: runtime decreased from
  `427.6461785 s` (Cycle 1) to `32.1933978 s` (Cycle 2), a `13.28x` speedup
  and `92.47%` reduction. Modulation, demodulated bit count, accepted FEC,
  interleaver, CRC, and exact payload were checked; see the
  [runtime-only evidence](reports/tier2/fixes/cycle2_runtime_only/README.md).
- Frozen T2-01, T2-02, T2-03, T2-07, and T2-08 all passed CRC and exact-payload
  checks with `expected_payload_length=None` and no payload contents supplied.
  T2-07 took `136.6177 s` without the declared length, `4.24x` its Cycle 2
  declared-length runtime. See the [CRC phase-search evidence](reports/tier2/fixes/cycle2_crc_phase_search/README.md).
- A single controlled synthetic 8-samples/symbol QPSK run recovered timing but
  classified the signal as 8PSK and did not pass FEC/CRC or recover payload.
  Oversampled end-to-end decode is therefore not validated; the run is kept
  separate from Tier-2 evidence.
- T2-04 remains an unresolved 16-QAM classification limitation without
  ground-truth CFO. T2-05 now reports `PARTIAL / phase orientation unresolved`
  instead of an unqualified complete state.
- The automated GUI tests passed. A manual native-window run of **Run Demo**
  and visual inspection were not verified in the available desktop session.
- The Tier-2 evidence was generated without GNU Radio and does not establish
  real over-the-air performance. See the [historical Tier-2 methodology and
  results](docs/tier2_validation_note.md) and its final-pass addendum.

These results describe the named captures and test vectors only. They are not
an overall accuracy estimate or an independent live-radio benchmark.

## Run the desktop application

From the repository root with the locked dependencies installed:

```powershell
.venv\Scripts\Activate.ps1
python -m gui
```

Choose **Run Demo** for the deterministic offline demonstration, or load a
capture and enter required metadata in the left panel. The GUI displays the
analyzer's result; it does not recalculate DSP. Normal analysis does not require
a network service.

## Run tests

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest tests -q
```

## Production model contract

The checked-in [`production_ml_contract.json`](production_ml_contract.json)
records the ONNX digest, tensor shapes, feature order, class order, embedded
scaler, architecture, runtime provider, and calibration metadata. Current
inference expects raw feature values and does not apply external scaling or
temperature calibration. The model was not retrained or modified during this
pass.

## Evidence and claim guidance

- [Final pre-freeze report](reports/pre_freeze/final_pre_freeze_report.md)
- [Presentation claim sheet](docs/presentation_claims.md)
- [Cycle 2 runtime-only report](reports/tier2/fixes/cycle2_runtime_only/README.md)
- [Cycle 2 CRC phase-search report](reports/tier2/fixes/cycle2_crc_phase_search/README.md)
- [Controlled synthetic oversampled report](reports/validation/controlled_oversampled/README.md)

Do not present the measured T2-07 speedup as a general performance guarantee,
claim temperature-calibrated confidence, universal OOD rejection, GNU Radio or
OTA validation, or successful oversampled end-to-end decoding.
