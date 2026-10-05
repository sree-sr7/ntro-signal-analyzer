# Tier-2 Independent Validation Note

## Objective and boundary

Tier-2 measures the current application's behavior on independently
constructed captures. The primary application call is the public
`Analyzer.analyze_file(...)` API. This run does not change DSP, synchronization,
classifier/model, demodulation, FEC, interleaver, or analyzer code and does not
retrain or re-export the production model.

## Frozen baseline and environment

The pre-harness source snapshot and environment manifest were recorded before
Tier-2 files were added. The checkout has no `.git` directory, so a repository
revision and `git status` are unavailable. The production ONNX file was present
and its SHA-256 was recorded in the manifest. No previous validated model hash
was found in the repository or supplied records, so historical model continuity
is explicitly unavailable. GNU Radio was not installed.

The timestamped environment manifest and snapshot hash are stored in
`reports/tier2/manifests/`. Runtime versions, model digest, and the limitation
above are repeated in the baseline run manifest.

## Independence methodology

`tools/tier2/generate_signals.py` imports Python standard-library modules,
NumPy, SciPy's WAV writer (only for T2-05), and the external Komm library. It
does not import `core.*`, `tests`, `scripts`, `generators`, `data`, `models`,
training arrays, or the production demo. It independently creates the baseband
symbols/waveforms, gain, phase, carrier offset, AWGN, and OFDM IFFT waveform.

Convolutional encoding uses the separately installed `komm.ConvolutionalCode`
API and feedforward polynomials. RS(255,223), CRC-16/CCITT-FALSE, rectangular
block, circular-branch convolutional, and 16×16 diagonal interleavers are
implemented locally in the Tier-2 generator. Their mappings are based on
separately reviewed definitions and the documented project-scoped diagonal
mapping; none is imported from application code. The local RS result was
compared against the installed `reedsolo` 1.7.0 implementation as a parameter
convention check, but `reedsolo` is not a generator dependency. CRC check
vector `123456789` returned `0x29b1`.

The runner records and prints a static AST import audit for the generator. That
audit checks direct generator imports; it is not a whole-process sandbox or
proof about arbitrary behavior of third-party dependencies. The analyzer
runner imports the normal application and is intentionally outside that
generator-only import audit.

## Capture and ground-truth method

The exact case configurations are in
`tools/tier2/config/baseline_scenarios.json`. Each capture directory contains
the raw waveform, `ground_truth.json`, packed payload bits when applicable,
the complete serialized `AnalysisResult`, a comparison JSON file, and a local
console log. Raw IQ is little-endian complex64. T2-05 is stereo signed PCM16
WAV with channel 0=I and channel 1=Q; this quantizes each component to the
nearest signed 16-bit value after clipping to the documented PCM range.

The runner supplies only raw-IQ sample-rate metadata, the WAV stereo-as-IQ
flag, and the FSK samples-per-symbol/tone values required to load or demodulate
those captures. It does not pass expected modulation, FEC, interleaver, payload
contents, payload length, SNR, CFO, gain, phase, or seed to the analyzer. CRC
presence is enabled only for generated in-scope FEC cases. T2-05 disables FEC
search because its declared ground truth has no FEC. OOD cases run the
analyzer's default FEC search so any accepted recovery is recorded.

Seeds, payloads, gains, and phases are deterministic and case-specific. They
are stored in each ground-truth manifest. The selected seeds were checked
against the repository's visible training/generator configuration locations;
no match was found. Timing offset is zero for all cases, and timing recovery
is not requested because PSK/QAM captures are symbol-rate samples. Exact
sample counts and duration are present in each ground-truth manifest.

## Scenario list

| ID | Signal | FEC | Interleaver | SNR | CFO |
|---|---|---|---|---:|---:|
| T2-01 | QPSK | Conv_R12_K5 | Block_8x8 | 12 dB | +150 Hz |
| T2-02 | BPSK | Conv_R12_K7 | None | 8 dB | −200 Hz |
| T2-03 | 8PSK | Conv_R12_K7 | Convolutional_Depth_12 | 15 dB | +400 Hz |
| T2-04 | 16QAM | RS_255_223 | Block_32x32 | 18 dB | +100 Hz |
| T2-05 | QPSK | None | None | 20 dB | 0 Hz |
| T2-06 | genuine 2FSK | Conv_R12_K3 | None | 10 dB | 0 Hz |
| T2-07 | QPSK | Concat_RS223_Conv7 | Block_16x16 | 12 dB | +250 Hz |
| T2-08 | QPSK | Conv_R12_K7 | Diag_16x16_S1 | 15 dB | +300 Hz |
| T2-09 | complex Gaussian noise | None expected | None | n/a | 0 Hz |
| T2-10 | independent OFDM | None expected | None | 20 dB | 0 Hz |

T2-04 contains 256 complete RS words. Appending its two-byte CRC to the
57,086-byte payload yields exactly 256 × 223 data bytes; the encoded output is
also an exact multiple of 32×32 interleaver blocks. T2-07 contains 64 complete
concatenated codewords, including CRC in the RS data, and 1,023 complete
16×16 blocks. The generator does not pad or truncate these frames.

## Result gates and status definitions

For in-scope captures, the comparison reports separate input, modulation,
demodulation, FEC/interleaver/CRC, and payload gates. `PASS` means all
applicable gates pass, including exact payload hash equality when a payload is
defined. `PARTIAL` means input, modulation, and demodulation succeeded but a
downstream required gate did not. `FAIL` means an in-scope gate failed before a
partial result can be claimed. `REJECTED` is the analyzer's explicit rejection
state. `ERROR` records a harness exception or input failure. `BLOCKED` denotes
a case that could not be run for an explicitly recorded prerequisite reason.

For OOD captures, `OOD_PASS` requires explicit rejection/unknown classification
and no accepted FEC or CRC-valid recovery. Any trusted supported classification
or accepted recovery is `OOD_FAIL`. A lack of process crash alone is never
counted as OOD success.

## OOD methodology

T2-09 is generated as independent complex circular Gaussian noise. T2-10 is an
OFDM waveform with 64-point NumPy IFFT, cyclic prefix, 52 occupied symmetric
subcarriers, and QPSK subcarrier symbols. Neither is assigned a supported
ground-truth modulation class. The comparison records the classifier result,
confidence, analyzer state, candidate count, CRC state, and whether any
recovery was accepted.

## Known limitations

- This environment did not provide GNU Radio. The run uses NumPy, SciPy's WAV
  writer, and Komm instead; it must not be described as GNU Radio-generated.
- No prior validated model digest was supplied or found, so historical model
  continuity could not be verified. The current digest is still recorded.
- The import audit is a static check on the generator source, not dependency
  sandboxing.
- The scenario set is a functional baseline, not a broad statistical
  benchmark. No overall accuracy percentage is inferred from it.
- The optional independent IEEE 802.11n QC-LDPC reference encoder was not
  established for this run; T2-11-LDPC is therefore blocked.
- Application outputs, including rejected and failed outcomes, are preserved
  as produced. This note does not recommend or apply a corrective change.

## Reproducibility

Focused harness checks:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\tier2\test_tier2_harness.py -q
```

Sequential capture run:

```powershell
.\.venv\Scripts\python.exe tools\tier2\run_tier2.py --run-dir reports\tier2\baseline\<new-unique-run-id>
```

The baseline used a newly created timestamped run directory that was checked
to be empty, and the aggregate summary paths were checked absent before the
run; prior Tier-2 results were not overwritten. Future invocations must use a
new unique run directory and preserve existing aggregate reports. Environment,
source snapshot hash, generator import audit, seeds, capture hashes, analyzer
configuration keys, and outcomes are saved with the run artifacts. Tier-2 references are listed in
`docs/research_references.md` and `docs/references.bib`.

## Observed baseline outcomes

Run `20260930_201445_269` used the frozen source snapshot SHA-256
`f7afd5e055ca91474950f0bc6ab0853d87fff23c949e6468e5f9b2763d2cffbc` and model
SHA-256
`0e0ea216116f8859775b1e55167520db63f95dd93a25a77243bff803ed7fb677`. Current
environment: Python 3.12.10 on Windows 11 build 26200, NumPy 2.5.3, SciPy
1.18.1, ONNX Runtime 1.30.0, PyQt6 6.11.0, pyqtgraph 0.14.0, and Komm 0.36.0.
GNU Radio was unavailable. No prior model digest was found for a continuity
comparison.

The table below reflects the reviewed canonical comparison. Injected and
estimated CFO values are in hertz. `—` means the corresponding value or
recovery was not available.

| Scenario | Predicted | Confidence | CFO injected / estimated | Accepted FEC / interleaver | CRC | Payload | Status |
|---|---|---:|---:|---|---|---|---|
| T2-01 | 8PSK | 0.999046 | +150 / +145.202 | — / — | False | — | FAIL |
| T2-02 | 8PSK | 0.999998 | −200 / −198.853 | — / — | False | — | FAIL |
| T2-03 | 8PSK | 0.999998 | +400 / +400.973 | — / — | False | — | PARTIAL |
| T2-04 | Unknown | 0.054236 | +100 / — | — / — | — | — | REJECTED |
| T2-05 | QPSK | 0.997927 | 0 / +0.037 | None / None | — | False | PARTIAL |
| T2-06 | 2FSK | 0.999999 | 0 / — | Conv_R12_K3 / None | True | True | PASS |
| T2-07 | 8PSK | 0.9999998 | +250 / +249.999 | — / — | False | — | FAIL |
| T2-08 | 8PSK | 0.9999991 | +300 / +298.866 | — / — | False | — | FAIL |
| T2-09 | 16QAM | 1.000000 | 0 / — | None / None | — | — | OOD_FAIL |
| T2-10 | 16QAM | 1.000000 | 0 / — | None / None | — | — | OOD_FAIL |

T2-01, T2-02, T2-07, and T2-08 were QPSK/BPSK ground-truth captures labeled
8PSK. T2-03's modulation label matched, but its FEC/interleaver/CRC gates did
not. T2-04 was explicitly rejected as unknown rather than accepted as 16QAM.
T2-05 classified and demodulated as QPSK, but all 8,192 recovered hard-decision
bits did not match the independently stored payload. T2-06's supplemental
same-capture run recovered the exact payload with Conv_R12_K3 and CRC. Its
first invocation is preserved and shows the harness omitted the required tone
pair; the corrected invocation supplied `(-3000, +3000)` Hz. Its raw capture
SHA-256 is identical in both directories.

For T2-09 and T2-10, the classifier accepted 16QAM rather than explicitly
rejecting an unsupported/out-of-distribution signal. The FEC search reported
50 and 45 accepted trial candidates respectively, while neither reported a
CRC-valid recovery. Both are OOD failures because no false trusted modulation
should be accepted.

Reviewed aggregate counts: 10 scenarios attempted; PASS 1, PARTIAL 2, FAIL 4,
REJECTED 1, ERROR 0, BLOCKED 0, OOD_PASS 0, and OOD_FAIL 2. The in-scope count
is 1 successful and 7 failed/partial/rejected; both OOD scenarios failed.
These are counts over the ten specified captures, not a general accuracy
estimate. T2-07 required 2,618.441 seconds (43 minutes 38 seconds) in the
analyzer's FEC/interleaver search.

The original runner comparisons and summary are retained with `_initial`
filenames. First reviewed summaries are retained with `_review_v1` filenames.
The canonical summaries include only comparison corrections described in the
run manifest; raw waveform captures and `AnalysisResult` files were not
changed.

## Final controlled pass addendum — 2026-10-04

This section records checks made after the historical Cycle 1 run. It does not
edit or replace the frozen baseline or Cycle 1 evidence.

- The production ONNX digest was independently checked again and matched the
  digest recorded above. ONNX Runtime 1.30.0 opened the model on
  `CPUExecutionProvider`; its input remained `[batch, 17]` float32 and output
  `[batch, 5]` float32. The production model and scaler were not changed.
- The Cycle 1 checksum manifest revalidated at **117/117 unchanged**. Cycle 2
  outputs were written separately under `reports/tier2/fixes/cycle2_*`.
- On the same frozen T2-07 capture, the runtime-only run with declared payload
  length took `32.1934 s` versus the historical Cycle 1 `427.6462 s`; CRC and
  exact payload matched. With `expected_payload_length=None`, the same capture
  took `136.6177 s`, still passing CRC and exact payload. Five frozen cases
  (T2-01/02/03/07/08) passed exact-payload and CRC checks without a declared
  payload length. None supplied payload contents.
- The randomized CRC-phase negative check evaluated 28,480 hypotheses and
  observed one CRC accept that passed structure/FEC validity. This does not
  establish a zero false-accept rate. See
  `reports/tier2/fixes/cycle2_crc_phase_search/false_accept_experiment.md`.
- The frozen T2-04 capture remains unknown/rejected without ground-truth CFO.
  A saved diagnostic that removes the known +100 Hz offset is oracle-only.
- The frozen T2-05 capture remains a four-way unresolved QPSK orientation. A
  fresh current-code analysis reports `partial`; the historical Cycle 1
  `analysis_result.json` is preserved as originally recorded. The GUI now
  renders the state as `PARTIAL / phase orientation unresolved`.
- The separate one-shot synthetic oversampled run recovered timing, but
  classified its QPSK input as 8PSK and failed FEC/CRC. It did not recover the
  payload and must not be counted as Tier-2 or successful end-to-end evidence.
- The complete Windows test suite passed 632 tests with Qt offscreen. Manual
  visual verification of the deterministic GUI demo remained unverified in
  the available desktop automation session. This addendum is not a declaration
  that the project passed every freeze gate.

## GUI validation

The actual `gui.main_window.MainWindow` was shown on the native Windows Qt
platform for the passing T2-06 supplemental capture and rejected T2-04 capture.
The real `select_file` and `start_analysis` handlers ran, `AnalysisWorker`
called the analyzer, and `populate_result` displayed the returned result.
All six tabs opened in both cases. Waveform and spectrum plot items were
present (two and one items respectively); FEC/frame detail tables contained
81 rows for T2-06 and one fallback row for T2-04. Native QWidget renderings are
saved under the run's `gui/` directory. The separate computer-use inventory
exposed no targetable native windows, so this validation used actual Qt
windows and saved native renderings rather than a desktop click-through.

The focused Tier-2 harness checks passed: 5 passed. No project-wide test suite
was run. No production/core code or model artifact was changed.

## Native GUI follow-up — 2026-10-05

After the 2026-10-04 validation report, the project developer completed native
Windows manual checks and reported PASS for application launch, deterministic
demo, T2-01 manual analysis, T2-02 manual IQ analysis, T2-09 rejection, T2-05
stereo-WAV/I-Q partial behavior, and T2-07 responsiveness during long-running
analysis. This follow-up supersedes the earlier statement that the manual GUI
demo gate was unverified. The 632-test result and other measurements above
were reused; no tests or Tier-2 analyses were rerun for this release task.

## Fresh GUI-branch validation — 2026-10-05

Run `20261005_post_gui_validation` was generated with the current Tier-2 runner
and analyzer code at revision `c9fdb7a3df3754bad59bc4641d351a6c7f31828b` on
`gui-redesign`. The NumPy/Komm independent-generator import audit passed. The
per-case captures, full analysis results, comparisons, and logs are in
`reports/tier2/baseline/20261005_post_gui_validation/`; aggregate reports are
in `reports/tier2/baseline_summaries/20261005_post_gui_validation/`.

| Case | Result | Runtime | Key evidence |
|---|---|---:|---|
| T2-01 QPSK | PASS | 0.914 s | `Conv_R12_K5`, `Block_8x8`, CRC pass, exact payload match |
| T2-02 BPSK | PASS | 0.383 s | `Conv_R12_K7`, CRC pass, exact payload match |
| T2-03 8PSK | PASS | 0.366 s | `Conv_R12_K7`, convolutional depth-12 interleaver, CRC pass, exact payload match |
| T2-04 16-QAM | REJECTED | 0.241 s | Classifier returned unknown; no demodulation/FEC/payload evidence |
| T2-05 QPSK stereo WAV | PARTIAL | 0.035 s | QPSK classified/demodulated; payload did not match |
| T2-06 2FSK | PASS | 0.191 s | `Conv_R12_K3`, CRC pass, exact payload match |
| T2-07 QPSK | PASS | 90.122 s | `Concat_RS223_Conv7`, `Block_16x16`, CRC pass, exact payload match |
| T2-08 QPSK | PASS | 2.847 s | `Conv_R12_K7`, `Diag_16x16_S1`, CRC pass, exact payload match |
| T2-09 noise | OOD_PASS | 0.016 s | Explicitly rejected; no accepted recovery |
| T2-10 OFDM | OOD_PASS | 0.011 s | Explicitly rejected; no accepted recovery |

Aggregate counts: 6 PASS, 1 PARTIAL, 1 REJECTED, 2 OOD_PASS, and zero FAIL or
ERROR. This is one fixed, controlled synthetic capture set; it does not imply
an overall accuracy rate. T2-04 remains unresolved, T2-05 remains partial,
and the two observed OOD rejections do not establish universal OOD rejection.
GNU Radio was unavailable, so this is not GNU Radio evidence. It is not OTA or
real-world evidence. The prior oversampled-QPSK failure and observed CRC false
accept remain limitations.

The existing runner had a hard-coded note that Git metadata was unavailable,
even though this checkout had `.git`. The run manifest was corrected using the
recorded Git revision. The runner now records revision and working-tree status
on future runs; that source change affects provenance metadata only, not signal
generation or analyzer behavior.
