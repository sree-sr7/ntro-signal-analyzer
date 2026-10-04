# FINAL PRE-FREEZE REPORT — SIH 26147

**Repository:** `C:\Users\HP\Documents\NTRO-Signal-Analyzer`  
**Pass date:** 2026-10-04  
**Decision:** **DO NOT FREEZE** — the required manual native-window demo check
could not be performed in this desktop session. No freeze tag was created.

## 1. Patch package inspected

Inspected `C:\Users\HP\Downloads\ntro_patches.zip`, its README, both patch
files, and the supplied patch overlay. The package contained Patch A, the
optional Patch B, and their tests. The archive was not copied wholesale into
the repository.

Patch A adds a vectorized hard-decision Viterbi path while preserving the
reference decoder behavior, resolves near-equal periodicity candidates in
favor of the shorter period, discards the configured initial transient only
for classifier feature extraction after timing recovery, and updates fixtures
and regressions for the signal-support gate and numerical ties. It does not
change Viterbi into a soft-decision decoder.

Patch B allows PSK phase-orientation search without a declared payload length
when CRC presence is enabled. It retains CRC-based selection and does not use
payload contents.

## 2. Patch A applied — exact files

- `core/fec/convolutional.py`
- `core/bitstream/frame_analyzer.py`
- `core/pipeline/analyzer.py`
- `tests/integration/test_analyzer_pipeline.py`
- `tests/integration/test_oversampled_classification.py` (new)
- `tests/unit/test_modulation_features.py`
- `tests/unit/test_viterbi_vectorized_equivalence.py` (new)

The equivalence test seed was changed from Python `hash()` to a deterministic
SHA-256-derived seed. A stale GUI test assertion found by the full suite was
corrected. A separate current-code issue found in the T2-05 review was fixed:
unresolved phase orientation now prevents an unqualified `complete` analyzer
state, and the GUI shows `PARTIAL / phase orientation unresolved`.

## 3. Patch B applied — exact files

Patch B was applied as a small change to `core/pipeline/analyzer.py`. The
resulting no-length phase-search runs still require CRC presence and CRC
selection. No payload contents or ground-truth payload bits were supplied.

## 4–5. Windows suite and Viterbi equivalence

Final full suite, using the project Windows virtual environment and
`QT_QPA_PLATFORM=offscreen`:

- Collected/passed: **632 / 632**
- Failed: **0**
- Skipped: **0**
- Errors: **0**
- Runtime: **34.04 seconds**

The dedicated Viterbi suite reported **44 regression cases** and **160 exact
reference decode comparisons**, with **0 mismatches**. Comparisons covered
decoded bits, path metrics, termination modes, traceback survivors, and
tie-breaking behavior across the supported configurations.

## 6. Production model integrity

- SHA-256: `0e0ea216116f8859775b1e55167520db63f95dd93a25a77243bff803ed7fb677`
- ONNX Runtime opened the model with `CPUExecutionProvider`.
- Input: float32 `[batch, 17]`; output: float32 `[batch, 5]`.
- The 17-feature sequence and class order are recorded in
  `production_ml_contract.json` and are unchanged.
- The StandardScaler is embedded in the ONNX graph; inference expects raw
  feature values and applies no external second scaling.
- Calibration temperature `3.824104` is recorded as metadata and is not used
  by current inference.
- No retraining or scaler refit occurred; the ONNX file was not modified.

## 7. Baseline and Cycle 1 integrity

- Cycle 1 baseline checksum manifest: **117/117 unchanged**.
- Independent Cycle 1 safety snapshot: **63/63 unchanged**.
- Protected snapshot covering the frozen baseline, Cycle 1, scenario config,
  and model: **169/169 unchanged**.
- `reports/tier2/baseline/20260930_201445_269/`,
  `reports/tier2/fixes/cycle1/`, and
  `tools/tier2/config/baseline_scenarios.json` were not modified.

## 8. T2-07 Cycle 2 runtime-only validation

The same frozen T2-07 capture was analyzed; it was not regenerated. New output
is isolated in `reports/tier2/fixes/cycle2_runtime_only/`.

| Measurement | Cycle 1 | Cycle 2 runtime-only |
|---|---:|---:|
| Runtime | 427.6461785 s | 32.1933978 s |
| Speedup | — | 13.2837× |
| Runtime reduction | — | 92.47195% |
| Modulation | QPSK | QPSK |
| Demodulated bits | 261,888 | 261,888 |
| Structurally compatible candidates | — | 9 |
| Accepted FEC | — | `Concat_RS223_Conv7` |
| Interleaver | — | `Block_16x16` |
| CRC | — | PASS |
| Payload | — | 114,160 bits; exact match |

Cycle 2 considered 89 theoretical hypotheses and recorded 82 trial-log entries.
The accepted payload matches both the frozen payload truth and Cycle 1's
recovered payload; candidate trial logs match Cycle 1. The speedup is reported
for this complete Cycle 2 change set, not attributed solely to Viterbi.

## 9. Patch B no-length frozen-capture results

All five runs used `expected_payload_length=None`, CRC enabled, and no payload
contents. Each reported phase status `crc_selected`, correct modulation/FEC/
interleaver, CRC PASS, and exact payload match.

| Capture | Runtime | Modulation | FEC | Interleaver | Recovered bits |
|---|---:|---|---|---|---:|
| T2-01 | 2.9092 s | QPSK | `Conv_R12_K5` | `Block_8x8` | 1,932 |
| T2-02 | 1.2825 s | BPSK | `Conv_R12_K7` | None | 2,048 |
| T2-03 | 1.0468 s | 8PSK | `Conv_R12_K7` | `Convolutional_Depth_12` | 2,048 |
| T2-07 | 136.6177 s | QPSK | `Concat_RS223_Conv7` | `Block_16x16` | 114,160 |
| T2-08 | 1.5630 s | QPSK | `Conv_R12_K7` | `Diag_16x16_S1` | 4,074 |

No-length T2-07 took 104.4243 seconds longer than its Cycle 2 declared-length
run: a 324.37% increase, or 4.2437× the runtime. It still recovered the exact
payload with CRC PASS.

## 10. Randomized CRC false-accept check

Reproducible seed `2614701`; 80 random QPSK frames, 512 symbols per frame, four
phase rotations, and 89 theoretical FEC/interleaver hypotheses per phase:

- Theoretical hypotheses: **28,480**
- Actual trial evaluations: **27,520** (960 skipped as structurally
  incompatible)
- Structurally compatible evaluations: **16,000**
- FEC-valid candidate outputs: **12,800**
- CRC accepts / accepted candidates / final accepted garbage frames: **1 / 1 / 1**

The accepted candidate was an uncoded `fec_type=None` candidate with CRC PASS
and 1,008 decoded data bits. This is an observed false accept, not evidence of
a zero false-positive rate. For an independent random candidate, CRC-16 has an
approximate `2^-16` collision probability before structural and FEC
validation. Full details are in
`reports/tier2/fixes/cycle2_crc_phase_search/false_accept_experiment.md`.

## 11. Controlled synthetic oversampled check

One synthetic capture only; no retries or same-capture tuning. QPSK with RRC,
8 samples/symbol, +120 Hz CFO, 20 dB AWGN, 1,000 payload bits, CRC-16,
terminated `Conv_R12_K9`, and `Block_8x8` interleaving.

- Classifier returned **8PSK**, not the generated QPSK.
- Timing recovery ran; classifier transient discard was 307 samples.
- 3,072 demodulated bits versus 2,048 transmitted coded bits.
- Common-prefix mismatch was 0.509765625; this is not a valid BER because the
  modulation label and bit count differed.
- No FEC candidate; CRC failed; no payload was recovered.

Conclusion: this controlled oversampled end-to-end decode did **not** validate.
It is separate from Tier-2 evidence. No tuning was performed.

## 12. T2-04 limitation

The frozen T2-04 capture was analyzed without ground-truth CFO. The analyzer
returned `unknown` / `rejected`; the bounded signal-support score passed
(`0.57809` against `0.35`), but class scores remained ambiguous and no coarse
CFO estimate was available. Removing the known +100 Hz CFO in the historical
diagnostic is oracle-only and is not an analyzer input. T2-04 remains an
unresolved 16-QAM / RS(255,223) / `Block_32x32` case.

## 13. T2-05 phase status

The frozen capture remains QPSK with four unresolved phase orientations and no
FEC/CRC recovery requested. A fresh current-code analysis reports `partial`.
The GUI status is `PARTIAL / phase orientation unresolved`; it does not present
the case as a successful decode. The old Cycle 1 result file is preserved as
historical evidence.

## 14. GUI manual result

Automated GUI tests passed as part of the 632-test suite. The requested manual
Windows check—launching the native window, running **Run Demo**, and visually
checking plots, parameters, result, progress, and offline behavior—remains
**unverified**. The available computer-use surface returned no native apps or
windows (only browser surfaces), so this session could not perform a visual
click-through. Historical Cycle 1 native renderings cover T2-04 and T2-06, not
the final deterministic demo.

This missing required gate is the direct reason for `DO NOT FREEZE`.

## 15. README and claim reconciliation

`README.md` was rewritten against current code and evidence. It no longer says
the model is absent or Tier-2 validation is merely upcoming, and it describes
the actual model, supported scope, hard-decision Viterbi, bounded trial engine,
unresolved limitations, and current validation. It does not claim a fixed
1,456-trial sweep, 768-seed blind search, rate-1/3 convolutional decoding,
temperature-calibrated confidence, universal OOD rejection, simplistic
pre-classification CFO correction, GNU Radio/OTA validation, successful
oversampled decoding, or T2-05 completion.

`docs/presentation_claims.md` gives exact safe slide wording and claims to
avoid, with the no-length metadata caveat and CRC false-accept evidence.
`docs/tier2_validation_note.md` now has a dated addendum distinguishing current
checks from preserved historical records.

## 16. Production ML contract

`production_ml_contract.json` is present. It records the exact ONNX digest,
tensor shapes and types, 17 feature names in order, five-class order, embedded
StandardScaler, architecture, CPU runtime, calibration metadata marked unused,
and the bounded support-gate meaning. Its scaler parameters were sourced from
the hash-matched model contract; no model or scaler refit was done.

## 17. OOD/support-gate claim check

The current support gate threshold is **0.35**. The frozen noise and OFDM
examples scored **0.1853** and **0.1924** and were rejected. These values are
bounded geometric signal-fit evidence, not calibrated OOD probabilities,
confidence guarantees, or universal unknown-signal rejection.

## 18. Diagnostic path cleanup

`tools/tier2/fix_cycle1/diag_ood_paths.py` now resolves the contract relative to
the repository root and gets classes from the contract. Its docstring and
threshold notes identify it as a historical local diagnostic, not production
runtime. No Downloads or `USERPROFILE` path remains.

## 19. Secrets scan

Scanned **319 text files** for high-confidence AWS, GitHub, Slack, OpenAI-style,
Google API, JWT, private-key, credential-assignment, and URL-credential
patterns. Result: **0 matches** and **0 credential-like filenames**. The scan
does not treat the local `.venv` package contents as project source.

## 20. Source-package cleanup and manifest

`.gitignore` excludes `.venv`, Python/pytest caches, ZIP backups, the prior
Step 78 local audit artifacts, and credential/key files from the source
package. They remain in the working folder where applicable. The `data/`
directory is empty. No large source datasets were found; the largest retained
historical report is a 17.5 MB analyzer JSON in the frozen baseline. The
historical validation reports and frozen evidence remain present.

The SHA-256 manifest for the final source/package tree is
`reports/pre_freeze/final_source_sha256.txt`. It excludes `.venv`, `.git`,
backup ZIPs, local audit artifacts, `__pycache__`, `.pytest_cache`, and itself.

## 21. Version control and release decision

The workspace did not contain Git initially. Git is initialized for this
pre-freeze source state, but there is **no freeze tag** because the manual GUI
gate is unverified. Any local commit is explicitly a pre-freeze validation
commit, not `v1.0.0-sih26147-freeze`.

## 22. Exact remaining limitations

- Manual native Windows deterministic-demo visualization is unverified.
- One of 28,480 randomized hypotheses produced an accepted CRC-16 false
  candidate; do not claim zero false accepts.
- Controlled oversampled QPSK was classified as 8PSK and did not decode.
- T2-04 16-QAM remains unknown/rejected without ground-truth CFO.
- T2-05 phase orientation remains unresolved and is correctly partial.
- The OOD/support gate is bounded fit evidence with a 0.35 threshold, not a
  calibrated probability or universal rejection guarantee.
- Current model scores do not use temperature calibration.
- The Tier-2 evidence is a small frozen-capture set, not GNU Radio or OTA
  validation and not a general accuracy benchmark.
- No end-to-end DQPSK, 64-QAM, or 4FSK result was established in this pass.

## 23. Exact safe presentation wording

The following slide statements are safe when their scope stays with them:

- **“On frozen T2-07, the analyzer runtime fell from 427.646 seconds in Cycle
  1 to 32.193 seconds in Cycle 2, a 13.28× speedup (92.47% less runtime), with
  CRC pass and exact payload recovery.”** One frozen capture, CRC enabled, and
  the 114,160-bit expected length supplied; do not attribute the entire gain
  solely to Viterbi or generalize it.
- **“Five frozen cases—T2-01, T2-02, T2-03, T2-07, and T2-08—passed CRC and
  exact-payload checks without a declared payload length.”** CRC presence and
  sample rate were supplied; no payload contents were supplied. This is not a
  no-metadata claim.
- **“The current Windows test suite passed 632 of 632 tests, including
  headless GUI tests.”** The required manual visual demo check remains
  unverified.
- **“The checked-in ONNX classifier uses 17 ordered input features and five
  output classes, with its StandardScaler embedded in the ONNX graph.”** The
  recorded temperature is not used in current inference.
- **“The signal-support gate uses a 0.35 bounded geometric fit-evidence
  threshold; the recorded noise and OFDM examples scored 0.1853 and 0.1924 and
  were rejected.”** These two examples do not establish universal rejection.
- **“One controlled 8-samples/symbol synthetic run exercised timing recovery;
  the signal was misclassified and did not pass FEC/CRC.”** Do not describe
  this as successful oversampled decoding.

## 24. Exact unsafe presentation claims

Do not claim “100% accuracy,” “6/8 blind recovery,” “recovery with no
metadata,” “1,456 FEC trials,” or a “768-seed blind search.” The five no-length
cases still used sample-rate and CRC-presence metadata; the trial engine prunes
structurally incompatible candidates and its pseudo-random support is
project-scoped and pinned to configured profiles. Do not claim rate-1/3,
soft-decision or punctured convolutional Viterbi; the implemented Viterbi path
is hard-decision rate-1/2. Do not claim temperature-calibrated confidence,
universal OOD rejection, guaranteed CFO correction before classification,
zero CRC false accepts, GNU Radio validation, real OTA validation, successful
oversampled decoding, T2-04 recovery, T2-05 completion, or verified end-to-end
DQPSK/64-QAM/4FSK.

## 25. Final verdict

**FINAL VERDICT = DO NOT FREEZE.** The code, model hash, preserved baselines,
automated tests, Viterbi equivalence, and Cycle 2 capture checks are recorded.
The explicitly required manual native-window demo check was not completed in
this session. No freeze commit or `v1.0.0-sih26147-freeze` tag is claimed.
