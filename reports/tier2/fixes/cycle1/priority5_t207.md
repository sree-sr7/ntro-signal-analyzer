# Priority 5 — T2-07 FEC-search runtime

## Finding

T2-07 previously took 2,618.441 seconds (43m 38s). The raw-CFO classifier selected 8PSK for a QPSK capture, producing 392,832 received bits instead of the 261,888 bits implied by the corrected QPSK result. The baseline FEC log had 81 trial entries, of which 36 were structurally compatible when no expected payload length was supplied.

## Change

`core/fec/trial_engine.py` now checks length-preserving interleaver blocks and exact FEC/codeword/frame lengths before deinterleaving or decoding. Incompatible candidates remain in the trial log with reasons. Diagnostics report total hypotheses, structurally compatible candidates after pruning, and pruned candidates. `tests/integration/test_trial_pruning.py` verifies that known expected frame length preserves a real K7 candidate and calls no impossible convolutional code.

The one full T2-07 rerun used the same frozen capture and supplied its existing 114,160-bit expected payload length plus CRC flag. No payload bits were passed to analysis.

## Before / after

| Measure | Baseline | After |
|---|---:|---:|
| Analyzer modulation | 8PSK | QPSK |
| Received/demodulated bits | 392,832 | 261,888 |
| Total candidate hypotheses | 89 theoretical; 81 trial entries | 89 theoretical; 82 trial entries |
| Structurally compatible after pruning | 36 | 9 |
| Ground-truth Concat_RS223_Conv7 / Block_16x16 | present, failed CRC | evaluated and accepted |
| CRC / exact payload | fail / no recovery | pass / exact 114,160 bits |
| Runtime | 2,618.441 s | 427.646 s |

Runtime improved 6.12× (83.7% lower). Artifacts: `reports/tier2/fixes/cycle1/T2-07/analysis_result.json`, `comparison.json`, `runtime.json`, `run.log`, and `recovered_payload_bits.bin`.

Focused structural checks plus the T2-03 regression: 11 passed. Only one full T2-07 FEC run was performed in this cycle.
