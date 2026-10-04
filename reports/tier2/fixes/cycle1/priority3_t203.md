# Priority 3 — T2-03 FEC/CRC

## Finding

The trace proved the convolutional code, interleaver, Viterbi decoder, and CRC are mutually compatible: true transmitted bits passed project deinterleaving, K7 decoding, and CRC with zero path metric. The first divergence was symbol-to-bit demodulation. The saved capture had a constant 8PSK phase offset, and carrier recovery rejected its linear-fit phase estimate. Even the best fixed 45-degree rotation left 12.10% BER.

After coarse CFO correction, a circular mean of the 8th-power phase had coherence 0.613 and estimated the continuous phase component. The remaining eightfold phase ambiguity cannot be selected from the constellation alone, but the supplied expected payload length and CRC provide legitimate frame evidence.

## Change

`core/pipeline/analyzer.py` now uses a phase-only circular estimate when coarse CFO correction succeeded and linear carrier recovery was rejected. With CRC and an expected payload length, it tries the finite PSK phase rotations and selects only a CRC-accepted FEC result. OOD rejection still exits before this path.

## Same-capture result

Using the frozen T2-03 capture, expected payload length 2,048, and CRC enabled: Conv_R12_K7 / Convolutional_Depth_12 passed CRC and recovered all 2,048 reference payload bits exactly. Phase candidate 1 was selected after 2 candidates were tested. Runtime was 2.078 s. Result: `reports/tier2/fixes/cycle1/T2-03/priority3_focused_result.json`.

Focused regression: `pytest tests/integration/test_t203_phase_recovery.py -q`: 1 passed.

The later final end-to-end rerun is recorded in `T2-03/comparison.json` and `runtime.json`: it reproduced the same CRC pass and exact payload at 3.880 s with the full analyzer output enabled. The 2.078 s measurement above is from the focused phase-recovery capture check.

## Remaining limitation

When CRC and expected payload length are not supplied, rotational ambiguity remains unresolved and is reported as such. This cycle does not infer those values from hidden ground truth.
