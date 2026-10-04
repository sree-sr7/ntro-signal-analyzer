# Priority 4 — T2-05 WAV payload

## Finding

`trace_t205.py` confirmed the WAV loader reads stereo channel 0 as I, channel 1 as Q, preserves 1 MHz sample rate, and matches manual PCM/32768 scaling exactly. Clipping was present on 24.5% of PCM components, but with the recorded carrier phase removed the maximum sample angular error was 16.4 degrees, well inside the QPSK decision half-sector. Clipping and WAV channel handling do not explain the payload mismatch.

The concrete defect was in QPSK carrier phase recovery. It used the fourth-power phase as if the QPSK constellation anchor were 0 degrees, while this repository's QPSK mapping is anchored at 45 degrees. The returned phase was 0.222 rad instead of a channel phase equivalent to 0.981 rad modulo 90 degrees.

## Change

`core/sync/carrier_recovery.py` now accepts a constellation phase anchor (default 0 preserves callers using zero-anchored PSK). `core/pipeline/analyzer.py` passes 45 degrees for QPSK. A focused carrier regression covers the same nonzero phase.

## Same-capture check and limitation

On the frozen WAV, the recovered QPSK phase is now −0.564 rad, equivalent to the transmitted channel phase modulo 90 degrees. The independent payload maps exactly under phase candidate 2 of 4 (0 BER); candidate selection remains unresolved because this capture supplies no CRC or sync word. The analyzer correctly reports four unresolved rotations. It does not use the saved payload to choose a phase. WAV-specific output is in `reports/tier2/fixes/cycle1/T2-05/priority4_focused_result.json`.

`pytest tests/unit/test_carrier_recovery.py -q`: 4 passed. The WAV loader was not modified.
