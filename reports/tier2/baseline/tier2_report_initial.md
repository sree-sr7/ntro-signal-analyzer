# Tier-2 Independent Validation Baseline

Run: `20260930_201445_269`
Scenarios attempted: 10

No single accuracy percentage is reported; this is a fixed, small scenario set.

| Scenario | Ground truth | Predicted | Confidence | CFO in / estimated | FEC / interleaver expected | Accepted | CRC | Payload | Status | Failure reason |
|---|---|---|---:|---:|---|---|---|---|---|---|
| T2-01 | QPSK | 8psk | 0.9990460375766295 | 150.0 / 145.2021974980235 | Conv_R12_K5 / Block_8x8 | None / None | False | None | FAIL | failed gates: modulation, fec_interleaver_crc |
| T2-02 | BPSK | 8psk | 0.9999977528614875 | -200.0 / -198.8529920697958 | Conv_R12_K7 / None | None / None | False | None | FAIL | failed gates: modulation, fec_interleaver_crc |
| T2-03 | 8PSK | 8psk | 0.9999980433091609 | 400.0 / 400.97310813945046 | Conv_R12_K7 / Convolutional_Depth_12 | None / None | False | None | PARTIAL | failed gates: fec_interleaver_crc |
| T2-04 | 16QAM | unknown | 0.05423551766129037 | 100.0 / None | RS_255_223 / Block_32x32 | None / None | None | None | REJECTED | failed gates: modulation, demodulation, fec_interleaver_crc |
| T2-05 | QPSK | qpsk | 0.9979265862527592 | 0.0 / 0.03724615310063768 | None / None | None / None | None | None | PASS |  |
| T2-06 | 2FSK | 2fsk | 0.999998780208137 | 0.0 / None | Conv_R12_K3 / None | None / None | None | None | FAIL | 2FSK demodulation requires explicit (bit-0, bit-1) tone_frequencies_hz. |
| T2-07 | QPSK | 8psk | 0.9999997812165518 | 250.0 / 249.99903077507028 | Concat_RS223_Conv7 / Block_16x16 | None / None | False | None | FAIL | failed gates: modulation, fec_interleaver_crc |
| T2-08 | QPSK | 8psk | 0.9999991017281685 | 300.0 / 298.86634467124617 | Conv_R12_K7 / Diag_16x16_S1 | None / None | False | None | FAIL | failed gates: modulation, fec_interleaver_crc |
| T2-09 | Noise-only | 16qam | 0.9999999999999998 | 0.0 / None | None / None | None / None | None | None | FAIL | failed gates: modulation |
| T2-10 | OFDM | 16qam | 0.9999999999198896 | 0.0 / None | None / None | None / None | None | None | FAIL | failed gates: modulation |

## Aggregate counts

- Scenarios attempted: 10
- PASS: 1
- PARTIAL: 1
- FAIL: 7
- REJECTED: 1
- BLOCKED: 0
- ERROR: 0
- OOD_PASS: 0
- OOD_FAIL: 0
- In-scope successful: 1
- In-scope failed: 9
- OOD successful: 0
- OOD failed: 0

## Optional LDPC extension

T2-11-LDPC: BLOCKED. No independent verified encoder was available; no project LDPC generation code was used.
