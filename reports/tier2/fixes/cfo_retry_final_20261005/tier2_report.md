# Tier-2 Independent Validation Baseline

Run: `20261005_cfo_retry_final`
Scenarios attempted: 10

No single accuracy percentage is reported; this is a fixed, small scenario set.

| Scenario | Ground truth | Predicted | Confidence | CFO in / estimated | FEC / interleaver expected | Accepted | CRC | Payload | Status | Failure reason |
|---|---|---|---:|---:|---|---|---|---|---|---|
| T2-01 | QPSK | qpsk | 0.9999997851739901 | 150.0 / 149.02104605428218 | Conv_R12_K5 / Block_8x8 | Conv_R12_K5 / Block_8x8 | True | True | PASS |  |
| T2-02 | BPSK | bpsk | 0.9999991400326638 | -200.0 / -199.2679758267934 | Conv_R12_K7 / None | Conv_R12_K7 / None | True | True | PASS |  |
| T2-03 | 8PSK | 8psk | 0.9999999485259768 | 400.0 / 400.97310813945046 | Conv_R12_K7 / Convolutional_Depth_12 | Conv_R12_K7 / Convolutional_Depth_12 | True | True | PASS |  |
| T2-04 | 16QAM | unknown | 0.05423551766129037 | 100.0 / None | RS_255_223 / Block_32x32 | None / None | None | None | REJECTED | failed gates: modulation, demodulation, fec_interleaver_crc |
| T2-05 | QPSK | qpsk | 0.9979252138844094 | 0.0 / 0.03724615310063768 | None / None | None / None | None | False | PARTIAL | failed gates: payload |
| T2-06 | 2FSK | 2fsk | 0.999998780208137 | 0.0 / None | Conv_R12_K3 / None | Conv_R12_K3 / None | True | True | PASS |  |
| T2-07 | QPSK | qpsk | 0.9999998619036202 | 250.0 / 249.99931827944692 | Concat_RS223_Conv7 / Block_16x16 | Concat_RS223_Conv7 / Block_16x16 | True | True | PASS |  |
| T2-08 | QPSK | qpsk | 0.9999997704345865 | 300.0 / 299.5300680900776 | Conv_R12_K7 / Diag_16x16_S1 | Conv_R12_K7 / Diag_16x16_S1 | True | True | PASS |  |
| T2-09 | Noise-only | unknown | None | 0.0 / None | None / None | None / None | None | None | OOD_PASS |  |
| T2-10 | OFDM | unknown | None | 0.0 / None | None / None | None / None | None | None | OOD_PASS |  |

## Aggregate counts

- Scenarios attempted: 10
- PASS: 6
- PARTIAL: 1
- FAIL: 0
- REJECTED: 1
- BLOCKED: 0
- ERROR: 0
- OOD_PASS: 2
- OOD_FAIL: 0
- In-scope successful: 6
- In-scope failed: 2
- OOD successful: 2
- OOD failed: 0

## Optional LDPC extension

T2-11-LDPC: BLOCKED. No independent verified encoder was available; no project LDPC generation code was used.
