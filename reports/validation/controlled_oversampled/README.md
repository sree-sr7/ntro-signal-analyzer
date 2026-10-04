# Controlled synthetic validation

This single synthetic oversampled QPSK run is kept separate from Tier-2 evidence. No retry or parameter tuning was performed.

- **label:** Controlled synthetic validation — separate from Tier-2 evidence
- **seed:** 2614702
- **waveform:** QPSK, RRC pulse shaping, 8 samples/symbol, 120 Hz CFO, 20 dB AWGN
- **modulation:** QPSK
- **samples per symbol:** 8
- **symbol count:** 1024
- **sample count:** 8192
- **sample rate hz:** 96000.0
- **fec:** Conv_R12_K9 terminated
- **interleaver:** Block_8x8
- **crc:** CRC-16/CCITT-FALSE
- **payload bits expected:** 1000
- **payload contents supplied to analyzer:** False
- **runtime seconds:** 0.2628209999820683
- **classifier result:** 8psk
- **classification correct:** False
- **timing recovery applied:** True
- **classifier transient discarded samples:** 307
- **demodulated bit count:** 3072
- **coded bits transmitted:** 2048
- **ber definition:** Descriptive common-prefix mismatch only; the value is not a valid modulation BER because the analyzer classified the QPSK signal as 8PSK and produced a different bit count.
- **ber comparison bits:** 2048
- **ber:** 0.509765625
- **fec result:** no_fec_candidate
- **interleaver result:** None
- **crc pass:** False
- **recovered payload bits:** 0
- **exact payload match:** False
- **artifact files:** ['controlled_qpsk_8sps.iq', 'analysis_result.json', 'recovered_payload_bits.bin']
- **interpretation:** Controlled synthetic E2E did not validate: target QPSK was classified as 8PSK, FEC/CRC were not accepted, and no payload was recovered. One capture only; no transient-discard or parameter tuning was performed.
