# T2-05 phase status verification

The original frozen capture was re-analyzed without payload contents, CRC, expected payload length, or FEC. Current code reports the unresolved four-way QPSK orientation as partial. The Cycle 1 record is preserved unchanged.

{
  "label": "Cycle 2 T2-05 status verification \u2014 frozen capture reused",
  "capture": "reports/tier2/baseline/20260930_201445_269/T2-05/capture.wav",
  "analyzer_inputs": {
    "treat_stereo_as_iq": true,
    "run_fec": false,
    "crc_present": false,
    "expected_payload_length": null,
    "payload_contents_supplied": false
  },
  "current_status": "partial",
  "classification": "qpsk",
  "phase_resolution": "unresolved",
  "phase_candidate_count": 4,
  "fec_status": "not_requested",
  "crc_pass": null,
  "historical_cycle1_status": "complete (preserved record; predates final status correction)",
  "passes_partial_orientation_requirement": true
}
