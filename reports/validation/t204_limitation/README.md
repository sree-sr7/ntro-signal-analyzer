# T2-04 limitation verification

The frozen capture was analyzed without using its ground-truth CFO. The bounded signal-support gate passed; the classifier returned unknown because its class scores were ambiguous. Saved diagnostics that remove +100 Hz CFO are oracle-only and do not justify passing ground-truth CFO into production.

{
  "label": "T2-04 limitation verification \u2014 frozen capture reused",
  "capture": "reports/tier2/baseline/20260930_201445_269/T2-04/capture.iq",
  "analyzer_inputs": {
    "sample_rate_hz": 1000000.0,
    "ground_truth_cfo_supplied": false,
    "expected_payload_length": null,
    "crc_present": false,
    "run_fec": false
  },
  "runtime_seconds": 0.2931749999988824,
  "status": "rejected",
  "classification": "unknown",
  "classifier_status": "success",
  "confidence": 0.05423551766129037,
  "signal_support": {
    "status": "passed",
    "geometric_fit_score": 0.5780948681531969,
    "threshold": 0.35,
    "interpretation": "bounded geometric signal-support evidence, not a probability"
  },
  "coarse_cfo_estimate": null,
  "fec_status": "unavailable",
  "crc_pass": null,
  "limitation": "16QAM / RS(255,223) / Block_32x32 / 18 dB / +100 Hz ground truth remains unclassified by the production path. Saved diagnostics that remove +100 Hz CFO are oracle-only and do not justify passing ground-truth CFO into production.",
  "cfo_estimate": null,
  "classification_outcome": "unknown (top candidates are ambiguous and fail the configured acceptance margin)"
}
