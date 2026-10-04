import numpy as np

from core.bitstream.crc import append_crc16
from core.fec import trial_engine
from core.fec.convolutional import encode_convolutional
from core.fec.trial_engine import run_fec_trials


def test_expected_frame_length_prunes_impossible_codes_before_viterbi(monkeypatch):
    rng = np.random.default_rng(5512)
    payload = rng.integers(0, 2, size=100, dtype=np.uint8)
    frame = append_crc16(payload)
    transmitted = encode_convolutional(frame, "Conv_R12_K7")
    original_decode = trial_engine.decode_viterbi
    decoded_codes = []

    def recording_decode(bits, fec_type, *, terminated):
        decoded_codes.append(fec_type)
        return original_decode(bits, fec_type, terminated=terminated)

    monkeypatch.setattr(trial_engine, "decode_viterbi", recording_decode)
    search = run_fec_trials(
        transmitted,
        expected_payload_length=payload.size,
        crc_present=True,
    )

    true_candidate = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == "Conv_R12_K7" and trial.interleaver_type is None
    )
    assert true_candidate.accepted and true_candidate.crc_pass is True
    assert set(decoded_codes) == {"Conv_R12_K7"}
    assert search.diagnostics["total_candidate_count"] == 89
    assert search.diagnostics["candidate_count_after_structural_pruning"] == 5
    assert search.diagnostics["structurally_pruned_candidate_count"] == 84
