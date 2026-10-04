import numpy as np

from core.fec.ldpc_decoder import LDPC_CONFIGS, encode_ldpc_frame
from core.fec.trial_engine import run_fec_trials


def test_trial_engine_runs_the_selected_ieee_ldpc_candidate_with_diagnostics():
    config = LDPC_CONFIGS["LDPC_648_R1_2"]
    rng = np.random.default_rng(82501)
    payload = rng.integers(0, 2, size=3 * config.K, dtype=np.uint8)
    coded = encode_ldpc_frame(payload, config)

    search = run_fec_trials(
        coded,
        expected_payload_length=3 * config.K,
        crc_present=False,
    )

    candidate = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == config.name and trial.interleaver_type is None
    )
    assert candidate.accepted
    assert candidate.codeword_count == 3
    assert candidate.codeword_length_bits == config.N
    assert candidate.successful_fec_decodes == 3
    assert candidate.failed_fec_decodes == 0
    assert candidate.diagnostics["ldpc_batch"]["fec_name"] == "LDPC"
    assert candidate.diagnostics["ldpc_batch"]["configuration"] == config.name
    assert candidate.diagnostics["ldpc_batch"]["decoder_algorithm"] == "normalized_min_sum_llr"
    assert candidate.diagnostics["ldpc_batch"]["syndrome_status"] == [True, True, True]
    np.testing.assert_array_equal(candidate.decoded_bits, payload)


def test_trial_engine_rejects_wrong_ldpc_configuration_by_exact_frame_structure():
    config = LDPC_CONFIGS["LDPC_648_R1_2"]
    wrong_config = LDPC_CONFIGS["LDPC_648_R2_3"]
    rng = np.random.default_rng(82502)
    payload = rng.integers(0, 2, size=config.K, dtype=np.uint8)
    coded = encode_ldpc_frame(payload, config)

    search = run_fec_trials(
        coded,
        expected_payload_length=config.K,
        crc_present=False,
    )

    candidate = next(
        trial for trial in search.ranked_trials
        if trial.fec_type == wrong_config.name and trial.interleaver_type is None
    )
    assert not candidate.frame_length_compatible
    assert not candidate.accepted
    assert any("produces" in warning for warning in candidate.warnings)
