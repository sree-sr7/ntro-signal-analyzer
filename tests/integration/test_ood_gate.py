import numpy as np

from core.common.enums import SampleDtype
from core.common.models import IQConfig
from core.pipeline.analyzer import Analyzer
from libs.modulation_library import bits_to_symbols
from core.common.enums import ModulationType


def _write_iq(path, samples):
    np.asarray(samples, dtype=np.complex64).tofile(path)
    return IQConfig(dtype=SampleDtype.COMPLEX64, sample_rate=1_000_000.0)


def _ofdm_unknown(seed=10010):
    rng = np.random.default_rng(seed)
    nfft, cp, symbol_count = 64, 16, 24
    half_active = 26
    active = np.r_[np.arange(1, half_active + 1), np.arange(nfft - half_active, nfft)]
    symbols = []
    for _ in range(symbol_count):
        bins = np.zeros(nfft, dtype=np.complex128)
        bins[active] = np.exp(
            1j * (np.pi / 4 + rng.integers(0, 4, active.size) * np.pi / 2)
        )
        useful = np.fft.ifft(bins) * np.sqrt(nfft)
        symbols.append(np.r_[useful[-cp:], useful])
    return np.concatenate(symbols)


def _assert_explicit_ood_rejection(result):
    assert result.status == "rejected"
    assert result.classification["modulation"] == "unknown"
    assert result.classification["confidence"] is None
    assert result.classification["model_prediction_before_signal_gate"]["confidence"] > 0.9
    gate = result.classification["signal_detection_gate"]
    assert gate["decision"] == "unknown_ood"
    assert gate["stage"] == "signal_detection"
    assert "not a probability" in gate["score_interpretation"]
    assert result.classification["rejection"]["stage"] == "signal_detection"
    assert result.fec["status"] == "unavailable"
    assert not any(stage["name"] == "fec_interleaver_trials" for stage in result.stages)


def test_noise_only_is_rejected_before_fec(tmp_path):
    rng = np.random.default_rng(9009)
    noise = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)) / np.sqrt(2)
    path = tmp_path / "noise.iq"
    config = _write_iq(path, noise)

    result = Analyzer().analyze_file(path, iq_config=config)

    _assert_explicit_ood_rejection(result)


def test_strong_ofdm_unknown_is_rejected_before_fec(tmp_path):
    path = tmp_path / "ofdm.iq"
    config = _write_iq(path, _ofdm_unknown())

    result = Analyzer().analyze_file(path, iq_config=config)

    _assert_explicit_ood_rejection(result)


def test_supported_qpsk_signal_passes_signal_detection_gate(tmp_path):
    rng = np.random.default_rng(5010)
    bits = rng.integers(0, 2, 4096, dtype=np.uint8)
    symbols = bits_to_symbols(bits, ModulationType.QPSK)
    samples = symbols * np.exp(1j * 0.31)
    path = tmp_path / "qpsk.iq"
    config = _write_iq(path, samples)

    result = Analyzer().analyze_file(path, iq_config=config, run_fec=False)

    assert result.classification["signal_detection_gate"]["status"] == "passed"
    assert result.classification["signal_detection_gate"]["decision"] == "signal_candidate"
    assert result.classification["signal_detection_gate"]["stage"] == "signal_detection"
