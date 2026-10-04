"""Oversampled, RRC-shaped PSK/QAM must classify correctly with the production model.

Regression for the Gardner-loop acquisition transient: at >= 6 samples/symbol
the loop's unsettled early output used to reach the classifier and make QPSK
look like 16-QAM (reported "complete" with confidence ~1.0).
"""

import numpy as np
import pytest
from scipy.signal import fftconvolve

from core.common.enums import ModulationType
from core.common.models import IQConfig
from core.dsp.pulse_shaping import root_raised_cosine
from core.pipeline.analyzer import Analyzer
from libs.modulation_library import bits_to_symbols

_WIDTH = {
    ModulationType.BPSK: 1,
    ModulationType.QPSK: 2,
    ModulationType.PSK8: 3,
    ModulationType.QAM16: 4,
}


def _shaped_capture(modulation, symbols, sps, snr_db, seed):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, symbols * _WIDTH[modulation], dtype=np.uint8)
    up = np.zeros(symbols * sps, dtype=np.complex128)
    up[::sps] = bits_to_symbols(bits, modulation)
    x = fftconvolve(up, root_raised_cosine(0.35, sps, 8), mode="same")
    power = float(np.mean(np.abs(x) ** 2))
    noise = np.sqrt(power / 10 ** (snr_db / 10) / 2) * (
        rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
    )
    return (x + noise).astype(np.complex64)


@pytest.mark.parametrize("sps", [4, 8, 16])
@pytest.mark.parametrize(
    "modulation",
    [ModulationType.BPSK, ModulationType.QPSK, ModulationType.PSK8, ModulationType.QAM16],
)
def test_oversampled_shaped_signal_is_classified_correctly(tmp_path, modulation, sps):
    capture = _shaped_capture(modulation, 3000, sps, 18.0, seed=sps * 10 + _WIDTH[modulation])
    path = tmp_path / "shaped.iq"
    capture.tofile(path)

    result = Analyzer().analyze_file(
        path,
        iq_config=IQConfig(sample_rate=1_000_000.0),
        samples_per_symbol=float(sps),
        matched_filter_rolloff=0.35,
        recover_symbol_timing=True,
        run_fec=False,
    )

    assert result.classification["modulation"] == modulation.value
    preprocessing = result.synchronization["preprocessing"]
    assert preprocessing["timing_recovery_applied"] is True
    assert preprocessing["classifier_timing_transient_discarded_samples"] > 0
