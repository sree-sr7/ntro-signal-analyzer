import numpy as np
import pytest

from core.common.enums import ModulationType
from core.dsp.demod_qam import demodulate_qam
from libs.modulation_library import bits_to_symbols
from tests.modulation_helpers import add_awgn, random_bits


def test_16qam_demodulation_round_trips_mapping_and_gain_phase():
    rng = np.random.default_rng(701)
    bits = random_bits(4096, rng)
    symbols = bits_to_symbols(bits, ModulationType.QAM16)
    impaired = 3.4 * symbols * np.exp(0.29j)

    result = demodulate_qam(impaired, phase_reference=0.29)

    np.testing.assert_array_equal(result.bits, bits)
    assert result.symbol_indices.size == bits.size // 4
    assert result.diagnostics["amplitude_scale_applied"] is True


def test_16qam_awgn_bit_error_rate_is_bounded():
    rng = np.random.default_rng(809)
    bits = random_bits(8000, rng)
    symbols = bits_to_symbols(bits, ModulationType.QAM16)
    received = add_awgn(symbols, 20.0, rng)

    result = demodulate_qam(received)

    ber = float(np.mean(result.bits != bits))
    assert ber < 0.02, f"16-QAM BER={ber:.6f} at 20 dB SNR"


def test_16qam_demodulation_reports_short_or_silent_inputs():
    short = demodulate_qam(np.ones(3, dtype=np.complex64))
    silent = demodulate_qam(np.zeros(32, dtype=np.complex64))

    assert short.warnings and short.bits.size == 0
    assert silent.warnings and silent.bits.size == 0


def test_16qam_rejects_other_modulations():
    with pytest.raises(ValueError, match="16QAM"):
        demodulate_qam(np.ones(32), ModulationType.QPSK)
