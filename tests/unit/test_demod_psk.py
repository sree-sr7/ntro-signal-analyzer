import numpy as np
import pytest

from core.common.enums import ModulationType
from core.dsp.demod_psk import decide_psk_symbols, demodulate_psk
from libs.modulation_library import bits_to_symbols
from tests.modulation_helpers import add_awgn, random_bits


@pytest.mark.parametrize(
    "modulation,width",
    [(ModulationType.BPSK, 1), (ModulationType.QPSK, 2), (ModulationType.PSK8, 3)],
)
def test_psk_demodulation_round_trips_bits(modulation, width):
    rng = np.random.default_rng(401 + width)
    bits = random_bits(2048 * width, rng)
    symbols = bits_to_symbols(bits, modulation)

    result = demodulate_psk(symbols, modulation)

    np.testing.assert_array_equal(result.bits, bits)
    assert result.modulation is modulation
    assert result.diagnostics["symbol_count"] == symbols.size
    assert result.confidence > 0.9


@pytest.mark.parametrize(
    "modulation,width",
    [(ModulationType.BPSK, 1), (ModulationType.QPSK, 2), (ModulationType.PSK8, 3)],
)
def test_psk_demodulation_removes_supplied_phase_and_tracks_small_drift(modulation, width):
    rng = np.random.default_rng(503 + width)
    bits = random_bits(1024 * width, rng)
    symbols = bits_to_symbols(bits, modulation)
    symbol_index = np.arange(symbols.size)
    phase = 0.17 + 1.0e-5 * symbol_index
    impaired = 1.9 * symbols * np.exp(1j * phase)

    result = demodulate_psk(impaired, modulation, phase_reference=0.17)

    np.testing.assert_array_equal(result.bits, bits)


@pytest.mark.parametrize(
    "modulation,width,snr_db,max_ber",
    [
        (ModulationType.BPSK, 1, 8.0, 0.01),
        (ModulationType.QPSK, 2, 10.0, 0.015),
        (ModulationType.PSK8, 3, 18.0, 0.02),
    ],
)
def test_psk_awgn_bit_error_rate_is_bounded(modulation, width, snr_db, max_ber):
    rng = np.random.default_rng(607 + width)
    bits = random_bits(4096 * width, rng)
    symbols = bits_to_symbols(bits, modulation)
    received = add_awgn(symbols, snr_db, rng)

    recovered = demodulate_psk(received, modulation).bits
    ber = float(np.mean(recovered != bits))

    assert ber < max_ber, f"{modulation.value} BER={ber:.6f} at {snr_db:.1f} dB SNR"


def test_psk_symbol_decisions_report_short_and_silent_inputs():
    assert decide_psk_symbols(np.ones(3), ModulationType.BPSK).warnings
    silent = decide_psk_symbols(np.zeros(16), ModulationType.QPSK)
    assert silent.symbol_count == 0
    assert silent.warnings


def test_psk_rejects_non_psk_modulation():
    with pytest.raises(ValueError, match="PSK"):
        demodulate_psk(np.ones(32), ModulationType.QAM16)
