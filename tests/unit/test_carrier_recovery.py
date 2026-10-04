import numpy as np
import pytest

from core.common.models import CarrierRecoveryResult
from core.sync.carrier_recovery import recover_carrier
from tests.signal_helpers import add_awgn, psk_symbols


@pytest.mark.parametrize("order,phase", [(2, 0.73), (4, -1.12)])
def test_carrier_recovery_removes_phase_and_residual_frequency(order, phase):
    rng = np.random.default_rng(901 + order)
    symbol_rate = 4800.0
    cfo_hz = 83.0
    symbols = psk_symbols(order, 1600, rng)
    index = np.arange(symbols.size)
    received = symbols * np.exp(1j * (phase + 2 * np.pi * cfo_hz * index / symbol_rate))
    received = add_awgn(received, 18.0, rng)
    result = recover_carrier(received, symbol_rate, order)

    assert isinstance(result, CarrierRecoveryResult)
    assert result.frequency_offset == pytest.approx(cfo_hz, abs=2.0)
    # M-th power resolves phase only modulo 2π/M; compare within that equivalence class.
    phase_step = 2.0 * np.pi / order
    phase_error = (result.phase_offset - phase + phase_step / 2) % phase_step - phase_step / 2
    assert abs(phase_error) < 0.04
    assert result.recovered_samples.size == received.size
    # RMS error includes the specified 18 dB AWGN and should stay near that scale.
    aligned = result.recovered_samples * np.exp(-1j * np.round(
        np.angle(np.mean(result.recovered_samples * np.conjugate(symbols))) / (2 * np.pi / order)
    ) * (2 * np.pi / order))
    assert np.sqrt(np.mean(np.abs(aligned - symbols) ** 2)) < 0.25


def test_carrier_recovery_rejects_noise_and_zero_signal():
    rng = np.random.default_rng(909)
    noise = rng.standard_normal(2048) + 1j * rng.standard_normal(2048)
    result = recover_carrier(noise, 8_000.0, 4)
    zero = recover_carrier(np.zeros(64), 8_000.0, 4)

    assert result.frequency_offset is None and result.warnings
    assert zero.phase_offset is None and zero.warnings
    with pytest.raises(ValueError, match="finite"):
        recover_carrier(np.array([1 + 0j, np.nan + 0j]), 8_000.0)


def test_qpsk_carrier_recovery_subtracts_constellation_phase_anchor():
    phase = 2.5516129220225663
    index = np.arange(512)
    symbols = np.exp(1j * (np.pi / 4.0 + index * np.pi / 2.0))
    received = symbols * np.exp(1j * phase)

    result = recover_carrier(
        received,
        1_000_000.0,
        4,
        constellation_phase_rad=np.pi / 4.0,
    )

    assert result.phase_offset is not None
    phase_step = np.pi / 2.0
    phase_error = (result.phase_offset - phase + phase_step / 2) % phase_step - phase_step / 2
    assert abs(phase_error) < 1e-10
    assert result.recovered_samples.size == received.size
