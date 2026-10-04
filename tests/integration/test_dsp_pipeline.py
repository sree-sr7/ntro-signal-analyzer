import numpy as np
import pytest

from core.common.models import (
    CFOResult,
    CarrierRecoveryResult,
    DetectionResult,
    SignalData,
    SignalMetadata,
    SpectralResult,
    TimingResult,
)
from core.common.validation import validate_signal_data
from core.dsp.energy_detector import detect_activity
from core.dsp.spectral import analyze_spectrum
from core.sync.carrier_recovery import recover_carrier
from core.sync.cfo_estimator import estimate_cfo
from core.sync.timing_recovery import recover_timing
from tests.signal_helpers import add_awgn, fractional_delay, psk_symbols, psk_waveform


def test_typed_qpsk_synchronization_pipeline_estimates_injected_impairments():
    rng = np.random.default_rng(1001)
    sample_rate = 48_000.0
    symbol_rate = 6_000.0
    samples_per_symbol = int(sample_rate / symbol_rate)
    cfo_hz = 180.0
    phase_rad = 0.55
    timing_delay = 1.6
    symbols = psk_symbols(4, 2400, rng)
    clean = psk_waveform(symbols, samples_per_symbol, rolloff=0.25, span=10)
    delayed = fractional_delay(clean, timing_delay)
    index = np.arange(delayed.size)
    impaired = delayed * np.exp(1j * (phase_rad + 2 * np.pi * cfo_hz * index / sample_rate))
    burst = add_awgn(impaired, 24.0, rng)
    # Silence on either side supplies a measurable background interval to the activity detector.
    guard = 24_000
    received = np.concatenate(
        (np.zeros(guard, dtype=np.complex128), burst, np.zeros(guard, dtype=np.complex128))
    )
    signal = SignalData(
        samples=received,
        metadata=SignalMetadata(sample_rate=sample_rate, sample_count=received.size),
    )
    assert validate_signal_data(signal).valid

    spectrum = analyze_spectrum(signal.samples, signal.metadata.sample_rate, nfft=2048)
    detection = detect_activity(signal.samples, window_size=256, threshold_db=5.0)
    assert isinstance(spectrum, SpectralResult)
    assert isinstance(detection, DetectionResult) and detection.detected
    assert detection.start_index is not None and detection.end_index is not None
    detected = signal.samples[detection.start_index:detection.end_index]
    coarse = estimate_cfo(detected, sample_rate, modulation_order=4)
    assert isinstance(coarse, CFOResult) and coarse.estimated_offset_hz is not None
    # CFO tolerance accounts for raised-cosine transitions in the oversampled fourth-power signal.
    assert coarse.estimated_offset_hz == pytest.approx(cfo_hz, abs=70.0)

    detected_index = np.arange(detected.size)
    corrected = detected * np.exp(-2j * np.pi * coarse.estimated_offset_hz * detected_index / sample_rate)
    timing = recover_timing(corrected, samples_per_symbol, loop_bandwidth=0.008)
    assert isinstance(timing, TimingResult)
    assert timing.synchronized_samples.size > 1000
    recovered = recover_carrier(timing.synchronized_samples, symbol_rate, modulation_order=4)
    assert isinstance(recovered, CarrierRecoveryResult)
    assert recovered.frequency_offset is not None
    assert recovered.phase_offset is not None
    # Carrier recovery operates on the symbol-rate stream; this checks the residual estimate, not demodulation.
    assert abs(recovered.frequency_offset) < 60.0
    assert recovered.confidence > 0.15
