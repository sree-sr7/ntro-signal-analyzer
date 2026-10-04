"""Deterministic local IQ source used by the GUI's offline demonstration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

from core.bitstream.crc import append_crc16
from core.common.enums import ModulationType, SampleDtype
from core.common.models import IQConfig
from core.fec.convolutional import CONVOLUTIONAL_CODE_SPECS, encode_convolutional
from core.interleaver.pseudorandom import (
    PseudoRandomInterleaver,
    PseudoRandomInterleaverConfig,
)
from core.pipeline.analyzer import AnalysisResult, Analyzer
from core.ml.classifier import OnnxModulationClassifier
from libs.modulation_library import bits_to_symbols


DEMO_SOURCE = "DEMO / SYNTHETIC"
DEMO_PAYLOAD_HEX = "26147D3AA55AC33CF00F966981"


@dataclass(frozen=True)
class DemoConfig:
    """Fixed, inspectable parameters for the local BPSK demonstration."""

    modulation: ModulationType = ModulationType.BPSK
    sample_rate_hz: float = 48_000.0
    samples_per_symbol: int = 1
    snr_db: float = 30.0
    gain: float = 1.0
    carrier_offset_hz: float = 0.0
    random_seed: int = 26147
    payload_hex: str = DEMO_PAYLOAD_HEX
    fec: str = "Conv_R12_K9"
    interleaver: str = "pseudo_random"
    interleaver_frame_length: int = 256
    interleaver_seed: int = 26147
    crc_enabled: bool = True

    def __post_init__(self) -> None:
        if self.modulation is not ModulationType.BPSK:
            raise ValueError("The deterministic demo currently supports BPSK only")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive and finite")
        if (
            not isinstance(self.samples_per_symbol, (int, np.integer))
            or isinstance(self.samples_per_symbol, (bool, np.bool_))
            or self.samples_per_symbol != 1
        ):
            raise ValueError("The demo uses symbol-rate IQ samples (samples_per_symbol=1)")
        if not np.isfinite(self.snr_db) or not -100.0 <= self.snr_db <= 100.0:
            raise ValueError("snr_db must be finite and between -100 and 100 dB")
        if not np.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("gain must be positive and finite")
        if (
            not np.isfinite(self.carrier_offset_hz)
            or abs(self.carrier_offset_hz) >= self.sample_rate_hz / 4.0
        ):
            raise ValueError("BPSK carrier offset must be finite and within ±sample_rate_hz/4")
        for label, seed in (("random_seed", self.random_seed), ("interleaver_seed", self.interleaver_seed)):
            if (
                not isinstance(seed, (int, np.integer))
                or isinstance(seed, (bool, np.bool_))
                or not 0 <= int(seed) < 2**64
            ):
                raise ValueError(f"{label} must be an integer in [0, 2**64 - 1]")
        try:
            payload = bytes.fromhex(self.payload_hex)
        except (TypeError, ValueError) as exc:
            raise ValueError("payload_hex must be a non-empty hexadecimal byte string") from exc
        if not payload:
            raise ValueError("payload_hex must contain at least one payload byte")
        if self.fec not in CONVOLUTIONAL_CODE_SPECS:
            raise ValueError(f"Unsupported demo convolutional FEC: {self.fec}")
        if self.interleaver != "pseudo_random":
            raise ValueError("The deterministic demo currently supports pseudo_random interleaving")
        interleaver_config = PseudoRandomInterleaverConfig(
            frame_length=self.interleaver_frame_length,
            seed=self.interleaver_seed,
        )
        if self.crc_enabled is not True:
            raise ValueError("The demo requires CRC-16 to verify the known payload")
        payload_bit_count = len(payload) * 8
        spec = CONVOLUTIONAL_CODE_SPECS[self.fec]
        coded_bit_count = 2 * (payload_bit_count + 16 + spec.termination_tail_bits)
        if coded_bit_count != interleaver_config.frame_length:
            raise ValueError(
                "Demo payload, CRC, FEC, and interleaver frame length must form exactly one frame "
                f"(coded bits={coded_bit_count}, frame length={interleaver_config.frame_length})"
            )

    @property
    def payload_bits(self) -> np.ndarray:
        """Return the fixed expected payload as MSB-first bits."""
        return np.unpackbits(np.frombuffer(bytes.fromhex(self.payload_hex), dtype=np.uint8)).copy()

    @property
    def payload_length_bits(self) -> int:
        return len(bytes.fromhex(self.payload_hex)) * 8


DEFAULT_DEMO_CONFIG = DemoConfig()


@dataclass(frozen=True)
class DemoSignal:
    """Generated symbol-rate complex samples and their known payload."""

    samples: np.ndarray
    expected_payload_bits: np.ndarray
    configuration: DemoConfig


def generate_demo_signal(configuration: DemoConfig = DEFAULT_DEMO_CONFIG) -> DemoSignal:
    """Generate one reproducible AWGN-impaired BPSK codeword."""
    if not isinstance(configuration, DemoConfig):
        raise TypeError("configuration must be a DemoConfig")
    expected_payload = configuration.payload_bits
    payload_with_crc = append_crc16(expected_payload)
    coded = encode_convolutional(payload_with_crc, configuration.fec, terminated=True)
    interleaver_configuration = PseudoRandomInterleaverConfig(
        frame_length=configuration.interleaver_frame_length,
        seed=configuration.interleaver_seed,
    )
    interleaved = PseudoRandomInterleaver(interleaver_configuration).interleave(coded)
    clean = bits_to_symbols(interleaved, configuration.modulation).astype(np.complex128)
    indices = np.arange(clean.size, dtype=np.float64)
    clean *= configuration.gain * np.exp(
        2j * np.pi * configuration.carrier_offset_hz * indices / configuration.sample_rate_hz
    )
    signal_power = float(np.mean(np.abs(clean) ** 2))
    noise_power = signal_power / (10.0 ** (configuration.snr_db / 10.0))
    rng = np.random.Generator(np.random.PCG64(int(configuration.random_seed)))
    noise = math.sqrt(noise_power / 2.0) * (
        rng.standard_normal(clean.size) + 1j * rng.standard_normal(clean.size)
    )
    samples = np.asarray(clean + noise, dtype=np.complex64)
    return DemoSignal(samples, expected_payload, configuration)


def _demo_metadata(signal: DemoSignal) -> dict[str, Any]:
    configuration = signal.configuration
    return {
        "source": DEMO_SOURCE,
        "note": "Deterministic local demonstration signal. Not an independent validation dataset.",
        "configuration": {
            "modulation": configuration.modulation.value,
            "sample_rate_hz": configuration.sample_rate_hz,
            "samples_per_symbol": configuration.samples_per_symbol,
            "snr_db": configuration.snr_db,
            "gain": configuration.gain,
            "carrier_offset_hz": configuration.carrier_offset_hz,
            "random_seed": int(configuration.random_seed),
            "payload_length_bits": configuration.payload_length_bits,
            "fec": configuration.fec,
            "interleaver": configuration.interleaver,
            "interleaver_frame_length": configuration.interleaver_frame_length,
            "interleaver_seed": int(configuration.interleaver_seed),
            "crc_enabled": configuration.crc_enabled,
        },
        "expected_payload_hex": configuration.payload_hex.upper(),
        "expected_payload_bits": signal.expected_payload_bits.astype(int).tolist(),
        "payload_recovery": "not_available",
        "payload_match": None,
    }


def _verify_payload(result: AnalysisResult, signal: DemoSignal) -> None:
    recovered = result.visualization.get("fec_recovered_bits")
    if recovered is None:
        recovered = result.visualization.get("crc_verified_bits")
    if recovered is None:
        return
    recovered_bits = np.asarray(recovered, dtype=np.uint8).reshape(-1)
    matches = np.array_equal(recovered_bits, signal.expected_payload_bits)
    result.demo["payload_recovery"] = "matched" if matches else "mismatch"
    result.demo["payload_match"] = bool(matches)
    result.demo["recovered_payload_hex"] = np.packbits(recovered_bits).tobytes().hex().upper()
    if not matches:
        result.warnings.append(
            "Demo payload mismatch: recovered payload does not match the known expected payload."
        )
        if result.status == "complete":
            result.status = "partial"


def analyze_demo(
    analyzer: Analyzer,
    *,
    configuration: DemoConfig = DEFAULT_DEMO_CONFIG,
    progress_callback: Callable[[str, int], None] | None = None,
) -> AnalysisResult:
    """Generate a temporary IQ file and send it through ``Analyzer.analyze_file``."""
    if not isinstance(analyzer, Analyzer):
        raise TypeError("analyzer must be an Analyzer instance")
    report = progress_callback or (lambda _message, _percent: None)
    report("Generating demo signal...", 1)
    signal = generate_demo_signal(configuration)
    classifier = analyzer.classifier
    with tempfile.TemporaryDirectory(prefix="ntro-signal-demo-") as temporary_directory:
        iq_path = Path(temporary_directory) / "deterministic_demo.iq"
        signal.samples.tofile(iq_path)
        report("Loading generated IQ file...", 8)
        result = analyzer.analyze_file(
            iq_path,
            iq_config=IQConfig(
                dtype=SampleDtype.COMPLEX64,
                sample_rate=configuration.sample_rate_hz,
            ),
            samples_per_symbol=float(configuration.samples_per_symbol),
            expected_payload_length=configuration.payload_length_bits,
            crc_present=configuration.crc_enabled,
            run_fec=True,
            progress_callback=progress_callback,
        )
    result.source = DEMO_SOURCE
    result.filename = "deterministic_demo.iq"
    result.input_path = DEMO_SOURCE
    result.file_metadata["source"] = DEMO_SOURCE
    result.demo = _demo_metadata(signal)
    result.demo.update({
        "classifier_wrapper": type(classifier).__name__,
        "production_classifier_wrapper_invoked": (
            isinstance(classifier, OnnxModulationClassifier)
            and result.classification.get("method") == "onnx"
        ),
        "production_onnx_session_loaded": (
            isinstance(classifier, OnnxModulationClassifier) and classifier.available
        ),
        "production_onnx_inference_succeeded": (
            result.classification.get("method") == "onnx"
            and result.classification.get("status") == "success"
        ),
    })
    _verify_payload(result, signal)
    return result
