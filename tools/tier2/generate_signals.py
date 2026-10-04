"""Independent NumPy/Komm signal generator for the Tier-2 baseline.

This module deliberately imports no project code, tests, training data, model
artifacts, reedsolo, or project generator. Coding and interleaving mappings are
implemented locally from separately reviewed definitions. Only NumPy and the
external Komm convolutional-code implementation are used as dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import komm
import numpy as np


GENERATOR_VERSION = "tier2-independent-numpy-komm-1.0.0"


def _pack_bits(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").tobytes()


def _crc16_bits(bits: np.ndarray) -> np.ndarray:
    crc = 0xFFFF
    for bit in np.asarray(bits, dtype=np.uint8):
        feedback = ((crc >> 15) & 1) ^ int(bit)
        crc = (crc << 1) & 0xFFFF
        if feedback:
            crc ^= 0x1021
    value = crc
    return np.unpackbits(np.frombuffer(value.to_bytes(2, "big"), dtype=np.uint8), bitorder="big")


def _append_crc(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    return np.concatenate((values, _crc16_bits(values))).astype(np.uint8, copy=False)


def _gf_multiply(left: int, right: int) -> int:
    result = 0
    a, b = int(left), int(right)
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= 0x11D
    return result


def _gf_power(value: int, exponent: int) -> int:
    out = 1
    for _ in range(exponent):
        out = _gf_multiply(out, value)
    return out


def _rs_generator(parity_symbols: int) -> list[int]:
    polynomial = [1]
    for root_power in range(parity_symbols):
        root = _gf_power(2, root_power)
        updated = [0] * (len(polynomial) + 1)
        for index, coefficient in enumerate(polynomial):
            updated[index] ^= coefficient
            updated[index + 1] ^= _gf_multiply(coefficient, root)
        polynomial = updated
    return polynomial


def _rs_encode(data: bytes, parity_symbols: int = 32) -> bytes:
    """Systematic GF(256) RS encoder, alpha=2, primitive 0x11d, fcr=0."""
    if len(data) != 223 or parity_symbols != 32:
        raise ValueError("independent RS path is explicitly RS(255,223)")
    generator = _rs_generator(parity_symbols)
    work = list(data) + [0] * parity_symbols
    for position in range(len(data)):
        coefficient = work[position]
        if coefficient:
            for offset, term in enumerate(generator):
                work[position + offset] ^= _gf_multiply(term, coefficient)
    return data + bytes(work[-parity_symbols:])


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    values = np.asarray(bits, dtype=np.uint8)
    if values.size % 8:
        raise ValueError("byte conversion requires a multiple of eight bits")
    return _pack_bits(values)


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big").astype(np.uint8)


def _conv_encode(bits: np.ndarray, constraint_length: int) -> np.ndarray:
    feedforward = {
        3: (0o7, 0o5),
        5: (0o23, 0o35),
        7: (0o171, 0o133),
    }[constraint_length]
    code = komm.ConvolutionalCode(feedforward_polynomials=[feedforward])
    source = np.concatenate((np.asarray(bits, dtype=np.uint8), np.zeros(constraint_length - 1, dtype=np.uint8)))
    return np.asarray(code.encode(source), dtype=np.uint8).reshape(-1)


def _conv_interleave(bits: np.ndarray, depth: int, branches: int = 4) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    banks = [values[branch::branches].copy() for branch in range(branches)]
    for branch, bank in enumerate(banks):
        if bank.size:
            banks[branch] = np.roll(bank, (branch * depth) % bank.size)
    output = np.empty(values.size, dtype=np.uint8)
    for index in range(values.size):
        output[index] = banks[index % branches][index // branches]
    return output


def _block_interleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    size = rows * cols
    if not values.size or values.size % size:
        raise ValueError(f"bit length {values.size} is not divisible by {size}")
    return values.reshape(-1, rows, cols).transpose(0, 2, 1).reshape(-1).copy()


def _diagonal_interleave(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    if not values.size or values.size % 256:
        raise ValueError("diagonal frame must contain complete 256-bit blocks")
    matrices = values.reshape(-1, 16, 16)
    output = np.empty_like(matrices)
    for diagonal in range(16):
        for column in range(16):
            output[:, diagonal, column] = matrices[:, (diagonal + column) % 16, column]
    return output.reshape(-1)


def _gray_qpsk(bit_groups: np.ndarray) -> np.ndarray:
    # Gray phase sequence: 00, 01, 11, 10 beginning at pi/4.
    indices_by_label = {0b00: 0, 0b01: 1, 0b11: 2, 0b10: 3}
    phase_indices = np.array([indices_by_label[int(a) * 2 + int(b)] for a, b in bit_groups], dtype=np.int64)
    return np.exp(1j * (np.pi / 4 + phase_indices * np.pi / 2))


def _gray_8psk(bit_groups: np.ndarray) -> np.ndarray:
    labels = ("000", "001", "011", "010", "110", "111", "101", "100")
    lookup = {int(label, 2): index for index, label in enumerate(labels)}
    indices = np.array([lookup[int("".join(str(int(bit)) for bit in group), 2)] for group in bit_groups])
    return np.exp(1j * indices * (2 * np.pi / 8))


def _map_symbols(bits: np.ndarray, modulation: str) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    widths = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4}
    width = widths[modulation]
    if values.size % width:
        raise ValueError(f"{modulation} requires a multiple of {width} input bits")
    groups = values.reshape(-1, width)
    if modulation == "BPSK":
        return (1.0 - 2.0 * groups[:, 0]).astype(np.complex128)
    if modulation == "QPSK":
        return _gray_qpsk(groups)
    if modulation == "8PSK":
        return _gray_8psk(groups)
    # Two-bit Gray axis map: 00 -> -3, 01 -> -1, 10 -> +3, 11 -> +1.
    axis = np.array([-3.0, -1.0, 3.0, 1.0])
    i = 2 * groups[:, 0] + groups[:, 1]
    q = 2 * groups[:, 2] + groups[:, 3]
    return (axis[i] + 1j * axis[q]) / np.sqrt(10.0)


def _apply_channel(
    waveform: np.ndarray,
    *,
    sample_rate: float,
    snr_db: float | None,
    cfo_hz: float,
    phase_rad: float,
    gain: float,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(waveform, dtype=np.complex128) * float(gain)
    index = np.arange(values.size, dtype=np.float64)
    values = values * np.exp(1j * (phase_rad + 2.0 * np.pi * cfo_hz * index / sample_rate))
    if snr_db is not None:
        power = float(np.mean(np.abs(values) ** 2))
        noise_power = power / (10.0 ** (float(snr_db) / 10.0))
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal(values.size) + 1j * rng.standard_normal(values.size)
        )
        values = values + noise
    return values.astype(np.complex64)


def _make_fsk(bits: np.ndarray, fs: float, sps: int, tones: tuple[float, float]) -> np.ndarray:
    samples_per_bit = np.arange(sps, dtype=np.float64)
    out = np.empty(bits.size * sps, dtype=np.complex128)
    phase = 0.0
    cursor = 0
    for bit in bits:
        frequency = tones[int(bit)]
        increments = 2.0 * np.pi * frequency / fs
        phases = phase + increments * samples_per_bit
        out[cursor:cursor + sps] = np.exp(1j * phases)
        phase = float((phase + increments * sps) % (2.0 * np.pi))
        cursor += sps
    return out


def _make_ofdm(rng: np.random.Generator, *, nfft: int, cp: int, active_count: int, symbols: int) -> np.ndarray:
    if active_count > nfft - 2:
        raise ValueError("active subcarriers exceed available non-DC bins")
    positive_count = active_count // 2
    active = np.r_[np.arange(1, positive_count + 1), np.arange(nfft - positive_count, nfft)]
    out: list[np.ndarray] = []
    qpsk_bits = rng.integers(0, 2, size=(symbols, active_count, 2), dtype=np.uint8).reshape(-1, 2)
    qpsk = _gray_qpsk(qpsk_bits).reshape(symbols, active_count)
    for symbol_index in range(symbols):
        bins = np.zeros(nfft, dtype=np.complex128)
        bins[active] = qpsk[symbol_index]
        time = np.fft.ifft(bins) * np.sqrt(nfft)
        out.append(np.concatenate((time[-cp:], time)))
    return np.concatenate(out)


def generate_scenario(scenario: dict[str, Any], sample_rates: dict[str, float], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_id = str(scenario["scenario_id"])
    rng = np.random.Generator(np.random.PCG64(int(scenario["seed"])))
    modulation = str(scenario["modulation"])
    sample_rate = float(sample_rates[scenario["sample_rate_key"]])
    phase = float(rng.uniform(-np.pi, np.pi))
    gain = float(rng.uniform(0.65, 1.45))
    payload_bits = np.empty(0, dtype=np.uint8)
    crc_enabled = scenario.get("fec") not in ("None", "Noise-only")
    pre_modulation_bits = np.empty(0, dtype=np.uint8)
    expected_encoded_bits = 0

    if scenario_id == "T2-09":
        base = (rng.standard_normal(int(scenario["sample_count"])) + 1j * rng.standard_normal(int(scenario["sample_count"]))) / np.sqrt(2.0)
        received = _apply_channel(base, sample_rate=sample_rate, snr_db=None, cfo_hz=0.0, phase_rad=phase, gain=gain, rng=rng)
        fmt = "complex64_le_iq"
    elif scenario_id == "T2-10":
        base = _make_ofdm(rng, nfft=int(scenario["nfft"]), cp=int(scenario["cyclic_prefix"]), active_count=int(scenario["active_subcarriers"]), symbols=int(scenario["ofdm_symbols"]))
        received = _apply_channel(base, sample_rate=sample_rate, snr_db=float(scenario["snr_db"]), cfo_hz=float(scenario["cfo_hz"]), phase_rad=phase, gain=gain, rng=rng)
        fmt = "complex64_le_iq"
    elif scenario["fec"] == "RS_255_223":
        payload_bits = rng.integers(0, 2, int(scenario["payload_bytes"]) * 8, dtype=np.uint8)
        payload = _bits_to_bytes(payload_bits)
        with_crc = payload + _pack_bits(_crc16_bits(payload_bits))
        coded_words = []
        for offset in range(0, len(with_crc), 223):
            word = with_crc[offset:offset + 223]
            if len(word) != 223:
                raise ValueError("RS payload+CRC did not fill whole RS data words")
            coded_words.append(_bytes_to_bits(_rs_encode(word)))
        coded = np.concatenate(coded_words)
        pre_modulation_bits = _block_interleave(coded, 32, 32)
        expected_encoded_bits = int(pre_modulation_bits.size)
        symbols = _map_symbols(pre_modulation_bits, modulation)
        received = _apply_channel(symbols, sample_rate=sample_rate, snr_db=float(scenario["snr_db"]), cfo_hz=float(scenario["cfo_hz"]), phase_rad=phase, gain=gain, rng=rng)
        fmt = "complex64_le_iq"
    elif scenario["fec"] == "Concat_RS223_Conv7":
        payload_bits = rng.integers(0, 2, int(scenario["payload_bytes"]) * 8, dtype=np.uint8)
        payload = _bits_to_bytes(payload_bits)
        with_crc = payload + _pack_bits(_crc16_bits(payload_bits))
        coded_frames: list[np.ndarray] = []
        for offset in range(0, len(with_crc), 223):
            data_word = with_crc[offset:offset + 223]
            if len(data_word) != 223:
                raise ValueError("concatenated payload+CRC did not fill whole RS data words")
            coded_frames.append(_conv_encode(_bytes_to_bits(_rs_encode(data_word)), 7))
        coded = np.concatenate(coded_frames)
        pre_modulation_bits = _block_interleave(coded, 16, 16)
        expected_encoded_bits = int(pre_modulation_bits.size)
        symbols = _map_symbols(pre_modulation_bits, modulation)
        received = _apply_channel(symbols, sample_rate=sample_rate, snr_db=float(scenario["snr_db"]), cfo_hz=float(scenario["cfo_hz"]), phase_rad=phase, gain=gain, rng=rng)
        fmt = "complex64_le_iq"
    elif scenario["fec"] == "None" and modulation == "QPSK":
        payload_bits = rng.integers(0, 2, int(scenario["payload_bits"]), dtype=np.uint8)
        symbols = _map_symbols(payload_bits, modulation)
        expected_encoded_bits = int(payload_bits.size)
        pre_modulation_bits = payload_bits.copy()
        received = _apply_channel(symbols, sample_rate=sample_rate, snr_db=float(scenario["snr_db"]), cfo_hz=float(scenario["cfo_hz"]), phase_rad=phase, gain=gain, rng=rng)
        fmt = "stereo_wav_iq" if scenario.get("file_format") == "stereo_wav_iq" else "complex64_le_iq"
    else:
        payload_bits = rng.integers(0, 2, int(scenario["payload_bits"]), dtype=np.uint8)
        if crc_enabled:
            frame_bits = _append_crc(payload_bits)
        else:
            frame_bits = payload_bits.copy()
        if scenario["fec"].startswith("Conv_R12_K"):
            constraint = int(scenario["fec"].rsplit("K", 1)[1])
            coded = _conv_encode(frame_bits, constraint)
            expected_encoded_bits = int(coded.size)
            if scenario["interleaver"].startswith("Block_"):
                rows, cols = (int(part) for part in scenario["interleaver"].split("_")[1].split("x"))
                pre_modulation_bits = _block_interleave(coded, rows, cols)
            elif scenario["interleaver"].startswith("Convolutional_Depth_"):
                depth = int(scenario["interleaver"].rsplit("_", 1)[1])
                pre_modulation_bits = _conv_interleave(coded, depth)
            elif scenario["interleaver"] == "Diag_16x16_S1":
                pre_modulation_bits = _diagonal_interleave(coded)
            else:
                pre_modulation_bits = coded
            if pre_modulation_bits.size != expected_encoded_bits:
                raise RuntimeError("interleaver changed coded frame length")
            if modulation == "2FSK":
                tones = tuple(float(value) for value in scenario["tone_frequencies_hz"])
                base = _make_fsk(pre_modulation_bits, sample_rate, int(scenario["samples_per_symbol"]), tones)
            else:
                base = _map_symbols(pre_modulation_bits, modulation)
            received = _apply_channel(base, sample_rate=sample_rate, snr_db=float(scenario["snr_db"]), cfo_hz=float(scenario["cfo_hz"]), phase_rad=phase, gain=gain, rng=rng)
            fmt = "complex64_le_iq"
        else:
            raise ValueError(f"No independent generation recipe for {scenario_id}")

    if fmt == "stereo_wav_iq":
        from scipy.io import wavfile

        # Quantization to signed PCM16 is explicitly recorded; preserve I then Q channel order.
        interleaved = np.column_stack((received.real, received.imag))
        pcm = np.rint(np.clip(interleaved, -1.0, 32767.0 / 32768.0) * 32768.0).astype("<i2")
        capture_path = output_dir / "capture.wav"
        wavfile.write(str(capture_path), int(sample_rate), pcm)
        actual_samples = pcm[:, 0].astype(np.float32) / 32768.0 + 1j * (pcm[:, 1].astype(np.float32) / 32768.0)
    else:
        capture_path = output_dir / "capture.iq"
        received.astype("<c8", copy=False).tofile(capture_path)
        actual_samples = received

    payload_path: str | None = None
    if payload_bits.size:
        payload_path = "payload_bits.bin"
        (output_dir / payload_path).write_bytes(_pack_bits(payload_bits))
    gt = {
        "scenario_id": scenario_id,
        "modulation": modulation,
        "ood": bool(scenario.get("ood", False)),
        "sample_rate_hz": sample_rate,
        "samples_per_symbol": scenario.get("samples_per_symbol", 1 if modulation in {"BPSK", "QPSK", "8PSK", "16QAM"} else None),
        "sample_count": int(actual_samples.size),
        "duration_seconds": float(actual_samples.size / sample_rate),
        "snr_db": scenario.get("snr_db"),
        "gain": gain,
        "carrier_offset_hz": float(scenario.get("cfo_hz", 0.0)),
        "initial_phase_rad": phase,
        "timing_offset_samples": 0,
        "fec": scenario.get("fec"),
        "interleaver": scenario.get("interleaver"),
        "tone_frequencies_hz": scenario.get("tone_frequencies_hz"),
        "crc": "CRC-16/CCITT-FALSE" if crc_enabled else None,
        "payload_bit_count": int(payload_bits.size),
        "payload_sha256": hashlib.sha256(_pack_bits(payload_bits)).hexdigest() if payload_bits.size else None,
        "payload_bits_file": payload_path,
        "payload_packed_bitorder": "big",
        "encoded_modulation_input_bit_count": expected_encoded_bits or int(pre_modulation_bits.size),
        "random_seed": int(scenario["seed"]),
        "generator": {
            "name": GENERATOR_VERSION,
            "language": "Python",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "komm_version": getattr(komm, "__version__", "unknown"),
            "configuration": scenario,
            "import_allowlist": ["stdlib", "numpy", "scipy.io.wavfile", "komm"],
            "fec_method": "Komm ConvolutionalCode for convolutional coding; local GF(256) RS encoder (alpha=2, primitive 0x11d, fcr=0); locally implemented CRC and interleaver mappings",
            "channel_method": "NumPy complex baseband mapping, explicit CFO/phase/gain, and AWGN; OFDM constructed with NumPy IFFT",
            "file_conversion": "stereo WAV is signed PCM16 with channel 0=I, channel 1=Q; raw IQ is little-endian complex64",
        },
        "file_format": fmt,
        "file_path": capture_path.name,
        "file_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        "unknown_to_analyzer": ["payload bits", "random seed", "transmitted modulation", "FEC identity", "interleaver identity", "gain", "initial phase", "injected SNR", "carrier offset"],
        "analyzer_configuration_allowed": {
            "raw_iq_sample_rate_hz": sample_rate if fmt == "complex64_le_iq" else None,
            "stereo_wav_as_iq": fmt == "stereo_wav_iq",
            "samples_per_symbol": scenario.get("samples_per_symbol"),
            "tone_frequencies_hz": scenario.get("tone_frequencies_hz"),
            "crc_present": bool(crc_enabled),
            "run_fec": scenario_id not in {"T2-05"},
        },
    }
    (output_dir / "ground_truth.json").write_text(json.dumps(gt, indent=2, sort_keys=True), encoding="utf-8")
    return gt


def load_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config" / "baseline_scenarios.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configuration = load_config(args.config)
    for scenario in configuration["scenarios"]:
        print(generate_scenario(scenario, configuration["sample_rates_hz"], args.output / scenario["scenario_id"]))
