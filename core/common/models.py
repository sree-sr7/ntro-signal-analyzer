from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import numpy as np
from .enums import (
    SignalFormat, IQLayout, Endianness, SampleDtype, LoadStatus,
    ModulationType, ClassifierStatus,
)

@dataclass
class IQConfig:
    """Explicit configuration for loading raw IQ files.
    
    Raw IQ binary files contain no metadata. The caller MUST provide
    enough information to interpret the binary data correctly.
    """
    dtype: SampleDtype = SampleDtype.COMPLEX64
    sample_rate: Optional[float] = None
    center_frequency: Optional[float] = None
    iq_layout: IQLayout = IQLayout.COMPLEX_NATIVE
    endianness: Endianness = Endianness.LITTLE
    header_offset_bytes: int = 0

@dataclass
class SignalMetadata:
    """Normalized metadata for a loaded signal.
    
    Fields that could not be determined are set to None.
    The 'warnings' list records any assumptions or heuristics used.
    """
    source_path: Optional[Path] = None
    format: SignalFormat = SignalFormat.UNKNOWN
    sample_rate: Optional[float] = None
    center_frequency: Optional[float] = None
    sample_count: int = 0
    duration_seconds: Optional[float] = None
    dtype: Optional[SampleDtype] = None
    iq_layout: IQLayout = IQLayout.UNKNOWN
    endianness: Endianness = Endianness.NATIVE
    num_channels: int = 1
    bits_per_sample: Optional[int] = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_sample_rate(self) -> bool:
        return self.sample_rate is not None and self.sample_rate > 0

    @property
    def has_center_frequency(self) -> bool:
        return self.center_frequency is not None

@dataclass
class SignalData:
    """Container for loaded signal samples and their metadata."""
    samples: np.ndarray  # Always complex64 or complex128 after loading
    metadata: SignalMetadata

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def is_complex(self) -> bool:
        return np.iscomplexobj(self.samples)

@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: 'ValidationResult') -> None:
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

@dataclass
class LoadResult:
    """Result of attempting to load a signal file."""
    status: LoadStatus
    signal: Optional[SignalData] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status in (LoadStatus.SUCCESS, LoadStatus.PARTIAL)

    @staticmethod
    def ok(signal: SignalData, warnings: Optional[list[str]] = None) -> 'LoadResult':
        status = LoadStatus.PARTIAL if warnings else LoadStatus.SUCCESS
        return LoadResult(
            status=status,
            signal=signal,
            warnings=warnings or [],
        )

    @staticmethod
    def fail(errors: list[str], warnings: Optional[list[str]] = None) -> 'LoadResult':
        return LoadResult(
            status=LoadStatus.FAILED,
            errors=errors,
            warnings=warnings or [],
        )


@dataclass
class SpectralResult:
    """Welch power spectral density and derived spectral measurements.

    ``power`` is a linear power spectral density (sample-units squared per Hz).
    ``occupied_bandwidth`` is the width containing 99% of integrated PSD power.
    """
    frequencies: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    power: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    sample_rate: Optional[float] = None
    peak_frequency: Optional[float] = None
    occupied_bandwidth: Optional[float] = None
    noise_floor: Optional[float] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class DetectionResult:
    """Energy detector result; indices use a half-open ``[start, end)`` range."""
    detected: bool
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    estimated_snr: Optional[float] = None
    noise_floor: Optional[float] = None
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    activity_regions: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class SymbolRateResult:
    """Bounded symbol-rate estimate and ranked alternatives."""
    estimated_rate: Optional[float] = None
    confidence: float = 0.0
    method: str = "unavailable"
    candidates: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CFOResult:
    """Coarse carrier-frequency offset estimate in Hz."""
    estimated_offset_hz: Optional[float] = None
    confidence: float = 0.0
    method: str = "unavailable"
    warnings: list[str] = field(default_factory=list)


@dataclass
class TimingResult:
    """Gardner timing-recovery output and timing diagnostics."""
    samples_per_symbol: float
    timing_offset: Optional[float] = None
    confidence: float = 0.0
    synchronized_samples: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.complex64))
    warnings: list[str] = field(default_factory=list)


@dataclass
class CarrierRecoveryResult:
    """PSK carrier recovery output and estimated residual impairments."""
    frequency_offset: Optional[float] = None
    phase_offset: Optional[float] = None
    confidence: float = 0.0
    recovered_samples: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.complex64))
    warnings: list[str] = field(default_factory=list)


@dataclass
class PhaseAmbiguityResult:
    """Candidate rotational corrections for a rotationally symmetric PSK signal."""
    modulation_order: int
    candidate_phases_rad: list[float] = field(default_factory=list)
    candidate_samples: list[np.ndarray] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ModulationFeatureResult:
    """Named numerical feature vector extracted from synchronized samples."""
    feature_names: tuple[str, ...] = ()
    values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    diagnostics: dict[str, float] = field(default_factory=dict)
    sample_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        """Return the feature vector as a name-to-value mapping."""
        return {name: float(value) for name, value in zip(self.feature_names, self.values)}


@dataclass
class ClassificationResult:
    """Ranked bounded modulation-classification result."""
    modulation: ModulationType = ModulationType.UNKNOWN
    confidence: float = 0.0
    method: str = "unavailable"
    probabilities: dict[ModulationType, float] = field(default_factory=dict)
    top_candidates: list[tuple[ModulationType, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    status: ClassifierStatus = ClassifierStatus.SUCCESS
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class SymbolResult:
    """Hard symbol decisions produced by a demodulator."""
    symbols: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.complex128))
    modulation: ModulationType = ModulationType.UNKNOWN
    confidence: float = 0.0
    symbol_count: int = 0
    warnings: list[str] = field(default_factory=list)
    symbol_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


@dataclass
class DemodulationResult:
    """Recovered bit and symbol decisions with demodulator diagnostics."""
    bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))
    llrs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    symbol_values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.complex128))
    modulation: ModulationType = ModulationType.UNKNOWN
    samples_per_symbol: Optional[float] = None
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    symbol_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
