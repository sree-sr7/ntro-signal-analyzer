"""Enumerations for signal formats, data types, layouts, and statuses."""

from enum import Enum


class SignalFormat(str, Enum):
    IQ = "iq"
    WAV = "wav"
    SIGMF = "sigmf"
    UNKNOWN = "unknown"

class IQLayout(str, Enum):
    IQ_INTERLEAVED = "iq_interleaved"    # float pairs: I0,Q0,I1,Q1,...
    QI_INTERLEAVED = "qi_interleaved"    # float pairs: Q0,I0,Q1,I0,...
    COMPLEX_NATIVE = "complex_native"    # numpy complex64/complex128
    REAL_ONLY = "real_only"              # single real-valued channel
    UNKNOWN = "unknown"

class Endianness(str, Enum):
    LITTLE = "little"
    BIG = "big"
    NATIVE = "native"

class SampleDtype(str, Enum):
    COMPLEX64 = "complex64"     # numpy complex64 (2x float32)
    COMPLEX128 = "complex128"   # numpy complex128 (2x float64)
    FLOAT32 = "float32"         # interleaved float32 IQ
    FLOAT64 = "float64"         # interleaved float64 IQ
    INT16 = "int16"             # interleaved int16 IQ
    INT8 = "int8"               # interleaved int8 IQ
    UINT8 = "uint8"             # interleaved uint8 IQ (RTL-SDR)

class LoadStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"         # loaded with warnings
    FAILED = "failed"


class ModulationType(str, Enum):
    """Modulation families currently supported by Phase 3."""

    UNKNOWN = "unknown"
    BPSK = "bpsk"
    QPSK = "qpsk"
    PSK8 = "8psk"
    QAM16 = "16qam"
    FSK2 = "2fsk"


class ClassifierStatus(str, Enum):
    """Operational state for classical and ONNX classification results."""

    SUCCESS = "success"
    INSUFFICIENT_DATA = "insufficient_data"
    MODEL_UNAVAILABLE = "model_unavailable"
    ERROR = "error"
