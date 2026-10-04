from pathlib import Path
from typing import Optional
from core.common.enums import SignalFormat, SampleDtype

def detect_format(path: Path) -> SignalFormat:
    ext = path.suffix.lower()
    if ext == '.wav':
        return SignalFormat.WAV
    elif ext == '.sigmf-meta':
        return SignalFormat.SIGMF
    elif ext == '.sigmf-data':
        if path.with_suffix('.sigmf-meta').exists():
            return SignalFormat.SIGMF
        return SignalFormat.UNKNOWN
    elif ext in ['.iq', '.bin', '.raw', '.cf32', '.cf64', '.cs16', '.cs8', '.cu8', '.ci16', '.ci8']:
        return SignalFormat.IQ
    
    try:
        with open(path, 'rb') as f:
            if f.read(4) == b'RIFF':
                return SignalFormat.WAV
    except Exception:
        pass
        
    return SignalFormat.UNKNOWN

def detect_iq_dtype_hint(path: Path) -> Optional[SampleDtype]:
    ext = path.suffix.lower()
    if ext == '.cf32':
        return SampleDtype.COMPLEX64
    elif ext == '.cf64':
        return SampleDtype.COMPLEX128
    elif ext in ['.cs16', '.ci16']:
        return SampleDtype.INT16
    elif ext in ['.cs8', '.ci8']:
        return SampleDtype.INT8
    elif ext == '.cu8':
        return SampleDtype.UINT8
    return None

def detect_format_from_content(path: Path) -> SignalFormat:
    try:
        with open(path, 'rb') as f:
            if f.read(4) == b'RIFF':
                return SignalFormat.WAV
    except Exception:
        pass
        
    if path.with_suffix('.sigmf-meta').exists() or path.name.endswith('.sigmf-meta'):
        return SignalFormat.SIGMF
        
    return SignalFormat.UNKNOWN
