import json
import numpy as np
from pathlib import Path
from core.common.enums import SignalFormat, IQLayout, SampleDtype
from core.common.models import SignalMetadata, SignalData, LoadResult

def load_sigmf(meta_path: Path) -> LoadResult:
    warnings = []
    try:
        if not meta_path.exists():
            return LoadResult.fail([f"File not found: {meta_path}"])

        try:
            from sigmf import SigMFFile
            smf = SigMFFile(metadata=str(meta_path))
            global_meta = smf.get_global_info()
            captures = smf.get_captures()
            capture_meta = captures[0] if captures else {}
            datatype = global_meta.get('core:datatype')
            sample_rate = global_meta.get('core:sample_rate')
            center_freq = capture_meta.get('core:frequency')
        except Exception:
            # Fallback to direct json parse
            with open(meta_path, 'r') as f:
                meta_dict = json.load(f)
            global_meta = meta_dict.get('global', {})
            captures = meta_dict.get('captures', [])
            capture_meta = captures[0] if captures else {}
            datatype = global_meta.get('core:datatype')
            sample_rate = global_meta.get('core:sample_rate')
            center_freq = capture_meta.get('core:frequency')
            
        if not datatype:
            return LoadResult.fail(["Missing core:datatype in SigMF metadata"])
            
        if sample_rate is None:
            warnings.append("Missing sample rate (core:sample_rate) in SigMF metadata")
        if center_freq is None:
            warnings.append("Missing core:frequency in SigMF metadata")

        is_complex = datatype.startswith('c')
        is_le = datatype.endswith('_le')
        
        if 'f32' in datatype:
            np_dt, enum_dt = np.float32, SampleDtype.FLOAT32
        elif 'f64' in datatype:
            np_dt, enum_dt = np.float64, SampleDtype.FLOAT64
        elif 'i16' in datatype:
            np_dt, enum_dt = np.int16, SampleDtype.INT16
        elif 'i8' in datatype:
            np_dt, enum_dt = np.int8, SampleDtype.INT8
        elif 'u8' in datatype:
            np_dt, enum_dt = np.uint8, SampleDtype.UINT8
        else:
            return LoadResult.fail([f"Unsupported core:datatype: {datatype}"])
            
        dt = np.dtype(np_dt).newbyteorder('<' if is_le else '>')

        data_path = meta_path.with_suffix('.sigmf-data')
        if not data_path.exists():
            return LoadResult.fail([f"Data file not found: {data_path}"])

        raw_data = np.fromfile(data_path, dtype=dt)
        
        if is_complex:
            i, q = raw_data[0::2], raw_data[1::2]
        else:
            i, q = raw_data, np.zeros_like(raw_data)
            
        if enum_dt == SampleDtype.INT16:
            i, q = i.astype(np.float32) / 32768.0, q.astype(np.float32) / 32768.0
        elif enum_dt == SampleDtype.INT8:
            i, q = i.astype(np.float32) / 128.0, q.astype(np.float32) / 128.0
        elif enum_dt == SampleDtype.UINT8:
            i, q = (i.astype(np.float32) - 127.5) / 128.0, (q.astype(np.float32) - 127.5) / 128.0
            
        if enum_dt == SampleDtype.FLOAT64:
            samples = (i + 1j * q).astype(np.complex128)
        else:
            samples = i.astype(np.float32) + 1j * q.astype(np.float32)
            
        sample_count = len(samples)
            
        metadata = SignalMetadata(
            source_path=meta_path,
            format=SignalFormat.SIGMF,
            sample_rate=float(sample_rate) if sample_rate else None,
            center_frequency=float(center_freq) if center_freq else None,
            sample_count=sample_count,
            duration_seconds=sample_count / sample_rate if sample_rate else None,
            dtype=enum_dt,
            iq_layout=IQLayout.IQ_INTERLEAVED if is_complex else IQLayout.REAL_ONLY,
            source_metadata={"sigmf_datatype": datatype},
            warnings=list(warnings),
        )
        return LoadResult.ok(SignalData(samples, metadata), warnings)
        
    except Exception as e:
        return LoadResult.fail([str(e)])
