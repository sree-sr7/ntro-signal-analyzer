"""Modulation-type demodulator router for future pipeline integration."""

from __future__ import annotations

import numpy as np

from core.common.enums import ModulationType
from core.common.models import DemodulationResult
from .demod_fsk import demodulate_fsk
from .demod_psk import demodulate_psk
from .demod_qam import demodulate_qam


def demodulate(
    samples: np.ndarray,
    modulation: ModulationType,
    **configuration: object,
) -> DemodulationResult:
    """Dispatch synchronized samples to the matching supported demodulator."""
    modulation = ModulationType(modulation)
    if modulation in (ModulationType.BPSK, ModulationType.QPSK, ModulationType.PSK8):
        return demodulate_psk(samples, modulation, **configuration)
    if modulation is ModulationType.QAM16:
        return demodulate_qam(samples, modulation, **configuration)
    if modulation is ModulationType.FSK2:
        return demodulate_fsk(samples, **configuration)
    raise ValueError("UNKNOWN or unsupported modulation cannot be demodulated")
