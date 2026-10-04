import numpy as np
import pytest

from core.common.enums import ModulationType
from libs.modulation_library import (
    bits_to_symbols,
    indices_to_symbols,
    modulation_definition,
    symbol_indices_to_bits,
    symbols_to_indices,
)


@pytest.mark.parametrize(
    "modulation,bits_per_symbol",
    [
        (ModulationType.BPSK, 1),
        (ModulationType.QPSK, 2),
        (ModulationType.PSK8, 3),
        (ModulationType.QAM16, 4),
        (ModulationType.FSK2, 1),
    ],
)
def test_central_bit_symbol_index_mappings_round_trip(modulation, bits_per_symbol):
    definition = modulation_definition(modulation)
    bits = np.tile(np.arange(2, dtype=np.uint8), 64 * bits_per_symbol)
    symbols = bits_to_symbols(bits, modulation)
    indices = symbols_to_indices(symbols, modulation)

    np.testing.assert_array_equal(indices_to_symbols(indices, modulation), symbols)
    np.testing.assert_array_equal(symbol_indices_to_bits(indices, modulation), bits)
    assert definition.bits_per_symbol == bits_per_symbol
    assert np.isclose(np.mean(np.abs(definition.constellation) ** 2), 1.0)
