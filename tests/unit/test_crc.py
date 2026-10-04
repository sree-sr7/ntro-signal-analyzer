import numpy as np

from core.bitstream.crc import append_crc16, compute_crc16, verify_crc16
from core.bitstream.extractor import bytes_to_bits


def test_crc16_ccitt_false_reference_vector():
    assert compute_crc16(b"123456789") == 0x29B1
    assert compute_crc16(bytes_to_bits(b"123456789")) == 0x29B1


def test_crc_append_and_verify_for_bytes_and_bits():
    message_bytes = b"NTRO step 79"
    frame_bytes = append_crc16(message_bytes)
    assert isinstance(frame_bytes, bytes)
    assert frame_bytes[-2:] == compute_crc16(message_bytes).to_bytes(2, "big")
    assert verify_crc16(frame_bytes)

    message_bits = bytes_to_bits(message_bytes)
    frame_bits = append_crc16(message_bits)
    assert frame_bits.dtype == np.uint8
    assert frame_bits.size == message_bits.size + 16
    assert verify_crc16(frame_bits)


def test_single_bit_corruption_fails_crc():
    frame = append_crc16(bytes_to_bits(b"payload"))
    corrupted = frame.copy()
    corrupted[11] ^= np.uint8(1)

    assert not verify_crc16(corrupted)
