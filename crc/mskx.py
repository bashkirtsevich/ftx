import typing

from consts.mskx import MSKX_LDPC_K_BYTES
from .crc import compute_crc
from tools import byte

# FIXME: Replace to "consts"
MSKX_CRC_POLYNOMIAL = 0x15D7
MSKX_CRC_WIDTH = 13
MSKX_CRC_TOP_BIT = 1 << (MSKX_CRC_WIDTH - 1)
MSKX_PAYLOAD_BITS = 96
MSKX_MESSAGE_BITS = MSKX_PAYLOAD_BITS - MSKX_CRC_WIDTH


def mskx_compute_crc(message: typing.ByteString, num_bits: int) -> int:
    return compute_crc(
        message, num_bits,
        crc_width=MSKX_CRC_WIDTH,
        crc_top_bit=MSKX_CRC_TOP_BIT,
        crc_polynomial=MSKX_CRC_POLYNOMIAL
    )


def mskx_add_crc(payload: typing.ByteString) -> typing.ByteString:
    message = payload + (b"\x00" * (MSKX_LDPC_K_BYTES - len(payload)))

    message[-3] &= 0xfc
    message[-2] = 0

    checksum = mskx_compute_crc(message, MSKX_MESSAGE_BITS)

    message[-3] |= byte(checksum >> 10)
    message[-2] = byte(checksum >> 2)
    message[-1] = byte(checksum << 6)

    return message
