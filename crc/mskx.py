import typing

import numpy as np
import numpy.typing as npt
from consts.mskx import *
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


def mskx_extract_crc(msg: typing.ByteString) -> int:
    return ((msg[9] & 0x07) << 11) | (msg[10] << 3) | (msg[11] >> 5)


def mskx_add_crc(payload: typing.ByteString) -> typing.ByteString:
    message = payload + (b"\x00" * (MSKX_LDPC_K_BYTES - len(payload)))

    message[-3] &= 0xfc
    message[-2] = 0

    checksum = mskx_compute_crc(message, MSKX_MESSAGE_BITS)

    message[-3] |= byte(checksum >> 10)
    message[-2] = byte(checksum >> 2)
    message[-1] = byte(checksum << 6)

    return message


def mskx_crc(msg_bits: npt.NDArray[np.uint8], msg_len: int) -> npt.NDArray[np.uint8]:
    div = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1]  # 0x15D7 in binary representation

    # FIXME: Use concat
    msg = np.zeros(msg_len + MSKX_CRC_WIDTH, dtype=np.uint8)
    for i in range(msg_len + MSKX_CRC_WIDTH):
        if i < 77:
            msg[i] = msg_bits[i]

    for i in range(msg_len):
        if msg[i] != 0:
            for j, d in enumerate(div):
                msg[i + j] = msg[i + j] ^ d

    return msg[msg_len:msg_len + MSKX_CRC_WIDTH]


def mskx_check_crc(msg_bits: npt.NDArray[np.uint8]) -> bool:
    # [1]: 'The CRC is calculated on the source-encoded message, zero-extended from 77 to 83 bits.'
    crc = mskx_crc(msg_bits, 83)
    return np.array_equal(crc, msg_bits[MSKX_LDPC_K - MSKX_CRC_WIDTH:MSKX_LDPC_K])
