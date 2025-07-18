import typing

from consts import *
from tools import byte

FTX_CRC_POLYNOMIAL = 0x2757
FTX_CRC_WIDTH = 14
FFTX_CRC_TOP_BIT = 1 << (FTX_CRC_WIDTH - 1)
FTX_PAYLOAD_BITS = 96
FTX_MESSAGE_BITS = FTX_PAYLOAD_BITS - FTX_CRC_WIDTH


def ftx_compute_crc(message: typing.ByteString, num_bits: int) -> int:
    remainder = 0
    idx_byte = 0

    for idx_bit in range(num_bits):
        if idx_bit % 8 == 0:
            remainder ^= message[idx_byte] << (FTX_CRC_WIDTH - 8)
            idx_byte += 1

        if remainder & FFTX_CRC_TOP_BIT != 0:
            remainder = (remainder << 1) ^ FTX_CRC_POLYNOMIAL
        else:
            remainder = remainder << 1

    return remainder & ((FFTX_CRC_TOP_BIT << 1) - 1)


def ftx_extract_crc(a91: typing.ByteString) -> int:
    return ((a91[9] & 0x07) << 11) | (a91[10] << 3) | (a91[11] >> 5)


def ftx_add_crc(payload: typing.ByteString) -> typing.ByteString:
    # Copy 77 bits of payload data
    message = payload + (b"\x00" * (FTX_LDPC_K_BYTES - len(payload)))

    # Clear 3 bits after the payload to make 82 bits
    message[-3] &= 0xf8
    message[-2] = 0

    # Calculate CRC of 82 bits (77 + 5 zeros)
    # 'The CRC is calculated on the source-encoded message, zero-extended from 77 to 82 bits'
    checksum = ftx_compute_crc(message, FTX_MESSAGE_BITS)

    # Store the CRC at the end of 77 bit message
    message[-3] |= byte(checksum >> 11)
    message[-2] = byte(checksum >> 3)
    message[-1] = byte(checksum << 5)

    return message


def ftx_crc(msg1: typing.ByteString, msglen: int) -> typing.ByteString:
    div = [1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]

    # FIXME: Use concat
    msg = bytearray(b"\x00" * (FTX_LDPC_M + FTX_CRC_WIDTH))
    for i in range(msglen + FTX_CRC_WIDTH):
        if i < 77:
            msg[i] = msg1[i]

    for i in range(msglen):
        if msg[i] != 0:
            for j, d in enumerate(div):
                msg[i + j] = msg[i + j] ^ d

    return msg[msglen:msglen + FTX_CRC_WIDTH]


def ftx_check_crc(a91: typing.ByteString) -> bool:
    # [1]: 'The CRC is calculated on the source-encoded message, zero-extended from 77 to 82 bits.'
    out1 = ftx_crc(a91, 82)
    for i, b in enumerate(out1):
        if b != a91[FTX_LDPC_K - FTX_CRC_WIDTH + i]:
            return False
    return True
