import typing

from consts import *

FT8_CRC_POLYNOMIAL = 0x2757
FT8_CRC_WIDTH = 14
TOPBIT = 1 << (FT8_CRC_WIDTH - 1)


def ftx_compute_crc(message: bytes, num_bits: int) -> int:
    remainder = 0
    idx_byte = 0

    for idx_bit in range(num_bits):
        if idx_bit % 8 == 0:
            remainder ^= message[idx_byte] << (FT8_CRC_WIDTH - 8)
            idx_byte += 1

        if remainder & TOPBIT != 0:
            remainder = (remainder << 1) ^ FT8_CRC_POLYNOMIAL
        else:
            remainder = remainder << 1

    return remainder & ((TOPBIT << 1) - 1)


def ftx_extract_crc(a91: bytes) -> int:
    return ((a91[9] & 0x07) << 11) | (a91[10] << 3) | (a91[11] >> 5)


def ftx_add_crc(payload: bytes) -> bytes:
    # Copy 77 bits of payload data
    a91 = payload + (b"\x00" * (FTX_LDPC_K_BYTES - len(payload)))

    # Clear 3 bits after the payload to make 82 bits
    a91[9] &= 0xf8
    a91[10] = 0

    # Calculate CRC of 82 bits (77 + 5 zeros)
    # 'The CRC is calculated on the source-encoded message, zero-extended from 77 to 82 bits'
    checksum = ftx_compute_crc(a91, 96 - FT8_CRC_WIDTH)  # FIXME: unnamed constant

    # Store the CRC at the end of 77 bit message
    a91[9] |= checksum >> 11 & 0xff
    a91[10] = checksum >> 3 & 0xff
    a91[11] = checksum << 5 & 0xff

    # print("checksum", checksum)

    return a91


def ftx_crc(msg1: typing.ByteString, msglen: int) -> typing.ByteString:
    div = [1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]

    # FIXME: Use concat
    msg = bytearray(b"\x00" * (FTX_LDPC_M + FT8_CRC_WIDTH))
    for i in range(msglen + FT8_CRC_WIDTH):
        if i < 77:
            msg[i] = msg1[i]

    for i in range(msglen):
        if msg[i] != 0:
            for j, d in enumerate(div):
                msg[i + j] = msg[i + j] ^ d

    return msg[msglen:msglen + FT8_CRC_WIDTH]


def ftx_check_crc(a91: typing.ByteString) -> bool:
    # [1]: 'The CRC is calculated on the source-encoded message, zero-extended from 77 to 82 bits.'
    out1 = ftx_crc(a91, 82)
    for i, b in enumerate(out1):
        if b != a91[FTX_LDPC_K - FT8_CRC_WIDTH + i]:
            return False
    return True
