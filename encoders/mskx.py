import typing

import numpy as np
import numpy.typing as npt

from consts.mskx import MSKX_LDPC_K_BYTES
from consts.mskx import MSKX_LDPC_N_BYTES
from consts.mskx import MSKX_LDPC_K
from consts.mskx import MSKX_LDPC_M
from consts.mskx import MSK144_LDPC_GENERATOR
from consts.mskx import MSKX_LDPC_N
from crc.mskx import mskx_add_crc
from encoders.ftx import parity8


def mskx_encode(message: typing.ByteString) -> typing.ByteString:
    # This implementation accesses the generator bits straight from the packed binary representation in kFTX_LDPC_generator
    # Fill the codeword with message and zeros, as we will only update binary ones later
    codeword = bytearray(message[i] if i < MSKX_LDPC_K_BYTES else 0 for i in range(MSKX_LDPC_N_BYTES))

    # Compute the byte index and bit mask for the first checksum bit
    col_mask = 0x80 >> (MSKX_LDPC_K % 8)  # bitmask of current byte
    col_idx = MSKX_LDPC_K_BYTES - 1  # index into byte array

    # Compute the LDPC checksum bits and store them in codeword
    for i in range(MSKX_LDPC_M):
        # Fast implementation of bitwise multiplication and parity checking
        # Normally nsum would contain the result of dot product between message and kFTX_LDPC_generator[i],
        # but we only compute the sum modulo 2.
        nsum = 0
        for j in range(MSKX_LDPC_K_BYTES):
            bits = message[j] & MSK144_LDPC_GENERATOR[i][j]  # bitwise AND (bitwise multiplication)
            nsum ^= parity8(bits)  # bitwise XOR (addition modulo 2)

        # Set the current checksum bit in codeword if nsum is odd
        if nsum % 2:
            codeword[col_idx] |= col_mask

        # Update the byte index and bit mask for the next checksum bit
        col_mask >>= 1
        if col_mask == 0:
            col_mask = 0x80
            col_idx += 1

    return codeword


def msk144_encode(payload: typing.ByteString) -> npt.NDArray[np.int64]:
    a96 = mskx_add_crc(payload)
    codeword = mskx_encode(a96)

    sync = b"\x72"  # 0,1,1,1,0,0,1,0
    envelope = sync + codeword[0:6] + sync + codeword[6:16]

    MSK144_BITS = MSKX_LDPC_N + 16

    signs = [2 * ((envelope[i // 8] >> (7 - (i % 8))) & 1) - 1
             for i in range(MSK144_BITS)] + [0]  # FIXME: + [0] -- is a shit!

    tones = [0 for _ in range(MSK144_BITS)]
    for i in range(MSK144_BITS // 2):
        tones[2 * i - 0] = (signs[2 * i + 1] * signs[2 * i - 0] + 1) // 2
        tones[2 * i + 1] = -(signs[2 * i + 1] * signs[2 * i + 2] - 1) // 2

    return np.fromiter((-tone + 1 for tone in tones), dtype=np.int64)
