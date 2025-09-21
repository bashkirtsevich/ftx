import typing

from consts_ftx import FT4_NN
from consts_ftx import FT8_NN
from consts_ftx import FTX_LDPC_K
from consts_ftx import FTX_LDPC_K_BYTES
from consts_ftx import FTX_LDPC_M
from consts_ftx import FTX_LDPC_N_BYTES
from consts_ftx import FT4_COSTAS_PATTERN
from consts_ftx import FT4_GRAY_MAP
from consts_ftx import FT4_XOR_SEQUENCE
from consts_ftx import FT8_COSTAS_PATTERN
from consts_ftx import FT8_GRAY_MAP
from consts_ftx import FTX_LDPC_GENERATOR
from consts_mskx import MSKX_LDPC_K, MSKX_LDPC_M, MSKX_LDPC_N, MSKX_LDPC_N_BYTES, MSKX_LDPC_K_BYTES, MSK144_LDPC_GENERATOR
from crc import ftx_add_crc
from crc import mskx_add_crc
from tools import byte


def parity8(x: int) -> int:
    for i in [4, 2, 1]:
        x ^= x >> i
    # x ^= x >> 4  # a b c d ae bf cg dh
    # x ^= x >> 2  # a b ac bd cae dbf aecg bfdh
    # x ^= x >> 1  # a ab bac acbd bdcae caedbf aecgbfdh
    return byte(x % 2)  # modulo 2


# Encode via LDPC a 91-bit message and return a 174-bit codeword.
# The generator matrix has dimensions (87,87).
# The code is a (174,91) regular LDPC code with column weight 3.
# Arguments:
# [IN] message   - array of 91 bits stored as 12 bytes (MSB first)
# [OUT] codeword - array of 174 bits stored as 22 bytes (MSB first)
def ftx_encode(message: typing.ByteString) -> typing.ByteString:
    # This implementation accesses the generator bits straight from the packed binary representation in kFTX_LDPC_generator
    # Fill the codeword with message and zeros, as we will only update binary ones later
    codeword = bytearray(message[i] if i < FTX_LDPC_K_BYTES else 0 for i in range(FTX_LDPC_N_BYTES))

    # Compute the byte index and bit mask for the first checksum bit
    col_mask = 0x80 >> (FTX_LDPC_K % 8)  # bitmask of current byte
    col_idx = FTX_LDPC_K_BYTES - 1  # index into byte array

    # Compute the LDPC checksum bits and store them in codeword
    for i in range(FTX_LDPC_M):
        # Fast implementation of bitwise multiplication and parity checking
        # Normally nsum would contain the result of dot product between message and kFTX_LDPC_generator[i],
        # but we only compute the sum modulo 2.
        nsum = 0
        for j in range(FTX_LDPC_K_BYTES):
            bits = message[j] & FTX_LDPC_GENERATOR[i][j]  # bitwise AND (bitwise multiplication)
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


def ft8_encode(payload: typing.ByteString) -> typing.Generator[int, None, None]:
    # Compute and add CRC at the end of the message
    # a91 contains 77 bits of payload + 14 bits of CRC
    a91 = ftx_add_crc(payload)
    codeword = ftx_encode(a91)

    # Message structure: S7 D29 S7 D29 S7
    # Total symbols: 79 (FT8_NN)
    mask = 0x80  # Mask to extract 1 bit from codeword
    i_byte = 0  # Index of the current byte of the codeword

    for i_tone in range(FT8_NN):
        # FIXME: Optimize/normalize
        if 7 > i_tone >= 0:
            yield FT8_COSTAS_PATTERN[i_tone]
        elif 43 > i_tone >= 36:
            yield FT8_COSTAS_PATTERN[i_tone - 36]
        elif 79 > i_tone >= 72:
            yield FT8_COSTAS_PATTERN[i_tone - 72]
        else:
            # Extract 3 bits from codeword at i-th position
            bits3 = 0

            for bit_or in [4, 2, 1]:
                # print("codeword len", len(codeword), "i_byte", i_byte)
                if codeword[i_byte] & mask:
                    bits3 |= bit_or

                mask >>= 1
                if mask == 0:
                    mask = 0x80
                    i_byte += 1

            yield FT8_GRAY_MAP[bits3]


def ft4_encode(payload: typing.ByteString) -> typing.Generator[int, None, None]:
    # '[..] for FT4 only, in order to avoid transmitting a long string of zeros when sending CQ messages,
    # the assembled 77-bit message is bitwise exclusive-OR’ed with [a] pseudorandom sequence before computing the CRC and FEC parity bits'
    payload_xor = bytearray(payload[i] ^ FT4_XOR_SEQUENCE[i] for i in range(10))

    # Compute and add CRC at the end of the message
    # a91 contains 77 bits of payload + 14 bits of CRC
    a91 = ftx_add_crc(payload_xor)

    codeword = ftx_encode(a91)  # 91 bits -> 174 bits

    # Message structure: R S4_1 D29 S4_2 D29 S4_3 D29 S4_4 R
    # Total symbols: 105 (FT4_NN)

    mask = 0x80  # Mask to extract 1 bit from codeword
    i_byte = 0  # Index of the current byte of the codeword
    for i_tone in range(FT4_NN):
        if i_tone == 0 or i_tone == 104:
            yield 0  # R (ramp) symbol
        # FIXME: Optimize
        elif 5 > i_tone >= 1:
            yield FT4_COSTAS_PATTERN[0][i_tone - 1]
        elif 38 > i_tone >= 34:
            yield FT4_COSTAS_PATTERN[1][i_tone - 34]
        elif 71 > i_tone >= 67:
            yield FT4_COSTAS_PATTERN[2][i_tone - 67]
        elif 104 > i_tone >= 100:
            yield FT4_COSTAS_PATTERN[3][i_tone - 100]
        else:
            bits2 = 0  # Extract 2 bits from codeword at i-th position

            for bit_or in [2, 1]:
                if codeword[i_byte] & mask:
                    bits2 |= bit_or

                mask >>= 1
                if mask == 0:
                    mask = 0x80
                    i_byte += 1

            yield FT4_GRAY_MAP[bits2]


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


def msk144_encode(payload: typing.ByteString) -> typing.Generator[int, None, None]:
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

    for tone in tones:
        yield -tone + 1
