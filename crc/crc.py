import typing


def compute_crc(message: typing.ByteString, num_bits: int,
                crc_width: int, crc_top_bit: int, crc_polynomial: int) -> int:
    remainder = 0
    idx_byte = 0

    for idx_bit in range(num_bits):
        if idx_bit % 8 == 0:
            remainder ^= message[idx_byte] << (crc_width - 8)
            idx_byte += 1

        if remainder & crc_top_bit != 0:
            remainder = (remainder << 1) ^ crc_polynomial
        else:
            remainder = remainder << 1

    return remainder & ((crc_top_bit << 1) - 1)
