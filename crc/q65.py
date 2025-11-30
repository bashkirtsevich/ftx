import numpy as np
from numba import njit
from numpy import typing as npt

CRC6_POLYNOMIAL = 0x30
CRC12_POLYNOMIAL = 0xF01


@njit
def crc_six_bit(x: npt.NDArray[np.uint8], polynomial: int) -> int:
    sr = 0
    for t in x:
        for _ in range(6):
            if (t ^ sr) & 0x01:
                sr = (sr >> 1) ^ polynomial
            else:
                sr >>= 1
            t >>= 1
    return sr


def crc6(x: npt.NDArray[np.uint8]) -> int:
    return crc_six_bit(x, CRC6_POLYNOMIAL)


def crc12(x: npt.NDArray[np.uint8]) -> int:
    return crc_six_bit(x, CRC12_POLYNOMIAL)
