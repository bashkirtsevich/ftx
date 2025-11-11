import numpy as np
from numba import jit
from numpy import typing as npt

CRC12_POLYNOMIAL = 0xF01


@jit(nopython=True)
def crc12(x: npt.NDArray[np.uint8]) -> int:
    sr = 0
    for t in x:
        for _ in range(6):
            if (t ^ sr) & 0x01:
                sr = (sr >> 1) ^ CRC12_POLYNOMIAL
            else:
                sr >>= 1
            t >>= 1
    return sr
