import typing

import numpy as np

from consts.wspr import WSPR_PR3
from consts.wspr import WSPR_CONV_POLY


# Encode call, locator, and dBm into WSPR codeblock.
def wspr_encode(payload: typing.ByteString) -> typing.Generator[int, None, None]:
    # convolutional encoding K=32, r=1/2, Layland-Lushbaugh polynomials
    k = 0
    state = 0
    symbol = np.zeros(176, dtype=np.uint8)

    for p in payload:
        for i in range(7, -1, -1):
            state = (state << 1) | ((p >> i) & 1)
            for s in range(2):  # convolve
                n = state & WSPR_CONV_POLY[s]
                even = 0
                while n:
                    even = 1 - even
                    n &= n - 1

                symbol[k] = even
                k += 1

    tones = np.zeros(162, dtype=np.uint8)
    for i in range(162):
        p = -1
        k = 0
        j0 = 0
        while p != i:
            for j in range(8):
                j0 = (((k >> j) & 1) | (j0 << 1)) & 0xff

            if j0 < 162:
                p += 1

            k += 1

        tones[j0] = WSPR_PR3[j0] | symbol[i] << 1  # interleave and add sync std::vector

    for tone in tones:
        yield tone
