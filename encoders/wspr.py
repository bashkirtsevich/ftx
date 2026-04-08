import typing

import numpy as np
import numpy.typing as npt

from consts.wspr import WSPR_PR3, WSPR_ND
from consts.wspr import WSPR_CONV_POLY
from consts.wspr import WSPR_CONV_SYMBOLS


def convolutional_encode(payload: typing.ByteString) -> npt.NDArray:
    # convolutional encoding K=32, r=1/2, Layland-Lushbaugh polynomials
    k = 0
    state = 0
    symbols = np.zeros(WSPR_CONV_SYMBOLS, dtype=np.uint8)

    for p in payload:
        for i in range(7, -1, -1):
            state = (state << 1) | ((p >> i) & 1)
            for poly in WSPR_CONV_POLY:
                n = state & poly
                even = 0
                while n:
                    even = 1 - even
                    n &= n - 1

                symbols[k] = even
                k += 1

    return symbols


# Encode call, locator, and dBm into WSPR codeblock.
def wspr_encode(payload: typing.ByteString) -> typing.Iterator[int]:
    symbols = convolutional_encode(payload)

    tones = np.zeros(WSPR_ND, dtype=np.uint8)
    for i in range(WSPR_ND):
        p = -1
        k = 0
        j0 = 0
        while p != i:
            for j in range(8):
                j0 = (((k >> j) & 1) | (j0 << 1)) & 0xff

            if j0 < WSPR_ND:
                p += 1

            k += 1

        tones[j0] = WSPR_PR3[j0] | symbols[i] << 1  # interleave and add sync std::vector

    for tone in tones:
        yield tone

# def wspr_encode(payload: bytes) -> typing.Generator[int, None, None]:
#     symbols = convolutional_encode(payload)
#
#     # 1. Генерируем таблицу перемежения (interleaving table) за один проход
#     interleave_order = []
#     j0 = 0
#     for k in range(1024):  # Достаточный диапазон для поиска 162 индексов
#         # Разворот 8 бит (bit-reversal)
#         j0 = int(f'{k & 0xFF:08b}'[::-1], 2)
#         if j0 < 162:
#             interleave_order.append(j0)
#             if len(interleave_order) == 162:
#                 break
#
#     # 2. Применяем перемежение и добавляем биты синхронизации
#     # WSPR_PR3 — массив синхропоследовательности (162 элемента)
#     tones = np.zeros(162, dtype=np.uint8)
#     for i, pos in enumerate(interleave_order):
#         tones[pos] = (symbols[i] << 1) | WSPR_PR3[pos]
#
#     yield from tones
