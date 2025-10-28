import typing

import numpy as np
from numba import jit
import numpy.typing as npt
from crc.q65 import crc12

from consts.q65 import Q65_SYNC
from consts.q65 import qra_NC
from consts.q65 import qra_M
from consts.q65 import qra_a
from consts.q65 import qra_acc_input_idx
from consts.q65 import qra_acc_input_wlog
from consts.q65 import qra_log
from consts.q65 import qra_exp


@jit(nopython=True)
def qra_encode(x: npt.NDArray[np.int64], concat: bool = False) -> npt.NDArray[np.int64]:
    y = np.zeros(qra_NC, dtype=np.int64)

    # compute the code check symbols as a weighted accumulation of a permutated
    # sequence of the (repeated) systematic input symbols:
    # chk(k+1) = x(idx(k))*alfa^(logw(k)) + chk(k)
    # (all operations performed over GF(M))

    chk = 0
    for k in np.arange(qra_NC):
        kk = qra_a * k
        for j in np.arange(qra_a):
            jj = kk + j
            # irregular grouping support
            # if qra_acc_input_idx[jj] < 0:
            #     continue

            t = x[qra_acc_input_idx[jj]]
            if t:
                # multiply input by weight[k] and xor it with previous check
                t = (qra_log[t] + qra_acc_input_wlog[jj]) % (qra_M - 1)
                t = qra_exp[t]
                chk ^= t

        y[k] = chk

    if concat:
        return np.concat((x, y,))

    return y


def q65_encode_msg(msg: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    # compute and append the appropriate CRC
    crc = crc12(msg)
    data_in = np.concat((msg, (crc & 0x3F, crc >> 6),))
    # encode with the given qra code
    data_out = qra_encode(data_in)
    # exclude crc from result data array for this encoding type
    return np.concat((msg, data_out,))


def q65_6bit_encode(bits: npt.NDArray[np.uint8]) -> npt.NDArray[np.int64]:
    n_chunks = bits.shape[0] // 6
    chunks_reshaped = bits.reshape((n_chunks, 6))
    values = chunks_reshaped.dot(1 << np.arange(5, -1, -1))
    return values.tolist()


def q65_encode(payload: typing.ByteString) -> npt.NDArray[np.int64]:
    arr = np.frombuffer(payload, dtype=np.uint8)
    bits = np.unpackbits(arr)

    msg = q65_6bit_encode(bits)
    codeword = q65_encode_msg(msg)

    tones_count = 85
    tones = np.zeros(tones_count, dtype=np.int64)

    tone_indices = np.setdiff1d(np.arange(tones_count), Q65_SYNC - 1)
    tones[tone_indices] = codeword + 1

    return tones
