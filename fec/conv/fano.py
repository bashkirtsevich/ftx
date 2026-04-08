import typing
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit

from .encoder import PARITY_TAB
from .node import Node
from .poly import LL_POLY1, LL_POLY2


@dataclass
class NodeFano(Node):
    metrics: npt.NDArray[np.int64]  # Metrics indexed by all possible tx syms
    tm: npt.NDArray[np.int64]  # Sorted metrics for current hypotheses
    i: int  # Current branch being tested


# Convolutionally encode a packet. The input data bytes are read
# high bit first and the encoded packet is written into 'symbols',
# one symbol per byte. The first symbol is generated from POLY1,
# the second from POLY2.
#
# Storing only one symbol per byte uses more space, but it is faster
# and easier than trying to pack them more compactly.
# def encode(
#         symbols: npt.NDArray[np.uint8],  # Output buffer, 2*nbytes*8
#         data: npt.NDArray[np.uint8],  # Input buffer, nbytes
#         nbytes: int,  # Number of bytes in data
#         poly1: int = LL_POLY1,
#         poly2: int = LL_POLY2,
# ):
#     enc_state = 0
#     for i in range(nbytes):
#         for j in range(8):
#             enc_state = (enc_state << 1) | ((data[i] >> (7 - j)) & 1)
#
#             sym = ENCODE(enc_state, poly1, poly2)
#
#             symbols[i * 2] = sym >> 1
#             symbols[i * 2 + 1] = sym & 1

@njit
def fano(
        symbols: npt.NDArray[np.uint8],  # Raw deinterleaved input symbols
        bits: int,  # Number of output bits
        metric_table: npt.NDArray[np.int64],  # [2][256] Metric table, [sent sym][rx symbol]
        delta: int,  # Threshold adjust parameter
        max_iter: int,
        poly1: int = LL_POLY1,
        poly2: int = LL_POLY2,
) -> typing.Optional[typing.Tuple[int, int, npt.NDArray[np.uint8]]]:
    s0 = symbols[0::2]
    s1 = symbols[1::2]

    metrics_all = np.empty((bits, 4), dtype=np.int64)
    metrics_all[:, 0] = metric_table[0, s0] + metric_table[0, s1]
    metrics_all[:, 1] = metric_table[0, s0] + metric_table[1, s1]
    metrics_all[:, 2] = metric_table[1, s0] + metric_table[0, s1]
    metrics_all[:, 3] = metric_table[1, s0] + metric_table[1, s1]

    encstates = np.zeros(bits + 1, dtype=np.uint64)
    gammas = np.zeros(bits + 1, dtype=np.int64)
    tms = np.zeros((bits, 2), dtype=np.int64)
    current_i = np.zeros(bits, dtype=np.uint8)

    node_id = 0
    node_id_max = bits - 1
    node_id_tail = bits - 31

    lsym = 0
    m0 = metrics_all[0, lsym]
    m1 = metrics_all[0, 3 ^ lsym]

    if m0 >= m1:
        tms[0, 0] = m0
        tms[0, 1] = m1
    else:
        tms[0, 0] = m1
        tms[0, 1] = m0
        encstates[0] = 1

    threshold = 0
    total_iters = max_iter * bits

    for cycle in range(1, total_iters + 1):
        current_gamma = gammas[node_id] + (tms[node_id, 0] if current_i[node_id] == 0 else tms[node_id, 1])

        if current_gamma >= threshold:
            if gammas[node_id] < threshold + delta:
                if current_gamma >= threshold + delta:
                    threshold += ((current_gamma - threshold) // delta) * delta

            node_id += 1
            gammas[node_id] = current_gamma

            if node_id > node_id_max:
                data = np.zeros(bits // 8, dtype=np.uint8)
                for j in range(bits // 8):
                    data[j] = encstates[(j + 1) * 8 - 1] & 0xff
                return current_gamma, cycle, data

            next_state = (encstates[node_id - 1] << 1)
            encstates[node_id] = next_state

            tmp = next_state & poly1
            tmp ^= tmp >> 16

            lsym = PARITY_TAB[(tmp ^ (tmp >> 8)) & 0xff] << 1
            tmp = next_state & poly2

            tmp ^= tmp >> 16
            lsym |= PARITY_TAB[(tmp ^ (tmp >> 8)) & 0xff]

            if node_id >= node_id_tail:
                tms[node_id, 0] = metrics_all[node_id, lsym]
                tms[node_id, 1] = -1000000
            else:
                m0 = metrics_all[node_id, lsym]
                m1 = metrics_all[node_id, 3 ^ lsym]
                if m0 >= m1:
                    tms[node_id, 0] = m0
                    tms[node_id, 1] = m1
                else:
                    tms[node_id, 0] = m1
                    tms[node_id, 1] = m0
                    encstates[node_id] |= 1

            current_i[node_id] = 0
            continue

        else:
            while True:
                if node_id == 0 or gammas[node_id - 1] < threshold:
                    threshold -= delta
                    if current_i[node_id] != 0:
                        current_i[node_id] = 0
                        encstates[node_id] ^= 1
                    break

                node_id -= 1
                if node_id < node_id_tail and current_i[node_id] != 1:
                    current_i[node_id] = 1
                    encstates[node_id] ^= 1
                    break
    return None
