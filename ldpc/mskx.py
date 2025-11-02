import typing

import numpy as np
import numpy.typing as ntp
from consts.mskx import MSKX_LDPC_M
from consts.mskx import MSKX_LDPC_N
from consts.mskx import MSK144_LDPC_MN
from consts.mskx import MSK144_LDPC_NM
from consts.mskx import MSK144_LDPC_NUM_ROWS

from ldpc.bp_decoder import belief_propagation


def bp_decode(codeword: ntp.NDArray[np.float64], max_iters: int) -> typing.Tuple[int, ntp.NDArray[np.uint8]]:
    return belief_propagation(
        codeword, max_iters,
        ldpc_n=MSKX_LDPC_N, ldpc_m=MSKX_LDPC_M,
        n_v=3, m_c=11,
        ldpc_num_rows=MSK144_LDPC_NUM_ROWS,
        ldpc_nm=MSK144_LDPC_NM - 1,
        ldpc_mn=MSK144_LDPC_MN - 1,
    )
