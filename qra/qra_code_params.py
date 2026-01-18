from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from consts.q65 import QRAType
from qra.exceptions import InvalidQRAType


@dataclass(frozen=True)
class QRACodeParams:
    # code parameters
    K: int  # number of information symbols
    N: int  # codeword length in symbols
    m: int  # bits/symbol
    M: int  # Symbol alphabet cardinality (2^m)
    a: int  # code grouping factor
    # NC: int  # number of check symbols (N-K) # FIXME: To be deleted
    V: int  # number of variables in the code graph (N)
    C: int  # number of factors in the code graph (N +(N-K)+1)
    NMSG: int  # number of msgs in the code graph
    MAXVDEG: int  # maximum variable degree
    MAXCDEG: int  # maximum factor degree
    type: QRAType
    R: float  # code rate (K/N)
    # name: str  # code name # FIXME: To be deleted
    # tables used by the encoder
    acc_input_idx: npt.NDArray[np.int64]  # FIXME: To be deleted
    acc_input_wlog: npt.NDArray[np.int64]  # FIXME: To be deleted
    gflog: npt.NDArray[np.int64]  # FIXME: To be deleted
    gfexp: npt.NDArray[np.int64]  # FIXME: To be deleted
    # tables used by the decoder -------------------------
    msgw: npt.NDArray[np.int64]
    vdeg: npt.NDArray[np.int64]
    cdeg: npt.NDArray[np.int64]
    v2cmidx: npt.NDArray[np.int64]
    c2vmidx: npt.NDArray[np.int64]
    gfpmat: npt.NDArray[np.int64]  # Permutation matrix

    @property
    def bits_per_symbol(self):
        return self.m

    @property
    def code_rate(self):
        return 1.0 * self.message_length / self.codeword_length

    @property
    def message_length(self):
        # return the actual information message length (in symbols)
        # excluding any punctured symbol

        if self.type == QRAType.NORMAL:
            return self.K
        elif self.type in (QRAType.CRC, QRAType.CRC_PUNCTURED):
            return self.K - 1  # one information symbol of the underlying qra code is reserved for CRC
        elif self.type == QRAType.CRC_PUNCTURED2:
            return self.K - 2  # two code information symbols are reserved for CRC

        raise InvalidQRAType

    @property
    def codeword_length(self):
        # return the actual codeword length (in symbols)
        # excluding any punctured symbol

        if self.type in (QRAType.NORMAL, QRAType.CRC):
            return self.N  # no puncturing
        elif self.type == QRAType.CRC_PUNCTURED:
            return self.N - 1  # the CRC symbol is punctured
        elif self.type == QRAType.CRC_PUNCTURED2:
            return self.N - 2  # the two CRC symbols are punctured

        raise InvalidQRAType

    @property
    def alphabet_size(self):
        return self.M
