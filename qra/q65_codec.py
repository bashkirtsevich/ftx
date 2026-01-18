from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from qra.qra_code_params import QRACodeParams


@dataclass
class Q65Codec:
    qra_code: QRACodeParams  # qra code to be used by the codec
    decoderEsNoMetric: float  # value for which we optimize the decoder metric
    x: npt.NDArray[np.int64]  # codec input
    y: npt.NDArray[np.int64]  # codec output
    qra_v2cmsg: npt.NDArray[np.float64]  # decoder v->c messages
    qra_c2vmsg: npt.NDArray[np.float64]  # decoder c->v messages
    ix: npt.NDArray[np.float64]  # decoder intrinsic information
    ex: npt.NDArray[np.float64]  # decoder extrinsic information
    # variables used to compute the intrinsics in the fast-fading case
    BinsPerTone: int
    BinsPerSymbol: int
    NoiseVar: float
    EsNoMetric: float
    WeightsCount: int  # FIXME: To be deleted
    FastFadingWeights: npt.NDArray[np.float64]  # FIXME: To be deleted
