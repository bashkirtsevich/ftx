import numpy as np


def dB(x: float) -> float:
    val = -99.0

    if x > 1.259e-10:
        x = max(x, 0.000001)
        val = 10.0 * np.log10(x)

    return val
