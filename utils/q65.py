import numpy as np
import numpy.typing as npt
from numba import njit


def smooth_121(a: np.ndarray):
    v = np.array([0.25, 0.5, 0.25])
    convolved = np.convolve(a, v, mode='same')
    return np.concat([a[:1], convolved[1:-1], a[-1:]])


@njit
def shell(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = len(arr)
    inc = 1
    while 3 * inc + 1 <= n:
        inc = 3 * inc + 1

    while inc > 1:
        for i in range(inc, n):
            v = arr[i]
            j = i

            while j >= inc and arr[j - inc] > v:
                arr[j] = arr[j - inc]
                j -= inc

            arr[j] = v

        inc //= 3

    return arr


def shell_sort_percentile(arr: npt.NDArray[np.float64], percentile: float) -> np.float64:
    points = len(arr)
    tmp = shell(arr.copy())
    i = min(max(int(points * 0.01 * percentile), 0), points - 1)
    return tmp[i]


def q65_6bit_decode(values: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    c77 = np.zeros(13 * 6 + 5, dtype=np.bool)

    bc = 0
    for i in range(13):
        bits = 6
        b = int(values[i])

        izz = bits - 1
        for j in range(bits):
            c77[bc] = (1 & (b >> -(j - izz)))
            bc += 1

    return np.packbits(c77[:bc - 1])


# @njit
def bzap(s3: npt.NDArray[np.float64], LL: int):
    NBZAP = 15
    hist = np.zeros(LL, dtype=np.int64)

    for j in range(63):
        beg = j * LL
        ipk1 = np.argmax(s3[beg:beg + LL])
        hist[ipk1] += 1

    if np.max(hist) > NBZAP:
        for i in range(LL):
            if hist[i] > NBZAP:
                for j in range(63):
                    s3[j * LL + i] = 1.0
