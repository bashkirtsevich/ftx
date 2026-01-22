import numpy as np
import numpy.typing as npt


def wh_bfy(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64], base: int, offs: int, dist: int):
    dst[base + offs] = src[base + offs] + src[base + offs + dist]
    dst[base + offs + dist] = src[base + offs] - src[base + offs + dist]


def fwht_1(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64]):
    dst[0] = src[0]


def fwht_2(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64]):
    t = np.zeros(2, dtype=np.float64)

    wh_bfy(t, src, 0, 0, 1)
    dst[0] = t[0]
    dst[1] = t[1]


def fwht_4(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64]):
    t = np.zeros(4, dtype=np.float64)

    # group 1
    for i in range(2):
        wh_bfy(t, src, 0, i, 2)
    # group 2
    for i in range(2):
        wh_bfy(dst, t, i * 2, 0, 1)


def fwht_8(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64]):
    t1 = np.zeros(8, dtype=np.float64)
    t2 = np.zeros(8, dtype=np.float64)

    # group 1
    for i in range(4):
        wh_bfy(t1, src, 0, i, 4)
    # group 2
    for i in range(2):
        for j in range(2):
            wh_bfy(t2, t1, i * 4, j, 2)
    # group 3
    for i in range(4):
        wh_bfy(dst, t2, i * 2, 0, 1)


def fwht_16(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64]):
    t1 = np.zeros(16, dtype=np.float64)
    t2 = np.zeros(16, dtype=np.float64)

    # group 1
    for i in range(8):
        wh_bfy(t1, src, 0, i, 8)
    # group 2
    for i in range(2):
        for j in range(4):
            wh_bfy(t2, t1, i * 8, j, 4)
    # group 3
    for i in range(4):
        for j in range(2):
            wh_bfy(t1, t2, i * 4, j, 2)
    # group 4
    for i in range(8):
        wh_bfy(dst, t1, i * 2, 0, 1)


def fwht_32(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64]):
    t1 = np.zeros(32, dtype=np.float64)
    t2 = np.zeros(32, dtype=np.float64)

    # group 1
    for i in range(16):
        wh_bfy(t1, src, 0, i, 16)
    # group 2
    for i in range(2):
        for j in range(8):
            wh_bfy(t2, t1, i * 16, j, 8)
    # group 3
    for i in range(4):
        for j in range(4):
            wh_bfy(t1, t2, i * 8, j, 4)
    # group 4
    for i in range(8):
        for j in range(2):
            wh_bfy(t2, t1, i * 4, j, 2)
    # group 5
    for i in range(16):
        wh_bfy(dst, t2, i * 2, 0, 1)


def fwht_64(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64]):
    t1 = np.zeros(64, dtype=np.float64)
    t2 = np.zeros(64, dtype=np.float64)

    # group 1
    for i in range(32):
        wh_bfy(t1, src, 0, i, 32)
    # group 2
    for i in range(2):
        for j in range(16):
            wh_bfy(t2, t1, i * 32, j, 16)
    # group 3
    for i in range(4):
        for j in range(8):
            wh_bfy(t1, t2, i * 16, j, 8)
    # group 4
    for i in range(8):
        for j in range(4):
            wh_bfy(t2, t1, i * 8, j, 4)
    # group 5
    for i in range(16):
        for j in range(2):
            wh_bfy(t1, t2, i * 4, j, 2)
    # group 6
    for i in range(32):
        wh_bfy(dst, t1, i * 2, 0, 1)


fwht_tab = [
    fwht_1,
    fwht_2,
    fwht_4,
    fwht_8,
    fwht_16,
    fwht_32,
    fwht_64
]


# Fast Walsh-Hadamard direct transform
def fwht(N: int, dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64]):
    fwht_tab[N](dst, src)
