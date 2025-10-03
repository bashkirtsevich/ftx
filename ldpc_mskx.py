# LDPC decoder for FT8.
#
# given a 174-bit codeword as an array of log-likelihood of zero,
# return a 174-bit corrected codeword, or zero-length array.
# last 87 bits are the (systematic) plain-text.
# this is an implementation of the sum-product algorithm
# from Sarah Johnson's Iterative Error Correction book.
# codeword[i] = log ( P(x=0) / P(x=1) )

import typing

from consts.mskx import MSKX_LDPC_K
from consts.mskx import MSKX_LDPC_M
from consts.mskx import MSKX_LDPC_N
from consts.mskx import MSK144_LDPC_MN
from consts.mskx import MSK144_LDPC_NM
from consts.mskx import MSK144_LDPC_NUM_ROWS
from numba import jit


# Ideas for approximating tanh/atanh:
# * https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
# * http://functions.wolfram.com/ElementaryFunctions/ArcTanh/10/0001/
# * https://mathr.co.uk/blog/2017-09-06_approximating_hyperbolic_tangent.html
# * https://math.stackexchange.com/a/446411

@jit(nopython=True)
def fast_tanh(x: float) -> float:
    if x < -4.97:
        return -1.0

    if x > 4.97:
        return 1.0

    x2 = x ** 2
    a = x * (945.0 + x2 * (105.0 + x2))
    b = 945.0 + x2 * (420.0 + x2 * 15.0)
    return a / b


@jit(nopython=True)
def fast_atanh(x: float) -> float:
    x2 = x ** 2
    a = x * (945.0 + x2 * (-735.0 + x2 * 64.0))
    b = (945.0 + x2 * (-1050.0 + x2 * 225.0))
    return a / b


def ldpc_check(codeword: bytes) -> int:
    """
    does a 174-bit codeword pass the FT8's LDPC parity checks?
    :param codeword:
    :return: number of parity errors, 0 means total success.
    """
    errors = 0

    for m in range(MSKX_LDPC_M):
        x = 0
        for i in range(MSK144_LDPC_NUM_ROWS[m]):
            x ^= codeword[MSK144_LDPC_NM[m][i] - 1]

        if x:
            errors += 1

    return errors


def bp_decode(codeword: typing.List[float], max_iters: int) -> typing.Tuple[int, typing.ByteString]:
    min_errors = MSKX_LDPC_M

    # initialize message data
    tov = [[0.0] * 3 for _ in range(MSKX_LDPC_N)]
    toc = [[0.0] * 11 for _ in range(MSKX_LDPC_M)]

    plain = bytearray(b"\x00" * MSKX_LDPC_N)

    for _ in range(max_iters):
        # Do a hard decision guess (tov=0 in iter 0)
        plain_sum = 0
        for n in range(MSKX_LDPC_N):
            plain[n] = int((codeword[n] + tov[n][0] + tov[n][1] + tov[n][2]) > 0)
            plain_sum += plain[n]

        if plain_sum == 0:
            min_errors = MSKX_LDPC_M
            break  # message converged to all-zeros, which is prohibited

        # Check to see if we have a codeword (check before we do any iter)
        if (errors := ldpc_check(plain)) < min_errors:
            min_errors = errors  # we have a better guess - update the result

            if errors == 0:
                break  # Found a perfect answer

        # Send messages from bits to check nodes
        for m in range(MSKX_LDPC_M):
            for n_idx in range(MSK144_LDPC_NUM_ROWS[m]):
                n = MSK144_LDPC_NM[m][n_idx] - 1
                Tnm = codeword[n]
                for m_idx in range(3):
                    if (MSK144_LDPC_MN[n][m_idx] - 1) != m:
                        Tnm += tov[n][m_idx]
                toc[m][n_idx] = fast_tanh(-Tnm / 2)

        # send messages from check nodes to variable nodes
        for n in range(MSKX_LDPC_N):
            for m_idx in range(3):
                m = MSK144_LDPC_MN[n][m_idx] - 1
                Tmn = 1.0
                for n_idx in range(MSK144_LDPC_NUM_ROWS[m]):
                    if (MSK144_LDPC_NM[m][n_idx] - 1) != n:
                        Tmn *= toc[m][n_idx]
                tov[n][m_idx] = -2 * fast_atanh(Tmn)

    return min_errors, plain
