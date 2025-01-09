# LDPC decoder for FT8.
#
# given a 174-bit codeword as an array of log-likelihood of zero,
# return a 174-bit corrected codeword, or zero-length array.
# last 87 bits are the (systematic) plain-text.
# this is an implementation of the sum-product algorithm
# from Sarah Johnson's Iterative Error Correction book.
# codeword[i] = log ( P(x=0) / P(x=1) )

import typing

from consts import FTX_LDPC_M
from consts import FTX_LDPC_N
from consts import kFTX_LDPC_Mn
from consts import kFTX_LDPC_Nm
from consts import kFTX_LDPC_Num_rows


# Ideas for approximating tanh/atanh:
# * https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
# * http://functions.wolfram.com/ElementaryFunctions/ArcTanh/10/0001/
# * https://mathr.co.uk/blog/2017-09-06_approximating_hyperbolic_tangent.html
# * https://math.stackexchange.com/a/446411

def fast_tanh(x: float) -> float:
    if x < -4.97:
        return -1.0

    if x > 4.97:
        return 1.0

    x2 = x ** 2
    a = x * (945.0 + x2 * (105.0 + x2))
    b = 945.0 + x2 * (420.0 + x2 * 15.0)
    return a / b


def fast_atanh(x: float) -> float:
    x2 = x ** 2
    a = x * (945.0 + x2 * (-735.0 + x2 * 64.0))
    b = (945.0 + x2 * (-1050.0 + x2 * 225.0))
    return a / b


# codeword is 174 log-likelihoods.
# plain is a return value, 174 ints, to be 0 or 1.
# max_iters is how hard to try.
# ok == 87 means success.
def ldpc_decode(codeword: typing.List[float], max_iters: int) -> typing.Tuple[int, bytes]:
    min_errors = FTX_LDPC_M

    m = [[0.0] * FTX_LDPC_N for _ in range(FTX_LDPC_M)]
    e = [[0.0] * FTX_LDPC_N for _ in range(FTX_LDPC_M)]

    plain = bytearray(b"\x00" * FTX_LDPC_N)

    for j in range(FTX_LDPC_M):
        for i in range(FTX_LDPC_N):
            m[j][i] = codeword[i]
            e[j][i] = 0.0

    for _ in range(max_iters):
        for j in range(FTX_LDPC_M):
            for ii1 in range(kFTX_LDPC_Num_rows[j]):
                i1 = kFTX_LDPC_Nm[j][ii1] - 1
                a = 1.0
                for ii2 in range(kFTX_LDPC_Num_rows[j]):
                    i2 = kFTX_LDPC_Nm[j][ii2] - 1
                    if i2 != i1:
                        a *= fast_tanh(-m[j][i2] / 2.0)

                e[j][i1] = -2.0 * fast_atanh(a)

        for i in range(FTX_LDPC_N):
            l = codeword[i]
            for j in range(3):
                l += e[kFTX_LDPC_Mn[i][j] - 1][i]

            plain[i] = int(l > 0)

        if (errors := ldpc_check(plain)) < min_errors:
            # Update the current best result
            min_errors = errors

            if errors == 0:
                break  # Found a perfect answer

        for i in range(FTX_LDPC_N):
            for ji1 in range(3):
                j1 = kFTX_LDPC_Mn[i][ji1] - 1
                l = codeword[i]
                for ji2 in range(3):
                    if ji1 != ji2:
                        j2 = kFTX_LDPC_Mn[i][ji2] - 1
                        l += e[j2][i]
                m[j1][i] = l

    return min_errors, plain


def ldpc_check(codeword: bytes) -> int:
    """
    does a 174-bit codeword pass the FT8's LDPC parity checks?
    :param codeword:
    :return: number of parity errors, 0 means total success.
    """
    errors = 0

    for m in range(FTX_LDPC_M):
        x = 0
        for i in range(kFTX_LDPC_Num_rows[m]):
            x ^= codeword[kFTX_LDPC_Nm[m][i] - 1]

        if x:
            errors += 1

    return errors


def bp_decode(codeword: typing.List[float], max_iters: int) -> typing.Tuple[int, bytes]:
    min_errors = FTX_LDPC_M

    # initialize message data
    tov = [[0.0] * 3 for _ in range(FTX_LDPC_N)]
    toc = [[0.0] * 7 for _ in range(FTX_LDPC_M)]

    plain = bytearray(b"\x00" * FTX_LDPC_N)

    for _ in range(max_iters):
        # Do a hard decision guess (tov=0 in iter 0)
        plain_sum = 0
        for n in range(FTX_LDPC_N):
            plain[n] = int((codeword[n] + tov[n][0] + tov[n][1] + tov[n][2]) > 0)
            plain_sum += plain[n]

        if plain_sum == 0:
            break  # message converged to all-zeros, which is prohibited

        # Check to see if we have a codeword (check before we do any iter)
        if (errors := ldpc_check(plain)) < min_errors:
            min_errors = errors  # we have a better guess - update the result

            if errors == 0:
                break  # Found a perfect answer

        # Send messages from bits to check nodes
        for m in range(FTX_LDPC_M):
            for n_idx in range(kFTX_LDPC_Num_rows[m]):
                n = kFTX_LDPC_Nm[m][n_idx] - 1
                Tnm = codeword[n]
                for m_idx in range(3):
                    if (kFTX_LDPC_Mn[n][m_idx] - 1) != m:
                        Tnm += tov[n][m_idx]
                toc[m][n_idx] = fast_tanh(-Tnm / 2)

        # send messages from check nodes to variable nodes
        for n in range(FTX_LDPC_N):
            for m_idx in range(3):
                m = kFTX_LDPC_Mn[n][m_idx] - 1
                Tmn = 1.0
                for n_idx in range(kFTX_LDPC_Num_rows[m]):
                    if (kFTX_LDPC_Nm[m][n_idx] - 1) != n:
                        Tmn *= toc[m][n_idx]
                tov[n][m_idx] = -2 * fast_atanh(Tmn)

    return min_errors, plain
