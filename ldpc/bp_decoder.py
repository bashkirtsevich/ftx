import typing

import numpy as np
import numpy.typing as ntp
from numba import jit


@jit(nopython=True)
def ldpc_check(
        codeword: ntp.NDArray[np.uint8],
        ldpc_m: int,
        ldpc_num_rows: ntp.NDArray[np.int64],
        ldpc_nm: ntp.NDArray[np.int64],
) -> int:
    errors = 0

    for m in np.arange(ldpc_m):
        x = 0
        for i in np.arange(ldpc_num_rows[m]):
            x ^= codeword[ldpc_nm[m][i] - 1]

        if x:
            errors += 1

    return errors


@jit(nopython=True)
def belief_propagation(
        codeword: ntp.NDArray[np.float64], max_iters: int,
        ldpc_n: int, ldpc_m: int,
        n_v: int, m_c: int,
        ldpc_num_rows: ntp.NDArray[np.int64],
        ldpc_nm: ntp.NDArray[np.int64],
        ldpc_mn: ntp.NDArray[np.int64],
) -> typing.Tuple[np.int64, ntp.NDArray[np.int64]]:
    min_errors = ldpc_m

    # initialize message data
    tov = np.zeros((ldpc_n, n_v), dtype=np.float64)
    toc = np.zeros((ldpc_m, m_c), dtype=np.float64)

    for _ in np.arange(max_iters):
        # Do a hard decision guess (tov=0 in iter 0)
        plain = (codeword + np.sum(tov, axis=1) > 0).astype(np.uint8)
        plain_sum = np.sum(plain)

        if plain_sum == 0:
            min_errors = ldpc_m
            break  # message converged to all-zeros, which is prohibited

        # Check to see if we have a codeword (check before we do any iter)
        errors = ldpc_check(plain, ldpc_m=ldpc_m, ldpc_num_rows=ldpc_num_rows, ldpc_nm=ldpc_nm)
        if errors < min_errors:
            min_errors = errors  # we have a better guess - update the result

            if errors == 0:
                break  # Found a perfect answer

        # Send messages from bits to check nodes
        for m in np.arange(ldpc_m):
            for n_idx in np.arange(ldpc_num_rows[m]):
                n = ldpc_nm[m][n_idx] - 1
                Tnm = codeword[n]
                for m_idx in np.arange(3):
                    if (ldpc_mn[n][m_idx] - 1) != m:
                        Tnm += tov[n][m_idx]
                toc[m][n_idx] = np.tanh(-Tnm / 2)

        # send messages from check nodes to variable nodes
        for n in np.arange(ldpc_n):
            for m_idx in np.arange(3):
                m = ldpc_mn[n][m_idx] - 1
                Tmn = 1.0
                for n_idx in np.arange(ldpc_num_rows[m]):
                    if (ldpc_nm[m][n_idx] - 1) != n:
                        Tmn *= toc[m][n_idx]
                tov[n][m_idx] = -2 * np.atanh(Tmn)

    return min_errors, plain
