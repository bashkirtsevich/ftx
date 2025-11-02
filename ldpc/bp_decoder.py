import typing

import numpy as np
import numpy.typing as ntp
from numba import njit


@njit
def ldpc_check(
        codeword: ntp.NDArray[np.uint8],
        ldpc_m: int,
        ldpc_num_rows: ntp.NDArray[np.int64],
        ldpc_nm: ntp.NDArray[np.int64],
) -> int:
    errors = 0

    for m in range(ldpc_m):
        x = 0
        for i in range(ldpc_num_rows[m]):
            x ^= codeword[ldpc_nm[m][i] - 1]

        if x:
            errors += 1

    return errors


@njit
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

    ldpc_nm_idx = ldpc_nm - 1
    ldpc_mn_idx = ldpc_mn - 1

    for _ in range(max_iters):
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
        for m in range(ldpc_m):
            for n_idx in range(ldpc_num_rows[m]):
                n = ldpc_nm_idx[m][n_idx]

                total = 0.0
                for m_idx in range(n_v):
                    if ldpc_mn_idx[n][m_idx] != m:
                        total += tov[n][m_idx]

                Tnm = codeword[n] + total

                toc[m][n_idx] = np.tanh(-Tnm / 2)

        # send messages from check nodes to variable nodes
        for n in range(ldpc_n):
            for m_idx in range(3):
                m = ldpc_mn_idx[n][m_idx]

                Tmn = 1.0
                num_rows_m = ldpc_num_rows[m]
                for idx in range(num_rows_m):
                    if ldpc_nm_idx[m][idx] != n:
                        Tmn *= toc[m][idx]

                tov[n][m_idx] = -2 * np.atanh(Tmn)

    return min_errors, plain
