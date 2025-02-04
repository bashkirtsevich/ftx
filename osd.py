import math
import typing
from functools import partial, cmp_to_key

from consts import *
from crc import ftx_check_crc
from ldpc import ldpc_encode


# Performs Gauss-Jordan elimination on a binary matrix to find its inverse.
#
# The function manipulates the input matrix `m` of dimensions
# [FTX_LDPC_N][2*FTX_LDPC_K] (174x182) to compute the inverse of its left half,
# storing the result in its upper-right quarter. The `rows` parameter is the
# number of rows (91), and `cols` is the number of columns (174).
#
# The `which` array keeps track of column swaps to ensure the correct inverse is
# found. The initial right half of `m` should be zeros, and the function
# constructs an identity matrix in the upper-right quarter during execution.
#
# Returns 1 if the matrix is successfully inverted, or 0 if it is singular.
#
#
# int gauss_jordan(uint8_t m[FTX_LDPC_N][2 * FTX_LDPC_K], uint8_t which[FTX_LDPC_K])
def gauss_jordan(m: typing.List[typing.List[int]], which: typing.List[int]) -> bool:
    rows = FTX_LDPC_K
    cols = FTX_LDPC_N

    for row in range(rows):
        if m[row][row] != 1:
            for row1 in range(row + 1, cols):
                if m[row1][row] == 1:
                    # swap m[row] and m[row1]
                    for col in range(2 * rows):
                        temp = m[row][col]
                        m[row][col] = m[row1][col]
                        m[row1][col] = temp

                    tmp = which[row]
                    which[row] = which[row1]
                    which[row1] = tmp
                    break
        if m[row][row] != 1:
            # could not invert
            return False
        # lazy creation of identity matrix in the upper-right quarter
        m[row][rows + row] ^= 1
        # now eliminate
        for row1 in range(cols):
            if row1 == row:
                continue
            if m[row1][row] == 0:
                continue

            for col in range(2 * rows):
                m[row1][col] ^= m[row][col]

    return True


#  Score how well a given possible original codeword matches what was received.
#
#  @param xplain Possible original codeword (91 bits).
#  @param ll174 Received codeword (174 bits) with log-likelihoods.
#  @return A score, where higher means better match.
#
# osd_score(const uint8_t xplain[FTX_LDPC_K], const float ll174[FTX_LDPC_N])
def osd_score(xplain: typing.ByteString, ll174: typing.List[float]) -> float:
    # uint8_t xcode[FTX_LDPC_N];
    xcode = ldpc_encode(xplain)

    score = 0.0
    for i in range(FTX_LDPC_N):
        if xcode[i]:
            # one-bit, expect ll to be negative.
            score -= ll174[i]
        else:
            # zero-bit, expect ll to be positive.
            score += ll174[i]

    return score


#  @brief Checks the plausibility of a decoded message.
#
#  This function verifies whether a given decoded message, represented by
#  the 91-bit array `plain`, is plausible. It first checks if the message
#  is not all zeros. Then, it checks the message's CRC for validity.
#
#  @param[in] plain The 91-bit array representing the decoded message.
#
#  @return Returns 1 if the message is plausible (non-zero and valid CRC),
#          otherwise returns 0.
#
# osd_check(const uint8_t plain[FTX_LDPC_K])
def osd_check(plain: typing.ByteString) -> bool:
    for i in range(FTX_LDPC_K):
        if plain[i] != 0:
            return ftx_check_crc(plain)

    # all zeros
    return False


#  @brief Performs matrix multiplication and modulo operation.
#
#  This function multiplies a matrix 'a' with a vector 'b' and stores the result in vector 'c'.
#  The multiplication is followed by a modulo 2 operation on each resulting element.
#
#  @param[in] a A 2D array representing the matrix with dimensions [FTX_LDPC_K][FTX_LDPC_K].
#  @param[in] b A 1D array representing the vector with dimension [FTX_LDPC_K].
#  @param[out] c A 1D array representing the resulting vector after matrix multiplication and modulo operation.
#
# matmul(const uint8_t a[FTX_LDPC_K][FTX_LDPC_K], const uint8_t b[FTX_LDPC_K], uint8_t c[FTX_LDPC_K])
def matmul(a: typing.List[typing.List[int]], b: typing.List[int]) -> typing.ByteString:
    c = bytearray(b"\x00" * FTX_LDPC_K)

    for i in range(FTX_LDPC_K):
        sum = 0
        for j in range(FTX_LDPC_K):
            sum ^= a[i][j] & b[j]  # one bit multiply
        c[i] = sum

    return c


#  @brief qsort comparison function for sorting codeword indices.
#
#  Compares two indices 'a' and 'b' based on the absolute value of the elements in the
#  'codeword' array at positions 'a' and 'b'. The comparison is in descending order.
#
#  @param[in] a A pointer to the first index.
#  @param[in] b A pointer to the second index.
#  @param[in] c A pointer to the 'codeword' array.
#
#  @return A negative value if 'a' is greater than 'b', zero if equal, and a positive value if 'a' is less than 'b'.
def osd_cmp(a, b, c) -> int:
    # float* codeword = (float*)c;
    # uint8_t aa = *(uint8_t*)a;
    # uint8_t bb = *(uint8_t*)b;
    codeword = c
    aa = a
    bb = b

    fabs_a = math.fabs(codeword[aa])
    fabs_b = math.fabs(codeword[bb])

    # Reverse order (descending)
    if fabs_a > fabs_b:
        return -1
    if fabs_a < fabs_b:
        return 1
    return 0


#  @brief Ordered statistics decoder for LDPC and new FT8.
#
#  Decodes a received codeword using the ordered statistics decoder.
#  The decoder sorts the received codeword by strength of the bits,
#  and then reorders the columns of the generator matrix accordingly.
#  Then, it does a gaussian elimination to find a solution to the
#  system of linear equations.
#
#  @param[in] codeword The received codeword as an array of 174 log-likelihoods. codeword[i] = log ( P(x=0) / P(x=1) )
#  @param[in] depth The maximum number of bits to flip in the received codeword.
#  @param[out] out The decoded plain bits in an array of 91 bits.
#  @param[out] out_depth The actual number of bits that were flipped in the received codeword.
#
#  @return Returns 1 if the decode was successful, 0 otherwise.
# int osd_decode(const float codeword[FTX_LDPC_N], int depth, uint8_t out[FTX_LDPC_K], int* out_depth)
def osd_decode(codeword: typing.List[float], depth: int) -> typing.Optional[typing.Tuple[typing.ByteString, int]]:
    osd_thresh = -100.0

    # sort, strongest first; we'll use strongest 91.
    which = [i for i in range(FTX_LDPC_N)]
    which.sort(key=cmp_to_key(partial(osd_cmp, c=codeword)))
    # gen_sys[174 rows][91 cols] has a row per each of the 174 codeword bits,
    # indicating how to generate it by xor with each of the 91 plain bits.

    # generator matrix, reordered strongest codeword bit first.
    b = [[0] * FTX_LDPC_K * 2 for _ in range(FTX_LDPC_N)]
    for i in range(FTX_LDPC_N):
        ii = which[i]
        if ii < FTX_LDPC_K:
            b[i][ii] = 1
        else:
            kk = ii - FTX_LDPC_K
            for j in range(FTX_LDPC_K):
                b[i][j] = GEN_SYS[kk][j]

    if gauss_jordan(b, which) == 0:
        return None

    gen1_inv = [[b[i][FTX_LDPC_K + j] for j in range(FTX_LDPC_K)] for i in range(FTX_LDPC_K)]

    # y1 is the received bits, same order as gen1_inv,
    # more or less strongest-first, converted from
    # log-likihood to 0/1.
    y1 = [int(codeword[which[i]] > 0) for i in range(FTX_LDPC_K)]

    # can we decode without flipping any bits?
    xplain = matmul(gen1_inv, y1)  # also does mod 2

    xscore = osd_score(xplain, codeword)
    ch = osd_check(xplain)
    if xscore < osd_thresh and ch:
        # just accept this, since no bits had to be flipped.
        return xplain, 0

    # uint8_t best_plain[FTX_LDPC_K];
    best_plain = None
    best_score = 0.0
    got_a_best = 0
    best_depth = -1

    # flip a few bits, see if decode works.
    for ii in range(depth):
        i = FTX_LDPC_K - 1 - ii
        y1[i] ^= 1
        xplain = matmul(gen1_inv, y1)
        y1[i] ^= 1
        xscore = osd_score(xplain, codeword)
        ch = osd_check(xplain)
        if xscore < osd_thresh and ch:
            if got_a_best == 0 or xscore < best_score:
                got_a_best = 1
                best_plain = xplain
                best_score = xscore
                best_depth = ii

    if got_a_best:
        return best_plain, best_depth

    return None
