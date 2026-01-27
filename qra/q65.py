import typing

import numpy as np
import numpy.typing as npt
from functools import cache

from fwht import fwht
from utils.common import dB
from crc.q65 import crc6, crc12

from consts.q65 import *
from encoders.qra import qra_encode
from qra.exceptions import CRCMismatch, MExceeded, DecodeFailed
from qra.q65_codec import Q65Codec
from qra.qra_code_params import QRACodeParams, qra15_65_64_irr_e23


def pd_imul(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64], dim: int):
    idx = int(2 ** dim)
    dst[:idx] *= src[:idx]


@cache
def pd_uniform(dim: int) -> npt.NDArray[np.float64]:
    # define uniform distributions of given size
    return np.full(2 ** dim, 1 / 2 ** dim, dtype=np.float64)


def pd_norm_tab(data: npt.NDArray[np.float64], dim: int) -> float:
    dim = min(dim, qra_m)

    if dim == 0:
        t = data[0]
        data[0] = 1.0
        return t

    c = 2 ** dim
    t = np.sum(data[:c])

    if t <= 0:
        data[:c] = pd_uniform(dim)[:c]
        return t

    data *= 1 / t
    return t


def pd_norm(data: npt.NDArray[np.float64], dim: int) -> float:
    return pd_norm_tab(data, dim)


def pd_backward_permutation(
        dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64],
        perm: npt.NDArray[np.int64], dim: int):
    dst[perm[:dim]] = src[:dim]


def pd_forward_permutation(
        dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64],
        perm: npt.NDArray[np.int64], dim: int):
    dst[:dim] = src[perm[:dim]]


def qra_extrinsic(
        code: QRACodeParams,
        ex: npt.NDArray[np.float64],
        ix: npt.NDArray[np.float64],
        max_iter: int,
        qra_v2cmsg: npt.NDArray[np.float64],
        qra_c2vmsg: npt.NDArray[np.float64],
):
    qra_M = code.M
    qra_m = code.m
    qra_V = code.V
    qra_MAXVDEG = code.MAX_V_DEG
    qra_vdeg = code.v_deg
    qra_C = code.C
    qra_MAXCDEG = code.MAX_C_DEG
    qra_cdeg = code.c_deg
    qra_v2cmidx = code.v2cm_idx
    qra_c2vmidx = code.c2vm_idx
    qra_pmat = code.gfp_mat.reshape((-1, qra_M))
    qra_msgw = code.msg_w

    # float msg_out[QRACODE_MAX_M]; # we use a fixed size in order to avoid mallocs
    msg_out = np.zeros(QRACODE_MAX_M, dtype=np.float64)

    rc = -1  # rc>=0  extrinsic converged to 1 at iteration rc (rc=0..maxiter-1)
    # rc=-1  no convergence in the given number of iterations
    # rc=-2  error in the code tables (code checks degrees must be >1)
    # rc=-3  M is larger than QRACODE_MAX_M

    if qra_M > QRACODE_MAX_M:
        # return -3
        raise MExceeded

    # message initialization -------------------------------------------------------

    # init c->v variable intrinsic msgs
    qra_c2vmsg[:qra_V, :qra_M] = ix[:qra_V, :qra_M]

    # init the v->c messages directed to code factors (k=1..deg) with the intrinsic info
    for v in range(qra_V):  # current variable
        v_deg = qra_vdeg[v]  # degree of current node
        msg_base = v * qra_MAXVDEG  # base to msg index row for the current node

        # copy intrinsics on v->c
        for k in range(1, v_deg):
            idx_msg = qra_v2cmidx[msg_base + k]  # current message index
            qra_v2cmsg[idx_msg, :qra_M] = ix[v, :qra_M]

    # message passing algorithm iterations ------------------------------

    for iter in range(max_iter):  # current iteration
        # c->v step -----------------------------------------------------
        # Computes messages from code checks to code variables.
        # As the first qra_V checks are associated with intrinsic information
        # (the code tables have been constructed in this way)
        # we need to do this step only for code checks in the range [qra_V..qra_C)

        # The convolutions of probability distributions over the alphabet of a finite field GF(qra_M)
        # are performed with a fast convolution algorithm over the given field.
        #
        # I.e. given the code check x1+x2+x3 = 0 (with x1,x2,x3 in GF(2^m))
        # and given Prob(x2) and Prob(x3), we have that:
        # Prob(x1=X1) = Prob((x2+x3)=X1) = sum((Prob(x2=X2)*Prob(x3=(X1+X2))) for all the X2s in the field
        # This translates to Prob(x1) = IWHT(WHT(Prob(x2))*WHT(Prob(x3)))
        # where WHT and IWHT are the direct and inverse Walsh-Hadamard transforms of the argument.
        # Note that the WHT and the IWHF differs only by a multiplicative coefficent and since in this step
        # we don't need that the output distribution is normalized we use the relationship
        # Prob(x1) =(proportional to) WH(WH(Prob(x2))*WH(Prob(x3)))

        # In general given the check code x1+x2+x3+..+xm = 0
        # the output distribution of a variable given the distributions of the other m-1 variables
        # is the inverse WHT of the product of the WHTs of the distribution of the other m-1 variables
        # The complexity of this algorithm scales with M*log2(M) instead of the M^2 complexity of
        # the brute force approach (M=size of the alphabet)

        for nc in range(qra_V, qra_C):  # current check
            deg = qra_cdeg[nc]  # degree of current node

            if deg == 1:  # this should never happen (code factors must have deg>1)
                return -2  # bad code tables

            msg_base = nc * qra_MAXCDEG  # base to msg index row for the current node

            # transforms inputs in the Walsh-Hadamard "frequency" domain
            # v->c  -> fwht(v->c)
            for k in range(deg):
                idx_msg = qra_c2vmidx[msg_base + k]  # msg index
                fwht(qra_m, qra_v2cmsg[idx_msg, :], qra_v2cmsg[idx_msg, :])  # compute fwht

            # compute products and transform them back in the WH "time" domain
            for k in range(deg):  # loop indexes
                # init output message to uniform distribution
                msg_out[:qra_M] = pd_uniform(qra_m)[:qra_M]

                # c->v = prod(fwht(v->c))
                # TODO: we assume that checks degrees are not larger than three but
                # if they are larger the products can be computed more efficiently
                for kk in range(deg):  # loop indexes
                    if kk != k:
                        idx_msg = qra_c2vmidx[msg_base + kk]
                        pd_imul(msg_out, qra_v2cmsg[idx_msg, :], qra_m)

                # transform product back in the WH "time" domain

                # Very important trick:
                # we bias WHT[0] so that the sum of output pd components is always strictly positive
                # this helps avoiding the effects of underflows in the v->c steps when multipling
                # small fp numbers
                msg_out[0] += 1E-7  # TODO: define the bias accordingly to the field size

                fwht(qra_m, msg_out, msg_out)

                # inverse weight and output
                idx_msg = qra_c2vmidx[msg_base + k]  # current output msg index
                w_msg = qra_msgw[idx_msg]  # current msg weight

                if w_msg == 0:
                    qra_c2vmsg[idx_msg, :qra_M] = msg_out[:qra_M]
                else:
                    # output p(alfa^(-w)*x)
                    pd_backward_permutation(qra_c2vmsg[idx_msg, :], msg_out, qra_pmat[w_msg, :], qra_M)

        # v->c step -----------------------------------------------------
        for v in range(qra_V):
            deg = qra_vdeg[v]  # degree of current node
            msg_base = v * qra_MAXVDEG  # base to msg index row for the current node

            for k in range(deg):
                # init output message to uniform distribution
                msg_out[:qra_M] = pd_uniform(qra_m)[:qra_M]

                # v->c msg = prod(c->v)
                # TODO: factor factors to reduce the number of computations for high degree nodes
                for kk in range(deg):
                    if kk != k:
                        idx_msg = qra_v2cmidx[msg_base + kk]
                        pd_imul(msg_out, qra_c2vmsg[idx_msg, :], qra_m)

                # normalize the result to a probability distribution
                pd_norm(msg_out, qra_m)
                # weight and output
                idx_msg = qra_v2cmidx[msg_base + k]  # current output msg index
                w_msg = qra_msgw[idx_msg]  # current msg weight

                if w_msg == 0:
                    qra_v2cmsg[idx_msg, :qra_M] = msg_out[:qra_M]
                else:
                    # output p(alfa^w*x)
                    pd_forward_permutation(qra_v2cmsg[idx_msg, :qra_M], msg_out, qra_pmat[w_msg, :], qra_M)

        # check extrinsic information ------------------------------
        # We assume that decoding is successful if each of the extrinsic
        # symbol probability is close to ej, where ej = [0 0 0 1(j-th position) 0 0 0 ]
        # Therefore, for each symbol k in the codeword we compute max(prob(Xk))
        # and we stop the iterations if sum(max(prob(xk)) is close to the codeword length
        # Note: this is a more restrictive criterium than that of computing the a
        # posteriori probability of each symbol, making a hard decision and then check
        # if the codeword syndrome is null.
        # WARNING: this is tricky and probably works only for the particular class of RA codes
        # we are coping with (we designed the code weights so that for any input symbol the
        # sum of its weigths is always 0, thus terminating the accumulator trellis to zero
        # for every combination of the systematic symbols).
        # More generally we should instead compute the max a posteriori probabilities
        # (as a product of the intrinsic and extrinsic information), make a symbol by symbol hard
        # decision and then check that the syndrome of the result is indeed null.

        totex = 0  # total extrinsic information
        for v in range(qra_V):
            totex += np.max(qra_v2cmsg[v, :qra_M])

        if totex > (1.0 * (qra_V) - 0.01):
            # the total maximum extrinsic information of each symbol in the codeword
            # is very close to one. This means that we have reached the (1,1) point in the
            # code EXIT chart(s) and we have successfully decoded the input.
            rc = iter
            break  # remove the break to evaluate the decoder speed performance as a function of the max iterations number)

    # copy extrinsic information to output to do the actual max a posteriori prob decoding
    ex[:qra_V, :qra_M] = qra_v2cmsg[:qra_V, :qra_M]
    return rc


def qra_map_decode(code: QRACodeParams, x_dec: npt.NDArray[np.uint8],
                   ex: npt.NDArray[np.float64], ix: npt.NDArray[np.float64]):
    # Maximum a posteriori probability decoding.
    # Given the intrinsic information (pix) and extrinsic information (pex) (computed with qra_extrinsic(...))
    # compute pmap = pex*pix and decode each (information) symbol of the received codeword
    # as the symbol which maximizes pmap

    # Returns:
    #	xdec[k] = decoded (information) symbols k=[0..K-1]

    #  Note: pex is destroyed and overwritten with mapp

    M = code.M
    m = code.m
    K = code.K

    for k in range(K):
        # compute a posteriori prob
        pd_imul(ex[k, :], ix[k, :], m)
        x_dec[k] = np.argmax(ex[k, :M])


def q65_init() -> Q65Codec:
    qra_code = qra15_65_64_irr_e23
    # Eb/No value for which we optimize the decoder metric (AWGN/Rayleigh cases)
    EbNodBMetric = 2.8
    EbNoMetric = 10 ** (EbNodBMetric / 10)

    # compute and store the AWGN/Rayleigh Es/No ratio for which we optimize
    # the decoder metric
    nm = qra_code.bits_per_symbol
    R = qra_code.code_rate

    return Q65Codec(
        qra_code=qra_code,
        decoderEsNoMetric=1.0 * nm * R * EbNoMetric,
        x=np.zeros(qra_code.K, dtype=np.uint8),
        y=np.zeros(qra_code.N, dtype=np.uint8),
        qra_v2cmsg=np.zeros((qra_code.N_MSG, qra_code.M), dtype=np.float64),
        qra_c2vmsg=np.zeros((qra_code.N_MSG, qra_code.M), dtype=np.float64),
        ix=np.zeros((qra_code.N, qra_code.M), dtype=np.float64),
        ex=np.zeros((qra_code.N, qra_code.M), dtype=np.float64),

        BinsPerTone=0,
        BinsPerSymbol=0,
        NoiseVar=0,
        EsNoMetric=0,
        WeightsCount=0,
        FastFadingWeights=np.zeros(Q65_FASTFADING_MAXWEIGTHS, dtype=np.float64)
    )


def q65_fastfading_intrinsics(
        codec: Q65Codec,
        input_energies: npt.NDArray[np.float64],  # received energies input
        sub_mode: int,  # submode idx (0=A ... 4=E)
        B90Ts: float,  # spread bandwidth (90% fractional energy)
        fading_model: FadingModel  # 0=Gaussian 1=Lorentzian fade model
) -> npt.NDArray[np.float64]:
    # As the symbol duration in q65 is different than in QRA64,
    # the fading tables continue to be valid if the B90Ts parameter
    # is properly scaled to the QRA64 symbol interval
    # Compute index to most appropriate weighting function coefficients
    B90 = B90Ts / TS_QRA64

    # Unlike in QRA64 we accept any B90, anyway limiting it to
    # the extreme cases (0.9 to 210 Hz approx.)
    h_idx = int(np.log(B90) / np.log(1.09) - 0.499)
    h_idx = min(63, max(0, h_idx))

    # select the appropriate weighting fading coefficients array
    model = {
        FadingModel.Gaussian: fm_tab_gauss,  # gaussian fading model
        FadingModel.Lorentzian: fm_tab_lorentz,  # lorentzian fading model
    }[fading_model]

    weights = model[h_idx]
    weights_count = len(weights)  # weights_count = (L+1)/2 (where L=(odd) number of taps of w fun)

    # compute (heuristically) the optimal decoder metric accordingly the given spread amount
    # We assume that the decoder 50% decoding threshold is:
    # Es/No(dB) = Es/No(AWGN)(dB) + 8*log(B90)/log(240)(dB)
    # that's to say, at the maximum Doppler spread bandwidth (240 Hz for QRA64)
    # there's a ~8 dB Es/No degradation over the AWGN case
    EsNo_deg = 8.0 * np.log(B90) / np.log(240.0)  # assumed Es/No degradation for the given fading bandwidth
    EsNo_metric = codec.decoderEsNoMetric * np.pow(10.0, EsNo_deg / 10.0)

    M = codec.qra_code.alphabet_size
    N = codec.qra_code.codeword_length

    tone_span = 1 << sub_mode
    sym_span = M * (2 + tone_span)

    # In the fast fading case, the intrinsic probabilities can be computed only
    # if both the noise spectral density and the average Es/No ratio are known.

    # Assuming that the energy of a tone is spread, on average, over adjacent bins
    # with the weights given in the precomputed fast-fading tables, it turns out
    # that the probability that the transmitted tone was tone j when we observed
    # the energies En(1)...En(N) is:

    # prob(tone j| en1....enN) proportional to exp(sum(En(k,j)*w(k)/No))
    # where w(k) = (g(k)*Es/No)/(1 + g(k)*Es/No),
    # g(k) are constant coefficients given on the fading tables,
    # and En(k,j) denotes the Energy at offset k from the central bin of tone j

    # Therefore we:
    # 1) compute No - the noise spectral density (or noise variance)
    # 2) compute the coefficients w(k) given the coefficient g(k) for the given decoder Es/No metric
    # 3) compute the logarithm of prob(tone j| en1....enN) which is simply = sum(En(k,j)*w(k)/No
    # 4) subtract from the logarithm of the probabilities their maximum,
    # 5) exponentiate the logarithms
    # 6) normalize the result to a probability distribution dividing each value
    #    by the sum of all of them

    # Evaluate the average noise spectral density
    noise_var = np.mean(input_energies)
    # The noise spectral density so computed includes also the signal power.
    # Therefore we scale it accordingly to the Es/No assumed by the decoder
    noise_var = noise_var / (1.0 + EsNo_metric / sym_span)
    # The value so computed is an overestimate of the true noise spectral density
    # by the (unknown) factor (1+Es/No(true)/nBinsPerSymbol)/(1+EsNoMetric/nBinsPerSymbol)
    # We will take this factor in account when computing the true Es/No ratio

    # store in the pCodec structure for later use in the estimation of the Es/No ratio
    codec.NoiseVar = noise_var
    codec.EsNoMetric = EsNo_metric
    codec.BinsPerTone = tone_span
    codec.BinsPerSymbol = sym_span
    codec.WeightsCount = weights_count

    # compute the fast fading weights accordingly to the Es/No ratio
    # for which we compute the exact intrinsics probabilities
    EsNo_gain = EsNo_metric * weights
    weight = EsNo_gain / (EsNo_gain + 1) / noise_var
    codec.FastFadingWeights = weight

    # Compute now the intrinsics as indicated above
    sym_probs = np.zeros((N, M), dtype=np.float64)

    tap_half = weights_count - 1  # number of symmetric taps
    tap_center = 2 * tap_half  # index of the central tap

    for n in range(N):  # for each symbol in the message
        # compute the logarithm of the tone probability
        # as a weighted sum of the pertaining energies
        # M - point to the central bin of the first symbol tone
        sym_start = M - weights_count + 1  # point to the first bin of the current symbol

        for k in range(M):  # for each tone in the current symbol
            # do a symmetric weighted sum
            log_prob = 0
            for j in range(tap_half):
                log_prob += weight[j] * (
                        input_energies[n, sym_start + j] + input_energies[n, sym_start + tap_center - j]
                )

            log_prob += weight[tap_half] * input_energies[n, sym_start + tap_half]
            sym_probs[n, k] = log_prob

            sym_start += tone_span  # next tone

        log_prob_max = np.max(sym_probs[n])  # keep track of the max

        # exponentiate and accumulate the normalization constant
        sum_prob = 0
        for k in range(M):
            rel_log = sym_probs[n, k] - log_prob_max
            rel_log = min(85.0, max(-85.0, rel_log))

            rel_prob = np.exp(rel_log)
            sym_probs[n, k] = rel_prob
            sum_prob += rel_prob

        # scale to a probability distribution
        sym_probs[n, :] *= 1.0 / sum_prob

    return sym_probs


def q65_fastfading_EsNodB(
        codec: Q65Codec,
        y_dec: npt.NDArray[np.int64],
        input_energies: npt.NDArray[np.float64],
) -> float:
    # Estimate the Es/No ratio of the decoded codeword

    N = codec.qra_code.codeword_length
    M = codec.qra_code.alphabet_size

    bins_per_tone = codec.BinsPerTone
    bins_per_symbol = codec.BinsPerSymbol
    weights = codec.WeightsCount
    noise_var = codec.NoiseVar
    EsNo_metric = codec.EsNoMetric
    tot_weights = 2 * weights - 1

    # compute symbols energy (noise included) summing the
    # energies pertaining to the decoded symbols in the codeword

    EsPlusWNo = 0.0
    for n in range(N):
        cur_tone_idx = M + y_dec[n] * bins_per_tone  # point to the central bin of the current decoded symbol
        cur_bin_idx = cur_tone_idx - weights + 1  # point to first bin

        # sum over all the pertaining bins
        EsPlusWNo += np.sum(input_energies[n, cur_bin_idx: cur_bin_idx + tot_weights])

    EsPlusWNo = EsPlusWNo / N  # Es + nTotWeigths*No

    # The noise power noise_var computed in the q65_intrisics_fastading(...) function
    # is not the true noise power as it includes part of the signal energy.
    # The true noise variance is:
    # No = noise_var*(1+EsNoMetric/bins_per_symbol)/(1+EsNo/bins_per_symbol)

    # Therefore:
    # Es/No = EsPlusWNo/No - W = EsPlusWNo/noise_var*(1+Es/No/bins_per_symbol)/(1+Es/NoMetric/bins_per_symbol) - W
    # and:
    # Es/No*(1-u/bins_per_symbol) = u-W or Es/No = (u-W)/(1-u/bins_per_symbol)
    # where:
    # u = EsPlusNo/noise_var/(1+EsNoMetric/bins_per_symbol)

    u = EsPlusWNo / (noise_var * (1 + EsNo_metric / bins_per_symbol))
    u = max(u, tot_weights + 0.316)  # Limit the minimum Es/No to -5 dB approx.
    u = (u - tot_weights) / (1 - u / bins_per_symbol)  # linear scale Es/No

    EsNodB = dB(u)
    return EsNodB


# def q65_mask(qra_code: QRACodeParams, ix: npt.NDArray[np.float64], mask: npt.NDArray[np.int64],
#              x: npt.NDArray[np.int64]):
#     # mask intrinsic information ix with available a priori knowledge
#     M = qra_code.M
#     m = qra_code.m
#
#     # Exclude from masking the symbols which have been punctured.
#     # K is the length of the mask and x arrays, which do
#     # not include any punctured symbol
#     K = qra_code.message_length
#
#     # for each symbol set to zero the probability
#     # of the values which are not allowed by
#     # the a priori information
#     for k in range(K):
#         if s := mask[k]:
#             for kk in range(M):
#                 if (kk ^ x[k]) & s != 0:
#                     # This symbol value is not allowed
#                     # by the AP information
#                     # Set its probability to zero
#                     ix[k, kk] = 0.0
#
#             # normalize to a probability distribution
#             pd_norm(ix[k, :], m)


def q65_decode(
        codec: Q65Codec,
        sym_prob: npt.NDArray[np.float64],
        # APMask: npt.NDArray[np.int64],
        # APSymbols: npt.NDArray[np.int64],
        max_iters: int
):
    qra_code = codec.qra_code
    ix = codec.ix
    ex = codec.ex

    K = qra_code.message_length
    N = qra_code.codeword_length
    M = qra_code.M
    bits = qra_code.m

    x = codec.x
    y = codec.y

    # Depuncture intrinsics observations as required by the code type
    if qra_code.type == QRAType.CRC_PUNCTURED:
        ix[:K, :M] = sym_prob[:K, :M]

        uniform = pd_uniform(bits)
        ix[K, :M] = uniform[:M]  # crc

        ix[K + 1: K + 1 + N - K, :M] = sym_prob[K:K + N - K, :M]  # parity checks

    elif qra_code.type == QRAType.CRC_PUNCTURED2:
        ix[:K, :M] = sym_prob[:K, :M]

        uniform = pd_uniform(bits)
        ix[K, :M] = uniform[:M]  # crc
        ix[K + 1, :M] = uniform[:M]  # crc

        ix[K + 2: K + 2 + N - K, :M] = sym_prob[K: K + N - K, :M]  # parity checks

    else:
        # no puncturing
        ix[:K, :M] = sym_prob[:K, :M]  # as they are

    # mask the intrinsics with the available a priori knowledge
    # q65_mask(qra_code, ix, APMask, APSymbols)

    # Compute the extrinsic symbols probabilities with the message-passing algorithm
    # Stop if the extrinsics information does not converges to unity
    # within the given number of iterations
    rc = qra_extrinsic(qra_code, ex, ix, max_iters, codec.qra_v2cmsg, codec.qra_c2vmsg)
    if rc < 0:
        # failed to converge to a solution
        raise DecodeFailed

    # decode the information symbols (punctured information symbols included)
    qra_map_decode(qra_code, x, ex, ix)

    # verify CRC match
    if qra_code.type in (QRAType.CRC, QRAType.CRC_PUNCTURED):
        crc = crc6(x[:K])  # compute crc-6
        if crc != x[K]:
            raise CRCMismatch  # crc doesn't match

    elif qra_code.type == QRAType.CRC_PUNCTURED2:
        crc = crc12(x[:K])  # compute crc-12
        if (crc & 0x3F) != x[K] or (crc >> 6) != x[K + 1]:
            raise CRCMismatch  # crc doesn't match

    # copy the decoded msg to the user buffer (excluding punctured symbols)
    # decoded_msg[:K] = x[:K]
    decoded_msg = x[:K]

    # if (pDecodedCodeword==NULL)		# user is not interested in the decoded codeword
    #     return rc;					# return the number of iterations required to decode

    # crc matches therefore we can reconstruct the transmitted codeword
    #  reencoding the information available in x...

    # qra_encode(qra_code, y, x)
    y[:] = qra_encode(qra_code, x, concat=True)

    # ...and strip the punctured symbols from the codeword
    if qra_code.type == QRAType.CRC_PUNCTURED:
        # puncture crc-6 symbol
        decoded_codeword = np.concat([y[:K], y[K + 1:K + (N - K) + 1]])
    elif qra_code.type == QRAType.CRC_PUNCTURED2:
        # puncture crc-12 symbol
        decoded_codeword = np.concat([y[:K], y[K + 2:K + (N - K) + 2]])
    else:
        decoded_codeword = y[:N]  # no puncturing

    # return the number of iterations required to decode
    return decoded_codeword, decoded_msg


def q65_dec(
        codec: Q65Codec,
        sym_spectra: npt.NDArray[np.float64],  # [LL,NN] Symbol spectra
        sym_prob: npt.NDArray[np.float64],  # [LL,NN] Symbol-value intrinsic probabilities
        # APmask: npt.NDArray[np.int64],  # [13]  AP information to be used in decoding
        # APsymbols: npt.NDArray[np.int64],  # [13] Available AP informtion
        max_iter: int,
        # x_dec: npt.NDArray[np.int64],  # [13] Decoded 78-bit message as 13 six-bit integers
) -> typing.Tuple[float, npt.NDArray[np.uint8]]:  # Return code from q65_decode(); Estimated Es/No (dB)
    # rc, ydec, xdec = q65_decode(codec, s3_prob, APmask, APsymbols, max_iters)
    ydec, xdec = q65_decode(codec, sym_prob, max_iter)

    # rc = -1:  Invalid params
    # rc = -2:  Decode failed
    # rc = -3:  CRC mismatch
    # if (rc<0) return;

    EsNodB = q65_fastfading_EsNodB(codec, ydec, sym_spectra)
    # if (rc<0)
    #     printf("error in q65_esnodb_fastfading()\n");

    return EsNodB, xdec
