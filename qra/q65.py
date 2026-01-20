import typing

import numpy as np
import numpy.typing as npt

from fwht import fwht
from utils.common import dB
from crc.q65 import crc6, crc12

from consts.q65 import *
from encoders.qra import qra_encode
from qra.exceptions import InvalidFadingModel, CRCMismatch, MExceeded
from qra.q65_codec import Q65Codec
from qra.qra_code_params import QRACodeParams

qra15_65_64_irr_e23 = QRACodeParams(
    qra_K,
    qra_N,
    qra_m,
    qra_M,
    qra_a,
    # qra_NC, # FIXME: To be deleted
    qra_V,
    qra_C,
    qra_NMSG,
    qra_MAXVDEG,
    qra_MAXCDEG,
    QRAType.CRC_PUNCTURED2,
    qra_R,
    # CODE_NAME, # FIXME: To be deleted
    qra_acc_input_idx,
    qra_acc_input_wlog,
    qra_log,
    qra_exp,
    qra_msgw,
    qra_vdeg,
    qra_cdeg,
    qra_v2cmidx,
    qra_c2vmidx,
    qra_pmat
)


def q65_init() -> Q65Codec:
    qra_code = qra15_65_64_irr_e23
    # Eb/No value for which we optimize the decoder metric (AWGN/Rayleigh cases)
    EbNodBMetric = 2.8
    EbNoMetric = 10 ** (EbNodBMetric / 10)

    # compute and store the AWGN/Rayleigh Es/No ratio for which we optimize
    # the decoder metric
    nm = qra_code.bits_per_symbol
    R = qra_code.code_rate

    codec = Q65Codec(
        qra_code=qra_code,
        decoderEsNoMetric=1.0 * nm * R * EbNoMetric,
        x=np.zeros(qra_code.K, dtype=np.int64),
        y=np.zeros(qra_code.N, dtype=np.int64),
        qra_v2cmsg=np.zeros((qra_code.NMSG, qra_code.M), dtype=np.float64),
        qra_c2vmsg=np.zeros((qra_code.NMSG, qra_code.M), dtype=np.float64),
        ix=np.zeros((qra_code.N, qra_code.M), dtype=np.float64),
        ex=np.zeros((qra_code.N, qra_code.M), dtype=np.float64),

        BinsPerTone=0,
        BinsPerSymbol=0,
        NoiseVar=0,
        EsNoMetric=0,
        WeightsCount=0,
        FastFadingWeights=np.zeros(Q65_FASTFADING_MAXWEIGTHS, dtype=np.float64)
    )
    return codec


def pd_uniform(log_dim: int) -> npt.NDArray[np.float64]:
    return pd_uniform_tab[log_dim]


def pd_imul(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64], log_dim: int):
    idx = int(2 ** log_dim)
    dst[:idx] *= src[:idx]


def pd_norm_tab(ppd: npt.NDArray[np.float64], c0: int) -> float:
    c0 = min(c0, 6)  # FIXME: Use named const
    if c0 == 0:
        t = ppd[0]
        ppd[0] = 1.0
        return t

    c1 = np.pow(2, c0)
    t = np.sum(ppd[:c1])

    if t <= 0:
        dim = pd_log2dim[c0]
        ppd[:dim] = pd_uniform(c0)[:dim]
        return t

    ppd *= 1 / t
    return t


def pd_norm(pd: npt.NDArray[np.float64], nlogdim: int) -> float:
    return pd_norm_tab(pd, nlogdim)


def pd_bwdperm(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64], perm: npt.NDArray[np.int64], ndim: int):
    for i in range(ndim):
        dst[perm[i]] = src[i]


def pd_fwdperm(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64], perm: npt.NDArray[np.int64], ndim: int):
    for i in range(ndim):
        dst[i] = src[perm[i]]


def q65_intrinsics_fastfading(
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
    if fading_model == FadingModel.Gaussian:
        # gaussian fading model
        # fm_tab_len = glen_tab_gauss[h_idx]  # fm_tab_len = (L+1)/2 (where L=(odd) number of taps of w fun)
        weights = fm_tab_gauss[h_idx]  # pointer to the first (L+1)/2 coefficients of w fun
    elif fading_model == FadingModel.Lorentzian:
        # lorentzian fading model
        # point to lorentzian energy weighting taps
        weights = fm_tab_lorentz[h_idx]  # pointer to the first (L+1)/2 coefficients of w funfun)

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


def q65_esnodb_fastfading(
        codec: Q65Codec,
        y_dec: npt.NDArray[np.int64],
        input_energies: npt.NDArray[np.float64],
) -> float:
    # Estimate the Es/No ratio of the decoded codeword

    qra_N = codec.qra_code.codeword_length
    qra_M = codec.qra_code.alphabet_size

    bins_per_tone = codec.BinsPerTone
    bins_per_symbol = codec.BinsPerSymbol
    weights = codec.WeightsCount
    noise_var = codec.NoiseVar
    EsNo_metric = codec.EsNoMetric
    tot_weights = 2 * weights - 1

    # compute symbols energy (noise included) summing the
    # energies pertaining to the decoded symbols in the codeword

    EsPlusWNo = 0.0
    for n in range(qra_N):
        cur_tone_idx = qra_M + y_dec[n] * bins_per_tone  # point to the central bin of the current decoded symbol
        cur_bin_idx = cur_tone_idx - weights + 1  # point to first bin

        # sum over all the pertaining bins
        EsPlusWNo += np.sum(input_energies[n, cur_bin_idx: cur_bin_idx + tot_weights])

    EsPlusWNo = EsPlusWNo / qra_N  # Es + nTotWeigths*No

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


def q65_mask(qra_code: QRACodeParams, ix: npt.NDArray[np.float64], mask: npt.NDArray[np.int64],
             x: npt.NDArray[np.int64]):
    # mask intrinsic information ix with available a priori knowledge
    M = qra_code.M
    m = qra_code.m

    # Exclude from masking the symbols which have been punctured.
    # K is the length of the mask and x arrays, which do
    # not include any punctured symbol
    K = qra_code.message_length

    # for each symbol set to zero the probability
    # of the values which are not allowed by
    # the a priori information
    for k in range(K):
        if s := mask[k]:
            for kk in range(M):
                if (kk ^ x[k]) & s != 0:
                    # This symbol value is not allowed
                    # by the AP information
                    # Set its probability to zero
                    ix[k, kk] = 0.0

            # normalize to a probability distribution
            pd_norm(ix[k, :], m)


def qra_extrinsic(
        qra_code: QRACodeParams,
        ex: npt.NDArray[np.float64],
        ix: npt.NDArray[np.float64],
        max_iter: int,
        qra_v2cmsg: npt.NDArray[np.float64],
        qra_c2vmsg: npt.NDArray[np.float64],
):
    qra_M = qra_code.M
    qra_m = qra_code.m
    qra_V = qra_code.V
    qra_MAXVDEG = qra_code.MAXVDEG
    qra_vdeg = qra_code.vdeg
    qra_C = qra_code.C
    qra_MAXCDEG = qra_code.MAXCDEG
    qra_cdeg = qra_code.cdeg
    qra_v2cmidx = qra_code.v2cmidx
    qra_c2vmidx = qra_code.c2vmidx
    qra_pmat = qra_code.gfpmat.reshape((-1, qra_M))
    qra_msgw = qra_code.msgw

    # float msgout[QRACODE_MAX_M]; # we use a fixed size in order to avoid mallocs
    msgout = np.zeros(QRACODE_MAX_M, dtype=np.float64)

    rc = -1  # rc>=0  extrinsic converged to 1 at iteration rc (rc=0..maxiter-1)
    # rc=-1  no convergence in the given number of iterations
    # rc=-2  error in the code tables (code checks degrees must be >1)
    # rc=-3  M is larger than QRACODE_MAX_M

    if qra_M > QRACODE_MAX_M:
        raise MExceeded

    # message initialization -------------------------------------------------------

    # init c->v variable intrinsic msgs
    qra_c2vmsg[:qra_V, :qra_M] = ix[:qra_V, :qra_M]

    # init the v->c messages directed to code factors (k=1..ndeg) with the intrinsic info
    for nv in range(qra_V):  # current variable
        ndeg = qra_vdeg[nv]  # degree of current node
        msgbase = nv * qra_MAXVDEG  # base to msg index row for the current node

        # copy intrinsics on v->c
        for k in range(1, ndeg):
            msg_idx = qra_v2cmidx[msgbase + k]  # current message index
            qra_v2cmsg[msg_idx, :qra_M] = ix[nv, :qra_M]

    # message passing algorithm iterations ------------------------------

    for nit in range(max_iter):  # current iteration
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
            ndeg = qra_cdeg[nc]  # degree of current node

            if ndeg == 1:  # this should never happen (code factors must have deg>1)
                return -2  # bad code tables

            msgbase = nc * qra_MAXCDEG  # base to msg index row for the current node

            # transforms inputs in the Walsh-Hadamard "frequency" domain
            # v->c  -> fwht(v->c)
            for k in range(ndeg):
                msg_idx = qra_c2vmidx[msgbase + k]  # msg index
                fwht(qra_m, qra_v2cmsg[msg_idx, :], qra_v2cmsg[msg_idx, :])  # compute fwht

            # compute products and transform them back in the WH "time" domain
            for k in range(ndeg):  # loop indexes
                # init output message to uniform distribution
                msgout[:qra_M] = pd_uniform(qra_m)[:qra_M]

                # c->v = prod(fwht(v->c))
                # TODO: we assume that checks degrees are not larger than three but
                # if they are larger the products can be computed more efficiently
                for kk in range(ndeg):  # loop indexes
                    if kk != k:
                        msg_idx = qra_c2vmidx[msgbase + kk]
                        pd_imul(msgout, qra_v2cmsg[msg_idx, :], qra_m)

                # transform product back in the WH "time" domain

                # Very important trick:
                # we bias WHT[0] so that the sum of output pd components is always strictly positive
                # this helps avoiding the effects of underflows in the v->c steps when multipling
                # small fp numbers
                msgout[0] += 1E-7  # TODO: define the bias accordingly to the field size

                fwht(qra_m, msgout, msgout)

                # inverse weight and output
                msg_idx = qra_c2vmidx[msgbase + k]  # current output msg index
                wmsg = qra_msgw[msg_idx]  # current msg weight

                if wmsg == 0:
                    qra_c2vmsg[msg_idx, :qra_M] = msgout[:qra_M]
                else:
                    # output p(alfa^(-w)*x)
                    pd_bwdperm(qra_c2vmsg[msg_idx, :], msgout, qra_pmat[wmsg, :], qra_M)

        # v->c step -----------------------------------------------------
        for nv in range(qra_V):
            ndeg = qra_vdeg[nv]  # degree of current node
            msgbase = nv * qra_MAXVDEG  # base to msg index row for the current node

            for k in range(ndeg):
                # init output message to uniform distribution
                msgout[:qra_M] = pd_uniform(qra_m)[:qra_M]

                # v->c msg = prod(c->v)
                # TODO: factor factors to reduce the number of computations for high degree nodes
                for kk in range(ndeg):
                    if kk != k:
                        msg_idx = qra_v2cmidx[msgbase + kk]
                        pd_imul(msgout, qra_c2vmsg[msg_idx, :], qra_m)

                # normalize the result to a probability distribution
                pd_norm(msgout, qra_m)
                # weight and output
                msg_idx = qra_v2cmidx[msgbase + k]  # current output msg index
                wmsg = qra_msgw[msg_idx]  # current msg weight

                if wmsg == 0:
                    qra_v2cmsg[msg_idx, :qra_M] = msgout[:qra_M]
                else:
                    # output p(alfa^w*x)
                    pd_fwdperm(qra_v2cmsg[msg_idx, :qra_M], msgout, qra_pmat[wmsg, :], qra_M)

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
        for nv in range(qra_V):
            totex += np.max(qra_v2cmsg[nv, :qra_M])

        if totex > (1.0 * (qra_V) - 0.01):
            # the total maximum extrinsic information of each symbol in the codeword
            # is very close to one. This means that we have reached the (1,1) point in the
            # code EXIT chart(s) and we have successfully decoded the input.
            rc = nit
            break  # remove the break to evaluate the decoder speed performance as a function of the max iterations number)

    # copy extrinsic information to output to do the actual max a posteriori prob decoding
    ex[:qra_V, :qra_M] = qra_v2cmsg[:qra_V, :qra_M]
    return rc


def qra_mapdecode(pcode: QRACodeParams, xdec: npt.NDArray[np.int64], pex: npt.NDArray[np.float64],
                  pix: npt.NDArray[np.float64]):
    # Maximum a posteriori probability decoding.
    # Given the intrinsic information (pix) and extrinsic information (pex) (computed with qra_extrinsic(...))
    # compute pmap = pex*pix and decode each (information) symbol of the received codeword
    # as the symbol which maximizes pmap

    # Returns:
    #	xdec[k] = decoded (information) symbols k=[0..qra_K-1]

    #  Note: pex is destroyed and overwritten with mapp

    qra_M = pcode.M
    qra_m = pcode.m
    qra_K = pcode.K

    for k in range(qra_K):
        # compute a posteriori prob
        pd_imul(pex[k, :], pix[k, :], qra_m)
        xdec[k] = np.argmax(pex[k, :qra_M])


def q65_decode(
        codec: Q65Codec,
        intrinsics: npt.NDArray[np.float64],
        APMask: npt.NDArray[np.int64],
        APSymbols: npt.NDArray[np.int64],
        max_iters: int
):
    qra_code = codec.qra_code
    ix = codec.ix
    ex = codec.ex

    nK = qra_code.message_length
    nN = qra_code.codeword_length
    nM = qra_code.M
    nBits = qra_code.m

    px = codec.x
    py = codec.y

    # Depuncture intrinsics observations as required by the code type
    if qra_code.type == QRAType.CRC_PUNCTURED:
        ix[:nK, :nM] = intrinsics[:nK, :nM]

        uniform = pd_uniform(nBits)
        ix[nK, :nM] = uniform[:nM]  # crc

        ix[nK + 1: nK + 1 + nN - nK, :nM] = intrinsics[nK:nK + nN - nK, :nM]  # parity checks

    elif qra_code.type == QRAType.CRC_PUNCTURED2:
        ix[:nK, :nM] = intrinsics[:nK, :nM]

        uniform = pd_uniform(nBits)
        ix[nK, :nM] = uniform[:nM]  # crc
        ix[nK + 1, :nM] = uniform[:nM]  # crc

        ix[nK + 2: nK + 2 + nN - nK, :nM] = intrinsics[nK: nK + nN - nK, :nM]  # parity checks

    else:
        # no puncturing
        ix[:nK, :nM] = intrinsics[:nK, :nM]  # as they are

    # mask the intrinsics with the available a priori knowledge
    q65_mask(qra_code, ix, APMask, APSymbols)

    # Compute the extrinsic symbols probabilities with the message-passing algorithm
    # Stop if the extrinsics information does not converges to unity
    # within the given number of iterations
    rc = qra_extrinsic(
        qra_code,
        ex,
        ix,
        max_iters,
        codec.qra_v2cmsg,
        codec.qra_c2vmsg
    )
    if rc < 0:
        # failed to converge to a solution
        # return Q65_DECODE_FAILED
        raise Exception("Q65_DECODE_FAILED")

    # decode the information symbols (punctured information symbols included)
    qra_mapdecode(qra_code, px, ex, ix)

    # verify CRC match
    if qra_code.type in (QRAType.CRC, QRAType.CRC_PUNCTURED):
        crc = crc6(px[:nK])  # compute crc-6
        if crc != px[nK]:
            raise CRCMismatch  # crc doesn't match

    elif qra_code.type == QRAType.CRC_PUNCTURED2:
        crc = crc12(px[:nK])  # compute crc-12
        if (crc & 0x3F) != px[nK] or (crc >> 6) != px[nK + 1]:
            raise CRCMismatch  # crc doesn't match

    # copy the decoded msg to the user buffer (excluding punctured symbols)
    # decoded_msg[:nK] = px[:nK]
    decoded_msg = px[:nK].copy()

    # if (pDecodedCodeword==NULL)		# user is not interested in the decoded codeword
    #     return rc;					# return the number of iterations required to decode

    # crc matches therefore we can reconstruct the transmitted codeword
    #  reencoding the information available in px...

    # qra_encode(qra_code, py, px)
    py[:] = qra_encode(px, concat=True)

    # ...and strip the punctured symbols from the codeword
    if qra_code.type == QRAType.CRC_PUNCTURED:
        # puncture crc-6 symbol
        decoded_codeword = np.concat([py[:nK], py[nK + 1:nK + (nN - nK) + 1]])
    elif qra_code.type == QRAType.CRC_PUNCTURED2:
        # puncture crc-12 symbol
        decoded_codeword = np.concat([py[:nK], py[nK + 2:nK + (nN - nK) + 2]])
    else:
        decoded_codeword = py[:nN].copy()  # no puncturing

    # return the number of iterations required to decode
    return rc, decoded_codeword, decoded_msg


def q65_dec(
        codec: Q65Codec,
        s3: npt.NDArray[np.float64],  # [LL,NN] Symbol spectra
        s3_prob: npt.NDArray[np.float64],  # [LL,NN] Symbol-value intrinsic probabilities
        APmask: npt.NDArray[np.int64],  # [13]  AP information to be used in decoding
        APsymbols: npt.NDArray[np.int64],  # [13] Available AP informtion

        max_iters: int,
        # x_dec: npt.NDArray[np.int64],  # [13] Decoded 78-bit message as 13 six-bit integers
) -> typing.Tuple[int, float, npt.NDArray[np.int64]]:  # Return code from q65_decode(); Estimated Es/No (dB)
    rc, ydec, xdec = q65_decode(codec, s3_prob, APmask, APsymbols, max_iters)

    # rc = -1:  Invalid params
    # rc = -2:  Decode failed
    # rc = -3:  CRC mismatch
    # if (rc<0) return;

    esnodb = q65_esnodb_fastfading(codec, ydec, s3)
    # if (rc<0)
    #     printf("error in q65_esnodb_fastfading()\n");

    return rc, esnodb, xdec[:13]
