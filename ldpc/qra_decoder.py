import json
import math
import typing
from dataclasses import dataclass
from functools import partial

import numpy as np
from numba import jit
import numpy.typing as npt
from crc.q65 import crc6, crc12

from consts.q65 import *
from encoders.qra import qra_encode


class QRAException(Exception):
    ...


class InvalidQRAType(QRAException):
    ...


class InvalidFadingModel(QRAException):
    ...


@dataclass
class QRACode:
    # code parameters
    K: int  # number of information symbols
    N: int  # codeword length in symbols
    m: int  # bits/symbol
    M: int  # Symbol alphabet cardinality (2^m)
    a: int  # code grouping factor
    NC: int  # number of check symbols (N-K)
    V: int  # number of variables in the code graph (N)
    C: int  # number of factors in the code graph (N +(N-K)+1)
    NMSG: int  # number of msgs in the code graph
    MAXVDEG: int  # maximum variable degree
    MAXCDEG: int  # maximum factor degree
    type: int  # see QRATYPE_xx defines # FIXME: use enum
    R: float  # code rate (K/N)
    name: str  # code name
    # tables used by the encoder
    acc_input_idx: npt.NDArray[np.int64]
    acc_input_wlog: npt.NDArray[np.int64]
    gflog: npt.NDArray[np.int64]
    gfexp: npt.NDArray[np.int64]
    # tables used by the decoder -------------------------
    msgw: npt.NDArray[np.int64]
    vdeg: npt.NDArray[np.int64]
    cdeg: npt.NDArray[np.int64]
    v2cmidx: npt.NDArray[np.int64]
    c2vmidx: npt.NDArray[np.int64]
    gfpmat: npt.NDArray[np.int64]

    def q65_get_bits_per_symbol(self):
        return self.m

    def q65_get_code_rate(self):
        return 1.0 * self.q65_get_message_length() / self.q65_get_codeword_length()

    def q65_get_message_length(self):
        # return the actual information message length (in symbols)
        # excluding any punctured symbol

        if self.type == QRATYPE_NORMAL:
            return self.K
        elif self.type in (QRATYPE_CRC, QRATYPE_CRCPUNCTURED):
            # one information symbol of the underlying qra code is reserved for CRC
            return self.K - 1
        elif self.type == QRATYPE_CRCPUNCTURED2:
            # two code information symbols are reserved for CRC
            return self.K - 2

        raise InvalidQRAType

    def q65_get_codeword_length(self):
        # return the actual codeword length (in symbols)
        # excluding any punctured symbol

        if self.type in (QRATYPE_NORMAL, QRATYPE_CRC):
            # no puncturing
            return self.N
        elif self.type == QRATYPE_CRCPUNCTURED:
            # the CRC symbol is punctured
            return self.N - 1
        elif self.type == QRATYPE_CRCPUNCTURED2:
            # the two CRC symbols are punctured
            return self.N - 2

        raise InvalidQRAType

    def q65_get_alphabet_size(self):
        return self.M


qra15_65_64_irr_e23 = QRACode(
    qra_K,
    qra_N,
    qra_m,
    qra_M,
    qra_a,
    qra_NC,
    qra_V,
    qra_C,
    qra_NMSG,
    qra_MAXVDEG,
    qra_MAXCDEG,
    QRATYPE_CRCPUNCTURED2,
    qra_R,
    CODE_NAME,
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

Q65_FASTFADING_MAXWEIGTHS = 65


@dataclass
class q65_codec_ds:
    pQraCode: QRACode  # qra code to be used by the codec
    decoderEsNoMetric: float  # value for which we optimize the decoder metric
    x: npt.NDArray[np.int64]  # codec input
    y: npt.NDArray[np.int64]  # codec output
    qra_v2cmsg: npt.NDArray[np.float64]  # decoder v->c messages
    qra_c2vmsg: npt.NDArray[np.float64]  # decoder c->v messages
    ix: npt.NDArray[np.float64]  # decoder intrinsic information
    ex: npt.NDArray[np.float64]  # decoder extrinsic information
    # variables used to compute the intrinsics in the fast-fading case
    nBinsPerTone: int
    nBinsPerSymbol: int
    ffNoiseVar: float
    ffEsNoMetric: float
    nWeights: int
    ffWeight: npt.NDArray[np.float64]  # len = Q65_FASTFADING_MAXWEIGTHS


def np_fwht(nlogdim: int, dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64]):
    return np_fwht_tab[nlogdim](dst, src)


def pd_uniform(nlogdim: int) -> npt.NDArray[np.float64]:
    return pd_uniform_tab[nlogdim]


def pd_init(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64], ndim: int):
    dst[:ndim] = src[:ndim]


def PD_ROWADDR(arr: npt.NDArray[np.float64], ndim: int, idx: int) -> npt.NDArray[np.float64]:
    return arr[ndim * idx:]


def pd_imul(dst: npt.NDArray[np.float64], src: npt.NDArray[np.float64], nlogdim: int):
    idx = int(2 ** nlogdim)
    dst[:idx] *= src[:idx]


def pd_max(src: npt.NDArray[np.float64], ndim: int) -> float:
    return np.max(src[:ndim])


def pd_argmax(src: npt.NDArray[np.float64], ndim: int) -> int:
    return np.argmax(src[:ndim])


def q65_init():
    pqracode = qra15_65_64_irr_e23
    # Eb/No value for which we optimize the decoder metric (AWGN/Rayleigh cases)
    EbNodBMetric = 2.8
    EbNoMetric = 10 ** (EbNodBMetric / 10)

    # compute and store the AWGN/Rayleigh Es/No ratio for which we optimize
    # the decoder metric
    nm = pqracode.q65_get_bits_per_symbol()
    R = pqracode.q65_get_code_rate()

    pCodec = q65_codec_ds(
        pQraCode=pqracode,
        decoderEsNoMetric=1.0 * nm * R * EbNoMetric,
        x=np.zeros(pqracode.K, dtype=np.int64),
        y=np.zeros(pqracode.N, dtype=np.int64),
        qra_v2cmsg=np.zeros(pqracode.NMSG * pqracode.M, dtype=np.float64),
        qra_c2vmsg=np.zeros(pqracode.NMSG * pqracode.M, dtype=np.float64),
        ix=np.zeros(pqracode.N * pqracode.M, dtype=np.float64),
        ex=np.zeros(pqracode.N * pqracode.M, dtype=np.float64),

        nBinsPerTone=0,
        nBinsPerSymbol=0,
        ffNoiseVar=0,
        ffEsNoMetric=0,
        nWeights=0,
        ffWeight=np.zeros(Q65_FASTFADING_MAXWEIGTHS, dtype=np.float64)
    )
    return pCodec


def pd_norm_tab(ppd: npt.NDArray[np.float64], c0: int) -> float:
    c0 = min(c0, 6)  # FIXME: Use named const
    if c0 == 0:
        t = ppd[0]
        ppd[0] = 1.0
        return t

    c1 = np.pow(2, c0)
    t = np.sum(ppd[:c1])

    if t <= 0:
        pd_init(ppd, pd_uniform(c0), pd_log2dim[c0])
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
        codec: q65_codec_ds,
        # pIntrinsics: npt.NDArray[np.float64], # intrinsic symbol probabilities output
        input_energies: npt.NDArray[np.float64],  # received energies input
        sub_mode: int,  # submode idx (0=A ... 4=E)
        B90Ts: float,  # spread bandwidth (90% fractional energy)
        fading_model: int  # 0=Gaussian 1=Lorentzian fade model
):  # -> npt.NDArray[np.float64]:
    # As the symbol duration in q65 is different than in QRA64,
    # the fading tables continue to be valid if the B90Ts parameter
    # is properly scaled to the QRA64 symbol interval
    # Compute index to most appropriate weighting function coefficients
    B90 = B90Ts / TS_QRA64
    hidx = int(np.log(B90) / np.log(1.09) - 0.499)

    # Unlike in QRA64 we accept any B90, anyway limiting it to
    # the extreme cases (0.9 to 210 Hz approx.)
    hidx = min(63, max(0, hidx))

    # select the appropriate weighting fading coefficients array
    if fading_model == 0:
        # gaussian fading model
        # point to gaussian energy weighting taps
        hlen = glen_tab_gauss[hidx]  # hlen = (L+1)/2 (where L=(odd) number of taps of w fun)
        hptr = gptr_tab_gauss[hidx]  # pointer to the first (L+1)/2 coefficients of w fun
    elif fading_model == 1:
        # point to lorentzian energy weighting taps
        hlen = glen_tab_lorentz[hidx]  # hlen = (L+1)/2 (where L=(odd) number of taps of w fun)
        hptr = gptr_tab_lorentz[hidx]  # pointer to the first (L+1)/2 coefficients of w fun
    else:
        raise InvalidFadingModel

    # compute (euristically) the optimal decoder metric accordingly the given spread amount
    # We assume that the decoder 50% decoding threshold is:
    # Es/No(dB) = Es/No(AWGN)(dB) + 8*log(B90)/log(240)(dB)
    # that's to say, at the maximum Doppler spread bandwidth (240 Hz for QRA64)
    # there's a ~8 dB Es/No degradation over the AWGN case
    fTemp = 8.0 * np.log(B90) / np.log(240.0)  # assumed Es/No degradation for the given fading bandwidth
    # EsNoMetric = pCodec->decoderEsNoMetric * 10.0 ** (fTemp/10.0)

    nm = 6
    R = 0.206349
    decoderEsNoMetric = 1.0 * nm * R * EbNoMetric
    EsNoMetric = decoderEsNoMetric * np.pow(10.0, fTemp / 10.0)

    nM = 64  # q65_get_alphabet_size(pCodec)
    nN = 63  # q65_get_codeword_length(pCodec)
    nBinsPerTone = 1 << sub_mode

    nBinsPerSymbol = nM * (2 + nBinsPerTone)
    nBinsPerCodeword = nN * nBinsPerSymbol

    # In the fast fading case , the intrinsic probabilities can be computed only
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
    # 2) compute the coefficients w(k) given the coefficient g(k) for the given decodeer Es/No metric
    # 3) compute the logarithm of prob(tone j| en1....enN) which is simply = sum(En(k,j)*w(k)/No
    # 4) subtract from the logarithm of the probabilities their maximum,
    # 5) exponentiate the logarithms
    # 6) normalize the result to a probability distribution dividing each value
    #    by the sum of all of them

    # Evaluate the average noise spectral density
    fNoiseVar = np.mean(input_energies[:nBinsPerCodeword])

    # The noise spectral density so computed includes also the signal power.
    # Therefore we scale it accordingly to the Es/No assumed by the decoder
    fNoiseVar = fNoiseVar / (1.0 + EsNoMetric / nBinsPerSymbol)
    # The value so computed is an overestimate of the true noise spectral density
    # by the (unknown) factor (1+Es/No(true)/nBinsPerSymbol)/(1+EsNoMetric/nBinsPerSymbol)
    # We will take this factor in account when computing the true Es/No ratio

    # store in the pCodec structure for later use in the estimation of the Es/No ratio
    codec.ffNoiseVar = fNoiseVar
    codec.ffEsNoMetric = EsNoMetric
    codec.nBinsPerTone = nBinsPerTone
    codec.nBinsPerSymbol = nBinsPerSymbol
    codec.nWeights = hlen

    # compute the fast fading weights accordingly to the Es/No ratio
    # for which we compute the exact intrinsics probabilities
    weight = np.zeros(hlen, dtype=np.float64)
    codec.ffWeight = weight

    for k in range(hlen):
        fTemp = hptr[k] * EsNoMetric
        weight[k] = fTemp / (1.0 + fTemp) / fNoiseVar

    # Compute now the instrinsics as indicated above
    cur_sym_id = nM  # point to the central bin of the first symbol tone
    cur_ix_id = 0  # point to the first intrinsic
    # cur_ix = np.zeros(nN * nM, dtype=np.float64)
    cur_ix = np.zeros(nN * nM, dtype=np.float64)

    hhsz = hlen - 1  # number of symmetric taps
    hlast = 2 * hhsz  # index of the central tap
    for n in range(nN):  # for each symbol in the message
        # compute the logarithm of the tone probability
        # as a weighted sum of the pertaining energies
        cur_bin_id = cur_sym_id - hlen + 1  # point to the first bin of the current symbol

        maxlogp = 0.0
        for k in range(nM):  # for each tone in the current symbol
            # do a symmetric weighted sum
            fTemp = 0.0
            for j in range(hhsz):
                fTemp += weight[j] * (input_energies[cur_bin_id + j] + input_energies[cur_bin_id + hlast - j])

            fTemp += weight[hhsz] * input_energies[cur_bin_id + hhsz]

            maxlogp = max(maxlogp, fTemp)  # keep track of the max
            cur_ix[cur_ix_id + k] = fTemp
            # cur_ix[n, k] = fTemp

            cur_bin_id += nBinsPerTone  # next tone

        # exponentiate and accumulate the normalization constant
        sumix = 0.0
        for k in range(nM):
            # x = cur_ix[n, k] - maxlogp
            x = cur_ix[cur_ix_id + k] - maxlogp
            x = min(85.0, max(-85.0, x))
            fTemp = np.exp(x)
            # cur_ix[n, k] = fTemp
            cur_ix[cur_ix_id + k] = fTemp
            sumix += fTemp

        # scale to a probability distribution
        cur_ix[cur_ix_id:cur_ix_id + nM] *= 1.0 / sumix
        # cur_ix[n, :] *= 1.0 / sumix

        cur_sym_id += nBinsPerSymbol  # next symbol input energies
        cur_ix_id += nM  # next symbol intrinsics

    return cur_ix


def q65_esnodb_fastfading(
        codec: q65_codec_ds,
        y_dec: npt.NDArray[np.int64],
        input_energies: npt.NDArray[np.float64],
) -> float:
    # Estimate the Es/No ratio of the decoded codeword

    qra_N = codec.pQraCode.q65_get_codeword_length()
    qra_M = codec.pQraCode.q65_get_alphabet_size()

    nBinsPerTone = codec.nBinsPerTone
    nBinsPerSymbol = codec.nBinsPerSymbol
    nWeights = codec.nWeights
    ffNoiseVar = codec.ffNoiseVar
    ffEsNoMetric = codec.ffEsNoMetric
    nTotWeights = 2 * nWeights - 1

    # compute symbols energy (noise included) summing the
    # energies pertaining to the decoded symbols in the codeword

    EsPlusWNo = 0.0
    cur_sym_idx = qra_M  # pInputEnergies + qra_M	# point to first central bin of first symbol tone
    for n in range(qra_N):
        cur_tone_idx = cur_sym_idx + y_dec[n] * nBinsPerTone  # point to the central bin of the current decoded symbol
        cur_bin_idx = cur_tone_idx - nWeights + 1  # point to first bin

        # sum over all the pertaining bins
        # for j in range(nTotWeights):
        #     EsPlusWNo += input_energies[cur_bin_idx + j]
        EsPlusWNo += np.sum(input_energies[cur_bin_idx: cur_bin_idx + nTotWeights])

        cur_sym_idx += nBinsPerSymbol

    EsPlusWNo = EsPlusWNo / qra_N  # Es + nTotWeigths*No

    # The noise power ffNoiseVar computed in the q65_intrisics_fastading(...) function
    # is not the true noise power as it includes part of the signal energy.
    # The true noise variance is:
    # No = ffNoiseVar*(1+EsNoMetric/nBinsPerSymbol)/(1+EsNo/nBinsPerSymbol)

    # Therefore:
    # Es/No = EsPlusWNo/No - W = EsPlusWNo/ffNoiseVar*(1+Es/No/nBinsPerSymbol)/(1+Es/NoMetric/nBinsPerSymbol) - W
    # and:
    # Es/No*(1-u/nBinsPerSymbol) = u-W or Es/No = (u-W)/(1-u/nBinsPerSymbol)
    # where:
    # u = EsPlusNo/ffNoiseVar/(1+EsNoMetric/nBinsPerSymbol)

    u = EsPlusWNo / (ffNoiseVar * (1 + ffEsNoMetric / nBinsPerSymbol))
    u = max(u, nTotWeights + 0.316)  # Limit the minimum Es/No to -5 dB approx.
    u = (u - float(nTotWeights)) / (1 - u / float(nBinsPerSymbol))  # linear scale Es/No

    EsNodB = 10.0 * np.log10(u)
    return EsNodB


def q65_intrinsics_ff(codec: q65_codec_ds, s3: npt.NDArray[np.float64], sub_mode: int, B90Ts: float,
                      fading_model: int) -> npt.NDArray[
    np.float64]:
    # Input:   s3[LL,NN]       Received energies
    #          submode         0=A, 4=E
    #          B90             Spread bandwidth, 90% fractional energy
    #          fadingModel     0=Gaussian, 1=Lorentzian
    # Output:  s3prob[LL,NN]   Symbol-value intrinsic probabilities

    # static int first=1;
    #
    # if (first)
    #     // Set the QRA code, allocate memory, and initialize
    #     int rc = q65_init(&codec,&qra15_65_64_irr_e23);
    #     if (rc<0)
    #         printf("error in q65_init()\n");
    #         exit(0);
    #     first=0;
    # rc = q65_intrinsics_fastfading(&codec,s3prob,s3,submode,B90Ts,fadingModel);
    s3prob = q65_intrinsics_fastfading(codec, s3, sub_mode, B90Ts, fading_model)
    return s3prob
    # if (rc<0)
    #     printf("error in q65_intrinsics()\n");
    #     ///qDebug()<<"hgjhgdjhgj";
    #     exit(0);


# Max codeword list size in q65_decode_fullaplist
Q65_FULLAPLIST_SIZE = 256
# Minimum codeword loglikelihood for decoding
Q65_LLH_THRESHOLD = -260.0


# Full AP decoding from a list of codewords
# Compute and verify the loglikelihood of the decoded codeword
def q65_check_llh(intrin: npt.NDArray[np.float64], ydec: npt.NDArray[np.int64], N: int, M: int):
    t = 0
    i = 0
    for k in range(N):
        x = intrin[i + ydec[k]]
        x = max(x, 1.0e-36)
        t += np.log(x)
        i += M

    return (t, t >= Q65_LLH_THRESHOLD)


def q65_mask(qra_code: QRACode, ix: npt.NDArray[np.float64], mask: npt.NDArray[np.int64], x: npt.NDArray[np.int64]):
    # mask intrinsic information ix with available a priori knowledge
    qra_M = qra_code.M
    qra_m = qra_code.m

    # Exclude from masking the symbols which have been punctured.
    # qra_K is the length of the mask and x arrays, which do
    # not include any punctured symbol
    qra_K = qra_code.q65_get_message_length()

    # for each symbol set to zero the probability
    # of the values which are not allowed by
    # the a priori information
    for k in range(qra_K):
        s_mask = mask[k]
        if s_mask:
            for kk in range(qra_M):
                if ((kk ^ x[k]) & s_mask) != 0:
                    # This symbol value is not allowed
                    # by the AP information
                    # Set its probability to zero
                    PD_ROWADDR(ix, qra_M, k)[kk] = 0.0

            # normalize to a probability distribution
            pd_norm(PD_ROWADDR(ix, qra_M, k), qra_m)


QRACODE_MAX_M = 256


def qra_extrinsic(
        qra_code: QRACode,
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
    qra_pmat = qra_code.gfpmat
    qra_msgw = qra_code.msgw

    # define ADDRMSG(fp, msgidx)    PD_ROWADDR(fp,qra_M,msgidx)
    # define C2VMSG(msgidx)         PD_ROWADDR(qra_c2vmsg,qra_M,msgidx)
    # define V2CMSG(msgidx)         PD_ROWADDR(qra_v2cmsg,qra_M,msgidx)
    # define MSGPERM(logw)          PD_ROWADDR(qra_pmat,qra_M,logw)
    C2VMSG = partial(PD_ROWADDR, qra_c2vmsg, qra_M)
    V2CMSG = partial(PD_ROWADDR, qra_v2cmsg, qra_M)
    MSGPERM = partial(PD_ROWADDR, qra_pmat, qra_M)
    ADDRMSG = partial(PD_ROWADDR, ix, qra_M)

    # float msgout[QRACODE_MAX_M]; # we use a fixed size in order to avoid mallocs
    msgout = np.zeros(QRACODE_MAX_M, dtype=np.float64)

    rc = -1  # rc>=0  extrinsic converged to 1 at iteration rc (rc=0..maxiter-1)
    # # rc=-1  no convergence in the given number of iterations
    # # rc=-2  error in the code tables (code checks degrees must be >1)
    # # rc=-3  M is larger than QRACODE_MAX_M

    assert qra_M <= QRACODE_MAX_M
    #     if (qra_M>QRACODE_MAX_M)
    #         return -3;

    # message initialization -------------------------------------------------------

    # init c->v variable intrinsic msgs
    pd_init(C2VMSG(0), ix, qra_M * qra_V)

    # init the v->c messages directed to code factors (k=1..ndeg) with the intrinsic info
    for nv in range(qra_V):  # current variable
        ndeg = qra_vdeg[nv]  # degree of current node
        msgbase = nv * qra_MAXVDEG  # base to msg index row for the current node

        # copy intrinsics on v->c
        for k in range(1, ndeg):
            msg_idx = qra_v2cmidx[msgbase + k]  # current message index
            pd_init(V2CMSG(msg_idx), ADDRMSG(nv), qra_M)

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
                np_fwht(qra_m, V2CMSG(msg_idx), V2CMSG(msg_idx))  # compute fwht

            # compute products and transform them back in the WH "time" domain
            for k in range(ndeg):  # loop indexes
                # init output message to uniform distribution
                pd_init(msgout, pd_uniform(qra_m), qra_M)

                # c->v = prod(fwht(v->c))
                # TODO: we assume that checks degrees are not larger than three but
                # if they are larger the products can be computed more efficiently
                for kk in range(ndeg):  # loop indexes
                    if kk != k:
                        msg_idx = qra_c2vmidx[msgbase + kk]
                        pd_imul(msgout, V2CMSG(msg_idx), qra_m)

                # transform product back in the WH "time" domain

                # Very important trick:
                # we bias WHT[0] so that the sum of output pd components is always strictly positive
                # this helps avoiding the effects of underflows in the v->c steps when multipling
                # small fp numbers
                msgout[0] += 1E-7  # TODO: define the bias accordingly to the field size

                np_fwht(qra_m, msgout, msgout)

                # inverse weight and output
                msg_idx = qra_c2vmidx[msgbase + k]  # current output msg index
                wmsg = qra_msgw[msg_idx]  # current msg weight

                if wmsg == 0:
                    pd_init(C2VMSG(msg_idx), msgout, qra_M)
                else:
                    # output p(alfa^(-w)*x)
                    pd_bwdperm(C2VMSG(msg_idx), msgout, MSGPERM(wmsg), qra_M)

        # v->c step -----------------------------------------------------
        for nv in range(qra_V):
            ndeg = qra_vdeg[nv]  # degree of current node
            msgbase = nv * qra_MAXVDEG  # base to msg index row for the current node

            for k in range(ndeg):
                # init output message to uniform distribution
                pd_init(msgout, pd_uniform(qra_m), qra_M)

                # v->c msg = prod(c->v)
                # TODO: factor factors to reduce the number of computations for high degree nodes
                for kk in range(ndeg):
                    if kk != k:
                        msg_idx = qra_v2cmidx[msgbase + kk]
                        pd_imul(msgout, C2VMSG(msg_idx), qra_m)

                # normalize the result to a probability distribution
                pd_norm(msgout, qra_m)
                # weight and output
                msg_idx = qra_v2cmidx[msgbase + k]  # current output msg index
                wmsg = qra_msgw[msg_idx]  # current msg weight

                if wmsg == 0:
                    pd_init(V2CMSG(msg_idx), msgout, qra_M)
                else:
                    # output p(alfa^w*x)
                    pd_fwdperm(V2CMSG(msg_idx), msgout, MSGPERM(wmsg), qra_M)

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
            totex += pd_max(V2CMSG(nv), qra_M)

        if totex > (1.0 * (qra_V) - 0.01):
            # the total maximum extrinsic information of each symbol in the codeword
            # is very close to one. This means that we have reached the (1,1) point in the
            # code EXIT chart(s) and we have successfully decoded the input.
            rc = nit
            break  # remove the break to evaluate the decoder speed performance as a function of the max iterations number)

    # copy extrinsic information to output to do the actual max a posteriori prob decoding
    pd_init(ex, V2CMSG(0), (qra_M * qra_V))
    return rc


# void q65subs::qra_mapdecode(const qracode *pcode, int *xdec, float *pex, const float *pix)
def qra_mapdecode(pcode: QRACode, xdec: npt.NDArray[np.int64], pex: npt.NDArray[np.float64],
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
        pd_imul(PD_ROWADDR(pex, qra_M, k), PD_ROWADDR(pix, qra_M, k), qra_m)
        xdec[k] = pd_argmax(PD_ROWADDR(pex, qra_M, k), qra_M)


Q65_DECODE_CRCMISMATCH = -3


def q65_decode(
        codec: q65_codec_ds,
        pDecodedCodeword: npt.NDArray[np.int64],
        pDecodedMsg: npt.NDArray[np.int64],
        pIntrinsics: npt.NDArray[np.float64],
        APMask: npt.NDArray[np.int64],
        APSymbols: npt.NDArray[np.int64],
        maxiters: int
):
    qra_code = codec.pQraCode
    ix = codec.ix
    ex = codec.ex

    nK = qra_code.q65_get_message_length()
    nN = qra_code.q65_get_codeword_length()
    nM = qra_code.M
    nBits = qra_code.m

    px = codec.x
    py = codec.y

    # Depuncture intrinsics observations as required by the code type
    if qra_code.type == QRATYPE_CRCPUNCTURED:
        ix[:nK * nM] = pIntrinsics[:nK * nM]  # information symbols
        pd_init(PD_ROWADDR(ix, nM, nK), pd_uniform(nBits), nM)  # crc
        ix[(nK + 1) * nM:(nK + 1) * nM + (nN - nK) * nM] = pIntrinsics[
                                                           nK * nM: nK * nM + (nN - nK) * nM]  # parity checks
    elif qra_code.type == QRATYPE_CRCPUNCTURED2:
        ix[:nK * nM] = pIntrinsics[:nK * nM]  # information symbols
        pd_init(PD_ROWADDR(ix, nM, nK), pd_uniform(nBits), nM)  # crc
        pd_init(PD_ROWADDR(ix, nM, nK + 1), pd_uniform(nBits), nM)  # crc
        ix[(nK + 2) * nM: (nK + 2) * nM + (nN - nK) * nM] = pIntrinsics[
                                                            nK * nM: nK * nM + (nN - nK) * nM]  # parity checks
    else:
        # no puncturing
        ix[:nN * nM] = pIntrinsics[:nN * nM]  # as they are

    # mask the intrinsics with the available a priori knowledge
    q65_mask(qra_code, ix, APMask, APSymbols)

    # Compute the extrinsic symbols probabilities with the message-passing algorithm
    # Stop if the extrinsics information does not converges to unity
    # within the given number of iterations
    rc = qra_extrinsic(qra_code,
                       ex,
                       ix,
                       maxiters,
                       codec.qra_v2cmsg,
                       codec.qra_c2vmsg)
    if rc < 0:
        # failed to converge to a solution
        # return Q65_DECODE_FAILED
        raise Exception("Q65_DECODE_FAILED")

    # decode the information symbols (punctured information symbols included)
    qra_mapdecode(qra_code, px, ex, ix)

    # verify CRC match
    if qra_code.type in (QRATYPE_CRC, QRATYPE_CRCPUNCTURED):
        crc = crc6(px[:nK])  # compute crc-6
        if crc != px[nK]:
            return Q65_DECODE_CRCMISMATCH  # crc doesn't match
    elif qra_code.type == QRATYPE_CRCPUNCTURED2:
        crc = crc12(px[:nK])  # compute crc-12
        if (crc & 0x3F) != px[nK] or (crc >> 6) != px[nK + 1]:
            return Q65_DECODE_CRCMISMATCH  # crc doesn't match

    # copy the decoded msg to the user buffer (excluding punctured symbols)
    if pDecodedMsg is not None:
        # memcpy(pDecodedMsg,px,nK*sizeof(int));
        pDecodedMsg[:nK] = px[:nK]

    # if (pDecodedCodeword==NULL)		# user is not interested in the decoded codeword
    #     return rc;					# return the number of iterations required to decode

    # crc matches therefore we can reconstruct the transmitted codeword
    #  reencoding the information available in px...

    # qra_encode(qra_code, py, px)
    py[:] = qra_encode(px, concat=True)

    # ...and strip the punctured symbols from the codeword
    if qra_code.type == QRATYPE_CRCPUNCTURED:
        pDecodedCodeword[:nK] = py[:nK]
        pDecodedCodeword[nK:nK + (nN - nK)] = py[nK + 1:nK + (nN - nK) + 1]  # puncture crc-6 symbol
    elif qra_code.type == QRATYPE_CRCPUNCTURED2:
        pDecodedCodeword[:nK] = py[:nK]
        pDecodedCodeword[nK:nK + (nN - nK)] = py[nK + 2:nK + (nN - nK) + 2]  # puncture crc-12 symbol
    else:
        pDecodedCodeword[:nN] = py[:nN]  # no puncturing

    return rc  # return the number of iterations required to decode


def q65_dec(
        codec: q65_codec_ds,
        s3: npt.NDArray[np.float64],
        s3prob: npt.NDArray[np.float64],
        APmask: npt.NDArray[np.int64],
        APsymbols: npt.NDArray[np.int64],

        maxiters: int,
        xdec: npt.NDArray[np.int64],
):
    # Input:   s3[LL,NN]       Symbol spectra
    #          s3prob[LL,NN]   Symbol-value intrinsic probabilities
    #          APmask[13]      AP information to be used in decoding
    #          APsymbols[13]   Available AP informtion
    # Output:
    #          esnodb0         Estimated Es/No (dB)
    #          xdec[13]        Decoded 78-bit message as 13 six-bit integers
    #          rc0             Return code from q65_decode()
    # int rc;
    # int ydec[63];
    # float esnodb;

    ydec = np.zeros(63, dtype=np.int64)
    rc = q65_decode(codec, ydec, xdec, s3prob, APmask, APsymbols, maxiters)
    # rc0=rc;
    # # rc = -1:  Invalid params
    # # rc = -2:  Decode failed
    # # rc = -3:  CRC mismatch
    # esnodb0 = 0.0;             //Default Es/No for a failed decode
    # if (rc<0) return;

    # rc = q65_esnodb_fastfading(&codec,&esnodb,ydec,s3);
    esnodb = q65_esnodb_fastfading(codec, ydec, s3)
    # if (rc<0)
    #     printf("error in q65_esnodb_fastfading()\n");
    #     exit(0);
    # esnodb0 = esnodb;
    return rc, esnodb


if __name__ == '__main__':
    with open("../data2.json") as f:
        data = json.load(f)

    s3_1fa = data["s3_1fa"]
    nsubmode = 1
    b90ts = 0.516000
    nFadingModel = 1

    apmask = np.zeros(13, dtype=np.int64)
    apsymbols = np.zeros(13, dtype=np.int64)
    dat4 = np.zeros(13, dtype=np.int64)
    s_maxiters = 100

    codec = q65_init()
    s3prob = q65_intrinsics_ff(codec, s3_1fa, nsubmode, b90ts, nFadingModel)
    rc, esnodb = q65_dec(codec, s3_1fa, s3prob, apmask, apsymbols, s_maxiters, dat4)

    print(rc, esnodb)
    print(dat4)
