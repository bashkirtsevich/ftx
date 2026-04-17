import typing

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from encoders import wspr_encode
from fec.conv.fano import fano
from msg.message import WSPRMessage
from utils.common import dB
from consts.wspr import *

from decoders.monitor import AbstractMonitor, LogItem


@dataclass
class Candidate:
    freq: float
    snr: float
    shift: int = 0
    drift: float = 0
    sync: float = 0


@dataclass
class WSPRLogItem(LogItem):
    BER: int
    drift: float
    decode_pass: int


class WSPRMonitor(AbstractMonitor):
    __slots__ = [
        "sample_rate",
        "signal",
        "symbol_len",
        "df",
        "df2",
        "dt",
        "decode_passes",
        "quick_mode",
    ]

    MAX_CANDIDATES = 410
    CANDIDATE_FREQ_MIN = -110
    CANDIDATE_FREQ_MAX = 110

    def __init__(self, sample_rate: int):
        self.signal = np.zeros(0, dtype=np.float64)
        self.sample_rate = sample_rate  # Source sample rate
        self.quick_mode = False

        down_sample_rate = self.sample_rate / WSPR_DECIMATION  # 375
        self.symbol_len = int(down_sample_rate * WSPR_SYMBOL_PERIOD)  # 256

        self.df = down_sample_rate / self.symbol_len  # 375 / 256
        self.df2 = self.df / 2  # 375 / 256 / 2
        self.dt = 1 / down_sample_rate
        self.decode_passes = 3

    def downsample(self) -> npt.NDArray[np.complex128]:
        fft_target = 46080

        fft_src = fft_target * WSPR_DECIMATION

        frame = np.pad(self.signal[:fft_src], (0, max(0, fft_src - len(self.signal))))
        spec = np.fft.rfft(frame)

        df = self.sample_rate / fft_src
        i0 = int(WSPR_CENTER_FREQ / df + 0.5)

        sub_spec = spec[i0: i0 + fft_target]
        signal = np.fft.ifft(sub_spec) * (fft_target / fft_src) * 2  # AGC
        return signal.astype(np.complex128)

    @staticmethod
    def deinterleave(sym: npt.NDArray[np.uint8]):
        tmp = np.zeros(WSPR_ND, dtype=np.uint8)

        p = 0
        i = 0
        while p < WSPR_ND:
            j = (((i * 0x80200802) & 0x0884422110) * 0x0101010101 >> 32) & 0xff
            if j < WSPR_ND:
                tmp[p] = sym[j]
                p = p + 1
            i += 1

        sym[:] = tmp[:]

    @staticmethod
    def count_sym_err(symbols: npt.NDArray[np.uint8], channel_symbols: npt.NDArray[np.uint8]) -> int:
        cw = (channel_symbols >= 2).astype(np.uint8)
        WSPRMonitor.deinterleave(cw)

        sym_bin = (symbols > 127).astype(np.uint8)
        err_count = int(np.sum(sym_bin != cw))

        return err_count

    def find_candidates(self, smooth_spec: npt.NDArray[np.float64]) -> typing.List[Candidate]:
        # Find all local maxima in smoothed spectrum.
        spec_slice = smooth_spec[:self.MAX_CANDIDATES]

        is_max = (spec_slice[1:-1] > spec_slice[:-2]) & (spec_slice[1:-1] > spec_slice[2:])

        indices = np.where(is_max)[0] + 1
        freqs = (indices - 205) * self.df2  # 205 -- freq shift

        mask = (freqs >= self.CANDIDATE_FREQ_MIN) & (freqs <= self.CANDIDATE_FREQ_MAX)
        cand_indices = indices[mask]
        cand_freqs = freqs[mask]

        snrs = [dB(ss) - WSPR_SNR_SCALING_FACTOR for ss in smooth_spec[cand_indices]]
        heap = [Candidate(freq=freq, snr=snr) for freq, snr in zip(cand_freqs, snrs)]

        heap.sort(key=lambda it: it.snr, reverse=True)
        return heap

    def sync(
            self,
            iq_signal: npt.NDArray[np.complex128],
            f1: float,
            f_min: int, f_max: int, f_step: float,
            shift1: int,
            lag_min: int, lag_max: int, lag_step: int,
            drift1: float,
            mode: int  # FIXME: Use enum
    ) -> typing.Tuple[float, int, float]:
        # ************************************************************************
        # * mode = 0: no frequency or drift search. find best time lag.          *
        # *        1: no time lag or drift search. find best frequency.          *
        # ************************************************************************
        df_offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * self.df
        two_pi_dt = 2 * np.pi * self.dt

        sync_max = -1e30
        best_shift = shift1
        f_best = f1

        if mode == 0:
            f_min, f_max, f_step = 0, 0, 0.0

        if mode == 1:
            lag_min, lag_max = shift1, shift1

        tone_pwr = np.zeros(WSPR_TONES_COUNT, dtype=np.float64)

        for freq in range(f_min, f_max + 1):
            f0 = f1 + freq * f_step

            for lag in range(lag_min, lag_max + lag_step, lag_step):
                ss = 0.0
                total_pwr = 0.0

                for sym in range(WSPR_ND):
                    fp = f0 + (drift1 / 2.0) * (sym - WSPR_NUM_BITS) / WSPR_NUM_BITS

                    tone_pwr.fill(0)

                    freqs = fp + df_offsets
                    t_indices = np.arange(self.symbol_len)

                    phasors = np.exp(-1j * two_pi_dt * np.outer(freqs, t_indices))

                    for tone in range(WSPR_TONES_COUNT):
                        start = lag + sym * self.symbol_len
                        end = start + self.symbol_len

                        segment = iq_signal[max(0, start): min(len(iq_signal), end)]

                        current_phasors = phasors[tone, :len(segment)]

                        acc = np.dot(current_phasors, segment)
                        tone_pwr[tone] = np.abs(acc)

                    total_pwr += np.sum(tone_pwr)
                    channel_metric = np.sum(tone_pwr[1::2]) - np.sum(tone_pwr[0::2])

                    ss += channel_metric * WSPR_PR3_SIG[sym]

                if total_pwr > 0:
                    ss /= total_pwr

                if ss > sync_max:
                    sync_max = ss
                    best_shift = lag
                    f_best = f0

        return (sync_max, best_shift, f_best)

    def noncoherent_sequence_detection(
            self,
            iq_signal: npt.NDArray[np.complex128],
            symbols: npt.NDArray[np.uint8],
            f1: float,
            shift1: int,
            drift1: float,
            sym_fac: int,
            block_size: int,
            bit_metric: bool
    ):
        #  ************************************************************************
        #  *  Noncoherent sequence detection for wspr.                            *
        #  *  Allowed block lengths are block=1,2,3,6, or 9 symbols.             *
        #  *  Longer block lengths require longer channel coherence time.         *
        #  *  The whole block is estimated at once.                               *
        #  *  block=1 corresponds to noncoherent detection of individual symbols *
        #  *     like the original wsprd symbol demodulator.                      *
        #  ************************************************************************
        df15 = self.df * 1.5
        df05 = self.df * 0.5

        ref_tones = np.zeros((WSPR_TONES_COUNT, WSPR_ND), dtype=np.complex128)
        phasor_ends = np.zeros((WSPR_TONES_COUNT, WSPR_ND), dtype=np.complex128)
        symb = np.zeros(WSPR_ND, dtype=np.float64)

        two_pi_dt = 2 * np.pi * self.dt
        f0 = f1

        fp = f0 + (drift1 / 2.0) * (np.arange(WSPR_ND) - WSPR_NUM_BITS) / WSPR_NUM_BITS
        d_phi = two_pi_dt * np.array([fp - df15, fp - df05, fp + df05, fp + df15])  # Shape: (4, 162)

        phases = d_phi[:, :, np.newaxis] * np.arange(self.symbol_len + 1)  # (4, 162, 257)
        phasors = np.exp(1j * phases)

        lag = int(shift1)
        indices = lag + np.arange(WSPR_ND)[:, np.newaxis] * self.symbol_len + np.arange(self.symbol_len)
        indices = np.clip(indices, 0, len(iq_signal) - 1)

        segments = iq_signal[indices]
        segments[indices >= len(iq_signal)].fill(0)

        ref_tones[:, :WSPR_ND] += np.einsum('tfj,fj->tf', np.conj(phasors[:, :, :self.symbol_len]), segments)

        phasor_ends[:, :WSPR_ND] = phasors[:, :, -1]

        # fplast = -10000.0
        # lag = int(shift1)
        # phasors = np.zeros((WSPR_TONES_COUNT, 257), dtype=np.complex128)
        # for i in range(WSPR_ND):
        #     fp = f0 + (drift1 / 2.0) * (i - WSPR_NUM_BITS) / WSPR_NUM_BITS
        #     if i == 0 or fp != fplast:  # only calculate sin/cos if necessary
        #         j = np.arange(257)
        #
        #         d_phis = two_pi_dt * np.array([
        #             fp - df15,
        #             fp - df05,
        #             fp + df05,
        #             fp + df15
        #         ])
        #
        #         phases = np.outer(d_phis, j)
        #         phasors[:, :] = np.exp(1j * phases)
        #
        #         fplast = fp
        #
        #     phasor_ends[:, i] = phasors[:, 256]
        #
        #     start = lag + i * 256
        #     end = start + 256
        #     segment = iq_signal[max(0, start): min(len(iq_signal), end)]
        #
        #     if len(segment) > 0:
        #         ref_tones[:, i] += np.dot(segment, np.conj(phasors[:, :len(segment)].T))

        seq = 1 << block_size
        seq_weights = np.zeros(seq, dtype=np.float64)
        bit_mask = np.arange(seq)
        bit_exp = np.arange(block_size - 1, -1, -1)
        phases = np.ones(block_size, dtype=np.complex128)
        for i in range(0, WSPR_ND, block_size):
            time_block = np.arange(i, i + block_size)
            pr3_block = WSPR_PR3[i: i + block_size]

            for j in range(seq):
                bits = (j >> bit_exp) & 1
                tones = pr3_block + 2 * bits

                phasor_proj = ref_tones[tones, time_block]
                phasor_shifts = phasor_ends[tones, time_block]

                phases.fill(1 + 0j)
                phases[1:] = np.cumprod(phasor_shifts[:-1])

                res = np.sum(phasor_proj * phases)
                seq_weights[j] = np.abs(res)

            for ib in range(block_size):
                mask = (bit_mask & (1 << (block_size - 1 - ib))) == 0
                xm0 = seq_weights[:seq][mask].max()
                xm1 = seq_weights[:seq][~mask].max()

                metric = xm1 - xm0
                if bit_metric:
                    metric /= np.max(xm0, xm1)

                symb[i + ib] = metric

        std = np.std(symb)
        symb *= sym_fac / std
        symbols[:] = np.clip(symb, -128.0, 127.0) + 128

    #  ******************************************************************************
    #   Subtract the coherent component of a signal
    #  *******************************************************************************
    def subtract_signal(self, signal: npt.NDArray[np.complex128], f0, shift0: int, drift0,
                        channel_symbols: npt.NDArray[np.uint8]):
        filtr = 360  # filtr must be even number.
        samples = WSPR_ND * self.symbol_len

        cs_rep = np.repeat(channel_symbols - 1.5, self.symbol_len)

        idx = np.repeat(np.arange(WSPR_ND), self.symbol_len)
        drift_eff = (drift0 / 2.0) * (idx - WSPR_ND / 2.0) / (WSPR_ND / 2.0)

        freq = f0 + drift_eff + cs_rep * self.df
        phase = 2.0 * np.pi * np.cumsum(freq) * self.dt

        ref = np.exp(1j * phase)

        start = max(shift0, 0)
        end = min(start + samples, len(signal))

        segment = signal[start:end]

        env = segment * np.conj(ref)

        weights = np.sin(np.pi * np.arange(filtr) / (filtr - 1))
        weights /= weights.sum()

        env_sm = np.convolve(env, weights, mode="same")

        partial_sum = np.cumsum(weights)

        filtr_half = filtr // 2
        corr_f = np.ones(samples)
        corr_f[:filtr_half] = partial_sum[filtr_half: filtr]  # Приблизительно, согласно логике оригинала
        corr_f[-filtr_half:] = np.flip(partial_sum[filtr_half: filtr])

        reconstruction = (env_sm / corr_f) * ref

        signal[start:end] -= reconstruction[start:end]

    def decode(self, **kwargs) -> typing.Iterator[WSPRLogItem]:
        iq_signal = self.downsample()

        # *************** main loop starts here *****************
        symbols = np.zeros(WSPR_NUM_BITS * 2, dtype=np.uint8)

        fft_step = 128
        fft_size = fft_step * 4
        fft_count = 4 * (len(iq_signal) // fft_size) - 1

        smooth = np.sin((np.pi / fft_size) * np.arange(fft_size))

        pwr_spec = np.zeros((fft_count, fft_size))

        block_size = 1
        max_drift = 4
        min_sync_2 = WSPR_MIN_SYNC_2

        decodes_pass = 0

        for iteration in range(self.decode_passes):
            if iteration == 1 and decodes_pass == 0 and self.decode_passes > 2:
                iteration = 2

            if iteration == 2:
                block_size = 4  # try 3 blocksizes plus bitbybit normalization
                max_drift = 0  # no drift for smaller frequency estimator variance
                min_sync_2 = WSPR_MIN_SYNC_1

            decodes_pass = 0

            for i in range(fft_count):
                start = i * fft_step

                segment = iq_signal[start: start + fft_size]
                if len(segment) < fft_size:
                    segment = np.pad(segment, (0, fft_size - len(segment)), mode="constant")

                cpx_spec = np.fft.fft(segment * smooth)
                cpx_spec = np.fft.fftshift(cpx_spec)
                pwr_spec[i, :] = np.abs(cpx_spec) ** 2

            pwr_spec_avg = np.mean(pwr_spec, axis=0) * fft_count

            center = fft_size // 2
            span = 205
            relevant_part = pwr_spec_avg[center - span: center + span + 1]

            smooth_spec = np.convolve(relevant_part, WSPR_SMOOTH_KERNEL, mode="same")

            noise_level = np.percentile(smooth_spec, WSPR_NOISE_PERCENTILE)

            # Renormalize spectrum so that (large) peaks represent an estimate of snr.
            # We know from experience that threshold snr is near -7dB in wspr bandwidth,
            # corresponding to -7-26.3=-33.3dB in 2500 Hz bandwidth.
            # The corresponding threshold is -42.3 dB in 2500 Hz bandwidth for WSPR-15. *

            min_snr = np.pow(10.0, -8.0 / 10.0)  # this is min snr in wspr bw

            smooth_spec = smooth_spec / noise_level - 1.0
            smooth_spec[smooth_spec < min_snr] = 0.1 * min_snr

            # Find all local maxima in smoothed spectrum.
            candidates = self.find_candidates(smooth_spec)

            # * Make coarse estimates of shift (DT), freq, and drift
            #
            # * Look for time offsets up to +/- 8 symbols (about +/- 5.4 s) relative
            # to nominal start time, which is 2 seconds into the file
            #
            # * Calculates shift relative to the beginning of the file
            #
            # * Negative shifts mean that signal started before start of file
            #
            # * The program prints DT = shift-2 s
            #
            # * Shifts that cause sync vector to fall off of either end of the data
            # vector are accommodated by "partial decoding", such that missing
            # symbols produce a soft-decision symbol value of 128
            #
            # * The frequency drift model is linear, deviation of +/- drift/2 over the
            # span of 162 symbols, with deviation equal to 0 at the center of the
            # signal vector.
            # *

            for cand in candidates:  # For each candidate...
                smax = -1e30
                if0 = int(cand.freq / self.df2 + self.symbol_len)
                for ifr in range(if0 - 2, if0 + 2 + 1):  # Freq search
                    for k0 in range(-10, 22):  # Time search
                        for drift in range(-max_drift, max_drift + 1):  # Drift search
                            ss = 0.0
                            pow = 0.0
                            for k in range(WSPR_ND):  # Sum over symbols
                                ifd = int(ifr + (k - WSPR_NUM_BITS) / WSPR_NUM_BITS * drift / (2.0 * self.df2))
                                k_idx = k0 + 2 * k
                                if k_idx >= 0 and k_idx < fft_count:
                                    pwrs = pwr_spec[k_idx, ifd - 3: ifd + 4: 2]
                                    pwrs_sqrt = np.sqrt(pwrs)

                                    ss += WSPR_PR3_SIG[k] * pwrs_sqrt.dot([-1, 1, -1, 1])
                                    pow += np.sum(pwrs_sqrt)

                            sync1 = ss / pow
                            if sync1 > smax:  # Save coarse parameters
                                smax = sync1
                                cand.shift = 128 * (k0 + 1)
                                cand.drift = drift
                                cand.freq = (ifr - self.symbol_len) * self.df2
                                cand.sync = sync1

            # tcandidates += (float)(clock()-t0)/CLOCKS_PER_SEC;

            # Refine the estimates of freq, shift using sync as a metric.
            # Sync is calculated such that it is a float taking values in the range
            # [0.0,1.0].
            #
            # Function sync_and_demodulate has three modes of operation
            # mode is the last argument:
            #
            # 0 = no frequency or drift search. find best time lag.
            # 1 = no time lag or drift search. find best frequency.
            # 2 = no frequency or time lag search. Calculate soft-decision
            # symbols using passed frequency and shift.
            #
            # NB: best possibility for OpenMP may be here: several worker threads
            # could each work on one candidate at a time.

            for cand in candidates:
                f1 = cand.freq
                drift1 = cand.drift
                shift1 = cand.shift

                # coarse-grid lag and freq search, then if sync>minsync1 continue
                f_step = 0.0
                f_min = 0
                f_max = 0
                lag_min = shift1 - 128
                lag_max = shift1 + 128
                lag_step = 64

                sync1, shift1, f1 = self.sync(iq_signal, f1, f_min, f_max, f_step, shift1, lag_min, lag_max, lag_step,
                                              drift1, 0)

                f_step = 0.25
                f_min = -2
                f_max = 2

                sync1, shift1, f1 = self.sync(iq_signal, f1, f_min, f_max, f_step, shift1, lag_min, lag_max, lag_step,
                                              drift1, 1)

                if iteration < 2:
                    # refine drift estimate
                    f_step = 0.0
                    f_min = 0
                    f_max = 0

                    driftp = drift1 + 0.5
                    syncp, shift1, f1 = self.sync(iq_signal, f1, f_min, f_max, f_step, shift1, lag_min, lag_max,
                                                  lag_step, driftp, 1)

                    driftm = drift1 - 0.5
                    syncm, shift1, f1 = self.sync(iq_signal, f1, f_min, f_max, f_step, shift1, lag_min, lag_max,
                                                  lag_step, driftm, 1)

                    if syncp > sync1:
                        drift1 = driftp
                        sync1 = syncp

                    elif syncm > sync1:
                        drift1 = driftm
                        sync1 = syncm

                # fine-grid lag and freq search
                if sync1 > WSPR_MIN_SYNC_1:
                    lag_min = shift1 - 32
                    lag_max = shift1 + 32
                    lag_step = 16

                    sync1, shift1, f1 = self.sync(iq_signal, f1, f_min, f_max, f_step, shift1, lag_min, lag_max,
                                                  lag_step, drift1, 0)

                    # fine search over frequency
                    f_step = 0.05
                    f_min = -2
                    f_max = 2

                    sync1, shift1, f1 = self.sync(iq_signal, f1, f_min, f_max, f_step, shift1, lag_min, lag_max,
                                                  lag_step, drift1, 1)

                    cand.freq = f1
                    cand.shift = shift1
                    cand.drift = drift1
                    cand.sync = sync1

            # remove duplicates
            cands = 0
            for j in range(len(candidates)):
                dupe = False
                for k in range(cands):
                    if abs(candidates[j].freq - candidates[k].freq) < 0.05 and abs(
                            candidates[j].shift - candidates[k].shift) < 16:
                        dupe = True
                        break

                if dupe:
                    if candidates[j].sync > candidates[k].sync:
                        candidates[k] = candidates[j]
                elif candidates[j].sync > min_sync_2:
                    candidates[cands] = candidates[j]
                    cands += 1

            for cand in candidates:
                f1 = cand.freq
                shift1 = cand.shift
                drift1 = cand.drift
                decoded = None

                ib = 1
                while ib <= block_size and decoded is None:
                    if ib < 4:
                        block_size = ib
                        bit_metric = False
                    else:  # if ib == 4:
                        block_size = 1
                        bit_metric = True

                    idt = 0
                    while decoded is None and idt <= (128 / WSPR_IIFAC):
                        ii = (idt + 1) / 2
                        if idt % 2 == 1:
                            ii = -ii

                        ii = WSPR_IIFAC * ii
                        jittered_shift = shift1 + ii

                        # Get soft-decision symbols
                        self.noncoherent_sequence_detection(
                            iq_signal, symbols, f1, jittered_shift, drift1, WSPR_SOFT_SYM_FAC, block_size, bit_metric
                        )

                        rms = np.sqrt(np.mean(np.square(symbols.astype(np.float32) - 128)))
                        if rms > WSPR_MIN_RMS:
                            self.deinterleave(symbols)

                            # decoded = jelinek(symbols, WSPR_NUM_BITS, stacksize, WSPR_METRIC_TABLE, WSPR_DECODER_LIM)
                            decoded = fano(
                                symbols, WSPR_NUM_BITS, WSPR_METRIC_TABLE, WSPR_FANO_THRESHOLD, WSPR_DECODER_LIM
                            )

                            idt += 1

                            if self.quick_mode:
                                break
                        ib += 1

                    if decoded is not None:
                        metric, cycles, dec_data = decoded

                        decodes_pass += 1

                        payload = dec_data.tobytes()
                        msg = WSPRMessage.decode(payload)  # FIXME: Check for exception when message is invalid

                        dt_print = shift1 * self.dt - 1.0
                        freq_print = (WSPR_CENTER_FREQ + f1) / 1e6

                        payload = msg.encode()
                        tones = wspr_encode(payload)
                        chan_sym = np.fromiter(tones, dtype=np.uint8)

                        # Unpack the decoded message, update the hashtable, apply
                        # sanity checks on grid and power, and return
                        # call_loc_pow string and also callsign (for de-duping).
                        self.subtract_signal(iq_signal, f1, shift1, drift1, chan_sym)
                        sym_err = self.count_sym_err(symbols, chan_sym)

                        yield WSPRLogItem(
                            snr=cand.snr,
                            dT=dt_print,
                            dF=freq_print,
                            payload=payload,
                            crc=0,
                            BER=sym_err,
                            drift=drift1,
                            decode_pass=decodes_pass
                        )

    def monitor_process(self, frame: npt.NDArray[np.int16]) -> None:
        frame_float = frame.astype(np.float32) / np.iinfo(np.int16).max
        self.signal = np.concat([self.signal, frame_float])
