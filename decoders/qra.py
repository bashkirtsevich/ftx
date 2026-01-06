import typing

import numpy as np
import numpy.typing as npt
from consts.q65 import *
from decoders.monitor import AbstractMonitor, LogItem
from utils.common import dB
from utils.q65 import smooth_121, bzap, q65_6bit_decode
from qra.q65 import q65_intrinsics_ff, q65_dec, q65_init


class Q65Monitor(AbstractMonitor):
    __slots__ = ["sample_rate", "signal"]

    CCF_OFFSET_R = 70
    CCF_OFFSET_C = 7000

    def __init__(self, eme_delay: bool, q65_type: typing.Literal[1, 2, 3, 4],
                 period: typing.Optional[typing.Literal[15, 30, 60, 120, 300]] = None):
        self.signal = np.zeros(0, dtype=np.float64)

        self.q65_type = q65_type

        if period == 30:
            self.sym_samps = 3600
        elif period == 60:
            self.sym_samps = 7200
        elif period == 120:
            self.sym_samps = 16000
        else:
            self.sym_samps = 3600

        self.fft_size = self.sym_samps
        self.df = 12000.0 / self.fft_size  # !Freq resolution = baud

        self.smooth = max(1, int(0.5 * self.q65_type ** 2))

        self.sym_steps = self.sym_samps // NSTEP
        self.dt_step = self.sym_samps / (NSTEP * 12000.0)  # !Step size in seconds

        # dt_step = self.sym_samps / (NSTEP * 12000.0)
        self.lag1 = int(-1.0 / self.dt_step)
        # !Include EME
        self.lag2 = int((5.5 if self.sym_samps >= 3600 and eme_delay else 1) / self.dt_step + 0.9999)

        self.i0 = 0

        # !Nominal start-signal index if(nsps.ge.7200) j0=1.0/dt_step
        self.j0 = int((1 if self.sym_samps >= 7200 else 0.5) / self.dt_step)

        # self.i0 = 0
        # self.j0 = 0

        self.NN = 63

        self.max_iters = 100

        self.nf_a = 214.333328
        self.nf_b = 2000.000000

        self.bw_a = 1
        self.bw_b = 11

        self.f_max_drift = False

        self.q65_codec = q65_init()

        # !Generate the sync vector
        self.sync = np.full(Q65_TONES_COUNT, -22.0 / 63.0, dtype=np.float64)  # !Sync tone OFF
        self.sync[Q65_SYNC - 1] = 1  # !Sync tone ON

    # !Compute symbol spectra with NSTEP time-steps per symbol.
    def symbol_spectra(self, num_bins: int, num_times: int) -> npt.NDArray[np.float64]:
        fac = (1 / np.iinfo(np.int16).max) * 0.01

        sym_spec = np.zeros((num_times, num_bins), dtype=np.float64)
        for i in range(0, num_times, 2):  # !Compute symbol spectra at 2*step size
            f_beg = i * self.sym_steps
            f_end = f_beg + self.sym_samps

            spec = np.fft.fft(self.signal[f_beg:f_end] * fac, n=self.fft_size)[:num_bins]  # iwave * fac ?
            sym_spec[i, :] = np.abs(spec) ** 2

            # !For large Doppler spreads, should we smooth the spectra here?
            if self.smooth > 1:
                for _ in range(self.smooth):
                    sym_spec[i] = smooth_121(sym_spec[i])

            # ! Interpolate to fill in the skipped-over spectra.
            if i >= 2:
                sym_spec[i - 1] = 0.5 * (sym_spec[i - 2] + sym_spec[i])

        return sym_spec

    def sync_ccf(self, sym_spec: npt.NDArray[np.float64], f0: float) -> typing.Tuple[int, int, float, float]:
        jz, iz = sym_spec.shape

        dec_df = 50
        snf_a = f0 - dec_df
        snf_b = f0 + dec_df

        max_drift = min(100 / self.df, 60) if self.f_max_drift else 0

        bin_start = int(max(self.nf_a, 100) / self.df)
        bin_end = int(min(self.nf_b, 4900) / self.df)

        # ccf_sync = np.zeros(bin_end - bin_start, dtype=np.float64)
        # time_offsets = np.zeros(bin_end - bin_start, dtype=np.float64)
        sym_spec_avg = np.sum(sym_spec[:, bin_start:bin_end], axis=0)

        ccf_best = 0
        best = 0
        lag_best = 0
        for i in range(bin_start, bin_end):
            ccf_max_s = 0
            ccf_max_m = 0
            lag_peak_s = 0
            # lag_peak_m = 0

            for lag in range(self.lag1, self.lag2 + 1):
                for drift in range(-max_drift, max_drift + 1):
                    ccf_t = 0
                    for kk in range(Q65_SYNC_TONES_COUNT):
                        k = Q65_SYNC[kk] - 1
                        zz = drift * (k - Q65_TONES_CENTER)
                        ii = i + zz // Q65_TONES_COUNT

                        if ii < 0 or ii >= iz:
                            continue

                        n = NSTEP * k
                        j = n + lag + self.j0
                        if j > -1 and j < jz:
                            ccf_t += sym_spec[j, ii]

                    ccf_t -= (Q65_SYNC_TONES_COUNT / jz) * sym_spec_avg[i - bin_start]

                    if ccf_t > ccf_max_s:
                        ccf_max_s = ccf_t
                        lag_peak_s = lag

                    if ccf_t > ccf_max_m and drift == 0:
                        ccf_max_m = ccf_t
                        # lag_peak_m = lag

            # ccf_sync[i - bin_start] = ccf_max_m
            # time_offsets[i - bin_start] = lag_peak_m * self.dt_step

            f = i * self.df
            if ccf_max_s > ccf_best and snf_a <= f <= snf_b:
                ccf_best = ccf_max_s
                best = i
                lag_best = lag_peak_s

        # ! Parameters for the top candidate:
        i_peak = best - self.i0
        j_peak = lag_best

        f0 += i_peak * self.df
        dt = j_peak * self.dt_step

        return i_peak, j_peak, f0, dt

    def get_data_sym(self, sym_spec: npt.NDArray[np.float64], i_peak: int, j_peak: int, LL: int) -> npt.NDArray[
        np.float64]:
        # data_sym = np.zeros(Q65_DATA_TONES_COUNT * 640, dtype=np.float64)  # attention = 63*640=40320 q65d from q65_subs

        jz, iz = sym_spec.shape

        # ! Copy synchronized symbol energies from s1 (or s1a) into data_sym.
        i1 = self.i0 + i_peak + self.q65_type - 64
        i2 = i1 + LL  # int LL=64*(2+mode_q65);
        i3 = i2 - i1  # A=192 .... D=640

        # attention = 63*640=40320 q65d from q65_subs
        data_sym = np.zeros((Q65_DATA_TONES_COUNT, i3), dtype=np.float64)

        if i1 > 0 and i2 < iz:
            j = self.j0 + j_peak - 8
            n = 0

            for k in range(Q65_TONES_COUNT):
                j += 8
                if self.sync[k] > 0.0:
                    continue

                if j > 0 and j < jz:
                    data_sym[n, :i3] = sym_spec[j, i1:i2]

                    n += 1

        bzap(data_sym)  # !Zap birdies

        return data_sym

    def decode_2(self, s3: npt.NDArray[np.float64], sub_mode: int, b90ts: float) -> typing.Optional[
        typing.Tuple[float, npt.NDArray]
    ]:
        # ! Attempt a q0, q1, or q2 decode using spcified AP information.
        s3prob = q65_intrinsics_ff(self.q65_codec, s3, sub_mode, b90ts, fading_model=FadingModel.Lorentzian)

        ap_mask = np.zeros(13, dtype=np.int64)  # !Try first with no AP information
        ap_symbols = np.zeros(13, dtype=np.int64)

        status, EsNo_dB, decoded = q65_dec(self.q65_codec, s3, s3prob, ap_mask, ap_symbols, self.max_iters)

        if status < 0 or np.sum(decoded) <= 0:
            return None

        payload = q65_6bit_decode(decoded)
        return EsNo_dB, payload

    def decode_q012(self, s3: npt.NDArray[np.float64]) -> typing.Optional[typing.Tuple[float, npt.NDArray]]:
        # ! Do separate passes attempting q0, q1, q2 decodes.
        if self.q65_type == 2:
            sub_mode = 1
        elif self.q65_type == 4:
            sub_mode = 2
        elif self.q65_type == 8:
            sub_mode = 3
        else:
            sub_mode = 0

        baud = 12000.0 / self.sym_samps
        for bw in range(self.bw_a, self.bw_b + 1):
            b90 = 1.72 ** bw
            b90ts = b90 / baud

            if (decoded := self.decode_2(s3, sub_mode, b90ts)) is not None:
                EsNo_dB, payload = decoded

                snr = EsNo_dB - dB(2500.0 / baud) + 3.0  # !Empirical adjustment
                return snr, payload

        return None

    def decode_0(self, f0: float) -> typing.Tuple[float, float, npt.NDArray[np.int64], float]:
        # Top-level routine in q65 module
        # !   - Compute symbol spectra
        # !   - Attempt sync and q3 decode using all 85 symbols
        # !   - If that fails, try sync with 22 symbols and standard q[0124] decode
        #
        # ! Input:  iavg                   0 for single-period decode, 1 for average
        # !         iwave(0:nmax-1)        Raw data
        # !         ntrperiod              T/R sequence length (s)
        # !         nfqso                  Target frequency (Hz)
        # !         ntol                   Search range around nfqso (Hz)
        # !         ndepth                 Requested decoding depth
        # !         lclearave              Flag to clear the accumulating array
        # !         emedelay               Extra delay for EME signals
        # ! Output: xdt                    Time offset from nominal (s)
        # !         f0                     Frequency of sync tone
        # !         snr1                   Relative SNR of sync signal
        # !         width                  Estimated Doppler spread
        # !         dat4(13)               Decoded message as 13 six-bit integers
        # !         snr2                   Estimated SNR of decoded signal
        # !         idec                   Flag for decing results
        # !            -1  No decode
        # !             0  No AP
        # !             1  "CQ        ?    ?"
        # !             2  "Mycall    ?    ?"
        # !             3  "MyCall HisCall ?"

        LL = 64 * (2 + self.q65_type)  # mode_q65 -- 1, 2, 3, 4

        txt = 85.0 * self.sym_samps / 12000.0 + (2 if self.sym_samps >= 6912 else 1)  # !For TR 60 s and higher

        iz = int(5000.0 / self.df)  # !Uppermost frequency bin, at 5000 Hz
        jz = int(txt * 12000.0 / self.sym_steps)  # !Number of symbol/NSTEP bins

        self.i0 = min(max(int(f0 / self.df), 64), iz + 64 - LL)  # !Target QSO frequency

        # ! Compute symbol spectra with NSTEP time bins per symbol
        sym_spec = self.symbol_spectra(iz, jz)

        for j in range(jz):
            t_s = sym_spec[j, self.i0 - 64:self.i0 - 64 + LL]
            if (base := np.percentile(t_s, 45)) == 0:
                base = 0.000001

            sym_spec[j, :] /= base

        for j in range(jz):
            # ! Apply fast AGC to the symbol spectra
            s1_max = 20.0  # !Empirical choice
            s_max = np.max(sym_spec[j, :])  # s_max=maxval(s1(ii1:ii2,j))

            if s_max > s1_max:
                sym_spec[j, :] *= s1_max / s_max

        # ! Get 2d CCF and ccf2 using sync symbols only
        i_peak, j_peak, ccf_freq, time_d = self.sync_ccf(sym_spec, f0)  # maybe out of bandwidth df

        # ! The q3 decode attempt failed. Copy synchronized symbol energies from s1
        # ! into data_sym and prepare to try a more general decode.

        data_sym = self.get_data_sym(sym_spec, i_peak, j_peak, LL)

        snr, data = self.decode_q012(data_sym)

        return time_d, ccf_freq, data, snr

    def decode(self, f0: int, **kwargs) -> typing.Generator[LogItem, None, None]:
        dT, ccf_freq, payload, snr = self.decode_0(f0=f0)

        yield LogItem(snr, dT, ccf_freq - f0, payload.tobytes(), 0)

    def monitor_process(self, frame: npt.NDArray[np.int16]) -> None:
        self.signal = np.concat([self.signal, frame])
