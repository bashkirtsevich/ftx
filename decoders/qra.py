import typing
import numpy as np
import numpy.typing as npt
from consts.q65 import *
from decoders.monitor import AbstractMonitor, LogItem
from utils.common import dB
from utils.q65 import smooth_121, shell_sort_percentile, bzap, q65_6bit_decode
from qra.q65 import q65_intrinsics_ff, q65_dec, q65_init


class Q65Monitor(AbstractMonitor):
    __slots__ = ["sample_rate", "signal"]

    CCF_OFFSET_R = 70
    CCF_OFFSET_C = 7000

    def __init__(self, q65_type: typing.Literal[1, 2, 3, 4],
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

        self.smooth = max(1, int(0.5 * self.q65_type ** 2))

        self.sym_steps = int(self.sym_samps / NSTEP)
        self.dt_step = self.sym_samps / (NSTEP * 12000.0)

        self.lag1 = int(-1.0 / self.dt_step)
        self.lag2 = int(1.0 / self.dt_step + 0.9999)

        self.i0 = 0
        self.j0 = 0

        self.LL0 = 0
        self.iz0 = 0
        self.jz0 = 0

        # LL = 64 * (2 + mode_q65) # 64 * 10
        # self.LL = 64 * 10
        self.NN = 63

        self.max_iters = 100

        self.fft_size = self.sym_samps
        self.df = 12000.0 / self.fft_size  # !Freq resolution = baud

        # self.iz = int(5000.0 / self.df)
        # self.txt = 85.0 * self.samp_per_sym / 12000.0
        # self.jz = int((self.txt + 1.0) * 12000.0 / self.istep)

        self.nfa = 214.333328
        self.nfb = 2000.000000

        self.bw_a = 1
        self.bw_b = 11

        # self.navg = np.zeros(2, dtype=np.int64)
        self.candidates_ = np.zeros((2, 20), dtype=np.float64)

        self.max_drift = 0
        self.f_max_drift = False

        self.q65_codec = q65_init()

        # !Generate the sync vector
        self.sync = np.full(Q65_TONES_COUNT, -22.0 / 63.0, dtype=np.float64)  # !Sync tone OFF
        self.sync[Q65_SYNC - 1] = 1.0  # !Sync tone ON

    # !Compute symbol spectra with NSTEP time-steps per symbol.
    def symbol_spectra(self, num_bins: int, num_times: int) -> npt.NDArray[np.float64]:
        fac = (1.0 / np.iinfo(np.int16).max) * 0.01

        sym_spec = np.zeros((800, 7000), dtype=np.float64)
        for j in range(0, num_times, 2):  # !Compute symbol spectra at 2*step size
            f_beg = j * self.sym_steps
            f_end = f_beg + self.sym_samps

            spec = np.fft.fft(self.signal[f_beg:f_end] * fac, n=self.fft_size)[:num_bins]  # iwave * fac ?
            sym_spec[j, :num_bins] = np.abs(spec) ** 2

            # !For large Doppler spreads, should we smooth the spectra here?
            if self.smooth > 1:
                for _ in range(self.smooth):
                    sym_spec[j] = smooth_121(sym_spec[j])

            # ! Interpolate to fill in the skipped-over spectra.
            if j >= 2:
                sym_spec[j - 1] = 0.5 * (sym_spec[j - 2] + sym_spec[j])

        return sym_spec

    def ccf_22(self, s1: npt.NDArray[np.float64], iz: int, jz: int, target_freq: float):
        xdt2 = np.zeros(7000, dtype=np.float64)
        ccf3 = np.zeros(7000, dtype=np.float64)
        s1avg = np.zeros(7000, dtype=np.float64)

        mdec_df = 50
        snfa = target_freq - mdec_df
        snfb = target_freq + mdec_df

        self.max_drift = 0

        if self.f_max_drift:
            self.max_drift = min(100.0 / self.df, 60)

        ia = int(max(self.nfa, 100.0) / self.df)
        ib = int(min(self.nfb, 4900.0) / self.df)

        for i in range(ia, ib):
            s1avg[i] = np.sum(s1[:jz, i])

        ccfbest = 0.0
        ibest = 0
        lagbest = 0
        idrift_best = 0
        for i in range(ia, ib):
            ccfmax_s = 0.0
            ccfmax_m = 0.0
            lagpk_s = 0
            lagpk_m = 0
            idrift_max_s = 0
            for lag in range(self.lag1, self.lag2 + 1):
                for idrift in range(-self.max_drift, self.max_drift + 1):
                    ccft = 0.0
                    for kk in range(22):
                        k = Q65_SYNC[kk] - 1
                        zz = idrift * (k - 43)
                        ii = i + (int)(zz / 85.0)
                        if ii < 0 or ii >= iz:
                            continue
                        n = NSTEP * k
                        j = n + lag + self.j0
                        if j > -1 and j < jz:
                            ccft += s1[j, ii]

                    ccft -= (22.0 / jz) * s1avg[i]
                    if ccft > ccfmax_s:
                        ccfmax_s = ccft
                        lagpk_s = lag
                        idrift_max_s = idrift
                    if ccft > ccfmax_m and idrift == 0:
                        ccfmax_m = ccft
                        lagpk_m = lag

            ccf3[i] = ccfmax_m
            xdt2[i] = lagpk_m * self.dt_step

            f = i * self.df
            if ccfmax_s > ccfbest and (f >= snfa and f <= snfb):
                ccfbest = ccfmax_s
                ibest = i
                lagbest = lagpk_s
                idrift_best = idrift_max_s

        # corrp = np.argmax(ccf3[int(snfa / self.df):int(snfb / self.df)]) + int(snfa / self.df)
        # self.xdtnd = xdt2[corrp]
        # self.f0nd = target_freq + (corrp - self.i0) * self.df

        # ! Parameters for the top candidate:
        ipk = ibest - self.i0
        jpk = lagbest
        f0 = target_freq + ipk * self.df
        xdt = jpk * self.dt_step
        self.drift = self.df * idrift_best

        ccf3[0:ia] = 0.0
        ccf3[ib:iz] = 0.0

        # ! Save parameters for best candidates
        jzz = min(ib - ia, 25)

        t_s = np.zeros(7000, dtype=np.float64)
        for z in range(jzz):
            t_s[z] = ccf3[z + ia]

        indices = np.argsort(t_s[:jzz])
        ave = shell_sort_percentile(t_s[:jzz], 50)
        base = shell_sort_percentile(t_s[:jzz], 84)

        if (rms := base - ave) == 0.0:
            rms = 0.000001

        ncand = 0
        maxcand = 20

        for j in range(maxcand):
            k = jzz - j - 1
            if k < 0 or k >= iz:
                continue
            i = indices[k] + ia
            f = i * self.df
            i3 = int(max(0, i - self.q65_type))
            i4 = int(min(iz, i + self.q65_type))

            biggest = np.max(ccf3[i3:i4])
            if ccf3[i] != biggest:
                continue

            snr = (ccf3[i] - ave) / rms
            if snr < 6.0:
                break

            self.candidates_[0, ncand] = xdt2[i]
            self.candidates_[1, ncand] = f

            ncand += 1
            if ncand > maxcand - 1:
                break  # no needed

        # ! Resort the candidates back into frequency order
        tmp = np.zeros((2, 25), dtype=np.float64)
        for j in range(ncand):
            tmp[0, j] = self.candidates_[0, j]
            tmp[1, j] = self.candidates_[1, j]

            self.candidates_[0, j] = 0.0
            self.candidates_[1, j] = 0.0
            indices[j] = 0

        if ncand > 0:
            indices = np.argsort(tmp[1, :ncand])

        for i in range(ncand):
            self.candidates_[0, i] = tmp[0, indices[i]]
            self.candidates_[1, i] = tmp[1, indices[i]]

        return ipk, jpk, f0, xdt

    def s1_to_s3(
            self,
            s1: npt.NDArray[np.float64],
            iz: int, jz: int, i_peak: int, j_peak: int, LL: int,
            s3: npt.NDArray[np.float64]
    ):
        # ! Copy synchronized symbol energies from s1 (or s1a) into s3.
        i1 = self.i0 + i_peak + self.q65_type - 64
        i2 = i1 + LL  # int LL=64*(2+mode_q65);
        i3 = i2 - i1  # A=192 .... D=640

        if i1 > 0 and i2 < iz:
            j = self.j0 + j_peak - 8
            n = 0

            for k in range(85):
                j += 8
                if self.sync[k] > 0.0:
                    continue

                if j > 0 and j < jz:
                    for i in range(i3):
                        s3[n] = s1[j, i + i1]
                        n += 1

        bzap(s3, LL)  # !Zap birdies

    def decode_2(self, s3: npt.NDArray[np.float64], sub_mode: int, b90ts: float) -> typing.Optional[
        typing.Tuple[float, npt.NDArray]
    ]:
        # ! Attempt a q0, q1, or q2 decode using spcified AP information.
        s3 = s3.reshape((-1, self.NN))
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
        for ibw in range(self.bw_a, self.bw_b + 1):
            b90 = 1.72 ** ibw
            b90ts = b90 / baud

            if (decoded := self.decode_2(s3, sub_mode, b90ts)) is not None:
                EsNo_dB, payload = decoded

                snr = EsNo_dB - dB(2500.0 / baud) + 3.0  # !Empirical adjustment
                return snr, payload

    def decode_0(self, f0: float, eme_delay: bool) -> typing.Tuple[float, float, npt.NDArray[np.int64], float]:
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
        istep = self.sym_samps / NSTEP
        iz = int(5000.0 / self.df)  # !Uppermost frequency bin, at 5000 Hz
        txt = 85.0 * self.sym_samps / 12000.0
        jz = int((txt + 1.0) * 12000.0 / istep)  # !Number of symbol/NSTEP bins
        if self.sym_samps >= 6912:
            jz = int((txt + 2.0) * 12000.0 / istep)  # !For TR 60 s and higher



        s3_1fa = np.zeros(63 * 640, dtype=np.float64)  # attention = 63*640=40320 q65d from q65_subs

        t_s = np.zeros(700, dtype=np.float64)

        if LL != self.LL0 or iz != self.iz0 or jz != self.jz0:
            self.LL0 = LL
            self.iz0 = iz
            self.jz0 = jz

        dt_step = self.sym_samps / (NSTEP * 12000.0)  # !Step size in seconds
        self.lag1 = int(-1.0 / dt_step)
        self.lag2 = int(1.0 / dt_step + 0.9999)

        if self.sym_samps >= 3600 and eme_delay:
            self.lag2 = int(5.5 / dt_step + 0.9999)  # !Include EME

        self.j0 = int(0.5 / dt_step)
        if self.sym_samps >= 7200:
            self.j0 = int(1.0 / dt_step)  # !Nominal start-signal index if(nsps.ge.7200) j0=1.0/dt_step

        # ! Compute symbol spectra with NSTEP time bins per symbol
        sym_spec = self.symbol_spectra(iz, jz)

        self.i0 = int(f0 / self.df)  # !Target QSO frequency
        if self.i0 - 64 < 0:
            self.i0 = 64

        if self.i0 - 64 + LL > iz - 1:
            self.i0 = iz + 64 - LL

        for j in range(jz):
            for z in range(LL):
                t_s[z] = sym_spec[j, z + self.i0 - 64]

            base = shell_sort_percentile(t_s[:LL], 45)
            if base == 0.0:
                base = 0.000001

            for z in range(iz):
                sym_spec[j, z] /= base

        for j in range(jz):
            # ! Apply fast AGC to the symbol spectra
            s1max = 20.0  # !Empirical choice
            smax = np.max(sym_spec[j, :iz])  # smax=maxval(s1(ii1:ii2,j))
            if smax > s1max:
                sym_spec[j, :iz] *= s1max / smax

        # ! Get 2d CCF and ccf2 using sync symbols only
        i_peak, j_peak, ccf_freq, time_d = self.ccf_22(sym_spec, iz, jz, f0)  # maybe out of bandwidth df

        # ! The q3 decode attempt failed. Copy synchronized symbol energies from s1
        # ! into s3 and prepare to try a more general decode.
        self.s1_to_s3(sym_spec, iz, jz, i_peak, j_peak, LL, s3_1fa)

        snr, data = self.decode_q012(s3_1fa)

        return time_d, ccf_freq, data, snr

    def decode(self, f0: int, eme_delay: bool, **kwargs) -> typing.Generator[LogItem, None, None]:
        dT, ccf_freq, payload, snr = self.decode_0(f0=f0, eme_delay=eme_delay)

        yield LogItem(snr, dT, ccf_freq - f0, payload.tobytes(), 0)

    def monitor_process(self, frame: npt.NDArray[np.int16]) -> None:
        self.signal = np.concat([self.signal, frame])
