import numpy as np
import numpy.typing as npt
from contextlib import suppress
from functools import cache
from numba import jit
from dataclasses import dataclass

from crc.mskx import mskx_check_crc, mskx_extract_crc
from encode import msk144_encode
from ldpc.mskx import bp_decode
from consts.mskx import *
import typing

from monitor import DecodeStatus, AbstractMonitor


@dataclass
class LogItem:
    snr: float
    dT: float
    dF: float
    p_avg: int
    freq: int
    eye_open: float
    bit_err: int

    payload: typing.ByteString
    crc: int


class NextCand(Exception):
    pass


class MSK144Monitor(AbstractMonitor):
    MAX_LDPC_ERRORS = 18
    LDPC_ITERATIONS = 10

    @staticmethod
    @cache
    @jit(nopython=True)
    def filter_response(n_fft: int, sample_rate: int) -> npt.NDArray[np.float64]:
        t = 1 / 2000
        beta = 0.1
        df = sample_rate / n_fft

        nh = (n_fft // 2) + 1

        response = np.full(nh, 1, dtype=np.float64)
        for i in range(nh):
            f = abs(df * i - MSK144_CENTER_FREQ)

            if (1 + beta) / (2 * t) >= f > (1 - beta) / (2 * t):
                response[i] = response[i] * 0.5 * (1 + np.cos((np.pi * t / beta) * (f - (1 - beta) / (2 * t))))

            elif f > (1 + beta) / (2 * t):
                response[i] = 0

        return response

    @staticmethod
    def fourier_bpf(signal: npt.NDArray[np.float64], n_fft: int,
                    response: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        # Time domain -> Freq domain
        freq_d = np.fft.fft(signal, n=n_fft)

        # Frequency attenuation
        response_len = len(response)
        freq_d[:response_len] = freq_d[:response_len] * response
        freq_d[0] = 0.5 * freq_d[0]
        # Attenuate other
        freq_d[len(response): n_fft] = complex(0, 0)

        # Freq domain -> Time domain
        time_d = np.fft.ifft(freq_d, n=n_fft)

        return time_d

    @staticmethod
    @jit(nopython=True)
    def shift_freq(complex_signal: npt.NDArray[np.complex128],
                   freq: float, sample_rate: int) -> npt.NDArray[np.complex128]:
        phi = 2 * np.pi * freq / sample_rate
        step = complex(np.cos(phi), np.sin(phi))

        phasor = complex(1, 1)
        signal = np.zeros(len(complex_signal), dtype=np.complex128)

        for i, cs_val in enumerate(complex_signal):
            phasor *= step
            signal[i] = phasor * cs_val

        return signal

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.signal = np.zeros(0, dtype=np.float64)

    # DecodeStatus(ldpc_errors, crc_extracted), payload, eye_opening, bit_errors
    def decode_fame(self, frame: npt.NDArray[np.complex128],
                    max_iterations: int) -> typing.Optional[typing.Tuple[DecodeStatus, typing.ByteString, float, int]]:
        cca = sum(frame[:SYNC_WAVEFORM_LEN] * SYNC_WAVEFORM_CONJ)
        ccb = sum(frame[56 * 6: 56 * 6 + SYNC_WAVEFORM_LEN] * SYNC_WAVEFORM_CONJ)
        cc = cca + ccb

        phase_0 = np.atan2(cc.imag, cc.real)

        fac = complex(np.cos(phase_0), np.sin(phase_0))
        frame *= fac.conjugate()

        soft_bits = np.zeros(MSK144_BITS_COUNT, dtype=np.float64)
        hard_bits = np.zeros(MSK144_BITS_COUNT, dtype=np.int64)

        for i in range(6):
            soft_bits[0] += frame[i].imag * SAMPLES_PER_WORD[i + 6]
            soft_bits[0] += frame[i + (MSK144_NSPM - 5 - 1)].imag * SAMPLES_PER_WORD[i]

        for i in range(12):
            soft_bits[1] += frame[i].real * SAMPLES_PER_WORD[i]

        for i in range(1, 72):
            sum_01 = 0
            for j in range(12):
                sum_01 += frame[i * 12 - 6 + j].imag * SAMPLES_PER_WORD[j]
            soft_bits[2 * i] = sum_01

            sum_01 = 0
            for j in range(12):
                sum_01 += frame[i * 12 + j].real * SAMPLES_PER_WORD[j]
            soft_bits[2 * i + 1] = sum_01

            if soft_bits[2 * i] >= 0:
                hard_bits[2 * i] = 1

            if soft_bits[2 * i + 1] >= 0:
                hard_bits[2 * i + 1] = 1

        if soft_bits[0] >= 0:
            hard_bits[0] = 1

        if soft_bits[1] >= 0:
            hard_bits[1] = 1

        bad_sync_1 = 0
        bad_sync_2 = 0
        for i in range(8):
            bad_sync_1 += (2 * hard_bits[i] - 1) * SYNC_WORDS[i]
            bad_sync_2 += ((2 * hard_bits[i + 57 - 1] - 1) * SYNC_WORDS[i])

        bad_sync_1 = (8 - bad_sync_1) // 2
        bad_sync_2 = (8 - bad_sync_2) // 2

        bad_sync = bad_sync_1 + bad_sync_2
        if bad_sync > 4:
            return

        sav = 0.0
        s2av = 0.0
        for i in range(MSK144_BITS_COUNT):
            sav += soft_bits[i]
            s2av += soft_bits[i] * soft_bits[i]

        sav /= MSK144_BITS_COUNT
        s2av /= MSK144_BITS_COUNT

        ssig = np.sqrt(s2av - (sav * sav))
        if ssig == 0:
            ssig = 1

        for i in range(MSK144_BITS_COUNT):
            soft_bits[i] = soft_bits[i] / ssig

        sigma = 0.60

        lratio = np.concat([soft_bits[8: 9 + 47], soft_bits[64: 65 + 80 - 1]])
        llr = 2 * lratio / (sigma * sigma)

        ldpc_errors, plain128 = bp_decode(llr, max_iterations)

        if ldpc_errors > self.MAX_LDPC_ERRORS:
            return None

        if not mskx_check_crc(plain128):
            return None

        # Extract payload + CRC (first FTX_LDPC_K bits) packed into a byte array
        payload = self.pack_bits(plain128, MSKX_LDPC_K)
        # Extract CRC
        crc_extracted = mskx_extract_crc(payload)

        tones = list(msk144_encode(payload))
        msg_bits = np.zeros(MSK144_BITS_COUNT)
        for i in range(MSK144_BITS_COUNT - 1):
            j = i + 1
            if tones[i] == 0:
                msg_bits[j] = msg_bits[i] if j % 2 == 1 else (msg_bits[i] + 1) % 2
            else:
                msg_bits[j] = (msg_bits[i] + 1) % 2 if j % 2 == 1 else msg_bits[i]

        eye_top = 1
        eye_bot = -1
        for i in range(MSK144_BITS_COUNT):
            if msg_bits[i] == 1:
                eye_top = min(soft_bits[i], eye_top)
            else:
                eye_bot = max(soft_bits[i], eye_bot)
        eye_opening = eye_top - eye_bot

        bit_errors = np.count_nonzero(hard_bits != msg_bits)

        return DecodeStatus(ldpc_errors, crc_extracted), payload, eye_opening, bit_errors

    def detect_signals(self, signal: npt.NDArray[np.complex128], signal_len: int, start: float,
                       max_cand: int = 16, tolerance: int = 150) -> typing.Generator[LogItem, None, None]:
        # ! define half-sine pulse and raised-cosine edge window
        dt = 1 / self.sample_rate

        df = self.sample_rate / MSK144_NFFT

        step_size = 216
        steps_count = (signal_len - MSK144_NPTS) // step_size

        freq_hi = 2 * (MSK144_CENTER_FREQ + 500)
        freq_lo = 2 * (MSK144_CENTER_FREQ - 500)

        ih_lo = int((freq_hi - 2 * tolerance) / df)
        ih_hi = int((freq_hi + 2 * tolerance) / df)

        il_lo = int((freq_lo - 2 * tolerance) / df)
        il_hi = int((freq_lo + 2 * tolerance) / df)

        freq_2000hz = int(freq_lo / df)
        freq_4000hz = int(freq_hi / df)

        time_arr = np.zeros(max_cand, dtype=np.float64)
        freq_arr = np.zeros(max_cand, dtype=np.float64)
        snr_arr = np.zeros(max_cand, dtype=np.float64)

        detmet = np.zeros(steps_count)
        detmet2 = np.zeros(steps_count)
        detfer = np.full(steps_count, -999.99)

        steps_real = 0
        for step in range(steps_count):
            part_start = step_size * step
            part_end = part_start + MSK144_NSPM

            if part_end > signal_len:
                break

            part = signal[part_start: part_end]

            # Coarse carrier frequency sync - seek tones at 2000 Hz and 4000 Hz in
            # squared signal spectrum.
            # search range for coarse frequency error is +/- 100 Hz
            part = part ** 2
            part[:12] = part[:12] * SMOOTH_WINDOW
            part[MSK144_NSPM - 12:MSK144_NSPM] = part[MSK144_NSPM - 12:MSK144_NSPM] * SMOOTH_WINDOW[::-1]  # Looks like window smooth function

            spec = np.fft.fft(part, MSK144_NFFT)
            amps = np.abs(spec) ** 2

            # Hi freq
            h_peak = ih_lo + np.argmax(amps[ih_lo:ih_hi])

            delta_hi = -((spec[h_peak - 1] - spec[h_peak + 1]) / (
                    2 * spec[h_peak] - spec[h_peak - 1] - spec[h_peak + 1])).real
            mag_hi = amps[h_peak]

            ahavp = (np.sum(amps[ih_lo:ih_hi]) - mag_hi) / (ih_hi - ih_lo)
            trath = mag_hi / (ahavp + 0.01)
            # Low freq
            l_peak = il_lo + np.argmax(amps[il_lo:il_hi])

            delta_lo = -((spec[l_peak - 1] - spec[l_peak + 1]) / (
                    2 * spec[l_peak] - spec[l_peak - 1] - spec[l_peak + 1])).real
            mag_lo = amps[l_peak]

            alavp = (np.sum(amps[il_lo:il_hi]) - mag_lo) / (il_hi - il_lo)
            tratl = mag_lo / (alavp + 0.01)
            # Errors
            freq_err_h = (h_peak + delta_hi - freq_4000hz) * df / 2
            freq_err_l = (l_peak + delta_lo - freq_2000hz) * df / 2

            if mag_hi >= mag_lo:
                f_error = freq_err_h
            else:
                f_error = freq_err_l

            detmet[step] = max(mag_hi, mag_lo)
            detmet2[step] = max(trath, tratl)
            detfer[step] = f_error

            steps_real += 1

        indices = np.argsort(detmet[:steps_real])

        xmedian = detmet[indices[steps_real // 4]]
        if xmedian == 0:
            xmedian = 1

        detmet[:steps_real] = detmet[:steps_real] / xmedian

        count_cand = 0
        for ip in range(max_cand):
            il = np.argmax(detmet[:steps_real])

            if detmet[il] < 3.5:
                break

            if abs(detfer[il]) <= tolerance:
                time_arr[count_cand] = ((il - 0) * step_size + MSK144_NSPM / 2) * dt
                freq_arr[count_cand] = detfer[il]
                snr_arr[count_cand] = 12 * np.log10(detmet[il]) / 2 - 9

                count_cand += 1

            detmet[il] = 0

        if count_cand < 3:  # for Tropo/ES
            for ip in range(max_cand - count_cand):
                if ip >= max_cand - count_cand:
                    break

                # Find candidates
                il = np.argmax(detmet2[:steps_real])

                if detmet2[il] < 12.0:
                    break

                if abs(detfer[il]) <= tolerance:
                    time_arr[count_cand] = ((il - 0) * step_size + MSK144_NSPM / 2) * dt
                    freq_arr[count_cand] = detfer[il]
                    snr_arr[count_cand] = 12 * np.log10(detmet2[il]) / 2 - 9

                    count_cand += 1

                detmet2[il] = 0

        if count_cand > 0:
            indices = np.argsort(time_arr[:count_cand])

        # ! Try to sync/demod/decode each candidate.
        hashes = set()
        for iip in range(count_cand):
            with suppress(NextCand):
                ip = indices[iip]
                imid = int(time_arr[ip] * self.sample_rate)

                if imid < MSK144_NPTS / 2:
                    imid = MSK144_NPTS // 2

                if imid > signal_len - MSK144_NPTS / 2:
                    imid = signal_len - MSK144_NPTS // 2

                t0 = time_arr[ip] + dt * (start)

                part = signal[imid - MSK144_NPTS // 2: imid + MSK144_NPTS // 2]

                f_error = freq_arr[ip]
                snr = 2 * int(snr_arr[ip] / 2)
                snr = max(-4.0, min(24.0, snr))

                # ! remove coarse freq error - should now be within a few Hz
                part = self.shift_freq(part, -(MSK144_CENTER_FREQ + f_error), self.sample_rate)

                cc1 = np.zeros(MSK144_NPTS, dtype=np.complex128)
                cc2 = np.zeros(MSK144_NPTS, dtype=np.complex128)

                for i in range(MSK144_NPTS - (56 * 6 + 42)):
                    cc1[i] = sum(part[i: i + SYNC_WAVEFORM_LEN] * SYNC_WAVEFORM_CONJ)
                    cc2[i] = sum(part[i + 56 * 6: i + 56 * 6 + SYNC_WAVEFORM_LEN] * SYNC_WAVEFORM_CONJ)

                dd = abs(cc1) * abs(cc2)

                # ! Find 6 largest peaks
                peaks = []
                for ipk in range(6):
                    # HV Good work cc ic1 no dd and ic2
                    ic2 = np.argmax(dd)
                    dd[max(0, ic2 - 7): min(MSK144_NPTS - 56 * 6 - 42, ic2 + 7)] = 0.0
                    peaks.append(ic2)

                # ! we want ic to be the index of the first sample of the frame
                for ic0 in peaks:  # ic0=peaks[ipk]
                    # ! fine adjustment of sync index
                    # ! bb lag used to place the sampling index at the center of the eye
                    bb = np.zeros(6, dtype=np.complex128)
                    for i in range(6):
                        cd_b = ic0 + i
                        if ic0 + 11 + MSK144_NSPM < MSK144_NPTS:
                            bb[i] = np.sum(
                                (part[cd_b + 6: cd_b + 6 + MSK144_NSPM: 6] * np.conj(part[cd_b:cd_b + MSK144_NSPM:6])) ** 2)
                        else:
                            bb[i] = np.sum((part[cd_b + 6: MSK144_NPTS: 6] * np.conj(part[cd_b:MSK144_NPTS - 6:6])) ** 2)

                    ibb = np.argmax(np.abs(bb))

                    if ibb <= 2:
                        ibb -= 1
                    if ibb > 2:
                        ibb -= 7

                    for id in range(3):
                        if id == 1:
                            sign = -1
                        elif id == 2:
                            sign = 1
                        else:
                            sign = 0

                        # ! Adjust frame index to place peak of bb at desired lag
                        ic = ic0 + ibb + sign

                        if ic < 0:
                            ic = ic + MSK144_NSPM

                        # ! Estimate fine frequency error.
                        # ! Should a larger separation be used when frames are averaged?
                        cca = np.sum(part[ic:ic + SYNC_WAVEFORM_LEN] * SYNC_WAVEFORM_CONJ)
                        if ic + 56 * 6 + 42 < MSK144_NPTS:
                            ccb = np.sum(part[ic + 56 * 6:ic + 56 * 6 + SYNC_WAVEFORM_LEN] * SYNC_WAVEFORM_CONJ)
                            cfac = ccb * np.conj(cca)
                            f_error_2 = np.atan2(cfac.imag, cfac.real) / (2 * np.pi * 56 * 6 * dt)
                        else:
                            ccb = np.sum(part[ic - 88 * 6:ic - 88 * 6 + SYNC_WAVEFORM_LEN] * SYNC_WAVEFORM_CONJ)
                            cfac = ccb * np.conj(cca)
                            f_error_2 = np.atan2(cfac.imag, cfac.real) / (2 * np.pi * 88 * 6 * dt)

                        # ! Final estimate of the carrier frequency - returned to the calling program
                        freq_est = int(MSK144_CENTER_FREQ + f_error + f_error_2)

                        for idf in range(5):  # frequency jitter
                            if idf == 0:
                                delta_f = 0
                            elif idf % 2 == 0:
                                delta_f = idf
                            else:
                                delta_f = -(idf + 1)

                            # ! Remove fine frequency error
                            subpart = self.shift_freq(part, -(f_error_2 + delta_f), self.sample_rate)
                            # ! place the beginning of frame at index NSPM+1
                            subpart = np.roll(subpart, -(ic - MSK144_NSPM))

                            # ! try each of 7 averaging patterns, hope that one works
                            for avg_pattern in range(8):
                                if avg_pattern == 0:
                                    frame = subpart[MSK144_NSPM: MSK144_NSPM + MSK144_NSPM]
                                elif avg_pattern == 1:
                                    frame = subpart[MSK144_NSPM - 432: MSK144_NSPM - 432 + MSK144_NSPM]
                                    frame = np.roll(frame, 432)  # frame = np.roll(frame, -432)
                                elif avg_pattern == 2:
                                    frame = subpart[2 * MSK144_NSPM - 432: 2 * MSK144_NSPM - 432 + MSK144_NSPM]
                                    frame = np.roll(frame, 432)  # frame = np.roll(frame, -432)
                                elif avg_pattern == 3:
                                    frame = subpart[:MSK144_NSPM]
                                elif avg_pattern == 4:
                                    frame = subpart[2 * MSK144_NSPM: 2 * MSK144_NSPM + MSK144_NSPM]
                                elif avg_pattern == 5:
                                    frame = subpart[:MSK144_NSPM] + subpart[MSK144_NSPM:MSK144_NSPM + MSK144_NSPM]
                                elif avg_pattern == 6:
                                    frame = subpart[MSK144_NSPM: MSK144_NSPM + MSK144_NSPM] + subpart[2 * MSK144_NSPM: 2 * MSK144_NSPM + MSK144_NSPM]
                                elif avg_pattern == 7:
                                    frame = subpart[:MSK144_NSPM] + subpart[MSK144_NSPM:MSK144_NSPM + MSK144_NSPM] + subpart[
                                                                                                                     2 * MSK144_NSPM: 2 * MSK144_NSPM + MSK144_NSPM]

                                if x := self.decode_fame(frame, self.LDPC_ITERATIONS):
                                    status, payload, eye_opening, bit_errors = x

                                    if status.crc_extracted in hashes:
                                        raise NextCand

                                    hashes.add(status.crc_extracted)

                                    df_hv = freq_est - MSK144_CENTER_FREQ

                                    log_item = LogItem(
                                        snr=snr,
                                        dT=t0,
                                        dF=df_hv,
                                        p_avg=avg_pattern + 1,
                                        freq=freq_est,
                                        eye_open=eye_opening,
                                        bit_err=bit_errors,

                                        payload=payload,
                                        crc=status.crc_extracted
                                    )
                                    yield log_item

                                    raise NextCand
            if len(hashes) >= 3:
                break

    def decode(self, tm_slot_start: float) -> typing.Generator[LogItem, None, None]:
        signal_len = min(len(self.signal), 30 * self.sample_rate)

        signal_part = self.signal[:signal_len]

        rms = np.sqrt(np.mean(signal_part ** 2))
        if rms == 0 or np.isnan(rms):
            rms = 1

        part_norm = signal_part / (rms / 2)

        n = int(np.log(signal_len) / np.log(2) + 1)
        nfft = int(min(2 ** n, 1024 ** 2))

        filter_response = self.filter_response(nfft, self.sample_rate)
        part_filtered = self.fourier_bpf(part_norm, nfft, filter_response)

        yield from self.detect_signals(part_filtered, signal_len, tm_slot_start)

    def monitor_process(self, frame: npt.NDArray):
        wave = frame * 0.000390625
        avg = np.mean(wave)
        wave -= avg

        # FIXME: Concat (?)
        self.signal = wave
