import itertools
import operator
from functools import cache

# import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import json
from scipy.io.wavfile import read

from crc.mskx import mskx_check_crc, mskx_extract_crc
from encode import msk144_encode
from ldpc_mskx import bp_decode
from consts.mskx import *
import typing
from numba import jit

from decode import msk_filter_response, fourier_bpf, shift_freq
from message import message_decode
from collections import namedtuple

# ss_msk144ms = False

kMaxLDPCErrors = 18
kLDPC_iterations = 10

DecodeStatus = namedtuple("DecodeStatus", ["ldpc_errors", "crc_extracted"])

MSK144_BITS_COUNT = 144
# s8ms = [0, 1, 0, 0, 1, 1, 1, 0]
MSK144_SYNC = [0, 1, 1, 1, 0, 0, 1, 0]

pp_msk144 = np.sin([i * np.pi / 12 for i in range(12)])
rcw_msk144 = (1 - np.cos([i * np.pi / 12 for i in range(12)])) / 2

# s_msk144_2s8 = np.array([2 * s8 - 1 for s8 in (s8ms if ss_msk144ms else s8)])
sync_words = np.array([2 * s8 - 1 for s8 in MSK144_SYNC])

sync_I = np.array([
    pp * sync_words[j * 2 + 1]
    for j in range(4)
    for pp in pp_msk144
])

sync_Q = np.array([
    pp * sync_words[j * 2]
    for j in range(4)
    for pp in (pp_msk144[6:] if j == 0 else pp_msk144)
])

SYNC_WAVEFORM = np.array([complex(sync_I[i], sync_Q[i]) for i in range(42)])


def pack_bits(bit_array: npt.NDArray[np.uint8], num_bits: int) -> typing.ByteString:
    # Packs a string of bits each represented as a zero/non-zero byte in plain[],
    # as a string of packed bits starting from the MSB of the first byte of packed[]
    num_bytes = (num_bits + 7) // 8
    packed = bytearray(b"\x00" * num_bytes)

    mask = 0x80
    byte_idx = 0
    for i in range(num_bits):
        if bit_array[i]:
            packed[byte_idx] |= mask

        mask >>= 1
        if not mask:
            mask = 0x80
            byte_idx += 1

    return packed


def msk144_decode_fame(frame: npt.NDArray[np.complex128], max_iterations: int):
    cca = sum(frame[:len(SYNC_WAVEFORM)] * np.conj(SYNC_WAVEFORM))
    ccb = sum(frame[56 * 6: 56 * 6 + len(SYNC_WAVEFORM)] * np.conj(SYNC_WAVEFORM))
    cc = cca + ccb

    phase_0 = np.atan2(cc.imag, cc.real)

    NSPM = 864

    fac = complex(np.cos(phase_0), np.sin(phase_0))
    frame *= fac.conjugate()

    soft_bits = np.zeros(MSK144_BITS_COUNT, dtype=np.float64)
    hard_bits = np.zeros(MSK144_BITS_COUNT, dtype=np.int64)

    for i in range(6):
        soft_bits[0] += frame[i].imag * pp_msk144[i + 6]
        soft_bits[0] += frame[i + (NSPM - 5 - 1)].imag * pp_msk144[i]

    for i in range(12):
        soft_bits[1] += frame[i].real * pp_msk144[i]

    for i in range(1, 72):
        sum_01 = 0
        for j in range(12):
            sum_01 += frame[i * 12 - 6 + j].imag * pp_msk144[j]
        soft_bits[2 * i] = sum_01

        sum_01 = 0
        for j in range(12):
            sum_01 += frame[i * 12 + j].real * pp_msk144[j]
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
        bad_sync_1 += (2 * hard_bits[i] - 1) * sync_words[i]
        bad_sync_2 += ((2 * hard_bits[i + 57 - 1] - 1) * sync_words[i])

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
        ssig = 1.0

    for i in range(MSK144_BITS_COUNT):
        soft_bits[i] = soft_bits[i] / ssig

    sigma = 0.60  # 0.75 if ss_msk144ms else 0.60

    lratio = np.concat([soft_bits[8:9 + 47], soft_bits[64:65 + 80 - 1]])
    llr = 2 * lratio / (sigma * sigma)

    ldpc_errors, plain128 = bp_decode(llr, max_iterations)

    if ldpc_errors > kMaxLDPCErrors:
        return None

    if not mskx_check_crc(plain128):
        return None

    # Extract payload + CRC (first FTX_LDPC_K bits) packed into a byte array
    payload = pack_bits(plain128, MSKX_LDPC_K)
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

    bit_errors = np.count_nonzero(hard_bits != msg_bits)

    eye_opening = eye_top - eye_bot

    return DecodeStatus(ldpc_errors, crc_extracted), payload, eye_opening, bit_errors


def detect_msk144(signal: np.typing.ArrayLike, n: int, start: float, sample_rate: int,
                  max_cand: int = 16, tolerance: int = 150):
    NSPM = 864
    NPTS = 3 * NSPM
    NFFT = 864

    rx_freq = 1500

    # ! define half-sine pulse and raised-cosine edge window
    dt_msk144 = 1 / sample_rate

    df = sample_rate / NFFT

    n_step_size = 216
    n_steps = (n - NPTS) // n_step_size

    nfhi = 2 * (rx_freq + 500)
    nflo = 2 * (rx_freq - 500)

    ihlo_msk144 = int((nfhi - 2 * tolerance) / df)
    ihhi_msk144 = int((nfhi + 2 * tolerance) / df)

    illo_msk144 = int((nflo - 2 * tolerance) / df)
    ilhi_msk144 = int((nflo + 2 * tolerance) / df)

    i2000_msk144 = int(nflo / df)
    i4000_msk144 = int(nfhi / df)

    times = np.zeros(max_cand, dtype=np.float64)
    freq_errs = np.zeros(max_cand, dtype=np.float64)
    snrs = np.zeros(max_cand, dtype=np.float64)

    detmet = np.zeros(n_steps)
    detmet2 = np.zeros(n_steps)
    detfer = np.full(n_steps, -999.99)

    steps_real = 0
    for step in range(n_steps):
        part_start = n_step_size * step
        part_end = part_start + NSPM

        if part_end > n:
            break

        part = signal[part_start: part_end]

        # Coarse carrier frequency sync - seek tones at 2000 Hz and 4000 Hz in
        # squared signal spectrum.
        # search range for coarse frequency error is +/- 100 Hz
        part = part ** 2
        part[:12] = part[:12] * rcw_msk144
        part[NSPM - 12:NSPM] = part[NSPM - 12:NSPM] * rcw_msk144[::-1]  # Looks like window smooth function

        spec = np.fft.fft(part, NFFT)
        amps = np.abs(spec) ** 2

        # MAD: Find index with max value in amps[ihlo_msk144:ihhi_msk144]
        ##########
        h_peak = ihlo_msk144 + np.argmax(amps[ihlo_msk144:ihhi_msk144])

        delta_hi = -((spec[h_peak - 1] - spec[h_peak + 1]) / (
                2 * spec[h_peak] - spec[h_peak - 1] - spec[h_peak + 1])).real
        mag_hi = amps[h_peak]

        ahavp = (np.sum(amps[ihlo_msk144:ihhi_msk144]) - mag_hi) / (ihhi_msk144 - ihlo_msk144)
        trath = mag_hi / (ahavp + 0.01)
        ##########
        l_peak = illo_msk144 + np.argmax(amps[illo_msk144:ilhi_msk144])

        delta_lo = -((spec[l_peak - 1] - spec[l_peak + 1]) / (
                2 * spec[l_peak] - spec[l_peak - 1] - spec[l_peak + 1])).real
        mag_lo = amps[l_peak]

        alavp = (np.sum(amps[illo_msk144:ilhi_msk144]) - mag_lo) / (ilhi_msk144 - illo_msk144)
        tratl = mag_lo / (alavp + 0.01)
        ##########

        freq_err_h = (h_peak + delta_hi - i4000_msk144) * df / 2
        freq_err_l = (l_peak + delta_lo - i2000_msk144) * df / 2

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
    if xmedian == 0.0:
        xmedian = 1.0

    detmet[:steps_real] = detmet[:steps_real] / xmedian

    count_cand = 0
    for ip in range(max_cand):
        il = np.argmax(detmet[:steps_real])

        if detmet[il] < 3.5:
            break

        if abs(detfer[il]) <= tolerance:
            times[count_cand] = ((il - 0) * n_step_size + NSPM / 2) * dt_msk144
            freq_errs[count_cand] = detfer[il]
            snrs[count_cand] = 12 * np.log10(detmet[il]) / 2 - 9

            count_cand += 1

        detmet[il] = 0.0

    if count_cand < 3:  # for Tropo/ES
        for ip in range(max_cand - count_cand):
            if ip >= max_cand - count_cand:
                break

            # Find candidates
            il = np.argmax(detmet2[:steps_real])

            if detmet2[il] < 12.0:
                break

            if abs(detfer[il]) <= tolerance:
                times[count_cand] = ((il - 0) * n_step_size + NSPM / 2) * dt_msk144
                freq_errs[count_cand] = detfer[il]
                snrs[count_cand] = 12 * np.log10(detmet2[il]) / 2 - 9

                count_cand += 1

            detmet2[il] = 0

    if count_cand > 0:
        indices = np.argsort(times[:count_cand])

    # ! Try to sync/demod/decode each candidate.
    for iip in range(count_cand):
        ip = indices[iip]
        imid = int(times[ip] * sample_rate)

        if imid < NPTS / 2:
            imid = NPTS // 2
        if imid > n - NPTS / 2:
            imid = n - NPTS // 2

        t0 = times[ip] + dt_msk144 * (start)

        part = signal[imid - NPTS // 2: imid + NPTS // 2]

        f_error = freq_errs[ip]
        snr = 2.0 * int(snrs[ip] / 2.0)
        snr = max(-4.0, min(24.0, snr))

        # ! remove coarse freq error - should now be within a few Hz
        part = shift_freq(part, -(rx_freq + f_error), sample_rate)

        cc1 = np.zeros(NPTS, dtype=np.complex128)
        cc2 = np.zeros(NPTS, dtype=np.complex128)

        for i in range(NPTS - (56 * 6 + 42)):
            cc1[i] = sum(part[i: i + len(SYNC_WAVEFORM)] * np.conj(SYNC_WAVEFORM))
            cc2[i] = sum(part[i + 56 * 6: i + 56 * 6 + len(SYNC_WAVEFORM)] * np.conj(SYNC_WAVEFORM))

        dd = abs(cc1) * abs(cc2)

        # ! Find 6 largest peaks
        peaks = []
        for ipk in range(6):
            # HV Good work cc ic1 no dd and ic2
            ic2 = np.argmax(dd)
            dd[max(0, ic2 - 7): min(NPTS - 56 * 6 - 42, ic2 + 7)] = 0.0
            peaks.append(ic2)

        # ! we want ic to be the index of the first sample of the frame
        for ic0 in peaks:  # ic0=peaks[ipk]
            # ! fine adjustment of sync index
            # ! bb lag used to place the sampling index at the center of the eye
            bb = np.zeros(6, dtype=np.complex128)
            for i in range(6):
                cd_b = ic0 + i
                if ic0 + 11 + NSPM < NPTS:
                    bb[i] = np.sum((part[cd_b + 6: cd_b + 6 + NSPM: 6] * np.conj(part[cd_b:cd_b + NSPM:6])) ** 2)
                else:
                    bb[i] = np.sum((part[cd_b + 6: NPTS: 6] * np.conj(part[cd_b:NPTS - 6:6])) ** 2)

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
                    ic = ic + NSPM

                # ! Estimate fine frequency error.
                # ! Should a larger separation be used when frames are averaged?
                cca = np.sum(part[ic:ic + len(SYNC_WAVEFORM)] * np.conj(SYNC_WAVEFORM))
                if ic + 56 * 6 + 42 < NPTS:
                    ccb = np.sum(part[ic + 56 * 6:ic + 56 * 6 + len(SYNC_WAVEFORM)] * np.conj(SYNC_WAVEFORM))
                    cfac = ccb * np.conj(cca)
                    f_error_2 = np.atan2(cfac.imag, cfac.real) / (2 * np.pi * 56 * 6 * dt_msk144)
                else:
                    ccb = np.sum(part[ic - 88 * 6:ic - 88 * 6 + len(SYNC_WAVEFORM)] * np.conj(SYNC_WAVEFORM))
                    cfac = ccb * np.conj(cca)
                    f_error_2 = np.atan2(cfac.imag, cfac.real) / (2 * np.pi * 88 * 6 * dt_msk144)

                # ! Final estimate of the carrier frequency - returned to the calling program
                freq_est = int(rx_freq + f_error + f_error_2)

                for idf in range(5):  # frequency jitter
                    if idf == 0:
                        delta_f = 0
                    elif idf % 2 == 0:
                        delta_f = idf
                    else:
                        delta_f = -(idf + 1)

                    # ! Remove fine frequency error
                    subpart = shift_freq(part, -(f_error_2 + delta_f), sample_rate)
                    # ! place the beginning of frame at index NSPM+1
                    subpart = np.roll(subpart, -(ic - NSPM))

                    # ! try each of 7 averaging patterns, hope that one works
                    for avg_pattern in range(8):
                        if avg_pattern == 0:
                            frame = subpart[NSPM: NSPM + NSPM]
                        elif avg_pattern == 1:
                            frame = subpart[NSPM - 432: NSPM - 432 + NSPM]
                            frame = np.roll(frame, 432)  # frame = np.roll(frame, -432)
                        elif avg_pattern == 2:
                            frame = subpart[2 * NSPM - 432: 2 * NSPM - 432 + NSPM]
                            frame = np.roll(frame, 432)  # frame = np.roll(frame, -432)
                        elif avg_pattern == 3:
                            frame = subpart[:NSPM]
                        elif avg_pattern == 4:
                            frame = subpart[2 * NSPM: 2 * NSPM + NSPM]
                        elif avg_pattern == 5:
                            frame = subpart[:NSPM] + subpart[NSPM:NSPM + NSPM]
                        elif avg_pattern == 6:
                            frame = subpart[NSPM: NSPM + NSPM] + subpart[2 * NSPM: 2 * NSPM + NSPM]
                        elif avg_pattern == 7:
                            frame = subpart[:NSPM] + subpart[NSPM:NSPM + NSPM] + subpart[2 * NSPM:2 * NSPM + NSPM]

                        if x := msk144_decode_fame(frame, kLDPC_iterations):
                            status, payload, eye_opening, bit_errors = x

                            df_hv = freq_est - rx_freq

                            print("dB:", snr)
                            print("T:", t0)
                            print("DF:", df_hv)
                            print("Averaging pattern:", avg_pattern + 1)
                            print("Freq:", freq_est)

                            print("Eye opening:", eye_opening)
                            print("Bit errors:", bit_errors)

                            msg = message_decode(payload)
                            print("Msg:", " ".join(s for s in msg if isinstance(s, str)))
                            print("CRC:", status.crc_extracted)

                            return


#                         if nsuccess > 0:
#                             ndupe=0
#                             for (int im = 0; im<nmessages; im++)
#                                 if ( allmessages[im] == msgreceived ) ndupe=1
#                             ncorrected = 0
#                             eyeopening = 0.0
#                             msk144signalquality(c,snr,fest,t0,softbits,msgreceived,s_HisCall,ncorrected,
#                                                 eyeopening,s_trained_msk144,s_pcoeffs_msk144,false)
#                             if  ndupe == 0 and nmessages<20 :
#                                 if f_only_one_color:
#                                     f_only_one_color = false
#                                     SetBackColor()
#                                 allmessages[nmessages]=msgreceived
#                                 int df_hv = fest-nrxfreq#int df_hv = freq2-nrxfreq+idf1;
#                                 QStringList list;
#                                 list <<s_time<<QString("%1").arg(t0,0,'f',1)<<""<<
#                                 QString("%1").arg((int)snr)<<""<<QString("%1").arg((int)df_hv)
#                                 <<msgreceived<<QString("%1").arg(iav+1)  //navg +1 za tuk  no ne 0 hv 1.31
#                                 <<QString("%1").arg(ncorrected)<<QString("%1").arg(eyeopening,0,'f',1)
#                                 <<QString(ident)+" "+QString("%1").arg((int)fest);
#                                 if (ss_msk144ms)
#                                     list.replace(2,str_round_20ms(p_duration));
#                                     list.replace(4,GetStandardRPT(p_duration,snr));
#                                 SetDecodetTextMsk2DL(list);//2.46
#                                 nmessages++;
#                                 if (s_mousebutton == 0) // && t2!=0.0 1.32 ia is no real 0.0   mousebutton Left=1, Right=3 fullfile=0 rtd=2
#                                     emit EmitDecLinesPosToDisplay(nmessages,t0,t0,s_time);
#                             goto c999;
#         msgreceived=""
# c999:
#         if ( nmessages >= 3 )// nai mnogo 3 razli4ni
#             return


def msk144_decode(signal: npt.NDArray[np.float64], s_istart: float, sample_rate: int):
    npts = min(len(signal), 30 * sample_rate)

    part = signal[:npts]

    rms = np.sqrt(np.mean(part ** 2))
    if rms == 0 or np.isnan(rms):
        rms = 1

    dat = part / (rms / 2)

    n = int(np.log(npts) / np.log(2) + 1)

    nfft = int(min(2 ** n, 1024 ** 2))

    response = msk_filter_response(nfft, sample_rate)
    c = fourier_bpf(dat, nfft, response)
    detect_msk144(c, npts, s_istart, sample_rate)


def read_wav(path: str) -> typing.Tuple[int, npt.NDArray]:
    sample_rate, raw = read(path)

    wave = raw * 0.000390625
    avg = np.mean(wave)
    wave -= avg

    return sample_rate, wave


wav_path = "/home/mad/projects/MSHV_2763/bin/RxWavs/msk144-cq-r9feu-lo87.wav"
# wav_path = "/home/mad/projects/MSHV_2763/bin/RxWavs/msk144.wav"
sample_rate, wave = read_wav(wav_path)
msk144_decode(wave, 0, sample_rate)
