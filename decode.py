import math
import typing
from collections import namedtuple
from copy import copy
from itertools import cycle

import numpy as np

from consts import *
from crc import ftx_extract_crc, ftx_check_crc
from encode import ft4_encode, ft8_encode
from gfsk import M_PI
from ldpc import bp_decode
from message import ftx_message_decode
from osd import osd_decode

kMin_score = 5  # Minimum sync score threshold for candidates
kMax_candidates = 140
kLDPC_iterations = 25
kMaxLDPCErrors = 32


class Waterfall:
    # Input structure to ftx_find_sync() function. This structure describes stored waterfall data over the whole message slot.
    # Fields time_osr and freq_osr specify additional oversampling rate for time and frequency resolution.
    # If time_osr=1, FFT magnitude data is collected once for every symbol transmitted, i.e. every 1/6.25 = 0.16 seconds.
    # Values time_osr > 1 mean each symbol is further subdivided in time.
    # If freq_osr=1, each bin in the FFT magnitude data corresponds to 6.25 Hz, which is the tone spacing.
    # Values freq_osr > 1 mean the tone spacing is further subdivided by FFT analysis.

    def __init__(self, max_blocks: int, num_bins: int, time_osr: int, freq_osr: int, protocol):
        # max_blocks;       ///< number of blocks (symbols) allocated in the mag array
        # num_blocks;       ///< number of blocks (symbols) stored in the mag array
        # num_bins;         ///< number of FFT bins in terms of 6.25 Hz
        # time_osr;         ///< number of time subdivisions
        # freq_osr;         ///< number of frequency subdivisions
        # WF_ELEM_T* mag;   ///< FFT magnitudes stored as uint8_t[blocks][time_osr][freq_osr][num_bins]
        # block_stride;     ///< Helper value = time_osr * freq_osr * num_bins
        # protocol;         ///< Indicate if using FT4 or FT8

        self.max_blocks = max_blocks
        self.num_blocks = 0
        self.num_bins = num_bins
        self.time_osr = time_osr
        self.freq_osr = freq_osr
        self.block_stride = (time_osr * freq_osr * num_bins)
        self.mag = [0] * (max_blocks * time_osr * freq_osr * num_bins)
        # self.mag = np.zeros(max_blocks * time_osr * freq_osr * num_bins, dtype=np.int16)
        self.protocol = protocol


class Candidate:
    # Output structure of ftx_find_sync() and input structure of ftx_decode().
    # Holds the position of potential start of a message in time and frequency.
    def __init__(self, time_offset, freq_offset, time_sub, freq_sub, score=0):
        self.score: int = score  # < Candidate score (non-negative number; higher score means higher likelihood)
        self.time_offset: int = time_offset  # < Index of the time block
        self.freq_offset: int = freq_offset  # < Index of the frequency bin
        self.time_sub: int = time_sub  # < Index of the time subdivision used
        self.freq_sub: int = freq_sub  # < Index of the frequency subdivision used

    def __repr__(self):
        return str(self)

    def __str__(self):
        return (f"[score: {self.score}; "
                f"time_offset: {self.time_offset}; "
                f"freq_offset: {self.freq_offset}; "
                f"time_sub: {self.time_sub}; "
                f"freq_sub: {self.freq_sub}]")


DecodeStatus = namedtuple("DecodeStatus", ["ldpc_errors", "crc_extracted"])


class Monitor:
    # FT4/FT8 monitor object that manages DSP processing of incoming audio data and prepares a waterfall object
    @staticmethod
    def hann_i(i: int, N: int) -> float:
        x = math.sin(M_PI * i / N)
        return x ** 2

    @staticmethod
    def pack_bits(bit_array: typing.ByteString, num_bits: int) -> typing.ByteString:
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

    @staticmethod
    def ftx_normalize_logl(log174: typing.List[float]) -> typing.Generator[float, None, None]:
        # FIXME: Optimize
        # Compute the variance of log174
        sum = 0
        sum2 = 0
        for it in log174:
            sum += it
            sum2 += it ** 2

        inv_n = 1.0 / FTX_LDPC_N
        variance = (sum2 - (sum * sum * inv_n)) * inv_n

        # Normalize log174 distribution and scale it with experimentally found coefficient
        norm_factor = math.sqrt(24.0 / variance)

        for it in log174:
            yield it * norm_factor

    def __init__(self, f_min: int, f_max: int, sample_rate: int, time_osr: int, freq_osr: int, protocol):
        # symbol_period;    ///< FT4/FT8 symbol period in seconds
        # min_bin;          ///< First FFT bin in the frequency range (begin)
        # max_bin;          ///< First FFT bin outside the frequency range (end)
        # block_size;       ///< Number of samples per symbol (block)
        # subblock_size;    ///< Analysis shift size (number of samples)
        # nfft;             ///< FFT size
        # fft_norm;         ///< FFT normalization factor
        # window;           ///< Window function for STFT analysis (nfft samples)
        # last_frame;       ///< Current STFT analysis frame (nfft samples)
        # wf;               ///< Waterfall object
        # max_mag;          ///< Maximum detected magnitude (debug stats)

        slot_time = FT4_SLOT_TIME if protocol == FTX_PROTOCOL_FT4 else FT8_SLOT_TIME
        symbol_period = FT4_SYMBOL_PERIOD if protocol == FTX_PROTOCOL_FT4 else FT8_SYMBOL_PERIOD

        # Compute DSP parameters that depend on the sample rate
        self.block_size = int(sample_rate * symbol_period)  # samples corresponding to one FSK symbol
        self.subblock_size = int(self.block_size / time_osr)
        self.nfft = self.block_size * freq_osr
        self.fft_norm = 2.0 / self.nfft
        # const int len_window = 1.8f * me->block_size; // hand-picked and optimized

        self.window = [self.fft_norm * self.hann_i(i, self.nfft) for i in range(self.nfft)]
        self.last_frame = [0.0] * self.nfft

        # Allocate enough blocks to fit the entire FT8/FT4 slot in memory
        max_blocks = int(slot_time / symbol_period)

        # Keep only FFT bins in the specified frequency range (f_min/f_max)
        self.min_bin = int(f_min * symbol_period)
        self.max_bin = int(f_max * symbol_period + 1)
        num_bins = self.max_bin - self.min_bin

        self.wf = Waterfall(max_blocks, num_bins, time_osr, freq_osr, protocol)

        self.symbol_period = symbol_period

        self.max_mag = -120.0

    def monitor_process(self, frame: typing.List[float]):
        # Check if we can still store more waterfall data
        if self.wf.num_blocks >= self.wf.max_blocks:
            return False

        offset = self.wf.num_blocks * self.wf.block_stride
        frame_pos = 0

        # Loop over block subdivisions
        for time_sub in range(self.wf.time_osr):
            # Shift the new data into analysis frame
            for pos in range(self.nfft - self.subblock_size):
                self.last_frame[pos] = self.last_frame[pos + self.subblock_size]

            for pos in range(self.nfft - self.subblock_size, self.nfft):
                self.last_frame[pos] = frame[frame_pos]
                frame_pos += 1

            # Do DFT of windowed analysis frame
            timedata = [self.window[pos] * self.last_frame[pos] for pos in range(self.nfft)]
            freqdata = np.fft.fft(timedata)[:self.nfft // 2 + 1]

            # Loop over possible frequency OSR offsets
            for freq_sub in range(self.wf.freq_osr):
                for bin in range(self.min_bin, self.max_bin):
                    src_bin = (bin * self.wf.freq_osr) + freq_sub
                    mag2 = freqdata[src_bin].imag ** 2 + freqdata[src_bin].real ** 2
                    db = 10.0 * math.log10(1E-12 + mag2)

                    # Scale decibels to unsigned 8-bit range and clamp the value
                    # Range 0-240 covers -120..0 dB in 0.5 dB steps
                    scaled = int(2 * db + 240)
                    self.wf.mag[offset] = max(min(scaled, 255), 0)
                    offset += 1

                    self.max_mag = max(self.max_mag, db)

        self.wf.num_blocks += 1

        return True

    def ft8_sync_score(self, candidate: Candidate) -> int:
        wf = self.wf

        score = 0
        num_average = 0

        # Get the pointer to symbol 0 of the candidate
        mag_cand = self.get_cand_mag_idx(candidate)

        # Compute average score over sync symbols (m+k = 0-7, 36-43, 72-79)
        for m in range(FT8_NUM_SYNC):
            for k in range(FT8_LENGTH_SYNC):
                block = FT8_SYNC_OFFSET * m + k  # relative to the message
                block_abs = candidate.time_offset + block  # relative to the captured signal

                if block_abs < 0:  # Check for time boundaries
                    continue

                if block_abs >= wf.num_blocks:
                    break

                # Get the pointer to symbol 'block' of the candidate
                p8 = mag_cand + block * wf.block_stride

                # Check only the neighbors of the expected symbol frequency- and time-wise
                sm = kFT8_Costas_pattern[k]  # Index of the expected bin
                p8sm = p8 + sm
                if sm > 0:  # look at one frequency bin lower
                    score += wf.mag[p8sm] - wf.mag[p8sm - 1]
                    num_average += 1
                if sm < 7:  # look at one frequency bin higher
                    score += wf.mag[p8sm] - wf.mag[p8sm + 1]
                    num_average += 1
                if k > 0 and block_abs > 0:  # look one symbol back in time
                    score += wf.mag[p8sm] - wf.mag[p8sm - wf.block_stride]
                    num_average += 1
                if k + 1 < FT8_LENGTH_SYNC and block_abs + 1 < wf.num_blocks:  # look one symbol forward in time
                    score += wf.mag[p8sm] - wf.mag[p8sm + wf.block_stride]
                    num_average += 1

        if num_average > 0:
            score = int(score / num_average)

        # if score != 0:
        #     print(f"ft8_sync_score score={score}")
        return score

    def ft4_sync_score(self, candidate: Candidate) -> int:
        wf = self.wf
        score = 0
        num_average = 0

        # Get the pointer to symbol 0 of the candidate
        mag_cand = self.get_cand_mag_idx(candidate)

        # Compute average score over sync symbols (block = 1-4, 34-37, 67-70, 100-103)
        for m in range(FT4_NUM_SYNC):
            for k in range(FT4_LENGTH_SYNC):
                block = 1 + (FT4_SYNC_OFFSET * m) + k
                block_abs = candidate.time_offset + block
                # Check for time boundaries
                if block_abs < 0:
                    continue
                if block_abs >= wf.num_blocks:
                    break

                # Get the pointer to symbol 'block' of the candidate
                p4 = mag_cand + (block * wf.block_stride)

                sm = kFT4_Costas_pattern[m][k]  # Index of the expected bin
                p4sm = p4 + sm

                # score += (4 * p4[sm]) - p4[0] - p4[1] - p4[2] - p4[3];
                # num_average += 4;

                # Check only the neighbors of the expected symbol frequency- and time-wise
                if sm > 0:
                    # look at one frequency bin lower
                    score += wf.mag[p4sm] - wf.mag[p4sm - 1]
                    num_average += 1
                if sm < 3:
                    # look at one frequency bin higher
                    score += wf.mag[p4sm] - wf.mag[p4sm + 1]
                    num_average += 1
                if k > 0 and block_abs > 0:
                    # look one symbol back in time
                    score += wf.mag[p4sm] - wf.mag[p4sm - wf.block_stride]
                    num_average += 1
                if k + 1 < FT4_LENGTH_SYNC and block_abs + 1 < wf.num_blocks:
                    # look one symbol forward in time
                    score += wf.mag[p4sm] - wf.mag[p4sm + wf.block_stride]
                    num_average += 1

        if num_average > 0:
            score = int(score / num_average)

        return score

    def ftx_find_candidates(self, num_candidates: int, min_score: int) -> typing.List[Candidate]:
        wf = self.wf

        if wf.protocol == FTX_PROTOCOL_FT4:
            num_tones = 4
            time_offset_range = range(
                -FT4_LENGTH_SYNC, int(FT4_SLOT_TIME / FT4_SYMBOL_PERIOD - FT4_NN + FT4_LENGTH_SYNC)
            )

            sync_fun = self.ft4_sync_score
        else:
            num_tones = 8
            time_offset_range = range(
                -FT8_LENGTH_SYNC, int(FT8_SLOT_TIME / FT8_SYMBOL_PERIOD - FT8_NN + FT8_LENGTH_SYNC)
            )

            sync_fun = self.ft8_sync_score

        # Here we allow time offsets that exceed signal boundaries, as long as we still have all data bits.
        # I.e. we can afford to skip the first 7 or the last 7 Costas symbols, as long as we track how many
        # sync symbols we included in the score, so the score is averaged.
        heap = []
        can = Candidate(0, 0, 0, 0)
        for time_sub in range(wf.time_osr):
            for freq_sub in range(wf.freq_osr):
                for time_offset in time_offset_range:
                    # (candidate.freq_offset + num_tones - 1) < wf->num_bin
                    for freq_offset in range(wf.num_bins - num_tones):
                        can.time_sub = time_sub
                        can.freq_sub = freq_sub
                        can.time_offset = time_offset
                        can.freq_offset = freq_offset

                        if (score := sync_fun(can)) < min_score:
                            continue

                        candidate = copy(can)
                        candidate.score = score

                        heap.insert(0, candidate)

        heap.sort(key=lambda x: x.score, reverse=True)
        return heap[:num_candidates]

    def get_cand_mag_idx(self, candidate: Candidate) -> int:
        wf = self.wf

        offset = candidate.time_offset
        offset = offset * wf.time_osr + candidate.time_sub
        offset = offset * wf.freq_osr + candidate.freq_sub
        offset = offset * wf.num_bins + candidate.freq_offset

        return offset

    def ft4_extract_likelihood(self, cand: Candidate) -> typing.List[float]:
        log174 = [0.0] * FTX_LDPC_N

        mag = self.get_cand_mag_idx(cand)  # Pointer to 4 magnitude bins of the first symbol

        # Go over FSK tones and skip Costas sync symbols
        for k in range(FT4_ND):
            # Skip either 5, 9 or 13 sync symbols
            # TODO: replace magic numbers with constants
            sym_idx = k + (5 if k < 29 else 9 if k < 58 else 13)
            bit_idx = 2 * k

            # Check for time boundaries
            block = cand.time_offset + sym_idx
            if block < 0 or block >= self.wf.num_blocks:
                log174[bit_idx + 0] = 0
                log174[bit_idx + 1] = 0
            else:
                logl_0, logl_1 = self.ft4_extract_symbol(mag + sym_idx * self.wf.block_stride)
                log174[bit_idx + 0] = logl_0
                log174[bit_idx + 1] = logl_1

        return log174

    def ft8_extract_likelihood(self, cand: Candidate) -> typing.List[float]:
        log174 = [0.0] * FTX_LDPC_N

        mag = self.get_cand_mag_idx(cand)  # Pointer to 8 magnitude bins of the first symbol

        # Go over FSK tones and skip Costas sync symbols
        for k in range(FT8_ND):
            # Skip either 7 or 14 sync symbols
            # TODO: replace magic numbers with constants
            sym_idx = k + (7 if k < 29 else 14)
            bit_idx = 3 * k

            # Check for time boundaries
            block = cand.time_offset + sym_idx

            if block < 0 or block >= self.wf.num_blocks:
                log174[bit_idx + 0] = 0
                log174[bit_idx + 1] = 0
                log174[bit_idx + 2] = 0
            else:
                logl_0, logl_1, logl_2 = self.ft8_extract_symbol(mag + sym_idx * self.wf.block_stride)
                log174[bit_idx + 0] = logl_0
                log174[bit_idx + 1] = logl_1
                log174[bit_idx + 2] = logl_2

        return log174

    def ft4_extract_symbol(self, mag_idx: int) -> typing.Tuple[float, float]:
        # Compute unnormalized log likelihood log(p(1) / p(0)) of 2 message bits (1 FSK symbol)
        # Cleaned up code for the simple case of n_syms==1
        s2 = [self.wf.mag[mag_idx + kFT4_Gray_map[j]] for j in range(4)]

        logl_0 = max(s2[2], s2[3]) - max(s2[0], s2[1])
        logl_1 = max(s2[1], s2[3]) - max(s2[0], s2[2])

        return logl_0, logl_1

    def ft8_extract_symbol(self, mag_idx: int) -> typing.Tuple[float, float, float]:
        # Compute unnormalized log likelihood log(p(1) / p(0)) of 3 message bits (1 FSK symbol)
        # Cleaned up code for the simple case of n_syms==1
        s2 = [self.wf.mag[mag_idx + kFT8_Gray_map[j]] for j in range(8)]

        logl_0 = max(s2[4], s2[5], s2[6], s2[7]) - max(s2[0], s2[1], s2[2], s2[3])
        logl_1 = max(s2[2], s2[3], s2[6], s2[7]) - max(s2[0], s2[1], s2[4], s2[5])
        logl_2 = max(s2[1], s2[3], s2[5], s2[7]) - max(s2[0], s2[2], s2[4], s2[6])

        return logl_0, logl_1, logl_2

    def ftx_decode_candidate(
            self, cand: Candidate,
            max_iterations: int) -> typing.Optional[typing.Tuple[DecodeStatus, typing.Optional[bytes], float]]:
        wf = self.wf

        if wf.protocol == FTX_PROTOCOL_FT4:
            log174 = self.ft4_extract_likelihood(cand)
        else:
            log174 = self.ft8_extract_likelihood(cand)

        log174 = list(self.ftx_normalize_logl(log174))
        ldpc_errors, plain174 = bp_decode(log174, max_iterations)

        # FIXME: Slow code
        if ldpc_errors > kMaxLDPCErrors:
            return None

        if ldpc_errors > 0:
            if not (x := osd_decode(log174, 6)):
                return None

            plain174, got_depth = x
        # EOF Slow code block

        if not ftx_check_crc(plain174):
            return None

        # Extract payload + CRC (first FTX_LDPC_K bits) packed into a byte array
        a91 = self.pack_bits(plain174, FTX_LDPC_K)
        # Extract CRC and check it
        crc_extracted = ftx_extract_crc(a91)

        if wf.protocol == FTX_PROTOCOL_FT4:
            # '[..] for FT4 only, in order to avoid transmitting a long string of zeros when sending CQ messages,
            # the assembled 77-bit message is bitwise exclusive-OR’ed with [a] pseudorandom sequence before computing the CRC and FEC parity bits'
            payload = bytearray(a91[i] ^ xor for i, xor in enumerate(kFT4_XOR_sequence))
            tones = ft4_encode(payload)
        else:
            payload = a91
            tones = ft8_encode(payload)

        # snr = self.ftx_subtract(cand, tones)
        snr = self.ftx_get_snr(cand, tones)
        return DecodeStatus(ldpc_errors, crc_extracted), payload, snr

    def ftx_get_snr(self, candidate: Candidate, tones: typing.Iterable[int]) -> float:
        n_items = 4 if self.wf.protocol == FTX_PROTOCOL_FT4 else 8

        mag_cand = self.get_cand_mag_idx(candidate)

        signal = 0
        noise = 0
        num_average = 0

        tones = cycle(tones)
        for i, tone in enumerate(tones):
            block_abs = candidate.time_offset + i  # relative to the captured signal
            # Check for time boundaries
            if block_abs < 0:
                continue

            if block_abs >= self.wf.num_blocks:
                break

            wf_el = mag_cand + i * self.wf.block_stride

            min_val = 255
            for s in range(n_items):
                wf_mag = self.wf.mag[wf_el + s]

                if s == tone:
                    signal += wf_mag
                else:
                    min_val = min(min_val, wf_mag)

            noise += min_val
            num_average += 1

            # Mute
            if tone == 0:
                self.wf.mag[wf_el + 0] = self.wf.mag[wf_el + 1]
            elif tone == 7:
                self.wf.mag[wf_el + 7] = self.wf.mag[wf_el + 6]
            else:
                self.wf.mag[wf_el + tone] = int(self.wf.mag[wf_el + tone + 1] / 2 + self.wf.mag[wf_el + tone - 1] / 2)

        return (signal - noise) / (2 * num_average) - 26

    def ftx_subtract(self, candidate: Candidate, tones: typing.Iterable[int]) -> float:
        # Subtract the estimated noise from the signal, given a candidate and a sequence of tones
        # This function takes a candidate and a sequence of tones, and subtracts the estimated noise from the signal.
        # The noise is estimated as the minimum signal power of all tones except the one of the candidate.
        # The signal power is then subtracted from the signal.

        n_items = 4 if self.wf.protocol == FTX_PROTOCOL_FT4 else 8

        can = copy(candidate)
        snr_all = 0.0

        tones = cycle(tones)
        for freq_sub in range(self.wf.freq_osr):
            can.freq_sub = freq_sub

            mag_cand = self.get_cand_mag_idx(can)
            noise = 0.0
            signal = 0.0
            num_average = 0

            for i, tone in enumerate(tones):
                block_abs = candidate.time_offset + i  # relative to the captured signal
                # Check for time boundaries
                if block_abs < 0:
                    continue

                if block_abs >= self.wf.num_blocks:
                    break

                # Get the pointer to symbol 'block' of the candidate
                wf_el = mag_cand + i * self.wf.block_stride

                noise_val = 100000.0
                for s in filter(lambda x: x != tone, range(n_items)):
                    noise_val = min(noise_val, self.wf.mag[wf_el + s] * 0.5 - 120.0)

                noise += noise_val
                signal += self.wf.mag[wf_el + tone] * 0.5 - 120.0
                num_average += 1

            noise /= num_average
            signal /= num_average
            snr = signal - noise

            for i, tone in enumerate(tones):
                block_abs = candidate.time_offset + i  # relative to the captured signal
                # Check for time boundaries
                if block_abs < 0:
                    continue

                if block_abs >= self.wf.num_blocks:
                    break

                # Get the pointer to symbol 'block' of the candidate
                wf_el = mag_cand + i * self.wf.block_stride
                self.wf.mag[wf_el + tone] -= snr * 2 + 240

            snr_all += snr
            # print(
            #     f"Freq: {candidate.freq_offset} Noise: {noise}, Signal: {signal}, SNR: {snr} score: {candidate.score}"
            # )

        return snr_all / self.wf.freq_osr

    def decode(self, tm_slot_start) -> typing.Generator[typing.Tuple[float, float, float, str], None, None]:
        hashes = set()

        # Find top candidates by Costas sync score and localize them in time and frequency
        wf = self.wf

        candidate_list = self.ftx_find_candidates(kMax_candidates, kMin_score)
        # Go over candidates and attempt to decode messages
        for i, cand in enumerate(candidate_list):
            # print(f"{i}\t{cand.score}\t{cand.time_offset}\t{cand.freq_offset}\t{cand.time_sub}\t{cand.freq_sub}")
            # continue
            freq_hz = (self.min_bin + cand.freq_offset + cand.freq_sub / wf.freq_osr) / self.symbol_period
            time_sec = (cand.time_offset + cand.time_sub / wf.time_osr) * self.symbol_period - 0.65

            if not (x := self.ftx_decode_candidate(cand, kLDPC_iterations)):
                continue

            status, message, snr = x

            if (crc := status.crc_extracted) in hashes:
                continue

            hashes.add(crc)

            call_to_rx, call_de_rx, extra_rx = ftx_message_decode(message)

            yield snr, time_sec, freq_hz, " ".join([call_to_rx, call_de_rx or "", extra_rx or ""])
