import typing
from copy import copy
from dataclasses import dataclass
from itertools import cycle

import numpy as np
import numpy.typing as npt

import multiprocessing as mp

from consts.ftx import *
from crc.ftx import ftx_extract_crc, ftx_check_crc
from encoders import ft4_encode, ft8_encode
from ldpc.ftx import bp_decode
from .monitor import DecodeStatus, AbstractMonitor, LogItem

from numba import jit


@dataclass
class FTXLogItem(LogItem):
    err: int
    payload: typing.ByteString


@dataclass(slots=True)
class Candidate:
    time_offset: int  # < Index of the time block
    freq_offset: int  # < Index of the frequency bin
    time_sub: int  # < Index of the time subdivision used
    freq_sub: int  # < Index of the frequency subdivision used
    score: int = 0  # < Candidate score (non-negative number; higher score means higher likelihood)


class FTXMonitor(AbstractMonitor):
    __slots__ = (
        "symbol_period",  # < FT4/FT8 symbol period in seconds
        "min_bin",  # < First FFT bin in the frequency range (begin)
        "max_bin",  # < First FFT bin outside the frequency range (end)
        "block_size",  # < Number of samples per symbol (block)
        "subblock_size",  # < Analysis shift size (number of samples)
        "nfft",  # < FFT size
        "window",  # < Window function for STFT analysis (nfft samples)
        "last_frame",  # < Current STFT analysis frame (nfft samples)

        # Input structure to ftx_find_sync() function. This structure describes stored waterfall data over the whole message slot.
        # Fields time_osr and freq_osr specify additional oversampling rate for time and frequency resolution.
        # If time_osr=1, FFT magnitude data is collected once for every symbol transmitted, i.e. every 1/6.25 = 0.16 seconds.
        # Values time_osr > 1 mean each symbol is further subdivided in time.
        # If freq_osr=1, each bin in the FFT magnitude data corresponds to 6.25 Hz, which is the tone spacing.
        # Values freq_osr > 1 mean the tone spacing is further subdivided by FFT analysis.

        "num_bins",  # < number of FFT bins in terms of 6.25 Hz
        "time_osr",  # < number of time subdivisions
        "freq_osr",  # < number of frequency subdivisions
        "protocol",  # < Indicate if using FT4 or FT8
        "mag",  # < FFT magnitudes stored as uint8_t[blocks][time_osr][freq_osr][num_bins]
        "max_blocks",  # < number of blocks (symbols) allocated in the mag array
        "num_blocks",  # < number of blocks (symbols) stored in the mag array
        "block_stride"  # < Helper value = time_osr * freq_osr * num_bins
    )

    MIN_SCORE = 5  # Minimum sync score threshold for candidates
    MAX_CANDIDATES = 140
    LDPC_ITERATIONS = 25
    MAX_LDPC_ERRORS = 32

    def __init__(self, f_min: int, f_max: int, sample_rate: int, time_osr: int, freq_osr: int, protocol):
        self.num_blocks = 0
        self.time_osr = time_osr
        self.freq_osr = freq_osr
        self.protocol = protocol

        slot_time = FTX_SLOT_TIMES[protocol]
        symbol_period = FTX_SYMBOL_PERIODS[protocol]

        # Compute DSP parameters that depend on the sample rate
        self.block_size = int(sample_rate * symbol_period)  # samples corresponding to one FSK symbol
        self.subblock_size = int(self.block_size / time_osr)
        self.nfft = self.block_size * freq_osr

        fft_norm = 2.0 / self.nfft
        self.window = fft_norm * np.hanning(self.nfft)

        self.last_frame = np.zeros(self.nfft, dtype=np.float64)

        # Keep only FFT bins in the specified frequency range (f_min/f_max)
        self.min_bin = int(f_min * symbol_period)
        self.max_bin = int(f_max * symbol_period + 1)
        self.num_bins = self.max_bin - self.min_bin

        self.block_stride = (self.time_osr * self.freq_osr * self.num_bins)

        # Allocate enough blocks to fit the entire FT8/FT4 slot in memory
        self.max_blocks = int(slot_time / symbol_period)
        self.mag = np.zeros((self.max_blocks, self.time_osr, self.freq_osr, self.num_bins), dtype=np.int64)

        self.symbol_period = symbol_period

    def monitor_process(self, frame: typing.List[float]):
        # Check if we can still store more waterfall data
        if self.num_blocks >= self.max_blocks:
            return False

        frame_pos = 0

        # Loop over block subdivisions
        for time_sub in range(self.time_osr):
            # Shift the new data into analysis frame
            for pos in range(self.nfft - self.subblock_size):
                self.last_frame[pos] = self.last_frame[pos + self.subblock_size]

            for pos in range(self.nfft - self.subblock_size, self.nfft):
                self.last_frame[pos] = frame[frame_pos]
                frame_pos += 1

            # Do DFT of windowed analysis frame
            time_data = self.last_frame * self.window
            freq_data = np.fft.fft(time_data)[:self.nfft // 2 + 1]

            # Loop over possible frequency OSR offsets
            for freq_sub in range(self.freq_osr):
                for bin in range(self.min_bin, self.max_bin):
                    src_bin = (bin * self.freq_osr) + freq_sub

                    mag2 = freq_data[src_bin].imag ** 2 + freq_data[src_bin].real ** 2
                    db = 10.0 * np.log10(1E-12 + mag2)

                    # Scale decibels to unsigned 8-bit range and clamp the value
                    # Range 0-240 covers -120..0 dB in 0.5 dB steps
                    scaled = int(2 * db + 240)
                    self.mag[self.num_blocks, time_sub, freq_sub, bin - self.min_bin] = scaled

        self.num_blocks += 1

        return True

    def get_candidate_mag_idx(self, candidate: Candidate) -> int:
        offset = candidate.time_offset
        offset = offset * self.time_osr + candidate.time_sub
        offset = offset * self.freq_osr + candidate.freq_sub
        offset = offset * self.num_bins + candidate.freq_offset

        return offset

    @staticmethod
    @jit(nopython=True)
    def ft8_sync_score(mag: npt.NDArray[np.int64], mag_idx: int,
                       time_offset: int, num_blocks: int, block_stride: int) -> int:
        score = 0
        num_average = 0

        # Compute average score over sync symbols (m+k = 0-7, 36-43, 72-79)
        for m in range(FT8_NUM_SYNC):
            block_offset = FT8_SYNC_OFFSET * m

            for i, sm in np.ndenumerate(FT8_COSTAS_PATTERN):  # Index of the expected bin
                k = i[0]

                block = block_offset + k  # relative to the message
                block_abs = time_offset + block  # relative to the captured signal

                if block_abs < 0:  # Check for time boundaries
                    continue

                if block_abs >= num_blocks:
                    break

                # Get the pointer to symbol 'block' of the candidate
                p8 = mag_idx + block * block_stride

                # Check only the neighbors of the expected symbol frequency- and time-wise
                p8sm = p8 + sm

                # look at one frequency bin lower
                if sm > 0:
                    score += mag[p8sm] - mag[p8sm - 1]
                    num_average += 1

                # look at one frequency bin higher
                if sm < 7:
                    score += mag[p8sm] - mag[p8sm + 1]
                    num_average += 1

                # look one symbol back in time
                if k > 0 and block_abs > 0:
                    score += mag[p8sm] - mag[p8sm - block_stride]
                    num_average += 1

                # look one symbol forward in time
                if k + 1 < FT8_LENGTH_SYNC and block_abs + 1 < num_blocks:
                    score += mag[p8sm] - mag[p8sm + block_stride]
                    num_average += 1

        if num_average > 0:
            score = int(score / num_average)

        return score

    @staticmethod
    @jit(nopython=True)
    def ft4_sync_score(mag: npt.NDArray[np.int64], mag_idx: int,
                       time_offset: int, num_blocks: int, block_stride: int) -> int:
        score = 0
        num_average = 0

        # Get the pointer to symbol 0 of the candidate
        # Compute average score over sync symbols (block = 1-4, 34-37, 67-70, 100-103)
        for m in range(FT4_NUM_SYNC):
            block_offset = FT4_SYNC_OFFSET * m + 1

            for k in range(FT4_LENGTH_SYNC):
                block = block_offset + k
                block_abs = time_offset + block

                if block_abs < 0:  # Check for time boundaries
                    continue

                if block_abs >= num_blocks:
                    break

                # Get the pointer to symbol 'block' of the candidate
                p4 = mag_idx + (block * block_stride)

                sm = FT4_COSTAS_PATTERN[m][k]  # Index of the expected bin
                p4sm = p4 + sm

                # Check only the neighbors of the expected symbol frequency- and time-wise
                if sm > 0:
                    # look at one frequency bin lower
                    score += mag[p4sm] - mag[p4sm - 1]
                    num_average += 1

                if sm < 3:
                    # look at one frequency bin higher
                    score += mag[p4sm] - mag[p4sm + 1]
                    num_average += 1

                if k > 0 and block_abs > 0:
                    # look one symbol back in time
                    score += mag[p4sm] - mag[p4sm - block_stride]
                    num_average += 1

                if k + 1 < FT4_LENGTH_SYNC and block_abs + 1 < num_blocks:
                    # look one symbol forward in time
                    score += mag[p4sm] - mag[p4sm + block_stride]
                    num_average += 1

        if num_average > 0:
            score = int(score / num_average)

        return score

    def ftx_sync_score(self, candidate: Candidate) -> int:
        if self.protocol == FTX_PROTOCOL_FT4:
            sync_fun = self.ft4_sync_score
        elif self.protocol == FTX_PROTOCOL_FT8:
            sync_fun = self.ft8_sync_score
        else:
            raise ValueError("Invalid protocol")

        mag_cand = self.get_candidate_mag_idx(candidate)
        return sync_fun(self.mag.ravel(), mag_cand, candidate.time_offset, self.num_blocks, self.block_stride)

    def ftx_find_candidates(self, num_candidates: int, min_score: int, **kwargs) -> typing.List[Candidate]:
        time_offset_range = range(-FTX_LENGTH_SYNC[self.protocol], int(FTX_TIME_RANGE[self.protocol]))

        # Here we allow time offsets that exceed signal boundaries, as long as we still have all data bits.
        # I.e. we can afford to skip the first 7 or the last 7 Costas symbols, as long as we track how many
        # sync symbols we included in the score, so the score is averaged.
        num_tones = FTX_TONES_COUNT[self.protocol]
        period = FTX_SYMBOL_PERIODS[self.protocol]

        if isinstance(f_min := kwargs.get("f_min"), int):
            min_bin = max(0, int(f_min * period) - self.min_bin)
        else:
            min_bin = kwargs.get("kwargs", self.min_bin)

        if isinstance(f_max := kwargs.get("f_max"), int):
            max_bin = min(self.num_bins - num_tones, int(f_max * period + 1) - self.min_bin)
        else:
            max_bin = kwargs.get("max_bin", self.num_bins - num_tones)

        heap = []
        can = Candidate(0, 0, 0, 0)
        for time_sub in range(self.time_osr):
            for freq_sub in range(self.freq_osr):
                for time_offset in time_offset_range:
                    for freq_offset in range(min_bin, max_bin):
                        can.time_sub = time_sub
                        can.freq_sub = freq_sub
                        can.time_offset = time_offset
                        can.freq_offset = freq_offset

                        if (score := self.ftx_sync_score(can)) < min_score:
                            continue

                        candidate = copy(can)
                        candidate.score = score

                        heap.insert(0, candidate)

        heap.sort(key=lambda x: x.score, reverse=True)
        return heap[:num_candidates]

    def ft4_extract_likelihood(self, cand: Candidate) -> npt.NDArray[np.float64]:
        log174 = np.zeros(FTX_LDPC_N, dtype=np.float64)

        mag = self.get_candidate_mag_idx(cand)  # Pointer to 4 magnitude bins of the first symbol

        # Go over FSK tones and skip Costas sync symbols
        for k in range(FT4_ND):
            # Skip either 5, 9 or 13 sync symbols
            # TODO: replace magic numbers with constants
            sym_idx = k + (5 if k < 29 else 9 if k < 58 else 13)
            bit_idx = 2 * k

            # Check for time boundaries
            block = cand.time_offset + sym_idx
            if 0 <= block < self.num_blocks:
                log174[bit_idx:bit_idx + 2] = self.ft4_extract_symbol(mag + sym_idx * self.block_stride)

        return log174

    def ft8_extract_likelihood(self, cand: Candidate) -> npt.NDArray[np.float64]:
        log174 = np.zeros(FTX_LDPC_N, dtype=np.float64)

        mag = self.get_candidate_mag_idx(cand)  # Pointer to 8 magnitude bins of the first symbol

        # Go over FSK tones and skip Costas sync symbols
        for k in range(FT8_ND):
            # Skip either 7 or 14 sync symbols
            # TODO: replace magic numbers with constants
            sym_idx = k + (7 if k < 29 else 14)
            bit_idx = 3 * k

            # Check for time boundaries
            block = cand.time_offset + sym_idx

            if 0 <= block < self.num_blocks:
                log174[bit_idx:bit_idx + 3] = self.ft8_extract_symbol(mag + sym_idx * self.block_stride)

        return log174

    def ftx_extract_symbol(self, gray_map: npt.NDArray[np.int64], mag_idx: int,
                           bit_map: typing.Tuple) -> npt.NDArray[np.float64]:
        # Compute unnormalized log likelihood log(p(1) / p(0)) of n message bits (1 FSK symbol)
        # Cleaned up code for the simple case of n_syms==1
        s2 = self.mag.ravel()[gray_map + mag_idx]

        logl = np.fromiter((
            np.max(s2[np.array(l)]) - np.max(s2[np.array(r)])
            for l, r in bit_map
        ), dtype=np.float64)

        return logl

    def ft4_extract_symbol(self, mag_idx: int) -> npt.NDArray[np.float64]:
        bit_map = (
            ((2, 3), (0, 1)),
            ((1, 3), (0, 2)),
        )

        return self.ftx_extract_symbol(FT4_GRAY_MAP, mag_idx, bit_map)

    def ft8_extract_symbol(self, mag_idx: int) -> npt.NDArray[np.float64]:
        bit_map = (
            ((4, 5, 6, 7), (0, 1, 2, 3)),
            ((2, 3, 6, 7), (0, 1, 4, 5)),
            ((1, 3, 5, 7), (0, 2, 4, 6)),
        )

        return self.ftx_extract_symbol(FT8_GRAY_MAP, mag_idx, bit_map)

    def ftx_decode_candidate(
            self, cand: Candidate,
            max_iterations: int) -> typing.Optional[typing.Tuple[DecodeStatus, typing.Optional[bytes], float]]:
        if self.protocol == FTX_PROTOCOL_FT4:
            log174 = self.ft4_extract_likelihood(cand)
        else:
            log174 = self.ft8_extract_likelihood(cand)

        # Compute the variance of log174
        variance = np.var(log174, dtype=np.float64)
        # Normalize log174 distribution and scale it with experimentally found coefficient
        norm_factor = np.sqrt(24.0 / variance) if variance != 0.0 else 1.0

        log174 *= norm_factor

        ldpc_errors, plain174 = bp_decode(log174, max_iterations)

        if ldpc_errors > 0:
            return None

        # FIXME: Slow code
        # if ldpc_errors > self.MAX_LDPC_ERRORS:
        #     return None
        #
        # if ldpc_errors > 0:
        #     if not (x := osd_decode(log174, 6)):
        #         return None
        #
        #     plain174, got_depth = x
        # EOF Slow code block

        if not ftx_check_crc(plain174):
            return None

        # Extract payload + CRC (first FTX_LDPC_K bits) packed into a byte array
        a91 = self.pack_bits(plain174, FTX_LDPC_K)
        # Extract CRC and check it
        crc_extracted = ftx_extract_crc(a91)

        if self.protocol == FTX_PROTOCOL_FT4:
            # '[..] for FT4 only, in order to avoid transmitting a long string of zeros when sending CQ messages,
            # the assembled 77-bit message is bitwise exclusive-OR’ed with [a] pseudorandom sequence before computing the CRC and FEC parity bits'
            payload = bytearray(a91[i] ^ xor for i, xor in enumerate(FT4_XOR_SEQUENCE))
            tones = ft4_encode(payload)
        else:
            payload = a91
            tones = ft8_encode(payload)

        snr = self.ftx_subtract(cand, tones)
        # snr = self.ftx_get_snr(cand, tones)
        return DecodeStatus(ldpc_errors, crc_extracted), payload, snr

    # def ftx_get_snr(self, candidate: Candidate, tones: typing.Iterable[int]) -> float:
    #     num_tones = FTX_TONES_COUNT[self.protocol]
    #
    #     mag_cand = self.get_candidate_mag_idx(candidate)
    #
    #     signal = 0
    #     noise = 0
    #     num_average = 0
    #
    #     tones = cycle(tones)
    #     for i, tone in enumerate(tones):
    #         block_abs = candidate.time_offset + i  # relative to the captured signal
    #         # Check for time boundaries
    #         if block_abs < 0:
    #             continue
    #
    #         if block_abs >= self.num_blocks:
    #             break
    #
    #         wf_el = mag_cand + i * self.block_stride
    #
    #         min_val = 255
    #         for s in range(num_tones):
    #             wf_mag = self.mag[wf_el + s]
    #
    #             if s == tone:
    #                 signal += wf_mag
    #             else:
    #                 min_val = min(min_val, wf_mag)
    #
    #         noise += min_val
    #         num_average += 1
    #
    #         # Mute
    #         if tone == 0:
    #             self.mag[wf_el + 0] = self.mag[wf_el + 1]
    #         elif tone == 7:
    #             self.mag[wf_el + 7] = self.mag[wf_el + 6]
    #         else:
    #             self.mag[wf_el + tone] = int(self.mag[wf_el + tone + 1] / 2 + self.mag[wf_el + tone - 1] / 2)
    #
    #     return (signal - noise) / (2 * num_average) - 26

    def ftx_subtract(self, candidate: Candidate, tones: typing.Iterable[int]) -> float:
        # Subtract the estimated noise from the signal, given a candidate and a sequence of tones
        # This function takes a candidate and a sequence of tones, and subtracts the estimated noise from the signal.
        # The noise is estimated as the minimum signal power of all tones except the one of the candidate.
        # The signal power is then subtracted from the signal.

        num_tones = FTX_TONES_COUNT[self.protocol]

        can = copy(candidate)
        snr_all = 0.0

        mag = self.mag.ravel()

        tones = cycle(tones)
        for freq_sub in range(self.freq_osr):
            can.freq_sub = freq_sub

            mag_cand = self.get_candidate_mag_idx(can)
            noise = 0.0
            signal = 0.0
            num_average = 0

            for i, tone in enumerate(tones):
                block_abs = candidate.time_offset + i  # relative to the captured signal
                # Check for time boundaries
                if block_abs < 0:
                    continue

                if block_abs >= self.num_blocks:
                    break

                # Get the pointer to symbol 'block' of the candidate
                wf_el = mag_cand + i * self.block_stride

                noise_val = 100000.0
                for s in filter(lambda x: x != tone, range(num_tones)):
                    noise_val = min(noise_val, mag[wf_el + s] * 0.5 - 120.0)

                noise += noise_val
                signal += mag[wf_el + tone] * 0.5 - 120.0
                num_average += 1

            noise /= num_average
            signal /= num_average
            snr = signal - noise

            for i, tone in enumerate(tones):
                block_abs = candidate.time_offset + i  # relative to the captured signal
                # Check for time boundaries
                if block_abs < 0:
                    continue

                if block_abs >= self.num_blocks:
                    break

                # Get the pointer to symbol 'block' of the candidate
                wf_el = mag_cand + i * self.block_stride
                mag[wf_el + tone] -= snr * 2 + 240

            snr_all += snr

        return snr_all / self.freq_osr / 2 - 22

    def decode(self, **kwargs) -> typing.Generator[LogItem, None, None]:
        # Find top candidates by Costas sync score and localize them in time and frequency
        items = set()

        candidate_list = self.ftx_find_candidates(self.MAX_CANDIDATES, self.MIN_SCORE, **kwargs)
        # Go over candidates and attempt to decode messages
        for cand in candidate_list:
            freq_hz = (self.min_bin + cand.freq_offset + cand.freq_sub / self.freq_osr) / self.symbol_period
            time_sec = (cand.time_offset + cand.time_sub / self.time_osr) * self.symbol_period - 0.65

            if not (x := self.ftx_decode_candidate(cand, self.LDPC_ITERATIONS)):
                continue

            status, message, snr = x

            if (crc := status.crc_extracted) in items:
                continue

            items.add(crc)

            yield FTXLogItem(
                snr=snr,
                dT=time_sec,
                dF=freq_hz,
                err=status.ldpc_errors,
                payload=message,
                crc=crc
            )

    def _decode_mp_wrap(self, min_bin: int, max_bin: int):
        return list(self.decode(min_bin=min_bin, max_bin=max_bin))

    def decode_mp(self, pool_size: typing.Optional[int] = None) -> typing.Generator[LogItem, None, None]:
        num_tones = FTX_TONES_COUNT[self.protocol]

        bin_overlap = num_tones * 3
        bin_step = num_tones * 5

        args = [
            (i - bin_overlap, i + bin_overlap)
            for i in range(self.min_bin + bin_overlap, self.max_bin, bin_step)
        ]

        with mp.Pool(processes=pool_size) as pool:
            logs = pool.starmap(self._decode_mp_wrap, args)

            hash_set = set()
            for log in logs:
                for it in log:
                    if not it.crc in hash_set:
                        hash_set.add(it.crc)
                        yield it
