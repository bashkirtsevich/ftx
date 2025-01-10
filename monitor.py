import math
import typing

import numpy as np

from consts import FT4_SLOT_TIME, FT4_SYMBOL_PERIOD, FTX_PROTOCOL_FT4, FT8_SLOT_TIME, FT8_SYMBOL_PERIOD
from decode import monitor_t, ftx_waterfall_t
from gfsk import M_PI


def hann_i(i: int, N: int) -> float:
    x = math.sin(M_PI * i / N)
    return x ** 2


def waterfall_init(max_blocks: int, num_bins: int, time_osr: int, freq_osr: int) -> ftx_waterfall_t:
    me = ftx_waterfall_t()
    # size_t mag_size = max_blocks * time_osr * freq_osr * num_bins * sizeof(me->mag[0])
    me.max_blocks = max_blocks
    me.num_blocks = 0
    me.num_bins = num_bins
    me.time_osr = time_osr
    me.freq_osr = freq_osr
    me.block_stride = (time_osr * freq_osr * num_bins)
    me.mag = [0] * (max_blocks * time_osr * freq_osr * num_bins)
    # LOG(LOG_DEBUG, "Waterfall size = %zu\n", mag_size)
    return me


def monitor_init(f_min, f_max, sample_rate, time_osr, freq_osr, protocol) -> monitor_t:
    me = monitor_t()

    slot_time = FT4_SLOT_TIME if protocol == FTX_PROTOCOL_FT4 else FT8_SLOT_TIME
    symbol_period = FT4_SYMBOL_PERIOD if protocol == FTX_PROTOCOL_FT4 else FT8_SYMBOL_PERIOD
    # Compute DSP parameters that depend on the sample rate
    me.block_size = int(sample_rate * symbol_period)  # samples corresponding to one FSK symbol
    me.subblock_size = int(me.block_size / time_osr)
    me.nfft = me.block_size * freq_osr
    me.fft_norm = 2.0 / me.nfft
    # const int len_window = 1.8f * me->block_size; // hand-picked and optimized

    me.window = [me.fft_norm * hann_i(i, me.nfft) for i in range(me.nfft)]
    me.last_frame = [0.0] * me.nfft

    # Allocate enough blocks to fit the entire FT8/FT4 slot in memory
    max_blocks = int(slot_time / symbol_period)
    # Keep only FFT bins in the specified frequency range (f_min/f_max)
    me.min_bin = int(f_min * symbol_period)
    me.max_bin = int(f_max * symbol_period + 1)
    num_bins = me.max_bin - me.min_bin

    me.wf = waterfall_init(max_blocks, num_bins, time_osr, freq_osr);
    me.wf.protocol = protocol

    me.symbol_period = symbol_period

    me.max_mag = -120.0

    return me


def monitor_process(me: monitor_t, frame: typing.List[float]):
    # Check if we can still store more waterfall data
    if me.wf.num_blocks >= me.wf.max_blocks:
        return

    offset = me.wf.num_blocks * me.wf.block_stride
    frame_pos = 0

    # Loop over block subdivisions
    for time_sub in range(me.wf.time_osr):
        # Shift the new data into analysis frame
        for pos in range(me.nfft - me.subblock_size):
            me.last_frame[pos] = me.last_frame[pos + me.subblock_size]

        for pos in range(me.nfft - me.subblock_size, me.nfft):
            me.last_frame[pos] = frame[frame_pos]
            frame_pos += 1

        # Do DFT of windowed analysis frame
        timedata = [me.window[pos] * me.last_frame[pos] for pos in range(me.nfft)]
        # timedata = [me.last_frame[pos] for pos in range(me.nfft)]
        # window = np.hanning(me.nfft) * me.fft_norm
        freqdata = np.fft.fft(timedata)[:me.nfft // 2 + 1]

        # Loop over possible frequency OSR offsets
        for freq_sub in range(me.wf.freq_osr):
            for bin in range(me.min_bin, me.max_bin):
                src_bin = (bin * me.wf.freq_osr) + freq_sub
                mag2 = freqdata[src_bin].imag ** 2 + freqdata[src_bin].real ** 2
                db = 10.0 * math.log10(1E-12 + mag2)

                # Scale decibels to unsigned 8-bit range and clamp the value
                # Range 0-240 covers -120..0 dB in 0.5 dB steps
                scaled = int(2 * db + 240)
                me.wf.mag[offset] = 0 if scaled < 0 else 255 if scaled > 255 else scaled
                offset += 1

                if db > me.max_mag:
                    me.max_mag = db

    me.wf.num_blocks += 1
