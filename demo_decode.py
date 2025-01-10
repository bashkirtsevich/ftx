import math
from time import clock_gettime, CLOCK_REALTIME, gmtime

import numpy as np
from scipy.io.wavfile import read

from consts import FT4_SLOT_TIME, FT8_SLOT_TIME, FTX_PROTOCOL_FT8
from decode import decode
from monitor import monitor_process, monitor_init

kFreq_osr = 2  # Frequency oversampling rate (bin subdivision)
kTime_osr = 2  # Time oversampling rate (symbol subdivision)


def main():
    time_shift = 0.8

    is_ft4 = False
    slot_period = FT4_SLOT_TIME if is_ft4 else FT8_SLOT_TIME

    is_live = False

    # signal = gen_signal(is_ft4)
    sample_rate, data = read("examples/ft8.wav")
    # sample_rate, data = read("210703_133430.wav")

    num_samples = int(slot_period * sample_rate)

    amplitude = np.iinfo(np.int16).max
    signal = data / amplitude

    protocol = FTX_PROTOCOL_FT8

    mon = monitor_init(
        f_min=200,
        f_max=3000,
        sample_rate=sample_rate,
        time_osr=kTime_osr,
        freq_osr=kFreq_osr,
        protocol=protocol
    )

    while True:
        tm_slot_start = 0
        if is_live:
            # Wait for the start of time slot
            while True:
                time = clock_gettime(CLOCK_REALTIME)
                time_within_slot = math.fmod(time - time_shift, slot_period)
                if time_within_slot > slot_period / 4:
                    # audio_read(signal, mon.block_size) # Read data from soundcard
                    pass
                else:
                    time_slot_start = time - time_within_slot
                    tm_slot_start = gmtime(time_slot_start)
                    # LOG(LOG_INFO, "Time within slot %02d%02d%02d: %.3f s\n", tm_slot_start.tm_hour,
                    #     tm_slot_start.tm_min, tm_slot_start.tm_sec, time_within_slot)
                    break

        # Process and accumulate audio data in a monitor/waterfall instance
        for frame_pos in range(0, num_samples - mon.block_size, mon.block_size):
            # frame_pos = 0
            # while frame_pos + mon.block_size <= num_samples:
            #     frame_pos += mon.block_size
            # if (dev_name != NULL) # Read audio from soundcard
            #     audio_read(signal + frame_pos, mon.block_size);
            # LOG(LOG_DEBUG, "Frame pos: %.3fs\n", (float)(frame_pos + mon.block_size) / sample_rate);
            # print("#")
            # Process the waveform data frame by frame - you could have a live loop here with data from an audio device
            monitor_process(mon, signal[frame_pos:frame_pos + mon.block_size])

        # LOG(LOG_DEBUG, "Waterfall accumulated %d symbols\n", mon.wf.num_blocks);
        print(f"Waterfall accumulated {mon.wf.num_blocks} symbols")
        # LOG(LOG_INFO, "Max magnitude: %.1f dB\n", mon.max_mag);
        print(f"Max magnitude: {mon.max_mag} dB")

        # Decode accumulated data (containing slightly less than a full time slot)
        decode(mon, tm_slot_start)

        # Reset internal variables for the next time slot
        # mon = monitor_reset(mon)
        mon.wf.num_blocks = 0
        mon.max_mag = -120.0

        if not is_live:
            break


if __name__ == '__main__':
    main()
