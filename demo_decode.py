import time

import numpy as np
from scipy.io.wavfile import read

from consts import FTX_PROTOCOL_FT8, FTX_PROTOCOL_FT4
from decode import Monitor

kFreq_osr = 2  # Frequency oversampling rate (bin subdivision)
kTime_osr = 4  # Time oversampling rate (symbol subdivision)


def main():
    # time_shift = 0.8

    is_ft4 = False

    sample_rate, data = read("examples/example1.wav")
    # sample_rate, data = read("examples/7signals.wav")
    # sample_rate, data = read("210703_133430.wav")

    amplitude = np.iinfo(data.dtype).max
    signal = data / amplitude

    protocol = FTX_PROTOCOL_FT4 if is_ft4 else FTX_PROTOCOL_FT8

    mon = Monitor(
        f_min=200,
        f_max=3000,
        sample_rate=sample_rate,
        time_osr=kTime_osr,
        freq_osr=kFreq_osr,
        protocol=protocol
    )

    frame_pos = 0
    while True:
        eof = frame_pos >= len(signal) - mon.block_size

        if eof or not mon.monitor_process(signal[frame_pos:frame_pos + mon.block_size]):
            print(f"Waterfall accumulated {mon.wf.num_blocks} symbols")
            print(f"Max magnitude: {mon.max_mag:+.2f} dB")

            tm_slot_start = 0
            ts1 = time.monotonic()
            for i, (snr, time_sec, freq_hz, text) in enumerate(mon.decode(tm_slot_start)):
                # Fake WSJT-X-like output for now
                print(
                    f"{i + 1:03}\t"
                    f"{snr:+06.2f}dB\t"
                    f"{time_sec:-.2f}sec\t"
                    f"{freq_hz:.2f}Hz\t"
                    f"{text}"
                )

            mon.wf.num_blocks = 0
            mon.max_mag = -120.0

            ts2 = time.monotonic()

            print("-" * 20, "decoded @", ts2 - ts1, "sec")

        if eof:
            break

        frame_pos += mon.block_size


if __name__ == '__main__':
    # python -m cProfile -s time demo_decode.py
    main()
