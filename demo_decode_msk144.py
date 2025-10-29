import time

from scipy.io.wavfile import read

from decoders import MSK144Monitor
from message import message_decode


def main():
    sample_rate, data = read("examples/signal.wav")

    mon = MSK144Monitor(sample_rate=sample_rate)
    mon.monitor_process(data)

    ts1 = time.monotonic()

    for it in mon.decode(0):
        msg = message_decode(it.payload)
        msg_text = " ".join(it for it in msg if isinstance(it, str))

        print(
            f"dB: {it.snr:.3f}\t"
            f"T: {it.dT:.3f}\t"
            f"DF: {it.dF:.3f}\t"
            f"Averaging pattern: {it.p_avg}\t"
            f"Freq: {it.freq}\t"
            f"Eye opening: {it.eye_open:.3f}\t"
            f"Bit errors: {it.bit_err}\t"
            f"CRC: {it.crc}\t"
            f"Message text: {msg_text}"
        )

    ts2 = time.monotonic()

    print("-" * 20, "decoded @", ts2 - ts1, "sec")


if __name__ == '__main__':
    # python -m cProfile -s time demo_decode_msk144.py
    main()
