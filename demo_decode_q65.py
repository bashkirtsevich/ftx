import time
from scipy.io.wavfile import read

from decoders import Q65Monitor
from msg.message import MsgServer


def main():
    sample_rate, data = read("examples/signal.wav")

    msg_svr = MsgServer()

    mon = Q65Monitor(q65_type=2, period=15, eme_delay=False)
    mon.monitor_process(data)

    ts1 = time.monotonic()

    for it in mon.decode(f0=1000, eme_delay=False):
        msg = msg_svr.decode(it.payload)

        print(
            f"dB: {it.snr:.3f}\t"
            f"T: {it.dT:.3f}\t"
            f"DF: {it.dF:.3f}\t"
            f"CRC: {it.crc}\t"
            f"Message text: {msg}"
        )

    ts2 = time.monotonic()

    print("-" * 20, "decoded @", ts2 - ts1, "sec")


if __name__ == '__main__':
    # python -m cProfile -s time demo_decode_q65.py
    main()
