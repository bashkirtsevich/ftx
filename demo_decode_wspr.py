import time
from scipy.io.wavfile import read

from decoders.wspr import WSPRMonitor
from msg.message import WSPRMsgServer


def main():
    sample_rate, data = read("examples/signal.wav")

    msg_svr = WSPRMsgServer()

    mon = WSPRMonitor(
        sample_rate=sample_rate
    )
    mon.monitor_process(data)

    ts1 = time.monotonic()

    for it in mon.decode():
        msg = msg_svr.decode(it.payload)

        print(
            f"dB: {it.snr:.3f}\t"
            f"T: {it.dT:.3f}\t"
            f"DF: {it.dF:.3f}\t"
            f"BER: {it.BER}\t"
            f"drift: {it.drift}\t"
            f"pass: {it.decode_pass}\t"
            f"Message text: {msg}"
        )

    ts2 = time.monotonic()

    print("-" * 20, "decoded @", ts2 - ts1, "sec")


if __name__ == '__main__':
    # python -m cProfile -s time demo_decode_wspr.py
    main()
