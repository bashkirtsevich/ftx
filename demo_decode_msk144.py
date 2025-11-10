import os
import time

from scipy.io.wavfile import read

from decoders import MSK144Monitor
from msg.message import MsgServer


def main():
    sample_rate, data = read("examples/signal.wav")

    msg_svr = MsgServer()

    db_path = "examples/cs_db.pkl"
    if os.path.isfile(db_path):
        msg_svr.load(db_path)

    mon = MSK144Monitor(sample_rate=sample_rate)
    mon.monitor_process(data)

    ts1 = time.monotonic()

    for it in mon.decode(tm_slot_start=0):
        msg = msg_svr.decode(it.payload)

        print(
            f"dB: {it.snr:.3f}\t"
            f"T: {it.dT:.3f}\t"
            f"DF: {it.dF:.3f}\t"
            f"Averaging pattern: {it.p_avg}\t"
            f"Freq: {it.freq}\t"
            f"Eye opening: {it.eye_open:.3f}\t"
            f"Bit errors: {it.bit_err}\t"
            f"CRC: {it.crc}\t"
            f"Message text: {msg}"
        )

    ts2 = time.monotonic()

    print("-" * 20, "decoded @", ts2 - ts1, "sec")

    msg_svr.save(db_path)


if __name__ == '__main__':
    # python -m cProfile -s time demo_decode_msk144.py
    main()
