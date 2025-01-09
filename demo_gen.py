import numpy as np
from scipy.io.wavfile import write

from consts import FT8_SYMBOL_PERIOD, FT4_SYMBOL_PERIOD, FT4_NN, FT8_NN
from encode import ft8_encode, ft4_encode
from gfsk import synth_gfsk, FT4_SYMBOL_BT, FT8_SYMBOL_BT
from message import ftx_message_encode


def main():
    try:
        # call = "R9FEU/P R1AAA/R R+01"
        # call = "CQ R9FEU LO87"
        payload = ftx_message_encode("R9FEU", "R1AAA", "R+20")
    except Exception as e:
        print(f"Cannot parse message: {type(e)}")
        return

    is_ft4 = False
    # is_ft4 = True

    if is_ft4:
        tones = ft4_encode(payload)
    else:
        tones = ft8_encode(payload)

    tones = list(tones)

    print("FSK tones:", "".join(str(i) for i in tones))

    frequency = 1000

    symbol_period = FT4_SYMBOL_PERIOD if is_ft4 else FT8_SYMBOL_PERIOD
    symbol_bt = FT4_SYMBOL_BT if is_ft4 else FT8_SYMBOL_BT

    sample_rate = 12000
    num_tones = FT4_NN if is_ft4 else FT8_NN

    signal = synth_gfsk(tones, num_tones, frequency, symbol_bt, symbol_period, sample_rate)
    # save_wav(signal, sample_rate, "example.wav")
    # data = np.array(signal)
    # write("example.wav", sample_rate, data)

    amplitude = np.iinfo(np.int16).max
    data = amplitude * np.array(signal)
    write("signal.wav", sample_rate, data.astype(np.int16))


if __name__ == '__main__':
    main()
