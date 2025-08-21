import numpy as np
from scipy.io.wavfile import write

from consts import *
from encode import ft8_encode, ft4_encode
from gfsk import synth_gfsk, FT4_SYMBOL_BT, FT8_SYMBOL_BT
from message import ftx_message_encode, ftx_message_decode, ftx_message_encode_free, ftx_message_decode_free


def main():
    try:
        # call = "R9FEU/P R1AAA/R R+01"
        # call = "R9FEU R1AAA R+01"
        call = "CQ R9FEU LO88"
        # call = "R2CBA R1ABC RR73"
        payload = ftx_message_encode(*call.split())
        # payload = ftx_message_encode_free("0123456789AB")
    except Exception as e:
        print(f"Cannot parse message: {type(e)}")
        return

    print("Payload", ''.join('{:08b}'.format(x) for x in payload))

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
    slot_time = FT4_SLOT_TIME if is_ft4 else FT8_SLOT_TIME

    num_samples = int(0.5 + num_tones * symbol_period * sample_rate)
    num_silence = int((slot_time * sample_rate - num_samples) / 2)

    signal = np.fromiter(synth_gfsk(tones, num_tones, frequency, symbol_bt, symbol_period, sample_rate), dtype=float)
    # save_wav(signal, sample_rate, "example.wav")
    # data = np.array(signal)
    # write("example.wav", sample_rate, data)

    silence = np.zeros(num_silence)
    amplitude = np.iinfo(np.int16).max
    data = np.concat([silence, amplitude * signal, silence])
    write("examples/signal.wav", sample_rate, data.astype(np.int16))


if __name__ == '__main__':
    main()
