import typing

import numpy as np
from scipy.io.wavfile import write

from consts.ftx import *
from encode import ft8_encode, ft4_encode
from gfsk import synth_gfsk, FT4_SYMBOL_BT, FT8_SYMBOL_BT
from message import message_encode, message_encode_free


def gen_signal(tones: typing.List[int], frequency: int, sample_rate: int, is_ft4: bool):
    symbol_period = FT4_SYMBOL_PERIOD if is_ft4 else FT8_SYMBOL_PERIOD
    symbol_bt = FT4_SYMBOL_BT if is_ft4 else FT8_SYMBOL_BT

    num_tones = FT4_NN if is_ft4 else FT8_NN

    signal = np.fromiter(synth_gfsk(tones, num_tones, frequency, symbol_bt, symbol_period, sample_rate), dtype=float)

    return signal


def gen_ftx_tones(payload: typing.ByteString, is_ft4: bool) -> typing.List[int]:
    if is_ft4:
        tones = ft4_encode(payload)
    else:
        tones = ft8_encode(payload)

    tones = list(tones)

    return tones


def gen_free_text_tones(msg: str, is_ft4: bool) -> typing.List[int]:
    payload = message_encode_free(msg)

    return gen_ftx_tones(payload, is_ft4)


def gen_msg_tones(call_to: str, call_de: str, extra: str, is_ft4: bool) -> typing.List[int]:
    payload = message_encode(call_to, call_de, extra)

    return gen_ftx_tones(payload, is_ft4)


def main():
    sample_rate = 12000
    is_ft4 = False

    print("Gen tones")
    # tones = gen_free_text_tones(f"0123456789AB", is_ft4=is_ft4)
    tones = gen_msg_tones("CQ", "R1ABC", "AA00", is_ft4=is_ft4)

    print("Gen signal")
    signal = gen_signal(tones, 1000, sample_rate=sample_rate, is_ft4=is_ft4)

    symbol_period = FT4_SYMBOL_PERIOD if is_ft4 else FT8_SYMBOL_PERIOD

    num_tones = FT4_NN if is_ft4 else FT8_NN
    slot_time = FT4_SLOT_TIME if is_ft4 else FT8_SLOT_TIME

    num_samples = int(0.5 + num_tones * symbol_period * sample_rate)
    num_silence = int((slot_time * sample_rate - num_samples) / 2)

    print("Write signals")
    silence = np.zeros(num_silence)
    amplitude = np.iinfo(np.int16).max
    data = np.concat([silence, amplitude * signal, silence])
    write("examples/signal.wav", sample_rate, data.astype(np.int16))


if __name__ == '__main__':
    main()
