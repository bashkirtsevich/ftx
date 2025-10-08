import typing

import numpy as np
from scipy.io.wavfile import write
from scipy.signal import decimate

from encode import msk144_encode
from message import message_encode, message_encode_free
from msk import msk_gen_signal


def gen_signal(tones: typing.List[int], sample_rate: int):
    signal = msk_gen_signal(tones, sample_rate=sample_rate)

    return signal


def gen_msk144_tones(payload: typing.ByteString) -> typing.List[int]:
    tones = msk144_encode(payload)

    tones = list(tones)

    return tones


def gen_free_text_tones(msg: str) -> typing.List[int]:
    payload = message_encode_free(msg)

    return gen_msk144_tones(payload)


def gen_msg_tones(call_to: str, call_de: str, extra: str = "") -> typing.List[int]:
    payload = message_encode(call_to, call_de, extra)

    return gen_msk144_tones(payload)


def main():
    tones = gen_msg_tones("CQ", "R9FEU", "LO87")

    # Gen base signal
    sample_rate = 48000
    signal = gen_signal(tones, sample_rate=sample_rate)

    # Adjust volume
    volume = 0.5
    signal *= volume

    # Downsample
    downsampling = 4
    signal = decimate(signal, downsampling)

    # Gen wav file
    amplitude = np.iinfo(np.int16).max
    wave = np.concat([signal * amplitude] * 10)

    write("examples/signal.wav", sample_rate // downsampling, wave.astype(np.int16))


if __name__ == '__main__':
    main()
