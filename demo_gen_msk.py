import typing

import numpy as np
import numpy.typing as npt

from scipy.io.wavfile import write
from scipy.signal import decimate

from encoders import msk144_encode
from msg.message import message_encode, message_encode_free
from mod import synth_msk


def gen_signal(tones: npt.NDArray[np.int64], sample_rate: int) -> npt.NDArray[np.float64]:
    signal = synth_msk(tones, sample_rate=sample_rate)

    return signal


def gen_msk144_tones(payload: typing.ByteString) -> npt.NDArray[np.int64]:
    tones = msk144_encode(payload)
    return tones


def gen_free_text_tones(msg: str) -> npt.NDArray[np.int64]:
    payload = message_encode_free(msg)
    return gen_msk144_tones(payload)


def gen_msg_tones(call_to: str, call_de: str, extra: str = "") -> npt.NDArray[np.int64]:
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
    wave = np.concat([signal * amplitude] * 420)

    write("examples/signal.wav", sample_rate // downsampling, wave.astype(np.int16))


if __name__ == '__main__':
    main()
