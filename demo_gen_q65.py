import typing

import numpy as np
import numpy.typing as npt

from scipy.io.wavfile import write
from scipy.signal import decimate

from encoders import q65_encode
from message import message_encode, message_encode_free
from mod import synth_fsk


def gen_signal(tones: npt.NDArray[np.int64], sample_rate: int,
               period: typing.Optional[typing.Literal[30, 60, 120, 300]],
               f0: float, q65_type: typing.Literal[1, 2, 3, 4]):
    if period == 30:
        nsps = 3600
    elif period == 60:
        nsps = 7200
    elif period == 120:
        nsps = 16000
    elif period == 300:
        nsps = 41472
    else:
        nsps = 1800

    samples_per_symbol = 4 * nsps  # 48000 Hz sampling

    baud = sample_rate / samples_per_symbol
    bandwidth = baud * q65_type

    signal = synth_fsk(tones, sample_rate=sample_rate, samples_per_symbol=samples_per_symbol, f0=f0,
                       bandwidth=bandwidth)

    return signal


def gen_q65_tones(payload: typing.ByteString) -> npt.NDArray[np.int64]:
    tones = q65_encode(payload)
    return tones


def gen_free_text_tones(msg: str) -> npt.NDArray[np.int64]:
    payload = message_encode_free(msg)
    return gen_q65_tones(payload)


def gen_msg_tones(call_to: str, call_de: str, extra: str = "") -> npt.NDArray[np.int64]:
    payload = message_encode(call_to, call_de, extra)
    return gen_q65_tones(payload)


def main():
    tones = gen_msg_tones("CQ", "R9FEU", "LO87")

    sample_rate = 48000
    signal = gen_signal(tones, sample_rate, period=30, f0=1000, q65_type=1)

    # Adjust volume
    volume = 0.5
    signal *= volume

    # Downsample
    downsampling = 4
    signal = decimate(signal, downsampling)

    # Gen wav file
    amplitude = np.iinfo(np.int16).max
    wave = signal * amplitude

    write("examples/signal.wav", sample_rate // downsampling, wave.astype(np.int16))


if __name__ == '__main__':
    main()
