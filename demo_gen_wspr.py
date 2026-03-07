import typing

import numpy as np
import numpy.typing as npt

from scipy.io.wavfile import write

from consts.wspr import *
from encoders import wspr_encode
from mod import synth_fsk
from msg.message import *


def gen_signal(tones: npt.NDArray[np.uint8], frequency: int, sample_rate: int) -> npt.NDArray[np.float64]:
    samples_per_symbol = int(sample_rate * WSPR_SYMBOL_PERIOD)

    bandwidth = sample_rate / samples_per_symbol

    signal = np.fromiter(
        synth_fsk(tones, sample_rate, samples_per_symbol, frequency, bandwidth),
        dtype=np.float64
    )

    return signal


def gen_wspr_tones(payload: typing.ByteString) -> npt.NDArray[np.uint8]:
    tones = wspr_encode(payload)
    tones = np.fromiter(tones, dtype=np.uint8)

    return tones


def main():
    sample_rate = 12000

    msg = WSPRMessage(WSPRCallsign("R9FEU"), WSPRGrid("LO87"), 50)
    payload = msg.encode()

    tones = gen_wspr_tones(payload)

    signal = gen_signal(tones, frequency=1500, sample_rate=sample_rate)

    amplitude = np.iinfo(np.int16).max
    data = signal * amplitude
    write("examples/signal.wav", sample_rate, data.astype(np.int16))


if __name__ == '__main__':
    main()
