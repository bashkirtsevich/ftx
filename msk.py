import numpy as np
import typing


def mskx_gen_signal(
        tones: typing.List[int], sample_rate: int,
        carrier_freq: int = 1000, delta_freq: int = 1000,
        sampling_factor: int = 1, sampling_rate_coef: int = 4):
    dt = 1.0 / (sampling_factor * sample_rate)
    samples_per_symbol = 6 * sampling_rate_coef

    phase = 0.0

    signal = np.zeros(len(tones) * samples_per_symbol)
    for i, tone in enumerate(tones):
        freq = carrier_freq + tone * delta_freq
        phase_delta = 2 * np.pi * freq * dt

        t_start = i * samples_per_symbol
        t_end = t_start + samples_per_symbol

        phases = np.fromiter(
            (np.fmod(phase_delta * i + phase, 2 * np.pi)
             for i in range(samples_per_symbol)),
            dtype=np.float32
        )

        signal[t_start:t_end] = np.sin(phases)

        phase = np.fmod(phase_delta * samples_per_symbol + phase, 2 * np.pi)

    return signal
