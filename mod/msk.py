import numpy as np
import numpy.typing as npt


def synth_msk(
        tones: npt.NDArray[np.int64], sample_rate: int,
        carrier_freq: int = 1000, delta_freq: int = 1000,
        sampling_factor: int = 1, sampling_rate_coef: int = 4) -> npt.NDArray[np.float64]:
    dt = 1.0 / (sampling_factor * sample_rate)
    samples_per_symbol = 6 * sampling_rate_coef

    phase = 0.0

    signal = np.zeros(tones.shape[0] * samples_per_symbol)
    for i, tone in np.ndenumerate(tones):
        freq = carrier_freq + tone * delta_freq
        phase_delta = 2 * np.pi * freq * dt

        phases = np.fromiter(
            (np.fmod(phase_delta * i + phase, 2 * np.pi)
             for i in range(samples_per_symbol)),
            dtype=np.float64
        )

        t_start = i * samples_per_symbol
        t_end = t_start + samples_per_symbol

        signal[t_start:t_end] = np.sin(phases)

        phase = np.fmod(phase_delta * samples_per_symbol + phase, 2 * np.pi)

    return signal
