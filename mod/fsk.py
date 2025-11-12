import numpy as np
import numpy.typing as npt


def synth_fsk(
        tones: npt.NDArray[np.uint8], sample_rate: int,
        samples_per_symbol: int,
        f0: float, bandwidth: float) -> npt.NDArray[np.float64]:
    dt = 1.0 / sample_rate

    phase = 0.0

    signal = np.zeros(tones.shape[0] * samples_per_symbol, dtype=np.float64)
    for i, tone in np.ndenumerate(tones):
        idx, *_ = i

        freq = f0 + bandwidth * tone
        phase_delta = 2 * np.pi * freq * dt

        phases = np.fromiter(
            (np.fmod(phase_delta * i + phase, 2 * np.pi)
             for i in range(samples_per_symbol)),
            dtype=np.float64
        )

        t_start = idx * samples_per_symbol
        t_end = t_start + samples_per_symbol

        signal[t_start:t_end] = np.sin(phases)

        phase = np.fmod(phase_delta * samples_per_symbol + phase, 2 * np.pi)

    return signal
