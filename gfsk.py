import math
import typing

FT8_SYMBOL_BT = 2.0  # < symbol smoothing filter bandwidth factor (BT)
FT4_SYMBOL_BT = 1.0  # < symbol smoothing filter bandwidth factor (BT)

M_PI = 3.14159265358979323846
GFSK_CONST_K = 5.336446  # < == pi * sqrt(2 / log(2))


def gfsk_pulse(n_spsym: int, symbol_bt: float) -> typing.Generator[float, None, None]:
    """
    Computes a GFSK smoothing pulse.
    The pulse is theoretically infinitely long, however, here it's truncated at 3 times the symbol length.
    This means the pulse array has to have space for 3*n_spsym elements.

    :param n_spsym: Number of samples per symbol
    :param symbol_bt: Shape parameter (values defined for FT8/FT4)
    :return: Generator wirh float values
    """
    for i in range(3 * n_spsym):
        t = i / n_spsym - 1.5
        arg1 = GFSK_CONST_K * symbol_bt * (t + 0.5)
        arg2 = GFSK_CONST_K * symbol_bt * (t - 0.5)
        val = (math.erf(arg1) - math.erf(arg2)) / 2

        yield val


def synth_gfsk(symbols: typing.List[int], n_sym: int,
               f0: float,
               symbol_bt: float, symbol_period: float,
               signal_rate: int) -> typing.List[float]:
    """
    Synthesize waveform data using GFSK phase shaping.
    The output waveform will contain n_sym symbols.
    :param symbols: Array of symbols (tones) (0-7 for FT8)
    :param n_sym: Number of symbols in the symbol array
    :param f0: Audio frequency in Hertz for the symbol 0 (base frequency)
    :param symbol_bt: Symbol smoothing filter bandwidth (2 for FT8, 1 for FT4)
    :param symbol_period: Symbol period (duration), seconds
    :param signal_rate: Sample rate of synthesized signal, Hertz
    :return: List with signal wave values [-1..1]
    """
    n_spsym = int(0.5 + signal_rate * symbol_period)  # Samples per symbol
    n_wave = n_sym * n_spsym  # Number of output samples
    hmod = 1.0

    # Compute the smoothed frequency waveform.
    # Length = (nsym+2)*n_spsym samples, first and last symbols extended
    dphi_peak = 2 * M_PI * hmod / n_spsym
    # Shift frequency up by f0
    dphi = [2 * M_PI * f0 / signal_rate] * (n_wave + 2 * n_spsym)

    pulse = list(gfsk_pulse(n_spsym, symbol_bt))

    for i in range(n_sym):
        ib = i * n_spsym
        for j in range(3 * n_spsym):
            dphi[j + ib] += dphi_peak * symbols[i] * pulse[j]

    # Add dummy symbols at beginning and end with tone values equal to 1st and last symbol, respectively
    for j in range(2 * n_spsym):
        dphi[j] += dphi_peak * pulse[j + n_spsym] * symbols[0]
        dphi[j + n_sym * n_spsym] += dphi_peak * pulse[j] * symbols[n_sym - 1]

    # Calculate and insert the audio waveform
    signal = []
    phi = 0.0
    for k in range(n_wave):
        #  Don't include dummy symbols
        signal.append(math.sin(phi))
        phi = math.fmod(phi + dphi[k + n_spsym], 2 * M_PI)

    # Apply envelope shaping to the first and last symbols
    n_ramp = n_spsym // 8
    for i in range(n_ramp):
        env = (1 - math.cos(2 * M_PI * i / (2 * n_ramp))) / 2
        signal[i] *= env
        signal[n_wave - 1 - i] *= env

    return signal
