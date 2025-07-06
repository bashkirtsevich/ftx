import math
import typing

FT8_SYMBOL_BT = 2.0  # < symbol smoothing filter bandwidth factor (BT)
FT4_SYMBOL_BT = 1.0  # < symbol smoothing filter bandwidth factor (BT)

GFSK_K = math.pi * math.sqrt(2 / math.log(2))  # pi * sqrt(2 / log(2)) = 5.336446


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
        arg1 = GFSK_K * symbol_bt * (t + 0.5)
        arg2 = GFSK_K * symbol_bt * (t - 0.5)
        val = (math.erf(arg1) - math.erf(arg2)) / 2

        yield val


def synth_gfsk(symbols: typing.List[int], n_sym: int,
               f0: float,
               symbol_bt: float, symbol_period: float,
               signal_rate: int) -> typing.Generator[float, None, None]:
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
    dphi_peak = 2 * math.pi * hmod / n_spsym
    # Shift frequency up by f0
    dphi = [2 * math.pi * f0 / signal_rate] * (n_wave + 2 * n_spsym)

    pulse = list(gfsk_pulse(n_spsym, symbol_bt))

    for i in range(n_sym):
        ib = i * n_spsym
        for j in range(3 * n_spsym):
            dphi[j + ib] += dphi_peak * symbols[i] * pulse[j]

    # Add dummy symbols at beginning and end with tone values equal to 1st and last symbol, respectively
    for j in range(2 * n_spsym):
        dphi[j] += dphi_peak * pulse[j + n_spsym] * symbols[0]
        dphi[j + n_sym * n_spsym] += dphi_peak * pulse[j] * symbols[n_sym - 1]

    # Calculate the audio waveform
    phi = 0.0
    n_ramp = n_spsym // 8
    for k in range(n_wave):
        val = math.sin(phi)
        phi = math.fmod(phi + dphi[k + n_spsym], 2 * math.pi)

        # Apply envelope shaping to the first and last symbols
        if k < n_ramp or k >= n_wave - n_ramp:
            i_ramp = (k if k < n_ramp else n_wave - k - 1)
            env = (1 - math.cos(2 * math.pi * i_ramp / (2 * n_ramp))) / 2
            val *= env

        yield val
