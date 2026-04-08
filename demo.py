import numpy as np
import matplotlib.pyplot as plt

# Фиксация сида для воспроизводимости
np.random.seed(95651675)

# Параметры
fs = 4000  # Частота дискретизации
t = np.arange(fs) / fs  # 1 секунда сигнала
f_signal = 1000  # Частота сигнала (Гц)
snr_db = -30  # Очень низкий SNR

# Настройки накопления
accum_steps = [1, 10, 100]  # Количество усредняемых спектров

# Расчет амплитуды шума
# Мощность сигнала P = A^2 / 2. Для A=1, P=0.5.
signal = np.sin(2 * np.pi * f_signal * t)
p_signal = np.mean(signal ** 2)
p_noise = p_signal / (10 ** (snr_db / 10))
noise_sigma = np.sqrt(p_noise)

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

for i, N in enumerate(accum_steps):
    accum_fft = np.zeros(fs // 2)

    for _ in range(N):
        noise = np.random.normal(0, noise_sigma, len(t))
        mixed_signal = signal + noise

        # Вычисляем амплитудный спектр
        fft_vals = np.abs(np.fft.fft(mixed_signal))[:fs // 2]
        accum_fft += fft_vals

    # Усредняем
    avg_fft = accum_fft / N
    freqs = np.fft.fftfreq(fs, 1 / fs)[:fs // 2]

    axes[i].plot(freqs, avg_fft, color='royalblue', lw=1)
    if i == 0:
        axes[i].axvline(f_signal, color='red', linestyle='--', alpha=0.6, label='Частота сигнала')
    axes[i].set_title(f'Усреднение {N} спектров (SNR = {snr_db} дБ)')
    axes[i].set_ylabel('Амплитуда FFT')
    axes[i].grid(True, alpha=0.3)
    if i == 0: axes[i].legend()

axes[-1].set_xlabel('Частота (Гц)')
plt.tight_layout()
plt.show()