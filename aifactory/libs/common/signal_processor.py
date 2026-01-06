import numpy as np
import matplotlib.pyplot as plt


def plot_fft_results(fft):
    # 可视化
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    # 原始信号
    N = len(fft['src'])
    axes[0].plot(np.arange(N), fft['src'])
    axes[0].set_title('Signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Magnitude')
    axes[0].grid(True)

    # 幅度谱（完整）
    axes[1].plot(fft['frequency'], fft['magnitude'])
    axes[1].set_title('Frequency Spectrum')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].grid(True)

    # 幅度谱（正频率部分）
    positive_freq = fft['frequency'][:N // 2]
    positive_magnitude = fft['magnitude'][:N // 2]
    axes[2].plot(positive_freq, positive_magnitude)
    axes[2].set_title('Positive Frequency Spectrum')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude')
    axes[2].grid(True)

    # 相位谱
    axes[3].plot(positive_freq, fft['phase'][:N // 2])
    axes[3].set_title('Phase Spectrum')
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('Phase (arc)')
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()


def get_frequency_info(x, plot=False):
    x = x.reshape(-1)
    fft = np.fft.fft(x)
    magnitude = np.abs(fft)
    phase = np.angle(fft)
    N = len(x)
    frequency = np.fft.fftfreq(N, 1/N)
    if plot:
        plot_info = {"src": x,
                     "fft": fft,
                     "magnitude": magnitude,
                     "phase": phase,
                     "frequency": frequency}
        plot_fft_results(plot_info)

    return {"fft": fft,
            "magnitude": magnitude,
            "phase": phase,
            "frequency": frequency}


def gradient2d(x):
    grad_y, grad_x = np.gradient(x)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    arc = np.arctan2(grad_y, grad_x)
    return {"grad_x": grad_x,
            "grad_y": grad_y,
            "magnitude": magnitude,
            "arc": arc}
