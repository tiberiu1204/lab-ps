import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import time
import math
from scipy.io import wavfile
from scipy.fft import rfft

plt_index = 0
PI = np.pi

DEFAULT = 5000


def plot_samples(x, y, l1="Time", l2="Value"):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.xlabel(l1)
    plt.ylabel(l2)
    plt.title("Sample Plot")
    plt.grid(True)


def plot_multiple_samples(xs, ys, labels, l1="Time", l2="Value"):
    plt.figure(figsize=(10, 5))

    for i, y in enumerate(ys):
        plt.plot(xs[i], y, label=labels[i])

    plt.title("Sample Arrays")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)


def plot_samples_subplots(data_list, figsize=(12, 8), titles=None):
    n = len(data_list)
    plt.figure(figsize=figsize)

    for i, (x, y, l1, l2) in enumerate(data_list, start=1):
        plt.subplot(n, 1, i)
        plt.plot(x, y)
        plt.xlabel(l1)
        plt.ylabel(l2)
        if titles:
            plt.title(titles[i-1])
        else:
            plt.title(f"Sample Plot {i}")
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=.5)
        plt.axvline(0, color='black', linewidth=.5)

    plt.tight_layout()


def show_and_save():
    global plt_index
    plt.savefig(f"plot_{plt_index}.pdf")
    plt_index += 1
    plt.show()


def saw(f, fs=DEFAULT, t=1, start=0):
    ts = 1 / fs
    time = np.arange(start, start + t, ts)
    return time, 2 * np.mod(f * time, 1) - 1, "Time", "Value"


def square(f, fs=DEFAULT, t=1, start=0):
    time, s, _, _ = sinusoid(A=1, f=f, fs=fs, t=t, start=start)
    return time, np.sign(s), "Time", "Value"


def sinusoid(A, f, phi=0, fs=DEFAULT, t=1, start=0, func=np.sin):
    ts = 1 / fs
    time = np.arange(start, start + t, ts)
    return time, A * func(2 * np.pi * f * time + phi), "Time", "Value"


def my_fft(x):

    n = len(x)
    if n <= 1:
        return x

    even = my_fft(x[0::2])
    odd = my_fft(x[1::2])

    coef = np.exp(-2j * PI * np.arange(n // 2) / n)

    return np.concatenate([even + coef * odd, even - coef * odd])


def make_fourier_matrix(N):
    m = []
    for i in range(N):
        m.append([])
        for n in range(N):
            m[i].append(math.e ** (-2 * PI * 1j * n * i / N))
    return 1 / math.sqrt(N) * np.array(m)


def my_dft(x):
    return make_fourier_matrix(len(x)) @ x


def _1():
    Ns = [128, 256, 512, 1024, 2048, 4096, 8192]

    times_dft = []
    times_fft = []
    times_npfft = []

    for N in Ns:
        print(f"N = {N}")
        x = np.random.random(N)

        start = time.time()
        my_dft(x)
        times_dft.append(time.time() - start)

        start = time.time()
        my_fft(x)
        times_fft.append(time.time() - start)

        start = time.time()
        np.fft.fft(x)
        times_npfft.append(time.time() - start)

    plt.figure(figsize=(8, 5))
    plt.plot(Ns, times_dft, 'o-', label='DFT O(N^2)')
    plt.plot(Ns, times_fft, 'o-', label='FFT O(N log N)')
    plt.plot(Ns, times_npfft, 'o-', label='NumPy FFT (optimizat)')
    plt.yscale('log')
    plt.xlabel("Dimensiunea vectorului N")
    plt.ylabel("Timp de execuție (secunde, scara log)")
    plt.title("Compararea timpilor de executie: DFT vs FFT vs np.fft.fft")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    show_and_save()


def _2():
    fs_sample = 600
    t = 0.01

    time_samp = np.arange(0, t, 1/fs_sample)
    f0 = 1900

    f_alias1 = 700  # f0 - 2 * fs_sample
    f_alias2 = 100  # f0 - 3 * fs_sample

    samples_orig = np.sin(2 * np.pi * f0 * time_samp)
    samples_alias1 = np.sin(2 * np.pi * f_alias1 * time_samp)
    samples_alias2 = np.sin(2 * np.pi * f_alias2 * time_samp)

    time_cont, x_cont, _, _ = sinusoid(1, f0, phi=0, fs=50000, t=t)
    _, x1_cont, _, _ = sinusoid(1, f_alias1, phi=0, fs=50000, t=t)
    _, x2_cont, _, _ = sinusoid(1, f_alias2, phi=0, fs=50000, t=t)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(time_cont, x_cont, label=f"Original {f0} Hz")
    axs[0].scatter(time_samp, samples_orig, color='red', zorder=3)
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    axs[1].plot(time_cont, x1_cont, label=f"Alias {
                f_alias1} Hz", color='orange')
    axs[1].scatter(time_samp, samples_alias1, color='red', zorder=3)
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()

    axs[2].plot(time_cont, x2_cont, label=f"Alias {
                f_alias2} Hz", color='green')
    axs[2].scatter(time_samp, samples_alias2, color='red', zorder=3)
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Amplitude")
    axs[2].legend()

    plt.suptitle(
        "Aliasing Demonstration: Same Sample Points for Different Frequencies")
    plt.tight_layout()
    show_and_save()


def _3():
    fs_sample = 3000
    t = 0.01

    time_samp = np.arange(0, t, 1/fs_sample)
    f0 = 1900

    f_alias1 = 700
    f_alias2 = 100

    samples_orig = np.sin(2 * np.pi * f0 * time_samp)
    samples_alias1 = np.sin(2 * np.pi * f_alias1 * time_samp)
    samples_alias2 = np.sin(2 * np.pi * f_alias2 * time_samp)

    time_cont, x_cont, _, _ = sinusoid(1, f0, phi=0, fs=50000, t=t)
    _, x1_cont, _, _ = sinusoid(1, f_alias1, phi=0, fs=50000, t=t)
    _, x2_cont, _, _ = sinusoid(1, f_alias2, phi=0, fs=50000, t=t)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(time_cont, x_cont, label=f"Original {f0} Hz")
    axs[0].scatter(time_samp, samples_orig, color='red', zorder=3)
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    axs[1].plot(time_cont, x1_cont, label=f"Alias {
                f_alias1} Hz", color='orange')
    axs[1].scatter(time_samp, samples_alias1, color='red', zorder=3)
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()

    axs[2].plot(time_cont, x2_cont, label=f"Alias {
                f_alias2} Hz", color='green')
    axs[2].scatter(time_samp, samples_alias2, color='red', zorder=3)
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Amplitude")
    axs[2].legend()

    plt.suptitle(
        "Aliasing Demonstration: Same Sample Points for Different Frequencies")
    plt.tight_layout()
    show_and_save()


def _4():
    print("2 * 200 = 400 hz")


def _6():
    fs, data = wavfile.read("vocale.wav")

    data = data.astype(float)
    if data.ndim > 1:
        data = data[:, 0]  # canalul stang

    sz_group = len(data) // 100
    step = sz_group // 2

    fig = []

    for i in range(0, len(data)-sz_group+1, step):
        segment = data[i:i+sz_group]
        fft_vals = np.abs(rfft(segment))
        fig.append(fft_vals)

    fig = np.column_stack(fig)

    freqs = np.fft.rfftfreq(sz_group, 1/fs)
    times = np.arange(fig.shape[1]) * step / fs

    fig_db = 10 * np.log10(fig)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, freqs, fig_db, shading='gouraud', cmap='inferno')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram')
    plt.colorbar(label='Amplitude [dB]')
    show_and_save()


def _7():
    # SNRdb = 10 log10 SNR (Psemnal / Pzgomot)
    # Psemnaldb = 90dB = 10 log10 Psemnal / P0
    # log10 Psemnal / P0 = 9 => Psemnal / P0 = 10 ^ 9 => Psemnal = P0 * 10 ^ 9
    # 10 log10 (P0 * 10^9 / Pzgomot) = 80
    # P0 * 10^9 / Pzgomot = 10^8
    # Pzgomot = P0 * 10
    # 10 * log10 (Pzgomot / P0) = 10dB
    # SAU
    # 10 log10 (Psemnal / Pzgomot) = 10log10 Psemnal - 10log10 Pzgomot = 90 => 10log10 Pzgomot = 10
    pass


_1()
_2()
_3()
_4()
_6()
