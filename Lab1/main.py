import numpy as np
import matplotlib.pyplot as plt


def x(t): return np.cos(520 * np.pi * t + np.pi / 3)
def y(t): return np.cos(280 * np.pi * t - np.pi / 3)
def z(t): return np.cos(120 * np.pi * t + np.pi / 3)


def s(t, f, phi=0): return np.sin(t * f * np.pi + phi)


def saw(t, f, phi=0):
    return 2 * np.mod(f * t, 1) - 1


def sqr(t, f, phi=0):
    return np.sign(s(t, f, phi))


# Funcție de plotare
def plot_signal(f, fs, phi=0, start=0, end=1, max_s=50, sig_type="sin"):
    t = np.arange(start, end, 1/fs)

    if sig_type == "sin":
        y = s(t, f, phi)
    elif sig_type == "sawtooth":
        y = saw(t, f)
    elif sig_type == "square":
        y = sqr(t, f)

    plt.suptitle(f"{sig_type.capitalize()} wave")
    plt.plot(t[:max_s], y[:max_s],)
    plt.grid(True)
    plt.show()


def plot_sinusoids():
    ax_time = np.arange(0, 0.03 + 0.0005, 0.0005)
    fig, axs = plt.subplots(3)
    fig.suptitle("Sinusoidale")
    axs[0].plot(ax_time, x(ax_time))
    axs[1].plot(ax_time, y(ax_time))
    axs[2].plot(ax_time, z(ax_time))

    fig.show()


def plot_sampling():
    ax_time = np.arange(0, 0.03, 1 / 200)
    fig, axs = plt.subplots(3)
    fig.suptitle("Sinusoidale esantionate")
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))

    axs[0].stem(ax_time, x(ax_time))
    axs[1].stem(ax_time, y(ax_time))
    axs[2].stem(ax_time, z(ax_time))

    fig.show()


def _2e():
    x, y = 128, 128
    I = np.random.rand(x, y)
    plt.imshow(I)
    plt.show()


def _2f():
    x, y = 128, 128
    I = np.arange(x * y).reshape(x, y)
    I = np.tan(I)
    plt.imshow(I)
    plt.show()


# plot_sinusoids()
# plot_sampling()


# 2 a
plot_signal(400, 1600)
# 2 b
plot_signal(f=800, fs=2000, end=3)
# 2 c
plot_signal(f=240, fs=100000, sig_type="sawtooth", max_s=5000)
# 2 d
plot_signal(f=300, fs=100000, sig_type="square", max_s=5000)
# 2 e
_2e()
# 2 f
_2f()


# 1 / 2000
# (3600 * 4 * 2000) / 2

input()
