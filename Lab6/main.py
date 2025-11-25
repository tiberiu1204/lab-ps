import numpy as np
import matplotlib.pyplot as plt
import random
import scipy

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


def _1():
    B = 2
    start = -3
    end = 3
    x = np.sinc(B * np.linspace(start=start, stop=end,
                endpoint=True, num=DEFAULT)) ** 2
    eps = .00001
    Fs = [1, 1.5, 2, 4]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    t_plot = np.linspace(start, end, DEFAULT)

    for i, fs in enumerate(Fs):
        ts = 1 / fs
        s1 = np.sinc(B * np.arange(start=0, stop=-3 - eps, step=-ts)) ** 2
        s2 = np.sinc(B * np.arange(start=0, stop=3 + eps, step=ts)) ** 2
        s = np.concatenate([s1[::-1], s2[1:]])
        x_r = [0] * DEFAULT

        t_reconstruct = np.linspace(start, end, DEFAULT)
        x_r = np.zeros(DEFAULT)

        t_s1 = np.arange(start=0, stop=-3 - eps, step=-ts)
        t_s2 = np.arange(start=0, stop=3 + eps, step=ts)
        t_samples = np.concatenate([t_s1[::-1], t_s2[1:]])

        for t in range(len(t_reconstruct)):
            t_val = t_reconstruct[t]
            x_r[t] = np.sum(s * np.sinc((t_val - t_samples) / ts))

        ax = axs[i]
        ax.axhline(0, color='black')

        ax.plot(t_plot, x, color='black', label='Original')

        ax.plot(t_plot, x_r, color='green',
                linestyle='--', label='Reconstructed')

        t_s1 = np.arange(start=0, stop=-3 - eps, step=-ts)
        t_s2 = np.arange(start=0, stop=3 + eps, step=ts)
        t_s = np.concatenate([t_s1[::-1], t_s2[1:]])

        markerline, stemlines, baseline = ax.stem(t_s, s, basefmt=" ")
        plt.setp(stemlines, 'color', 'orange')
        plt.setp(markerline, 'markerfacecolor',
                 'orange', 'markeredgecolor', 'orange')

        ax.set_title(f'Frecventa = {fs}, B = {B}')
        ax.set_xlabel('t[s]')
        ax.set_ylabel('Amplitudine')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    show_and_save()


def _2():
    N = 100

    def conv3(x):
        ax_x = np.arange(N)
        xs = []
        xs.append(x)
        dl = [[ax_x, x, "Sample", "Value"]]
        for i in range(3):
            xs.append(np.convolve(xs[-1], xs[-1]))
            dl.append([np.arange(len(xs[-1])), xs[-1], "Sample", "Value"])
        plot_samples_subplots(data_list=dl)
        show_and_save()
    conv3(np.random.random(N))
    ones = 10
    init_zeros = 10
    conv3(np.concatenate([np.zeros(init_zeros),
          np.ones(ones), np.zeros(N-ones-init_zeros)]))


def _3():
    N = 100
    p = np.random.randint(low=-100, high=100, size=N)
    q = np.random.randint(low=-100, high=100, size=N)
    print(np.convolve(p, q))
    p_fft = np.fft.fft(p, N * 2 - 1)
    q_fft = np.fft.fft(q, N * 2 - 1)
    res = np.rint(np.real(np.fft.ifft(p_fft * q_fft))).astype(np.int64)
    print(res)
    print(np.all(res == np.convolve(p, q)))


def _4():
    N = 20

    x = np.sinc(np.arange(N)/N) ** 3
    d = random.randint(0, N)
    y = np.roll(x, d)

    X = np.fft.fft(x)
    Y = np.fft.fft(y)

    f1 = np.fft.ifft(np.matrix(X).H * Y)
    d1 = int(np.argmax(np.abs(f1)))

    eps = 1e-12
    Xs = np.where(np.abs(X) < eps, eps, X)
    f2 = np.fft.ifft(Y / Xs)
    d2 = int(np.argmax(np.abs(f2)))

    print(d, d1, d2)


def _5():
    Nw = 200
    f = 100
    A = 1
    phi = 0

    t, x, l1, l2 = sinusoid(A, f, phi=phi, fs=Nw, t=1)

    rect_window = np.ones(Nw)
    hanning_window = np.hanning(Nw)

    x_rect = x * rect_window
    x_hann = x * hanning_window

    data_list = [
        (t, x, l1, l2),
        (t, x_rect, l1, l2),
        (t, x_hann, l1, l2)
    ]
    titles = ["Original signal", "Rectangular window", "Hanning window"]
    plot_samples_subplots(data_list, titles=titles)
    show_and_save()


def _6():
    x = np.ndarray.flatten(np.genfromtxt('train.csv', delimiter=',')[
                           10000:10000 + 24 * 3, 2:3])
    time_axis = np.arange(72)
    Ws = [5, 9, 13, 17]
    labels = ["Original signal"]
    Xs = [time_axis]
    Ys = [x]
    for w in Ws:
        x_f = np.convolve(x, np.ones(w), mode='valid') / w
        Xs.append(time_axis[:len(x_f)])
        Ys.append(x_f)
        labels.append(f"w = {w}")
    plot_multiple_samples(Xs, Ys, labels)
    show_and_save()

    # Elimin frecventele mai mici de 'o zi'
    Wn = 1 / (24 * 3600)
    Wn_norm = Wn / (1 / 3600 / 2)
    print(f"Wn Norm: {Wn_norm}")

    def plot_filters(N=5, rp=5):
        b_butter, a_butter = scipy.signal.butter(N=N, Wn=Wn_norm)
        b_cheby1, a_cheby1 = scipy.signal.cheby1(N=N, rp=rp, Wn=Wn_norm)

        x_butter = scipy.signal.filtfilt(b_butter, a_butter, x)
        x_cheby1 = scipy.signal.filtfilt(b_cheby1, a_cheby1, x)

        plot_multiple_samples([time_axis[:len(x)], time_axis[:len(x_butter)], time_axis[:len(x_cheby1)]],
                              [x, x_butter, x_cheby1], labels=["Original", f"Butter filter (N={N})", f"Cheby1 filter (N={N}, rp={rp})"])
        show_and_save()
        # Aleg filtrul butter sau cheby1 cu rb <= 1, deoarece urmaresc forma generala a semnalului mai bine
        # decat cheb1 cu rb = 5
    N = [2, 5, 10]
    RP = [1, 3, 5]
    for n in N:
        for rp in RP:
            plot_filters(n, rp)


_1()
_2()
_3()
_4()
_5()
_6()
