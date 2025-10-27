import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation

PI = np.pi

DEFAULT = 5000
FS = 44100
RATE = int(1e5)
plt_index = 0


def animate_wrap_multi(s1, freqs=(2, 3, 5, 7), interval=10, num_samples=DEFAULT):
    t, x = s1[0], s1[1]
    N = len(x)
    n = np.arange(N)

    ncols = len(freqs) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    ax_time = axes[0]
    ax_wraps = axes[1:]

    ax_time.plot(t, x, color='b')
    ax_time.set_xlabel("Timp")
    ax_time.set_ylabel("Amplitudine")
    ax_time.set_title("Semnal sinusoidal")
    ax_time.grid(True)
    ax_time.axhline(0, color='black', lw=1)

    point_time, = ax_time.plot([], [], 'ro', markersize=8)

    circles = []
    lines = []
    points = []
    limits = 1.2 * np.max(np.abs(x))

    for i, f in enumerate(freqs):
        ax = ax_wraps[i]
        u = unit_circle(scale_vector=x, f=f, num_samples=num_samples)
        c = u[0] + 1j * u[1]
        circles.append(c)
        ax.set_aspect('equal')
        ax.set_xlim(-limits, limits)
        ax.set_ylim(-limits, limits)
        ax.axhline(0, color='black', lw=1)
        ax.axvline(0, color='black', lw=1)
        ax.grid(True)
        ax.set_title(f"ω = {f}")
        line, = ax.plot([], [], 'b-', lw=2)
        point, = ax.plot([], [], 'ro', markersize=6)
        lines.append(line)
        points.append(point)

    def update(i):
        point_time.set_data([t[i]], [x[i]])

        for k, c in enumerate(circles):
            lines[k].set_data(c.real[:i], c.imag[:i])
            points[k].set_data(
                [c.real[i-1]], [c.imag[i-1]] if i > 0 else [0])
        return [point_time, *lines, *points]

    ani = animation.FuncAnimation(
        fig, update, frames=N, interval=interval, blit=True
    )

    plt.suptitle("Infasurarea sinusului pentru mai multe frecvente ω")
    plt.show()


def make_fourier_matrix(N):
    m = []
    for i in range(N):
        m.append([])
        for n in range(N):
            m[i].append(math.e ** (-2 * PI * 1j * n * i / N))
    return 1 / math.sqrt(N) * np.array(m)


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


def unit_circle(num_samples=DEFAULT, scale_vector=None, f=1):
    c = np.exp(-np.arange(num_samples) / num_samples * 2 * PI * 1j * f)
    if scale_vector is not None:
        c *= scale_vector
    return c.real, c.imag, "Real", "Imaginar"


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


def _1():
    N = 8
    m = make_fourier_matrix(N)

    print(np.allclose(m.conj().T @ m, np.eye(N)))

    fig, axes = plt.subplots(N, 2, figsize=(10, 12))
    fig.suptitle("Componente reale si imaginare")

    x = np.arange(N)
    for k in range(N):
        y = m[k, :].flatten()
        axes[k, 0].stem(x, y.real)
        axes[k, 0].set_title(f"Randul {k} - Partea reala")
        axes[k, 0].set_ylim(-0.5, 0.5)

        axes[k, 1].stem(x, y.imag)
        axes[k, 1].set_title(f"Randul {k} - Partea imaginara")
        axes[k, 1].set_ylim(-0.5, 0.5)

    plt.tight_layout()
    show_and_save()


def _2():

    s1 = sinusoid(A=1, f=7)
    u = unit_circle(num_samples=len(s1[1]), scale_vector=s1[1], f=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(s1[0], s1[1])
    axes[0].scatter([s1[0][300]], [s1[1][300]], color='red', s=100, zorder=5)
    axes[0].set_xlabel(s1[2])
    axes[0].set_ylabel(s1[3])
    axes[0].axhline(0, color='black')
    axes[0].grid()

    axes[1].plot(u[0], u[1], color='blue')
    axes[1].scatter([u[0][300]], [u[1][300]], color='red', s=100, zorder=5)
    axes[1].set_xlabel(u[2])
    axes[1].set_ylabel(u[3])
    axes[1].axhline(0, color='black')
    axes[1].axvline(0, color='black')
    axes[1].grid()
    axes[1].set_aspect('equal')

    plt.suptitle("Figura 1 – Sinusoid si desfasurarea pe cercul unitate")
    show_and_save()

    frecvente = [2, 3, 5, 7]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i, f in enumerate(frecvente):
        z = unit_circle(num_samples=len(s1[1]), scale_vector=s1[1], f=f)
        dist = np.sqrt(z[0]**2 + z[1]**2)

        color = plt.cm.plasma(dist / dist.max())

        axes[i].scatter(z[0], z[1], c=color, s=1)
        axes[i].set_aspect('equal')
        axes[i].set_title(f"ω = {f}")
        axes[i].axhline(0, color='black')
        axes[i].axvline(0, color='black')
        axes[i].grid()

    plt.suptitle("Figura 2 – Influenta frecventei de infasurare")
    show_and_save()

    animate_wrap_multi(s1)


def _3():
    fs = 2400
    s1 = sinusoid(A=3, f=32, phi=PI/4, fs=fs)
    s2 = sinusoid(A=.5, f=89, fs=fs)
    s3 = sinusoid(A=4, f=29, phi=PI/6, fs=fs)
    s4 = sinusoid(A=1, f=12, fs=fs)
    s5 = [s1[0], s1[1] + s2[1] + s3[1] + s4[1], "Time", "Value"]
    F = make_fourier_matrix(200)
    X = F @ s5[1][::12]

    freq = np.arange(100)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(s5[0], s5[1])
    axes[0].set_title("Semnal compus in timp")
    axes[0].set_xlabel(s5[2])
    axes[0].set_ylabel(s5[3])
    axes[0].grid(True)

    # Spectrul Fourier (magnitudinea)
    axes[1].stem(freq, np.abs(X)[:100], basefmt=" ")
    axes[1].set_title("Spectrul Fourier |X[k]|")
    axes[1].set_xlabel("Index frecventa (k)")
    axes[1].set_ylabel("|X[k]|")
    axes[1].grid(True)

    plt.tight_layout()
    show_and_save()
    animate_wrap_multi(s5, freqs=(32, 98, 29, 12), num_samples=fs, interval=30)


_1()
_2()
_3()
