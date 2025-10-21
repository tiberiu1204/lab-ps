import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice as sd

PI = np.pi

DEFAULT = int(1e5)
FS = 44100
RATE = int(1e5)


# def gaussian(m=0, s=1, fs=DEFAULT, t=1, start=-4):
#     num_samples = fs * t
#     x = np.arange(start, -start, -2 * start / num_samples)
#     return x, 1 / np.sqrt(2 * PI * s ** 2) * np.exp(-(x - m)**2 / (2 * s ** 2)), "X-axis", "Value"

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


def plot_samples_subplots(data_list, figsize=(12, 8)):
    n = len(data_list)
    plt.figure(figsize=figsize)

    for i, (x, y, l1, l2) in enumerate(data_list, start=1):
        plt.subplot(n, 1, i)
        plt.plot(x, y)
        plt.xlabel(l1)
        plt.ylabel(l2)
        plt.title(f"Sample Plot {i}")
        plt.grid(True)

    plt.tight_layout()


def _1():
    print("=================Ex1=================")
    plot_samples_subplots([sinusoid(2.5, 6), sinusoid(
        2.5, 6, func=np.cos, phi=-np.pi / 2)])
    plt.show()


def _2():
    print("=================Ex2=================")
    t1, s1, _, _ = sinusoid(1, 5, 0)
    t2, s2, _, _ = sinusoid(1, 5, PI / 2)
    t3, s3, _, _ = sinusoid(1, 5, PI)
    t4, s4, _, _ = sinusoid(1, 5, 1.5 * PI)
    plot_multiple_samples([t1, t2, t3, t4], [s1, s2, s3, s4],
                          ["0", "pi / 2", "pi", "3*pi/2"])
    plt.show()

    SNR = [0.1, 1, 10, 100]
    z = np.random.normal(size=DEFAULT)
    for snr in SNR:
        gamma = np.sqrt(np.linalg.norm(s1) ** 2 /
                        (snr * np.linalg.norm(z) ** 2))
        plot_samples(t1, s1 + gamma * z)
        plt.show()


def _3():
    print("=================Ex3=================")
    # 2 a
    s1 = sinusoid(A=1, f=400)[1]
    sd.play(s1, FS)
    input("Press anything to continue with next sound")
    # 2 b
    s2 = sinusoid(A=1, f=800)[1]
    sd.play(s2, FS)
    input("Press anything to continue with next sound")
    # 2 c
    s3 = saw(f=240)[1]
    sd.play(s3, FS)
    input("Press anything to continue with next sound")
    # 2 d
    s4 = square(f=300)[1]
    sd.play(s4, FS)
    input("Press anything to continue with next sound")

    scipy.io.wavfile.write("test.wav", RATE, s1)
    test_s1 = scipy.io.wavfile.read("test.wav")[1]
    assert (np.array_equal(test_s1, s1))


def _4():
    print("=================Ex4=================")
    s1 = sinusoid(A=1, f=240, t=.05)
    s2 = saw(f=400, t=.05)
    sum = [s1[0], s1[1] + s2[1], "Time", "Value"]
    plot_samples_subplots([s1, s2, sum])

    plt.show()


def _5():
    print("=================Ex5=================")
    s1 = sinusoid(A=1, f=240)[1]
    s2 = sinusoid(A=1, f=400)[1]
    concat = np.concatenate([s1, s2])
    sd.play(concat, FS)
    sd.wait()

    # Prima oara auzim s1, apoi s2


def _6():
    print("=================Ex6=================")
    a = sinusoid(A=1, f=100, fs=200)
    b = sinusoid(A=1, f=50, fs=200)
    c = sinusoid(A=1, f=0, fs=200)
    plot_samples_subplots([a, b, c])
    plt.show()

    # a arata ca zgomot
    # b are esantioane exact in varful "valurilor"
    # c este o linie dreapta (frecventa 0 + faza 0 -> sin(0) = 0)


def _7():
    print("=================Ex7=================")
    s1 = sinusoid(A=1, f=400, fs=1000, t=0.05)
    s2 = [s1[0][::4], s1[1][::4], s1[2], s1[3]]
    s3 = [s1[0][1::4], s1[1][1::4], s1[2], s1[3]]

    plot_samples_subplots([s1, s2, s3])
    plt.show()

    # In s1 prindem mai multe detalii; in s2, fiind mai putine sample-uri, avem mai putine detalii despre sinusoida
    # In s3 esantionam din s1 punctele imediat urmatoare fiecarui punct esantionat in s2


def _8():
    print("=================Ex8=================")
    alpha = np.linspace(-np.pi/2, np.pi/2, 500)
    sin_true = np.sin(alpha)
    sin_small = alpha
    sin_pade = (alpha - 7*alpha**3/60) / (1 + alpha**2/20)

    error_small = np.abs(sin_true - sin_small)
    error_pade = np.abs(sin_true - sin_pade)

    # Plot sin and approximations
    data_curves = [
        (alpha, sin_true, "alpha", "sin(alpha)"),
        (alpha, sin_small, "alpha", "sin(alpha)=alpha (small-angle)"),
        (alpha, sin_pade, "alpha", "Pade approx")
    ]
    plot_samples_subplots(data_curves)

    # Plot errors on log scale
    plt.figure(figsize=(10, 5))
    plt.plot(alpha, error_small, '--', label="Error: Small-angle")
    plt.plot(alpha, error_pade, ':', label="Error: Pade")
    plt.yscale('log')
    plt.xlabel("alpha")
    plt.ylabel("Absolute error")
    plt.title("Approximation Errors")
    plt.legend()
    plt.grid(True, which='both')
    plt.show()


_1()
_2()
_3()
_4()
_5()
_6()
_7()
_8()
