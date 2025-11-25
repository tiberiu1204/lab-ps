import numpy as np
import matplotlib.pyplot as plt

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


x = np.genfromtxt('train.csv', delimiter=',')
print(x[7 * 24 * 2 + 48 + 1])
x = np.ndarray.flatten(x[1:, 2:3])
N = len(x)
time_axis = np.arange(N) / 24

X = np.fft.fft(x)
X_orig = np.copy(X)
X = np.abs(X / N)
X = X[:N//2]
NX = N // 2

fs = 1 / 3600
freq_axis = fs * np.linspace(0, N/2, N//2)/N
plot_samples(freq_axis, X, "Frequency (Hz)",
             "|X| (Amplitude) (Number of Cars)")
show_and_save()

print(f"Frecventa de esantionare este 1 / 3600 = {fs} Hz")
print(f"Intervalul de timp acopera {
      N / 24} zile, {N / 24 / 12} luni, aproximativ {N / 24 / 365} ani")
print(f"Frecventa maxima prezenta in semnal este: {fs / 2} Hz")

X_complete = np.copy(X_orig)
X_complete[0] = 0
x_elim = np.fft.ifft(X_complete)

plot_samples_subplots([[time_axis, x, "Time (days since beginning)", "Number of Cars"],
                       [time_axis, x_elim, "Time (days since beginning)", "Number of Cars"]],
                      titles=["Original Signal", "Signal without constant component"])
show_and_save()

NFD = 10
top4_idx = np.argpartition(X, -NFD)[-NFD:]
top4_idx_sorted = top4_idx[np.argsort(X[top4_idx])[::-1]]

print(f"Top {NFD} cele mai principale frecvente (prima este 0 adica media, si inca {
      NFD - 1} apoi): {freq_axis[top4_idx_sorted]}")
aux = 1 / freq_axis[top4_idx_sorted[1:]] / 3600 / 24  # Hz -> D
np.set_printoptions(suppress=True)
print(f"Perioadele in zile, fara f = 0: {aux}")

# esantion 0 -> sambata ora 00:00
# esantion 7 * 24 * k + 48, k>=0 -> luni
# esantion > 1000 -> k > (1000 - 48) / 7 / 24

k = (1000 - 48) // (7 * 24) + 10
start = 7 * 24 * k + 48
period = 24 * 30

plot_samples(time_axis[start:start+period], x[start:start+period],
             "Time (days since beginning)", "Number of Cars")
show_and_save()

# Frecventa cea mai mare din top 10 frecvente (inclusiv media)
threshold = 1.15905307e-05
idx = np.searchsorted(freq_axis, threshold)

# Elimin frecventele aflate dupa threshold
# Adica, frecventele care cu siguranta nu se aflta in top 10 cele mai influente frecvente
X_filter = np.concatenate(
    [X_orig[:idx], np.zeros_like(X_orig[idx:N-idx]), X_orig[N-idx:]])
x_filter = np.fft.ifft(X_filter)

plot_samples_subplots([[time_axis, x, "Time (days since beginning)", "Number of Cars"],
                       [time_axis, x_filter, "Time (days since beginning)", "Number of Cars"]],
                      titles=["Original Signal", "Filtered signal"])
show_and_save()
