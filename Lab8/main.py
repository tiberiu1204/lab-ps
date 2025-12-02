import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

plt_index = 0


def show_and_save():
    global plt_index
    plt.savefig(f"plot_{plt_index}.pdf")
    plt_index += 1
    plt.show()


N = 1001

t = np.arange(N) / N
f1 = 10
f2 = 50
trend_ = 4 * (t ** 2)
season_ = np.sin(2 * np.pi * t * f1) + np.sin(2 * np.pi * t * f2)
noise_ = np.random.normal(loc=0, scale=.3, size=N)
signal_ = trend_ + season_ + noise_

N = 1000
t = t[:N]
trend = trend_[:N]
season = season_[:N]
noise = noise_[:N]
signal = signal_[:N]

plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(t, trend)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Trend')
plt.xlabel('Time (t)')
plt.ylabel('Value')

plt.subplot(4, 1, 2)
plt.plot(t, season)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Season')
plt.xlabel('Time (t)')
plt.ylabel('Value')

plt.subplot(4, 1, 3)
plt.plot(t, noise)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Noise')
plt.xlabel('Time (t)')
plt.ylabel('Value')

plt.subplot(4, 1, 4)
plt.plot(t, signal)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Signal')
plt.xlabel('Time (t)')
plt.ylabel('Value')

plt.tight_layout()
show_and_save()

cor_vec = np.convolve(signal, signal[::-1])
cor_vec_corr = np.correlate(signal, signal, mode='full')
print(np.all(cor_vec == cor_vec_corr))

plt.plot(np.arange(len(cor_vec)), cor_vec)
plt.title("Autocorrelation vector")
plt.xlabel('Component')
plt.ylabel('Value')
show_and_save()


def ar(m=246, p=1):
    if p >= m:
        return 0, 0, np.inf

    Y = []

    for k in range(p + N - m, N):
        Y.append(signal[k-p:k][::-1])

    Y = np.matrix(Y)
    target = np.matrix(signal[p + N - m:N]).T
    G = Y.T * Y

    try:
        x_star = np.linalg.pinv(G) * Y.T * target
    except np.linalg.LinAlgError:
        return 0, 0, np.inf

    y_pred = (x_star.T @ np.matrix(signal[N-p:N][::-1]).T)[0, 0]
    t_pred = 1

    mse = (signal_[-1] - y_pred) ** 2

    return t_pred, y_pred, mse


t_pred, y_pred, _ = ar()
plt.plot(t, signal)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Signal')
plt.xlabel('Time (t)')
plt.ylabel('Value')

plt.scatter(t_pred, y_pred, label='y_pred', color='red')
plt.scatter(t_pred, signal_[-1], label='y_actual', color='green')

plt.legend()
show_and_save()

min_mse = np.inf
m_opt = 0
p_opt = 0
MP = []

for m in range(1, N, 5):
    for p in range(1, N, 5):
        MP.append((m, p)) if p < m else None
for (m, p) in tqdm(MP, "Progress: "):
    _, _, mse = ar(m, p)
    if min_mse > mse:
        min_mse = mse
        p_opt = p
        m_opt = m

print(f"Opt Params: m={m_opt} p={p_opt}")  # 246, 1
