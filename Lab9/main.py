import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA

plt_index = 0


def show_and_save():
    global plt_index
    plt.savefig(f"plot_{plt_index}.pdf")
    plt_index += 1
    plt.show()


# 1

def generate_ts(N=1000, f1=10, f2=50):
    t = np.arange(N + 1) / (N + 1)
    f1 = 10
    f2 = 50
    trend = 4 * (t ** 2)
    season = np.sin(2 * np.pi * t * f1) + np.sin(2 * np.pi * t * f2)
    noise = np.random.normal(loc=0, scale=.3, size=N + 1)
    signal = trend + season + noise

    return t, trend, season, noise, signal


t_, trend_, season_, noise_, signal_ = generate_ts()
t, trend, season, noise, signal = t_[
    :-1], trend_[:-1], season_[:-1], noise_[:-1], signal_[:-1]
t_pred = 1001 / 1000

plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(t, trend)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Trend')
plt.xlabel('Time (s)')
plt.ylabel('Value')

plt.subplot(4, 1, 2)
plt.plot(t, season)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Season')
plt.xlabel('Time (s)')
plt.ylabel('Value')

plt.subplot(4, 1, 3)
plt.plot(t, noise)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Noise')
plt.xlabel('Time (s)')
plt.ylabel('Value')

plt.subplot(4, 1, 4)
plt.plot(t, signal)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Signal')
plt.xlabel('Time (s)')
plt.ylabel('Value')

plt.tight_layout()
show_and_save()

# 2


def _2():
    def me_1(x, alpha=.5):
        s = []
        for t in range(len(x)):
            sum_ind = np.arange(t)
            st = alpha * np.sum((1 - alpha) ** sum_ind *
                                x[t - sum_ind] + ((1 - alpha) ** t) * x[0])
            s.append(st)
        return s

    s = me_1(signal)

    plt.figure(figsize=(10, 6))

    plt.plot(t, signal, color='blue', label='Original signal')
    plt.scatter(t_pred, s[-1], color='red', label='S series predition')
    plt.scatter(t_pred, signal_[-1], color='blue', label='Actual')

    plt.axhline(0, color='black', linewidth=.8)

    plt.title('Original Signal and S Series')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')

    plt.legend()
    plt.tight_layout()
    show_and_save()

    n = 100
    alphas = np.arange(n) / (n - 1)
    alpha_opt = 0
    s_opt = s
    s_init = s

    def loss(s, x):
        s = s[:-1]
        x = x[1:]
        return np.sum((s - x) ** 2)

    min_loss = np.inf
    for alpha in tqdm(alphas):
        s = me_1(signal, alpha)
        L = loss(s, signal)
        if L < min_loss:
            min_loss = L
            alpha_opt = alpha
            s_opt = s

    print(f"Apha opt: {alpha_opt}")

    plt.figure(figsize=(10, 6))

    plt.plot(t, signal, label='Original signal')
    plt.scatter(t_pred, s_init[-1], color='red', label='Prediction alpha=.5')
    plt.scatter(t_pred, s_opt[-1], color='green',
                label=f'Prediction alpha={alpha_opt} (opt)')
    plt.scatter(t_pred, signal_[-1], color='blue', label='Actual')

    plt.axhline(0, color='black', linewidth=.8)

    plt.title('Original Signal vs S series')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')

    plt.legend()
    plt.tight_layout()
    show_and_save()

    def me_2(x, alpha=.5, beta=.5):
        n = len(x)
        s = [0] * n
        b = [0] * n
        s[0] = x[0]
        b[0] = x[1] - x[0]
        for t in range(1, n):
            s[t] = alpha * x[t] + (1 - alpha) * (s[t - 1] + b[t - 1])
            b[t] = beta * (s[t] - s[t - 1]) + (1 - beta) * b[t - 1]

        return s, b

    a = np.arange(100)
    i, j = np.meshgrid(a, a, indexing='ij')

    domain = np.stack([i.ravel(), j.ravel()], axis=1) / 99
    alpha2_opt = 0
    beta2_opt = 0
    s2_opt = None
    b2_opt = None

    def loss(s, b, x):
        x_pred = np.array(s) + np.array(b)
        return np.sum(x_pred[:-1] - x[1:]) ** 2

    min_loss = np.inf
    for alpha, beta in tqdm(domain):
        s, b = me_2(signal, alpha, beta)
        L = loss(s, b, signal)
        if L < min_loss:
            min_loss = L
            alpha2_opt = alpha
            beta2_opt = beta
            s2_opt = s
            b2_opt = b

    print(f"Alpha opt: {alpha2_opt}\nBeta opt: {beta2_opt}")

    plt.figure(figsize=(10, 6))

    plt.plot(t, signal, label='Original signal')
    plt.scatter(t_pred, s2_opt[-1] + b2_opt[-1], color='green',
                label=f'Prediction me2 alpha={alpha2_opt} beta={beta2_opt} (opt)')
    plt.scatter(t_pred, s_opt[-1], color='red',
                label=f'Prediction me1 alpha={alpha_opt} (opt)')
    plt.scatter(t_pred, signal_[-1], color='blue', label='Actual')

    plt.axhline(0, color='black', linewidth=.8)

    plt.title('Original Signal vs S series')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')

    plt.legend()
    plt.tight_layout()
    show_and_save()

    def me_3(x, L, alpha=.5, beta=.5, gamma=.5):
        n = len(x)
        s = np.zeros(n)
        b = np.zeros(n)
        c = np.ones(n)

        s[0] = x[0]
        b[0] = x[1] - x[0]

        for t in range(L):
            c[t] = x[t] / s[0]

        for t in range(1, n):
            if t - L >= 0:
                c_past = c[t - L]
            else:
                c_past = c[t % L]

            s[t] = alpha * (x[t] / c_past) + (1 - alpha) * \
                (s[t - 1] + b[t - 1])
            b[t] = beta * (s[t] - s[t - 1]) + (1 - beta) * b[t - 1]

            c[t] = gamma * (x[t] / s[t]) + (1 - gamma) * c_past

        return s, b, c

    def loss_me3(s, b, c, x, L):
        x_pred = []
        for t in range(1, len(x)):
            if t - L >= 0:
                season = c[t - L]
            else:
                season = c[t % L]
            x_pred.append((s[t-1] + b[t-1]) * season)

        x_pred = np.array(x_pred)
        return np.sum((x_pred - x[1:]) ** 2)

    n = 20
    a = np.arange(n)
    i, j, k = np.meshgrid(a, a, a, indexing='ij')
    domain3 = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) / (n - 1)

    alpha3_opt = beta3_opt = gamma3_opt = 0
    s3_opt = b3_opt = c3_opt = None
    min_loss = np.inf

    L = 10

    for alpha, beta, gamma in tqdm(domain3):
        s, b, c = me_3(signal, L, alpha, beta, gamma)
        Lval = loss_me3(s, b, c, signal, L)

        if Lval < min_loss:
            min_loss = Lval
            alpha3_opt, beta3_opt, gamma3_opt = alpha, beta, gamma
            s3_opt, b3_opt, c3_opt = s, b, c

    def forecast_me3(s, b, c, L, t, m):
        idx = t - L + 1 + ((m - 1) % L)
        seasonal = c[idx]
        return (s[t] + m * b[t]) * seasonal

    plt.figure(figsize=(10, 6))

    plt.plot(t, signal, label='Original signal')
    plt.scatter(t_pred,
                forecast_me3(s3_opt, b3_opt, c3_opt, L, len(signal)-1, 1),
                label=f'Prediction me3 a={alpha3_opt:.2f} b={beta3_opt:.2f} g={gamma3_opt:.2f}')
    plt.scatter(t_pred, s2_opt[-1] + b2_opt[-1],
                label=f'Prediction me2 alpha={alpha2_opt} beta={beta2_opt} (opt)')
    plt.scatter(t_pred, s_opt[-1],
                label=f'Prediction me1 alpha={alpha_opt} (opt)')
    plt.scatter(t_pred, signal_[-1], label='Actual')

    plt.legend()
    plt.tight_layout()
    show_and_save()


def _3():
    def solve_ma_model(signal, p):
        s = np.array(signal)
        ma = np.convolve(s, np.ones(p)/p, mode='valid')
        eps = s[p-1:] - ma

        X = np.column_stack([eps[p-1-j: len(eps)-1-j]
                            for j in range(p)] + [np.ones(len(eps)-p)])
        Y_reg = ma[p:]

        params, _, _, _ = np.linalg.lstsq(X, Y_reg, rcond=None)

        Y_pred = eps[p:] + X @ params

        return Y_pred, params

    p = 200

    Y_pred, params = solve_ma_model(signal, p)

    t_pred_ma = t[2*p-1:]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label='Original Signal')
    plt.plot(t_pred_ma, Y_pred, color='red', label='Y_pred (Model)')
    plt.legend()
    plt.title(f'MA({p}) Model: Signal vs Predictions')
    plt.xlabel('Time')
    plt.ylabel('Value')
    show_and_save()


def _4():
    limit = 20
    step = 5
    pairs = [(p, q) for p in range(1, limit + 1, step)
             for q in range(1, limit + 1, step)]

    best_aic = float('inf')
    best_order = None
    best_model_res = None

    for p, q in tqdm(pairs):
        try:
            model = ARIMA(signal, order=(p, 0, q))
            res = model.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_order = (p, q)
                best_model_res = res
        except:
            continue

    Y_fitted = best_model_res.predict()

    Y_future = best_model_res.forecast(steps=1)

    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label='Original Signal')
    plt.plot(t, Y_fitted, color='orange', linestyle='--',
             label=f'In-Sample Fit {best_order}')
    plt.scatter(t_pred, Y_future, color='red',
                marker='*', s=100, label='Next Step Forecast')
    plt.scatter(t_pred, signal_[-1], color='blue',
                marker='*', s=100, label='Next Step Actual')
    plt.legend()
    plt.title(f'ARMA{best_order} Model: Fit & Forecast')
    show_and_save()


_2()
_3()
_4()
