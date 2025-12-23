import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from l1regls import l1regls
import cvxopt as cv


plt_index = 0
N = 1000


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


t_, trend_, season_, noise_, signal_ = generate_ts(N)
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


def ar(m=246, p=1, N=1000):
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

    return x_star, t_pred, y_pred, mse


def create_design_matrix(signal, p_max):
    num_samples = len(signal)
    start = p_max

    Y = []
    target = []

    for k in range(start, num_samples):
        Y.append(signal[k-p_max:k][::-1])
        target.append(signal[k])

    return np.array(Y), np.array(target).reshape(-1, 1)


def ar_greedy(Y, target, max_features=5):
    n_samples, n_features = Y.shape

    selected_indices = []

    x_final = np.zeros(n_features)

    for _ in range(max_features):
        best_loss = np.inf
        best_i = -1
        x_best = None

        for i in range(n_features):
            try_indices = selected_indices + [i]

            Y_subset = Y[:, try_indices]

            x, _, _, _ = np.linalg.lstsq(
                Y_subset, target, rcond=None)

            predictions = Y_subset @ x
            loss = np.sum((target - predictions) ** 2)

            if loss < best_loss:
                best_loss = loss
                best_i = i
                x_best = x

        if best_i != -1:
            selected_indices.append(best_i)

            x_temp = np.zeros(n_features)
            x_temp[selected_indices] = x_best.flatten()
            x_final = x_temp

    return x_final, selected_indices


def ar_lasso_cvxopt(Y, target, l=1.0):
    Y = cv.matrix(Y)
    target = cv.matrix(target)
    return l1regls(Y, target, l)


p_max = 100
l = 10
Y, target = create_design_matrix(signal, p_max)

x_ar, _, _, _ = ar(N, p_max, N)
x_greedy, _ = ar_greedy(Y, target, max_features=p_max//3)
x_l1 = np.array(ar_lasso_cvxopt(Y, target, 10))


def _3():

    lags = np.arange(1, p_max + 1)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.scatter(lags, np.asarray(x_ar).flatten())
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Coeficienți AR (Least Squares)")
    plt.xlabel("Lag")
    plt.ylabel("Valoare coeficient")
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.scatter(lags, x_greedy)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f"Coeficienți AR Greedy (sparse) max_features={p_max//3}")
    plt.xlabel("Lag")
    plt.ylabel("Valoare coeficient")
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.scatter(lags, x_l1.flatten())
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f"Coeficienți AR LASSO (L1) l={l}")
    plt.xlabel("Lag")
    plt.ylabel("Valoare coeficient")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    show_and_save()


def get_roots(coeff):
    C = []
    for i, c in enumerate(coeff[::-1]):
        line = np.zeros(len(coeff))
        line[i-1] = 1
        if isinstance(c, np.ndarray):
            c = c.item()
        line[-1] = -c
        C.append(line)
    C = np.array(C)
    eig, _ = np.linalg.eig(C)
    return eig


def _4():
    # n = 100
    # coeff = np.random.rand(n)
    # x^2 + 2x + 1
    coeff = [2, 1]  # c0 aici este coef pt x^(n-1) si dau reverse in get_roots
    print(f"coeff = {coeff}")
    print(f"roots = {get_roots(coeff)}")


def _5():
    def prepare(x):
        x = np.asarray(x).ravel()
        x = np.append(x, -1)
        x /= x[0]
        return x

    roots_ar = get_roots(prepare(x_ar))
    roots_greedy = get_roots(prepare(x_greedy))
    roots_l1 = get_roots(prepare(x_l1))

    def is_stationary(roots):
        return "stationar" if np.all(np.abs(roots) > 1) else "nestationar"
    print(f"x_ar: {is_stationary(roots_ar)}")
    print(f"x_greedy: {is_stationary(roots_greedy)}")
    print(f"x_l1: {is_stationary(roots_l1)}")

    # Plot
    plt.figure(figsize=(12, 12))

    root_sets = [(roots_ar, "AR Least Squares"),
                 (roots_greedy, "AR Greedy"),
                 (roots_l1, "AR LASSO")]

    for i, (roots, title) in enumerate(root_sets, 1):
        ax = plt.subplot(3, 1, i)
        unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=1)
        ax.add_artist(unit_circle)
    
        inside = np.abs(roots) <= 1
        outside = np.abs(roots) > 1
    
        ax.scatter(roots[outside].real, roots[outside].imag, color='red', marker='o', s=5, label='|z| > 1')
        ax.scatter(roots[inside].real, roots[inside].imag, color='blue', marker='o', s=5, label='|z| ≤ 1')
    
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal', 'box')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(f"Roots: {title}")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    show_and_save()


_3()
_4()
_5()
