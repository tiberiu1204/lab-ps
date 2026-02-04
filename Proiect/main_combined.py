from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.stats import skew, kurtosis


def downsample(y, sr, target_sr=5512.5, plot_fft=False, save_output=False):

    from scipy.signal import butter, filtfilt, resample_poly
    from math import gcd

    nyquist_orig = sr / 2

    cutoff = (target_sr / 2) * 0.9
    normalized_cutoff = cutoff / nyquist_orig

    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99

    b, a = butter(8, normalized_cutoff, btype="low")
    y_filtered = filtfilt(b, a, y)

    up = int(target_sr * 2)
    down = int(sr * 2)

    g = gcd(up, down)
    up //= g
    down //= g

    y_down = resample_poly(y_filtered, up, down)

    print(f"Downsampled: {sr} Hz → {target_sr} Hz (anti-aliasing applied)")

    if plot_fft:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        fft_orig = np.abs(np.fft.rfft(y))
        freqs_orig = np.fft.rfftfreq(len(y), d=1 / sr)
        plt.plot(freqs_orig, fft_orig, label="Original", alpha=0.5)
        fft_down = np.abs(np.fft.rfft(y_down))
        freqs_down = np.fft.rfftfreq(len(y_down), d=1 / target_sr)
        plt.plot(freqs_down, fft_down, label="Downsampled")
        plt.axvline(
            target_sr / 2,
            color="r",
            linestyle="--",
            label=f"Nyquist ({target_sr / 2:.0f} Hz)",
        )
        plt.legend()
        plt.title("Anti-Aliased Downsampling")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.show()

    if save_output:
        sf.write("downsampled.wav", y_down, int(target_sr))

    return y_down, target_sr


def hanning_window(N):

    n = np.arange(N)
    w = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    return w


def compute_STFT(y, sr, window, win_length=512, n_fft=4096, hop_length=32):

    N_frames = 1 + (len(y) - win_length) // hop_length
    K_bins = n_fft // 2 + 1

    STFT = np.zeros((K_bins, N_frames), dtype=complex)

    for n in range(N_frames):
        start = n * hop_length
        frame = y[start: start + win_length]

        frame = frame * window

        frame_padded = np.zeros(n_fft)
        frame_padded[:win_length] = frame

        spectrum = np.fft.rfft(frame_padded)
        STFT[:, n] = spectrum

    freqs = np.linspace(0, sr / 2, K_bins)
    times = np.arange(N_frames) * hop_length / sr

    return STFT, freqs, times


def compute_STFT_spectrogram(y, sr, win_length=512, n_fft=4096, hop_length=32):

    window = hanning_window(win_length)

    STFT, freqs, times = compute_STFT(
        y, sr, window, win_length, n_fft, hop_length)

    M = np.abs(STFT)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        20 * np.log10(M + 1e-6),
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
    )
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("STFT Magnitude Spectrogram")
    plt.show()

    return M, STFT, freqs, times


def log_frequency_axis(n_semitones=780, fmin=29.1, f_ref=440):

    k_log = np.arange(n_semitones)
    semitones_from_A4 = 22 + (k_log / 10.0) - 69
    f_log = f_ref * (2 ** (semitones_from_A4 / 12))

    return f_log, k_log


def compute_window_derivative(win_length):

    n = np.arange(win_length)
    val = (np.pi / (win_length - 1)) * np.sin(2 * np.pi * n / (win_length - 1))
    return val


def compute_reassigned_spectrogram(
    y,
    sr,
    win_length=512,
    n_fft=4096,
    hop_length=32,
    n_semitones=780,
    fmin=29.1,
    plot_spectogram=True,
):

    window = hanning_window(win_length)
    window_prime = compute_window_derivative(win_length)

    STFT, freqs_lin, _ = compute_STFT(
        y, sr, window, win_length, n_fft, hop_length)
    STFT_prime, _, _ = compute_STFT(
        y, sr, window_prime, win_length, n_fft, hop_length)

    M = np.abs(STFT)

    freqs_norm = np.fft.rfftfreq(n_fft, d=1.0)
    freqs_grid = freqs_norm[:, np.newaxis]

    mag_sq = np.real(STFT * np.conj(STFT))
    valid_mask = mag_sq > 1e-10 * np.max(mag_sq)

    ratio = np.zeros_like(STFT)
    ratio[valid_mask] = STFT_prime[valid_mask] / STFT[valid_mask]

    correction = -np.imag(ratio) / (2 * np.pi)
    IF_norm = freqs_grid + correction

    IF_Hz = IF_norm * sr
    IF_Hz = np.clip(IF_Hz, 0, sr / 2)

    f_log, _ = log_frequency_axis(n_semitones, fmin)
    K_lin, N_frames = M.shape
    K_log = len(f_log)
    MIF = np.zeros((K_log, N_frames))

    for n in range(N_frames):
        for k_lin in range(K_lin):
            f_if = IF_Hz[k_lin, n]
            dist = np.abs(f_log - f_if)
            k_log_nearest = np.argmin(dist)
            MIF[k_log_nearest, n] += M[k_lin, n]

    if plot_spectogram:

        plt.figure(figsize=(10, 6))
        plt.imshow(20 * np.log10(M + 1e-6), aspect="auto", origin="lower")
        plt.title("Original Magnitude Spectrogram")
        plt.colorbar(label="Magnitude (dB)")
        plt.xlabel("Frame Index")
        plt.ylabel("Frequency Bin")
        plt.tight_layout()
        plt.savefig("plot_spectrogram_original.pdf")
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.imshow(20 * np.log10(MIF + 1e-6), aspect="auto", origin="lower")
        plt.title("Reassigned Magnitude Spectrogram MIF (log freq)")
        plt.colorbar(label="Magnitude (dB)")
        plt.xlabel("Frame Index")
        plt.ylabel("Log-frequency Bin")
        plt.tight_layout()
        plt.savefig("plot_spectrogram_reassigned.pdf")
        plt.show()
        plt.close()

    return MIF, IF_Hz, M, f_log, freqs_lin


def _find_peaks_manual(signal, height=None, distance=None):

    peaks = []
    last_peak_idx = -distance if distance else -999999

    for i in range(1, len(signal) - 1):
        current_val = signal[i]
        if signal[i - 1] < current_val and current_val >= signal[i + 1]:
            if height is not None and current_val < height:
                continue

            if distance is not None:
                if (i - last_peak_idx) < distance:
                    if signal[i] > signal[last_peak_idx]:
                        peaks.pop()
                        peaks.append(i)
                        last_peak_idx = i
                    continue

            peaks.append(i)
            last_peak_idx = i

    return np.array(peaks)


def compute_onset_detection_function(MIF, hop_length_seconds=None, plot_onset=True):

    vec_freq = np.array([[0.3], [1.0], [0.3]])
    vec_time = np.array([[1, 1, 1, 0, -1, -1, -1]])
    M_K = vec_freq @ vec_time

    MIF_K = convolve2d(MIF, M_K, mode="same", boundary="fill", fillvalue=0)

    alpha_On = np.max(MIF_K, axis=0)

    global_max = np.max(alpha_On)
    threshold = 0.30 * global_max

    onset_indices = _find_peaks_manual(alpha_On, height=threshold, distance=15)

    if len(onset_indices) == 0 or onset_indices[0] > 10:

        initial_energy = np.mean(
            np.max(MIF[:, :5], axis=0)) if MIF.shape[1] > 5 else 0
        if initial_energy > threshold * 0.5:
            onset_indices = np.insert(onset_indices, 0, 0)

    onset_times = None
    if hop_length_seconds is not None:
        onset_times = onset_indices * hop_length_seconds

    if plot_onset:
        times_axis = (
            np.arange(len(alpha_On)) * hop_length_seconds
            if hop_length_seconds
            else np.arange(len(alpha_On))
        )
        plt.figure(figsize=(12, 5))
        plt.plot(times_axis, alpha_On, "b-", label="Onset Function alpha_On")
        if len(onset_indices) > 0:
            plt.plot(
                times_axis[onset_indices],
                alpha_On[onset_indices],
                "ro",
                label="Detected Onsets",
            )
        plt.hlines(
            threshold,
            times_axis[0],
            times_axis[-1],
            colors="g",
            linestyles="--",
            label="Threshold (20% max)",
        )

        plt.title("Onset Detection Function")
        plt.xlabel("Time (s)" if hop_length_seconds else "Frame Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.tight_layout()
        plt.savefig("plot_onset_detection.pdf")
        plt.show()
        plt.close()

    return alpha_On, onset_indices, onset_times, MIF_K


def freq_to_log_bin(freq):

    if freq <= 0:
        return 0
    val = 10 * (12 * np.log2(freq / 440.0) + 47)
    return int(np.round(val))


def log_bin_to_freq(k):

    return 440.0 * (2.0 ** ((k / 10.0 - 47.0) / 12.0))


def create_harmonic_template(f0, beta, max_bins=781):

    template = np.zeros(max_bins)
    harmonics_info = []

    for h in range(10):

        term = 1 + beta * ((h + 1) ** 2)
        f_h = (h + 1) * f0 * np.sqrt(term)

        k = freq_to_log_bin(f_h)

        if 0 <= k < max_bins:

            magnitude = 2.0 if h < 2 else 1.0
            template[k] = magnitude

            harmonics_info.append((f_h, magnitude, h))

    return template, harmonics_info


def compute_template_correlation(MIF_frame, template):

    min_len = min(len(MIF_frame), len(template))
    return np.dot(MIF_frame[:min_len], template[:min_len])


def compute_harmonic_sum_interpolated(M_frame, freqs_lin, f0, beta, n_harmonics=10):

    total_sum = 0.0

    for h in range(n_harmonics):

        term = 1 + beta * ((h + 1) ** 2)
        fh = (h + 1) * f0 * np.sqrt(term)

        if fh > freqs_lin[-1]:
            continue

        ah = np.interp(fh, freqs_lin, M_frame)

        weight = 2.0 if h < 2 else 1.0
        total_sum += weight * ah

    return total_sum


def estimate_f0(
    M,
    freqs_lin,
    MIF,
    onset_frame,
    next_onset_frame,
    f_min=30,
    f_max=400,
    debug_plot=False,
):

    num_bins = MIF.shape[0]

    duration = next_onset_frame - onset_frame
    end_acc = onset_frame + int(0.2 * duration)
    end_acc = max(onset_frame + 1, min(end_acc, M.shape[1]))

    M_acc = np.mean(M[:, onset_frame:end_acc], axis=1)

    best_score = -1
    best_f0 = 0
    best_beta = 0

    betas = np.linspace(0, 0.001, 100)

    k_min = freq_to_log_bin(f_min)
    k_max = freq_to_log_bin(f_max)

    if debug_plot:
        plot_f0s = []
        plot_scores = []

    for k_candidate in range(k_min, k_max):
        f0_candidate = log_bin_to_freq(k_candidate)

        for beta in betas:

            score = compute_harmonic_sum_interpolated(
                M_acc, freqs_lin, f0_candidate, beta
            )

            if debug_plot and beta == 0:
                plot_f0s.append(f0_candidate)
                plot_scores.append(score)

            if score > best_score:
                best_score = score
                best_f0 = f0_candidate
                best_beta = beta

    if debug_plot:

        plt.figure(figsize=(10, 4))
        plt.plot(plot_f0s, plot_scores, color="k", label="Harmonic Sum Score")
        plt.axvline(
            x=best_f0, color="r", linestyle="--", label=f"Best f0: {best_f0:.2f} Hz"
        )
        plt.title("1. Grid Search: Harmonic Sum Score (Linear Spectrogram)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Score (Sum)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plot_f0_grid_search.pdf")
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(
            freqs_lin,
            M_acc,
            label="Accumulated Linear Spectrogram",
            color="green",
            alpha=0.6,
        )

        for h in range(10):
            term = 1 + best_beta * ((h + 1) ** 2)
            fh = (h + 1) * best_f0 * np.sqrt(term)
            if fh < freqs_lin[-1]:
                color = "r" if h < 2 else "b"

                label = (
                    f"Harmonic {h} (weighted 2.0)"
                    if h < 2
                    else ("Harmonic 2-9" if h == 2 else None)
                )
                plt.axvline(x=fh, color=color, linestyle="--",
                            alpha=0.5, label=label)

        plt.xlim(0, best_f0 * 12)
        plt.title(
            f"2. Harmonics Detected ($f_0$={best_f0:.1f}Hz, β={
                best_beta:.5f}) - Linear M"
        )
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("plot_f0_harmonics.pdf")
        plt.show()
        plt.close()

        template, harmonics_info = create_harmonic_template(
            best_f0, best_beta, max_bins=num_bins
        )
        plt.figure(figsize=(10, 4))
        plt.stem(
            np.arange(len(template)),
            template,
            linefmt="b-",
            markerfmt="bo",
            basefmt=" ",
            label="Template Magnitude",
        )

        for h in range(10):
            ideal_fh = (h + 1) * best_f0
            ideal_k = freq_to_log_bin(ideal_fh)
            if 0 <= ideal_k < len(template):
                plt.plot(
                    ideal_k,
                    0.1,
                    "rx",
                    markersize=10,
                    markeredgewidth=2,
                    label="Ideal (β=0)" if h == 0 else None,
                )

        for fh_hz, mag, h_idx in harmonics_info:
            k = freq_to_log_bin(fh_hz)
            if 0 <= k < len(template):
                plt.annotate(
                    f"H{h_idx}",
                    (k, mag),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                )
        plt.title(f"3. Harmonic Template (f0={
                  best_f0:.1f}Hz, β={best_beta:.5f})")
        plt.xlabel("Log-frequency Bin Index")
        plt.ylabel("Weight")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plot_harmonic_template.pdf")
        plt.show()
        plt.close()

    midi_pitch = int(np.floor(12 * np.log2(best_f0 / 440.0) + 69 + 0.5))

    distance = next_onset_frame - onset_frame
    n0_offset = int(0.10 * distance)
    n0 = onset_frame + max(1, n0_offset)

    num_frames = M.shape[1]
    if n0 >= num_frames:
        n0 = num_frames - 1
    if n0 >= next_onset_frame:
        n0 = next_onset_frame - 1

    current_f0_fwd = best_f0
    current_f0_bwd = best_f0

    bwd_f0s, bwd_frames = [], []
    fwd_f0s, fwd_frames = [], []

    for n in range(n0 - 1, onset_frame - 1, -1):

        best_score_local, best_f0_local = find_best_local_match(
            M[:, n], freqs_lin, current_f0_bwd, best_beta
        )
        current_f0_bwd = best_f0_local
        bwd_f0s.insert(0, current_f0_bwd)
        bwd_frames.insert(0, n)

    search_end = min(next_onset_frame + 10, num_frames)

    for n in range(n0, search_end):
        best_score_local, best_f0_local = find_best_local_match(
            M[:, n], freqs_lin, current_f0_fwd, best_beta
        )
        current_f0_fwd = best_f0_local
        fwd_f0s.append(current_f0_fwd)
        fwd_frames.append(n)

    full_frames = bwd_frames + fwd_frames
    full_f0s = bwd_f0s + fwd_f0s

    full_Cmax = []
    max_Cmax_so_far = 0.0

    num_bins_log = MIF.shape[0]

    for i, n in enumerate(full_frames):
        if n >= MIF.shape[1]:
            full_Cmax.append(0)
            continue

        f0_val = full_f0s[i]

        template, _ = create_harmonic_template(
            f0_val, best_beta, max_bins=num_bins_log)

        cmax_val = compute_template_correlation(MIF[:, n], template)
        full_Cmax.append(cmax_val)

        if cmax_val > max_Cmax_so_far:
            max_Cmax_so_far = cmax_val

    return (
        best_f0,
        best_beta,
        midi_pitch,
        full_frames,
        full_f0s,
        full_Cmax,
        max_Cmax_so_far,
    )


def find_best_local_match(MIF_frame, freqs_lin, current_f0, beta):

    k_current = freq_to_log_bin(current_f0)
    candidates_k = [k_current - 1, k_current, k_current + 1]

    best_score = -1
    best_f0 = current_f0

    for k_cand in candidates_k:
        if k_cand < 0:
            continue

        f_cand = log_bin_to_freq(k_cand)
        template, _ = create_harmonic_template(f_cand, beta)
        score = compute_template_correlation(MIF_frame, template)

        if score > best_score:
            best_score = score
            best_f0 = f_cand

    return best_score, best_f0


def detect_offset(
    full_frames,
    full_f0s,
    full_Cmax,
    max_Cmax,
    onset_frame,
    next_onset_frame,
    debug_plot=False,
):

    threshold = 0.05 * max_Cmax
    offset_frame = full_frames[-1]

    low_score_counter = 0

    try:
        start_idx = full_frames.index(
            onset_frame) if onset_frame in full_frames else 0
    except ValueError:
        start_idx = 0

    for i in range(start_idx, len(full_frames)):
        n = full_frames[i]
        cmax = full_Cmax[i]

        if n >= next_onset_frame:
            offset_frame = next_onset_frame
            break

        if cmax < threshold:
            low_score_counter += 1
        else:
            low_score_counter = 0

        if low_score_counter >= 4:

            offset_frame = full_frames[i - 3]
            break

    valid_indices = [
        i for i, f in enumerate(full_frames) if f <= offset_frame and f >= onset_frame
    ]
    if not valid_indices:

        f0_contour = np.array([full_f0s[0]])
        final_offset = onset_frame + 1
    else:
        f0_contour = np.array([full_f0s[i] for i in valid_indices])
        final_offset = offset_frame

    if debug_plot:
        plt.figure(figsize=(10, 4))
        ax1 = plt.gca()
        ax1.plot(full_frames, full_Cmax, "b-",
                 label="Cmax (Cross-correlation)")
        ax1.axhline(
            y=threshold,
            color="r",
            linestyle="--",
            label=f"Threshold (5% of max={max_Cmax:.1f})",
        )
        ax1.axvline(x=final_offset, color="g",
                    linestyle="-", label="Detected Offset")
        ax1.axvline(x=onset_frame, color="k", linestyle=":", label="Onset")
        ax1.axvline(x=next_onset_frame, color="k",
                    linestyle="-.", label="Next Onset")

        ax1.set_title("Step D: Offset Detection (Cmax Thresholding)")
        ax1.set_ylabel("Cmax")
        ax1.set_xlabel("Frame Index")
        ax1.legend()
        ax1.grid(True)
        from matplotlib.ticker import MaxNLocator

        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig("plot_offset_cmax.pdf")
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(full_frames, full_f0s, "m-", label="Tracked f0 (Hz)")
        plt.axvline(x=final_offset, color="g",
                    linestyle="-", label="Detected Offset")
        plt.axvline(x=onset_frame, color="k", linestyle=":", label="Onset")
        plt.title("f0 Contour (Truncated at Offset)")
        plt.xlabel("Frame Index")
        plt.ylabel("Frequency (Hz)")
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig("plot_offset_f0_contour.pdf")
        plt.show()
        plt.close()

    return final_offset, f0_contour


def find_best_local_match_MIF(MIF_frame, current_f0, beta, max_bins):

    k_current = freq_to_log_bin(current_f0)
    candidates_k = [k_current - 1, k_current, k_current + 1]

    best_Cmax = -1
    best_f0 = current_f0

    for k_cand in candidates_k:
        if k_cand < 0:
            continue

        f_cand = log_bin_to_freq(k_cand)
        template, _ = create_harmonic_template(f_cand, beta, max_bins=max_bins)
        Cmax = compute_template_correlation(MIF_frame, template)

        if Cmax > best_Cmax:
            best_Cmax = Cmax
            best_f0 = f_cand

    return best_Cmax, best_f0


def spectral_envelope_modeling(
    M, freqs_lin, onset_frame, offset_frame, f0_contour, beta
):

    n_harmonics = 10
    note_duration = offset_frame - onset_frame

    if note_duration <= 0:
        return np.zeros((n_harmonics, 1)), np.zeros(1)

    if len(f0_contour) < note_duration:

        f0_extended = np.zeros(note_duration)
        f0_extended[: len(f0_contour)] = f0_contour
        f0_extended[len(f0_contour):] = f0_contour[-1] if len(f0_contour) > 0 else 0
        f0_contour = f0_extended
    elif len(f0_contour) > note_duration:
        f0_contour = f0_contour[:note_duration]

    ah = np.zeros((n_harmonics, note_duration))

    for n_rel in range(note_duration):
        n_abs = onset_frame + n_rel

        if n_abs >= M.shape[1]:
            break

        f0_n = f0_contour[n_rel]

        if f0_n <= 0:
            continue

        M_frame = M[:, n_abs]

        for h in range(n_harmonics):

            term = 1 + beta * ((h + 1) ** 2)
            fh = (h + 1) * f0_n * np.sqrt(term)

            ah[h, n_rel] = np.interp(fh, freqs_lin, M_frame)

    a_envelope = np.sum(ah, axis=0)

    return ah, a_envelope


def extract_intensity_segmentation(
    M, freqs_lin, onset_frame, offset_frame, f0_contour, beta, debug_plot=False
):

    note_duration = offset_frame - onset_frame

    if note_duration <= 0:
        return onset_frame, 0

    ah, a_envelope = spectral_envelope_modeling(
        M, freqs_lin, onset_frame, offset_frame, f0_contour, beta
    )

    if len(a_envelope) == 0 or np.max(a_envelope) == 0:
        return onset_frame, 0

    peak_index_rel = np.argmax(a_envelope)
    n_peak = onset_frame + peak_index_rel

    a_n_peak = a_envelope[peak_index_rel]

    if a_n_peak <= 0:
        intensity = -80
    else:
        intensity = 20 * np.log10(a_n_peak)

    if debug_plot:
        from matplotlib.ticker import MaxNLocator

        plot_end = min(offset_frame + 5, M.shape[1])
        plot_duration = plot_end - onset_frame

        f0_extended = np.zeros(plot_duration)
        if len(f0_contour) > 0:
            current_len = min(len(f0_contour), plot_duration)
            f0_extended[:current_len] = f0_contour[:current_len]
            if plot_duration > current_len:
                f0_extended[current_len:] = f0_contour[-1]

        ah_plot, a_envelope_plot = spectral_envelope_modeling(
            M, freqs_lin, onset_frame, plot_end, f0_extended, beta
        )

        plt.figure(figsize=(10, 4))
        x_frames = np.arange(onset_frame, plot_end)

        plt.plot(
            x_frames, a_envelope_plot, label="Accumulated Envelope a(n) [Linear Interp]"
        )
        plt.axvline(x=n_peak, color="r", linestyle="--", label="Peak (nPeak)")
        plt.axvline(x=onset_frame, color="g", linestyle=":", label="Onset")
        plt.axvline(x=offset_frame, color="orange",
                    linestyle="-", label="Offset")
        plt.title(f"Section F: Intensity = {
                  intensity:.1f} dB | Peak at frame {n_peak}")
        plt.xlabel("Frame Index")
        plt.ylabel("Summed Magnitude (Linear)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig("plot_intensity_segmentation.pdf")
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.imshow(
            ah_plot,
            aspect="auto",
            origin="lower",
            extent=[onset_frame, plot_end, 0.5, 10.5],
        )
        plt.colorbar(label="Magnitude (Linear)")
        plt.title("Section E: Spectral Envelope ah (Harmonic Magnitudes)")
        plt.xlabel("Frame Index")
        plt.ylabel("Harmonic Index h")
        plt.yticks(range(1, 11), [f"H{i}" for i in range(1, 11)])
        plt.tight_layout()
        plt.savefig("plot_spectral_envelope_ah.pdf")
        plt.show()
        plt.close()

    return n_peak, intensity


def harmonic_peak_mask_from_template(T, threshold_ratio=0.2):

    T = np.asarray(T)
    if T.size == 0:
        return np.zeros(0, dtype=bool)
    mx = np.max(T)
    if mx <= 0:
        return np.zeros_like(T, dtype=bool)
    return T >= (threshold_ratio * mx)


def remove_harmonic_peaks(M_frame, T_harm, threshold_ratio=0.2):

    mask = harmonic_peak_mask_from_template(
        T_harm, threshold_ratio=threshold_ratio)
    MR = np.array(M_frame, copy=True)
    MR[mask] = 0.0
    return MR


def _add_hann_peak(template, k_center, width_bins=5, weight=1.0):

    if width_bins < 1:
        return
    if width_bins % 2 == 0:
        width_bins += 1

    half = width_bins // 2
    k0 = max(0, k_center - half)
    k1 = min(len(template), k_center + half + 1)

    w = np.hanning(k1 - k0)
    template[k0:k1] += weight * w


def create_linear_harmonic_template(freqs_lin, f0, beta, n_harmonics=10, width_bins=5):

    T = np.zeros_like(freqs_lin, dtype=float)

    for h in range(n_harmonics):
        idx = h + 1
        term = 1.0 + beta * (idx**2)
        fh = idx * f0 * np.sqrt(term)

        if fh <= 0 or fh > freqs_lin[-1]:
            continue

        k = int(np.argmin(np.abs(freqs_lin - fh)))
        weight = 2.0 if h < 2 else 1.0
        _add_hann_peak(T, k, width_bins=width_bins, weight=weight)

    return T


def create_linear_subharmonic_template(
    freqs_lin, f0, beta, m, n_harmonics=10, width_bins=5
):

    f_virtual = f0 / float(m)
    T = np.zeros_like(freqs_lin, dtype=float)

    for h in range(n_harmonics):
        idx = h + 1

        if (idx % m) == 0:
            continue

        term = 1.0 + beta * (idx**2)
        fh = idx * f_virtual * np.sqrt(term)

        if fh <= 0 or fh > freqs_lin[-1]:
            continue

        k = int(np.argmin(np.abs(freqs_lin - fh)))
        weight = 2.0 if h < 2 else 1.0
        _add_hann_peak(T, k, width_bins=width_bins, weight=weight)

    return T


def template_energy_ratio(M_frame, T):

    denom = np.sum(M_frame) + 1e-9
    return float(np.dot(M_frame, T) / denom)


def first_derivative(x):
    x = np.asarray(x)
    if len(x) < 2:
        return np.array([])
    return np.diff(x)


def compute_vector_stats(arr, prefix):

    if len(arr) == 0:

        return {
            f"{prefix}_{s}": 0.0
            for s in ["mean", "var", "min", "max", "median", "skew", "kurt"]
        }

    std = np.std(arr)
    if len(arr) > 2 and std > 1e-12:
        s = skew(arr, bias=False, nan_policy="omit")
        k = kurtosis(arr, bias=False, nan_policy="omit")
    else:
        s = 0.0
        k = 0.0

    arr = np.array(arr)

    return {
        f"{prefix}_mean": np.mean(arr),
        f"{prefix}_var": np.var(arr),
        f"{prefix}_min": np.min(arr),
        f"{prefix}_max": np.max(arr),
        f"{prefix}_median": np.median(arr),
        f"{prefix}_skew": s,
        f"{prefix}_kurt": k,
    }


def compute_spectral_features(frame, freqs_axis):

    total = np.sum(frame)
    if total < 1e-9:
        return {k: 0.0 for k in ["centroid", "spread", "rolloff", "crest", "slope"]}

    centroid = np.sum(freqs_axis * frame) / total

    variance = np.sum(((freqs_axis - centroid) ** 2) * frame) / total
    spread = np.sqrt(variance)

    cumsum = np.cumsum(frame)
    threshold = 0.85 * cumsum[-1]
    rolloff_idx = np.searchsorted(cumsum, threshold)
    rolloff_idx = min(rolloff_idx, len(freqs_axis) - 1)
    rolloff = freqs_axis[rolloff_idx]

    mean_mag = np.mean(frame)
    crest = np.max(frame) / (mean_mag + 1e-9)

    if len(frame) > 1:
        A = np.vstack([freqs_axis, np.ones(len(freqs_axis))]).T
        slope, _ = np.linalg.lstsq(A, frame, rcond=None)[0]
    else:
        slope = 0

    return {
        "centroid": centroid,
        "spread": spread,
        "rolloff": rolloff,
        "crest": crest,
        "slope": slope,
    }


def compute_tristimulus(harm_vec):

    total = np.sum(harm_vec) + 1e-9
    t1 = harm_vec[0] / total
    t2 = np.sum(harm_vec[1:4]) / total if len(harm_vec) >= 4 else 0
    t3 = np.sum(harm_vec[4:]) / total if len(harm_vec) > 4 else 0
    return t1, t2, t3


def compute_spectral_irregularity(harm_vec):

    if len(harm_vec) < 2:
        return 0
    diff = np.abs(np.diff(harm_vec))
    return np.sum(diff) / (np.sum(harm_vec) + 1e-9)


def extract_features(
    M,
    freqs_lin,
    onset,
    peak,
    offset,
    f0,
    ah,
    a_envelope,
    beta,
    sr=5512.5,
    hop_length=256,
    f0_contour=None,
):

    feats = {}

    if offset <= onset or peak < onset or len(a_envelope) == 0:
        return {}

    idx_peak_rel = peak - onset

    idx_peak_rel = max(0, min(idx_peak_rel, len(a_envelope) - 1))

    attack_sec = idx_peak_rel * (hop_length / sr)
    feats["log_attack_time"] = np.log10(attack_sec + 1e-5)

    times = np.arange(len(a_envelope))
    total_env = np.sum(a_envelope) + 1e-9
    temp_cent = np.sum(times * a_envelope) / total_env
    feats["temporal_centroid"] = temp_cent / (len(a_envelope) + 1e-5)

    if idx_peak_rel > 1:
        t_att = np.arange(idx_peak_rel)
        env_att = a_envelope[:idx_peak_rel]
        A = np.vstack([t_att, np.ones(len(t_att))]).T
        slope_att, _ = np.linalg.lstsq(A, env_att, rcond=None)[0]
        feats["slope_attack"] = slope_att
    else:
        feats["slope_attack"] = 0.0

    dec_len = len(a_envelope) - idx_peak_rel
    if dec_len > 1:
        t_dec = np.arange(dec_len)
        env_dec = a_envelope[idx_peak_rel:]
        env_dec_log = np.log(env_dec + 1e-9)
        A = np.vstack([t_dec, np.ones(dec_len)]).T
        slope_dec, _ = np.linalg.lstsq(A, env_dec_log, rcond=None)[0]
        feats["slope_decay"] = -slope_dec
    else:
        feats["slope_decay"] = 0.0

    env_attack = a_envelope[:idx_peak_rel]
    env_decay = a_envelope[idx_peak_rel:]

    feats.update(compute_vector_stats(env_attack, "att_env"))

    feats.update(compute_vector_stats(env_decay, "dec_env"))

    if len(env_attack) > 1:
        feats.update(compute_vector_stats(np.diff(env_attack), "att_env_diff"))
    else:
        feats.update(compute_vector_stats([], "att_env_diff"))

    if len(env_decay) > 1:
        feats.update(compute_vector_stats(np.diff(env_decay), "dec_env_diff"))
    else:
        feats.update(compute_vector_stats([], "dec_env_diff"))

    harm_mags_peak = ah[:, idx_peak_rel]
    base_mag = harm_mags_peak[0] + 1e-9

    for h in range(1, len(harm_mags_peak)):
        feats[f"harm_ratio_{h + 1}_1"] = harm_mags_peak[h] / base_mag

    chi_a_rel = []
    for h in range(
        1, min(len(harm_mags_peak), 10)
    ):
        chi_a_rel.append(harm_mags_peak[h] / base_mag)

    feats.update(compute_vector_stats(chi_a_rel, "harm_rel"))

    h_idx = np.arange(len(harm_mags_peak))
    if len(h_idx) > 1:
        A = np.vstack([h_idx, np.ones(len(h_idx))]).T
        gamma1, _ = np.linalg.lstsq(A, harm_mags_peak, rcond=None)[0]
        feats["spectral_decay"] = gamma1
    else:
        feats["spectral_decay"] = 0.0

    segments = [("att", onset, peak), ("dec", peak, offset)]

    for part, start, end in segments:
        if end <= start:

            for k in ["centroid", "spread", "rolloff", "crest", "slope"]:
                feats[f"{part}_{k}_mean"] = 0
                feats[f"{part}_{k}_var"] = 0
            continue

        part_spec = M[:, start:end]
        vals = {k: []
                for k in ["centroid", "spread", "rolloff", "crest", "slope"]}
        crest_series = []

        if end > start:
            part_spec = M[:, start:end]
            for n in range(part_spec.shape[1]):
                sf = compute_spectral_features(part_spec[:, n], freqs_lin)
                crest_series.append(sf["crest"])
                for k, v in sf.items():
                    vals[k].append(v)

        crest_diff = first_derivative(crest_series)
        feats.update(compute_vector_stats(crest_diff, f"{part}_crest_diff"))

        for k, arr in vals.items():
            stats = compute_vector_stats(arr, f"{part}_{k}")
            feats.update(stats)

    idx_end_rel = offset - onset
    slices = [("att", 0, idx_peak_rel), ("dec", idx_peak_rel, idx_end_rel)]

    for part, s_start, s_end in slices:
        if s_end <= s_start:
            continue

        ah_part = ah[:, s_start:s_end]

        t1s, t2s, t3s, irrs = [], [], [], []
        for n in range(ah_part.shape[1]):
            h_vec = ah_part[:, n]
            t1, t2, t3 = compute_tristimulus(h_vec)
            irrs.append(compute_spectral_irregularity(h_vec))
            t1s.append(t1)
            t2s.append(t2)
            t3s.append(t3)

        feats.update(compute_vector_stats(t1s, f"{part}_trist1"))
        feats.update(compute_vector_stats(t2s, f"{part}_trist2"))
        feats.update(compute_vector_stats(t3s, f"{part}_trist3"))
        feats.update(compute_vector_stats(irrs, f"{part}_irreg"))

    att_start = onset
    att_end = peak
    att_start = max(0, min(att_start, M.shape[1]))
    att_end = max(att_start, min(att_end, M.shape[1]))

    if att_end <= att_start:
        feats["noisiness"] = 0.0
    else:
        ratios = []
        for n_abs in range(att_start, att_end):

            f0_n = f0
            if f0_contour is not None:
                rel = n_abs - onset
                if 0 <= rel < len(f0_contour) and f0_contour[rel] > 1e-6:
                    f0_n = float(f0_contour[rel])

            T_harm = create_linear_harmonic_template(
                freqs_lin, f0_n, beta, n_harmonics=10, width_bins=5
            )

            M_frame = M[:, n_abs]
            MR_frame = remove_harmonic_peaks(
                M_frame, T_harm, threshold_ratio=0.2)

            denom = np.sum(M_frame) + 1e-9
            ratios.append(float(np.sum(MR_frame) / denom))

        feats["noisiness"] = float(np.mean(ratios)) if len(ratios) else 0.0

    note_len = max(1, offset - onset)
    decay_start = peak
    decay_end = offset

    decay_start = max(0, min(decay_start, M.shape[1] - 1))
    decay_end = max(decay_start + 1, min(decay_end, M.shape[1]))

    frames_for_templates = list(range(decay_start, decay_end))
    if len(frames_for_templates) == 0:
        frames_for_templates = [max(0, min(peak, M.shape[1] - 1))]

    for m in range(2, 8):
        T_sub = create_linear_subharmonic_template(
            freqs_lin, f0, beta, m, n_harmonics=10, width_bins=5
        )

        vals = []
        for n_abs in frames_for_templates:
            M_frame = M[:, n_abs]
            vals.append(template_energy_ratio(M_frame, T_sub))

        feats[f"subharm_{m}"] = float(np.mean(vals)) if len(vals) else 0.0

    open_strings = [41.2, 55.0, 73.4, 98.0]
    for i, f_str in enumerate(open_strings):
        T_str = create_linear_harmonic_template(
            freqs_lin, f_str, beta, n_harmonics=10, width_bins=5
        )

        vals = []
        for n_abs in frames_for_templates:
            M_frame = M[:, n_abs]
            vals.append(template_energy_ratio(M_frame, T_str))

        feats[f"string_likelihood_{
            i + 1}"] = float(np.mean(vals)) if len(vals) else 0.0

    freq_devs = []

    for h in range(1, len(harm_mags_peak)):
        idx = h + 1
        fh_expected = idx * f0 * np.sqrt(1 + beta * (idx**2))

        k_exp = int(np.searchsorted(freqs_lin, fh_expected))
        if k_exp <= 0:
            k_exp = 0
        elif k_exp >= len(freqs_lin):
            k_exp = len(freqs_lin) - 1
        else:

            if abs(freqs_lin[k_exp] - fh_expected) > abs(
                freqs_lin[k_exp - 1] - fh_expected
            ):
                k_exp = k_exp - 1

        k_min = max(0, k_exp - 3)
        k_max = min(len(freqs_lin), k_exp + 4)

        local_seg = M[k_min:k_max, peak] if k_max > k_min else None
        if local_seg is not None and len(local_seg) > 0:
            k_peak = k_min + int(np.argmax(local_seg))
            fh_meas = float(freqs_lin[k_peak])
            dev = (fh_meas - fh_expected) / (fh_expected + 1e-9)
            freq_devs.append(float(dev))
        else:
            freq_devs.append(0.0)

    feats.update(compute_vector_stats(freq_devs, "freq_dev"))

    if f0_contour is not None and len(f0_contour) > 10:

        f0_cent = f0_contour - np.mean(f0_contour)

        signs = np.sign(f0_cent)

        for i in range(1, len(signs)):
            if signs[i] == 0:
                signs[i] = signs[i - 1]
        zero_crossings = np.sum(signs[1:] * signs[:-1] < 0)
        feats["mod_quarter_periods"] = int(2 * zero_crossings)

        autocorr = np.correlate(f0_cent, f0_cent, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]

        peaks, _ = np.array([]), None

        if len(autocorr) > 2:

            peak_indices = [
                i
                for i in range(1, len(autocorr) - 1)
                if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]
            ]

            if peak_indices:
                lag = peak_indices[0]
                mod_freq = sr / (hop_length * lag)
                feats["mod_frequency"] = mod_freq
            else:
                feats["mod_frequency"] = 0
        else:
            feats["mod_frequency"] = 0

        df = np.diff(f0_contour)
        if len(df) == 0:
            feats["mod_quarter_periods"] = 0
        else:
            eps = max(1e-6, 0.01 * float(np.median(np.abs(df))))
            signs = np.sign(df)
            signs[np.abs(df) < eps] = 0
            nz = signs[signs != 0]
            if len(nz) == 0:
                feats["mod_quarter_periods"] = 0
            else:
                feats["mod_quarter_periods"] = int(
                    1 + np.sum(nz[1:] != nz[:-1]))

        f0_max, f0_min = np.max(f0_contour), np.min(f0_contour)
        feats["mod_lift"] = 1200 * np.log2((f0_max + 1e-9) / (f0_min + 1e-9))

        n_fr = len(f0_contour)
        idx_30 = int(n_fr * 0.3)
        if idx_30 > 0:
            avg_start = np.mean(f0_contour[:idx_30])
            avg_end = np.mean(f0_contour[-idx_30:])
            feats["pitch_progression"] = 1200 * np.log2(
                (avg_end + 1e-9) / (avg_start + 1e-9)
            )
        else:
            feats["pitch_progression"] = 0

    else:
        feats["mod_frequency"] = 0
        feats["mod_quarter_periods"] = 0
        feats["mod_lift"] = 0
        feats["pitch_progression"] = 0
        feats["mod_quarter_periods"] = 0

    feats["inharmonicity"] = beta

    return feats


class BassClassifier:

    def __init__(self, model_dir="models", n_features=220):
        from sklearn.feature_selection import SelectKBest, f_classif

        self.model_dir = model_dir
        self.n_features = n_features

        clf_args = {
            "kernel": "rbf",
            "probability": True,
            "class_weight": "balanced",
            "C": 10,
            "gamma": "scale",
        }

        k_val = min(n_features, 60)

        self.clf_style = make_pipeline(
            StandardScaler(), SelectKBest(f_classif, k=k_val), SVC(**clf_args)
        )
        self.clf_expr = make_pipeline(
            StandardScaler(), SelectKBest(f_classif, k=k_val), SVC(**clf_args)
        )
        self.clf_string = make_pipeline(
            StandardScaler(), SelectKBest(f_classif, k=k_val), SVC(**clf_args)
        )
        self.is_trained = False
        self.feature_keys = None

    def get_feature_vector(self, d):

        if self.feature_keys is None:
            self.feature_keys = sorted(d.keys())
        return [d.get(k, 0) for k in self.feature_keys]

    def train(self, X, y_style, y_expr, y_string):
        print(f"Training models on {len(X)} samples...")

        if len(X) > 0 and isinstance(X[0], dict):

            self.feature_keys = sorted(X[0].keys())
            print(f"  Using {len(self.feature_keys)} features")
            X_vec = [self.get_feature_vector(d) for d in X]
        else:
            X_vec = X

        X_arr = np.array(X_vec)
        nan_count = np.isnan(X_arr).sum()
        inf_count = np.isinf(X_arr).sum()
        if nan_count > 0 or inf_count > 0:
            print(
                f"  Warning: Found {nan_count} NaN and {
                    inf_count
                } Inf values, replacing with 0"
            )
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        X_vec = X_arr.tolist()

        if len(set(y_style)) > 1:
            self.clf_style.fit(X_vec, y_style)
        else:
            self._default_style = y_style[0] if y_style else "Finger"

        if len(set(y_expr)) > 1:
            self.clf_expr.fit(X_vec, y_expr)
        else:
            self._default_expr = y_expr[0] if y_expr else "Normal"

        if len(set(y_string)) > 1:
            self.clf_string.fit(X_vec, y_string)
        else:
            self._default_string = y_string[0] if y_string else 0

        self.is_trained = True
        print("Training complete.")

    def save_models(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        joblib.dump(self.clf_style, os.path.join(self.model_dir, "style.pkl"))
        joblib.dump(self.clf_expr, os.path.join(self.model_dir, "expr.pkl"))
        joblib.dump(self.clf_string, os.path.join(
            self.model_dir, "string.pkl"))
        joblib.dump(self.feature_keys, os.path.join(
            self.model_dir, "feature_keys.pkl"))
        print(f"Models saved to {self.model_dir}/")

    def load_models(self):
        try:
            self.clf_style = joblib.load(
                os.path.join(self.model_dir, "style.pkl"))
            self.clf_expr = joblib.load(
                os.path.join(self.model_dir, "expr.pkl"))
            self.clf_string = joblib.load(
                os.path.join(self.model_dir, "string.pkl"))
            self.feature_keys = joblib.load(
                os.path.join(self.model_dir, "feature_keys.pkl")
            )
            self.is_trained = True
            print(f"Models loaded successfully ({
                  len(self.feature_keys)} features).")
            return True
        except FileNotFoundError:
            print("No saved models found.")
            return False

    def predict(self, feature_dict):
        if not self.is_trained:
            return "Finger", "Normal", 0

        X = np.array([self.get_feature_vector(feature_dict)])

        nz = np.count_nonzero(X)
        total = X.shape[1]
        print(
            f"[DEBUG] Non-zero features: {nz}/{total} | min={
                float(X.min()):.4g} max={float(X.max()):.4g}"
        )
        if nz < max(5, total * 0.05):
            print(
                "[WARNING] Feature vector is almost empty or constant. Possible feature mismatch or NaNs."
            )

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        s = self.clf_style.predict(X)[0]
        e = self.clf_expr.predict(X)[0]
        st = self.clf_string.predict(X)[0]
        return s, e, st


def calculate_fret(midi_pitch, string_idx, expression):

    OPEN_STRINGS = [28, 33, 38, 43]

    if expression == "Dead-note":
        return -1

    if 0 <= string_idx < len(OPEN_STRINGS):
        fret = midi_pitch - OPEN_STRINGS[string_idx]
        return max(0, fret)
    return 0


if __name__ == "__main__":
    filename = "example.wav"

    try:
        print("--- Loading Audio ---")
        y, sr = sf.read(filename)

        if y.ndim > 1:
            print(f"Stereo detected (channels: {
                  y.shape[1]}). Converting to Mono.")
            y = np.mean(y, axis=1)

        y_down, sr_down = downsample(y, sr)

        hop_len = 32
        MIF, IF_Hz, M, f_log, freqs_lin = compute_reassigned_spectrogram(
            y_down,
            sr_down,
            win_length=512,
            n_fft=4096,
            hop_length=hop_len,
            plot_spectogram=True,
        )

        hop_sec = hop_len / sr_down
        alpha_On, onset_indices, onset_times, MIF_K = compute_onset_detection_function(
            MIF, hop_length_seconds=hop_sec, plot_onset=True
        )

        print(f"\n--- Detected {len(onset_indices)} Events. Processing... ---")

        print(
            f"{'Idx':<4} | {'Onset(s)':<8} | {'Pitch':<6} | {'Freq(Hz)':<9} | {
                'Dur(fr)':<7} | {'Int(dB)':<7}"
        )
        print("-" * 60)

        results = []

        for i, onset_frame in enumerate(onset_indices):

            if i < len(onset_indices) - 1:
                next_onset = onset_indices[i + 1]
            else:
                next_onset = MIF.shape[1]

            show_plots = i == 0

            f0, beta, midi, full_frames, full_f0s, full_Cmax, max_Cmax = estimate_f0(
                M, freqs_lin, MIF, onset_frame, next_onset, debug_plot=show_plots
            )

            offset_frame, f0_contour = detect_offset(
                full_frames,
                full_f0s,
                full_Cmax,
                max_Cmax,
                onset_frame,
                next_onset,
                debug_plot=show_plots,
            )

            if offset_frame > next_onset:
                offset_frame = next_onset

            n_peak, intensity = extract_intensity_segmentation(
                M,
                freqs_lin,
                onset_frame,
                offset_frame,
                f0_contour,
                beta,
                debug_plot=show_plots,
            )

            note_data = {
                "onset_time": onset_frame * hop_sec,
                "midi": midi,
                "f0": f0,
                "beta": beta,
                "duration_frames": offset_frame - onset_frame,
                "intensity_db": intensity,
                "peak_frame": n_peak,
            }
            results.append(note_data)

            print(
                f"{i:<4} | {note_data['onset_time']:<8.3f} | {midi:<6} | {f0:<9.2f} | {
                    note_data['duration_frames']:<7} | {intensity:<7.1f}"
            )

        print("\n--- Processing Complete ---")

    except FileNotFoundError:
        print(f"Error: File '{
              filename}' not found. Please provide a valid wav file.")
    except Exception as e:
        print(f"An error occurred: {e}")
