import main_combined as bass_algo
import time
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    make_scorer,
    confusion_matrix,
)
import joblib
import xml.etree.ElementTree as ET
import soundfile as sf
import numpy as np
import zipfile
import requests
import argparse
import glob
import os
import warnings

warnings.filterwarnings("ignore")


URL_TRAIN = "https://zenodo.org/records/7188892/files/IDMT-SMT-BASS.zip?download=1"
DIR_TRAIN = "./data_train_isolated"


URL_TEST = "https://zenodo.org/records/7544099/files/IDMT-SMT-BASS-SINGLE-TRACKS.zip?download=1"
DIR_TEST = "./data_test_tracks"

MODELS_DIR = "./models"
FEATURES_FILE = "./features.pkl"


def print_cm(y_true, y_pred, labels=None, title="Confusion Matrix"):

    if labels is None:
        from sklearn.utils.multiclass import unique_labels

        labels = unique_labels(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    print(f"\n>>> {title} (Normalized)")

    header = f"{'True\\Pred':<12} | " + \
        " | ".join(f"{str(l):<10}" for l in labels)
    sep_line = "-" * len(header)

    print(sep_line)
    print(header)
    print(sep_line)

    for i, row_label in enumerate(labels):
        row_str = " | ".join(f"{val:.2f}".center(10) for val in cm[i])
        print(f"{str(row_label):<12} | {row_str}")
    print(sep_line)


def download_and_extract(url, target_dir, zip_name):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    zip_path = os.path.join(target_dir, zip_name)

    if not os.path.exists(zip_path):
        print(f"Downloading {zip_name}...")
        try:
            r = requests.get(url, stream=True)
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = (downloaded / total) * 100
                        print(
                            f"\r  {
                                downloaded / (1024 * 1024):.1f} MB / {total / (1024 * 1024):.1f} MB ({pct:.0f}%)",
                            end="",
                            flush=True,
                        )
            print("\nDownload gata.")
        except Exception as e:
            print(f"Eroare download: {e}")
            return False

    extracted_check = os.path.join(target_dir, "extracted_marker")
    if not os.path.exists(extracted_check):
        print(f"Dezarhivare {zip_name}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(target_dir)
            open(extracted_check, "w").close()
            print("Dezarhivare gata.")
        except zipfile.BadZipFile:
            print("Eroare: Zip corupt.")
            return False
    return True


def parse_xml_gt(xml_path):

    if not os.path.exists(xml_path):
        return []
    tree = ET.parse(xml_path)
    events = []
    for e in tree.findall(".//event"):
        events.append(
            {
                "onset": float(e.find("onsetSec").text),
                "offset": float(e.find("offsetSec").text),
                "pitch": int(e.find("pitch").text),
                "string": int(e.find("string").text),
                "style": e.find("playingStyle").text
                if e.find("playingStyle") is not None
                else "Finger",
            }
        )
    return events


def parse_csv_gt(csv_path):

    if not os.path.exists(csv_path):
        return []
    events = []

    style_map = {
        "FS": "Finger",
        "PK": "Pick",
        "MU": "Muted",
        "SP": "Slap",
        "ST": "Slap",
        "Finger": "Finger",
        "Pick": "Pick",
        "Muted": "Muted",
        "Slap": "Slap",
    }
    expr_map = {
        "NO": "Normal",
        "VI": "Vibrato",
        "BE": "Bending",
        "SL": "Slide",
        "DN": "Dead-Note",
        "HA": "Harmonics",
        "Normal": "Normal",
        "Vibrato": "Vibrato",
        "Bending": "Bending",
        "Slide": "Slide",
        "Dead-Note": "Dead-Note",
        "Harmonics": "Harmonics",
    }

    with open(csv_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:

                raw_style = parts[5]
                raw_expr = parts[6] if len(parts) > 6 else "NO"

                events.append(
                    {
                        "onset": float(parts[0]),
                        "offset": float(parts[1]),
                        "pitch": int(parts[2]),
                        "string": int(parts[3]),
                        "style": style_map.get(
                            raw_style, "Finger"
                        ),
                        "expression": expr_map.get(
                            raw_expr, "Normal"
                        ),
                    }
                )
            except (ValueError, IndexError):
                continue
    return events


def parse_filename_gt(wav_path):

    fname = os.path.basename(wav_path).replace(".wav", "")
    parts = fname.split("_")

    style_map = {
        "FS": "Finger",
        "SP": "Slap",
        "PK": "Pick",
        "MU": "Muted",
        "ST": "Slap",
    }

    expr_map = {
        "VI": "Vibrato",
        "VIF": "Vibrato",
        "VIS": "Vibrato",
        "BE": "Bending",
        "BEQ": "Bending",
        "BES": "Bending",
        "SL": "Slide",
        "SLD": "Slide",
        "SLU": "Slide",
        "HA": "Harmonics",
        "DN": "Dead-Note",
        "NO": "Normal",
    }

    try:
        if len(parts) == 10:

            style_code = parts[5]
            expr_code = parts[7]
            string_num = int(parts[8])
            pitch_idx = int(parts[9])
            expression = expr_map.get(expr_code, "Normal")
        elif len(parts) == 8:

            style_code = parts[4]
            expr_code = parts[5]
            string_num = int(parts[6])
            pitch_idx = int(parts[7])
            expression = expr_map.get(expr_code, "Normal")
        else:
            return []

        base_midi = {1: 40, 2: 45, 3: 50, 4: 55}.get(string_num, 40)
        midi_pitch = base_midi + pitch_idx

        return [
            {
                "onset": 0.0,
                "offset": 2.0,
                "pitch": midi_pitch,
                "string": string_num,
                "style": style_map.get(style_code, "Finger"),
                "expression": expression,
            }
        ]
    except (IndexError, ValueError):
        return []


def _process_note_inference_task(
    MIF, M, freqs_lin, onset, next_onset, sr_down, hop_sec
):

    try:

        f0, beta, midi, full_frames, full_f0s, full_Cmax, max_Cmax = (
            bass_algo.estimate_f0(
                M, freqs_lin, MIF, onset, next_onset, debug_plot=False
            )
        )

        offset, f0_contour = bass_algo.detect_offset(
            full_frames,
            full_f0s,
            full_Cmax,
            max_Cmax,
            onset,
            next_onset,
            debug_plot=False,
        )
        if offset > next_onset:
            offset = next_onset

        peak_idx, intensity = bass_algo.extract_intensity_segmentation(
            M, freqs_lin, onset, offset, f0_contour, beta
        )

        ah, a_envelope = bass_algo.spectral_envelope_modeling(
            M, freqs_lin, onset, offset, f0_contour, beta
        )

        feats = bass_algo.extract_features(
            M=M,
            freqs_lin=freqs_lin,
            onset=onset,
            peak=peak_idx,
            offset=offset,
            f0=f0,
            ah=ah,
            a_envelope=a_envelope,
            beta=beta,
            sr=sr_down,
            hop_length=32,
            f0_contour=f0_contour,
        )

        return {"onset_time": onset * hop_sec, "midi": midi, "features": feats}
    except Exception as e:
        return None


def _evaluate_track_worker(wav, models_dir):

    import main_combined as bass_algo_worker

    clf = bass_algo_worker.BassClassifier(model_dir=models_dir)
    if not clf.load_models():
        return None

    num = os.path.basename(wav).replace(".wav", "")

    csv_path = os.path.join(DIR_TEST, "misc", "notes_csv", f"{
                            num}_note_parameters.csv")
    gt_events = parse_csv_gt(csv_path)

    if not gt_events:
        return None

    detected = process_audio(wav, ground_truth=None, n_jobs=1)

    return {
        "gt_events": gt_events,
        "detected": detected,
        "predictions": [clf.predict(d["features"]) for d in detected],
        "track": num,
    }


def process_audio(wav_path, ground_truth=None, n_jobs=-1):

    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    y_down, sr_down = bass_algo.downsample(y, sr)

    MIF, _, M, f_log, freqs_lin = bass_algo.compute_reassigned_spectrogram(
        y_down, sr_down, plot_spectogram=False
    )
    hop_sec = 32 / sr_down

    extracted_data = []

    if ground_truth:
        for note in ground_truth:
            onset_idx = int(note["onset"] / hop_sec)
            offset_idx = int(note["offset"] / hop_sec)

            if onset_idx >= MIF.shape[1]:
                continue
            if offset_idx > MIF.shape[1]:
                offset_idx = MIF.shape[1]
            if offset_idx <= onset_idx:
                continue

            f0_gt_val = 440.0 * (2 ** ((note["pitch"] - 69) / 12.0))
            beta_def = 0.0001

            _, _, _, frames_gt, f0s_gt, cmax_gt, max_cmax_gt = bass_algo.estimate_f0(
                M,
                freqs_lin,
                MIF,
                onset_idx,
                offset_idx,
                f_min=f0_gt_val * 0.9,
                f_max=f0_gt_val * 1.1,
                debug_plot=False,
            )

            _, f0_contour = bass_algo.detect_offset(
                frames_gt,
                f0s_gt,
                cmax_gt,
                max_cmax_gt,
                onset_idx,
                offset_idx,
                debug_plot=False,
            )

            peak_idx, intensity = bass_algo.extract_intensity_segmentation(
                M, freqs_lin, onset_idx, offset_idx, f0_contour, beta_def
            )

            ah, a_envelope = bass_algo.spectral_envelope_modeling(
                M, freqs_lin, onset_idx, offset_idx, f0_contour, beta_def
            )

            feats = bass_algo.extract_features(
                M=M,
                freqs_lin=freqs_lin,
                onset=onset_idx,
                peak=peak_idx,
                offset=offset_idx,
                f0=f0_gt_val,
                ah=ah,
                a_envelope=a_envelope,
                beta=beta_def,
                sr=sr_down,
                hop_length=32,
                f0_contour=f0_contour,
            )

            extracted_data.append(
                {
                    "features": feats,
                    "y_style": note["style"],
                    "y_expr": note.get("expression", "Normal"),
                    "y_string": note["string"] - 1,
                }
            )

    else:
        _, onset_inds, _, _ = bass_algo.compute_onset_detection_function(
            MIF, hop_length_seconds=hop_sec, plot_onset=False
        )
        onset_inds = np.atleast_1d(onset_inds)

        note_params = []
        for i, onset in enumerate(onset_inds[:-1]):
            next_onset = onset_inds[i + 1]
            note_params.append((onset, next_onset))

        if n_jobs == 1:
            results = []
            for onset, next_onset in note_params:
                results.append(
                    _process_note_inference_task(
                        MIF, M, freqs_lin, onset, next_onset, sr_down, hop_sec
                    )
                )
        else:
            results = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
                joblib.delayed(_process_note_inference_task)(
                    MIF, M, freqs_lin, onset, next_onset, sr_down, hop_sec
                )
                for onset, next_onset in note_params
            )

        extracted_data = [r for r in results if r is not None]

    return extracted_data


def _process_single_file(f):

    ground_truth = parse_filename_gt(f)
    if not ground_truth:
        return None

    samples = process_audio(f, ground_truth=ground_truth)
    return [
        {
            "features": s["features"],
            "style": s["y_style"],
            "expr": s.get("y_expr", "Normal"),
            "string": s["y_string"],
        }
        for s in samples
    ]


def cmd_preprocess():

    from joblib import Parallel, delayed
    import multiprocessing

    print("\n=== PREPROCESARE (Extragere Features) ===")
    download_and_extract(URL_TRAIN, DIR_TRAIN, "train.zip")

    train_files = []
    train_files.extend(
        glob.glob(os.path.join(DIR_TRAIN, "PS", "**", "*.wav"), recursive=True)
    )
    train_files.extend(
        glob.glob(os.path.join(DIR_TRAIN, "ES", "**", "*.wav"), recursive=True)
    )
    train_files = [f for f in train_files if "Tech" not in f]

    if not train_files:
        print("Nu am găsit fișiere .wav.")
        return

    n_cores = multiprocessing.cpu_count()
    print(f"Preprocesăm {len(train_files)} fișiere pe {n_cores} core-uri...")

    results = Parallel(n_jobs=n_cores, verbose=10)(
        delayed(_process_single_file)(f) for f in train_files
    )

    X, y_s, y_e, y_st = [], [], [], []
    for r in results:
        if r:
            for item in r:
                X.append(item["features"])
                y_s.append(item["style"])
                y_e.append(item["expr"])
                y_st.append(item["string"])

    if len(X) > 0:
        data = {"X": X, "y_style": y_s, "y_expr": y_e, "y_string": y_st}
        joblib.dump(data, FEATURES_FILE)
        print(f"\nFeatures salvate în {FEATURES_FILE} ({len(X)} samples)")
    else:
        print("EROARE: Nu s-au extras features.")


def cmd_train():

    print("\n=== ANTRENARE ===")
    if not os.path.exists(FEATURES_FILE):
        print(f"Nu există {FEATURES_FILE}. Rulează --preprocess întâi.")
        return

    data = joblib.load(FEATURES_FILE)
    print(f"Loaded {len(data['X'])} samples.")

    clf = bass_algo.BassClassifier(model_dir=MODELS_DIR)
    clf.train(
        data["X"],
        data["y_style"],
        data.get("y_expr", ["Normal"] * len(data["X"])),
        data["y_string"],
    )
    clf.save_models()
    print("\nAntrenare completă.")


def cmd_tune():

    print("\n=== OPTIMIZARE HYPERPARAMETRI (GridSearchCV) ===")
    if not os.path.exists(FEATURES_FILE):
        print("Rulează --preprocess întâi!")
        return

    data = joblib.load(FEATURES_FILE)
    X = data["X"]
    y_style = data["y_style"]

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.svm import SVC

    print("Pregătire date...")
    dummy = bass_algo.BassClassifier()
    if len(X) > 0 and isinstance(X[0], dict):
        dummy.feature_keys = sorted(X[0].keys())
        X_vec = [dummy.get_feature_vector(d) for d in X]

        X_arr = np.array(X_vec)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        X_vec = X_arr.tolist()
    else:
        X_vec = X

    pipeline = make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=50),
        SVC(kernel="rbf", class_weight="balanced"),
    )

    param_grid = {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "selectkbest__k": [40, 50, 60],
    }

    print(f"Rulează GridSearch pe {len(X_vec)} sample-uri. Asta poate dura...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_vec, y_style)

    print("\nREZULTATE OPTIMIZARE (Style):")
    print(f"Cel mai bun scor: {grid.best_score_:.3f}")
    print(f"Cei mai buni parametri: {grid.best_params_}")
    print("\nRecomandare: Actualizează parametrii în main_combined.py manual.")


def cmd_acc_isolated():

    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_score,
        cross_val_predict,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, accuracy_score

    print("\n" + "=" * 70)
    print("EVALUATION ON ISOLATED NOTES (5-Fold CV)")
    print("=" * 70)

    if not os.path.exists(FEATURES_FILE):
        print(f"Features not found. Run --preprocess first.")
        return

    data = joblib.load(FEATURES_FILE)
    print(f"Loaded {len(data['X'])} isolated note samples.")

    all_keys = sorted(set().union(*[d.keys() for d in data["X"]]))
    X = np.array([[sample.get(k, 0) for k in all_keys]
                 for sample in data["X"]])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y_style = np.array(data["y_style"])
    y_expr = np.array(data.get("y_expr", ["Normal"] * len(data["X"])))
    y_string = np.array(data["y_string"])

    k_val = min(60, X.shape[1])
    clf = make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=k_val),
        SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[1/3] EVALUATING PLUCKING STYLE...")
    scores_style = cross_val_score(clf, X, y_style, cv=cv, scoring="accuracy")
    print(f"  Accuracy: {scores_style.mean():.3f} ± {scores_style.std():.3f}")
    y_pred_style = cross_val_predict(clf, X, y_style, cv=cv)
    print(classification_report(y_style, y_pred_style, zero_division=0))
    print_cm(y_style, y_pred_style, title="Style Confusion Matrix")

    print("\n[2/3] EVALUATING EXPRESSION...")
    scores_expr = cross_val_score(clf, X, y_expr, cv=cv, scoring="accuracy")
    print(f"  Accuracy: {scores_expr.mean():.3f} ± {scores_expr.std():.3f}")
    y_pred_expr = cross_val_predict(clf, X, y_expr, cv=cv)
    print(classification_report(y_expr, y_pred_expr, zero_division=0))
    print_cm(y_expr, y_pred_expr, title="Expression Confusion Matrix")

    print("\n[3/3] EVALUATING STRING NUMBER...")
    scores_string = cross_val_score(
        clf, X, y_string, cv=cv, scoring="accuracy")
    print(f"  Accuracy: {scores_string.mean():.3f} ± {
          scores_string.std():.3f}")
    y_pred_string = cross_val_predict(clf, X, y_string, cv=cv)
    print(classification_report(y_string, y_pred_string, zero_division=0))
    labels_str = sorted(list(set(y_string) | set(y_pred_string)))
    print_cm(
        y_string, y_pred_string, labels=labels_str, title="String Confusion Matrix"
    )


def cmd_acc():

    print("\n=== EVALUARE ACURATEȚE (Dataset Linii Bas Reale) ===")
    if "URL_TEST" in globals() and "DIR_TEST" in globals():
        download_and_extract(URL_TEST, DIR_TEST, "tracks.zip")

    audio_path = os.path.join(DIR_TEST, "**", "*.wav")
    track_files = glob.glob(audio_path, recursive=True)
    if not track_files:
        print("Nu s-au găsit fișiere de test.")
        return

    print(f"\nEvaluare pe {len(track_files)} fișiere (Mod Paralel)...")

    results = joblib.Parallel(n_jobs=-1, verbose=10)(
        joblib.delayed(_evaluate_track_worker)(f, MODELS_DIR) for f in track_files
    )

    results = [r for r in results if r is not None]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pitch_errors = 0
    y_true_pitch, y_pred_pitch = [], []
    y_true_style, y_pred_style = [], []
    y_true_expr, y_pred_expr = [], []
    y_true_string, y_pred_string = [], []
    eval_results = []

    for res in results:
        gt_events = res["gt_events"]
        detected = res["detected"]
        predictions = res["predictions"]

        matched_gt_indices = set()
        matched_detect_indices = set()
        local_tp = 0
        local_pitch_errors = 0
        local_matches = []

        gt_events.sort(key=lambda x: x["onset"])
        detected.sort(
            key=lambda x: x["onset_time"]
        )

        det_with_pred = list(zip(detected, predictions))
        det_with_pred.sort(key=lambda x: x[0]["onset_time"])

        for g_idx, gt in enumerate(gt_events):
            best_d_idx = -1
            min_dist = 0.150
            for d_idx, (d, pred) in enumerate(det_with_pred):
                if d_idx in matched_detect_indices:
                    continue
                dist = abs(d["onset_time"] - gt["onset"])
                if dist < min_dist and (d["midi"] == gt["pitch"]):
                    min_dist = dist
                    best_d_idx = d_idx

            if best_d_idx != -1:
                matched_gt_indices.add(g_idx)
                matched_detect_indices.add(best_d_idx)
                local_tp += 1
                local_matches.append(
                    (gt, det_with_pred[best_d_idx][0],
                     det_with_pred[best_d_idx][1])
                )

        for g_idx, gt in enumerate(gt_events):
            if g_idx in matched_gt_indices:
                continue

            for d_idx, (d, pred) in enumerate(det_with_pred):
                if d_idx in matched_detect_indices:
                    continue

                dist = abs(d["onset_time"] - gt["onset"])
                if dist < 0.150:
                    local_pitch_errors += 1

                    matched_detect_indices.add(d_idx)
                    break

        real_fn = len(gt_events) - local_tp
        real_fp = len(detected) - local_tp

        local_fp = real_fp - local_pitch_errors

        total_tp += local_tp
        total_fp += real_fp
        total_fn += real_fn
        total_pitch_errors += local_pitch_errors

        track_true_s, track_pred_s = [], []
        track_true_e, track_pred_e = [], []

        for gt, det, (pred_s, pred_e, pred_st) in local_matches:
            y_true_pitch.append(gt["pitch"])
            y_pred_pitch.append(det["midi"])

            y_true_style.append(gt["style"])
            y_pred_style.append(pred_s)
            track_true_s.append(gt["style"])
            track_pred_s.append(pred_s)

            y_true_expr.append(gt.get("expression", "Normal"))
            y_pred_expr.append(pred_e)
            track_true_e.append(gt.get("expression", "Normal"))
            track_pred_e.append(pred_e)

            gst = gt.get("string", 1) - 1
            if gst < 0:
                gst = 0
            y_true_string.append(gst)
            y_pred_string.append(pred_st)

        eval_results.append(
            {
                "track": res["track"],
                "tp": local_tp,
                "fp": local_fp,
                "fn": real_fn,
                "pitch_errors": local_pitch_errors,
                "style_acc": accuracy_score(track_true_s, track_pred_s)
                if track_true_s
                else 0,
            }
        )

    total_detected = total_tp + total_fp
    total_gt = total_tp + total_fn
    recall = total_tp / (total_gt + 1e-9)
    precision = total_tp / (total_detected + 1e-9)
    f_measure = 2 * precision * recall / (precision + recall + 1e-9)

    print("\n" + "=" * 70)
    print("FINAL EVALUATION REPORT")
    print("=" * 70)
    print(f"  Recall (R)    : {recall:.3f}")
    print(f"  Precision (P) : {precision:.3f}")
    print(f"  F-measure (F) : {f_measure:.3f}")

    print("\n  --- Detailed Breakdown ---")
    print(f"  Total Detected Notes    : {total_detected}")
    print(f"    - Correct (TP)        : {total_tp}")
    print(
        f"    - Pitch Errors        : {
            total_pitch_errors} (Correct Onset, Wrong Pitch)"
    )
    print(
        f"    - Spurious (Extra)    : {total_detected -
                          total_tp - total_pitch_errors}"
    )
    print(f"  Total Ground Truth      : {total_gt}")
    print(f"    - Matched (TP)        : {total_tp}")
    print(f"    - Missed (FN)         : {total_gt - total_tp}")

    if len(y_true_style) > 0:

        pitch_acc = total_tp / (total_tp + total_pitch_errors + 1e-9)
        print(
            f"\n  Pitch Accuracy (A)          : {pitch_acc:.3f}  [{
                                  total_tp}/{total_tp + total_pitch_errors}]"
        )
        print(
            f"  Plucking Style Accuracy (A) : {accuracy_score(
                y_true_style, y_pred_style):.3f}"
        )
        print(
            f"  Expression Style Accuracy (A): {accuracy_score(
                y_true_expr, y_pred_expr):.3f}"
        )
        print(
            f"  String Number Accuracy (A)  : {accuracy_score(
                y_true_string, y_pred_string):.3f}"
        )

        print("\n=== PLUCKING STYLE ===")
        print(classification_report(y_true_style, y_pred_style, zero_division=0))
        print_cm(y_true_style, y_pred_style, title="Style Confusion Matrix")

        print("\n=== EXPRESSION STYLE ===")
        print(classification_report(y_true_expr, y_pred_expr, zero_division=0))
        print_cm(y_true_expr, y_pred_expr, title="Expression Confusion Matrix")

        print("\n=== STRING NUMBER ===")
        print(classification_report(y_true_string, y_pred_string, zero_division=0))
        labels_str = sorted(list(set(y_true_string) | set(y_pred_string)))
        print_cm(
            y_true_string,
            y_pred_string,
            labels=labels_str,
            title="String Confusion Matrix",
        )
    else:
        print("No matched notes.")


def print_ascii_tab(tab_notes, max_notes=16):

    strings = ["G", "D", "A", "E"]
    tab_lines = {s: [] for s in strings}

    notes = tab_notes[:max_notes]

    for note in notes:
        string_idx = note["string"]
        fret = note["fret"]

        string_name = ["E", "A", "D", "G"][string_idx]

        for s in strings:
            if s == string_name:
                tab_lines[s].append(str(fret).ljust(2, "-"))
            else:
                tab_lines[s].append("--")

    print("\nTAB (primele note):")
    for s in strings:
        print(f"{s}|{'-'.join(tab_lines[s])}")


def cmd_infer(file_path):

    print(f"\n=== INFERENȚĂ: {os.path.basename(file_path)} ===")

    clf = bass_algo.BassClassifier(model_dir=MODELS_DIR)
    if not clf.load_models():
        print("Warning: Modele neîncărcate. Stilurile vor fi default.")

    detected_notes = process_audio(file_path, ground_truth=None)

    print(
        f"\n{'Time (s)':<8} | {'MIDI':<4} | {'Note':<4} | {
                    'Style':<10} | {'Expr':<8} | {'Str':<3} | {'Fret':<4}"
    )
    print("-" * 75)

    NOTE_NAMES = ["C", "C
    tab_notes = []

    for note in detected_notes:
        style, expr, string_idx = clf.predict(note["features"])
        fret = bass_algo.calculate_fret(note["midi"], string_idx, expr)
        note_name = NOTE_NAMES[note["midi"] % 12] + str(note["midi"] // 12 - 1)

        tab_notes.append(
            {
                "time": note["onset_time"],
                "string": string_idx,
                "fret": fret,
                "style": style,
                "expr": expr,
            }
        )
        print(
            f"{note['onset_time']:<8.3f} | {note['midi']:<4} | {note_name:<4} | {style:<10} | {expr:<8} | {string_idx:<3} | {fret:<4}"
        )
    print_ascii_tab(tab_notes, max_notes=16)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bass Transcription Pipeline")
    parser.add_argument(
        "--preprocess", action="store_true", help="Preprocess audio and save features"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train from cached features"
    )
    parser.add_argument(
        "--acc", action="store_true", help="Evaluate on real bass tracks (ISBST)"
    )
    parser.add_argument(
        "--acc-isolated", action="store_true", help="Cross-validation on isolated notes"
    )
    parser.add_argument("--infer", type=str, help="Run inference on a WAV file")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")

    args = parser.parse_args()

    if args.preprocess:
        cmd_preprocess()
    elif args.train:
        cmd_train()
    elif args.acc:
        cmd_acc()
    elif getattr(args, "acc_isolated", False):
        cmd_acc_isolated()
    elif args.tune:
        cmd_tune()
    elif args.infer:
        if os.path.exists(args.infer):
            cmd_infer(args.infer)
        else:
            print(f"Fișierul nu există: {args.infer}")
    else:
        parser.print_help()
