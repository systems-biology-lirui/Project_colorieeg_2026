"""Exploratory analyses (clearly labeled, not part of the main claims).

Contents:
  A. Multi-electrode MVPA: ROI channels x bands as features, per time point,
     for Task 3 (red vs green) and Task 2 (memory color, cross-fruit).
  B. ERP-amplitude time-resolved decoding: single feature = +/-25 ms mean
     amplitude, per electrode, group-level.
  C. Exemplar repeat check: unique stimulus images vs trial counts per
     condition (sample-leakage risk assessment).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from analysis.common import (
    GRAY_FRUITS,
    STIMULI_ROOT,
    SUBJECTS,
    AnalysisVariant,
    baseline_subtract,
    common_channels,
    load_conditions,
    read_localization,
)
from analysis.decoding_timeresolved import (
    eval_frames,
    group_cluster_permutation,
    stft_band_power,
    window_mask,
)
from analysis.selection import prepare_signal

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _cv_curve(x0: np.ndarray, x1: np.ndarray, seed: int, n_folds: int = 5) -> np.ndarray:
    """5-fold CV accuracy per time point on flattened pattern features."""
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)])
    n_times = x.shape[-1]
    splitter = list(
        StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(x, y)
    )
    scores = np.zeros(n_times)
    for t in range(n_times):
        folds = []
        for train_idx, test_idx in splitter:
            model = make_pipeline(
                StandardScaler(), LinearSVC(C=1.0, max_iter=10000, dual=True)
            )
            model.fit(x[train_idx, :, t], y[train_idx])
            folds.append(accuracy_score(y[test_idx], model.predict(x[test_idx, :, t])))
        scores[t] = float(np.mean(folds))
    return scores


def _cross_fruit_pattern_curve(
    feats: dict[str, np.ndarray], frame_idx: np.ndarray, seed: int
) -> np.ndarray:
    """Cross-fruit MVPA: train strawberry+cabbage, test watermelon+kiwi."""
    names = ["strawberry", "watermelon", "cabbage", "kiwi"]
    rng = np.random.default_rng(seed)
    arrays = []
    n_min = min(feats[n].shape[0] for n in names)
    for n in names:
        arr = feats[n][:, :, frame_idx]
        idx = rng.permutation(arr.shape[0])[:n_min]
        arrays.append(arr[idx])
    folds = [(0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2)]
    n_times = len(frame_idx)
    scores = np.zeros(n_times)
    for t in range(n_times):
        fold_scores = []
        for tr_red, tr_green, te_red, te_green in folds:
            x_train = np.concatenate(
                [arrays[tr_red][:, :, t], arrays[tr_green][:, :, t]],
                axis=0,
            )
            y_train = np.concatenate(
                [np.zeros(len(arrays[tr_red]), dtype=int),
                 np.ones(len(arrays[tr_green]), dtype=int)]
            )
            x_test = np.concatenate(
                [arrays[te_red][:, :, t], arrays[te_green][:, :, t]],
                axis=0,
            )
            y_test = np.concatenate(
                [np.zeros(len(arrays[te_red]), dtype=int),
                 np.ones(len(arrays[te_green]), dtype=int)]
            )
            model = make_pipeline(
                StandardScaler(), LinearSVC(C=1.0, max_iter=10000, dual=True)
            )
            model.fit(x_train, y_train)
            fold_scores.append(accuracy_score(y_test, model.predict(x_test)))
        scores[t] = float(np.mean(fold_scores))
    return scores


def run_exploration_mvpa(
    out: Path,
    variant: AnalysisVariant,
    n_perms_group: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    stage = out / "stage06_exploration"
    stage.mkdir(parents=True, exist_ok=True)
    rows = []
    for analysis, task, cond0, cond1, cross in (
        ("task3_red_green", 3, "red", "green", False),
        ("task2_memory_color", 2, None, None, True),
    ):
        subject_curves = []
        for subject in SUBJECTS:
            loc = read_localization(subject).set_index("channel")
            channels = [
                c
                for c in common_channels(subject)
                if c in loc.index and bool(loc.loc[c, "is_target_roi"])
            ]
            if not channels:
                continue
            if not cross:
                raw, time_ms, _ = load_conditions(subject, task, [cond0, cond1], channels)
                raw = {
                    key: prepare_signal(values, time_ms, variant)
                    for key, values in raw.items()
                }
                f0, ft = stft_band_power(raw[cond0])
                f1, _ = stft_band_power(raw[cond1])
                grid, frame_idx = eval_frames(ft)
                curve = _cv_curve(
                    f0.reshape(f0.shape[0], -1, f0.shape[-1])[:, :, frame_idx],
                    f1.reshape(f1.shape[0], -1, f1.shape[-1])[:, :, frame_idx],
                    seed + int(subject[-3:]),
                )
            else:
                raw, time_ms, _ = load_conditions(
                    subject, 2, [f"{f}_gray" for f in GRAY_FRUITS], channels
                )
                raw = {
                    key: prepare_signal(values, time_ms, variant)
                    for key, values in raw.items()
                }
                feats = {}
                ft = None
                for f in GRAY_FRUITS:
                    feat, ft = stft_band_power(raw[f"{f}_gray"])
                    feats[f] = feat.reshape(feat.shape[0], -1, feat.shape[-1])
                grid, frame_idx = eval_frames(ft)
                curve = _cross_fruit_pattern_curve(
                    feats, frame_idx, seed + int(subject[-3:])
                )
            subject_curves.append(curve)
        curves = np.asarray(subject_curves)
        summary, clusters = group_cluster_permutation(
            curves, grid, n_perms=n_perms_group, seed=seed
        )
        for cl in clusters:
            rows.append(
                {
                    "analysis": analysis,
                    "window": variant.window_label,
                    "signal": variant.signal,
                    "exploratory": True,
                    "n_subjects": len(curves),
                    **cl,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(
        stage / f"exploration_mvpa_{variant.suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table


def run_exploration_erp_amplitude(
    out: Path,
    variant: AnalysisVariant,
    n_perms_group: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    stage = out / "stage06_exploration"
    stage.mkdir(parents=True, exist_ok=True)
    rows = []
    for analysis, task, cond0, cond1 in (
        ("task3_red_green", 3, "red", "green"),
        ("task2_memory_color", 2, "strawberry_gray", "cabbage_gray"),
    ):
        subject_curves = []
        for subject in SUBJECTS:
            loc = read_localization(subject).set_index("channel")
            channels = [
                c
                for c in common_channels(subject)
                if c in loc.index and bool(loc.loc[c, "is_target_roi"])
            ]
            if not channels:
                continue
            raw, time_ms, _ = load_conditions(subject, task, [cond0, cond1], channels)
            raw = {
                key: prepare_signal(values, time_ms, variant)
                for key, values in raw.items()
            }
            grid = np.arange(0.0, 500.0 + 1e-9, 10.0)
            curves = np.zeros((len(channels), len(grid)))
            for j in range(len(channels)):
                x0 = baseline_subtract(raw[cond0][:, j, :], time_ms)
                x1 = baseline_subtract(raw[cond1][:, j, :], time_ms)
                feats0 = np.stack(
                    [
                        np.nanmean(
                            x0[:, window_mask(time_ms, t - 25, t + 25)], axis=-1
                        )
                        for t in grid
                    ],
                    axis=-1,
                )
                feats1 = np.stack(
                    [
                        np.nanmean(
                            x1[:, window_mask(time_ms, t - 25, t + 25)], axis=-1
                        )
                        for t in grid
                    ],
                    axis=-1,
                )
                curves[j] = _cv_curve(feats0[:, None, :], feats1[:, None, :], seed + j)
            subject_curves.append(curves.mean(axis=0))
        curves = np.asarray(subject_curves)
        summary, clusters = group_cluster_permutation(
            curves, grid, n_perms=n_perms_group, seed=seed
        )
        for cl in clusters:
            rows.append(
                {
                    "analysis": analysis,
                    "window": variant.window_label,
                    "signal": variant.signal,
                    "exploratory": True,
                    "n_subjects": len(curves),
                    **cl,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(
        stage / f"exploration_erp_amplitude_{variant.suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table


def run_exemplar_repeat_check(out: Path) -> pd.DataFrame:
    stage = out / "stage06_exploration"
    stage.mkdir(parents=True, exist_ok=True)
    rows = []
    trial_counts = {
        ("task1", "face"): 70,
        ("task1", "object"): 70,
        ("task1", "body"): 70,
        ("task1", "place"): 70,
        ("task2", "cabbage"): 60,
        ("task2", "kiwi"): 60,
        ("task2", "strawberry"): 60,
        ("task2", "watermelon"): 60,
        ("task3", "red"): 60,
        ("task3", "green"): 60,
        ("task3", "blue"): 60,
        ("task3", "yellow"): 60,
        ("task3", "black"): 60,
        ("task3", "white"): 60,
    }
    patterns = {
        "task1": ("Stimuli_Task1", ["face_color", "face_gray", "object_color", "object_gray", "body_color", "body_gray", "place_color", "place_gray"], "{}_*"),
        "task2": ("Stimuli_Task2", ["Cabbage_Gray", "Kiwi_Gray", "Strawberry_Gray", "Watermelon_Gray"], "{}_*"),
        "task3": ("Stimuli_Task3", ["Red_Color", "Green_Color", "Blue_Color", "Yellow_Color", "Black_Color", "White_Color"], "{}_*"),
    }
    for task, (folder, conds, pat) in patterns.items():
        for cond in conds:
            n_files = len(list((STIMULI_ROOT / folder).glob(pat.format(cond) + ".bmp")))
            key_map = {
                "face_color": ("task1", "face"),
                "face_gray": ("task1", "face"),
                "object_color": ("task1", "object"),
                "object_gray": ("task1", "object"),
                "body_color": ("task1", "body"),
                "body_gray": ("task1", "body"),
                "place_color": ("task1", "place"),
                "place_gray": ("task1", "place"),
                "Cabbage_Gray": ("task2", "cabbage"),
                "Kiwi_Gray": ("task2", "kiwi"),
                "Strawberry_Gray": ("task2", "strawberry"),
                "Watermelon_Gray": ("task2", "watermelon"),
                "Red_Color": ("task3", "red"),
                "Green_Color": ("task3", "green"),
                "Blue_Color": ("task3", "blue"),
                "Yellow_Color": ("task3", "yellow"),
                "Black_Color": ("task3", "black"),
                "White_Color": ("task3", "white"),
            }
            k = key_map.get(cond)
            n_trials = trial_counts.get(k, np.nan) if k else np.nan
            rows.append(
                {
                    "task": task,
                    "condition": cond,
                    "unique_images": n_files,
                    "trials_per_subject": n_trials,
                    "repeat_per_exemplar": (
                        round(n_trials / n_files, 2) if n_files else np.nan
                    ),
                    "exemplar_leakage_risk": bool(n_files and n_files < n_trials),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(
        stage / "exploration_exemplar_repeat_check.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table
