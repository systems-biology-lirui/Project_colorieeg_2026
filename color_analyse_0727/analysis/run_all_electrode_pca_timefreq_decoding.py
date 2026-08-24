"""All-electrode PCA time-frequency decoding.

This is a new, resumable parameter experiment.  It uses all three-task common
HDF5 electrodes (548 contacts in the current data), fine-grained STFT log
power, trial-wise baseline subtraction, and a training-fold-only
StandardScaler -> PCA(10) -> LinearSVC pipeline.  The four reported branches
are Task3 within-task, Task2 leave-one-fruit-pair-out, and the two cross-task
directions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import stft
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from analysis.common import GRAY_FRUITS, SUBJECTS, common_channels, load_conditions, natural_key


ANALYSES = (
    "task3_within_pca_timefreq",
    "task2_cross_fruit_pca_timefreq",
    "task3_to_task2_pca_timefreq",
    "task2_to_task3_pca_timefreq",
)
RED_FRUITS = ("strawberry", "watermelon")
GREEN_FRUITS = ("cabbage", "kiwi")
FS = 500.0
BASELINE_MS = (-200.0, 0.0)
STFT_WINDOW_MS = 256.0
STFT_STEP_MS = 10.0
STFT_NPERSEG = 128
STFT_HOP = 5
EVAL_MS = (0.0, 800.0)
FREQ_RANGE_HZ = (5.0, 195.0)
NOISE_BANDS = ((45.0, 55.0), (95.0, 105.0), (145.0, 155.0))
N_COMPONENTS = 10
CLUSTER_FORM_P = 0.05
MIN_CLUSTER_MS = 20.0


@dataclass
class PreparedTimeFreq:
    features: np.ndarray  # (trials, frequency bins, evaluation times)
    frame_times_ms: np.ndarray
    eval_times_ms: np.ndarray
    eval_indices: np.ndarray
    frequencies_hz: np.ndarray


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_info(root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"commit": None, "status_porcelain": None}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
        info["status_porcelain"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.STDOUT
        ).splitlines()
    except Exception as exc:  # pragma: no cover - provenance must not stop analysis
        info["error"] = repr(exc)
    return info


def _in_noise_band(freqs: np.ndarray) -> np.ndarray:
    result = np.zeros(freqs.shape, dtype=bool)
    for lo, hi in NOISE_BANDS:
        result |= (freqs >= lo) & (freqs < hi)
    return result


def _prepare_timefreq(epochs: np.ndarray, time_ms: np.ndarray) -> PreparedTimeFreq:
    """Compute fine-bin log power and trial-wise baseline log-power change."""

    epochs = np.asarray(epochs, dtype=np.float64)
    if epochs.ndim != 2:
        raise ValueError(f"Expected (trials,time), got {epochs.shape}")
    freqs, frame_times, coeff = stft(
        epochs,
        fs=FS,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NPERSEG - STFT_HOP,
        axis=-1,
        boundary=None,
        padded=False,
    )
    frame_times_ms = frame_times * 1000.0 + float(time_ms[0])
    frequency_mask = (
        (freqs >= FREQ_RANGE_HZ[0])
        & (freqs <= FREQ_RANGE_HZ[1])
        & ~_in_noise_band(freqs)
    )
    selected_freqs = np.asarray(freqs[frequency_mask], dtype=float)
    if selected_freqs.size < N_COMPONENTS:
        raise RuntimeError(f"Only {selected_freqs.size} frequency bins available for PCA")
    log_power = np.log(np.abs(coeff[:, frequency_mask, :]) ** 2 + 1e-12)
    baseline_mask = (frame_times_ms >= BASELINE_MS[0]) & (frame_times_ms <= BASELINE_MS[1])
    if not baseline_mask.any():
        raise RuntimeError("STFT produced no baseline frames")
    baseline_mean = np.nanmean(log_power[:, :, baseline_mask], axis=-1, keepdims=True)
    features = log_power - baseline_mean
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    eval_times = np.arange(EVAL_MS[0], EVAL_MS[1] + 1e-9, STFT_STEP_MS, dtype=float)
    eval_indices = np.asarray(
        [int(np.argmin(np.abs(frame_times_ms - target))) for target in eval_times], dtype=int
    )
    if len(np.unique(eval_indices)) != len(eval_indices):
        raise RuntimeError("Evaluation time grid maps to duplicate STFT frames")
    return PreparedTimeFreq(
        features=np.asarray(features[:, :, eval_indices], dtype=np.float32),
        frame_times_ms=np.asarray(frame_times_ms, dtype=float),
        eval_times_ms=eval_times,
        eval_indices=eval_indices,
        frequencies_hz=selected_freqs,
    )


def _balance_pair(
    x0: np.ndarray, x1: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x0), len(x1))
    if n < 2:
        raise ValueError(f"Too few trials to balance classes: {len(x0)}, {len(x1)}")
    return x0[rng.permutation(len(x0))[:n]], x1[rng.permutation(len(x1))[:n]]


def _fit_transform(
    x_train: np.ndarray, x_test: np.ndarray, seed: int, use_pca: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(x_train)
    train_scaled = scaler.transform(x_train)
    test_scaled = scaler.transform(x_test)
    if not use_pca:
        return (
            train_scaled,
            test_scaled,
            np.empty(0, dtype=float),
            np.empty((0, train_scaled.shape[1]), dtype=float),
        )
    n_components = min(N_COMPONENTS, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full", whiten=False, random_state=seed)
    return (
        pca.fit_transform(train_scaled),
        pca.transform(test_scaled),
        np.asarray(pca.explained_variance_ratio_, dtype=float),
        np.asarray(pca.components_, dtype=float) ** 2,
    )


def _model_score(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    model = LinearSVC(C=1.0, max_iter=10000, dual=False)
    model.fit(x_train, y_train)
    return float(accuracy_score(y_test, model.predict(x_test)))


def _decode_folded(
    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    n_perms: int,
    n_jobs: int,
    use_pca: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode fixed folds over time; PCA is fitted once per fold/time point."""

    rng = np.random.default_rng(seed)
    n_times = folds[0][0].shape[-1]
    prepared: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = []
    for t in range(n_times):
        at_time = []
        for fold_index, (x_train, x_test, y_train, y_test) in enumerate(folds):
            xtr, xte, ev, loading_sq = _fit_transform(
                x_train[:, :, t], x_test[:, :, t], seed + t * 100 + fold_index, use_pca
            )
            at_time.append((xtr, xte, y_train, y_test, ev, loading_sq))
        prepared.append(at_time)

    def evaluate(label_overrides: list[np.ndarray] | None) -> np.ndarray:
        scores = np.empty(n_times, dtype=float)
        for t, folds_t in enumerate(prepared):
            fold_scores = []
            for fold_index, (xtr, xte, y_train, y_test, _, _) in enumerate(folds_t):
                labels = y_train if label_overrides is None else label_overrides[fold_index]
                fold_scores.append(_model_score(xtr, labels, xte, y_test))
            scores[t] = float(np.mean(fold_scores))
        return scores

    real = evaluate(None)
    def one_permutation(permutation_index: int) -> np.ndarray:
        local_rng = np.random.default_rng(seed + permutation_index * 1000003)
        labels = [local_rng.permutation(folds_t[0][2]) for folds_t in prepared]
        return evaluate(labels)

    null_rows = Parallel(n_jobs=min(max(1, n_jobs), max(1, n_perms)), prefer="processes")(
        delayed(one_permutation)(permutation_index) for permutation_index in range(n_perms)
    )
    null = np.asarray(null_rows, dtype=np.float32).reshape(n_perms, n_times)

    diagnostic_components = N_COMPONENTS if use_pca else 0
    ev_summary = np.full((n_times, diagnostic_components), np.nan, dtype=float)
    loading_summary = np.full((n_times, diagnostic_components, folds[0][0].shape[1]), np.nan, dtype=float)
    for t, folds_t in enumerate(prepared):
        evs = [item[4] for item in folds_t]
        loads = [item[5] for item in folds_t]
        ev_summary[t, : min(N_COMPONENTS, len(evs[0]))] = np.mean(evs, axis=0)
        loading_summary[t, : loads[0].shape[1], :] = np.mean(loads, axis=0)
    return real, null, ev_summary, loading_summary


def _decode_fixed(
    train0: np.ndarray,
    train1: np.ndarray,
    test0: np.ndarray,
    test1: np.ndarray,
    seed: int,
    n_perms: int,
    n_jobs: int,
    use_pca: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train0, train1 = _balance_pair(train0, train1, rng)
    test0, test1 = _balance_pair(test0, test1, rng)
    x_train = np.concatenate([train0, train1], axis=0)
    x_test = np.concatenate([test0, test1], axis=0)
    y_train = np.concatenate([np.zeros(len(train0), dtype=int), np.ones(len(train1), dtype=int)])
    y_test = np.concatenate([np.zeros(len(test0), dtype=int), np.ones(len(test1), dtype=int)])
    n_times = x_train.shape[-1]
    real = np.empty(n_times, dtype=float)
    null = np.empty((n_perms, n_times), dtype=np.float32)
    diagnostic_components = N_COMPONENTS if use_pca else 0
    ev_summary = np.full((n_times, diagnostic_components), np.nan, dtype=float)
    loading_summary = np.full((n_times, diagnostic_components, x_train.shape[1]), np.nan, dtype=float)
    prepared = []
    for t in range(n_times):
        xtr, xte, ev, load = _fit_transform(x_train[:, :, t], x_test[:, :, t], seed + t, use_pca)
        prepared.append((xtr, xte, ev, load))
        real[t] = _model_score(xtr, y_train, xte, y_test)
        ev_summary[t, : len(ev)] = ev
        loading_summary[t, : load.shape[0], :] = load
    def one_permutation(permutation_index: int) -> np.ndarray:
        local_rng = np.random.default_rng(seed + permutation_index * 1000003)
        labels = local_rng.permutation(y_train)
        return np.asarray([_model_score(xtr, labels, xte, y_test) for xtr, xte, _, _ in prepared], dtype=float)

    null_rows = Parallel(n_jobs=min(max(1, n_jobs), max(1, n_perms)), prefer="processes")(
        delayed(one_permutation)(permutation_index) for permutation_index in range(n_perms)
    )
    null[:] = np.asarray(null_rows, dtype=np.float32).reshape(n_perms, n_times)
    return real, null, ev_summary, loading_summary


def _task3_folds(
    red: np.ndarray, green: np.ndarray, red_shapes: np.ndarray, green_shapes: np.ndarray, seed: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    shapes = sorted(set(red_shapes.tolist()) & set(green_shapes.tolist()))
    if len(shapes) != 3:
        raise ValueError(f"Expected three common Task3 shapes, found {shapes}")
    folds = []
    for shape in shapes:
        train_red, test_red = red[red_shapes != shape], red[red_shapes == shape]
        train_green, test_green = green[green_shapes != shape], green[green_shapes == shape]
        train_red, train_green = _balance_pair(train_red, train_green, rng)
        test_red, test_green = _balance_pair(test_red, test_green, rng)
        x_train = np.concatenate([train_red, train_green], axis=0)
        y_train = np.concatenate([np.zeros(len(train_red), dtype=int), np.ones(len(train_green), dtype=int)])
        x_test = np.concatenate([test_red, test_green], axis=0)
        y_test = np.concatenate([np.zeros(len(test_red), dtype=int), np.ones(len(test_green), dtype=int)])
        folds.append((x_train, x_test, y_train, y_test))
    return folds


def _task2_folds(fruits: dict[str, np.ndarray], seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    names = ["strawberry", "watermelon", "cabbage", "kiwi"]
    n = min(fruits[name].shape[0] for name in names)
    arrays = [fruits[name][rng.permutation(fruits[name].shape[0])[:n]] for name in names]
    fold_specs = [(0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2)]
    folds = []
    for train_red, train_green, test_red, test_green in fold_specs:
        x_train = np.concatenate([arrays[train_red], arrays[train_green]], axis=0)
        y_train = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
        x_test = np.concatenate([arrays[test_red], arrays[test_green]], axis=0)
        y_test = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
        folds.append((x_train, x_test, y_train, y_test))
    return folds


def _clusters(real: np.ndarray, null: np.ndarray, times: np.ndarray) -> list[dict[str, float]]:
    if null.size == 0 or not np.isfinite(real).any():
        return []
    threshold = np.nanquantile(null, 1.0 - CLUSTER_FORM_P, axis=0)
    min_len = max(1, int(round(MIN_CLUSTER_MS / STFT_STEP_MS)))

    def masses(values: np.ndarray, mask: np.ndarray) -> list[tuple[int, int, float]]:
        output: list[tuple[int, int, float]] = []
        start: int | None = None
        for index, flag in enumerate(mask):
            if flag and start is None:
                start = index
            if start is not None and (not flag or index == len(mask) - 1):
                end = index if flag and index == len(mask) - 1 else index - 1
                if end - start + 1 >= min_len:
                    output.append((start, end, float(np.sum(values[start : end + 1] - 0.5))))
                start = None
        return output

    observed = masses(real, real > threshold)
    max_null = np.zeros(null.shape[0], dtype=float)
    for i, row in enumerate(null):
        candidate = masses(row, row > threshold)
        if candidate:
            max_null[i] = max(item[2] for item in candidate)
    output = []
    for start, end, mass in observed:
        p = float((1.0 + np.sum(max_null >= mass)) / (len(max_null) + 1.0))
        output.append({"start_ms": float(times[start]), "end_ms": float(times[end]), "mass": mass, "p": p})
    return output


def _shape_labels(index_path: Path, subject: str, counts: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(index_path)
    subset = table[
        (table["subject"] == subject)
        & (table["task"].astype(int) == 3)
        & (table["is_eeg_epoch"].astype(str).str.lower() == "true")
        & (table["color"].astype(str).str.lower().isin(["red", "green"]))
    ].copy()
    output = []
    for color, count in (("red", counts["red"]), ("green", counts["green"])):
        rows = subset[subset["color"].astype(str).str.lower() == color].copy()
        rows["order"] = pd.to_numeric(rows["color_trial_index_1based"], errors="coerce")
        rows = rows.sort_values("order")
        expected = np.arange(1, count + 1)
        if len(rows) != count or not np.array_equal(rows["order"].to_numpy(dtype=int), expected):
            raise RuntimeError(f"Task3 trial-index mismatch for {subject} {color}: rows={len(rows)}, count={count}")
        output.append(pd.to_numeric(rows["shape_id"], errors="raise").to_numpy(dtype=int))
    return output[0], output[1]


def _load_subject(subject: str, index_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, list[str], np.ndarray]:
    channels = common_channels(subject)
    task3, time3, labels3 = load_conditions(subject, 3, ["red", "green"], channels)
    task2, time2, labels2 = load_conditions(subject, 2, [f"{fruit}_gray" for fruit in GRAY_FRUITS], channels)
    if labels3 != labels2 or not np.allclose(time3, time2):
        raise RuntimeError(f"Task2/Task3 channel or time mismatch for {subject}")
    counts = {"red": task3["red"].shape[0], "green": task3["green"].shape[0]}
    red_shapes, green_shapes = _shape_labels(index_path, subject, counts)
    return task3, task2, np.asarray([red_shapes, green_shapes], dtype=object), channels, time3


def _electrode_result(
    subject: str,
    channel: str,
    task3: dict[str, np.ndarray],
    task2: dict[str, np.ndarray],
    shapes: np.ndarray,
    channel_index: int,
    time_ms: np.ndarray,
    seed: int,
    n_perms: int,
    n_jobs: int,
    use_pca: bool,
) -> dict[str, Any]:
    tf3 = {name: _prepare_timefreq(values[:, channel_index, :], time_ms) for name, values in task3.items()}
    tf2 = {name: _prepare_timefreq(values[:, channel_index, :], time_ms) for name, values in task2.items()}
    eval_times = tf3["red"].eval_times_ms
    frame_times = tf3["red"].frame_times_ms
    if not np.allclose(eval_times, tf2["cabbage_gray"].eval_times_ms):
        raise RuntimeError(f"Task2/Task3 evaluation grid mismatch for {subject} {channel}")
    red3, green3 = tf3["red"].features, tf3["green"].features
    t2fruits = {fruit: tf2[f"{fruit}_gray"].features for fruit in GRAY_FRUITS}
    red2 = np.concatenate([t2fruits[fruit] for fruit in RED_FRUITS], axis=0)
    green2 = np.concatenate([t2fruits[fruit] for fruit in GREEN_FRUITS], axis=0)
    results: dict[str, Any] = {
        "eval_times_ms": eval_times,
        "eval_frame_times_ms": tf3["red"].frame_times_ms[tf3["red"].eval_indices],
        "frame_times_ms": frame_times,
        "frequencies_hz": tf3["red"].frequencies_hz,
        "analyses": {},
    }

    folds3 = _task3_folds(red3, green3, shapes[0], shapes[1], seed + 11)
    outputs = {}
    outputs[ANALYSES[0]] = _decode_folded(folds3, seed + 101, n_perms, n_jobs, use_pca)
    outputs[ANALYSES[1]] = _decode_folded(_task2_folds(t2fruits, seed + 12), seed + 102, n_perms, n_jobs, use_pca)
    outputs[ANALYSES[2]] = _decode_fixed(red3, green3, red2, green2, seed + 103, n_perms, n_jobs, use_pca)
    outputs[ANALYSES[3]] = _decode_fixed(red2, green2, red3, green3, seed + 104, n_perms, n_jobs, use_pca)
    for name, (real, null, ev, loading) in outputs.items():
        results["analyses"][name] = {
            "real": real.astype(np.float32),
            "null": null.astype(np.float32),
            "clusters": _clusters(real, null, eval_times),
            "pca_explained_variance_ratio": ev.astype(np.float32),
            "pca_loading_squared": loading.astype(np.float32),
        }
    return results


def _save_electrode(path: Path, result: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    arrays: dict[str, Any] = {
        "eval_times_ms": result["eval_times_ms"],
        "eval_frame_times_ms": result["eval_frame_times_ms"],
        "frame_times_ms": result["frame_times_ms"],
        "frequencies_hz": result["frequencies_hz"],
    }
    for name, data in result["analyses"].items():
        key = _safe_name(name)
        arrays[f"{key}__real"] = data["real"]
        arrays[f"{key}__null"] = data["null"]
        arrays[f"{key}__pca_ev"] = data["pca_explained_variance_ratio"]
        arrays[f"{key}__pca_loading_sq"] = data["pca_loading_squared"]
        arrays[f"{key}__clusters_json"] = np.asarray(json.dumps(data["clusters"], ensure_ascii=False))
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def _load_electrode(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        result: dict[str, Any] = {
            "eval_times_ms": np.asarray(data["eval_times_ms"]),
            "eval_frame_times_ms": np.asarray(data["eval_frame_times_ms"]),
            "frame_times_ms": np.asarray(data["frame_times_ms"]),
            "frequencies_hz": np.asarray(data["frequencies_hz"]),
            "analyses": {},
        }
        for analysis in ANALYSES:
            key = _safe_name(analysis)
            result["analyses"][analysis] = {
                "real": np.asarray(data[f"{key}__real"]),
                "null": np.asarray(data[f"{key}__null"]),
                "pca_explained_variance_ratio": np.asarray(data[f"{key}__pca_ev"]),
                "pca_loading_squared": np.asarray(data[f"{key}__pca_loading_sq"]),
                "clusters": json.loads(str(data[f"{key}__clusters_json"].item())),
            }
    return result


def _plot_electrode(path: Path, subject: str, channel: str, result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    times = result["eval_times_ms"]
    for axis, analysis in zip(axes, ANALYSES):
        data = result["analyses"][analysis]
        axis.plot(times, data["real"], color="#245b9e", linewidth=1.2)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
        for cluster in data["clusters"]:
            if cluster["p"] <= 0.05:
                axis.axvspan(cluster["start_ms"], cluster["end_ms"], color="#e45756", alpha=0.18)
        axis.set_ylim(0.35, 0.95)
        axis.set_ylabel("accuracy")
        axis.set_title(analysis.replace("_", " "), loc="left", fontsize=9)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("nominal time (ms); STFT 256 ms / 10 ms")
    fig.suptitle(f"{subject} {channel} | fine-bin STFT + PCA(10) | exploratory clusters", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _aggregate(output_dir: Path, records: list[tuple[str, str, Path]], make_plots: bool) -> None:
    curve_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    ev_rows: list[dict[str, Any]] = []
    freq_rows: list[dict[str, Any]] = []
    fig_dir = output_dir / "figures" / "individual_electrodes"
    if make_plots:
        fig_dir.mkdir(parents=True, exist_ok=True)
    subject_curves: dict[str, dict[str, list[np.ndarray]]] = {}
    for subject, channel, npz_path in records:
        result = _load_electrode(npz_path)
        times = result["eval_times_ms"]
        subject_curves.setdefault(subject, {analysis: [] for analysis in ANALYSES})
        if make_plots:
            _plot_electrode(fig_dir / f"{_safe_name(subject)}_{_safe_name(channel)}.png", subject, channel, result)
        for analysis in ANALYSES:
            data = result["analyses"][analysis]
            real = np.asarray(data["real"], dtype=float)
            subject_curves[subject][analysis].append(real)
            peak_idx = int(np.nanargmax(real)) if np.isfinite(real).any() else -1
            clusters = data["clusters"]
            best_p = min((float(c["p"]) for c in clusters), default=np.nan)
            best_cluster = min(clusters, key=lambda c: c["p"]) if clusters else None
            summary_rows.append({
                "subject": subject, "channel": channel, "analysis": analysis,
                "mean_accuracy_0_800": float(np.nanmean(real)),
                "peak_accuracy": float(real[peak_idx]) if peak_idx >= 0 else np.nan,
                "peak_time_ms": float(times[peak_idx]) if peak_idx >= 0 else np.nan,
                "n_clusters_p_le_0.05": int(sum(float(c["p"]) <= 0.05 for c in clusters)),
                "min_cluster_p": best_p,
                "best_cluster_start_ms": best_cluster["start_ms"] if best_cluster else np.nan,
                "best_cluster_end_ms": best_cluster["end_ms"] if best_cluster else np.nan,
                "permutation_scope": "100 labels per electrode/branch; time-cluster only",
            })
            for time_index, nominal_time in enumerate(times):
                curve_rows.append({
                    "subject": subject, "channel": channel, "analysis": analysis,
                    "time_ms": float(nominal_time),
                    "stft_frame_ms": float(result["eval_frame_times_ms"][time_index]),
                    "accuracy": float(real[time_index]),
                })
                for pc_index, ev in enumerate(data["pca_explained_variance_ratio"][time_index], start=1):
                    ev_rows.append({"subject": subject, "channel": channel, "analysis": analysis, "time_ms": float(nominal_time), "pc": pc_index, "explained_variance_ratio": float(ev)})
            for pc_index in range(data["pca_loading_squared"].shape[1]):
                for freq_index, value in enumerate(data["pca_loading_squared"][:, pc_index, :].mean(axis=0)):
                    freq_rows.append({"subject": subject, "channel": channel, "analysis": analysis, "pc": pc_index + 1, "frequency_hz": float(result["frequencies_hz"][freq_index]), "mean_squared_loading_0_800": float(value)})
            for cluster_index, cluster in enumerate(clusters, start=1):
                cluster_rows.append({"subject": subject, "channel": channel, "analysis": analysis, "cluster": cluster_index, **cluster})
    pd.DataFrame(summary_rows).to_csv(output_dir / "electrode_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve_rows).to_csv(output_dir / "decoding_time_curves.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cluster_rows).to_csv(output_dir / "time_cluster_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ev_rows).to_csv(output_dir / "pca_explained_variance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(freq_rows).to_csv(output_dir / "pca_frequency_contribution.csv", index=False, encoding="utf-8-sig")
    if make_plots:
        group_dir = output_dir / "figures"
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        for axis, analysis in zip(axes, ANALYSES):
            for subject, by_analysis in subject_curves.items():
                if by_analysis[analysis]:
                    axis.plot(times, np.nanmean(np.stack(by_analysis[analysis]), axis=0), linewidth=1.0, alpha=0.7, label=subject)
            axis.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
            axis.set_title(analysis.replace("_", " "), loc="left", fontsize=9)
            axis.set_ylabel("mean accuracy")
            axis.grid(alpha=0.2)
        axes[0].legend(ncol=4, fontsize=8)
        axes[-1].set_xlabel("nominal time (ms)")
        fig.suptitle("All-electrode subject means | fine-bin STFT + PCA(10)")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(group_dir / "subject_mean_decoding_curves.png", dpi=160)
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perms", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1, help="Reserved for future process-level parallelism; per-electrode files are resumable.")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=list(SUBJECTS))
    parser.add_argument("--limit-electrodes", type=int, default=0, help="Debug/integration limit; 0 means all electrodes.")
    parser.add_argument("--output", type=Path, default=None, help="Existing/new result directory; defaults to a timestamped result directory.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Existing/new provenance directory; defaults to runs/<timestamp>_all_electrode_pca_timefreq_decoding.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--no-pca", action="store_true", help="Disable PCA; retain training-fold StandardScaler before LinearSVC.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = "all_electrode_no_pca_timefreq_decoding" if args.no_pca else "all_electrode_pca_timefreq_decoding"
    output_dir = (args.output or (root / "color_analyse_0727" / "result" / f"{run_label}_{timestamp}")).resolve()
    run_dir = (args.run_dir or (root / "runs" / f"{timestamp}_{run_label}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "electrode_npz"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start = _utc_now()
    config = {
        "analysis": "all-electrode fine-bin PCA time-frequency decoding",
        "subjects": args.subjects,
        "n_electrodes_target": 548,
        "signal": "raw200",
        "stft_window_ms": STFT_WINDOW_MS,
        "stft_step_ms": STFT_STEP_MS,
        "stft_nperseg": STFT_NPERSEG,
        "stft_hop_samples": STFT_HOP,
        "evaluation_ms": list(EVAL_MS),
        "baseline_ms": list(BASELINE_MS),
        "feature": "trial-wise log-power change at fine STFT frequency bins",
        "frequency_range_hz": list(FREQ_RANGE_HZ),
        "excluded_noise_bands_hz": [list(item) for item in NOISE_BANDS],
        "pca": {"enabled": not args.no_pca, "n_components": N_COMPONENTS if not args.no_pca else 0, "whiten": False, "fit_scope": "training fold/time point only" if not args.no_pca else "disabled"},
        "classifier": "LinearSVC(C=1, dual=False, max_iter=10000)",
        "branches": list(ANALYSES),
        "task3_cv": "3-fold leave-one-shape-out using passivecolorpatch trial index",
        "task2_cv": "4-fold leave-one-fruit-pair-out",
        "permutations": args.perms,
        "cluster": {"forming_p": CLUSTER_FORM_P, "min_cluster_ms": MIN_CLUSTER_MS, "correction": "within-electrode time cluster only"},
        "seed": args.seed,
        "workers": args.workers,
        "command": " ".join(sys.argv),
        "started_at_utc": start,
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_info(root),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    index_path = root / "color_analyse_0727" / "metadata" / "passivecolorpatch_shape_trial_index.csv"
    input_files = [root / "color_analyse_0727" / "process_data" / subject / f"task{task}_epoched_1_200Hz.h5" for subject in args.subjects for task in (2, 3)]
    input_files.append(index_path)
    input_identity = []
    for path in input_files:
        stat = path.stat()
        input_identity.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": _sha256(path)})
    (run_dir / "input_identity.json").write_text(json.dumps(input_identity, indent=2, ensure_ascii=False), encoding="utf-8")
    records: list[tuple[str, str, Path]] = []
    processed = 0
    try:
        for subject in args.subjects:
            task3, task2, shapes, channels, time3 = _load_subject(subject, index_path)
            if args.limit_electrodes:
                channels = channels[: args.limit_electrodes]
            print(f"[{subject}] {len(channels)} electrodes", flush=True)
            pending: list[tuple[int, str, Path]] = []
            for channel_index, channel in enumerate(channels):
                checkpoint = checkpoint_dir / f"{_safe_name(subject)}__{_safe_name(channel)}.npz"
                if args.resume and checkpoint.exists():
                    records.append((subject, channel, checkpoint))
                    processed += 1
                else:
                    pending.append((channel_index, channel, checkpoint))

            def run_one(item: tuple[int, str, Path]) -> tuple[str, Path, dict[str, Any]]:
                channel_index, channel, checkpoint = item
                electrode_seed = args.seed + int(subject[-3:]) * 10000 + sum(ord(c) for c in channel) * 10
                return channel, checkpoint, _electrode_result(subject, channel, task3, task2, shapes, channel_index, time3, electrode_seed, args.perms, max(1, int(args.workers)), not args.no_pca)

            # ``workers`` parallelizes the independent label permutations inside
            # one electrode.  Keep the outer electrode loop serial to avoid
            # nested process pools and oversubscription.
            max_workers = 1
            if pending:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(run_one, item) for item in pending]
                    for future in concurrent.futures.as_completed(futures):
                        channel, checkpoint, result = future.result()
                        _save_electrode(checkpoint, result)
                        records.append((subject, channel, checkpoint))
                        processed += 1
                        progress = {"completed_electrodes": processed, "last_subject": subject, "last_channel": channel, "updated_at_utc": _utc_now(), "workers": max_workers}
                        (run_dir / "progress.json").write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")
                        print(f"  {processed}: {subject} {channel}", flush=True)
        _aggregate(output_dir, records, not args.skip_plots)
        finish = _utc_now()
        summary = {"status": "completed", "started_at_utc": start, "finished_at_utc": finish, "n_electrodes": processed, "output_dir": str(output_dir), "run_dir": str(run_dir)}
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    except Exception as exc:
        failure = {"status": "failed", "started_at_utc": start, "failed_at_utc": _utc_now(), "completed_electrodes": processed, "error": repr(exc), "output_dir": str(output_dir), "run_dir": str(run_dir)}
        (run_dir / "run_summary.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
        raise


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
