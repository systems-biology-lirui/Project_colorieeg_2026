"""Benchmark all requested decoding modes on one real SEEG electrode.

This is a timing benchmark, not a scientific result. It uses one electrode
and 100 label permutations with the current project preprocessing and model
families. Cross-task decoding is defined as train on one task and test on the
other task, with balanced red/green labels.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from analysis.common import GRAY_FRUITS, AnalysisVariant, load_conditions
from analysis.decoding import _fit_binary_cv, _fit_cross_fruit
from analysis.decoding_timeresolved import eval_frames, stft_band_power
from analysis.selection import prepare_signal
from pipeline.spectral_features import (
    band_power_baseline_z,
    welch_band_power,
    window_mask,
)


def _cross_task_fixed(
    train0: np.ndarray,
    train1: np.ndarray,
    test0: np.ndarray,
    test1: np.ndarray,
    n_perms: int,
    seed: int,
    workers: int,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_train = min(len(train0), len(train1))
    n_test = min(len(test0), len(test1))
    train0 = train0[rng.permutation(len(train0))[:n_train]]
    train1 = train1[rng.permutation(len(train1))[:n_train]]
    test0 = test0[rng.permutation(len(test0))[:n_test]]
    test1 = test1[rng.permutation(len(test1))[:n_test]]
    x_train = np.concatenate([train0, train1])
    y_train = np.concatenate(
        [np.zeros(n_train, dtype=int), np.ones(n_train, dtype=int)]
    )
    x_test = np.concatenate([test0, test1])
    y_test = np.concatenate(
        [np.zeros(n_test, dtype=int), np.ones(n_test, dtype=int)]
    )

    def evaluate(labels: np.ndarray) -> float:
        model = make_pipeline(
            StandardScaler(), LinearSVC(C=1.0, max_iter=10000, dual=True)
        )
        model.fit(x_train, labels)
        return float(accuracy_score(y_test, model.predict(x_test)))

    real = evaluate(y_train)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_perms, dtype=np.int64)
    null = Parallel(
        n_jobs=min(max(1, workers), max(1, n_perms)), prefer="processes"
    )(
        delayed(evaluate)(np.random.default_rng(int(s)).permutation(y_train))
        for s in seeds
    )
    return real, np.asarray(null, dtype=np.float32)


def _cross_task_curve(
    train0: np.ndarray,
    train1: np.ndarray,
    test0: np.ndarray,
    test1: np.ndarray,
    n_perms: int,
    seed: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_train = min(len(train0), len(train1))
    n_test = min(len(test0), len(test1))
    train0 = train0[rng.permutation(len(train0))[:n_train]]
    train1 = train1[rng.permutation(len(train1))[:n_train]]
    test0 = test0[rng.permutation(len(test0))[:n_test]]
    test1 = test1[rng.permutation(len(test1))[:n_test]]
    x_train = np.concatenate([train0, train1], axis=0)
    y_train = np.concatenate(
        [np.zeros(n_train, dtype=int), np.ones(n_train, dtype=int)]
    )
    x_test = np.concatenate([test0, test1], axis=0)
    y_test = np.concatenate(
        [np.zeros(n_test, dtype=int), np.ones(n_test, dtype=int)]
    )
    n_times = x_train.shape[-1]

    def evaluate(labels: np.ndarray) -> np.ndarray:
        scores = np.zeros(n_times, dtype=float)
        for t in range(n_times):
            model = make_pipeline(
                StandardScaler(), LinearSVC(C=1.0, max_iter=10000, dual=True)
            )
            model.fit(x_train[:, :, t], labels)
            scores[t] = accuracy_score(y_test, model.predict(x_test[:, :, t]))
        return scores

    real = evaluate(y_train)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_perms, dtype=np.int64)
    null = Parallel(
        n_jobs=min(max(1, workers), max(1, n_perms)), prefer="processes"
    )(
        delayed(evaluate)(np.random.default_rng(int(s)).permutation(y_train))
        for s in seeds
    )
    return real, np.asarray(null, dtype=np.float32)


def _cross_fruit_curve_fixed(
    feats: dict[str, np.ndarray],
    frame_indices: np.ndarray,
    channel: int,
    n_perms: int,
    seed: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Correctly ordered leave-one-fruit-pair-out time curve for benchmarking."""
    names = ["strawberry", "watermelon", "cabbage", "kiwi"]
    rng = np.random.default_rng(seed)
    n_min = min(feats[n].shape[0] for n in names)
    arrays = []
    for name in names:
        arr = feats[name][:, channel, :, :]
        idx = rng.permutation(arr.shape[0])[:n_min]
        arrays.append(np.take(arr[idx], frame_indices, axis=-1))
    folds = [(0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2)]
    n_times = len(frame_indices)

    def evaluate(shuffles: list[np.ndarray] | None) -> np.ndarray:
        scores = np.zeros(n_times, dtype=float)
        for t in range(n_times):
            fold_scores = []
            for fi, (tr_red, tr_green, te_red, te_green) in enumerate(folds):
                x_train = np.concatenate(
                    [arrays[tr_red][:, :, t], arrays[tr_green][:, :, t]], axis=0
                )
                y_train = np.concatenate(
                    [np.zeros(len(arrays[tr_red]), dtype=int), np.ones(len(arrays[tr_green]), dtype=int)]
                )
                x_test = np.concatenate(
                    [arrays[te_red][:, :, t], arrays[te_green][:, :, t]], axis=0
                )
                y_test = np.concatenate(
                    [np.zeros(len(arrays[te_red]), dtype=int), np.ones(len(arrays[te_green]), dtype=int)]
                )
                labels = y_train if shuffles is None else shuffles[fi]
                model = make_pipeline(
                    StandardScaler(), LinearSVC(C=1.0, max_iter=10000, dual=True)
                )
                model.fit(x_train, labels)
                fold_scores.append(accuracy_score(y_test, model.predict(x_test)))
            scores[t] = np.mean(fold_scores)
        return scores

    real = evaluate(None)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_perms, dtype=np.int64)

    def one_perm(seed_value: int) -> np.ndarray:
        local = np.random.default_rng(int(seed_value))
        shuffles = []
        for tr_red, tr_green, _, _ in folds:
            y = np.concatenate(
                [np.zeros(len(arrays[tr_red]), dtype=int), np.ones(len(arrays[tr_green]), dtype=int)]
            )
            shuffles.append(local.permutation(y))
        return evaluate(shuffles)

    null = Parallel(
        n_jobs=min(max(1, workers), max(1, n_perms)), prefer="processes"
    )(delayed(one_perm)(s) for s in seeds)
    return real, np.asarray(null, dtype=np.float32)


def _load_one_electrode(subject: str, channel: str, signal: str = "raw200"):
    signal_variant = AnalysisVariant((100.0, 400.0), signal)
    task3_names = ["red", "green"]
    task2_names = [f"{fruit}_gray" for fruit in GRAY_FRUITS]
    raw3, time3, _ = load_conditions(subject, 3, task3_names, [channel])
    raw2, time2, _ = load_conditions(subject, 2, task2_names, [channel])
    raw3 = {k: prepare_signal(v, time3, signal_variant) for k, v in raw3.items()}
    raw2 = {k: prepare_signal(v, time2, signal_variant) for k, v in raw2.items()}
    full3 = window_mask(time3, 1.0, 1000.0)
    full2 = window_mask(time2, 1.0, 1000.0)
    base3 = window_mask(time3, -200.0, 0.0)
    base2 = window_mask(time2, -200.0, 0.0)

    def spectrum(raw, names, base, full):
        out = {}
        for name in names:
            base_power = welch_band_power(raw[name][:, :, base])
            full_power = welch_band_power(raw[name][:, :, full])
            out[name] = band_power_baseline_z(full_power, base_power)[:, 0]
        return out

    spec3 = spectrum(raw3, task3_names, base3, full3)
    spec2 = spectrum(raw2, task2_names, base2, full2)

    tf3 = {}
    tf2 = {}
    frame_times = None
    for name in task3_names:
        tf3[name], frame_times = stft_band_power(raw3[name])
    for name in task2_names:
        tf2[name], frame_times2 = stft_band_power(raw2[name])
    if not np.allclose(frame_times, frame_times2):
        raise RuntimeError("Task2 and Task3 time grids differ")
    grid = np.arange(0.0, 1000.0 + 1e-9, 10.0)
    frame_indices = np.array([int(np.argmin(np.abs(frame_times - t))) for t in grid])
    return spec3, spec2, tf3, tf2, grid, frame_indices, frame_times


def _timed(name: str, fn: Callable[[], object], results: dict[str, object]):
    start = time.perf_counter()
    value = fn()
    results[name] = round(time.perf_counter() - start, 3)
    print(f"{name}: {results[name]} s", flush=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="test001")
    parser.add_argument("--channel", default="D6")
    parser.add_argument("--perms", type=int, default=100)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--signal", choices=("lf30", "raw200"), default="raw200")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    timings: dict[str, object] = {}
    start_all = time.perf_counter()
    spec3, spec2, tf3, tf2, grid, frame_indices, frame_times = _timed(
        "feature_loading_and_preparation",
        lambda: _load_one_electrode(args.subject, args.channel, args.signal),
        timings,
    )

    red3, green3 = spec3["red"], spec3["green"]
    red2 = np.concatenate([spec2["strawberry_gray"], spec2["watermelon_gray"]])
    green2 = np.concatenate([spec2["cabbage_gray"], spec2["kiwi_gray"]])
    _timed(
        "spectral_task3_within",
        lambda: _fit_binary_cv(red3, green3, args.perms, 1001, args.workers),
        timings,
    )
    _timed(
        "spectral_task2_cross_fruit",
        lambda: _fit_cross_fruit(
            {fruit.removesuffix("_gray"): spec2[fruit] for fruit in spec2},
            args.perms,
            1002,
            args.workers,
        ),
        timings,
    )
    _timed(
        "spectral_task3_to_task2",
        lambda: _cross_task_fixed(red3, green3, red2, green2, args.perms, 1003, args.workers),
        timings,
    )
    _timed(
        "spectral_task2_to_task3",
        lambda: _cross_task_fixed(red2, green2, red3, green3, args.perms, 1004, args.workers),
        timings,
    )

    t3_red = tf3["red"][:, 0]
    t3_green = tf3["green"][:, 0]
    tf2_by_fruit = {
        name.removesuffix("_gray"): value for name, value in tf2.items()
    }
    _timed(
        "timefreq_task3_within",
        lambda: __import__("analysis.decoding_timeresolved", fromlist=["decode_single_electrode_curve"]).decode_single_electrode_curve(
            t3_red, t3_green, args.perms, 2001, args.workers, frame_indices
        ),
        timings,
    )
    _timed(
        "timefreq_task2_cross_fruit",
        lambda: _cross_fruit_curve_fixed(
            tf2_by_fruit, frame_indices, 0, args.perms, 2002, args.workers
        ),
        timings,
    )
    _timed(
        "timefreq_task3_to_task2",
        lambda: _cross_task_curve(
            t3_red, t3_green,
            np.concatenate([tf2["strawberry_gray"][:, 0], tf2["watermelon_gray"][:, 0]]),
            np.concatenate([tf2["cabbage_gray"][:, 0], tf2["kiwi_gray"][:, 0]]),
            args.perms, 2003, args.workers,
        ),
        timings,
    )
    _timed(
        "timefreq_task2_to_task3",
        lambda: _cross_task_curve(
            np.concatenate([tf2["strawberry_gray"][:, 0], tf2["watermelon_gray"][:, 0]]),
            np.concatenate([tf2["cabbage_gray"][:, 0], tf2["kiwi_gray"][:, 0]]),
            t3_red, t3_green, args.perms, 2004, args.workers,
        ),
        timings,
    )

    timings["total_seconds"] = round(time.perf_counter() - start_all, 3)
    output = {
        "subject": args.subject,
        "channel": args.channel,
        "permutations": args.perms,
        "signal": args.signal,
        "workers": args.workers,
        "logical_cpus": os.cpu_count(),
        "python": sys.version,
        "platform": platform.platform(),
        "timefreq_grid_points": len(grid),
        "stft_frame_min_ms": float(frame_times[0]),
        "stft_frame_max_ms": float(frame_times[-1]),
        "note": "Current STFT uses 256 ms window and 10 ms hop; 1-1000 ms endpoints are nearest-frame mapped.",
        "timings_seconds": timings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
