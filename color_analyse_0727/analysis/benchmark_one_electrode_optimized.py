"""Optimized timing benchmark for one real SEEG electrode.

This keeps the baseline benchmark's data splits and permutation design, but
precomputes training-fold standardization and uses LinearSVC(dual=False).
It is a timing benchmark, not a replacement for the formal analysis code.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from analysis.benchmark_one_electrode import _load_one_electrode
from analysis.common import GRAY_FRUITS


def _model_accuracy(x_train: np.ndarray, labels: np.ndarray,
                    x_test: np.ndarray, y_test: np.ndarray) -> float:
    model = LinearSVC(C=1.0, max_iter=10000, dual=False)
    model.fit(x_train, labels)
    return float(accuracy_score(y_test, model.predict(x_test)))


def _scaled(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)


def _fit_binary_cv_optimized(
    x0: np.ndarray, x1: np.ndarray, n_perms: int, seed: int, n_jobs: int
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)])
    n_splits = min(5, int(np.min(np.bincount(y))))
    splitter = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed).split(x, y))
    folds = []
    for train_idx, test_idx in splitter:
        x_train, x_test = _scaled(x[train_idx], x[test_idx])
        folds.append((x_train, x_test, train_idx, y[test_idx]))

    def evaluate(labels: np.ndarray) -> float:
        scores = [
            _model_accuracy(x_train, labels[train_idx], x_test, y_test)
            for x_train, x_test, train_idx, y_test in folds
        ]
        return float(np.mean(scores))

    real = evaluate(y)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_perms, dtype=np.int64)
    null = Parallel(n_jobs=min(max(1, n_jobs), max(1, n_perms)), prefer="processes")(
        delayed(evaluate)(np.random.default_rng(int(s)).permutation(y)) for s in seeds
    )
    return real, np.asarray(null, dtype=np.float32)


def _fit_curve_cv_optimized(
    x0: np.ndarray, x1: np.ndarray, n_perms: int, seed: int, n_jobs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Time-resolved within-task CV with fold scaling cached across permutations."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)])
    n_splits = min(5, int(np.min(np.bincount(y))))
    splitter = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed).split(x, y))
    prepared = []
    for t in range(x.shape[-1]):
        time_folds = []
        for train_idx, test_idx in splitter:
            x_train, x_test = _scaled(x[train_idx, :, t], x[test_idx, :, t])
            time_folds.append((x_train, x_test, train_idx, y[test_idx]))
        prepared.append(time_folds)

    def evaluate(labels: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(prepared), dtype=float)
        for t, folds_t in enumerate(prepared):
            scores[t] = np.mean([
                _model_accuracy(x_train, labels[train_idx], x_test, y_test)
                for x_train, x_test, train_idx, y_test in folds_t
            ])
        return scores

    real = evaluate(y)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_perms, dtype=np.int64)
    null = Parallel(n_jobs=min(max(1, n_jobs), max(1, n_perms)), prefer="processes")(
        delayed(evaluate)(np.random.default_rng(int(s)).permutation(y)) for s in seeds
    )
    return real, np.asarray(null, dtype=np.float32)


def _fit_cross_fruit_optimized(
    fruits: dict[str, np.ndarray], n_perms: int, seed: int, n_jobs: int
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    names = ["strawberry", "watermelon", "cabbage", "kiwi"]
    n_min = min(fruits[name].shape[0] for name in names)
    arrays = [fruits[name][rng.permutation(fruits[name].shape[0])[:n_min]] for name in names]
    folds = [(0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2)]
    prepared = []
    for tr_red, tr_green, te_red, te_green in folds:
        x_train = np.concatenate([arrays[tr_red], arrays[tr_green]], axis=0)
        x_test = np.concatenate([arrays[te_red], arrays[te_green]], axis=0)
        y_train = np.concatenate([
            np.zeros(len(arrays[tr_red]), dtype=int),
            np.ones(len(arrays[tr_green]), dtype=int),
        ])
        y_test = np.concatenate([
            np.zeros(len(arrays[te_red]), dtype=int),
            np.ones(len(arrays[te_green]), dtype=int),
        ])
        x_train, x_test = _scaled(x_train, x_test)
        prepared.append((x_train, x_test, y_train, y_test))

    def evaluate(shuffles: list[np.ndarray] | None) -> float:
        scores = []
        for fi, (x_train, x_test, y_train, y_test) in enumerate(prepared):
            labels = y_train if shuffles is None else shuffles[fi]
            scores.append(_model_accuracy(x_train, labels, x_test, y_test))
        return float(np.mean(scores))

    real = evaluate(None)

    def one_perm(seed_value: int) -> float:
        local = np.random.default_rng(int(seed_value))
        shuffles = [local.permutation(y_train) for _, _, y_train, _ in prepared]
        return evaluate(shuffles)

    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_perms, dtype=np.int64)
    null = Parallel(n_jobs=min(max(1, n_jobs), max(1, n_perms)), prefer="processes")(
        delayed(one_perm)(s) for s in seeds
    )
    return real, np.asarray(null, dtype=np.float32)


def _cross_task_fixed_optimized(
    train0: np.ndarray, train1: np.ndarray, test0: np.ndarray, test1: np.ndarray,
    n_perms: int, seed: int, n_jobs: int,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_train = min(len(train0), len(train1))
    n_test = min(len(test0), len(test1))
    train0 = train0[rng.permutation(len(train0))[:n_train]]
    train1 = train1[rng.permutation(len(train1))[:n_train]]
    test0 = test0[rng.permutation(len(test0))[:n_test]]
    test1 = test1[rng.permutation(len(test1))[:n_test]]
    x_train = np.concatenate([train0, train1])
    x_test = np.concatenate([test0, test1])
    y_train = np.concatenate([np.zeros(n_train, dtype=int), np.ones(n_train, dtype=int)])
    y_test = np.concatenate([np.zeros(n_test, dtype=int), np.ones(n_test, dtype=int)])
    x_train, x_test = _scaled(x_train, x_test)

    def evaluate(labels: np.ndarray) -> float:
        return _model_accuracy(x_train, labels, x_test, y_test)

    real = evaluate(y_train)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_perms, dtype=np.int64)
    null = Parallel(n_jobs=min(max(1, n_jobs), max(1, n_perms)), prefer="processes")(
        delayed(evaluate)(np.random.default_rng(int(s)).permutation(y_train)) for s in seeds
    )
    return real, np.asarray(null, dtype=np.float32)


def _cross_task_curve_optimized(
    train0: np.ndarray, train1: np.ndarray, test0: np.ndarray, test1: np.ndarray,
    n_perms: int, seed: int, n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_train = min(len(train0), len(train1))
    n_test = min(len(test0), len(test1))
    train0 = train0[rng.permutation(len(train0))[:n_train]]
    train1 = train1[rng.permutation(len(train1))[:n_train]]
    test0 = test0[rng.permutation(len(test0))[:n_test]]
    test1 = test1[rng.permutation(len(test1))[:n_test]]
    x_train = np.concatenate([train0, train1], axis=0)
    x_test = np.concatenate([test0, test1], axis=0)
    y_train = np.concatenate([np.zeros(n_train, dtype=int), np.ones(n_train, dtype=int)])
    y_test = np.concatenate([np.zeros(n_test, dtype=int), np.ones(n_test, dtype=int)])
    prepared = [_scaled(x_train[:, :, t], x_test[:, :, t]) for t in range(x_train.shape[-1])]

    def evaluate(labels: np.ndarray) -> np.ndarray:
        return np.asarray([
            _model_accuracy(xtr, labels, xte, y_test) for xtr, xte in prepared
        ])

    real = evaluate(y_train)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_perms, dtype=np.int64)
    null = Parallel(n_jobs=min(max(1, n_jobs), max(1, n_perms)), prefer="processes")(
        delayed(evaluate)(np.random.default_rng(int(s)).permutation(y_train)) for s in seeds
    )
    return real, np.asarray(null, dtype=np.float32)


def _cross_fruit_curve_optimized(
    feats: dict[str, np.ndarray], frame_indices: np.ndarray, channel: int,
    n_perms: int, seed: int, n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    names = ["strawberry", "watermelon", "cabbage", "kiwi"]
    rng = np.random.default_rng(seed)
    n_min = min(feats[name].shape[0] for name in names)
    arrays = []
    for name in names:
        arr = feats[name][:, channel, :, :]
        idx = rng.permutation(arr.shape[0])[:n_min]
        arrays.append(np.take(arr[idx], frame_indices, axis=-1))
    folds = [(0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2)]
    prepared = []
    for t in range(len(frame_indices)):
        time_folds = []
        for tr_red, tr_green, te_red, te_green in folds:
            x_train = np.concatenate([arrays[tr_red][:, :, t], arrays[tr_green][:, :, t]])
            x_test = np.concatenate([arrays[te_red][:, :, t], arrays[te_green][:, :, t]])
            y_train = np.concatenate([
                np.zeros(len(arrays[tr_red]), dtype=int), np.ones(len(arrays[tr_green]), dtype=int)
            ])
            y_test = np.concatenate([
                np.zeros(len(arrays[te_red]), dtype=int), np.ones(len(arrays[te_green]), dtype=int)
            ])
            x_train, x_test = _scaled(x_train, x_test)
            time_folds.append((x_train, x_test, y_train, y_test))
        prepared.append(time_folds)

    def evaluate(shuffles: list[np.ndarray] | None) -> np.ndarray:
        scores = np.zeros(len(prepared), dtype=float)
        for t, folds_t in enumerate(prepared):
            fold_scores = []
            for fi, (x_train, x_test, y_train, y_test) in enumerate(folds_t):
                labels = y_train if shuffles is None else shuffles[t][fi]
                fold_scores.append(_model_accuracy(x_train, labels, x_test, y_test))
            scores[t] = np.mean(fold_scores)
        return scores

    real = evaluate(None)

    def one_perm(seed_value: int) -> np.ndarray:
        local = np.random.default_rng(int(seed_value))
        shuffles = []
        for folds_t in prepared:
            shuffles.append([local.permutation(y_train) for _, _, y_train, _ in folds_t])
        return evaluate(shuffles)

    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_perms, dtype=np.int64)
    null = Parallel(n_jobs=min(max(1, n_jobs), max(1, n_perms)), prefer="processes")(
        delayed(one_perm)(s) for s in seeds
    )
    return real, np.asarray(null, dtype=np.float32)


def _timed(name: str, fn: Callable[[], object], timings: dict[str, object]):
    start = time.perf_counter()
    value = fn()
    timings[name] = round(time.perf_counter() - start, 3)
    print(f"{name}: {timings[name]} s", flush=True)
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
        lambda: _load_one_electrode(args.subject, args.channel, args.signal), timings,
    )

    red3, green3 = spec3["red"], spec3["green"]
    red2 = np.concatenate([spec2["strawberry_gray"], spec2["watermelon_gray"]])
    green2 = np.concatenate([spec2["cabbage_gray"], spec2["kiwi_gray"]])
    _timed("spectral_task3_within", lambda: _fit_binary_cv_optimized(
        red3, green3, args.perms, 1001, args.workers), timings)
    _timed("spectral_task2_cross_fruit", lambda: _fit_cross_fruit_optimized(
        {fruit.removesuffix("_gray"): spec2[fruit] for fruit in spec2},
        args.perms, 1002, args.workers), timings)
    _timed("spectral_task3_to_task2", lambda: _cross_task_fixed_optimized(
        red3, green3, red2, green2, args.perms, 1003, args.workers), timings)
    _timed("spectral_task2_to_task3", lambda: _cross_task_fixed_optimized(
        red2, green2, red3, green3, args.perms, 1004, args.workers), timings)

    t3_red, t3_green = tf3["red"][:, 0], tf3["green"][:, 0]
    tf2_by_fruit = {name.removesuffix("_gray"): value for name, value in tf2.items()}
    _timed("timefreq_task3_within", lambda: _fit_curve_cv_optimized(
        t3_red[:, :, frame_indices], t3_green[:, :, frame_indices],
        args.perms, 2001, args.workers), timings)
    _timed("timefreq_task2_cross_fruit", lambda: _cross_fruit_curve_optimized(
        tf2_by_fruit, frame_indices, 0, args.perms, 2002, args.workers), timings)
    _timed("timefreq_task3_to_task2", lambda: _cross_task_curve_optimized(
        t3_red[:, :, frame_indices], t3_green[:, :, frame_indices],
        np.concatenate([tf2["strawberry_gray"][:, 0][:, :, frame_indices],
                        tf2["watermelon_gray"][:, 0][:, :, frame_indices]]),
        np.concatenate([tf2["cabbage_gray"][:, 0][:, :, frame_indices],
                        tf2["kiwi_gray"][:, 0][:, :, frame_indices]]),
        args.perms, 2003, args.workers), timings)
    _timed("timefreq_task2_to_task3", lambda: _cross_task_curve_optimized(
        np.concatenate([tf2["strawberry_gray"][:, 0][:, :, frame_indices],
                        tf2["watermelon_gray"][:, 0][:, :, frame_indices]]),
        np.concatenate([tf2["cabbage_gray"][:, 0][:, :, frame_indices],
                        tf2["kiwi_gray"][:, 0][:, :, frame_indices]]),
        t3_red[:, :, frame_indices], t3_green[:, :, frame_indices],
        args.perms, 2004, args.workers), timings)

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
        "optimization": {
            "classifier": "LinearSVC(C=1.0, max_iter=10000, dual=False)",
            "standardization": "precomputed once per training fold/time point and reused across permutations",
            "permutation_design": "same split and seeds as baseline benchmark",
        },
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
