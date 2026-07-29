"""Efficient, leakage-safe time-resolved decoding helpers."""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold


def make_time_windows(times_ms: np.ndarray, width_ms: float, step_ms: float):
    starts = np.arange(times_ms[0], times_ms[-1] - width_ms + 1e-9, step_ms)
    windows = []
    centers = []
    for start in starts:
        idx = np.flatnonzero((times_ms >= start) & (times_ms < start + width_ms))
        if idx.size:
            windows.append(idx)
            centers.append(float(times_ms[idx].mean()))
    return windows, np.asarray(centers)


def window_features(data: np.ndarray, windows: list[np.ndarray]) -> np.ndarray:
    """Convert [trial, channel, time] to [trial, window*channel]."""
    return np.concatenate([data[:, :, idx].mean(axis=2) for idx in windows], axis=1)


def window_tensor(data: np.ndarray, windows: list[np.ndarray]) -> np.ndarray:
    """Precompute [window, trial, channel] once for repeated/permuted fits."""
    return np.stack([data[:, :, idx].mean(axis=2) for idx in windows], axis=0).astype(np.float32)


def fixed_cv(y: np.ndarray, n_splits: int, seed: int):
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def decode_accuracy(
    data: np.ndarray,
    y: np.ndarray,
    windows: list[np.ndarray],
    cv_splits,
    estimator=None,
) -> np.ndarray:
    """Decode all windows using one precomputed CV partition set."""
    features = window_tensor(data, windows)
    return decode_precomputed(features, y, cv_splits, estimator)


def decode_precomputed(features, y, cv_splits, estimator=None) -> np.ndarray:
    estimator = estimator or make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto", random_state=0))
    scores = np.empty(features.shape[0], dtype=float)
    for wi, x in enumerate(features):
        fold_scores = []
        for train, test in cv_splits:
            model = clone(estimator)
            model.fit(x[train], y[train])
            pred = model.predict(x[test])
            fold_scores.append(np.mean(pred == y[test]))
        scores[wi] = np.mean(fold_scores)
    return scores


def decode_permutation(
    data: np.ndarray,
    y: np.ndarray,
    windows: list[np.ndarray],
    cv_splits,
    n_permutations: int,
    seed: int,
    n_jobs: int = -1,
) -> np.ndarray:
    """Parallel null distribution; CV splits are reused to reduce overhead."""
    rng = np.random.default_rng(seed)
    labels = [rng.permutation(y) for _ in range(n_permutations)]
    features = window_tensor(data, windows)

    def one(null_y):
        return decode_precomputed(features, null_y, cv_splits)

    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        return np.asarray(Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(one)(null_y) for null_y in labels
        ))


def temporal_generalization(train_data, test_data, y_train, y_test, windows, seed=0):
    """Train at each train window and test at every test window."""
    result = np.empty((len(windows), len(windows)), dtype=float)
    for i, train_idx in enumerate(windows):
        x_train = train_data[:, :, train_idx].mean(axis=2)
        model = make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto", random_state=seed))
        model.fit(x_train, y_train)
        for j, test_idx in enumerate(windows):
            pred = model.predict(test_data[:, :, test_idx].mean(axis=2))
            result[i, j] = np.mean(pred == y_test)
    return result
