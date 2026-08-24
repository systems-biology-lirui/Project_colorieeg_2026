"""Time-resolved decoding with cluster-based permutation correction.

This module answers the "when" question: at which post-stimulus time points a
single electrode (or an electrode group) carries classifiable color/memory
information. Features are sliding-window Welch band powers (STFT) for the 16
standard bands, baseline-z-scored per band across trials.

Statistical protocol (Maris-Oostenveld style):
  - per time point: label-permutation null (single-electrode) or sign-flip
    null (group level);
  - cluster-forming threshold: one-sided p<0.05 per time point;
  - cluster statistic: sum of (accuracy - 0.5) (single electrode) or sum of
    t-values (group level);
  - significance: observed cluster mass vs. max-mass null distribution.
"""

from __future__ import annotations

from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.signal import stft
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pipeline.spectral_features import FEATURE_BANDS, window_mask

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FS = 500.0
EVAL_MS = (0.0, 500.0)
STEP_MS = 10.0
NPERSEG = 128  # 256 ms window -> ~3.9 Hz frequency resolution
CLUSTER_FORM_P = 0.05
MIN_CLUSTER_MS = 20.0


def stft_band_power(
    epochs: np.ndarray,
    baseline_ms: tuple[float, float] = (-200.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window log band power.

    ``epochs``: (trials, channels, time). Returns (trials, channels, bands,
    frames) log power baseline-z-scored per channel/band across trials, plus
    the frame time axis in ms.
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    hop = max(1, int(round(STEP_MS / 1000.0 * FS)))
    n_trials, n_ch, n_time = epochs.shape
    n_bands = len(FEATURE_BANDS)
    frame_times_ms = None
    logpower_parts: list[np.ndarray] = []
    batch = 8
    for c0 in range(0, n_ch, batch):
        c1 = min(c0 + batch, n_ch)
        freqs, frame_times, z = stft(
            epochs[:, c0:c1, :],
            fs=FS,
            nperseg=NPERSEG,
            noverlap=NPERSEG - hop,
            axis=-1,
            boundary=None,
        )
        # scipy.stft reports frame times relative to the signal start; epochs
        # begin at -500 ms, so convert to epoch-centered milliseconds.
        frame_times_ms = frame_times * 1000.0 - 500.0
        power = np.abs(z) ** 2  # (trials, ch_batch, freq, frames)
        part = np.empty(
            (n_trials, c1 - c0, n_bands, power.shape[-1]), dtype=np.float32
        )
        for b, (lo, hi) in enumerate(FEATURE_BANDS):
            idx = np.where((freqs >= lo) & (freqs < hi))[0]
            if idx.size:
                part[:, :, b, :] = np.log(
                    np.mean(power[:, :, idx, :], axis=2) + 1e-12
                )
            else:
                part[:, :, b, :] = np.nan
        logpower_parts.append(part)
    logpower = np.concatenate(logpower_parts, axis=1)
    base_mask = window_mask(frame_times_ms, *baseline_ms)
    mu = np.nanmean(logpower[:, :, :, base_mask], axis=(0, 3), keepdims=True)
    sd = np.nanstd(logpower[:, :, :, base_mask], axis=(0, 3), keepdims=True)
    sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    zscore = (logpower - mu) / sd
    return np.asarray(zscore, dtype=np.float32), frame_times_ms


def eval_frames(
    frame_times_ms: np.ndarray, eval_ms: tuple[float, float] = EVAL_MS
) -> tuple[np.ndarray, np.ndarray]:
    """Select the evaluation time grid (10 ms steps) and its frame indices."""
    grid = np.arange(eval_ms[0], eval_ms[1] + 1e-9, STEP_MS)
    indices = np.array([int(np.argmin(np.abs(frame_times_ms - t))) for t in grid])
    return grid, indices


def decode_single_electrode_curve(
    x0: np.ndarray,
    x1: np.ndarray,
    n_perms: int,
    seed: int,
    workers: int,
    frame_indices: np.ndarray,
    n_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-time-point decoding of one electrode.

    ``x0/x1``: (trials, bands, frames). Returns real curve (n_eval,) and null
    matrix (n_perms, n_eval) of accuracies on the evaluation grid.
    """
    rng = np.random.default_rng(seed)
    x = np.concatenate([x0, x1], axis=0)[:, :, frame_indices]  # (N, bands, T)
    y = np.concatenate([np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)])
    n_times = x.shape[-1]
    n_splits = min(n_folds, int(np.min(np.bincount(y))))
    if n_splits < 2:
        return np.full(n_times, np.nan), np.full((n_perms, n_times), np.nan)
    splitter = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x, y)
    )

    def evaluate(y_train_override: np.ndarray | None) -> np.ndarray:
        scores = np.zeros(n_times, dtype=float)
        for t in range(n_times):
            folds = []
            for train_idx, test_idx in splitter:
                model = make_pipeline(
                    StandardScaler(), LinearSVC(C=1.0, max_iter=10000, dual=True)
                )
                y_train = (
                    y[train_idx] if y_train_override is None else y_train_override[train_idx]
                )
                model.fit(x[train_idx, :, t], y_train)
                folds.append(accuracy_score(y[test_idx], model.predict(x[test_idx, :, t])))
            scores[t] = float(np.mean(folds))
        return scores

    real = evaluate(None)
    if n_perms <= 0:
        return real, np.empty((0, n_times), dtype=np.float32)
    perm_labels = [rng.permutation(y) for _ in range(n_perms)]
    null = Parallel(n_jobs=min(max(1, workers), max(1, n_perms)), prefer="processes")(
        delayed(evaluate)(labels) for labels in perm_labels
    )
    return real, np.asarray(null, dtype=np.float32)


def cluster_permutation_1d(
    real: np.ndarray,
    null: np.ndarray,
    times_ms: np.ndarray,
    p_form: float = CLUSTER_FORM_P,
    min_cluster_ms: float = MIN_CLUSTER_MS,
) -> list[dict[str, float]]:
    """Cluster-based permutation test on a 1-D accuracy curve.

    Returns a list of significant clusters:
    {start_ms, end_ms, mass, p}. An empty list means nothing survived.
    """
    real = np.asarray(real, dtype=float)
    null = np.asarray(null, dtype=float)
    n_perms = null.shape[0]
    if n_perms == 0 or np.isnan(real).all():
        return []
    threshold = np.quantile(null, 1.0 - p_form, axis=0)  # per-time null threshold
    observed_mask = real > threshold
    min_len = max(1, int(round(min_cluster_ms / (times_ms[1] - times_ms[0]))))

    def clusters_from(mask: np.ndarray, values: np.ndarray) -> list[float]:
        masses = []
        start = None
        for i, flag in enumerate(mask):
            if flag and start is None:
                start = i
            if start is not None and (not flag or i == len(mask) - 1):
                end = i if (flag and i == len(mask) - 1) else i - 1
                if end - start + 1 >= min_len:
                    masses.append(float(np.sum(values[start : end + 1])))
                start = None
        return masses

    null_masses = []
    for i in range(n_perms):
        null_masses.extend(
            clusters_from(null[i] > threshold, null[i] - 0.5)
        )
    null_max = np.asarray(null_masses) if null_masses else np.zeros(1)
    result = []
    start = None
    for i, flag in enumerate(observed_mask):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(observed_mask) - 1):
            end = i if (flag and i == len(observed_mask) - 1) else i - 1
            if end - start + 1 >= min_len:
                mass = float(np.sum(real[start : end + 1] - 0.5))
                p = float(np.mean(null_max >= mass)) if null_max.size else 1.0
                result.append(
                    {
                        "start_ms": float(times_ms[start]),
                        "end_ms": float(times_ms[end]),
                        "mass": mass,
                        "p": p,
                    }
                )
            start = None
    return result


def group_cluster_permutation(
    subject_curves: np.ndarray,
    times_ms: np.ndarray,
    n_perms: int = 5000,
    seed: int = 0,
    p_form: float = CLUSTER_FORM_P,
    min_cluster_ms: float = MIN_CLUSTER_MS,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    """One-sample group cluster test by sign-flip permutation.

    ``subject_curves``: (n_subjects, n_times) accuracies. Returns a summary
    dict (mean curve, group t per time, significant clusters with p) and the
    significant-cluster list.
    """
    dev = np.asarray(subject_curves, dtype=float) - 0.5
    n_subj, n_times = dev.shape
    if n_subj < 2:
        return {"error": "need >=2 subjects"}, []
    mean_dev = dev.mean(axis=0)
    sd = dev.std(axis=0, ddof=1)
    t_obs = mean_dev / (sd / np.sqrt(n_subj) + 1e-12)
    t_crit = stats.t.ppf(1.0 - p_form, n_subj - 1)
    min_len = max(1, int(round(min_cluster_ms / (times_ms[1] - times_ms[0]))))
    dt = times_ms[1] - times_ms[0]

    def clusters(mask: np.ndarray, values: np.ndarray) -> list[float]:
        masses = []
        start = None
        for i, flag in enumerate(mask):
            if flag and start is None:
                start = i
            if start is not None and (not flag or i == len(mask) - 1):
                end = i if (flag and i == len(mask) - 1) else i - 1
                if end - start + 1 >= min_len:
                    masses.append(float(np.sum(values[start : end + 1])))
                start = None
        return masses

    rng = np.random.default_rng(seed)
    observed_mask = t_obs > t_crit
    null_max = []
    for _ in range(n_perms):
        flips = rng.choice([-1.0, 1.0], size=(n_subj, 1))
        perm_dev = flips * dev
        perm_mean = perm_dev.mean(axis=0)
        perm_sd = perm_dev.std(axis=0, ddof=1)
        perm_t = perm_mean / (perm_sd / np.sqrt(n_subj) + 1e-12)
        masses = clusters(perm_t > t_crit, perm_t)
        null_max.append(max(masses) if masses else 0.0)
    null_max = np.asarray(null_max)
    clusters_out: list[dict[str, float]] = []
    start = None
    for i, flag in enumerate(observed_mask):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(observed_mask) - 1):
            end = i if (flag and i == len(observed_mask) - 1) else i - 1
            if end - start + 1 >= min_len:
                mass = float(np.sum(t_obs[start : end + 1]))
                p = float(np.mean(null_max >= mass))
                clusters_out.append(
                    {
                        "start_ms": float(times_ms[start]),
                        "end_ms": float(times_ms[end]),
                        "mass": mass,
                        "p": p,
                    }
                )
            start = None
    summary = {
        "n_subjects": n_subj,
        "mean_curve": mean_dev + 0.5,
        "t_curve": t_obs,
        "times_ms": times_ms,
        "t_crit": float(t_crit),
        "n_permutations": n_perms,
        "significant_clusters": clusters_out,
    }
    return summary, clusters_out


def curves_to_table(
    rows: Iterable[dict[str, object]], columns: Sequence[str]
) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(columns))
