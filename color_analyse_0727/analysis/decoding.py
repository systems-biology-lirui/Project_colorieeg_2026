"""Stage 3: spectrum-level single-electrode decoding with permutation tests.

Features are the 16 Welch band-power values (5-195 Hz minus the line-noise
bands) baseline-z-scored against the per-condition baseline-trial distribution,
then standardized again with a StandardScaler fitted on the training folds
only (two-layer normalization). Significance comes from label permutations
(default 1000); peak-accuracy style statements are descriptive only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pipeline.spectral_features import (
    band_power_baseline_z,
    welch_band_power,
    window_mask,
)
from analysis.common import (
    BASELINE_MS,
    GRAY_FRUITS,
    SUBJECTS,
    AnalysisVariant,
    all_variants,
    bh_adjust,
    load_conditions,
)
from analysis.selection import prepare_signal


def _spectrum_features(
    subject: str,
    task: int,
    conditions: Iterable[str],
    channels: Iterable[str],
    time_ms: np.ndarray,
    variant: AnalysisVariant,
) -> dict[str, np.ndarray]:
    raw, time_ms, _ = load_conditions(subject, task, conditions, channels)
    raw = {
        condition: prepare_signal(values, time_ms, variant)
        for condition, values in raw.items()
    }
    base_win = window_mask(time_ms, *BASELINE_MS)
    win = window_mask(time_ms, *variant.window)
    features: dict[str, np.ndarray] = {}
    for condition in conditions:
        base_power = welch_band_power(raw[condition][:, :, base_win])
        analysis_power = welch_band_power(raw[condition][:, :, win])
        features[condition] = band_power_baseline_z(analysis_power, base_power)
    return features


def _balanced_trials(
    arrays: list[np.ndarray], rng: np.random.Generator
) -> list[np.ndarray]:
    n = min(a.shape[0] for a in arrays)
    return [a[rng.permutation(a.shape[0])[:n]] for a in arrays]


def _fit_binary_cv(
    x0: np.ndarray,
    x1: np.ndarray,
    n_perms: int,
    seed: int,
    n_jobs: int,
) -> tuple[float, np.ndarray, float]:
    """Balanced two-class CV decoding on fixed 16-band feature vectors."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)])
    n_splits = min(5, int(np.min(np.bincount(y))))
    if n_splits < 2:
        return np.nan, np.full(n_perms, np.nan), np.nan
    splitter = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x, y)
    )

    def evaluate(y_train_override: np.ndarray | None) -> float:
        scores = []
        for train_idx, test_idx in splitter:
            model = make_pipeline(
                StandardScaler(), LinearSVC(C=1.0, max_iter=3000, dual=True)
            )
            y_train = (
                y[train_idx] if y_train_override is None else y_train_override[train_idx]
            )
            model.fit(x[train_idx], y_train)
            scores.append(accuracy_score(y[test_idx], model.predict(x[test_idx])))
        return float(np.mean(scores))

    real = evaluate(None)
    perm_labels = [rng.permutation(y) for _ in range(n_perms)]
    null = Parallel(n_jobs=min(max(1, n_jobs), max(1, n_perms)), prefer="processes")(
        delayed(evaluate)(labels) for labels in perm_labels
    )
    null = np.asarray(null, dtype=float)
    p = (1.0 + float(np.sum(null >= real))) / (n_perms + 1.0)
    return real, null, p


def _fit_cross_fruit(
    fruits: dict[str, np.ndarray],
    n_perms: int,
    seed: int,
    n_jobs: int,
) -> tuple[float, np.ndarray, float]:
    """Leave-one-fruit-pair-out red-vs-green generalization (Task 2)."""
    rng = np.random.default_rng(seed)
    names = ["strawberry", "watermelon", "cabbage", "kiwi"]
    arrays = _balanced_trials([fruits[n] for n in names], rng)
    folds = [
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
    ]

    def evaluate(shuffles: list[np.ndarray] | None) -> float:
        fold_scores = []
        for fold_idx, (tr_red, tr_green, te_red, te_green) in enumerate(folds):
            x_train = np.concatenate([arrays[tr_red], arrays[tr_green]], axis=0)
            y_train = np.concatenate(
                [
                    np.zeros(len(arrays[tr_red]), dtype=int),
                    np.ones(len(arrays[tr_green]), dtype=int),
                ]
            )
            x_test = np.concatenate([arrays[te_red], arrays[te_green]], axis=0)
            y_test = np.concatenate(
                [
                    np.zeros(len(arrays[te_red]), dtype=int),
                    np.ones(len(arrays[te_green]), dtype=int),
                ]
            )
            y_fit = y_train if shuffles is None else shuffles[fold_idx]
            model = make_pipeline(
                StandardScaler(), LinearSVC(C=1.0, max_iter=3000, dual=True)
            )
            model.fit(x_train, y_fit)
            fold_scores.append(accuracy_score(y_test, model.predict(x_test)))
        return float(np.mean(fold_scores))

    real = evaluate(None)

    def one_permutation(random_seed: int) -> float:
        local_rng = np.random.default_rng(int(random_seed))
        shuffles = []
        for tr_red, tr_green, _, _ in folds:
            shuffles.append(
                local_rng.permutation(
                    np.concatenate(
                        [
                            np.zeros(len(arrays[tr_red]), dtype=int),
                            np.ones(len(arrays[tr_green]), dtype=int),
                        ]
                    )
                )
            )
        return evaluate(shuffles)

    perm_seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_perms, dtype=np.int64)
    null = Parallel(n_jobs=min(max(1, n_jobs), max(1, n_perms)), prefer="processes")(
        delayed(one_permutation)(random_seed) for random_seed in perm_seeds
    )
    null = np.asarray(null, dtype=float)
    p = (1.0 + float(np.sum(null >= real))) / (n_perms + 1.0)
    return real, null, p


def _channel_seed(subject: str, channel: str, variant: AnalysisVariant, base: int) -> int:
    subject_code = int(subject[-3:])
    channel_code = sum(ord(c) for c in channel)
    variant_code = all_variants().index(variant)
    return base + subject_code * 10000 + channel_code * 100 + variant_code


def _direction_consistency(
    out: Path, csc_table: pd.DataFrame, variant: AnalysisVariant
) -> pd.DataFrame:
    """Within-class sign consistency of the window-mean gray-fruit amplitude."""
    rows: list[dict[str, object]] = []
    for subject, grp in csc_table[csc_table.CSC].groupby("subject"):
        channels = list(grp.channel.astype(str))
        t2, time_ms, _ = load_conditions(
            subject, 2, [f"{fruit}_gray" for fruit in GRAY_FRUITS], channels
        )
        t2 = {
            key: prepare_signal(values, time_ms, variant)
            for key, values in t2.items()
        }
        win = window_mask(time_ms, *variant.window)
        for j, channel in enumerate(channels):
            means = {}
            for fruit in GRAY_FRUITS:
                trial_vals = np.nanmean(t2[f"{fruit}_gray"][:, j, win], axis=-1)
                means[fruit] = float(np.nanmean(trial_vals))
            red_consistent = (means["strawberry"] > 0) == (means["watermelon"] > 0)
            green_consistent = (means["cabbage"] > 0) == (means["kiwi"] > 0)
            rows.append(
                {
                    "subject": subject,
                    "channel": channel,
                    "window": variant.window_label,
                    "signal": variant.signal,
                    "strawberry_mean": means["strawberry"],
                    "watermelon_mean": means["watermelon"],
                    "cabbage_mean": means["cabbage"],
                    "kiwi_mean": means["kiwi"],
                    "red_pair_sign_consistent": bool(red_consistent),
                    "green_pair_sign_consistent": bool(green_consistent),
                    "red_minus_green_mean": float(
                        (means["strawberry"] + means["watermelon"])
                        / 2.0
                        - (means["cabbage"] + means["kiwi"]) / 2.0
                    ),
                }
            )
    table = pd.DataFrame(rows)
    stage = out / "stage03_decoding"
    stage.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        stage / f"decoding_direction_consistency_{variant.suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table


def run_spectrum_decoding(
    out: Path,
    csc_table: pd.DataFrame,
    variant: AnalysisVariant,
    n_perms: int = 1000,
    workers: int = 21,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    if csc_table.empty or not csc_table.CSC.any():
        return pd.DataFrame(), {}
    _direction_consistency(out, csc_table, variant)
    rows: list[dict[str, object]] = []
    curves: dict[str, dict[str, Any]] = {}
    for subject, grp in csc_table[csc_table.CSC].groupby("subject"):
        channels = list(grp.channel.astype(str))
        _, time_ms, _ = load_conditions(
            subject, 2, [f"{fruit}_gray" for fruit in GRAY_FRUITS], channels
        )
        t2_feat = _spectrum_features(
            subject,
            2,
            [f"{fruit}_gray" for fruit in GRAY_FRUITS],
            channels,
            time_ms,
            variant,
        )
        t3_feat = _spectrum_features(
            subject, 3, ["red", "green"], channels, time_ms, variant
        )
        for j, channel in enumerate(channels):
            seed = _channel_seed(subject, channel, variant, base=20260805)
            real_t3, null_t3, p_t3 = _fit_binary_cv(
                t3_feat["red"][:, j],
                t3_feat["green"][:, j],
                n_perms,
                seed,
                workers,
            )
            real_t2, null_t2, p_t2 = _fit_cross_fruit(
                {fruit: t2_feat[f"{fruit}_gray"][:, j] for fruit in GRAY_FRUITS},
                n_perms,
                seed + 17,
                workers,
            )
            rows.extend(
                [
                    {
                        "subject": subject,
                        "channel": channel,
                        "window": variant.window_label,
                        "signal": variant.signal,
                        "analysis": "task3_red_green",
                        "accuracy": real_t3,
                        "null95": float(np.nanquantile(null_t3, 0.95)),
                        "p_perm": p_t3,
                        "n_permutations": n_perms,
                    },
                    {
                        "subject": subject,
                        "channel": channel,
                        "window": variant.window_label,
                        "signal": variant.signal,
                        "analysis": "task2_gray_memory_color",
                        "accuracy": real_t2,
                        "null95": float(np.nanquantile(null_t2, 0.95)),
                        "p_perm": p_t2,
                        "n_permutations": n_perms,
                    },
                ]
            )
            curves[f"{subject}|{channel}|task3_red_green|{variant.suffix}"] = {
                "accuracy": real_t3,
                "null95": float(np.nanquantile(null_t3, 0.95)),
            }
            curves[f"{subject}|{channel}|task2_gray_memory_color|{variant.suffix}"] = {
                "accuracy": real_t2,
                "null95": float(np.nanquantile(null_t2, 0.95)),
            }
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["p_fdr_across_electrodes"] = np.nan
        for (window, signal, analysis), grp in summary.groupby(
            ["window", "signal", "analysis"]
        ):
            p_values = grp["p_perm"].to_numpy(dtype=float)
            summary.loc[grp.index, "p_fdr_across_electrodes"] = bh_adjust(p_values)
    stage = out / "stage03_decoding"
    stage.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        stage / f"decoding_summary_{variant.suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _plot_decoding_summary(out, summary, variant)
    return summary, curves


def _plot_decoding_summary(
    out: Path, summary: pd.DataFrame, variant: AnalysisVariant
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    figures = out / "stage03_decoding" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for analysis in sorted(summary.analysis.unique()):
        data = summary[summary.analysis == analysis].sort_values(
            ["subject", "channel"]
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"{s}-{c}" for s, c in zip(data.subject, data.channel)]
        colors = plt.cm.tab10(
            np.asarray([SUBJECTS.index(s) for s in data.subject]) % 10
        )
        ax.bar(np.arange(len(data)), data.accuracy, color=colors, alpha=0.75)
        ax.axhline(0.5, color="#333333", ls="--", lw=1)
        ax.axhline(data.null95.mean(), color="#d62728", ls=":", lw=1.2)
        ax.set_xticks(np.arange(len(data)), labels, rotation=60, fontsize=8)
        ax.set_ylim(0.25, 0.95)
        ax.set_ylabel("Decoding accuracy")
        ax.set_title(
            f"{analysis} ({variant.window_label} ms / {variant.signal_label})\n"
            "red dashed = mean 95th-percentile permutation null"
        )
        fig.tight_layout()
        fig.savefig(
            figures / f"decoding_{analysis}_{variant.suffix}.png", dpi=220
        )
        plt.close(fig)
