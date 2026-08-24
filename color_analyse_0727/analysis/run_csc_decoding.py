"""Run optimized decoding and plots for all CSC electrodes.

Primary variant: 100-400_raw200. The script saves raw real/null curves, per-
electrode cluster-permutation results, fixed-window spectral results, and
descriptive standardized LinearSVC feature-weight summaries.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from analysis.benchmark_one_electrode_optimized import (
    _cross_fruit_curve_optimized,
    _cross_task_curve_optimized,
    _cross_task_fixed_optimized,
    _fit_binary_cv_optimized,
    _fit_curve_cv_optimized,
    _fit_cross_fruit_optimized,
)
from analysis.common import GRAY_FRUITS, AnalysisVariant, load_conditions, natural_key
from analysis.decoding_timeresolved import stft_band_power
from analysis.selection import prepare_signal
from pipeline.spectral_features import (
    FEATURE_BANDS,
    band_power_baseline_z,
    welch_band_power,
    window_mask,
)


TIMEFREQ_ANALYSES = [
    "task3_within_timefreq",
    "task2_cross_fruit_timefreq",
    "task3_to_task2_timefreq",
    "task2_to_task3_timefreq",
]
SPECTRUM_ANALYSES = [
    "task3_within_spectrum",
    "task2_cross_fruit_spectrum",
    "task3_to_task2_spectrum",
    "task2_to_task3_spectrum",
]
ALL_ANALYSES = SPECTRUM_ANALYSES + TIMEFREQ_ANALYSES
FEATURE_LABELS = [f"{lo:g}-{hi:g} Hz" for lo, hi in FEATURE_BANDS]


def _bool_col(values: pd.Series) -> pd.Series:
    return values.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _seed_for(subject: str, channel: str, analysis_index: int, base: int) -> int:
    subject_code = int(str(subject)[-3:])
    channel_code = sum(ord(char) for char in str(channel))
    return int(base + subject_code * 100000 + channel_code * 10 + analysis_index)


def _p_value(real: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    if null.size == 0 or not np.isfinite(real):
        return float("nan")
    return float((1.0 + np.sum(null >= real)) / (null.size + 1.0))


def _cluster_permutation_1d_corrected(
    real: np.ndarray,
    null: np.ndarray,
    times_ms: np.ndarray,
    p_form: float = 0.05,
    min_cluster_ms: float = 20.0,
) -> list[dict[str, float]]:
    """Single-electrode max-cluster-mass permutation test with +1 p correction."""
    real = np.asarray(real, dtype=float)
    null = np.asarray(null, dtype=float)
    times_ms = np.asarray(times_ms, dtype=float)
    if null.ndim != 2 or null.shape[0] == 0 or np.isnan(real).all():
        return []
    threshold = np.nanquantile(null, 1.0 - p_form, axis=0)
    min_len = max(1, int(round(min_cluster_ms / (times_ms[1] - times_ms[0]))))

    def cluster_masses(values: np.ndarray, mask: np.ndarray) -> list[tuple[int, int, float]]:
        output = []
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

    observed = cluster_masses(real, real > threshold)
    null_max = np.zeros(null.shape[0], dtype=float)
    for permutation_index, values in enumerate(null):
        masses = cluster_masses(values, values > threshold)
        if masses:
            null_max[permutation_index] = max(mass for _, _, mass in masses)
    output = []
    for start, end, mass in observed:
        p_value = float((1.0 + np.sum(null_max >= mass)) / (len(null_max) + 1.0))
        output.append({
            "start_ms": float(times_ms[start]),
            "end_ms": float(times_ms[end]),
            "mass": mass,
            "p": p_value,
        })
    return output


def _make_spectral_features(
    raw: dict[str, np.ndarray],
    time_ms: np.ndarray,
    names: list[str],
    variant: AnalysisVariant,
) -> dict[str, np.ndarray]:
    base = window_mask(time_ms, -200.0, 0.0)
    analysis = window_mask(time_ms, 1.0, 1000.0)
    output: dict[str, np.ndarray] = {}
    for name in names:
        base_power = welch_band_power(raw[name][:, :, base])
        analysis_power = welch_band_power(raw[name][:, :, analysis])
        output[name] = band_power_baseline_z(analysis_power, base_power)
    return output


def _prepare_subject(
    subject: str, channels: list[str], variant: AnalysisVariant
) -> dict[str, object]:
    # h5py requires integer channel indices to be increasing; the CSC table is
    # ordered by MNI y for plotting, so restore the project's natural electrode
    # order for data loading and use channel-name lookup afterward.
    channels = sorted([str(channel) for channel in channels], key=natural_key)
    task3_names = ["red", "green"]
    task2_names = [f"{fruit}_gray" for fruit in GRAY_FRUITS]
    raw3, time3, _ = load_conditions(subject, 3, task3_names, channels)
    raw2, time2, _ = load_conditions(subject, 2, task2_names, channels)
    raw3 = {key: prepare_signal(value, time3, variant) for key, value in raw3.items()}
    raw2 = {key: prepare_signal(value, time2, variant) for key, value in raw2.items()}
    spec3 = _make_spectral_features(raw3, time3, task3_names, variant)
    spec2 = _make_spectral_features(raw2, time2, task2_names, variant)

    tf3: dict[str, np.ndarray] = {}
    tf2: dict[str, np.ndarray] = {}
    frame_times = None
    for name in task3_names:
        tf3[name], frame_times = stft_band_power(raw3[name])
    for name in task2_names:
        tf2[name], frame_times2 = stft_band_power(raw2[name])
        if frame_times is not None and not np.allclose(frame_times, frame_times2):
            raise RuntimeError(f"Task2/Task3 STFT grids differ for {subject}")
    if frame_times is None:
        raise RuntimeError(f"No STFT frame grid for {subject}")
    # Keep the original unpadded STFT implementation and restrict the formal
    # plotted/tested interval to 0-800 ms, avoiding nearest-frame extrapolation.
    grid = np.arange(0.0, 800.0 + 1e-9, 10.0)
    frame_indices = np.asarray(
        [int(np.argmin(np.abs(frame_times - time))) for time in grid], dtype=int
    )
    return {
        "spec3": spec3,
        "spec2": spec2,
        "tf3": tf3,
        "tf2": tf2,
        "grid": grid,
        "frame_indices": frame_indices,
        "frame_times": frame_times,
    }


def _scale_features(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)


def _classifier_weights(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    model = LinearSVC(C=1.0, max_iter=10000, dual=False)
    model.fit(x_train, y_train)
    return np.asarray(model.coef_[0], dtype=float)


def _weights_binary_cv(
    x0: np.ndarray, x1: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)])
    n_splits = min(5, int(np.min(np.bincount(y))))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    signed = []
    for train_idx, test_idx in splitter.split(x, y):
        x_train, _ = _scale_features(x[train_idx], x[test_idx])
        signed.append(_classifier_weights(x_train, y[train_idx]))
    signed_array = np.asarray(signed, dtype=float)
    return signed_array.mean(axis=0), np.abs(signed_array).mean(axis=0)


def _weights_cross_fruit(fruits: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    names = ["strawberry", "watermelon", "cabbage", "kiwi"]
    rng = np.random.default_rng(seed)
    n_min = min(fruits[name].shape[0] for name in names)
    arrays = [fruits[name][rng.permutation(fruits[name].shape[0])[:n_min]] for name in names]
    folds = [(0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2)]
    weights = []
    for tr_red, tr_green, _, _ in folds:
        x_train = np.concatenate([arrays[tr_red], arrays[tr_green]], axis=0)
        y_train = np.concatenate([
            np.zeros(len(arrays[tr_red]), dtype=int),
            np.ones(len(arrays[tr_green]), dtype=int),
        ])
        x_train, _ = _scale_features(x_train, x_train)
        weights.append(_classifier_weights(x_train, y_train))
    signed_array = np.asarray(weights, dtype=float)
    return signed_array.mean(axis=0), np.abs(signed_array).mean(axis=0)


def _balanced_cross_task(
    train0: np.ndarray, train1: np.ndarray, test0: np.ndarray, test1: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_train = min(len(train0), len(train1))
    n_test = min(len(test0), len(test1))
    train0 = train0[rng.permutation(len(train0))[:n_train]]
    train1 = train1[rng.permutation(len(train1))[:n_train]]
    test0 = test0[rng.permutation(len(test0))[:n_test]]
    test1 = test1[rng.permutation(len(test1))[:n_test]]
    x_train = np.concatenate([train0, train1], axis=0)
    y_train = np.concatenate([np.zeros(n_train, dtype=int), np.ones(n_train, dtype=int)])
    x_test = np.concatenate([test0, test1], axis=0)
    y_test = np.concatenate([np.zeros(n_test, dtype=int), np.ones(n_test, dtype=int)])
    return x_train, y_train, x_test, y_test


def _weights_cross_task(
    train0: np.ndarray, train1: np.ndarray, test0: np.ndarray, test1: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    x_train, y_train, x_test, _ = _balanced_cross_task(train0, train1, test0, test1, seed)
    x_train, _ = _scale_features(x_train, x_test)
    weights = _classifier_weights(x_train, y_train)
    return weights, np.abs(weights)


def _feature_rows(
    subject: str,
    channel: str,
    analysis: str,
    signed: np.ndarray,
    absolute: np.ndarray,
) -> list[dict[str, object]]:
    signed = np.asarray(signed, dtype=float)
    absolute = np.asarray(absolute, dtype=float)
    total = float(np.nansum(absolute))
    relative = absolute / total if total > 0 else np.full_like(absolute, np.nan)
    rows = []
    for index, (band, signed_value, abs_value, rel_value) in enumerate(
        zip(FEATURE_LABELS, signed, absolute, relative)
    ):
        rows.append({
            "subject": subject,
            "channel": channel,
            "analysis": analysis,
            "feature_index": index,
            "feature": band,
            "band_low_hz": FEATURE_BANDS[index][0],
            "band_high_hz": FEATURE_BANDS[index][1],
            "signed_weight": float(signed_value),
            "mean_abs_weight": float(abs_value),
            "relative_abs_weight": float(rel_value),
        })
    return rows


def _save_electrode_npz(
    path: Path,
    metadata: dict[str, object],
    outputs: dict[str, tuple[np.ndarray, np.ndarray]],
    grid: np.ndarray,
) -> None:
    payload: dict[str, object] = {"grid_ms": grid}
    payload.update(metadata)
    for name, (real, null) in outputs.items():
        payload[f"{name}_real"] = np.asarray(real)
        payload[f"{name}_null"] = np.asarray(null)
    np.savez_compressed(path, **payload)


def _summary_row(
    metadata: dict[str, object], analysis: str, real: np.ndarray, null: np.ndarray,
    grid: np.ndarray | None = None, clusters: list[dict[str, float]] | None = None,
) -> dict[str, object]:
    real_array = np.asarray(real, dtype=float)
    null_array = np.asarray(null, dtype=float)
    clusters = clusters or []
    if real_array.ndim == 0:
        real_mean = float(real_array)
        real_peak = float(real_array)
        peak_time = np.nan
        null_mean = float(np.nanmean(null_array)) if null_array.size else np.nan
        p_value = _p_value(real_mean, null_array.ravel())
    else:
        real_mean = float(np.nanmean(real_array))
        peak_index = int(np.nanargmax(real_array))
        real_peak = float(real_array[peak_index])
        peak_time = float(grid[peak_index]) if grid is not None else np.nan
        null_mean = float(np.nanmean(null_array)) if null_array.size else np.nan
        pointwise_p = np.mean(null_array >= real_array[None, :], axis=0) if null_array.size else np.array([])
        p_value = float(np.nanmin(pointwise_p)) if pointwise_p.size else np.nan
    return {
        **metadata,
        "analysis": analysis,
        "real_mean_accuracy": real_mean,
        "real_peak_accuracy": real_peak,
        "peak_time_ms": peak_time,
        "null_mean_accuracy": null_mean,
        "pointwise_or_fixed_p_min": p_value,
        "n_clusters_p_le_0.05": int(sum(float(cluster["p"]) <= 0.05 for cluster in clusters)),
    }


def _plot_individual(
    record: dict[str, object], curves: dict[str, tuple[np.ndarray, list[dict[str, float]]]],
    grid: np.ndarray, out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, analysis in zip(axes, TIMEFREQ_ANALYSES):
        real, clusters = curves[analysis]
        ax.plot(grid, real, color="#1f4e79", linewidth=1.8)
        ax.axhline(0.5, color="0.45", linestyle="--", linewidth=0.8)
        for cluster in clusters:
            if float(cluster["p"]) <= 0.05:
                ax.axvspan(cluster["start_ms"], cluster["end_ms"], color="#f4a261", alpha=0.3)
        ax.set_title(analysis.replace("_", " "))
        ax.set_xlim(float(grid[0]), float(grid[-1]))
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)
    fig.supxlabel("Time (ms)")
    fig.supylabel("Accuracy")
    fig.suptitle(
        f"{record['subject']}-{record['channel']} | MNI y={float(record['mni_y']):.2f} | "
        "single-electrode cluster permutation (100 permutations)",
        y=0.995,
    )
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.95))
    fig.savefig(out_dir / f"{_safe_name(record['subject'])}_{_safe_name(record['channel'])}_acc_time_cluster.png", dpi=180)
    plt.close(fig)


def _plot_combined(
    records: list[dict[str, object]],
    curve_store: dict[tuple[str, str], dict[str, tuple[np.ndarray, list[dict[str, float]]]]],
    grid: np.ndarray,
    out_path: Path,
) -> None:
    ordered = sorted(records, key=lambda row: float(row["mni_y"]))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    n_electrodes = max(1, len(ordered))
    cmap = plt.get_cmap("coolwarm")
    for index, record in enumerate(ordered):
        position = index / max(1, n_electrodes - 1)
        color = cmap(position)
        linewidth = 0.8 + 2.2 * position
        key = (str(record["subject"]), str(record["channel"]))
        for ax, analysis in zip(axes, TIMEFREQ_ANALYSES):
            ax.plot(
                grid,
                curve_store[key][analysis][0],
                color=color,
                linewidth=linewidth,
                alpha=0.78,
            )
    for ax, analysis in zip(axes, TIMEFREQ_ANALYSES):
        curves = np.asarray([
            curve_store[(str(row["subject"]), str(row["channel"]))][analysis][0]
            for row in ordered
        ])
        ax.plot(grid, np.nanmean(curves, axis=0), color="black", linewidth=3.0, label="CSC mean")
        ax.axhline(0.5, color="0.45", linestyle="--", linewidth=0.8)
        ax.set_title(analysis.replace("_", " "))
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="upper right")
    fig.supxlabel("Time (ms)")
    fig.supylabel("Accuracy")
    fig.suptitle("All CSC electrodes: posterior→anterior blue→red, thin→thick; no permutation shading", y=0.995)
    norm = Normalize(vmin=float(min(row["mni_y"] for row in ordered)), vmax=float(max(row["mni_y"] for row in ordered)))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("MNI y: posterior → anterior")
    fig.tight_layout(rect=(0.03, 0.03, 0.95, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_saved_results(result_dir: Path, suffix: str) -> None:
    stage = result_dir / "stage07_csc_decoding"
    fig_dir = stage / "figures"
    electrode_fig_dir = fig_dir / "individual_electrodes"
    fig_dir.mkdir(parents=True, exist_ok=True)
    electrode_fig_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(stage / "csc_electrode_order_posterior_to_anterior.csv").to_dict("records")
    curve_df = pd.read_csv(stage / "csc_decoding_time_curves_100perm.csv")
    cluster_df = pd.read_csv(stage / "csc_decoding_cluster_results_100perm.csv")
    grid = np.sort(curve_df["time_ms"].unique().astype(float))
    curve_store: dict[tuple[str, str], dict[str, tuple[np.ndarray, list[dict[str, float]]]]] = {}
    for record in records:
        subject = str(record["subject"])
        channel = str(record["channel"])
        curve_store[(subject, channel)] = {}
        for analysis in TIMEFREQ_ANALYSES:
            sub = curve_df[
                (curve_df["subject"].astype(str) == subject)
                & (curve_df["channel"].astype(str) == channel)
                & (curve_df["analysis"] == analysis)
            ].sort_values("time_ms")
            clusters = cluster_df[
                (cluster_df["subject"].astype(str) == subject)
                & (cluster_df["channel"].astype(str) == channel)
                & (cluster_df["analysis"] == analysis)
            ].to_dict("records")
            curve_store[(subject, channel)][analysis] = (
                sub["accuracy"].to_numpy(dtype=float), clusters
            )
    for record in records:
        key = (str(record["subject"]), str(record["channel"]))
        _plot_individual(record, curve_store[key], grid, electrode_fig_dir)
    _plot_combined(records, curve_store, grid, fig_dir / "csc_all_electrodes_acc_time_combined_no_permutation.png")
    feature_df = pd.read_csv(stage / "csc_decoding_feature_dominance_100perm.csv")
    _plot_feature_dominance(feature_df, fig_dir, suffix)


def _plot_feature_dominance(
    feature_df: pd.DataFrame, fig_dir: Path, suffix: str, label: str | None = None
) -> None:
    if label is None:
        label = "CSC" if "electrode_set" not in feature_df.columns else "/".join(
            sorted(feature_df["electrode_set"].dropna().astype(str).unique())
        )
    group = (
        feature_df.groupby(["analysis", "feature"], as_index=False)["relative_abs_weight"]
        .mean()
        .pivot(index="analysis", columns="feature", values="relative_abs_weight")
    )
    group = group.reindex(columns=FEATURE_LABELS)
    fig, ax = plt.subplots(figsize=(16, 5.5))
    im = ax.imshow(group.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0.0)
    ax.set_xticks(np.arange(len(FEATURE_LABELS)), FEATURE_LABELS, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(group.index)), [str(value).replace("_", " ") for value in group.index])
    ax.set_title(f"Descriptive spectral feature dominance across {label} electrodes")
    ax.set_xlabel("Frequency feature")
    ax.set_ylabel("Decoding analysis")
    fig.colorbar(im, ax=ax, label="Mean relative absolute SVM weight")
    fig.tight_layout()
    fig.savefig(fig_dir / f"csc_feature_dominance_group_{suffix}.png", dpi=200)
    plt.close(fig)

    feature_df = feature_df.copy()
    feature_df["electrode_analysis"] = feature_df["subject"].astype(str) + "-" + feature_df["channel"].astype(str) + " | " + feature_df["analysis"].astype(str)
    electrode = feature_df.pivot(index="electrode_analysis", columns="feature", values="relative_abs_weight").reindex(columns=FEATURE_LABELS)
    fig_height = max(6.0, 0.28 * len(electrode))
    fig, ax = plt.subplots(figsize=(16, fig_height))
    im = ax.imshow(electrode.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0.0)
    ax.set_xticks(np.arange(len(FEATURE_LABELS)), FEATURE_LABELS, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(electrode.index)), electrode.index, fontsize=7)
    ax.set_title("Feature dominance by CSC electrode and decoding analysis")
    ax.set_xlabel("Frequency feature")
    ax.set_ylabel("Electrode | analysis")
    fig.colorbar(im, ax=ax, label="Relative absolute SVM weight")
    fig.tight_layout()
    fig.savefig(fig_dir / f"csc_feature_dominance_electrode_{suffix}.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out = Path(args.result_dir)
    variant = AnalysisVariant((100.0, 400.0), args.signal)
    if args.plots_only:
        _plot_saved_results(out, variant.suffix)
        print(f"plots regenerated from {out / 'stage07_csc_decoding'}", flush=True)
        return
    stage = out / "stage07_csc_decoding"
    fig_dir = stage / "figures"
    electrode_fig_dir = fig_dir / "individual_electrodes"
    stage.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    electrode_fig_dir.mkdir(parents=True, exist_ok=True)

    selection_variant = AnalysisVariant((100.0, 400.0), args.selection_signal or args.signal)
    csc_path = out / "stage01_selection" / f"electrode_sets_and_csc_{selection_variant.suffix}.csv"
    csc = pd.read_csv(csc_path)
    csc["CSC_bool"] = _bool_col(csc["CSC"])
    csc = csc[csc["CSC_bool"]].copy()
    csc["mni_y"] = pd.to_numeric(csc["mni_y"], errors="coerce")
    csc = csc.sort_values(["mni_y", "subject", "channel"]).reset_index(drop=True)
    if csc.empty:
        raise RuntimeError(f"No CSC electrodes found in {csc_path}")

    records = csc[["subject", "channel", "mni_x", "mni_y", "mni_z", "roi"]].to_dict("records")
    grid: np.ndarray | None = None
    curve_store: dict[tuple[str, str], dict[str, tuple[np.ndarray, list[dict[str, float]]]]] = {}
    summary_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for subject, subject_table in csc.groupby("subject", sort=False):
        channels = subject_table.channel.astype(str).tolist()
        prepared = _prepare_subject(str(subject), channels, variant)
        spec3 = prepared["spec3"]
        spec2 = prepared["spec2"]
        tf3 = prepared["tf3"]
        tf2 = prepared["tf2"]
        subject_grid = np.asarray(prepared["grid"], dtype=float)
        frame_indices = np.asarray(prepared["frame_indices"], dtype=int)
        if grid is None:
            grid = subject_grid
        elif not np.array_equal(grid, subject_grid):
            raise RuntimeError("Subjects do not share the same decoding grid")

        for local_index, record in subject_table.reset_index(drop=True).iterrows():
            channel = str(record["channel"])
            j = channels.index(channel)
            metadata = {
                "subject": str(subject),
                "channel": channel,
                "mni_x": float(record["mni_x"]),
                "mni_y": float(record["mni_y"]),
                "mni_z": float(record["mni_z"]),
                "roi": str(record["roi"]),
                "window": variant.window_label,
                "signal": variant.signal,
            }
            spec3_red = spec3["red"][:, j]
            spec3_green = spec3["green"][:, j]
            spec2_fruits = {fruit: spec2[f"{fruit}_gray"][:, j] for fruit in GRAY_FRUITS}
            spec2_red = np.concatenate([spec2_fruits["strawberry"], spec2_fruits["watermelon"]])
            spec2_green = np.concatenate([spec2_fruits["cabbage"], spec2_fruits["kiwi"]])

            t3_red = tf3["red"][:, j, :, :][:, :, frame_indices]
            t3_green = tf3["green"][:, j, :, :][:, :, frame_indices]
            tf2_fruits = {fruit: tf2[f"{fruit}_gray"] for fruit in GRAY_FRUITS}
            tf2_red = np.concatenate([
                tf2["strawberry_gray"][:, j, :, :], tf2["watermelon_gray"][:, j, :, :]
            ], axis=0)[:, :, frame_indices]
            tf2_green = np.concatenate([
                tf2["cabbage_gray"][:, j, :, :], tf2["kiwi_gray"][:, j, :, :]
            ], axis=0)[:, :, frame_indices]

            outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            outputs["task3_within_spectrum"] = _fit_binary_cv_optimized(
                spec3_red, spec3_green, args.perms, _seed_for(str(subject), channel, 0, args.seed), args.workers
            )
            outputs["task2_cross_fruit_spectrum"] = _fit_cross_fruit_optimized(
                spec2_fruits, args.perms, _seed_for(str(subject), channel, 1, args.seed), args.workers
            )
            outputs["task3_to_task2_spectrum"] = _cross_task_fixed_optimized(
                spec3_red, spec3_green, spec2_red, spec2_green,
                args.perms, _seed_for(str(subject), channel, 2, args.seed), args.workers,
            )
            outputs["task2_to_task3_spectrum"] = _cross_task_fixed_optimized(
                spec2_red, spec2_green, spec3_red, spec3_green,
                args.perms, _seed_for(str(subject), channel, 3, args.seed), args.workers,
            )
            outputs["task3_within_timefreq"] = _fit_curve_cv_optimized(
                t3_red, t3_green, args.perms, _seed_for(str(subject), channel, 4, args.seed), args.workers
            )
            outputs["task2_cross_fruit_timefreq"] = _cross_fruit_curve_optimized(
                tf2_fruits, frame_indices, j, args.perms,
                _seed_for(str(subject), channel, 5, args.seed), args.workers,
            )
            outputs["task3_to_task2_timefreq"] = _cross_task_curve_optimized(
                t3_red, t3_green, tf2_red, tf2_green,
                args.perms, _seed_for(str(subject), channel, 6, args.seed), args.workers,
            )
            outputs["task2_to_task3_timefreq"] = _cross_task_curve_optimized(
                tf2_red, tf2_green, t3_red, t3_green,
                args.perms, _seed_for(str(subject), channel, 7, args.seed), args.workers,
            )

            electrode_curves: dict[str, tuple[np.ndarray, list[dict[str, float]]]] = {}
            for analysis, (real, null) in outputs.items():
                clusters = []
                if analysis in TIMEFREQ_ANALYSES:
                    clusters = _cluster_permutation_1d_corrected(
                        np.asarray(real), np.asarray(null), grid
                    )
                    electrode_curves[analysis] = (np.asarray(real), clusters)
                    for time_index, time_ms in enumerate(grid):
                        curve_rows.append({
                            **metadata,
                            "analysis": analysis,
                            "time_ms": float(time_ms),
                            "accuracy": float(real[time_index]),
                            "null_mean_accuracy": float(np.nanmean(null[:, time_index])),
                            "null_q95_accuracy": float(np.nanquantile(null[:, time_index], 0.95)),
                        })
                    for cluster in clusters:
                        cluster_rows.append({**metadata, "analysis": analysis, **cluster, "n_permutations": args.perms})
                summary_rows.append(_summary_row(metadata, analysis, real, null, grid if analysis in TIMEFREQ_ANALYSES else None, clusters))

            curve_store[(str(subject), channel)] = electrode_curves
            electrode_npz = stage / "electrode_npz" / f"{_safe_name(subject)}_{_safe_name(channel)}_100perm.npz"
            electrode_npz.parent.mkdir(parents=True, exist_ok=True)
            _save_electrode_npz(electrode_npz, metadata, outputs, grid)

            # Descriptive spectral feature dominance, based on real labels only.
            feature_rows.extend(_feature_rows(str(subject), channel, "task3_within_spectrum", *_weights_binary_cv(
                spec3_red, spec3_green, _seed_for(str(subject), channel, 20, args.seed)
            )))
            feature_rows.extend(_feature_rows(str(subject), channel, "task2_cross_fruit_spectrum", *_weights_cross_fruit(
                spec2_fruits, _seed_for(str(subject), channel, 21, args.seed)
            )))
            feature_rows.extend(_feature_rows(str(subject), channel, "task3_to_task2_spectrum", *_weights_cross_task(
                spec3_red, spec3_green, spec2_red, spec2_green,
                _seed_for(str(subject), channel, 22, args.seed)
            )))
            feature_rows.extend(_feature_rows(str(subject), channel, "task2_to_task3_spectrum", *_weights_cross_task(
                spec2_red, spec2_green, spec3_red, spec3_green,
                _seed_for(str(subject), channel, 23, args.seed)
            )))

            print(f"completed {subject}-{channel} ({len(summary_rows) // len(ALL_ANALYSES)}/{len(records)})", flush=True)

    if grid is None:
        raise RuntimeError("No CSC results were generated")
    summary_df = pd.DataFrame(summary_rows)
    cluster_df = pd.DataFrame(cluster_rows)
    curve_df = pd.DataFrame(curve_rows)
    feature_df = pd.DataFrame(feature_rows)
    summary_df.to_csv(stage / "csc_decoding_summary_100perm.csv", index=False, encoding="utf-8-sig")
    cluster_columns = list(csc.columns[:0]) + [
        "subject", "channel", "mni_x", "mni_y", "mni_z", "roi", "window", "signal",
        "analysis", "start_ms", "end_ms", "mass", "p", "n_permutations",
    ]
    cluster_df.reindex(columns=cluster_columns).to_csv(stage / "csc_decoding_cluster_results_100perm.csv", index=False, encoding="utf-8-sig")
    curve_df.to_csv(stage / "csc_decoding_time_curves_100perm.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(stage / "csc_decoding_feature_dominance_100perm.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(stage / "csc_electrode_order_posterior_to_anterior.csv", index=False, encoding="utf-8-sig")

    for record in records:
        key = (str(record["subject"]), str(record["channel"]))
        _plot_individual(record, curve_store[key], grid, electrode_fig_dir)
    _plot_combined(records, curve_store, grid, fig_dir / "csc_all_electrodes_acc_time_combined_no_permutation.png")
    _plot_feature_dominance(feature_df, fig_dir, variant.suffix)

    metadata = {
        "variant": variant.suffix,
        "window_ms": [100.0, 400.0],
        "signal": variant.signal,
        "selection_variant": selection_variant.suffix,
        "n_csc_electrodes": len(records),
        "n_permutations": args.perms,
        "workers_per_electrode": args.workers,
        "time_grid_ms": [float(grid[0]), float(grid[-1]), 10.0],
        "cluster_method": "single-electrode 1D max-cluster-mass permutation; p_form=0.05; min_cluster=20 ms; +1 p correction",
        "feature_dominance": "descriptive mean absolute standardized LinearSVC coefficient across real-label folds; no feature-level permutation test",
        "posterior_anterior_order": "ascending MNI y; blue to red; line width thin to thick",
        "notes": [
            "The optimized solver uses LinearSVC(dual=False) and reuses training-fold standardization across permutations.",
            "This batch uses the original unpadded 256 ms STFT window and 10 ms hop; formal time-resolved curves/tests are limited to 0-800 ms.",
            "Task2 time-resolved cross-fruit uses the corrected feature-axis implementation in the optimized benchmark helper.",
            "The decoding input uses the selected signal variant; this run keeps the prior lf30-derived CSC membership for a direct signal-source comparison.",
        ],
    }
    (stage / "csc_decoding_parameters_100perm.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("result/final_analysis_seeg_20260806_corrected"),
    )
    parser.add_argument("--perms", type=int, default=100)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--signal", choices=("lf30", "raw200"), default="raw200")
    parser.add_argument(
        "--selection-signal",
        choices=("lf30", "raw200"),
        default=None,
        help="Signal variant used only to select the electrode set; defaults to --signal.",
    )
    parser.add_argument("--plots-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
