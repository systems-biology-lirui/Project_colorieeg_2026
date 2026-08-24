"""Run the optimized single-electrode decoding suite on Norm1 S1 and S2.

The full 8-analysis suite is saved for every S1/S2 electrode. Figures focus on
Task 2 memory-color decoding (leave-one-fruit-pair-out) and relate the earliest
significant single-electrode cluster latency to MNI coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

from analysis.benchmark_one_electrode_optimized import (
    _cross_fruit_curve_optimized,
    _cross_task_curve_optimized,
    _cross_task_fixed_optimized,
    _fit_binary_cv_optimized,
    _fit_curve_cv_optimized,
    _fit_cross_fruit_optimized,
)
from analysis.common import GRAY_FRUITS, AnalysisVariant, natural_key
from analysis.run_csc_decoding import (
    ALL_ANALYSES,
    FEATURE_LABELS,
    SPECTRUM_ANALYSES,
    TIMEFREQ_ANALYSES,
    _cluster_permutation_1d_corrected,
    _feature_rows,
    _plot_feature_dominance,
    _prepare_subject,
    _safe_name,
    _seed_for,
    _summary_row,
)


MEMORY_ANALYSIS = "task2_cross_fruit_timefreq"


def _bool_col(values: pd.Series) -> pd.Series:
    return values.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def _metadata(
    row: pd.Series,
    electrode_set: str | None = None,
    signal: str = "raw200",
) -> dict[str, object]:
    output = {
        "subject": str(row["subject"]),
        "channel": str(row["channel"]),
        "mni_x": float(row["mni_x"]),
        "mni_y": float(row["mni_y"]),
        "mni_z": float(row["mni_z"]),
        "roi": str(row["roi"]),
        "window": "100-400",
        "signal": signal,
    }
    if electrode_set is not None:
        output["electrode_set"] = electrode_set
    return output


def _save_npz(path: Path, metadata: dict[str, object], outputs: dict[str, tuple[np.ndarray, np.ndarray]], grid: np.ndarray) -> None:
    payload: dict[str, object] = {"grid_ms": grid}
    payload.update(metadata)
    for name, (real, null) in outputs.items():
        payload[f"{name}_real"] = np.asarray(real)
        payload[f"{name}_null"] = np.asarray(null)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _curve_rows(metadata: dict[str, object], analysis: str, real: np.ndarray, null: np.ndarray, grid: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for index, time_ms in enumerate(grid):
        rows.append({
            **metadata,
            "analysis": analysis,
            "time_ms": float(time_ms),
            "accuracy": float(real[index]),
            "null_mean_accuracy": float(np.nanmean(null[:, index])),
            "null_q95_accuracy": float(np.nanquantile(null[:, index], 0.95)),
        })
    return rows


def _set_memory_plot(
    electrode_set: str,
    set_records: list[dict[str, object]],
    curve_store: dict[tuple[str, str], np.ndarray],
    sig_latency: pd.DataFrame,
    grid: np.ndarray,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(17, 11))
    layout = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.30)
    curve_ax = fig.add_subplot(layout[0, :2])
    xy_ax = fig.add_subplot(layout[0, 2])
    coordinate_axes = [fig.add_subplot(layout[1, index]) for index in range(3)]

    sig_keys = {
        (str(row["subject"]), str(row["channel"])) for _, row in sig_latency.iterrows()
    }
    onset_values = sig_latency["earliest_cluster_onset_ms"].to_numpy(dtype=float) if not sig_latency.empty else np.array([])
    onset_min = float(np.nanmin(onset_values)) if onset_values.size else 0.0
    onset_max = float(np.nanmax(onset_values)) if onset_values.size else 800.0
    onset_norm = Normalize(onset_min, onset_max if onset_max > onset_min else onset_min + 1.0)
    onset_cmap = plt.get_cmap("viridis")

    ordered = sorted(set_records, key=lambda row: float(row["mni_y"]))
    for record in ordered:
        key = (str(record["subject"]), str(record["channel"]))
        if key in sig_keys:
            onset = float(sig_latency.loc[
                (sig_latency["subject"].astype(str) == key[0]) &
                (sig_latency["channel"].astype(str) == key[1]),
                "earliest_cluster_onset_ms",
            ].iloc[0])
            color = onset_cmap(onset_norm(onset))
            linewidth = 2.0
            alpha = 0.9
        else:
            color = "#b9b9b9"
            linewidth = 0.65
            alpha = 0.45
        curve_ax.plot(grid, curve_store[key], color=color, linewidth=linewidth, alpha=alpha)
    curves = np.asarray([curve_store[(str(row["subject"]), str(row["channel"]))] for row in ordered])
    curve_ax.plot(grid, np.nanmean(curves, axis=0), color="black", linewidth=3, label=f"{electrode_set} mean")
    curve_ax.axhline(0.5, color="0.35", linestyle="--", linewidth=0.8, label="chance")
    curve_ax.set_title(f"{electrode_set}: Task2 memory-color cross-fruit decoding")
    curve_ax.set_xlabel("Time (ms)")
    curve_ax.set_ylabel("Accuracy")
    curve_ax.set_xlim(float(grid[0]), float(grid[-1]))
    curve_ax.set_ylim(0, 1)
    curve_ax.grid(alpha=0.2)
    curve_ax.legend(frameon=False, loc="upper right")

    all_x = np.asarray([float(row["mni_x"]) for row in ordered])
    all_y = np.asarray([float(row["mni_y"]) for row in ordered])
    xy_ax.scatter(all_x, all_y, color="#c7c7c7", edgecolor="white", s=32, label="all electrodes")
    if not sig_latency.empty:
        xy_ax.scatter(
            sig_latency["mni_x"], sig_latency["mni_y"],
            c=sig_latency["earliest_cluster_onset_ms"], cmap=onset_cmap, norm=onset_norm,
            s=45 + 5 * sig_latency["cluster_duration_ms"], edgecolor="black", linewidth=0.7,
            label="significant cluster",
        )
        for _, row in sig_latency.iterrows():
            xy_ax.annotate(f"{row['subject']}-{row['channel']}", (row["mni_x"], row["mni_y"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    xy_ax.axhline(0, color="0.7", linewidth=0.6)
    xy_ax.axvline(0, color="0.7", linewidth=0.6)
    xy_ax.set_title("MNI location of significant electrodes")
    xy_ax.set_xlabel("MNI x")
    xy_ax.set_ylabel("MNI y")
    xy_ax.grid(alpha=0.2)
    xy_ax.legend(frameon=False, fontsize=8, loc="best")

    for ax, coordinate, label in zip(coordinate_axes, ["mni_x", "mni_y", "mni_z"], ["MNI x", "MNI y", "MNI z"]):
        if sig_latency.empty:
            ax.text(0.5, 0.5, "No significant electrodes", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Latency vs {label}")
            continue
        x = sig_latency[coordinate].to_numpy(dtype=float)
        y = sig_latency["earliest_cluster_onset_ms"].to_numpy(dtype=float)
        ax.scatter(x, y, c=y, cmap=onset_cmap, norm=onset_norm, s=55, edgecolor="black", linewidth=0.6)
        for _, row in sig_latency.iterrows():
            ax.annotate(f"{row['subject']}-{row['channel']}", (row[coordinate], row["earliest_cluster_onset_ms"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        if len(x) >= 3 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
            rho, p_value = spearmanr(x, y)
            annotation = f"Spearman ρ={rho:.2f}, p={p_value:.3f}"
        else:
            annotation = "insufficient significant electrodes"
        ax.text(0.04, 0.95, annotation, transform=ax.transAxes, va="top", fontsize=8)
        ax.set_title(f"Earliest significant latency vs {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Earliest cluster onset (ms)")
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{electrode_set}: memory-color significant cluster latency and coordinates\n"
        "gray curves = nonsignificant; colored curves/points = p≤0.05 single-electrode cluster",
        y=0.995,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    result_dir = Path(args.result_dir)
    stage = result_dir / "stage08_s1_s2_single_electrode_decoding"
    figure_dir = stage / "figures"
    stage.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    variant = AnalysisVariant((100.0, 400.0), args.signal)
    selection_variant = AnalysisVariant((100.0, 400.0), args.selection_signal or args.signal)
    selection_path = result_dir / "stage01_selection" / f"electrode_sets_and_csc_{selection_variant.suffix}.csv"
    selection = pd.read_csv(selection_path)
    selection["S1_bool"] = _bool_col(selection["strategy1"])
    selection["S2_bool"] = _bool_col(selection["strategy2"])
    set_tables = {
        "S1": selection[selection["S1_bool"]].copy(),
        "S2": selection[selection["S2_bool"]].copy(),
    }
    for name in set_tables:
        set_tables[name] = set_tables[name].sort_values(["subject", "channel"]).reset_index(drop=True)
        set_tables[name].to_csv(stage / f"{name.lower()}_electrodes_{variant.suffix}.csv", index=False, encoding="utf-8-sig")

    unique_rows = pd.concat([set_tables["S1"], set_tables["S2"]], ignore_index=True).drop_duplicates(["subject", "channel"])
    unique_by_subject = {str(subject): group.copy() for subject, group in unique_rows.groupby("subject")}
    grid: np.ndarray | None = None
    cache: dict[tuple[str, str], dict[str, object]] = {}
    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for subject, subject_rows in unique_by_subject.items():
        channels = sorted(subject_rows.channel.astype(str).tolist(), key=natural_key)
        prepared = _prepare_subject(subject, channels, variant)
        spec3 = prepared["spec3"]
        spec2 = prepared["spec2"]
        tf3 = prepared["tf3"]
        tf2 = prepared["tf2"]
        subject_grid = np.asarray(prepared["grid"], dtype=float)
        frame_indices = np.asarray(prepared["frame_indices"], dtype=int)
        if grid is None:
            grid = subject_grid
        elif not np.array_equal(grid, subject_grid):
            raise RuntimeError("S1/S2 subjects do not share the same time grid")

        for _, row in subject_rows.iterrows():
            channel = str(row["channel"])
            j = channels.index(channel)
            key = (subject, channel)
            if key in cache:
                continue
            metadata = _metadata(row, signal=variant.signal)
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
            outputs = {
                "task3_within_spectrum": _fit_binary_cv_optimized(spec3_red, spec3_green, args.perms, _seed_for(subject, channel, 0, args.seed), args.workers),
                "task2_cross_fruit_spectrum": _fit_cross_fruit_optimized(spec2_fruits, args.perms, _seed_for(subject, channel, 1, args.seed), args.workers),
                "task3_to_task2_spectrum": _cross_task_fixed_optimized(spec3_red, spec3_green, spec2_red, spec2_green, args.perms, _seed_for(subject, channel, 2, args.seed), args.workers),
                "task2_to_task3_spectrum": _cross_task_fixed_optimized(spec2_red, spec2_green, spec3_red, spec3_green, args.perms, _seed_for(subject, channel, 3, args.seed), args.workers),
                "task3_within_timefreq": _fit_curve_cv_optimized(t3_red, t3_green, args.perms, _seed_for(subject, channel, 4, args.seed), args.workers),
                "task2_cross_fruit_timefreq": _cross_fruit_curve_optimized(tf2_fruits, frame_indices, j, args.perms, _seed_for(subject, channel, 5, args.seed), args.workers),
                "task3_to_task2_timefreq": _cross_task_curve_optimized(t3_red, t3_green, tf2_red, tf2_green, args.perms, _seed_for(subject, channel, 6, args.seed), args.workers),
                "task2_to_task3_timefreq": _cross_task_curve_optimized(tf2_red, tf2_green, t3_red, t3_green, args.perms, _seed_for(subject, channel, 7, args.seed), args.workers),
            }
            curve_data: dict[str, tuple[np.ndarray, list[dict[str, float]]]] = {}
            for analysis, (real, null) in outputs.items():
                clusters = []
                if analysis in TIMEFREQ_ANALYSES:
                    clusters = _cluster_permutation_1d_corrected(real, null, grid)
                curve_data[analysis] = (np.asarray(real), clusters)
            cache[key] = {"metadata": metadata, "outputs": outputs, "curves": curve_data}
            _save_npz(stage / "electrode_npz" / f"{_safe_name(subject)}_{_safe_name(channel)}_100perm.npz", metadata, outputs, grid)
            feature_rows.extend(_feature_rows(subject, channel, "task3_within_spectrum", *_weights_binary_cv_for_feature(spec3_red, spec3_green, _seed_for(subject, channel, 20, args.seed))))
            feature_rows.extend(_feature_rows(subject, channel, "task2_cross_fruit_spectrum", *_weights_cross_fruit_for_feature(spec2_fruits, _seed_for(subject, channel, 21, args.seed))))
            feature_rows.extend(_feature_rows(subject, channel, "task3_to_task2_spectrum", *_weights_cross_task_for_feature(spec3_red, spec3_green, spec2_red, spec2_green, _seed_for(subject, channel, 22, args.seed))))
            feature_rows.extend(_feature_rows(subject, channel, "task2_to_task3_spectrum", *_weights_cross_task_for_feature(spec2_red, spec2_green, spec3_red, spec3_green, _seed_for(subject, channel, 23, args.seed))))
            print(f"completed {subject}-{channel} ({len(cache)}/{len(unique_rows)})", flush=True)

    if grid is None:
        raise RuntimeError("No S1/S2 electrodes found")

    for electrode_set, table in set_tables.items():
        for _, row in table.iterrows():
            subject, channel = str(row["subject"]), str(row["channel"])
            key = (subject, channel)
            base_metadata = _metadata(row, electrode_set, signal=variant.signal)
            cached = cache[key]
            outputs = cached["outputs"]
            curves = cached["curves"]
            for analysis, (real, null) in outputs.items():
                clusters = curves[analysis][1]
                summary_rows.append(_summary_row(base_metadata, analysis, real, null, grid if analysis in TIMEFREQ_ANALYSES else None, clusters))
                if analysis in TIMEFREQ_ANALYSES:
                    curve_rows.extend(_curve_rows(base_metadata, analysis, real, null, grid))
                    for cluster in clusters:
                        cluster_rows.append({**base_metadata, "analysis": analysis, **cluster, "n_permutations": args.perms})

    summary_df = pd.DataFrame(summary_rows)
    curves_df = pd.DataFrame(curve_rows)
    cluster_columns = [
        "electrode_set", "subject", "channel", "mni_x", "mni_y", "mni_z", "roi",
        "window", "signal", "analysis", "start_ms", "end_ms", "mass", "p",
        "n_permutations",
    ]
    clusters_df = pd.DataFrame(cluster_rows, columns=cluster_columns)
    feature_base_df = pd.DataFrame(feature_rows)
    feature_sets = []
    for electrode_set, table in set_tables.items():
        keys = set(zip(table.subject.astype(str), table.channel.astype(str)))
        mask = feature_base_df.apply(
            lambda row: (str(row["subject"]), str(row["channel"])) in keys,
            axis=1,
        )
        feature_sets.append(feature_base_df[mask].assign(electrode_set=electrode_set))
    feature_df = pd.concat(feature_sets, ignore_index=True)
    summary_df.to_csv(stage / "s1s2_decoding_summary_100perm.csv", index=False, encoding="utf-8-sig")
    curves_df.to_csv(stage / "s1s2_decoding_time_curves_100perm.csv", index=False, encoding="utf-8-sig")
    clusters_df.to_csv(stage / "s1s2_decoding_cluster_results_100perm.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(stage / "s1s2_decoding_feature_dominance_100perm.csv", index=False, encoding="utf-8-sig")

    latency_rows = []
    for electrode_set, table in set_tables.items():
        memory = clusters_df[(clusters_df["electrode_set"] == electrode_set) & (clusters_df["analysis"] == MEMORY_ANALYSIS) & (clusters_df["p"] <= 0.05)]
        for (subject, channel), group in memory.groupby(["subject", "channel"]):
            row = table[(table.subject.astype(str) == str(subject)) & (table.channel.astype(str) == str(channel))].iloc[0]
            earliest = group.sort_values(["start_ms", "p"]).iloc[0]
            latency_rows.append({
                "electrode_set": electrode_set,
                "subject": str(subject),
                "channel": str(channel),
                "mni_x": float(row.mni_x),
                "mni_y": float(row.mni_y),
                "mni_z": float(row.mni_z),
                "roi": str(row.roi),
                "earliest_cluster_onset_ms": float(earliest.start_ms),
                "earliest_cluster_end_ms": float(earliest.end_ms),
                "cluster_duration_ms": float(earliest.end_ms - earliest.start_ms + 10.0),
                "cluster_p": float(earliest.p),
                "n_significant_clusters": int(len(group)),
            })
    latency_df = pd.DataFrame(latency_rows)
    latency_df.to_csv(stage / "memory_color_significant_latency_coordinates_100perm.csv", index=False, encoding="utf-8-sig")

    for electrode_set, table in set_tables.items():
        set_records = table[["subject", "channel", "mni_x", "mni_y", "mni_z", "roi"]].to_dict("records")
        curve_store = {
            (str(row["subject"]), str(row["channel"])): cache[(str(row["subject"]), str(row["channel"]))]["curves"][MEMORY_ANALYSIS][0]
            for _, row in table.iterrows()
        }
        set_latency = latency_df[latency_df["electrode_set"] == electrode_set].copy()
        _set_memory_plot(
            electrode_set,
            set_records,
            curve_store,
            set_latency,
            grid,
            figure_dir / f"{electrode_set}_memory_color_latency_coordinate.png",
        )
        _plot_feature_dominance(
            feature_df.query("electrode_set == @electrode_set"),
            figure_dir,
            f"{electrode_set}_{variant.suffix}",
            label=electrode_set,
        )

    parameters = {
        "sets": {name: int(len(table)) for name, table in set_tables.items()},
        "overlap": int(len(set(set_tables["S1"].apply(lambda row: (row.subject, row.channel), axis=1)) & set(set_tables["S2"].apply(lambda row: (row.subject, row.channel), axis=1)))),
        "variant": variant.suffix,
        "signal": variant.signal,
        "selection_variant": selection_variant.suffix,
        "n_permutations": args.perms,
        "workers_per_electrode": args.workers,
        "time_grid_ms": [float(grid[0]), float(grid[-1]), 10.0],
        "memory_color_analysis": MEMORY_ANALYSIS,
        "cluster_method": "single-electrode max-cluster-mass; p_form=0.05; min_cluster=20 ms; +1 correction",
        "latency_definition": "earliest onset of a p<=0.05 Task2 cross-fruit time-frequency cluster",
        "feature_dominance": "descriptive absolute standardized LinearSVC coefficient; no feature-level permutation test",
        "notes": [
            "S1=strategy1 (Task1 two-way ANOVA color main effect p<0.05); S2=strategy2 (at least one category Welch t-test p<0.05).",
            "No Norm2 or CSC spatial filter was applied.",
            "The 8-analysis suite includes within-task, cross-fruit, and cross-task spectrum/time-frequency decoding.",
            "Latency-coordinate correlations are descriptive and are not group-level inferential tests.",
            "The decoding input uses the selected signal variant; this run keeps the prior lf30-selected S1/S2 electrode membership for a direct signal-source comparison.",
        ],
    }
    (stage / "s1s2_decoding_parameters_100perm.json").write_text(json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(parameters, indent=2, ensure_ascii=False))


def _weights_binary_cv_for_feature(x0: np.ndarray, x1: np.ndarray, seed: int):
    from analysis.run_csc_decoding import _weights_binary_cv
    return _weights_binary_cv(x0, x1, seed)


def _weights_cross_fruit_for_feature(fruits: dict[str, np.ndarray], seed: int):
    from analysis.run_csc_decoding import _weights_cross_fruit
    return _weights_cross_fruit(fruits, seed)


def _weights_cross_task_for_feature(train0, train1, test0, test1, seed: int):
    from analysis.run_csc_decoding import _weights_cross_task
    return _weights_cross_task(train0, train1, test0, test1, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("result/final_analysis_seeg_20260806_corrected"))
    parser.add_argument("--perms", type=int, default=100)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8300)
    parser.add_argument("--signal", choices=("lf30", "raw200"), default="raw200")
    parser.add_argument(
        "--selection-signal",
        choices=("lf30", "raw200"),
        default=None,
        help="Signal variant used only to select S1/S2; defaults to --signal.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
