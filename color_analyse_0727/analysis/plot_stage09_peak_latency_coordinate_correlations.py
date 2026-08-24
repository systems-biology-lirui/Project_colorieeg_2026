"""Plot peak RSA-distance latency against electrode MNI coordinates.

This is a descriptive post-processing step.  It reads completed stage09
distance curves, selects one peak time per subject-channel, and does not rerun
STFT/RSA calculations.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PROJECT_ROOT / "color_analyse_0727"
RESULT_ROOT = MODULE_ROOT / "result" / "final_analysis_seeg_20260806_corrected"

STAGE_CONFIG = {
    "stage09_1_task2_grayfruit_rsa_raw200": {
        "curve_file": "memory_color_distance_curves.csv",
        "figure_name": "peak_memory_color_distance_latency_coordinate_correlation.png",
        "table_name": "peak_memory_color_distance_latency_coordinates.csv",
        "stats_name": "peak_memory_color_distance_latency_coordinate_statistics.json",
        "mode": "two_lines",
        "line_1": "within_memory_color_distance",
        "line_2": "between_memory_color_distance",
        "line_1_label": "within memory-color distance",
        "line_2_label": "between memory-color distance",
        "line_2_group_suffix": "between_greater_than_within",
        "line_2_short_suffix": "between_gt_within",
        "line_2_group_label": "between memory-color distance > within memory-color distance",
        "line_1_group_suffix": "within_greater_than_between",
        "line_1_short_suffix": "within_gt_between",
        "line_1_group_label": "within memory-color distance > between memory-color distance",
        "analysis_label": "stage09_1 Task2 gray-fruit RSA",
    },
    "stage09_2_task3_purecolor_rsa_raw200": {
        "curve_file": "memory_color_distance_curves.csv",
        "figure_name": "peak_red_green_distance_latency_coordinate_correlation.png",
        "table_name": "peak_red_green_distance_latency_coordinates.csv",
        "stats_name": "peak_red_green_distance_latency_coordinate_statistics.json",
        "mode": "single_distance",
        "distance": "red_green_distance",
        "analysis_label": "stage09_2 Task3 pure-color RSA",
    },
    "stage09_3_task2_task3_cross_rsa_raw200": {
        "curve_file": "cross_task_memory_color_distance_curves.csv",
        "figure_name": "peak_cross_task_memory_color_distance_latency_coordinate_correlation.png",
        "table_name": "peak_cross_task_memory_color_distance_latency_coordinates.csv",
        "stats_name": "peak_cross_task_memory_color_distance_latency_coordinate_statistics.json",
        "mode": "two_lines",
        "line_1": "same_memory_color_distance",
        "line_2": "different_memory_color_distance",
        "line_1_label": "same memory-color distance",
        "line_2_label": "different memory-color distance",
        "line_2_group_suffix": "different_greater_than_same",
        "line_2_short_suffix": "different_gt_same",
        "line_2_group_label": "different memory-color distance > same memory-color distance",
        "line_1_group_suffix": "same_greater_than_different",
        "line_1_short_suffix": "same_gt_different",
        "line_1_group_label": "same memory-color distance > different memory-color distance",
        "analysis_label": "stage09_3 Task2×Task3 cross-task RSA",
    },
}

COORDINATES = ("mni_x", "mni_y", "mni_z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().eq("true")


def _peak_table(
    curves: pd.DataFrame,
    config: dict[str, str],
    peak_range_ms: tuple[float, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (subject, channel), group in curves.groupby(["subject", "channel"], sort=False):
        group = group.sort_values("time_bin_index").reset_index(drop=True)
        if peak_range_ms is not None:
            group = group[
                (group["bin_start_ms"] >= peak_range_ms[0])
                & (group["bin_end_ms"] <= peak_range_ms[1])
            ].reset_index(drop=True)
            if group.empty:
                continue
        if config["mode"] == "two_lines":
            signed = group[config["line_2"]].to_numpy(dtype=float) - group[config["line_1"]].to_numpy(dtype=float)
            separation = np.abs(signed)
            if not np.isfinite(separation).any():
                continue
            peak_index = int(np.nanargmax(separation))
            peak = group.iloc[peak_index]
            record = {
                "subject": subject,
                "channel": channel,
                "peak_time_bin_index": int(peak["time_bin_index"]),
                "peak_bin_start_ms": float(peak["bin_start_ms"]),
                "peak_bin_end_ms": float(peak["bin_end_ms"]),
                "peak_time_ms": float((peak["bin_start_ms"] + peak["bin_end_ms"]) / 2.0),
                "line_1_distance_at_peak": float(peak[config["line_1"]]),
                "line_2_distance_at_peak": float(peak[config["line_2"]]),
                "peak_signed_difference": float(signed[peak_index]),
                "peak_absolute_difference": float(separation[peak_index]),
                "peak_direction": "line_2_greater" if signed[peak_index] > 0 else "line_1_greater" if signed[peak_index] < 0 else "tie",
                "peak_selection_range_ms": "all_bins" if peak_range_ms is None else f"{peak_range_ms[0]:g}-{peak_range_ms[1]:g}",
                "peak_selection_rule": f"maximum abs({config['line_2']} - {config['line_1']})",
            }
        else:
            values = group[config["distance"]].to_numpy(dtype=float)
            if not np.isfinite(values).any():
                continue
            peak_index = int(np.nanargmax(values))
            peak = group.iloc[peak_index]
            record = {
                "subject": subject,
                "channel": channel,
                "peak_time_bin_index": int(peak["time_bin_index"]),
                "peak_bin_start_ms": float(peak["bin_start_ms"]),
                "peak_bin_end_ms": float(peak["bin_end_ms"]),
                "peak_time_ms": float((peak["bin_start_ms"] + peak["bin_end_ms"]) / 2.0),
                "peak_red_green_distance": float(peak[config["distance"]]),
                "peak_selection_range_ms": "all_bins" if peak_range_ms is None else f"{peak_range_ms[0]:g}-{peak_range_ms[1]:g}",
                "peak_selection_rule": "maximum red_green_distance",
            }
        metadata = group.iloc[0]
        for column in ("mni_x", "mni_y", "mni_z", "roi", "electrode_sets", "S1", "S2", "CSC"):
            if column in group.columns:
                record[column] = metadata[column]
        rows.append(record)

    peak = pd.DataFrame(rows)
    if peak.empty:
        raise ValueError("No finite per-electrode peaks were found")
    for column in ("S1", "S2", "CSC"):
        if column in peak.columns:
            peak[column] = _bool_series(peak[column])
    return peak.sort_values(["subject", "channel"]).reset_index(drop=True)


def _correlation_stats(peak: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {"n_electrodes": int(len(peak)), "coordinates": {}}
    for coordinate in COORDINATES:
        valid = np.isfinite(peak[coordinate].to_numpy(dtype=float)) & np.isfinite(peak["peak_time_ms"].to_numpy(dtype=float))
        x = peak.loc[valid, coordinate].to_numpy(dtype=float)
        y = peak.loc[valid, "peak_time_ms"].to_numpy(dtype=float)
        if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            result["coordinates"][coordinate] = {"n": int(len(x)), "pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None}
            continue
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        result["coordinates"][coordinate] = {
            "n": int(len(x)),
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }
    return result


def _plot(
    peak: pd.DataFrame,
    config: dict[str, str],
    output_path: Path,
    stats: dict[str, object],
    title_suffix: str = "",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    csc = _bool_series(peak["CSC"]) if "CSC" in peak.columns else pd.Series(False, index=peak.index)
    colors = np.where(csc, "#6c5b7b", "#4c78a8")
    if config["mode"] == "two_lines":
        peak_magnitude = peak["peak_absolute_difference"].to_numpy(dtype=float)
        magnitude_label = "peak absolute line difference"
    else:
        peak_magnitude = peak["peak_red_green_distance"].to_numpy(dtype=float)
        magnitude_label = "peak red–green distance"
    finite_magnitude = np.isfinite(peak_magnitude)
    magnitude_min = float(np.nanmin(peak_magnitude[finite_magnitude]))
    magnitude_max = float(np.nanmax(peak_magnitude[finite_magnitude]))
    if magnitude_max > magnitude_min:
        point_sizes = 38.0 + 170.0 * (peak_magnitude - magnitude_min) / (magnitude_max - magnitude_min)
    else:
        point_sizes = np.full(len(peak), 100.0, dtype=float)
    point_sizes[~finite_magnitude] = 70.0
    for ax, coordinate in zip(axes, COORDINATES):
        valid = np.isfinite(peak[coordinate].to_numpy(dtype=float)) & np.isfinite(peak["peak_time_ms"].to_numpy(dtype=float))
        x = peak.loc[valid, coordinate].to_numpy(dtype=float)
        y = peak.loc[valid, "peak_time_ms"].to_numpy(dtype=float)
        point_colors = colors[valid]
        ax.scatter(x, y, c=point_colors, s=point_sizes[valid], alpha=0.85, edgecolor="white", linewidth=0.4)
        if len(x) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(np.min(x), np.max(x), 100)
            ax.plot(grid, slope * grid + intercept, color="#333333", linewidth=1.2)
        stat = stats["coordinates"][coordinate]
        if stat["pearson_r"] is None:
            annotation = f"n={stat['n']}\ncorrelation unavailable"
        else:
            annotation = (
                f"n={stat['n']}\n"
                f"Pearson r={stat['pearson_r']:.2f}, p={stat['pearson_p']:.3g}\n"
                f"Spearman ρ={stat['spearman_rho']:.2f}, p={stat['spearman_p']:.3g}"
            )
        ax.text(0.03, 0.97, annotation, transform=ax.transAxes, va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        ax.set_xlabel(f"MNI {coordinate[-1].upper()} (mm)")
        ax.set_ylabel("peak latency (ms)")
        ax.set_title(f"peak latency vs MNI {coordinate[-1].upper()}")
        ax.grid(alpha=0.2)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4c78a8", markeredgecolor="white", markersize=7, label="non-CSC selected electrode"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6c5b7b", markeredgecolor="white", markersize=7, label="CSC electrode"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#999999", markeredgecolor="white", markersize=5, label=f"small {magnitude_label} ({magnitude_min:.2f})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#999999", markeredgecolor="white", markersize=12, label=f"large {magnitude_label} ({magnitude_max:.2f})"),
    ]
    axes[0].legend(handles=legend_handles, frameon=False, fontsize=7, loc="lower right", ncol=2)
    if config["mode"] == "two_lines":
        peak_rule = f"peak = max |{config['line_2']} − {config['line_1']}|; signed difference retained in CSV"
    else:
        peak_rule = "peak = maximum red–green correlation distance"
    title = f"{config['analysis_label']}\n"
    if title_suffix:
        title += f"{title_suffix}\n"
    title += "peak latency and MNI-coordinate correlation"
    fig.suptitle(title, fontsize=14)
    fig.text(0.5, 0.01, f"{peak_rule}; point size ∝ {magnitude_label}", ha="center", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / f"{timestamp}_stage09_peak_latency_coordinate_correlations"
    run_dir.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc)
    (run_dir / "start_time.txt").write_text(start.isoformat() + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(subprocess.list2cmdline([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    summaries: list[dict[str, object]] = []
    try:
        for stage_name, config in STAGE_CONFIG.items():
            stage_dir = RESULT_ROOT / stage_name
            curve_path = stage_dir / config["curve_file"]
            curves = pd.read_csv(curve_path)
            peak = _peak_table(curves, config)
            stats = _correlation_stats(peak)
            table_path = stage_dir / config["table_name"]
            stats_path = stage_dir / config["stats_name"]
            figure_path = stage_dir / "figures" / config["figure_name"]
            peak.to_csv(table_path, index=False, encoding="utf-8-sig")
            stats_payload = {
                "analysis": config["analysis_label"],
                "stage": stage_name,
                "selection_rule": config["line_2"] + " minus " + config["line_1"] if config["mode"] == "two_lines" else "red_green_distance",
                "input_curve_file": str(curve_path),
                "statistics": stats,
            }
            stats_path.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            _plot(peak, config, figure_path, stats)
            outputs = [str(table_path), str(stats_path), str(figure_path)]
            direction_counts: dict[str, int] = {}
            if config["mode"] == "two_lines":
                split_peak = _peak_table(curves, config, peak_range_ms=(0.0, 400.0))
                split_suffix = "_0_400ms"
                for direction, suffix, label in (
                    ("line_2_greater", config["line_2_short_suffix"], config["line_2_group_label"]),
                    ("line_1_greater", config["line_1_short_suffix"], config["line_1_group_label"]),
                ):
                    group = split_peak[split_peak["peak_direction"] == direction].copy()
                    direction_counts[direction] = int(len(group))
                    group_table_path = stage_dir / f"peak_latency_coordinates_{suffix}{split_suffix}.csv"
                    group_stats_path = stage_dir / f"peak_latency_coordinate_statistics_{suffix}{split_suffix}.json"
                    group_figure_path = stage_dir / "figures" / f"peak_latency_coordinate_correlation_{suffix}{split_suffix}.png"
                    group.to_csv(group_table_path, index=False, encoding="utf-8-sig")
                    group_stats = _correlation_stats(group)
                    group_stats_payload = {
                        "analysis": config["analysis_label"],
                        "stage": stage_name,
                        "direction_group": direction,
                        "direction_label": label,
                        "peak_selection_range_ms": [0.0, 400.0],
                        "statistics": group_stats,
                    }
                    group_stats_path.write_text(json.dumps(group_stats_payload, indent=2, ensure_ascii=False), encoding="utf-8")
                    if not group.empty:
                        _plot(group, config, group_figure_path, group_stats, title_suffix=label)
                    outputs.extend([str(group_table_path), str(group_stats_path), str(group_figure_path)])
            summaries.append(
                {
                    "stage": stage_name,
                    "n_electrodes": int(len(peak)),
                    "direction_counts": direction_counts,
                    "input_sha256": _sha256(curve_path),
                    "outputs": outputs,
                }
            )
        end = datetime.now(timezone.utc)
        run_summary = {
            "status": "completed",
            "started_at_utc": start.isoformat(),
            "finished_at_utc": end.isoformat(),
            "stages": summaries,
            "run_dir": str(run_dir),
        }
        (run_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "output_paths.txt").write_text("\n".join(path for item in summaries for path in item["outputs"]) + "\n", encoding="utf-8")
        print(json.dumps(run_summary, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "started_at_utc": start.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(exc),
        }
        (run_dir / "run_summary.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    finally:
        (run_dir / "finish_time.txt").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
