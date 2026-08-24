"""Plot a standardized peak-strength/latency index against MNI y.

Composite index:
    early_strength_index = z(peak value) - z(peak latency)

The peak search is restricted to 0-400 ms for all three RSA branches.  This
is a descriptive post-processing analysis and does not rerun neural features.
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
from scipy.stats import pearsonr, spearmanr

from plot_stage09_peak_latency_coordinate_correlations import (
    RESULT_ROOT,
    STAGE_CONFIG,
    _bool_series,
    _peak_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PEAK_RANGE_MS = (0.0, 400.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zscore(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    finite = np.isfinite(array)
    mean = float(np.nanmean(array[finite]))
    std = float(np.nanstd(array[finite], ddof=0))
    if not np.isfinite(std) or std < 1e-12:
        raise ValueError("Cannot standardize a constant peak variable")
    return pd.Series((array - mean) / std, index=values.index, dtype=float)


def _group_label(stage_name: str, group_name: str) -> str:
    if stage_name == "stage09_1_task2_grayfruit_rsa_raw200":
        return {
            "line_2_greater": "between > within",
            "line_1_greater": "within > between",
        }.get(group_name, "all electrodes")
    if stage_name == "stage09_3_task2_task3_cross_rsa_raw200":
        return {
            "line_2_greater": "different > same",
            "line_1_greater": "same > different",
        }.get(group_name, "all electrodes")
    return "all electrodes"


def _prepare_stage(stage_name: str, config: dict[str, str]) -> tuple[pd.DataFrame, Path]:
    stage_dir = RESULT_ROOT / stage_name
    curve_path = stage_dir / config["curve_file"]
    curves = pd.read_csv(curve_path)
    peak = _peak_table(curves, config, peak_range_ms=PEAK_RANGE_MS)
    if config["mode"] == "two_lines":
        peak_value = peak["peak_absolute_difference"]
    else:
        peak_value = peak["peak_red_green_distance"]
    peak = peak.copy()
    peak["peak_value_for_composite"] = peak_value.astype(float)
    peak["peak_value_z"] = _zscore(peak["peak_value_for_composite"])
    peak["peak_latency_z"] = _zscore(peak["peak_time_ms"])
    peak["early_strength_index"] = peak["peak_value_z"] - peak["peak_latency_z"]
    peak["composite_definition"] = "z(peak_value) - z(peak_latency); higher = stronger and earlier"
    peak["peak_range_ms"] = "0-400"
    if config["mode"] == "two_lines":
        peak["direction_label"] = peak["peak_direction"].map(
            {
                "line_2_greater": _group_label(stage_name, "line_2_greater"),
                "line_1_greater": _group_label(stage_name, "line_1_greater"),
                "tie": "tie",
            }
        )
    else:
        peak["direction_label"] = "all electrodes"
    return peak, curve_path


def _stats(group: pd.DataFrame) -> dict[str, object]:
    valid = np.isfinite(group["early_strength_index"].to_numpy(dtype=float)) & np.isfinite(group["mni_y"].to_numpy(dtype=float))
    x = group.loc[valid, "early_strength_index"].to_numpy(dtype=float)
    y = group.loc[valid, "mni_y"].to_numpy(dtype=float)
    result: dict[str, object] = {"n": int(len(x))}
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        result.update({"pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None})
        return result
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    result.update(
        {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }
    )
    return result


def _plot_stage(stage_name: str, peak: pd.DataFrame, output_path: Path, stats_by_group: dict[str, object]) -> None:
    if stage_name == "stage09_2_task3_purecolor_rsa_raw200":
        groups = [("all electrodes", peak)]
    else:
        groups = [
            (label, peak[peak["direction_label"] == label].copy())
            for label in sorted(peak["direction_label"].dropna().unique())
            if label != "tie"
        ]
    fig, axes = plt.subplots(1, len(groups), figsize=(7.0 * len(groups), 5.5), squeeze=False, constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.80, wspace=0.28)
    axes_flat = axes.ravel()
    for ax, (label, group) in zip(axes_flat, groups):
        csc = _bool_series(group["CSC"]) if "CSC" in group.columns else pd.Series(False, index=group.index)
        colors = np.where(csc, "#6c5b7b", "#4c78a8")
        valid = np.isfinite(group["early_strength_index"].to_numpy(dtype=float)) & np.isfinite(group["mni_y"].to_numpy(dtype=float))
        x = group.loc[valid, "early_strength_index"].to_numpy(dtype=float)
        y = group.loc[valid, "mni_y"].to_numpy(dtype=float)
        ax.scatter(x, y, c=colors[valid], s=46, alpha=0.85, edgecolor="white", linewidth=0.4)
        if len(x) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(np.min(x), np.max(x), 100)
            ax.plot(grid, slope * grid + intercept, color="#333333", linewidth=1.2)
        stat = stats_by_group[label]
        if stat["pearson_r"] is None:
            annotation = f"n={stat['n']}\ncorrelation unavailable"
        else:
            annotation = (
                f"n={stat['n']}\n"
                f"Pearson r={stat['pearson_r']:.2f}, p={stat['pearson_p']:.3g}\n"
                f"Spearman ρ={stat['spearman_rho']:.2f}, p={stat['spearman_p']:.3g}"
            )
        ax.text(0.03, 0.97, annotation, transform=ax.transAxes, va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        ax.set_xlabel("early-strength index: z(peak value) − z(peak latency)")
        ax.set_ylabel("MNI Y (mm)")
        ax.set_title(label)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{stage_name}\ncomposite peak index versus MNI Y (peak restricted to 0–400 ms)", fontsize=14, y=0.96)
    fig.text(0.5, 0.035, "Higher index = larger peak value and earlier peak latency; standardization was across all electrodes in this stage.", ha="center", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / f"{timestamp}_stage09_composite_peak_index_mni_y"
    run_dir.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc)
    (run_dir / "start_time.txt").write_text(start.isoformat() + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(subprocess.list2cmdline([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    summaries = []
    try:
        for stage_name, config in STAGE_CONFIG.items():
            peak, curve_path = _prepare_stage(stage_name, config)
            stage_dir = RESULT_ROOT / stage_name
            table_path = stage_dir / "composite_peak_index_mni_y_0_400ms.csv"
            stats_path = stage_dir / "composite_peak_index_mni_y_0_400ms_statistics.json"
            figure_path = stage_dir / "figures" / "composite_peak_index_mni_y_0_400ms.png"
            if config["mode"] == "two_lines":
                groups = {
                    label: peak[peak["direction_label"] == label].copy()
                    for label in sorted(peak["direction_label"].dropna().unique())
                    if label != "tie"
                }
            else:
                groups = {"all electrodes": peak}
            stats_by_group = {label: _stats(group) for label, group in groups.items()}
            peak.to_csv(table_path, index=False, encoding="utf-8-sig")
            stats_payload = {
                "stage": stage_name,
                "peak_range_ms": list(PEAK_RANGE_MS),
                "composite_definition": "z(peak_value) - z(peak_latency)",
                "peak_value": "peak_absolute_difference for stage09_1/3; peak_red_green_distance for stage09_2",
                "input_curve_file": str(curve_path),
                "input_sha256": _sha256(curve_path),
                "groups": stats_by_group,
            }
            stats_path.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            _plot_stage(stage_name, peak, figure_path, stats_by_group)
            summaries.append(
                {
                    "stage": stage_name,
                    "n_electrodes": int(len(peak)),
                    "group_counts": {label: int(len(group)) for label, group in groups.items()},
                    "outputs": [str(table_path), str(stats_path), str(figure_path)],
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
