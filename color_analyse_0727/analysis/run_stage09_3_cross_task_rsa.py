"""Cross-task RSA for Task2 gray fruits and Task3 pure color patches.

This is an independent stage09_3 analysis.  It combines the four Task2 gray
fruit conditions with the Task3 red and green pure-color patches in one
trial-level raw200 time-frequency feature space, then measures whether a gray
fruit is closer to the pure-color patch with the same memory color than to the
opposite-color patch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PROJECT_ROOT / "color_analyse_0727"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from analysis.common import BASELINE_MS, SUBJECTS, h5_path, load_conditions, natural_key
from analysis.run_task1_condition_rsa import (
    EXPECTED_SET_COUNTS,
    FEATURE_BANDS,
    BAND_NAMES,
    SET_COLUMNS,
    TF_HOP_MS,
    TF_NPERSEG,
    _baseline_zscore,
    _correlation_rdm,
    _git_commit,
    _load_electrode_sets,
    _package_versions,
    _safe_name,
    _sha256,
    _timecourse_mean,
    _trial_logpower,
)


ANALYSIS_NAME = "stage09_3_task2_task3_cross_rsa_raw200"
STAGE_KEY = "stage09_3"
TIME_RANGE_MS = (0.0, 800.0)
BIN_MS = 50.0

TASK2_CONDITIONS = (
    "strawberry_gray",
    "watermelon_gray",
    "cabbage_gray",
    "kiwi_gray",
)
TASK3_CONDITIONS = ("red", "green")
CONDITIONS = TASK2_CONDITIONS + ("red_patch", "green_patch")
SHORT_NAMES = ("R1", "R2", "G1", "G2", "red", "green")
TRIGGERS = {
    "strawberry_gray": 123,
    "watermelon_gray": 133,
    "cabbage_gray": 103,
    "kiwi_gray": 113,
    "red_patch": 51,
    "green_patch": 54,
}
CONDITION_COLORS = {
    "strawberry_gray": "#d73027",
    "watermelon_gray": "#fc8d59",
    "cabbage_gray": "#4575b4",
    "kiwi_gray": "#74add1",
    "red_patch": "#8b0000",
    "green_patch": "#006d2c",
}


def _decode_scalar(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _audit_h5(
    subject: str,
    task: int,
    expected_conditions: tuple[str, ...],
    expected_triggers: dict[str, int],
) -> dict[str, Any]:
    path = h5_path(subject, task)
    with h5py.File(path, "r") as handle:
        names = [_decode_scalar(x) for x in handle["condition_names"][()]]
        triggers = [_decode_scalar(x) for x in handle["condition_triggers"][()]]
        counts = [int(x) for x in handle["trial_counts"][()]]
        time_ms = np.asarray(handle["time_ms"][()], dtype=float)
        labels = [_decode_scalar(x).strip().upper() for x in handle["labels"][()]]
    trigger_map = {name: int(trigger) for name, trigger in zip(names, triggers)}
    return {
        "subject": subject,
        "task": task,
        "path": str(path),
        "condition_names_in_hdf5": names,
        "condition_triggers_in_hdf5": trigger_map,
        "trial_counts": dict(zip(names, counts)),
        "n_channels": len(labels),
        "time_start_ms": float(time_ms[0]),
        "time_end_ms": float(time_ms[-1]),
        "n_timepoints": int(time_ms.size),
        "requested_conditions_present": set(expected_conditions).issubset(set(names)),
        "requested_triggers_match": all(
            trigger_map.get(name) == trigger for name, trigger in expected_triggers.items()
        ),
    }


def _input_manifest(selection_path: Path, subjects: tuple[str, ...]) -> list[dict[str, Any]]:
    paths = [selection_path]
    paths.extend(h5_path(subject, task) for subject in subjects for task in (2, 3))
    manifest: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        stat = path.stat()
        manifest.append(
            {
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "modified_ns": int(stat.st_mtime_ns),
                "sha256": _sha256(path),
            }
        )
    return manifest


def _time_bins() -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(TIME_RANGE_MS[0], TIME_RANGE_MS[1], BIN_MS, dtype=float)
    ends = starts + BIN_MS
    if len(starts) != 16:
        raise ValueError(f"Expected 16 time bins, got {len(starts)}")
    return starts, ends


def _finite_mean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def _cross_task_curve(rdm: np.ndarray) -> dict[str, float]:
    """Return gray-fruit to matching and mismatching pure-color distances."""
    pair_distances = {
        "strawberry_to_red_patch": float(rdm[0, 4]),
        "watermelon_to_red_patch": float(rdm[1, 4]),
        "cabbage_to_green_patch": float(rdm[2, 5]),
        "kiwi_to_green_patch": float(rdm[3, 5]),
        "strawberry_to_green_patch": float(rdm[0, 5]),
        "watermelon_to_green_patch": float(rdm[1, 5]),
        "cabbage_to_red_patch": float(rdm[2, 4]),
        "kiwi_to_red_patch": float(rdm[3, 4]),
    }
    same = _finite_mean(
        [
            pair_distances["strawberry_to_red_patch"],
            pair_distances["watermelon_to_red_patch"],
            pair_distances["cabbage_to_green_patch"],
            pair_distances["kiwi_to_green_patch"],
        ]
    )
    different = _finite_mean(
        [
            pair_distances["strawberry_to_green_patch"],
            pair_distances["watermelon_to_green_patch"],
            pair_distances["cabbage_to_red_patch"],
            pair_distances["kiwi_to_red_patch"],
        ]
    )
    return {
        **pair_distances,
        "same_memory_color_distance": same,
        "different_memory_color_distance": different,
        "different_minus_same": different - same
        if np.isfinite(same) and np.isfinite(different)
        else float("nan"),
    }


def _plot_stage_figure(
    result: dict[str, Any],
    output_path: Path,
    time_starts: np.ndarray,
    time_ends: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(20, 19))
    grid = GridSpec(
        6,
        4,
        figure=fig,
        height_ratios=[1.15, 1, 1, 1, 1, 1.05],
        hspace=0.8,
        wspace=0.55,
    )
    ax_time = fig.add_subplot(grid[0, :])
    for condition in CONDITIONS:
        ax_time.plot(
            result["time_ms"],
            result["timecourses"][condition],
            linewidth=1.0,
            label=condition,
            color=CONDITION_COLORS[condition],
        )
    ax_time.set_xlim(*TIME_RANGE_MS)
    ax_time.axvline(0, color="black", linestyle="--", linewidth=0.7)
    ax_time.set_title("Task2 gray-fruit and Task3 pure-color condition means")
    ax_time.set_xlabel("time (ms)")
    ax_time.set_ylabel("baseline-subtracted amplitude")
    ax_time.legend(ncol=6, fontsize=7, frameon=False)
    ax_time.grid(alpha=0.2)

    rdm_axes = []
    image = None
    for bin_index, rdm in enumerate(result["rdms"]):
        row, col = divmod(bin_index, 4)
        ax = fig.add_subplot(grid[row + 1, col])
        rdm_axes.append(ax)
        image = ax.imshow(
            rdm,
            cmap="viridis",
            vmin=0.0,
            vmax=2.0,
            interpolation="nearest",
        )
        ax.set_title(
            f"{time_starts[bin_index]:.0f}–{time_ends[bin_index]:.0f} ms",
            fontsize=8,
        )
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_yticks(range(len(CONDITIONS)))
        ax.set_xticklabels(SHORT_NAMES, rotation=90, fontsize=6)
        ax.set_yticklabels(SHORT_NAMES, fontsize=6)
        ax.axvline(3.5, color="white", linewidth=0.8)
        ax.axhline(3.5, color="white", linewidth=0.8)
    if image is not None:
        fig.colorbar(
            image,
            ax=rdm_axes,
            shrink=0.56,
            label="correlation distance (1 − Pearson r)",
        )

    curve = result["distance_curve"]
    x = time_starts + BIN_MS / 2
    ax_curve = fig.add_subplot(grid[5, :])
    ax_curve.plot(
        x,
        curve["same_memory_color_distance"],
        color="#542788",
        marker="o",
        label="same memory color: gray fruit → matching pure patch",
    )
    ax_curve.plot(
        x,
        curve["different_memory_color_distance"],
        color="#b35806",
        marker="o",
        label="different memory color: gray fruit → opposite pure patch",
    )
    ax_curve.set_xlabel("time-bin center (ms)")
    ax_curve.set_ylabel("correlation distance")
    ax_curve.set_title("cross-task memory-color geometry over time")
    ax_curve.legend(fontsize=8, frameon=False)
    ax_curve.grid(alpha=0.2)

    meta = result["metadata"]
    fig.suptitle(
        f"{ANALYSIS_NAME} | {meta['subject']} {meta['channel']} | "
        f"sets={meta['electrode_sets']} | "
        f"MNI=({meta['mni_x']:.1f}, {meta['mni_y']:.1f}, {meta['mni_z']:.1f})",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.978,
        "Task2 + Task3 trials combined → raw200 STFT → global −200–0 ms baseline z-score → six-condition mean RDM",
        ha="center",
        fontsize=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_set_figure(
    set_name: str,
    results: list[dict[str, Any]],
    output_path: Path,
    time_starts: np.ndarray,
    time_ends: np.ndarray,
) -> None:
    if not results:
        return
    rdms = np.nanmean(np.stack([r["rdms"] for r in results], axis=0), axis=0)
    for rdm in rdms:
        np.fill_diagonal(rdm, 0.0)
    curve = {
        key: np.nanmean(
            np.stack([r["distance_curve"][key] for r in results], axis=0), axis=0
        )
        for key in (
            "same_memory_color_distance",
            "different_memory_color_distance",
        )
    }
    proxy = {
        "time_ms": results[0]["time_ms"],
        "timecourses": {
            condition: np.nanmean(
                np.stack([r["timecourses"][condition] for r in results], axis=0),
                axis=0,
            )
            for condition in CONDITIONS
        },
        "rdms": rdms,
        "distance_curve": curve,
        "metadata": {
            "subject": f"{len(results)} electrodes",
            "channel": set_name,
            "electrode_sets": set_name,
            "mni_x": np.nan,
            "mni_y": np.nan,
            "mni_z": np.nan,
        },
    }
    _plot_stage_figure(proxy, output_path, time_starts, time_ends)


def _write_readme(result_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# {ANALYSIS_NAME}",
        "",
        "> 独立新增结果，不覆盖 stage01–08、stage09、stage09_1 或 stage09_2。",
        "",
        "## 方法",
        "",
        "- Task2 的四种灰色水果与 Task3 的 red/green 纯色色块在同一 subject-electrode 的六条件集合中联合分析。",
        "- 条件顺序固定为 R1=strawberry_gray、R2=watermelon_gray、G1=cabbage_gray、G2=kiwi_gray、red=Task3 red、green=Task3 green。",
        "- 六类 trial 合并后，对完整 −500–1000 ms epoch 做 trial-level raw200 STFT；log-power 相对于联合六条件的 −200–0 ms baseline 做 z-score，然后在条件内平均。",
        "- 在 0–800 ms 计算 16 个连续 50 ms 时间窗的 6×6 correlation-distance RDM，距离为 `1 − Pearson correlation`，特征为既有 16 个频带。",
        "- 同记忆颜色距离为 `(R1-red + R2-red + G1-green + G2-green) / 4`；异记忆颜色距离为 `(R1-green + R2-green + G1-red + G2-red) / 4`。",
        "- 集合图采用单电极 RDM 和距离曲线的直接算术平均；本版本不做 bootstrap、置换检验或跨电极显著性检验。",
        "",
        "## 解释边界",
        "",
        "同记忆颜色距离低于异记忆颜色距离，才是与跨任务共同记忆颜色结构一致的描述性模式；该结果仍不能排除 Task2/Task3 任务差异、图片/刺激身份或低层视觉差异。",
        "",
        "## 摘要",
        f"- unique electrodes: {summary['unique_electrodes']}",
        f"- set counts: {json.dumps(summary['set_counts'], ensure_ascii=False)}",
        f"- conditions: {json.dumps(list(CONDITIONS), ensure_ascii=False)}",
        f"- output directory: `{result_dir}`",
        "",
        "完整命令、输入 SHA256、Git commit、Python 环境、时间和 warnings 保存在项目根目录 `runs/` 的对应运行目录。",
    ]
    (result_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(
    selection: pd.DataFrame,
    subjects: tuple[str, ...],
    result_dir: Path,
) -> dict[str, Any]:
    time_starts, time_ends = _time_bins()
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "electrode_figures").mkdir(exist_ok=True)
    (result_dir / "set_figures").mkdir(exist_ok=True)
    metadata_by_key = {
        (str(row.subject), str(row.channel)): row._asdict()
        for row in selection.itertuples(index=False)
    }
    channels_by_subject = {
        subject: sorted(
            [channel for sub, channel in metadata_by_key if sub == subject],
            key=natural_key,
        )
        for subject in subjects
    }

    electrode_results: dict[tuple[str, str], dict[str, Any]] = {}
    timecourse_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    rdm_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    condition_counts: list[dict[str, Any]] = []

    for subject in subjects:
        channels = channels_by_subject[subject]
        if not channels:
            continue
        task2_raw, time2, returned2 = load_conditions(
            subject, 2, TASK2_CONDITIONS, channels
        )
        task3_raw, time3, returned3 = load_conditions(
            subject, 3, TASK3_CONDITIONS, channels
        )
        if [c.upper() for c in returned2] != [c.upper() for c in channels]:
            raise ValueError(f"Task2 channel order changed for {subject}")
        if [c.upper() for c in returned3] != [c.upper() for c in channels]:
            raise ValueError(f"Task3 channel order changed for {subject}")
        if not np.allclose(time2, time3, atol=1e-6, rtol=0):
            raise ValueError(f"Task2/Task3 time axes differ for {subject}")

        raw_by_condition = {
            **task2_raw,
            "red_patch": task3_raw["red"],
            "green_patch": task3_raw["green"],
        }
        stacked = np.concatenate([raw_by_condition[c] for c in CONDITIONS], axis=0)
        trial_logpower, frame_times = _trial_logpower(stacked)
        trial_z = _baseline_zscore(trial_logpower, frame_times, BASELINE_MS)

        offsets: dict[str, tuple[int, int]] = {}
        offset = 0
        for condition in CONDITIONS:
            n_trials = raw_by_condition[condition].shape[0]
            offsets[condition] = (offset, offset + n_trials)
            offset += n_trials

        trial_features_by_bin: list[dict[str, np.ndarray]] = []
        valid_counts = {condition: [] for condition in CONDITIONS}
        for start_ms, end_ms in zip(time_starts, time_ends):
            frame_mask = (frame_times >= start_ms) & (frame_times < end_ms)
            if not frame_mask.any():
                raise ValueError(f"No TF frames in {start_ms}-{end_ms} ms")
            trial_features: dict[str, np.ndarray] = {}
            for condition in CONDITIONS:
                start, stop = offsets[condition]
                values = trial_z[start:stop, :, :, frame_mask]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    features = np.nanmean(values, axis=3).astype(np.float32)
                trial_features[condition] = features
                valid_counts[condition].append(
                    int(np.isfinite(features).all(axis=(1, 2)).sum())
                )
            trial_features_by_bin.append(trial_features)

        for condition in CONDITIONS:
            condition_counts.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "source_task": 2 if condition in TASK2_CONDITIONS else 3,
                    "trigger": int(TRIGGERS[condition]),
                    "n_trials": int(raw_by_condition[condition].shape[0]),
                    "n_valid_trials_min_across_bins": int(min(valid_counts[condition])),
                    "n_valid_trials_max_across_bins": int(max(valid_counts[condition])),
                    "stage": STAGE_KEY,
                }
            )

        for channel_index, channel in enumerate(channels):
            key = (subject, channel)
            meta = metadata_by_key[key]
            timecourses = {
                condition: _timecourse_mean(
                    raw_by_condition[condition][:, channel_index : channel_index + 1, :],
                    time2,
                    BASELINE_MS,
                )[0]
                for condition in CONDITIONS
            }
            for condition_index, condition in enumerate(CONDITIONS):
                for time_index, time_value in enumerate(time2):
                    timecourse_rows.append(
                        {
                            **meta,
                            "stage": STAGE_KEY,
                            "condition": condition,
                            "condition_index": condition_index,
                            "source_task": 2 if condition in TASK2_CONDITIONS else 3,
                            "time_ms": float(time_value),
                            "mean_signal": float(timecourses[condition][time_index]),
                            "signal": "raw200",
                        }
                    )

            rdms: list[np.ndarray] = []
            curve_rows: list[dict[str, Any]] = []
            for bin_index, (start_ms, end_ms) in enumerate(zip(time_starts, time_ends)):
                feature_matrix = np.asarray(
                    [
                        np.nanmean(
                            trial_features_by_bin[bin_index][condition][:, channel_index, :],
                            axis=0,
                        )
                        for condition in CONDITIONS
                    ],
                    dtype=float,
                )
                for condition_index, condition in enumerate(CONDITIONS):
                    for band_index, band_name in enumerate(BAND_NAMES):
                        feature_rows.append(
                            {
                                **meta,
                                "stage": STAGE_KEY,
                                "condition": condition,
                                "condition_index": condition_index,
                                "source_task": 2 if condition in TASK2_CONDITIONS else 3,
                                "time_bin_index": bin_index,
                                "bin_start_ms": float(start_ms),
                                "bin_end_ms": float(end_ms),
                                "band_index": band_index,
                                "band_name": band_name,
                                "band_low_hz": float(FEATURE_BANDS[band_index][0]),
                                "band_high_hz": float(FEATURE_BANDS[band_index][1]),
                                "mean_tf_z": float(feature_matrix[condition_index, band_index]),
                                "signal": "raw200",
                            }
                        )
                rdm = _correlation_rdm(feature_matrix)
                rdms.append(rdm)
                for i, condition_i in enumerate(CONDITIONS):
                    for j, condition_j in enumerate(CONDITIONS):
                        rdm_rows.append(
                            {
                                **meta,
                                "stage": STAGE_KEY,
                                "time_bin_index": bin_index,
                                "bin_start_ms": float(start_ms),
                                "bin_end_ms": float(end_ms),
                                "condition_i": condition_i,
                                "condition_i_index": i,
                                "condition_j": condition_j,
                                "condition_j_index": j,
                                "distance": float(rdm[i, j]),
                                "distance_method": "1-Pearson correlation across 16 mean band features",
                                "signal": "raw200",
                            }
                        )
                curve = _cross_task_curve(rdm)
                curve_rows.append(
                    {
                        **meta,
                        "stage": STAGE_KEY,
                        "time_bin_index": bin_index,
                        "bin_start_ms": float(start_ms),
                        "bin_end_ms": float(end_ms),
                        **curve,
                        "distance_method": "1-Pearson correlation across 16 mean band features",
                        "signal": "raw200",
                    }
                )

            for row in curve_rows:
                distance_rows.append(row)
            electrode_results[key] = {
                "metadata": meta,
                "time_ms": np.asarray(time2, dtype=float),
                "timecourses": timecourses,
                "rdms": np.asarray(rdms, dtype=float),
                "distance_curve": {
                    key: np.asarray([row[key] for row in curve_rows], dtype=float)
                    for key in (
                        "same_memory_color_distance",
                        "different_memory_color_distance",
                    )
                },
            }
            _plot_stage_figure(
                electrode_results[key],
                result_dir / "electrode_figures" / f"{_safe_name(subject)}_{_safe_name(channel)}_{STAGE_KEY}.png",
                time_starts,
                time_ends,
            )
        print(f"completed {STAGE_KEY} {subject}: {len(channels)} electrodes", flush=True)

    if not electrode_results:
        raise ValueError("No results produced")
    pd.DataFrame(timecourse_rows).to_csv(
        result_dir / "condition_mean_timecourses.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(feature_rows).to_csv(
        result_dir / "condition_mean_tf_features.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(rdm_rows).to_csv(
        result_dir / "condition_rdm_long.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(distance_rows).to_csv(
        result_dir / "cross_task_memory_color_distance_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(condition_counts).to_csv(
        result_dir / "condition_trial_counts.csv", index=False, encoding="utf-8-sig"
    )
    selection.to_csv(result_dir / "electrode_sets_used.csv", index=False, encoding="utf-8-sig")

    set_counts: dict[str, int] = {}
    for set_name in ("S1", "S2", "CSC"):
        members = [
            result
            for result in electrode_results.values()
            if bool(result["metadata"].get(set_name, False))
        ]
        set_counts[set_name] = len(members)
        _plot_set_figure(
            set_name,
            members,
            result_dir / "set_figures" / f"{set_name}_mean_{STAGE_KEY}.png",
            time_starts,
            time_ends,
        )

    return {
        "analysis": ANALYSIS_NAME,
        "stage": STAGE_KEY,
        "unique_electrodes": len(electrode_results),
        "set_counts": set_counts,
        "conditions": list(CONDITIONS),
        "condition_triggers": TRIGGERS,
        "time_bins": len(time_starts),
        "feature_bands": len(FEATURE_BANDS),
        "output_dir": str(result_dir),
    }


def _write_parameters(
    result_dir: Path,
    args: argparse.Namespace,
    summary: dict[str, Any],
    audits: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    run_dir: Path,
) -> None:
    parameters = {
        "analysis": ANALYSIS_NAME,
        "stage": STAGE_KEY,
        "signal": "raw200",
        "tasks": {"task2": list(TASK2_CONDITIONS), "task3": list(TASK3_CONDITIONS)},
        "combined_condition_order": list(CONDITIONS),
        "condition_short_names": list(SHORT_NAMES),
        "condition_triggers": TRIGGERS,
        "baseline_ms": list(BASELINE_MS),
        "time_range_ms": list(TIME_RANGE_MS),
        "bin_ms": BIN_MS,
        "tf_method": "scipy.signal.stft on every trial over full epoch",
        "tf_nperseg_samples": TF_NPERSEG,
        "tf_hop_ms": TF_HOP_MS,
        "feature_bands_hz": [list(band) for band in FEATURE_BANDS],
        "feature_transform": "Task2 and Task3 trials combined, trial-level log power, joint baseline z-score across six requested conditions, then condition mean",
        "distance_method": "1-Pearson correlation across 16 mean band features",
        "same_memory_color_distance": "(R1-red + R2-red + G1-green + G2-green) / 4",
        "different_memory_color_distance": "(R1-green + R2-green + G1-red + G2-red) / 4",
        "set_average_method": "direct arithmetic average of single-electrode distance matrices and curves",
        "inference": "descriptive only; no bootstrap or permutation test",
        "electrode_selection_table": str(args.selection_table.resolve()),
        "electrode_selection_signal": "100-400_lf30 membership held fixed; neural input is raw200",
        "actual_set_counts": summary["set_counts"],
        "unique_electrode_count": summary["unique_electrodes"],
        "subjects": list(args.subjects),
        "h5_audit": audits,
        "input_manifest": manifest,
        "git_commit": _git_commit(),
        "python": sys.version,
        "python_executable": sys.executable,
        "package_versions": _package_versions(),
        "run_dir": str(run_dir),
        "result_dir": str(result_dir),
    }
    (result_dir / "analysis_parameters.json").write_text(
        json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_readme(result_dir, summary)
    parameters["output_paths"] = [
        str(path) for path in sorted(result_dir.rglob("*")) if path.is_file()
    ]
    (result_dir / "analysis_parameters.json").write_text(
        json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", choices=["raw200"], default="raw200")
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=list(SUBJECTS))
    parser.add_argument(
        "--selection-table",
        type=Path,
        default=MODULE_ROOT
        / "result"
        / "final_analysis_seeg_20260806_corrected"
        / "stage01_selection"
        / "electrode_sets_and_csc_100-400_lf30.csv",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=MODULE_ROOT / "result" / "final_analysis_seeg_20260806_corrected",
    )
    parser.add_argument("--overwrite-stage09-3", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.signal != "raw200":
        raise ValueError("This stage is fixed to --signal raw200")
    subjects = tuple(args.subjects)
    selection_path = args.selection_table.resolve()
    selection, _ = _load_electrode_sets(selection_path, ("S1", "S2", "CSC"))
    selection = selection[selection.subject.isin(subjects)].copy()
    if selection.empty:
        raise ValueError("No selected electrodes remain for requested subjects")
    if set(subjects) == set(SUBJECTS):
        for name, expected in EXPECTED_SET_COUNTS.items():
            actual = int(selection[name].sum())
            if actual != expected:
                raise ValueError(f"Expected {expected} {name} electrodes, found {actual}")

    manifest = _input_manifest(selection_path, subjects)
    audits = []
    for subject in subjects:
        audits.append(_audit_h5(subject, 2, TASK2_CONDITIONS, {k: TRIGGERS[k] for k in TASK2_CONDITIONS}))
        audits.append(_audit_h5(subject, 3, TASK3_CONDITIONS, {"red": 51, "green": 54}))
    for audit in audits:
        if not audit["requested_conditions_present"] or not audit["requested_triggers_match"]:
            raise ValueError(f"HDF5 audit failed: {audit}")

    result_dir = (args.result_root / ANALYSIS_NAME).resolve()
    if result_dir.exists() and any(result_dir.iterdir()) and not args.overwrite_stage09_3:
        raise FileExistsError(
            f"Refusing to overwrite non-empty result directory: {result_dir}. "
            "Use --overwrite-stage09-3 only for an explicit rerun."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / f"{timestamp}_{ANALYSIS_NAME}"
    run_dir.mkdir(parents=True, exist_ok=False)
    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "python_executable": sys.executable,
                "package_versions": _package_versions(),
                "platform": sys.platform,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    start = datetime.now(timezone.utc)
    (run_dir / "start_time.txt").write_text(start.isoformat() + "\n", encoding="utf-8")
    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            summary = _run(selection, subjects, result_dir)
            _write_parameters(result_dir, args, summary, audits, manifest, run_dir)
            caught.extend(caught_warnings)
        end = datetime.now(timezone.utc)
        run_summary = {
            "status": "completed",
            "started_at_utc": start.isoformat(),
            "finished_at_utc": end.isoformat(),
            "stage": STAGE_KEY,
            "summary": summary,
            "result_dir": str(result_dir),
        }
        (run_dir / "run_summary.json").write_text(
            json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / "output_paths.txt").write_text(str(result_dir) + "\n", encoding="utf-8")
        print(json.dumps(run_summary, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "started_at_utc": start.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        (run_dir / "run_summary.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise
    finally:
        warning_lines = [
            f"{message.category.__name__}: {message.message} ({message.filename}:{message.lineno})"
            for message in caught
        ]
        (run_dir / "warnings.log").write_text(
            "\n".join(warning_lines) + ("\n" if warning_lines else ""), encoding="utf-8"
        )
        (run_dir / "finish_time.txt").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    raise SystemExit(main())
