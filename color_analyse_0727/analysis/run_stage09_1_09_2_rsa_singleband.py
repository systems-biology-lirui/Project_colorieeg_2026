"""Task2 gray-fruit and Task3 red/green RSA plus single-band decoding.

This adds two independent stages without modifying the existing Task1 stage09:

* stage09_1: four Task2 gray fruits, R1/R2/G1/G2 geometry and cross-fruit
  single-band decoding;
* stage09_2: Task3 pure red/green patches, red-green geometry and within-task
  single-band decoding.

All neural features are computed trial-wise from raw200 HDF5 epochs, then
baseline-z-scored and averaged only for the descriptive RDM outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from analysis.common import FS, SUBJECTS, h5_path, load_conditions, natural_key
from analysis.run_task1_condition_rsa import (
    EXPECTED_SET_COUNTS,
    FEATURE_BANDS,
    BAND_NAMES,
    SET_COLUMNS,
    _baseline_zscore,
    _bool_col,
    _correlation_rdm,
    _git_commit,
    _load_electrode_sets,
    _package_versions,
    _safe_name,
    _sha256,
    _timecourse_mean,
    _trial_logpower,
)


STAGE_CONFIG: dict[str, dict[str, Any]] = {
    "stage09_1": {
        "task": 2,
        "analysis_name": "stage09_1_task2_grayfruit_rsa_raw200",
        "result_suffix": "stage09_1_task2_grayfruit_rsa_raw200",
        "conditions": (
            "strawberry_gray",
            "watermelon_gray",
            "cabbage_gray",
            "kiwi_gray",
        ),
        "triggers": {
            "strawberry_gray": 123,
            "watermelon_gray": 133,
            "cabbage_gray": 103,
            "kiwi_gray": 113,
        },
        "short_names": ("R1", "R2", "G1", "G2"),
        "mode": "cross_fruit",
    },
    "stage09_2": {
        "task": 3,
        "analysis_name": "stage09_2_task3_purecolor_rsa_raw200",
        "result_suffix": "stage09_2_task3_purecolor_rsa_raw200",
        "conditions": ("red", "green"),
        "triggers": {"red": 51, "green": 54},
        "short_names": ("red", "green"),
        "mode": "within_task",
    },
}

BASELINE_MS = (-200.0, 0.0)
TIME_RANGE_MS = (0.0, 800.0)
BIN_MS = 50.0
TF_NPERSEG = 128
TF_HOP_MS = 10.0
DEFAULT_WORKERS = 20
TASK2_FOLDS = (
    ("strawberry_gray", "cabbage_gray", "watermelon_gray", "kiwi_gray"),
    ("strawberry_gray", "kiwi_gray", "watermelon_gray", "cabbage_gray"),
    ("watermelon_gray", "cabbage_gray", "strawberry_gray", "kiwi_gray"),
    ("watermelon_gray", "kiwi_gray", "strawberry_gray", "cabbage_gray"),
)
CONDITION_COLORS = {
    "strawberry_gray": "#d73027",
    "watermelon_gray": "#fc8d59",
    "cabbage_gray": "#4575b4",
    "kiwi_gray": "#74add1",
    "red": "#d73027",
    "green": "#4575b4",
}


def _input_manifest(selection_path: Path, subjects: tuple[str, ...], tasks: tuple[int, ...]) -> list[dict[str, Any]]:
    paths = [selection_path]
    paths.extend(h5_path(subject, task) for subject in subjects for task in tasks)
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


def _decode_scalar(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _h5_audit(subject: str, config: dict[str, Any]) -> dict[str, Any]:
    path = h5_path(subject, int(config["task"]))
    with h5py.File(path, "r") as handle:
        names = [_decode_scalar(x) for x in handle["condition_names"][()]]
        triggers = [_decode_scalar(x) for x in handle["condition_triggers"][()]]
        counts = [int(x) for x in handle["trial_counts"][()]]
        time_ms = np.asarray(handle["time_ms"][()], dtype=float)
        labels = [_decode_scalar(x).strip().upper() for x in handle["labels"][()]]
    trigger_map = {name: int(trigger) for name, trigger in zip(names, triggers)}
    expected = {str(name): int(value) for name, value in config["triggers"].items()}
    return {
        "subject": subject,
        "task": int(config["task"]),
        "path": str(path),
        "condition_names_in_hdf5": names,
        "condition_triggers_in_hdf5": trigger_map,
        "trial_counts": dict(zip(names, counts)),
        "n_channels": len(labels),
        "time_start_ms": float(time_ms[0]),
        "time_end_ms": float(time_ms[-1]),
        "n_timepoints": int(time_ms.size),
        "requested_conditions_present": set(config["conditions"]).issubset(set(names)),
        "requested_triggers_match": all(trigger_map.get(name) == trigger for name, trigger in expected.items()),
    }


def _time_bins() -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(TIME_RANGE_MS[0], TIME_RANGE_MS[1], BIN_MS, dtype=float)
    ends = starts + BIN_MS
    if len(starts) != 16:
        raise ValueError(f"Expected 16 time bins, got {len(starts)}")
    return starts, ends


def _fit_accuracy(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model = LinearSVC(C=1.0, max_iter=10000, dual=False)
    model.fit(x_train_scaled, y_train)
    return float(balanced_accuracy_score(y_test, model.predict(x_test_scaled)))


def _task2_single_band_accuracy(features: dict[str, np.ndarray], band_index: int, channel_index: int) -> list[float]:
    results: list[float] = []
    for train_red, train_green, test_red, test_green in TASK2_FOLDS:
        n_train = min(len(features[train_red]), len(features[train_green]))
        n_test = min(len(features[test_red]), len(features[test_green]))
        x_train = np.concatenate(
            [features[train_red][:n_train, band_index, None], features[train_green][:n_train, band_index, None]],
            axis=0,
        )
        x_test = np.concatenate(
            [features[test_red][:n_test, band_index, None], features[test_green][:n_test, band_index, None]],
            axis=0,
        )
        y_train = np.concatenate([np.ones(n_train, dtype=int), np.zeros(n_train, dtype=int)])
        y_test = np.concatenate([np.ones(n_test, dtype=int), np.zeros(n_test, dtype=int)])
        results.append(_fit_accuracy(x_train, y_train, x_test, y_test))
    return results


def _task3_single_band_accuracy(red: np.ndarray, green: np.ndarray, band_index: int, seed: int) -> list[float]:
    n = min(len(red), len(green))
    x = np.concatenate([red[:n, band_index, None], green[:n, band_index, None]], axis=0)
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    n_splits = min(5, n)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_index, test_index in splitter.split(x, y):
        scores.append(_fit_accuracy(x[train_index], y[train_index], x[test_index], y[test_index]))
    return scores


def _decode_channel(
    config: dict[str, Any],
    subject: str,
    channel: str,
    channel_index: int,
    trial_features_by_bin: list[dict[str, np.ndarray]],
    time_starts: np.ndarray,
    time_ends: np.ndarray,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conditions = tuple(config["conditions"])
    for bin_index, (bin_start, bin_end) in enumerate(zip(time_starts, time_ends)):
        condition_features = {
            condition: trial_features_by_bin[bin_index][condition][:, channel_index, :]
            for condition in conditions
        }
        for band_index, band_name in enumerate(BAND_NAMES):
            if config["mode"] == "cross_fruit":
                scores = _task2_single_band_accuracy(condition_features, band_index, channel_index)
                row = {
                    **metadata,
                    "condition_set": "task2_gray_fruit_crossfruit",
                    "time_bin_index": bin_index,
                    "bin_start_ms": float(bin_start),
                    "bin_end_ms": float(bin_end),
                    "band_index": band_index,
                    "band_name": band_name,
                    "band_low_hz": float(FEATURE_BANDS[band_index][0]),
                    "band_high_hz": float(FEATURE_BANDS[band_index][1]),
                    "accuracy_split_1": scores[0],
                    "accuracy_split_2": scores[1],
                    "accuracy_split_3": scores[2],
                    "accuracy_split_4": scores[3],
                    "mean_accuracy": float(np.mean(scores)),
                    "min_accuracy": float(np.min(scores)),
                    "sharedness_margin": float(np.min(scores) - 0.5),
                    "signal": "raw200",
                    "workers": None,
                }
            else:
                scores = _task3_single_band_accuracy(
                    condition_features["red"], condition_features["green"], band_index,
                    seed=9000 + bin_index * 100 + band_index,
                )
                row = {
                    **metadata,
                    "condition_set": "task3_red_green",
                    "time_bin_index": bin_index,
                    "bin_start_ms": float(bin_start),
                    "bin_end_ms": float(bin_end),
                    "band_index": band_index,
                    "band_name": band_name,
                    "band_low_hz": float(FEATURE_BANDS[band_index][0]),
                    "band_high_hz": float(FEATURE_BANDS[band_index][1]),
                    "accuracy_mean_cv": float(np.mean(scores)),
                    "accuracy_min_cv": float(np.min(scores)),
                    "n_cv_folds": len(scores),
                    "signal": "raw200",
                    "workers": None,
                }
            rows.append(row)
    return rows


def _plot_stage_figure(
    config: dict[str, Any],
    result: dict[str, Any],
    output_path: Path,
    time_starts: np.ndarray,
    time_ends: np.ndarray,
) -> None:
    n_conditions = len(config["conditions"])
    fig = plt.figure(figsize=(20, 20))
    grid = GridSpec(6, 4, figure=fig, height_ratios=[1.2, 1, 1, 1, 1, 0.95], hspace=0.8, wspace=0.55)
    ax_time = fig.add_subplot(grid[0, :])
    for condition in config["conditions"]:
        ax_time.plot(
            result["time_ms"], result["timecourses"][condition],
            linewidth=1.0, label=condition, color=CONDITION_COLORS[condition],
        )
    ax_time.set_xlim(*TIME_RANGE_MS)
    ax_time.axvline(0, color="black", linestyle="--", linewidth=0.7)
    ax_time.set_title("condition-mean baseline-subtracted time courses")
    ax_time.set_xlabel("time (ms)")
    ax_time.set_ylabel("amplitude")
    ax_time.legend(ncol=max(2, n_conditions), fontsize=8)
    ax_time.grid(alpha=0.2)

    rdm_axes = []
    image = None
    for bin_index, rdm in enumerate(result["rdms"]):
        row, col = divmod(bin_index, 4)
        ax = fig.add_subplot(grid[row + 1, col])
        rdm_axes.append(ax)
        image = ax.imshow(rdm, cmap="viridis", vmin=0.0, vmax=2.0, interpolation="nearest")
        ax.set_title(f"{time_starts[bin_index]:.0f}–{time_ends[bin_index]:.0f} ms", fontsize=8)
        ax.set_xticks(range(n_conditions))
        ax.set_yticks(range(n_conditions))
        ax.set_xticklabels(config["short_names"], rotation=90, fontsize=6)
        ax.set_yticklabels(config["short_names"], fontsize=6)
        if config["mode"] == "cross_fruit":
            ax.axvline(1.5, color="white", linewidth=0.8)
            ax.axhline(1.5, color="white", linewidth=0.8)
    if image is not None:
        fig.colorbar(image, ax=rdm_axes, shrink=0.58, label="correlation distance (1 − Pearson r)")

    ax_curve = fig.add_subplot(grid[5, :2])
    if config["mode"] == "cross_fruit":
        curve = result["distance_curve"]
        x = time_starts + BIN_MS / 2
        ax_curve.plot(x, curve["within"], color="#542788", marker="o", label="within memory color: (R1R2 + G1G2)/2")
        ax_curve.plot(x, curve["between"], color="#b35806", marker="o", label="between memory color: (R1G1 + R1G2 + R2G1 + R2G2)/4")
        ax_curve.set_ylabel("correlation distance")
        ax_curve.legend(fontsize=7)
    else:
        curve = result["distance_curve"]
        x = time_starts + BIN_MS / 2
        ax_curve.plot(x, curve["red_green"], color="#7f0000", marker="o", label="red–green distance")
        ax_curve.set_ylabel("correlation distance")
        ax_curve.legend(fontsize=8)
    ax_curve.set_xlabel("time-bin center (ms)")
    ax_curve.set_title("condition geometry over time")
    ax_curve.grid(alpha=0.2)

    ax_dec = fig.add_subplot(grid[5, 2:])
    decoding = result["decoding_heatmap"]
    image_dec = ax_dec.imshow(decoding.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=0.0, vmax=1.0)
    ax_dec.set_xticks(range(len(time_starts)))
    ax_dec.set_xticklabels([f"{x:.0f}" for x in time_starts], rotation=90, fontsize=6)
    ax_dec.set_yticks(range(len(BAND_NAMES)))
    ax_dec.set_yticklabels(BAND_NAMES, fontsize=6)
    ax_dec.set_xlabel("bin start (ms)")
    ax_dec.set_ylabel("frequency band")
    ax_dec.set_title("single-band decoding accuracy")
    fig.colorbar(image_dec, ax=ax_dec, shrink=0.8, label="balanced accuracy")

    meta = result["metadata"]
    fig.suptitle(
        f"{config['analysis_name']} | {meta['subject']} {meta['channel']} | "
        f"sets={meta['electrode_sets']} | MNI=({meta['mni_x']:.1f}, {meta['mni_y']:.1f}, {meta['mni_z']:.1f})",
        fontsize=14,
        y=0.995,
    )
    fig.text(0.5, 0.978, "trial-level raw200 STFT → global -200–0 ms baseline z-score → condition mean RDM; decoding is trial-level single-band", ha="center", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_set_figure(
    config: dict[str, Any],
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
    curve_keys = ("within", "between") if config["mode"] == "cross_fruit" else ("red_green",)
    curve = {key: np.nanmean(np.stack([r["distance_curve"][key] for r in results], axis=0), axis=0) for key in curve_keys}
    decoding = np.nanmean(np.stack([r["decoding_heatmap"] for r in results], axis=0), axis=0)
    proxy = {
        "time_ms": results[0]["time_ms"],
        "timecourses": {
            condition: np.nanmean(np.stack([r["timecourses"][condition] for r in results], axis=0), axis=0)
            for condition in config["conditions"]
        },
        "rdms": rdms,
        "distance_curve": curve,
        "decoding_heatmap": decoding,
        "metadata": {
            "subject": f"{len(results)} electrodes",
            "channel": set_name,
            "electrode_sets": set_name,
            "mni_x": np.nan,
            "mni_y": np.nan,
            "mni_z": np.nan,
        },
    }
    _plot_stage_figure(config, proxy, output_path, time_starts, time_ends)


def _write_readme(result_dir: Path, config: dict[str, Any], summary: dict[str, Any]) -> None:
    if config["mode"] == "cross_fruit":
        distance_text = "within = (R1R2 + G1G2)/2；between = (R1G1 + R1G2 + R2G1 + R2G2)/4。"
        condition_text = "R1=strawberry_gray、R2=watermelon_gray、G1=cabbage_gray、G2=kiwi_gray。"
    else:
        distance_text = "每个时间窗计算 red 与 green 两个条件的 1−Pearson correlation distance。"
        condition_text = "只分析 Task3 的 red（trigger 51）与 green（trigger 54）纯色色块。"
    lines = [
        f"# {config['analysis_name']}",
        "",
        "> 独立新增结果，不覆盖现有 stage09 Task1 结果。",
        "",
        "## 方法",
        "",
        "- 输入为 raw200 HDF5；每个 trial 先在完整 −500–1000 ms epoch 上做 STFT。",
        "- log-power 相对于 −200–0 ms、该被试全部请求条件的 trial 做 baseline z-score，然后在条件内平均得到 RDM。",
        "- 使用 16 个既有频带和 0–800 ms 的 16 个连续 50 ms 时间窗。",
        f"- 条件：{condition_text}",
        f"- 距离：{distance_text}",
        "- 单频段 decoding 对每个电极、每个频带、每个时间窗独立进行；只报告 balanced accuracy，不做置换显著性检验。",
        "- Task2 使用四个无水果重叠的跨水果训练/测试方向；Task3 使用 red-vs-green 的五折 trial-level CV。",
        "- 图中每个电极包含四/两条条件均值时间信号、16 个 RDM 热图、距离曲线和单频段 decoding 热图。完整数值保存在 CSV，不把每个频段单独输出成图片。",
        "",
        "## 解释边界",
        "",
        "该结果是条件平均层面的几何和逐频段解码描述。Task2 四种水果中每种记忆颜色只有两种水果，因此跨水果泛化支持共同结构，但不能完全排除水果语义差异；Task3 red-green 是物理颜色参照，不等同于跨任务共享表征检验。",
        "",
        "## 摘要",
        f"- unique electrodes: {summary['unique_electrodes']}",
        f"- set counts: {json.dumps(summary['set_counts'], ensure_ascii=False)}",
        f"- workers: {summary['workers']}",
        f"- output directory: `{result_dir}`",
        "",
        "完整命令、输入 SHA256、Git commit、Python 环境、时间和 warnings 保存在项目根目录 `runs/` 的对应运行目录。",
    ]
    (result_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_stage(
    stage_key: str,
    config: dict[str, Any],
    selection: pd.DataFrame,
    subjects: tuple[str, ...],
    result_dir: Path,
    workers: int,
) -> dict[str, Any]:
    time_starts, time_ends = _time_bins()
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "electrode_figures").mkdir(exist_ok=True)
    (result_dir / "set_figures").mkdir(exist_ok=True)
    metadata_by_key = {
        (str(row.subject), str(row.channel)): row._asdict()
        for row in selection.itertuples(index=False)
    }
    unique_keys = list(metadata_by_key)
    channels_by_subject = {
        subject: sorted([channel for sub, channel in unique_keys if sub == subject], key=natural_key)
        for subject in subjects
    }
    electrode_results: dict[tuple[str, str], dict[str, Any]] = {}
    timecourse_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    rdm_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    decoding_rows: list[dict[str, Any]] = []
    condition_counts: list[dict[str, Any]] = []

    for subject in subjects:
        channels = channels_by_subject[subject]
        if not channels:
            continue
        raw_by_condition, time_ms, returned_channels = load_conditions(
            subject, int(config["task"]), config["conditions"], channels
        )
        if [c.upper() for c in returned_channels] != [c.upper() for c in channels]:
            raise ValueError(f"Channel order changed for {subject}")
        stacked = np.concatenate([raw_by_condition[c] for c in config["conditions"]], axis=0)
        trial_logpower, frame_times = _trial_logpower(stacked)
        trial_z = _baseline_zscore(trial_logpower, frame_times, BASELINE_MS)
        offsets: dict[str, tuple[int, int]] = {}
        offset = 0
        for condition in config["conditions"]:
            n = raw_by_condition[condition].shape[0]
            offsets[condition] = (offset, offset + n)
            offset += n

        trial_features_by_bin: list[dict[str, np.ndarray]] = []
        for bin_index, (start_ms, end_ms) in enumerate(zip(time_starts, time_ends)):
            frame_mask = (frame_times >= start_ms) & (frame_times < end_ms)
            if not frame_mask.any():
                raise ValueError(f"No TF frames in {start_ms}-{end_ms} ms")
            trial_features: dict[str, np.ndarray] = {}
            for condition in config["conditions"]:
                start, stop = offsets[condition]
                values = trial_z[start:stop, :, :, frame_mask]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    trial_features[condition] = np.nanmean(values, axis=3).astype(np.float32)
                condition_counts.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "trigger": int(config["triggers"][condition]),
                        "n_trials": int(stop - start),
                        "n_valid_trials": int(np.isfinite(trial_features[condition]).all(axis=(1, 2)).sum()),
                        "stage": stage_key,
                    }
                )
            trial_features_by_bin.append(trial_features)

        # The same feature arrays feed both trial-level decoding and the
        # condition-mean RDM, so no second TF transform is performed.  The
        # channel jobs are the 20-thread parallel unit requested by the user.
        channel_decoding_rows = Parallel(n_jobs=min(workers, 20), prefer="threads")(
            delayed(_decode_channel)(
                config,
                subject,
                channel,
                channel_index,
                trial_features_by_bin,
                time_starts,
                time_ends,
                metadata_by_key[(subject, channel)],
            )
            for channel_index, channel in enumerate(channels)
        )

        for channel_index, channel in enumerate(channels):
            key = (subject, channel)
            meta = metadata_by_key[key]
            timecourses = {
                condition: _timecourse_mean(
                    raw_by_condition[condition][:, channel_index : channel_index + 1, :],
                    time_ms,
                    BASELINE_MS,
                )[0]
                for condition in config["conditions"]
            }
            for condition_index, condition in enumerate(config["conditions"]):
                for time_index, time_value in enumerate(time_ms):
                    timecourse_rows.append(
                        {
                            **meta,
                            "stage": stage_key,
                            "condition": condition,
                            "condition_index": condition_index,
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
                        np.nanmean(trial_features_by_bin[bin_index][condition][:, channel_index, :], axis=0)
                        for condition in config["conditions"]
                    ],
                    dtype=float,
                )
                for condition_index, condition in enumerate(config["conditions"]):
                    for band_index, band_name in enumerate(BAND_NAMES):
                        feature_rows.append(
                            {
                                **meta,
                                "stage": stage_key,
                                "condition": condition,
                                "condition_index": condition_index,
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
                for i, condition_i in enumerate(config["conditions"]):
                    for j, condition_j in enumerate(config["conditions"]):
                        rdm_rows.append(
                            {
                                **meta,
                                "stage": stage_key,
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
                if config["mode"] == "cross_fruit":
                    within = float(np.nanmean([rdm[0, 1], rdm[2, 3]]))
                    between = float(np.nanmean([rdm[0, 2], rdm[0, 3], rdm[1, 2], rdm[1, 3]]))
                    curve_row = {
                        **meta,
                        "stage": stage_key,
                        "time_bin_index": bin_index,
                        "bin_start_ms": float(start_ms),
                        "bin_end_ms": float(end_ms),
                        "within_memory_color_distance": within,
                        "between_memory_color_distance": between,
                        "between_minus_within": between - within,
                    }
                else:
                    curve_row = {
                        **meta,
                        "stage": stage_key,
                        "time_bin_index": bin_index,
                        "bin_start_ms": float(start_ms),
                        "bin_end_ms": float(end_ms),
                        "red_green_distance": float(rdm[0, 1]),
                    }
                curve_rows.append(curve_row)

            channel_rows = channel_decoding_rows[channel_index]
            decoding_rows.extend(channel_rows)
            distance_rows.extend(curve_rows)
            decoding_heatmap = _decoding_heatmap(config, channel_rows)
            electrode_results[key] = {
                "metadata": meta,
                "time_ms": np.asarray(time_ms, dtype=float),
                "timecourses": timecourses,
                "rdms": np.asarray(rdms, dtype=float),
                "distance_curve": _curve_dict(config, curve_rows),
                "decoding_heatmap": decoding_heatmap,
            }
            _plot_stage_figure(
                config,
                electrode_results[key],
                result_dir / "electrode_figures" / f"{_safe_name(subject)}_{_safe_name(channel)}_{stage_key}.png",
                time_starts,
                time_ends,
            )
        print(f"completed {stage_key} {subject}: {len(channels)} electrodes", flush=True)

    if not electrode_results:
        raise ValueError(f"No results produced for {stage_key}")
    pd.DataFrame(timecourse_rows).to_csv(result_dir / "condition_mean_timecourses.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_rows).to_csv(result_dir / "condition_mean_tf_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rdm_rows).to_csv(result_dir / "condition_rdm_long.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(distance_rows).to_csv(result_dir / "memory_color_distance_curves.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(decoding_rows).to_csv(result_dir / "single_band_decoding.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(condition_counts).drop_duplicates().to_csv(result_dir / "condition_trial_counts.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(result_dir / "electrode_sets_used.csv", index=False, encoding="utf-8-sig")

    set_counts: dict[str, int] = {}
    for set_name in SET_COLUMNS:
        members = [r for r in electrode_results.values() if bool(r["metadata"].get(set_name, False))]
        set_counts[set_name] = len(members)
        if members:
            _plot_set_figure(
                config,
                set_name,
                members,
                result_dir / "set_figures" / f"{set_name}_mean_{stage_key}.png",
                time_starts,
                time_ends,
            )
    summary = {
        "analysis": config["analysis_name"],
        "stage": stage_key,
        "task": int(config["task"]),
        "unique_electrodes": len(electrode_results),
        "set_counts": set_counts,
        "conditions": list(config["conditions"]),
        "condition_triggers": config["triggers"],
        "time_bins": len(time_starts),
        "feature_bands": len(FEATURE_BANDS),
        "workers": int(workers),
        "output_dir": str(result_dir),
    }
    return {"summary": summary, "electrode_results": electrode_results}


def _curve_dict(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if config["mode"] == "cross_fruit":
        return {
            "within": np.asarray([row["within_memory_color_distance"] for row in rows], dtype=float),
            "between": np.asarray([row["between_memory_color_distance"] for row in rows], dtype=float),
        }
    return {"red_green": np.asarray([row["red_green_distance"] for row in rows], dtype=float)}


def _decoding_heatmap(config: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    heatmap = np.full((len(FEATURE_BANDS), 16), np.nan, dtype=float)
    for row in rows:
        if config["mode"] == "cross_fruit":
            value = row["mean_accuracy"]
        else:
            value = row["accuracy_mean_cv"]
        heatmap[int(row["band_index"]), int(row["time_bin_index"])] = float(value)
    return heatmap


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", choices=["raw200"], default="raw200")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGE_CONFIG), default=list(STAGE_CONFIG))
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=list(SUBJECTS))
    parser.add_argument(
        "--selection-table",
        type=Path,
        default=MODULE_ROOT / "result" / "final_analysis_seeg_20260806_corrected" / "stage01_selection" / "electrode_sets_and_csc_100-400_lf30.csv",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=MODULE_ROOT / "result" / "final_analysis_seeg_20260806_corrected",
    )
    parser.add_argument("--overwrite-stage09-substages", action="store_true")
    return parser.parse_args(argv)


def _write_stage_parameters(
    result_dir: Path,
    config: dict[str, Any],
    stage_key: str,
    args: argparse.Namespace,
    summary: dict[str, Any],
    audits: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    run_dir: Path,
) -> None:
    parameters = {
        "analysis": config["analysis_name"],
        "stage": stage_key,
        "task": int(config["task"]),
        "signal": args.signal,
        "conditions": list(config["conditions"]),
        "condition_triggers": config["triggers"],
        "baseline_ms": list(BASELINE_MS),
        "time_range_ms": list(TIME_RANGE_MS),
        "bin_ms": BIN_MS,
        "tf_method": "scipy.signal.stft on every trial over full epoch",
        "tf_nperseg_samples": TF_NPERSEG,
        "tf_hop_ms": TF_HOP_MS,
        "feature_bands_hz": [list(band) for band in FEATURE_BANDS],
        "feature_transform": "trial-level log power, baseline z-score across requested conditions/trials, then condition mean for RDM",
        "distance_method": "1-Pearson correlation across 16 condition-mean band features",
        "task2_distance_curves": {
            "within": "(R1R2 + G1G2)/2",
            "between": "(R1G1 + R1G2 + R2G1 + R2G2)/4",
        },
        "single_band_decoding": {
            "task2": "four fruit-disjoint cross-fruit directions",
            "task3": "red-vs-green five-fold trial-level CV",
            "classifier": "StandardScaler + LinearSVC(C=1, dual=False)",
            "metric": "balanced accuracy",
            "permutation": "none; descriptive only",
        },
        "electrode_selection_table": str(args.selection_table.resolve()),
        "electrode_selection_signal": "100-400_lf30 membership held fixed; neural input is raw200",
        "expected_set_counts": EXPECTED_SET_COUNTS,
        "actual_set_counts": summary["set_counts"],
        "unique_electrode_count": summary["unique_electrodes"],
        "workers_requested": int(args.workers),
        "worker_backend": "joblib threads for single-band decoding",
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
    (result_dir / "analysis_parameters.json").write_text(json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_readme(result_dir, config, {**summary, "workers": int(args.workers)})
    parameters["output_paths"] = [str(path) for path in sorted(result_dir.rglob("*")) if path.is_file()]
    (result_dir / "analysis_parameters.json").write_text(json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.workers != 20:
        raise ValueError("This requested run uses exactly --workers 20")
    if args.signal != "raw200":
        raise ValueError("This requested run uses --signal raw200")
    subjects = tuple(args.subjects)
    selection_path = args.selection_table.resolve()
    selection, _ = _load_electrode_sets(selection_path, tuple(SET_COLUMNS))
    selection = selection[selection.subject.isin(subjects)].copy()
    for name, expected in EXPECTED_SET_COUNTS.items():
        if int(selection[name].sum()) != expected:
            raise ValueError(f"Expected {expected} {name} electrodes, found {int(selection[name].sum())}")
    configs = {stage: STAGE_CONFIG[stage] for stage in args.stages}
    tasks = tuple(sorted({int(config["task"]) for config in configs.values()}))
    manifest = _input_manifest(selection_path, subjects, tasks)
    audits = [_h5_audit(subject, config) for config in configs.values() for subject in subjects]
    for audit in audits:
        if not audit["requested_conditions_present"] or not audit["requested_triggers_match"]:
            raise ValueError(f"HDF5 audit failed: {audit}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / f"{timestamp}_stage09_1_09_2_rsa_singleband"
    run_dir.mkdir(parents=True, exist_ok=False)
    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps({"python": sys.version, "python_executable": sys.executable, "package_versions": _package_versions(), "platform": sys.platform}, indent=2),
        encoding="utf-8",
    )
    start = datetime.now(timezone.utc)
    (run_dir / "start_time.txt").write_text(start.isoformat() + "\n", encoding="utf-8")
    caught: list[warnings.WarningMessage] = []
    stage_summaries: dict[str, Any] = {}
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            for stage_key, config in configs.items():
                result_dir = (args.result_root / config["result_suffix"]).resolve()
                if result_dir.exists() and any(result_dir.iterdir()) and not args.overwrite_stage09_substages:
                    raise FileExistsError(f"Refusing to overwrite non-empty result directory: {result_dir}")
                run_result = _run_stage(stage_key, config, selection, subjects, result_dir, args.workers)
                _write_stage_parameters(result_dir, config, stage_key, args, run_result["summary"], [a for a in audits if a["task"] == config["task"]], manifest, run_dir)
                stage_summaries[stage_key] = run_result["summary"]
            caught.extend(caught_warnings)
        end = datetime.now(timezone.utc)
        summary = {
            "status": "completed",
            "started_at_utc": start.isoformat(),
            "finished_at_utc": end.isoformat(),
            "workers": int(args.workers),
            "stages": stage_summaries,
            "result_dirs": [str((args.result_root / configs[k]["result_suffix"]).resolve()) for k in configs],
        }
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "output_paths.txt").write_text("\n".join(summary["result_dirs"]) + "\n", encoding="utf-8")
        print(json.dumps({"status": "completed", "run_dir": str(run_dir), "stages": stage_summaries}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "started_at_utc": start.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        (run_dir / "run_summary.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    finally:
        warning_lines = [f"{m.category.__name__}: {m.message} ({m.filename}:{m.lineno})" for m in caught]
        (run_dir / "warnings.log").write_text("\n".join(warning_lines) + ("\n" if warning_lines else ""), encoding="utf-8")
        (run_dir / "finish_time.txt").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
