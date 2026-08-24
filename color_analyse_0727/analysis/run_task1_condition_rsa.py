"""Task1 condition-mean time-frequency RSA using the raw200 HDF5 signal.

This is an independent descriptive analysis.  It deliberately does not touch
stage01--08 and computes a trial-level time-frequency representation before
averaging trials within each of Task1's eight conditions.
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
from matplotlib.gridspec import GridSpec
from scipy.signal import stft

from analysis.common import FS, SUBJECTS, h5_path, load_conditions, natural_key
from pipeline.spectral_features import BAND_NAMES, FEATURE_BANDS


ANALYSIS_NAME = "stage09_task1_condition_rsa_raw200"
CONDITIONS: tuple[str, ...] = (
    "face_color",
    "object_color",
    "body_color",
    "place_color",
    "face_gray",
    "object_gray",
    "body_gray",
    "place_gray",
)
CONDITION_TRIGGERS: dict[str, int] = {
    "face_color": 11,
    "face_gray": 12,
    "object_color": 21,
    "object_gray": 22,
    "body_color": 31,
    "body_gray": 32,
    "place_color": 41,
    "place_gray": 42,
}
SET_COLUMNS = {"S1": "strategy1", "S2": "strategy2", "CSC": "CSC"}
EXPECTED_SET_COUNTS = {"S1": 19, "S2": 46, "CSC": 12}
TF_NPERSEG = 128
TF_HOP_MS = 10.0
SHORT_CONDITION_NAMES = (
    "face-C",
    "object-C",
    "body-C",
    "place-C",
    "face-G",
    "object-G",
    "body-G",
    "place-G",
)
CONDITION_COLORS = (
    "#d73027",
    "#fc8d59",
    "#91bfdb",
    "#4575b4",
    "#7f7f7f",
    "#969696",
    "#bdbdbd",
    "#525252",
)


def _parse_float_pair(values: list[str], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f"{name} requires two numbers")
    start, end = float(values[0]), float(values[1])
    if end <= start:
        raise argparse.ArgumentTypeError(f"{name} end must be greater than start")
    return start, end


def _bool_col(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception as exc:  # pragma: no cover - provenance fallback
        return f"unavailable: {exc}"


def _package_versions() -> dict[str, str]:
    names = ("numpy", "scipy", "pandas", "h5py", "matplotlib", "scikit-learn")
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def _input_manifest(selection_path: Path, subjects: tuple[str, ...]) -> list[dict[str, Any]]:
    paths = [selection_path, *[h5_path(subject, 1) for subject in subjects]]
    manifest = []
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


def _read_h5_audit(subject: str) -> dict[str, Any]:
    path = h5_path(subject, 1)
    with h5py.File(path, "r") as handle:
        names = [str(x.decode() if isinstance(x, bytes) else x) for x in handle["condition_names"][()]]
        triggers = [int(x) for x in handle["condition_triggers"][()]]
        counts = [int(x) for x in handle["trial_counts"][()]]
        labels = [str(x.decode() if isinstance(x, bytes) else x).strip().upper() for x in handle["labels"][()]]
        time_ms = np.asarray(handle["time_ms"][()], dtype=float)
    trigger_map = dict(zip(names, triggers))
    return {
        "subject": subject,
        "path": str(path),
        "condition_names": names,
        "condition_triggers": trigger_map,
        "trial_counts": dict(zip(names, counts)),
        "n_channels": len(labels),
        "time_start_ms": float(time_ms[0]),
        "time_end_ms": float(time_ms[-1]),
        "n_timepoints": int(time_ms.size),
        # HDF5 stores its own registry order (currently color/gray within each
        # category); the analysis reorders conditions to the requested
        # color-first four + gray-four order when loading them.
        "condition_names_match": set(names) == set(CONDITIONS),
        "condition_order_in_hdf5": names,
        "trigger_mapping_matches": all(trigger_map.get(k) == v for k, v in CONDITION_TRIGGERS.items()),
    }


def _load_electrode_sets(selection_path: Path, requested_sets: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, list[tuple[str, str]]]]:
    table = pd.read_csv(selection_path)
    required = {"subject", "channel", "mni_x", "mni_y", "mni_z", "roi", *SET_COLUMNS.values()}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Selection table is missing columns: {sorted(missing)}")

    selected = table.copy()
    for name in SET_COLUMNS:
        selected[name] = selected[SET_COLUMNS[name]].map(_bool_col)
    selected = selected[selected[list(requested_sets)].any(axis=1)].copy()
    selected["subject"] = selected["subject"].astype(str)
    selected["channel"] = selected["channel"].astype(str).str.strip().str.upper()
    selected = selected.sort_values(["subject", "channel"], key=lambda col: col.map(natural_key) if col.name == "channel" else col)

    # A channel appears once in the current table.  Grouping here makes the
    # membership audit robust to future tables that contain duplicate rows.
    records: list[dict[str, Any]] = []
    for (subject, channel), group in selected.groupby(["subject", "channel"], sort=False):
        first = group.iloc[0]
        membership = [name for name in requested_sets if bool(group[name].any())]
        records.append(
            {
                "subject": subject,
                "channel": channel,
                "mni_x": float(first["mni_x"]),
                "mni_y": float(first["mni_y"]),
                "mni_z": float(first["mni_z"]),
                "roi": "" if pd.isna(first["roi"]) else str(first["roi"]),
                "electrode_sets": ";".join(membership),
                **{name: bool(group[name].any()) for name in requested_sets},
            }
        )
    unique = pd.DataFrame(records)
    if unique.empty:
        raise ValueError("No electrodes were selected")
    memberships = {
        name: [(str(row.subject), str(row.channel)) for row in unique.itertuples() if bool(getattr(row, name))]
        for name in requested_sets
    }
    return unique, memberships


def _trial_logpower(epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute full-epoch trial-level log power for all channels and bands."""
    values = np.asarray(epochs, dtype=np.float64)
    n_trials, n_channels, _ = values.shape
    hop = int(round(TF_HOP_MS / 1000.0 * FS))
    parts: list[np.ndarray] = []
    frame_times_ms: np.ndarray | None = None
    for start in range(0, n_channels, 8):
        stop = min(start + 8, n_channels)
        freqs, frame_times, coefficients = stft(
            values[:, start:stop, :],
            fs=FS,
            nperseg=TF_NPERSEG,
            noverlap=TF_NPERSEG - hop,
            axis=-1,
            boundary=None,
        )
        frame_times_ms = np.asarray(frame_times * 1000.0 - 500.0, dtype=float)
        power = np.abs(coefficients) ** 2
        part = np.full(
            (n_trials, stop - start, len(FEATURE_BANDS), power.shape[-1]),
            np.nan,
            dtype=np.float32,
        )
        for band_index, (low, high) in enumerate(FEATURE_BANDS):
            frequency_mask = (freqs >= low) & (freqs < high)
            if frequency_mask.any():
                part[:, :, band_index, :] = np.log(
                    np.nanmean(power[:, :, frequency_mask, :], axis=2) + 1e-12
                ).astype(np.float32)
        parts.append(part)
    if frame_times_ms is None:
        raise ValueError("No TF frames were produced")
    return np.concatenate(parts, axis=1), frame_times_ms


def _baseline_zscore(logpower: np.ndarray, frame_times_ms: np.ndarray, baseline: tuple[float, float]) -> np.ndarray:
    mask = (frame_times_ms >= baseline[0]) & (frame_times_ms <= baseline[1])
    if not mask.any():
        raise ValueError(f"Baseline {baseline} does not overlap TF frames")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(logpower[:, :, :, mask], axis=(0, 3), keepdims=True)
        std = np.nanstd(logpower[:, :, :, mask], axis=(0, 3), keepdims=True)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return np.asarray((logpower - mean) / std, dtype=np.float32)


def _correlation_rdm(features: np.ndarray) -> np.ndarray:
    """Return an 8x8 1-Pearson-r RDM from condition x band features."""
    values = np.asarray(features, dtype=float)
    n_conditions = values.shape[0]
    rdm = np.zeros((n_conditions, n_conditions), dtype=float)
    for i in range(n_conditions):
        for j in range(i + 1, n_conditions):
            valid = np.isfinite(values[i]) & np.isfinite(values[j])
            if valid.sum() < 2:
                distance = np.nan
            else:
                left = values[i, valid]
                right = values[j, valid]
                left = left - left.mean()
                right = right - right.mean()
                denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                distance = np.nan if denominator < 1e-12 else 1.0 - float(np.dot(left, right) / denominator)
                if np.isfinite(distance):
                    distance = float(np.clip(distance, 0.0, 2.0))
            rdm[i, j] = distance
            rdm[j, i] = distance
    np.fill_diagonal(rdm, 0.0)
    return rdm


def _timecourse_mean(epochs: np.ndarray, time_ms: np.ndarray, baseline: tuple[float, float]) -> np.ndarray:
    base_mask = (time_ms >= baseline[0]) & (time_ms <= baseline[1])
    if not base_mask.any():
        raise ValueError(f"Time-domain baseline {baseline} does not overlap the epoch")
    data = np.asarray(epochs, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = data - np.nanmean(data[..., base_mask], axis=-1, keepdims=True)
        return np.asarray(np.nanmean(data, axis=0), dtype=np.float32)


def _metadata_by_key(selection: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.subject), str(row.channel)): row._asdict()
        for row in selection.itertuples(index=False)
    }


def _plot_rdm_grid(
    rdms: np.ndarray,
    output_path: Path,
    title: str,
    subtitle: str = "",
) -> None:
    values = np.asarray(rdms, dtype=float)
    fig, axes = plt.subplots(4, 4, figsize=(17, 16), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    image = None
    for index, ax in enumerate(axes_flat):
        image = ax.imshow(values[index], cmap="viridis", vmin=0.0, vmax=2.0, interpolation="nearest")
        ax.set_title(f"{index * 50:.0f}–{index * 50 + 50:.0f} ms", fontsize=9)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(SHORT_CONDITION_NAMES, rotation=90, fontsize=5)
        ax.set_yticklabels(SHORT_CONDITION_NAMES, fontsize=5)
        ax.axvline(3.5, color="white", linewidth=0.8)
        ax.axhline(3.5, color="white", linewidth=0.8)
        ax.set_xlabel("color → gray", fontsize=6)
        ax.set_ylabel("color → gray", fontsize=6)
    if image is not None:
        fig.colorbar(image, ax=axes_flat.tolist(), shrink=0.72, label="correlation distance (1 − Pearson r)")
    fig.suptitle(title, fontsize=15, y=1.045)
    if subtitle:
        fig.text(0.5, 1.018, subtitle, ha="center", va="bottom", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_electrode(
    result: dict[str, Any],
    output_path: Path,
    time_range: tuple[float, float],
) -> None:
    fig = plt.figure(figsize=(19, 19))
    grid = GridSpec(5, 4, figure=fig, height_ratios=[1.2, 1, 1, 1, 1], hspace=0.72, wspace=0.5)
    time_axis = result["time_ms"]
    time_ax = fig.add_subplot(grid[0, :])
    for index, condition in enumerate(CONDITIONS):
        time_ax.plot(time_axis, result["timecourses"][condition], color=CONDITION_COLORS[index], linewidth=1.0, label=condition)
    time_ax.axvline(0.0, color="black", linewidth=0.7, linestyle="--")
    time_ax.set_xlim(*time_range)
    time_ax.set_xlabel("time (ms)")
    time_ax.set_ylabel("baseline-subtracted amplitude")
    time_ax.set_title("Task1 condition-mean single-channel time courses")
    time_ax.legend(ncol=4, fontsize=7, loc="upper right")
    time_ax.grid(alpha=0.18)

    axes = []
    image = None
    for index, rdm in enumerate(result["rdms"]):
        row, col = divmod(index, 4)
        ax = fig.add_subplot(grid[row + 1, col])
        axes.append(ax)
        image = ax.imshow(rdm, cmap="viridis", vmin=0.0, vmax=2.0, interpolation="nearest")
        ax.set_title(f"{index * 50:.0f}–{index * 50 + 50:.0f} ms", fontsize=8)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(SHORT_CONDITION_NAMES, rotation=90, fontsize=4.5)
        ax.set_yticklabels(SHORT_CONDITION_NAMES, fontsize=4.5)
        ax.axvline(3.5, color="white", linewidth=0.7)
        ax.axhline(3.5, color="white", linewidth=0.7)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.58, label="1 − Pearson r")
    metadata = result["metadata"]
    fig.suptitle(
        f"{metadata['subject']} {metadata['channel']} | sets={metadata['electrode_sets']} | "
        f"MNI=({metadata['mni_x']:.1f}, {metadata['mni_y']:.1f}, {metadata['mni_z']:.1f})",
        fontsize=14,
        y=0.995,
    )
    fig.text(0.5, 0.978, "raw200 trial-level log-power TF → baseline z-score → condition mean → correlation-distance RDM", ha="center", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_readme(result_dir: Path, parameters: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# Stage09 Task1 条件均值时频 RSA（raw200）",
        "",
        "> 这是独立的描述性分析，不覆盖 stage01–08。",
        "",
        "## 分析定义",
        "",
        "- 输入：Task1 HDF5 中已经完成 1–200 Hz 预处理的 epoch（raw200）。",
        "- 条件顺序：face_color, object_color, body_color, place_color, face_gray, object_gray, body_gray, place_gray。",
        "- 每个 trial 先做完整 -500–1000 ms epoch 的 STFT，再以每个 trial 的 log-power 相对于 -200–0 ms 做 baseline z-score；baseline 的均值和标准差在该被试全部八个条件的 trial 上估计，避免使用条件标签估计 baseline。",
        "- 条件均值在 trial 级 TF 特征上计算，然后按连续 50 ms 时间窗汇总 16 个频带。时间信号图也按 trial 先做 -200–0 ms 基线扣除，再在条件内平均。",
        "- 每个时间窗的 RDM 使用 16 个频带特征，距离为 `1 - Pearson correlation`；对角线为 0。",
        "- 集合平均对电极的距离矩阵直接取算术平均，不做 Fisher z；因此集合图是电极距离的平均描述，不是单电极推断。",
        "",
        "## 解释边界",
        "",
        "这些结果表示条件平均层面的神经表征几何结构。条件平均不能排除图片 exemplar、trial 构成或被试内样本方差的影响，因此不能直接解释为已经消除了图片身份或 trial variance。`Data/` 目录仅用于实验 trial 身份和触发器审计，本分析的神经特征直接来自 HDF5。",
        "",
        "## 输出",
        "",
        "- `electrode_figures/`：每个唯一 subject-channel 一张时间信号 + 16 个 RDM 热图。",
        "- `set_figures/`：S1、S2、CSC 的电极距离矩阵平均图。",
        "- `condition_mean_timecourses.csv`：条件均值时间信号。",
        "- `condition_mean_tf_features.csv`：条件均值、50 ms 分箱、16 频带 TF 特征。",
        "- `condition_rdm_long.csv`：每个电极和时间窗的完整 8×8 RDM。",
        "- `condition_trial_counts.csv`：每个被试、电极、条件的有效 trial 数和 QC 状态。",
        "- `electrode_sets_used.csv`：去重后的电极及 S1/S2/CSC 归属。",
        "- `rsa_parameters.json`：参数、触发器、输入和输出记录。",
        "",
        "## 运行摘要",
        "",
        f"- unique electrodes: {summary['unique_electrodes']}",
        f"- RDM time bins: {summary['rdm_time_bins']}",
        f"- set counts: {json.dumps(summary['set_counts'], ensure_ascii=False)}",
        f"- output path: `{result_dir}`",
        "",
        "完整的命令、输入 SHA256、Git commit、环境和 warnings 保存在项目根目录 `runs/` 下对应的本次运行目录。",
    ]
    (result_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", choices=["raw200"], default="raw200")
    parser.add_argument("--time-range", nargs=2, default=["0", "800"], metavar=("START", "END"))
    parser.add_argument("--bin-ms", type=float, default=50.0)
    parser.add_argument("--baseline", nargs=2, default=["-200", "0"], metavar=("START", "END"))
    parser.add_argument("--electrode-set", nargs="+", choices=tuple(SET_COLUMNS), default=list(SET_COLUMNS))
    parser.add_argument("--distance", choices=["correlation"], default="correlation")
    parser.add_argument(
        "--selection-table",
        type=Path,
        default=MODULE_ROOT / "result" / "final_analysis_seeg_20260806_corrected" / "stage01_selection" / "electrode_sets_and_csc_100-400_lf30.csv",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=MODULE_ROOT / "result" / "final_analysis_seeg_20260806_corrected" / ANALYSIS_NAME,
    )
    parser.add_argument(
        "--overwrite-stage09",
        action="store_true",
        help="Allow completion of the explicitly named stage09 directory after an interrupted/failed stage09 run.",
    )
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=list(SUBJECTS))
    return parser.parse_args(argv)


def _run_analysis(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    time_range = _parse_float_pair(list(args.time_range), "--time-range")
    baseline = _parse_float_pair(list(args.baseline), "--baseline")
    if args.signal != "raw200":
        raise ValueError("This stage only accepts --signal raw200")
    if abs(args.bin_ms - 50.0) > 1e-9:
        raise ValueError("The planned Task1 RSA uses --bin-ms 50")
    if time_range != (0.0, 800.0):
        raise ValueError("The planned Task1 RSA uses --time-range 0 800")
    if args.distance != "correlation":
        raise ValueError("The planned Task1 RSA uses correlation distance")
    requested_sets = tuple(args.electrode_set)
    subjects = tuple(args.subjects)
    result_dir = args.result_dir.resolve()
    if result_dir.exists():
        existing = list(result_dir.iterdir())
        # A process-level timeout can leave only a few already-generated
        # figures.  Those are incomplete outputs from this same stage and may
        # be completed safely; any tabular/parameter output means a finished
        # result exists and must remain immutable.
        finished_names = {
            "condition_mean_timecourses.csv",
            "condition_mean_tf_features.csv",
            "condition_rdm_long.csv",
            "condition_trial_counts.csv",
            "electrode_sets_used.csv",
            "rsa_parameters.json",
            "README.md",
        }
        if any(path.name in finished_names for path in existing) and not args.overwrite_stage09:
            raise FileExistsError(f"Refusing to overwrite non-empty result directory: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "electrode_figures").mkdir(exist_ok=True)
    (result_dir / "set_figures").mkdir(exist_ok=True)

    selection_path = args.selection_table.resolve()
    manifest = _input_manifest(selection_path, subjects)
    selection, memberships = _load_electrode_sets(selection_path, requested_sets)
    selection = selection[selection.subject.isin(subjects)].copy()
    actual_counts = {name: int(selection[name].sum()) for name in requested_sets}
    for name, expected in EXPECTED_SET_COUNTS.items():
        if name in requested_sets and actual_counts[name] != expected:
            raise ValueError(f"Expected {expected} {name} electrodes, found {actual_counts[name]}")
    unique_keys = [(str(row.subject), str(row.channel)) for row in selection.itertuples(index=False)]
    metadata = _metadata_by_key(selection)
    channels_by_subject: dict[str, list[str]] = {}
    for subject in subjects:
        channels_by_subject[subject] = sorted(
            [channel for sub, channel in unique_keys if sub == subject], key=natural_key
        )

    audit = [_read_h5_audit(subject) for subject in subjects]
    for record in audit:
        if not record["condition_names_match"] or not record["trigger_mapping_matches"]:
            raise ValueError(f"Task1 condition/trigger audit failed for {record['subject']}: {record}")

    bin_starts = np.arange(time_range[0], time_range[1], args.bin_ms, dtype=float)
    bin_ends = bin_starts + args.bin_ms
    if len(bin_starts) != 16:
        raise ValueError(f"Expected 16 time bins, got {len(bin_starts)}")

    electrode_results: dict[tuple[str, str], dict[str, Any]] = {}
    timecourse_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    rdm_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []

    for subject in subjects:
        subject_channels = channels_by_subject[subject]
        if not subject_channels:
            continue
        epochs_by_condition, time_ms, returned_channels = load_conditions(
            subject, 1, CONDITIONS, subject_channels
        )
        if [channel.upper() for channel in returned_channels] != [channel.upper() for channel in subject_channels]:
            raise ValueError(f"Channel order changed while loading {subject}")
        stacked = np.concatenate([epochs_by_condition[name] for name in CONDITIONS], axis=0)
        trial_logpower, frame_times_ms = _trial_logpower(stacked)
        trial_tf_z = _baseline_zscore(trial_logpower, frame_times_ms, baseline)
        trial_offsets: dict[str, tuple[int, int]] = {}
        offset = 0
        for condition in CONDITIONS:
            n_trials = epochs_by_condition[condition].shape[0]
            trial_offsets[condition] = (offset, offset + n_trials)
            offset += n_trials

        for channel_index, channel in enumerate(subject_channels):
            key = (subject, channel)
            meta = metadata[key]
            timecourses: dict[str, np.ndarray] = {}
            for condition_index, condition in enumerate(CONDITIONS):
                condition_epoch = epochs_by_condition[condition][:, channel_index, :]
                timecourses[condition] = _timecourse_mean(
                    epochs_by_condition[condition][:, channel_index : channel_index + 1, :], time_ms, baseline
                )[0]
                valid_signal = np.isfinite(condition_epoch).all(axis=1)
                start, stop = trial_offsets[condition]
                valid_tf = np.isfinite(trial_tf_z[start:stop, channel_index]).all(axis=(1, 2))
                count_rows.append(
                    {
                        **meta,
                        "condition": condition,
                        "condition_index": condition_index,
                        "trigger": CONDITION_TRIGGERS[condition],
                        "n_trials_raw": int(condition_epoch.shape[0]),
                        "n_valid_signal_trials": int(valid_signal.sum()),
                        "n_valid_tf_trials": int(valid_tf.sum()),
                        "qc_status": "ok" if bool(valid_signal.all() and valid_tf.all()) else "invalid_or_missing_trials",
                    }
                )
                for time_index, time_value in enumerate(time_ms):
                    timecourse_rows.append(
                        {
                            **meta,
                            "condition": condition,
                            "condition_index": condition_index,
                            "time_ms": float(time_value),
                            "mean_signal": float(timecourses[condition][time_index]),
                            "signal": args.signal,
                            "baseline_start_ms": baseline[0],
                            "baseline_end_ms": baseline[1],
                        }
                    )

            rdms: list[np.ndarray] = []
            tf_means: list[np.ndarray] = []
            for bin_index, (bin_start, bin_end) in enumerate(zip(bin_starts, bin_ends)):
                frame_mask = (frame_times_ms >= bin_start) & (frame_times_ms < bin_end)
                if not frame_mask.any():
                    raise ValueError(f"No TF frames in time bin {bin_start}-{bin_end} ms")
                condition_features = []
                for condition_index, condition in enumerate(CONDITIONS):
                    start, stop = trial_offsets[condition]
                    values = trial_tf_z[start:stop, channel_index, :, :][:, :, frame_mask]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        mean_feature = np.nanmean(values, axis=(0, 2))
                    condition_features.append(mean_feature)
                    for feature_index, value in enumerate(mean_feature):
                        low, high = FEATURE_BANDS[feature_index]
                        feature_rows.append(
                            {
                                **meta,
                                "condition": condition,
                                "condition_index": condition_index,
                                "time_bin_index": bin_index,
                                "bin_start_ms": float(bin_start),
                                "bin_end_ms": float(bin_end),
                                "feature_index": feature_index,
                                "band_name": BAND_NAMES[feature_index],
                                "band_low_hz": float(low),
                                "band_high_hz": float(high),
                                "mean_tf_z": float(value),
                                "signal": args.signal,
                            }
                        )
                feature_matrix = np.asarray(condition_features, dtype=float)
                rdm = _correlation_rdm(feature_matrix)
                rdms.append(rdm)
                tf_means.append(feature_matrix)
                for i, condition_i in enumerate(CONDITIONS):
                    for j, condition_j in enumerate(CONDITIONS):
                        rdm_rows.append(
                            {
                                **meta,
                                "time_bin_index": bin_index,
                                "bin_start_ms": float(bin_start),
                                "bin_end_ms": float(bin_end),
                                "condition_i": condition_i,
                                "condition_i_index": i,
                                "condition_j": condition_j,
                                "condition_j_index": j,
                                "distance": float(rdm[i, j]),
                                "distance_method": args.distance,
                                "signal": args.signal,
                            }
                        )
            electrode_results[key] = {
                "metadata": meta,
                "time_ms": np.asarray(time_ms, dtype=float),
                "timecourses": timecourses,
                "tf_means": np.asarray(tf_means, dtype=float),
                "rdms": np.asarray(rdms, dtype=float),
            }
            _plot_electrode(
                electrode_results[key],
                result_dir / "electrode_figures" / f"{_safe_name(subject)}_{_safe_name(channel)}_task1_condition_rsa.png",
                time_range,
            )
        print(f"completed {subject}: {len(subject_channels)} selected electrodes", flush=True)

    if not electrode_results:
        raise ValueError("No electrode results were produced")

    timecourses_df = pd.DataFrame(timecourse_rows)
    features_df = pd.DataFrame(feature_rows)
    rdms_df = pd.DataFrame(rdm_rows)
    counts_df = pd.DataFrame(count_rows)
    timecourses_df.to_csv(result_dir / "condition_mean_timecourses.csv", index=False, encoding="utf-8-sig")
    features_df.to_csv(result_dir / "condition_mean_tf_features.csv", index=False, encoding="utf-8-sig")
    rdms_df.to_csv(result_dir / "condition_rdm_long.csv", index=False, encoding="utf-8-sig")
    counts_df.to_csv(result_dir / "condition_trial_counts.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(result_dir / "electrode_sets_used.csv", index=False, encoding="utf-8-sig")

    set_counts: dict[str, int] = {}
    for set_name in requested_sets:
        set_results = [result["rdms"] for result in electrode_results.values() if bool(result["metadata"][set_name])]
        set_counts[set_name] = len(set_results)
        if not set_results:
            continue
        mean_rdms = np.nanmean(np.stack(set_results, axis=0), axis=0)
        for mean_rdm in mean_rdms:
            np.fill_diagonal(mean_rdm, 0.0)
        _plot_rdm_grid(
            mean_rdms,
            result_dir / "set_figures" / f"{set_name}_mean_rdm_task1_condition_rsa.png",
            f"{set_name} average Task1 condition RDM",
            f"Direct arithmetic average of {len(set_results)} electrode distance matrices; descriptive only",
        )

    summary = {
        "analysis": ANALYSIS_NAME,
        "unique_electrodes": int(len(electrode_results)),
        "rdm_time_bins": int(len(bin_starts)),
        "set_counts": set_counts,
        "condition_count": len(CONDITIONS),
        "condition_order": list(CONDITIONS),
        "time_range_ms": list(time_range),
        "bin_ms": float(args.bin_ms),
        "signal": args.signal,
        "distance": args.distance,
        "output_dir": str(result_dir),
        "h5_audit": audit,
        "input_manifest": manifest,
        "selection_table": str(selection_path),
        "data_audit_root": str(MODULE_ROOT / "Data"),
    }
    parameters = {
        "analysis": ANALYSIS_NAME,
        "description": "Task1 condition-mean time-frequency RSA; independent stage09; no inference",
        "signal": args.signal,
        "time_range_ms": list(time_range),
        "bin_ms": float(args.bin_ms),
        "n_time_bins": int(len(bin_starts)),
        "baseline_ms": list(baseline),
        "electrode_sets": list(requested_sets),
        "electrode_set_selection_table": str(selection_path),
        "electrode_set_selection_signal": "lf30 historical current functional membership (held fixed for raw200 input)",
        "expected_set_counts": {name: EXPECTED_SET_COUNTS[name] for name in requested_sets},
        "actual_set_counts": set_counts,
        "unique_electrode_count": int(len(electrode_results)),
        "condition_order": list(CONDITIONS),
        "condition_triggers": CONDITION_TRIGGERS,
        "tf_method": "scipy.signal.stft on every trial over full -500 to 1000 ms epoch",
        "tf_nperseg_samples": TF_NPERSEG,
        "tf_hop_ms": TF_HOP_MS,
        "tf_frame_times_ms": "STFT frame center; epoch origin is -500 ms",
        "feature_bands_hz": [list(band) for band in FEATURE_BANDS],
        "feature_band_names": list(BAND_NAMES),
        "feature_transform": "trial-level log(power), then baseline z-score across all conditions/trials per channel/band, then condition trial mean",
        "distance_method": "1 - Pearson correlation across 16 condition-mean band features",
        "set_average_method": "direct arithmetic average of electrode distance matrices; no Fisher z",
        "statistical_inference": "none; descriptive condition-mean geometry only",
        "data_role": "Data/ is used for trial identity/trigger audit only; neural features come directly from HDF5",
        "subjects": list(subjects),
        "h5_files": [record["path"] for record in audit],
        "input_manifest": manifest,
        "git_commit": _git_commit(),
        "python": sys.version,
        "python_executable": sys.executable,
        "package_versions": _package_versions(),
        "run_dir": str(run_dir),
        "result_dir": str(result_dir),
        "output_paths": [str(path) for path in sorted(result_dir.rglob("*")) if path.is_file()],
    }
    (result_dir / "rsa_parameters.json").write_text(json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_readme(result_dir, parameters, summary)
    # README and rsa_parameters.json themselves are also part of the final
    # output manifest; rewrite the parameter file after all outputs exist.
    parameters["output_paths"] = [str(path) for path in sorted(result_dir.rglob("*")) if path.is_file()]
    (result_dir / "rsa_parameters.json").write_text(json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["output_paths"] = [str(path) for path in sorted(result_dir.rglob("*")) if path.is_file()]
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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
            summary = _run_analysis(args, run_dir)
            caught.extend(caught_warnings)
        end = datetime.now(timezone.utc)
        summary["started_at_utc"] = start.isoformat()
        summary["finished_at_utc"] = end.isoformat()
        summary["status"] = "completed"
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "output_paths.txt").write_text("\n".join(summary["output_paths"]) + "\n", encoding="utf-8")
        print(json.dumps({"status": "completed", "result_dir": summary["output_dir"], "run_dir": str(run_dir)}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        end = datetime.now(timezone.utc)
        failure = {
            "status": "failed",
            "started_at_utc": start.isoformat(),
            "finished_at_utc": end.isoformat(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "result_dir": str(args.result_dir.resolve()),
        }
        (run_dir / "run_summary.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    finally:
        warning_lines = []
        for message in caught:
            warning_lines.append(f"{message.category.__name__}: {message.message} ({message.filename}:{message.lineno})")
        (run_dir / "warnings.log").write_text("\n".join(warning_lines) + ("\n" if warning_lines else ""), encoding="utf-8")
        (run_dir / "finish_time.txt").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
