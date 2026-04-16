import math
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import ttest_ind

from newanalyse_paths import get_feature_dir, get_roi_electrode_condition_dir, project_root

# ------------------------
#   绘制ROI下不同电极的时间信号差异
# ------------------------
BASE_PATH = project_root()

# =========================
# User Config
# =========================
SUBJECT = "test001"
FEATURE_KIND = "erp"  # "erp" | "lowgamma" | "highgamma"
TASK = "task1"        # "task1" | "task2" | "task3"

GROUP_A = [1, 3, 5, 7]   # 1-based condition i0dices
GROUP_B = [2, 4, 6, 8]   # 1-based condition indices
GROUP_A_LABEL = "Color"
GROUP_B_LABEL = "Gray"

ALPHA = 0.05
MIN_CONSECUTIVE_SIG_POINTS = 3
FS = 500.0
TMIN_MS = -100.0
BASELINE_START_MS = -100.0
BASELINE_END_MS = 0.0

ROI = None              # e.g. "Color_patch"; None means all ROI
ROI_PATTERN = "*.mat"
DPI = 300
OUTPUT_DIR = None       # None means auto-generate


FEATURE_CONFIG = {
    "erp": {
        "feature_subdir": "decoding_erp_features",
        "field_prefix": "erp",
    },
    "lowgamma": {
        "feature_subdir": "decoding_lowgamma_features",
        "field_prefix": "lg",
    },
    "highgamma": {
        "feature_subdir": "decoding_highgamma_features",
        "field_prefix": "hg",
    },
}


def validate_config():
    if FEATURE_KIND not in FEATURE_CONFIG:
        raise ValueError(f"Unsupported FEATURE_KIND: {FEATURE_KIND}")
    if TASK not in {"task1", "task2", "task3"}:
        raise ValueError(f"Unsupported TASK: {TASK}")
    if not GROUP_A or not GROUP_B:
        raise ValueError("GROUP_A and GROUP_B cannot be empty.")
    if set(GROUP_A) & set(GROUP_B):
        raise ValueError("GROUP_A and GROUP_B overlap. Please use non-overlapping conditions.")
    if min(GROUP_A + GROUP_B) < 1:
        raise ValueError("Condition indices must start from 1.")


def get_data_field():
    field_prefix = FEATURE_CONFIG[FEATURE_KIND]["field_prefix"]
    return f"{field_prefix}_{TASK}"


def to_zero_based_indices(indices_1based):
    return [v - 1 for v in indices_1based]


def sanitize_roi_name(name):
    name = re.sub(r"[^a-zA-Z0-9_]", "_", str(name))
    name = re.sub(r"^_+|_+$", "", name)
    return name or "Unknown"


def extract_channel_labels(epoch_struct):
    ch = epoch_struct.ch
    if isinstance(ch, np.ndarray):
        return [str(item.labels) for item in ch.flat]
    return [str(ch.labels)]


def build_subject_paths(subject):
    data_dir = BASE_PATH / "processed_data" / subject
    feature_dir = get_feature_dir(BASE_PATH, FEATURE_CONFIG[FEATURE_KIND]["feature_subdir"], subject)
    loc_file = data_dir / f"{subject}_ieegloc.xlsx"
    task_to_file = {
        "task1": data_dir / "task1_ERP_epoched.mat",
        "task2": data_dir / "task2_ERP_epoched.mat",
        "task3": data_dir / "task3_ERP_epoched.mat",
    }
    return data_dir, feature_dir, loc_file, task_to_file


def load_common_channels(task_to_file):
    channel_sets = []
    for task_name in ("task1", "task2", "task3"):
        mat = sio.loadmat(
            task_to_file[task_name],
            variable_names=["epoch"],
            squeeze_me=True,
            struct_as_record=False,
        )
        labels = extract_channel_labels(mat["epoch"])
        channel_sets.append(set(labels))
    common = sorted(channel_sets[0] & channel_sets[1] & channel_sets[2])
    if not common:
        raise RuntimeError("No common channels found across task1/task2/task3.")
    return common


def get_roi_map(loc_file, channel_labels):
    loc_table = pd.read_excel(loc_file)
    cols = list(loc_table.columns)
    lower_cols = [str(c).strip().lower() for c in cols]

    name_idx = None
    for candidate in ("name", "channel", "electrode", "label"):
        if candidate in lower_cols:
            name_idx = lower_cols.index(candidate)
            break
    if name_idx is None:
        name_idx = 0
    name_col = cols[name_idx]

    roi_idx = None
    for candidate in ("aal3", "aal3_mni_linear_", "aal3_label", "aal3 (mni-linear)"):
        if candidate in lower_cols:
            roi_idx = lower_cols.index(candidate)
            break
    if roi_idx is None:
        for candidate in ("roi", "region", "anatomy", "dk_lobe", "lobe"):
            if candidate in lower_cols:
                roi_idx = lower_cols.index(candidate)
                break
    if roi_idx is None:
        raise RuntimeError("Could not identify ROI column in location file.")
    roi_col = cols[roi_idx]

    roi_map = {}
    table_names = loc_table[name_col].astype(str).str.strip().str.lower()
    for ch_name in channel_labels:
        row_idxs = np.flatnonzero(table_names == ch_name.strip().lower())
        if len(row_idxs) == 0:
            target_rois = ["Unknown"]
        else:
            target_rois = []
            for idx in row_idxs:
                value = loc_table.iloc[idx][roi_col]
                if pd.isna(value):
                    roi_name = "Unknown"
                else:
                    roi_name = sanitize_roi_name(value)
                target_rois.append(roi_name)
            target_rois = sorted(set(target_rois))

        for roi_name in target_rois:
            roi_map.setdefault(roi_name, []).append(ch_name)
    return roi_map


def baseline_correct_erp(erp, times_ms, baseline_start_ms, baseline_end_ms):
    baseline_mask = (times_ms >= baseline_start_ms) & (times_ms <= baseline_end_ms)
    if not np.any(baseline_mask):
        raise ValueError(
            f"No baseline points found in {baseline_start_ms} to {baseline_end_ms} ms. "
            f"Available time range is {times_ms[0]} to {times_ms[-1]} ms."
        )
    baseline_mean = erp[..., baseline_mask].mean(axis=-1, keepdims=True)
    return erp - baseline_mean


def enforce_min_consecutive(mask, min_consecutive):
    if min_consecutive <= 1 or not np.any(mask):
        return mask

    filtered = np.zeros_like(mask, dtype=bool)
    start = None
    for idx, is_sig in enumerate(mask):
        if is_sig and start is None:
            start = idx
        elif not is_sig and start is not None:
            if idx - start >= min_consecutive:
                filtered[start:idx] = True
            start = None

    if start is not None and len(mask) - start >= min_consecutive:
        filtered[start:] = True

    return filtered


def prepare_group_trials(erp, condition_indices, ch_idx):
    selected = erp[condition_indices, :, ch_idx, :]
    return selected.reshape(-1, selected.shape[-1])


def compute_channel_stats(erp, group_a_indices, group_b_indices, alpha):
    n_channels = erp.shape[2]
    n_time = erp.shape[3]
    results = []

    for ch_idx in range(n_channels):
        group_a = prepare_group_trials(erp, group_a_indices, ch_idx)
        group_b = prepare_group_trials(erp, group_b_indices, ch_idx)

        mean_a = group_a.mean(axis=0)
        mean_b = group_b.mean(axis=0)
        sem_a = group_a.std(axis=0, ddof=1) / math.sqrt(group_a.shape[0])
        sem_b = group_b.std(axis=0, ddof=1) / math.sqrt(group_b.shape[0])
        sem_a = np.nan_to_num(sem_a, nan=0.0)
        sem_b = np.nan_to_num(sem_b, nan=0.0)

        _, p_values = ttest_ind(group_a, group_b, axis=0, equal_var=False, nan_policy="omit")
        p_values = np.nan_to_num(p_values, nan=1.0)

        sig_mask = p_values < alpha
        sig_mask = enforce_min_consecutive(sig_mask, MIN_CONSECUTIVE_SIG_POINTS)

        results.append(
            {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "sem_a": sem_a,
                "sem_b": sem_b,
                "p_values": p_values,
                "sig_mask": sig_mask,
                "n_trials_a": group_a.shape[0],
                "n_trials_b": group_b.shape[0],
            }
        )

    assert len(results) == n_channels
    assert results[0]["mean_a"].shape[0] == n_time
    return results


def plot_roi(roi_name, channel_names, times_ms, stats_per_channel, save_path):
    n_channels = len(channel_names)
    n_cols = int(math.ceil(math.sqrt(n_channels)))
    n_rows = int(math.ceil(n_channels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 3.9 * n_rows),
        squeeze=False,
        sharex=True,
    )

    color_a = "#1f77b4"
    color_b = "#d62728"

    group_a_str = ",".join(str(i) for i in GROUP_A)
    group_b_str = ",".join(str(i) for i in GROUP_B)
    fig.suptitle(
        (
            f"{SUBJECT} | {roi_name} | {FEATURE_KIND} | {TASK}\n"
            f"{GROUP_A_LABEL} (cond {group_a_str}) vs "
            f"{GROUP_B_LABEL} (cond {group_b_str}) | baseline {BASELINE_START_MS:g} to {BASELINE_END_MS:g} ms"
        ),
        fontsize=13,
        y=1.01,
    )

    for ch_idx, ch_name in enumerate(channel_names):
        ax = axes[ch_idx // n_cols, ch_idx % n_cols]
        stats = stats_per_channel[ch_idx]

        mean_a = stats["mean_a"]
        mean_b = stats["mean_b"]
        sem_a = stats["sem_a"]
        sem_b = stats["sem_b"]
        sig_mask = stats["sig_mask"]

        y_min = min(np.min(mean_a - sem_a), np.min(mean_b - sem_b))
        y_max = max(np.max(mean_a + sem_a), np.max(mean_b + sem_b))
        y_pad = max((y_max - y_min) * 0.12, 1e-6)
        sig_y = y_min - y_pad * 0.55

        ax.plot(times_ms, mean_a, color=color_a, linewidth=1.8, label=GROUP_A_LABEL)
        ax.fill_between(times_ms, mean_a - sem_a, mean_a + sem_a, color=color_a, alpha=0.2)
        ax.plot(times_ms, mean_b, color=color_b, linewidth=1.8, label=GROUP_B_LABEL)
        ax.fill_between(times_ms, mean_b - sem_b, mean_b + sem_b, color=color_b, alpha=0.2)

        if np.any(sig_mask):
            ax.scatter(
                times_ms[sig_mask],
                np.full(np.sum(sig_mask), sig_y),
                marker="s",
                s=12,
                color="black",
                linewidths=0,
                label="p < alpha",
            )

        ax.axvline(0, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_title(f"{ch_name} | sig={int(np.sum(sig_mask))} pts", fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(times_ms[0], times_ms[-1])
        ax.set_ylim(sig_y - y_pad * 0.35, y_max + y_pad * 0.25)
        ax.grid(True, alpha=0.25)

    for empty_idx in range(n_channels, n_rows * n_cols):
        fig.delaxes(axes[empty_idx // n_cols, empty_idx % n_cols])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_channel_stats(roi_name, channel_names, times_ms, stats_per_channel, save_path):
    rows = []
    for ch_name, stats in zip(channel_names, stats_per_channel):
        sig_times = times_ms[stats["sig_mask"]]
        rows.append(
            {
                "roi": roi_name,
                "channel": ch_name,
                "n_trials_a": stats["n_trials_a"],
                "n_trials_b": stats["n_trials_b"],
                "n_sig_points": int(np.sum(stats["sig_mask"])),
                "first_sig_ms": float(sig_times[0]) if sig_times.size else np.nan,
                "last_sig_ms": float(sig_times[-1]) if sig_times.size else np.nan,
                "min_p": float(np.min(stats["p_values"])),
            }
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(save_path, index=False)


def build_output_dir():
    if OUTPUT_DIR is not None:
        return Path(OUTPUT_DIR)
    cond_a = "-".join(str(v) for v in GROUP_A)
    cond_b = "-".join(str(v) for v in GROUP_B)
    return get_roi_electrode_condition_dir(BASE_PATH, TASK, FEATURE_KIND, SUBJECT, f"A_{cond_a}_vs_B_{cond_b}")


def main():
    validate_config()
    group_a_indices = to_zero_based_indices(GROUP_A)
    group_b_indices = to_zero_based_indices(GROUP_B)
    _, feature_dir, loc_file, task_to_file = build_subject_paths(SUBJECT)

    field = get_data_field()
    common_channels = load_common_channels(task_to_file)
    roi_map = get_roi_map(loc_file, common_channels)
    output_dir = build_output_dir()

    mat_files = sorted(feature_dir.glob(ROI_PATTERN))
    mat_files = [p for p in mat_files if p.is_file() and p.suffix == ".mat"]
    if ROI is not None:
        mat_files = [p for p in mat_files if p.stem == ROI]
    if not mat_files:
        raise FileNotFoundError(f"No ROI .mat files found in {feature_dir}")

    summary_rows = []
    for mat_file in mat_files:
        roi_name = mat_file.stem
        if roi_name not in roi_map:
            print(f"Skip {roi_name}: electrode labels not found in ROI map.")
            continue

        try:
            mat = sio.loadmat(mat_file)
        except Exception as exc:
            print(f"Skip {roi_name}: failed to read {mat_file.name} ({type(exc).__name__}: {exc})")
            continue
        if field not in mat:
            print(f"Skip {roi_name}: missing {field}.")
            continue

        erp = np.asarray(mat[field], dtype=float)
        if erp.ndim != 4:
            print(f"Skip {roi_name}: {field} shape is {erp.shape}, expected 4D.")
            continue

        max_cond = erp.shape[0] - 1
        if max(group_a_indices + group_b_indices) > max_cond:
            requested_max = max(group_a_indices + group_b_indices) + 1
            raise IndexError(
                f"{roi_name} only has {erp.shape[0]} conditions, but requested condition index up to {requested_max}."
            )

        channel_names = roi_map[roi_name]
        if len(channel_names) != erp.shape[2]:
            print(
                f"Warning: {roi_name} channel-name count ({len(channel_names)}) "
                f"does not match ERP channels ({erp.shape[2]}). Using fallback labels."
            )
            channel_names = [f"Ch{idx + 1}" for idx in range(erp.shape[2])]

        times_ms = TMIN_MS + np.arange(erp.shape[3]) * (1000.0 / FS)
        erp = baseline_correct_erp(
            erp,
            times_ms,
            baseline_start_ms=BASELINE_START_MS,
            baseline_end_ms=BASELINE_END_MS,
        )
        stats_per_channel = compute_channel_stats(
            erp=erp,
            group_a_indices=group_a_indices,
            group_b_indices=group_b_indices,
            alpha=ALPHA,
        )

        fig_path = output_dir / "figures" / f"{roi_name}.png"
        csv_path = output_dir / "stats" / f"{roi_name}.csv"

        plot_roi(roi_name, channel_names, times_ms, stats_per_channel, fig_path)
        save_channel_stats(roi_name, channel_names, times_ms, stats_per_channel, csv_path)

        summary_rows.append(
            {
                "roi": roi_name,
                "n_channels": len(channel_names),
                "n_channels_with_sig": int(sum(np.any(s["sig_mask"]) for s in stats_per_channel)),
                "max_sig_points_one_channel": int(max(np.sum(s["sig_mask"]) for s in stats_per_channel)),
                "figure": str(fig_path),
                "stats_csv": str(csv_path),
            }
        )
        print(f"Saved ROI figure: {fig_path}")

    if summary_rows:
        summary_path = output_dir / "roi_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    _script_start_time = time.time()
    try:
        main()
    finally:
        print(f"Total runtime: {time.time() - _script_start_time:.2f} s")
