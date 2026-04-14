import math
import os
import re
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


BASE_PATH = project_root()

# =========================
# User Config
# =========================
SUBJECT = "test001"
TASK = "task1"  # "task1" | "task2" | "task3"

GROUP_A = [1, 3, 7]  # 1-based condition indices
GROUP_B = [2, 4, 8]  # 1-based condition indices
GROUP_A_LABEL = "Color"
GROUP_B_LABEL = "Gray"

ALPHA = 0.05
MIN_CONSECUTIVE_SIG_POINTS = 3
FS = 500.0
TMIN_MS = -100.0
BASELINE_START_MS = -100.0
BASELINE_END_MS = 0.0

ROI = None              # e.g. "Color_with_sti"; None means all ROI
ROI_PATTERN = "Color_with*.mat"
DPI = 260
OUTPUT_DIR = None


FEATURE_SUBDIR = "decoding_gamma_multiband_features"
FIELD_PREFIX = "gmb"


def validate_config():
    if TASK not in {"task1", "task2", "task3"}:
        raise ValueError(f"Unsupported TASK: {TASK}")
    if not GROUP_A or not GROUP_B:
        raise ValueError("GROUP_A and GROUP_B cannot be empty.")
    if set(GROUP_A) & set(GROUP_B):
        raise ValueError("GROUP_A and GROUP_B overlap. Please use non-overlapping conditions.")
    if min(GROUP_A + GROUP_B) < 1:
        raise ValueError("Condition indices must start from 1.")


def get_data_field():
    return f"{FIELD_PREFIX}_{TASK}"


def to_zero_based_indices(indices_1based):
    return [v - 1 for v in indices_1based]


def sanitize_name(name):
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name))
    name = re.sub(r"^_+|_+$", "", name)
    return name or "Unknown"


def build_subject_paths(subject):
    feature_dir = get_feature_dir(BASE_PATH, FEATURE_SUBDIR, subject)
    return feature_dir


def matlab_cellstr_to_list(value):
    array = np.asarray(value)
    if array.size == 0:
        return []
    result = []
    for item in np.ravel(array, order="F"):
        if isinstance(item, np.ndarray):
            flattened = np.ravel(item, order="F")
            if flattened.size == 1:
                result.append(str(flattened[0]).strip())
            else:
                result.append("".join(str(v) for v in flattened.tolist()).strip())
        else:
            result.append(str(item).strip())
    return result


def get_time_vector(mat, n_time):
    if "gmb_time_ms" in mat:
        times = np.asarray(mat["gmb_time_ms"], dtype=float).reshape(-1)
        if times.size == n_time:
            return times
    return TMIN_MS + np.arange(n_time, dtype=float) * (1000.0 / FS)


def baseline_correct(data, times_ms, baseline_start_ms, baseline_end_ms):
    baseline_mask = (times_ms >= baseline_start_ms) & (times_ms <= baseline_end_ms)
    if not np.any(baseline_mask):
        raise ValueError(
            f"No baseline points found in {baseline_start_ms} to {baseline_end_ms} ms. "
            f"Available time range is {times_ms[0]} to {times_ms[-1]} ms."
        )
    baseline_mean = data[..., baseline_mask].mean(axis=-1, keepdims=True)
    return data - baseline_mean


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


def load_multiband_metadata(mat, n_features):
    channel_names = matlab_cellstr_to_list(mat.get("gmb_roi_channels", np.array([])))
    band_names = matlab_cellstr_to_list(mat.get("gmb_band_names", np.array([])))
    band_ranges = np.asarray(mat.get("gmb_band_ranges", np.empty((0, 2))), dtype=float)

    band_index = np.asarray(mat.get("gmb_feature_band_index", np.array([]))).reshape(-1)
    channel_index = np.asarray(mat.get("gmb_feature_channel_index", np.array([]))).reshape(-1)

    if band_index.size == 0 or channel_index.size == 0:
        if not band_names or not channel_names:
            raise ValueError("Missing multiband metadata: cannot reconstruct band/channel mapping.")
        n_bands = len(band_names)
        n_channels = len(channel_names)
        if n_bands * n_channels != n_features:
            raise ValueError(
                f"Metadata mismatch: {n_bands} bands x {n_channels} channels != {n_features} features."
            )
        band_index = np.repeat(np.arange(1, n_bands + 1), n_channels)
        channel_index = np.tile(np.arange(1, n_channels + 1), n_bands)

    if len(channel_names) == 0:
        max_channel = int(np.max(channel_index))
        channel_names = [f"Ch{idx + 1}" for idx in range(max_channel)]
    if len(band_names) == 0:
        max_band = int(np.max(band_index))
        band_names = [f"Band{idx + 1:02d}" for idx in range(max_band)]
    if band_ranges.size == 0:
        band_ranges = np.full((len(band_names), 2), np.nan)

    return channel_names, band_names, band_ranges, band_index.astype(int), channel_index.astype(int)


def reshape_to_band_channel(data, band_index, channel_index, n_bands, n_channels):
    n_cond, n_rep, n_features, n_time = data.shape
    reshaped = np.zeros((n_cond, n_rep, n_bands, n_channels, n_time), dtype=float)
    for feat_idx in range(n_features):
        band_pos = band_index[feat_idx] - 1
        channel_pos = channel_index[feat_idx] - 1
        reshaped[:, :, band_pos, channel_pos, :] = data[:, :, feat_idx, :]
    return reshaped


def prepare_group_trials(data, condition_indices, channel_idx):
    selected = data[condition_indices, :, :, channel_idx, :]
    n_cond, n_rep, n_bands, n_time = selected.shape
    return selected.reshape(n_cond * n_rep, n_bands, n_time)


def compute_channel_band_stats(data, group_a_indices, group_b_indices, alpha):
    n_bands = data.shape[2]
    n_channels = data.shape[3]
    results = []

    for ch_idx in range(n_channels):
        group_a = prepare_group_trials(data, group_a_indices, ch_idx)
        group_b = prepare_group_trials(data, group_b_indices, ch_idx)

        mean_a = group_a.mean(axis=0)
        mean_b = group_b.mean(axis=0)
        diff = mean_a - mean_b

        _, p_values = ttest_ind(group_a, group_b, axis=0, equal_var=False, nan_policy="omit")
        p_values = np.nan_to_num(p_values, nan=1.0)

        sig_mask = p_values < alpha
        for band_idx in range(n_bands):
            sig_mask[band_idx] = enforce_min_consecutive(sig_mask[band_idx], MIN_CONSECUTIVE_SIG_POINTS)

        results.append(
            {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "diff": diff,
                "p_values": p_values,
                "sig_mask": sig_mask,
                "n_trials_a": group_a.shape[0],
                "n_trials_b": group_b.shape[0],
            }
        )
    return results


def compute_global_limits(stats_per_channel):
    all_values = []
    all_abs_diff = []
    for stats in stats_per_channel:
        all_values.append(stats["mean_a"].ravel())
        all_values.append(stats["mean_b"].ravel())
        all_abs_diff.append(np.abs(stats["diff"]).ravel())
    combined = np.concatenate(all_values) if all_values else np.array([0.0, 1.0])
    low = float(np.percentile(combined, 5))
    high = float(np.percentile(combined, 95))
    if math.isclose(low, high):
        low -= 1.0
        high += 1.0
    diff_lim = float(np.percentile(np.concatenate(all_abs_diff), 95)) if all_abs_diff else 1.0
    diff_lim = max(diff_lim, 1e-6)
    return (low, high), (-diff_lim, diff_lim)


def plot_roi(roi_name, channel_names, band_names, band_ranges, times_ms, stats_per_channel, save_path):
    n_channels = len(channel_names)
    n_cols = 3
    fig, axes = plt.subplots(
        n_channels,
        n_cols,
        figsize=(5.9 * n_cols, max(3.0, 2.8 * n_channels)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    power_limits, diff_limits = compute_global_limits(stats_per_channel)
    y_positions = np.arange(len(band_names), dtype=float)
    if band_ranges.shape[0] == len(band_names):
        yticklabels = [f"{name}\n{low:.1f}-{high:.1f}Hz" for name, (low, high) in zip(band_names, band_ranges)]
    else:
        yticklabels = list(band_names)

    group_a_str = ",".join(str(i) for i in GROUP_A)
    group_b_str = ",".join(str(i) for i in GROUP_B)
    fig.suptitle(
        (
            f"{SUBJECT} | {roi_name} | gamma_multiband | {TASK}\n"
            f"{GROUP_A_LABEL} (cond {group_a_str}) vs {GROUP_B_LABEL} (cond {group_b_str}) | "
            f"baseline {BASELINE_START_MS:g} to {BASELINE_END_MS:g} ms"
        ),
        fontsize=13,
        y=0.995,
    )

    meshes = []
    for ch_idx, ch_name in enumerate(channel_names):
        stats = stats_per_channel[ch_idx]
        plot_items = [
            (stats["mean_a"], GROUP_A_LABEL, power_limits),
            (stats["mean_b"], GROUP_B_LABEL, power_limits),
            (stats["diff"], f"{GROUP_A_LABEL}-{GROUP_B_LABEL}", diff_limits),
        ]
        for col_idx, (matrix, title, limits) in enumerate(plot_items):
            ax = axes[ch_idx, col_idx]
            cmap = "RdBu_r" if col_idx == 2 else "viridis"
            mesh = ax.pcolormesh(times_ms, y_positions, matrix, shading="auto", cmap=cmap, vmin=limits[0], vmax=limits[1])
            meshes.append((mesh, col_idx))
            ax.axvline(0, color="white" if col_idx == 2 else "0.85", linestyle="--", linewidth=0.9)
            if col_idx == 0:
                ax.set_ylabel(ch_name)
                ax.set_yticks(y_positions)
                ax.set_yticklabels(yticklabels, fontsize=8)
            if ch_idx == 0:
                ax.set_title(title, fontsize=11)
            if ch_idx == n_channels - 1:
                ax.set_xlabel("Time (ms)")

            if col_idx == 2 and np.any(stats["sig_mask"]):
                sig_mask = stats["sig_mask"]
                for band_idx in range(sig_mask.shape[0]):
                    sig_times = times_ms[sig_mask[band_idx]]
                    if sig_times.size == 0:
                        continue
                    ax.scatter(sig_times, np.full(sig_times.shape, y_positions[band_idx]), s=5, c="black", marker="s", linewidths=0)

    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.05, top=0.95, hspace=0.35, wspace=0.08)
    cbar_mean_ax = fig.add_axes([0.88, 0.54, 0.016, 0.34])
    cbar_diff_ax = fig.add_axes([0.88, 0.12, 0.016, 0.34])
    cbar_mean = fig.colorbar(meshes[0][0], cax=cbar_mean_ax)
    cbar_mean.set_label("Baseline-corrected power")
    cbar_diff = fig.colorbar(next(mesh for mesh, col in meshes if col == 2), cax=cbar_diff_ax)
    cbar_diff.set_label(f"{GROUP_A_LABEL}-{GROUP_B_LABEL}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_channel_stats(roi_name, channel_names, band_names, band_ranges, times_ms, stats_per_channel, save_path):
    rows = []
    for ch_name, stats in zip(channel_names, stats_per_channel):
        for band_idx, band_name in enumerate(band_names):
            sig_times = times_ms[stats["sig_mask"][band_idx]]
            band_range = band_ranges[band_idx] if band_idx < band_ranges.shape[0] else (np.nan, np.nan)
            rows.append(
                {
                    "roi": roi_name,
                    "channel": ch_name,
                    "band": band_name,
                    "band_low_hz": float(band_range[0]),
                    "band_high_hz": float(band_range[1]),
                    "n_trials_a": stats["n_trials_a"],
                    "n_trials_b": stats["n_trials_b"],
                    "n_sig_points": int(np.sum(stats["sig_mask"][band_idx])),
                    "first_sig_ms": float(sig_times[0]) if sig_times.size else np.nan,
                    "last_sig_ms": float(sig_times[-1]) if sig_times.size else np.nan,
                    "min_p": float(np.min(stats["p_values"][band_idx])),
                    "mean_diff_abs": float(np.mean(np.abs(stats["diff"][band_idx]))),
                    "peak_diff_abs": float(np.max(np.abs(stats["diff"][band_idx]))),
                }
            )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(save_path, index=False)


def build_output_dir():
    if OUTPUT_DIR is not None:
        return Path(OUTPUT_DIR)
    cond_a = "-".join(str(v) for v in GROUP_A)
    cond_b = "-".join(str(v) for v in GROUP_B)
    return get_roi_electrode_condition_dir(BASE_PATH, TASK, 'gamma_multiband', SUBJECT, f"A_{cond_a}_vs_B_{cond_b}")


def main():
    validate_config()
    feature_dir = build_subject_paths(SUBJECT)
    output_dir = build_output_dir()
    field = get_data_field()
    group_a_indices = to_zero_based_indices(GROUP_A)
    group_b_indices = to_zero_based_indices(GROUP_B)

    mat_files = sorted(feature_dir.glob(ROI_PATTERN))
    mat_files = [path for path in mat_files if path.is_file() and path.suffix == ".mat"]
    if ROI is not None:
        mat_files = [path for path in mat_files if path.stem == ROI]
    if not mat_files:
        raise FileNotFoundError(f"No ROI .mat files found in {feature_dir}")

    summary_rows = []
    for mat_file in mat_files:
        roi_name = mat_file.stem
        try:
            mat = sio.loadmat(mat_file)
        except Exception as exc:
            print(f"Skip {roi_name}: failed to read {mat_file.name} ({type(exc).__name__}: {exc})")
            continue

        if field not in mat:
            print(f"Skip {roi_name}: missing {field}.")
            continue

        data = np.asarray(mat[field], dtype=float)
        if data.ndim != 4:
            print(f"Skip {roi_name}: {field} shape is {data.shape}, expected 4D [Cond, Rep, Feature, Time].")
            continue

        max_cond = data.shape[0] - 1
        if max(group_a_indices + group_b_indices) > max_cond:
            requested_max = max(group_a_indices + group_b_indices) + 1
            raise IndexError(
                f"{roi_name} only has {data.shape[0]} conditions, but requested condition index up to {requested_max}."
            )

        times_ms = get_time_vector(mat, n_time=data.shape[-1])
        channel_names, band_names, band_ranges, band_index, channel_index = load_multiband_metadata(mat, n_features=data.shape[2])
        data = reshape_to_band_channel(data, band_index, channel_index, len(band_names), len(channel_names))
        data = baseline_correct(data, times_ms, BASELINE_START_MS, BASELINE_END_MS)

        stats_per_channel = compute_channel_band_stats(
            data=data,
            group_a_indices=group_a_indices,
            group_b_indices=group_b_indices,
            alpha=ALPHA,
        )

        fig_path = output_dir / "figures" / f"{sanitize_name(roi_name)}.png"
        csv_path = output_dir / "stats" / f"{sanitize_name(roi_name)}.csv"

        plot_roi(roi_name, channel_names, band_names, band_ranges, times_ms, stats_per_channel, fig_path)
        save_channel_stats(roi_name, channel_names, band_names, band_ranges, times_ms, stats_per_channel, csv_path)

        summary_rows.append(
            {
                "roi": roi_name,
                "n_channels": len(channel_names),
                "n_bands": len(band_names),
                "n_channel_band_with_sig": int(sum(np.any(s["sig_mask"]) for s in stats_per_channel)),
                "max_sig_points_one_channel_band": int(max(np.max(np.sum(s["sig_mask"], axis=1)) for s in stats_per_channel)),
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
    main()