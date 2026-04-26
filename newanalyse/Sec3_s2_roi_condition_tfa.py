import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from newanalyse_paths import get_feature_dir, get_roi_condition_tfa_dir, project_root
# =========================
# 时频信号差异
# =========================

BASE_PATH = project_root()

# =========================
# User Config
# =========================
SUBJECT = "test001"
TASK = "task1"  # "task1" | "task2" | "task3"

GROUP_A = [1,3,7]  # 1-based condition indices
GROUP_B = [2,4,8]  # 1-based condition indices
GROUP_A_LABEL = "Condition 1"
GROUP_B_LABEL = "Condition 2"

ROI = None              # e.g. "Color_patch"; None means all ROI
ROI_PATTERN = "Color_with*.mat"

FS = 500.0
DEFAULT_TMIN_MS = -100.0
BASELINE_MS = (-100.0, -50.0)
BASELINE_MODE = "db"   # "db" | "ratio" | "percent" | "zscore" | "none"

FREQS = np.arange(4.0, 152.0, 4.0)
N_CYCLES = np.linspace(3.0, 10.0, FREQS.size)

DPI = 220
DIFF_CMAP = "RdBu_r"
POWER_CMAP = "turbo"
SAVE_TRIAL_LEVEL_POWER = True
OUTPUT_ROOT = BASE_PATH / "result" / "roi_condition_tfa" / SUBJECT


def validate_config():
    if TASK not in {"task1", "task2", "task3"}:
        raise ValueError(f"Unsupported TASK: {TASK}")
    if not GROUP_A or not GROUP_B:
        raise ValueError("GROUP_A and GROUP_B cannot be empty.")
    if set(GROUP_A) & set(GROUP_B):
        raise ValueError("GROUP_A and GROUP_B overlap. Please use non-overlapping conditions.")
    if min(GROUP_A + GROUP_B) < 1:
        raise ValueError("Condition indices must start from 1.")
    if BASELINE_MODE not in {"db", "ratio", "percent", "zscore", "none"}:
        raise ValueError(f"Unsupported BASELINE_MODE: {BASELINE_MODE}")
    freqs = np.asarray(FREQS, dtype=float)
    cycles = np.asarray(N_CYCLES, dtype=float)
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("FREQS must be a non-empty 1D array.")
    if cycles.ndim == 0:
        cycles = np.full(freqs.shape, float(cycles))
    if cycles.shape != freqs.shape:
        raise ValueError("N_CYCLES must be a scalar or have the same length as FREQS.")
    if np.any(freqs <= 0) or np.any(cycles <= 0):
        raise ValueError("FREQS and N_CYCLES must be positive.")


def build_feature_dir(subject):
    return get_feature_dir(BASE_PATH, 'decoding_tfa_features', subject)


def sanitize_token(text):
    token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text))
    token = token.strip("_")
    return token or "Unknown"


def build_comparison_id():
    left = "-".join(str(v) for v in GROUP_A)
    right = "-".join(str(v) for v in GROUP_B)
    return f"{TASK}_A_{left}_vs_B_{right}"


def to_zero_based_indices(indices_1based):
    return [v - 1 for v in indices_1based]


def matlab_string_array_to_list(value):
    array = np.asarray(value)
    if array.size == 0:
        return []
    flat = array.squeeze()
    if flat.ndim == 0:
        return [str(flat.item())]
    result = []
    for item in np.ravel(flat, order="F"):
        if isinstance(item, np.ndarray):
            result.append("".join(str(x) for x in item.tolist()).strip())
        else:
            result.append(str(item).strip())
    return result


def get_time_vector(mat, n_time):
    if "tfa_time_ms" in mat:
        times = np.asarray(mat["tfa_time_ms"], dtype=float).reshape(-1)
        if times.size == n_time:
            return times
    step_ms = 1000.0 / FS
    return DEFAULT_TMIN_MS + np.arange(n_time, dtype=float) * step_ms


def select_roi_files(feature_dir):
    mat_files = sorted(feature_dir.glob(ROI_PATTERN))
    mat_files = [path for path in mat_files if path.is_file() and path.suffix == ".mat"]
    if ROI is not None:
        mat_files = [path for path in mat_files if path.stem == ROI]
    return mat_files


def prepare_group_trials(data, condition_indices):
    selected = np.asarray(data[condition_indices, :, :, :], dtype=float)
    n_cond, n_rep, n_ch, n_time = selected.shape
    return selected.reshape(n_cond * n_rep, n_ch, n_time)


def morlet_power_batch(data, freqs, n_cycles, sfreq):
    data = np.asarray(data, dtype=float)
    n_trials, n_times = data.shape
    power = np.empty((n_trials, len(freqs), n_times), dtype=np.float32)

    centered = data - data.mean(axis=1, keepdims=True)
    for fi, (freq, cycles) in enumerate(zip(freqs, n_cycles)):
        sigma_t = cycles / (2.0 * math.pi * freq)
        half_width = max(1, int(math.ceil(5.0 * sigma_t * sfreq)))
        t = np.arange(-half_width, half_width + 1, dtype=float) / sfreq
        wavelet = np.exp(2j * math.pi * freq * t) * np.exp(-(t ** 2) / (2.0 * sigma_t ** 2))
        wavelet = wavelet - np.mean(wavelet)
        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))

        n_conv = n_times + wavelet.size - 1
        data_fft = np.fft.fft(centered, n=n_conv, axis=1)
        wavelet_fft = np.fft.fft(wavelet, n=n_conv)
        conv = np.fft.ifft(data_fft * wavelet_fft[None, :], axis=1)
        start = wavelet.size // 2
        conv = conv[:, start:start + n_times]
        power[:, fi, :] = np.abs(conv) ** 2

    return power


def apply_baseline(power, times_ms, baseline_ms, mode):
    if mode == "none":
        return power
    baseline_mask = (times_ms >= baseline_ms[0]) & (times_ms <= baseline_ms[1])
    if not np.any(baseline_mask):
        raise ValueError(
            f"No baseline points found in {baseline_ms}. Available time range is {times_ms[0]} to {times_ms[-1]} ms."
        )
    baseline = np.mean(power[..., baseline_mask], axis=-1, keepdims=True)
    baseline = np.maximum(baseline, np.finfo(float).eps)

    if mode == "db":
        return 10.0 * np.log10(np.maximum(power, np.finfo(float).eps) / baseline)
    if mode == "ratio":
        return power / baseline
    if mode == "percent":
        return (power - baseline) / baseline * 100.0
    if mode == "zscore":
        baseline_std = np.std(power[..., baseline_mask], axis=-1, ddof=0, keepdims=True)
        baseline_std = np.maximum(baseline_std, np.finfo(float).eps)
        return (power - baseline) / baseline_std
    raise ValueError(f"Unsupported baseline mode: {mode}")


def compute_group_tfr(trials, freqs, n_cycles, sfreq, times_ms):
    n_trials, n_ch, _ = trials.shape
    channel_powers = []
    for ch_idx in range(n_ch):
        channel_trials = trials[:, ch_idx, :]
        power = morlet_power_batch(channel_trials, freqs=freqs, n_cycles=n_cycles, sfreq=sfreq)
        power = apply_baseline(power, times_ms=times_ms, baseline_ms=BASELINE_MS, mode=BASELINE_MODE)
        channel_powers.append(power.astype(np.float32, copy=False))

    trial_power = np.stack(channel_powers, axis=-1)
    channel_mean = np.mean(trial_power, axis=0)
    roi_mean = np.mean(channel_mean, axis=-1)
    return trial_power, channel_mean, roi_mean


def build_output_dirs(output_root, comparison_id):
    if output_root is None:
        base_dir = get_roi_condition_tfa_dir(BASE_PATH, TASK, SUBJECT, comparison_id)
    else:
        base_dir = Path(output_root) / sanitize_token(comparison_id)
    dirs = {
        "base": base_dir,
        "mat": base_dir / "mat",
        "fig_roi_panel": base_dir / "figures" / "roi_panel",
        "fig_roi_diff": base_dir / "figures" / "roi_diff",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def compute_limits(group_a_map, group_b_map, diff_map):
    combined = np.concatenate([group_a_map.ravel(), group_b_map.ravel()])
    low = np.percentile(combined, 5)
    high = np.percentile(combined, 95)
    if math.isclose(low, high):
        low -= 1.0
        high += 1.0

    diff_lim = np.percentile(np.abs(diff_map), 95)
    diff_lim = max(float(diff_lim), 1e-6)
    return (float(low), float(high)), (-diff_lim, diff_lim)


def plot_roi_condition_grid(
    group_a_roi_mean,
    group_b_roi_mean,
    group_a_channel_mean,
    group_b_channel_mean,
    channel_names,
    times_ms,
    freqs,
    roi_name,
    save_path,
    vmin,
    vmax,
):
    row_labels = ["ROI mean"] + list(channel_names)
    n_channels = len(channel_names)
    n_rows = n_channels + 1
    height = max(3.0 * n_rows, 5.5)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(13.2, height),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    panel_data = [
        (GROUP_A_LABEL, group_a_roi_mean, group_a_channel_mean),
        (GROUP_B_LABEL, group_b_roi_mean, group_b_channel_mean),
    ]
    last_mesh = None

    for col_idx, (column_title, roi_mean, channel_mean) in enumerate(panel_data):
        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]
            current = roi_mean if row_idx == 0 else channel_mean[:, :, row_idx - 1]
            last_mesh = ax.pcolormesh(
                times_ms,
                freqs,
                current,
                shading="auto",
                cmap=POWER_CMAP,
                vmin=vmin,
                vmax=vmax,
            )
            ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
            if row_idx == 0:
                ax.set_title(column_title)
            if col_idx == 0:
                ax.set_ylabel(f"{row_labels[row_idx]}\nFreq (Hz)")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time (ms)")

    group_a_str = ",".join(str(v) for v in GROUP_A)
    group_b_str = ",".join(str(v) for v in GROUP_B)
    fig.suptitle(
        f"{SUBJECT} | {roi_name} | {TASK}\n"
        f"{GROUP_A_LABEL} (cond {group_a_str}) vs {GROUP_B_LABEL} (cond {group_b_str})",
        y=0.995,
    )
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.04, top=0.96, hspace=0.32, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.10, 0.018, 0.80])
    cbar = fig.colorbar(last_mesh, cax=cbar_ax)
    cbar.set_label(f"Power ({BASELINE_MODE})" if BASELINE_MODE != "none" else "Power")
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_roi_difference_grid(
    diff_roi_mean,
    diff_channel_mean,
    channel_names,
    times_ms,
    freqs,
    roi_name,
    save_path,
    vmin,
    vmax,
):
    row_labels = ["ROI diff"] + list(channel_names)
    n_rows = len(channel_names) + 1
    height = max(2.8 * n_rows, 5.0)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(7.2, height),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    last_mesh = None
    for row_idx in range(n_rows):
        ax = axes[row_idx, 0]
        current = diff_roi_mean if row_idx == 0 else diff_channel_mean[:, :, row_idx - 1]
        last_mesh = ax.pcolormesh(
            times_ms,
            freqs,
            current,
            shading="auto",
            cmap=DIFF_CMAP,
            vmin=vmin,
            vmax=vmax,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"{row_labels[row_idx]}\nFreq (Hz)")
        if row_idx == 0:
            ax.set_title(f"{GROUP_A_LABEL} - {GROUP_B_LABEL}")
        if row_idx == n_rows - 1:
            ax.set_xlabel("Time (ms)")

    group_a_str = ",".join(str(v) for v in GROUP_A)
    group_b_str = ",".join(str(v) for v in GROUP_B)
    fig.suptitle(
        f"{SUBJECT} | {roi_name} | {TASK}\n"
        f"Difference map: {GROUP_A_LABEL} (cond {group_a_str}) - {GROUP_B_LABEL} (cond {group_b_str})",
        y=0.995,
    )
    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.04, top=0.96, hspace=0.30)
    cbar_ax = fig.add_axes([0.90, 0.10, 0.018, 0.80])
    cbar = fig.colorbar(last_mesh, cax=cbar_ax)
    cbar.set_label("Power difference")
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_roi_result(
    save_path,
    roi_name,
    input_mat,
    channel_names,
    times_ms,
    freqs,
    group_a_trials,
    group_b_trials,
    group_a_trial_power,
    group_b_trial_power,
    group_a_channel_mean,
    group_b_channel_mean,
    diff_channel_mean,
    group_a_roi_mean,
    group_b_roi_mean,
):
    payload = {
        "roi_name": np.array([roi_name], dtype=object),
        "task": np.array([TASK], dtype=object),
        "group_a_conditions": np.asarray(GROUP_A, dtype=np.int16),
        "group_b_conditions": np.asarray(GROUP_B, dtype=np.int16),
        "group_a_label": np.array([GROUP_A_LABEL], dtype=object),
        "group_b_label": np.array([GROUP_B_LABEL], dtype=object),
        "times_ms": np.asarray(times_ms, dtype=np.float32),
        "freqs_hz": np.asarray(freqs, dtype=np.float32),
        "n_cycles": np.asarray(N_CYCLES, dtype=np.float32),
        "baseline_ms": np.asarray(BASELINE_MS, dtype=np.float32),
        "baseline_mode": np.array([BASELINE_MODE], dtype=object),
        "channel_names": np.asarray(channel_names, dtype=object),
        "group_a_trials": group_a_trials.astype(np.float32),
        "group_b_trials": group_b_trials.astype(np.float32),
        "group_a_channel_mean": group_a_channel_mean.astype(np.float32),
        "group_b_channel_mean": group_b_channel_mean.astype(np.float32),
        "diff_channel_mean": diff_channel_mean.astype(np.float32),
        "group_a_roi_mean": group_a_roi_mean.astype(np.float32),
        "group_b_roi_mean": group_b_roi_mean.astype(np.float32),
        "diff_roi_mean": (group_a_roi_mean - group_b_roi_mean).astype(np.float32),
        "config_json": np.array([
            json.dumps(
                {
                    "subject": SUBJECT,
                    "task": TASK,
                    "roi": ROI,
                    "group_a": GROUP_A,
                    "group_b": GROUP_B,
                    "group_a_label": GROUP_A_LABEL,
                    "group_b_label": GROUP_B_LABEL,
                    "baseline_ms": list(BASELINE_MS),
                    "baseline_mode": BASELINE_MODE,
                    "freqs": list(map(float, np.asarray(freqs).tolist())),
                    "n_cycles": list(map(float, np.asarray(N_CYCLES).tolist())),
                },
                ensure_ascii=False,
            )
        ], dtype=object),
    }

    if "tfa_band_names" in input_mat:
        payload["source_tfa_band_names"] = np.asarray(input_mat["tfa_band_names"], dtype=object)
    if "tfa_band_ranges" in input_mat:
        payload["source_tfa_band_ranges"] = np.asarray(input_mat["tfa_band_ranges"], dtype=np.float32)
    if SAVE_TRIAL_LEVEL_POWER:
        payload["group_a_trial_power"] = group_a_trial_power.astype(np.float32)
        payload["group_b_trial_power"] = group_b_trial_power.astype(np.float32)

    sio.savemat(save_path, payload, do_compression=True)


def write_summary_csv(rows, save_path):
    if not rows:
        return
    fieldnames = [
        "roi",
        "task",
        "group_a",
        "group_b",
        "n_trials_a",
        "n_trials_b",
        "n_channels",
        "n_timepoints",
        "n_freqs",
        "mat_path",
        "panel_figure",
        "diff_figure",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_roi(mat_file, output_dirs, freqs, n_cycles):
    mat = sio.loadmat(mat_file)
    field = f"tfa_{TASK}"
    if field not in mat:
        print(f"Skip {mat_file.stem}: missing {field}.")
        return None

    data = np.asarray(mat[field], dtype=float)
    if data.ndim != 4:
        print(f"Skip {mat_file.stem}: {field} shape is {data.shape}, expected 4D [Cond, Rep, Ch, Time].")
        return None

    group_a_indices = to_zero_based_indices(GROUP_A)
    group_b_indices = to_zero_based_indices(GROUP_B)
    max_cond = data.shape[0] - 1
    requested_max = max(group_a_indices + group_b_indices)
    if requested_max > max_cond:
        print(
            f"Skip {mat_file.stem}: only {data.shape[0]} conditions, requested up to {requested_max + 1}."
        )
        return None

    times_ms = get_time_vector(mat, n_time=data.shape[-1])
    channel_names = matlab_string_array_to_list(mat.get("tfa_roi_channels", np.array([])))
    if len(channel_names) != data.shape[2]:
        channel_names = [f"Ch{idx + 1}" for idx in range(data.shape[2])]

    group_a_trials = prepare_group_trials(data, group_a_indices)
    group_b_trials = prepare_group_trials(data, group_b_indices)

    group_a_trial_power, group_a_channel_mean, group_a_roi_mean = compute_group_tfr(
        group_a_trials,
        freqs=freqs,
        n_cycles=n_cycles,
        sfreq=FS,
        times_ms=times_ms,
    )
    group_b_trial_power, group_b_channel_mean, group_b_roi_mean = compute_group_tfr(
        group_b_trials,
        freqs=freqs,
        n_cycles=n_cycles,
        sfreq=FS,
        times_ms=times_ms,
    )
    diff_roi_mean = group_a_roi_mean - group_b_roi_mean
    diff_channel_mean = group_a_channel_mean - group_b_channel_mean

    power_limits, _ = compute_limits(group_a_roi_mean, group_b_roi_mean, diff_roi_mean)
    _, diff_limits = compute_limits(group_a_roi_mean, group_b_roi_mean, diff_roi_mean)
    roi_name = mat_file.stem
    safe_roi = sanitize_token(roi_name)

    panel_fig_path = output_dirs["fig_roi_panel"] / f"{safe_roi}.png"
    diff_fig_path = output_dirs["fig_roi_diff"] / f"{safe_roi}.png"
    mat_path = output_dirs["mat"] / f"{safe_roi}.mat"

    group_a_str = ",".join(str(v) for v in GROUP_A)
    group_b_str = ",".join(str(v) for v in GROUP_B)
    plot_roi_condition_grid(
        group_a_roi_mean,
        group_b_roi_mean,
        group_a_channel_mean,
        group_b_channel_mean,
        channel_names,
        times_ms=times_ms,
        freqs=freqs,
        roi_name=roi_name,
        save_path=panel_fig_path,
        vmin=power_limits[0],
        vmax=power_limits[1],
    )
    plot_roi_difference_grid(
        diff_roi_mean,
        diff_channel_mean,
        channel_names,
        times_ms=times_ms,
        freqs=freqs,
        roi_name=roi_name,
        save_path=diff_fig_path,
        vmin=diff_limits[0],
        vmax=diff_limits[1],
    )

    save_roi_result(
        save_path=mat_path,
        roi_name=roi_name,
        input_mat=mat,
        channel_names=channel_names,
        times_ms=times_ms,
        freqs=freqs,
        group_a_trials=group_a_trials,
        group_b_trials=group_b_trials,
        group_a_trial_power=group_a_trial_power,
        group_b_trial_power=group_b_trial_power,
        group_a_channel_mean=group_a_channel_mean,
        group_b_channel_mean=group_b_channel_mean,
        diff_channel_mean=diff_channel_mean,
        group_a_roi_mean=group_a_roi_mean,
        group_b_roi_mean=group_b_roi_mean,
    )

    print(f"Saved ROI TFA: {mat_path}")
    return {
        "roi": roi_name,
        "task": TASK,
        "group_a": group_a_str,
        "group_b": group_b_str,
        "n_trials_a": int(group_a_trials.shape[0]),
        "n_trials_b": int(group_b_trials.shape[0]),
        "n_channels": int(group_a_trials.shape[1]),
        "n_timepoints": int(group_a_trials.shape[2]),
        "n_freqs": int(len(freqs)),
        "mat_path": str(mat_path),
        "panel_figure": str(panel_fig_path),
        "diff_figure": str(diff_fig_path),
    }


def main():
    validate_config()
    feature_dir = build_feature_dir(SUBJECT)
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    comparison_id = build_comparison_id()
    output_dirs = build_output_dirs(OUTPUT_ROOT, comparison_id)
    freqs = np.asarray(FREQS, dtype=float)
    n_cycles = np.asarray(N_CYCLES, dtype=float)
    if n_cycles.ndim == 0:
        n_cycles = np.full(freqs.shape, float(n_cycles))

    mat_files = select_roi_files(feature_dir)
    if not mat_files:
        raise FileNotFoundError(f"No ROI .mat files found in {feature_dir}")

    summary_rows = []
    for mat_file in mat_files:
        row = process_roi(mat_file, output_dirs=output_dirs, freqs=freqs, n_cycles=n_cycles)
        if row is not None:
            summary_rows.append(row)

    summary_path = output_dirs["base"] / "roi_summary.csv"
    write_summary_csv(summary_rows, summary_path)
    if summary_rows:
        print(f"Saved summary: {summary_path}")
    else:
        print("No ROI results were generated.")


if __name__ == "__main__":
    _script_start_time = time.time()
    try:
        main()
    finally:
        print(f"Total runtime: {time.time() - _script_start_time:.2f} s")