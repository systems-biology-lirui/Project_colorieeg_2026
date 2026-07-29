"""CCEP 逐电极响应统计脚本。

独立版本只依赖当前打包目录中的 ROI 特征文件和本地 runtime_config，
默认输出写入 workspace/result/ccep/。
"""

import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import ttest_1samp


BASE_PATH = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from runtime_config import load_runtime_config


DEFAULT_CONFIG = {
    "subject": "test001",
    "modality": "erp",
    "alpha": 0.05,
    "apply_fdr": True,
    "artifact_start_ms": 10.0,
    "min_consecutive_sig_points": 3,
    "roi_pattern": "*.mat",
    "output_dir": None,
    "dpi": 220,
}


def sanitize_token(text):
    """把任意标签清洗成适合保存文件的安全字符串。"""
    token = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text))
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def build_config():
    """构建统计参数，并把输入输出路径固定到独立目录的 workspace 下。"""
    config = dict(DEFAULT_CONFIG)
    config.update(load_runtime_config(__file__, sections=("python_defaults", "ccep_defaults")))
    config["subject"] = str(config["subject"])
    config["modality"] = str(config["modality"]).lower()
    if config["modality"] not in {"erp", "tfa"}:
        raise ValueError(f"Unsupported modality: {config['modality']}")
    config["alpha"] = float(config["alpha"])
    config["artifact_start_ms"] = float(config["artifact_start_ms"])
    config["min_consecutive_sig_points"] = int(config["min_consecutive_sig_points"])
    config["apply_fdr"] = bool(config["apply_fdr"])
    config["dpi"] = int(config["dpi"])
    if config["output_dir"]:
        config["output_dir"] = Path(config["output_dir"])
    else:
        config["output_dir"] = BASE_PATH / "workspace" / "result" / "ccep" / config["subject"] / config["modality"]
    config["feature_dir"] = BASE_PATH / "workspace" / "feature" / f"ccep_{config['modality']}" / config["subject"]
    return config


def ensure_text_list(value):
    """把 MATLAB 读出的文本字段统一转成 Python 字符串列表。"""
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.flat]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def ensure_object_list(value):
    """把 MATLAB 的 object 数组或标量统一转成 Python 列表。"""
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value.flat)
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def load_roi_feature(mat_path):
    """读取单个 ROI 特征文件，并抽取后续统计用到的核心字段。"""
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "roi_feature" not in mat:
        raise KeyError(f"roi_feature not found in {mat_path}")

    roi = mat["roi_feature"]
    site_labels = ensure_text_list(roi.site_labels)
    channel_labels = ensure_text_list(roi.channel_labels)
    site_data = [np.asarray(item, dtype=float) for item in ensure_object_list(roi.site_data)]
    time_ms = np.asarray(roi.time_ms, dtype=float).reshape(-1)
    trial_count_per_site = np.asarray(roi.trial_count_per_site, dtype=int).reshape(-1)
    return {
        "roi_name": str(roi.roi_name),
        "subject": str(roi.subject),
        "modality": str(roi.modality),
        "site_labels": site_labels,
        "channel_labels": channel_labels,
        "site_data": site_data,
        "time_ms": time_ms,
        "trial_count_per_site": trial_count_per_site,
    }


def parse_contact_label(channel_label):
    """把电极标签拆成轴名和触点编号，例如 A12 -> (A, 12)。"""
    match = re.match(r"^([A-Za-z]+)(\d+)$", str(channel_label).strip())
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def derive_excluded_channels(site_label, all_channel_labels):
    """根据刺激位点标签推导需要排除的刺激相关电极。"""
    excluded = []
    for part in str(site_label).split("-"):
        shaft_name, contact_num = parse_contact_label(part.strip())
        if shaft_name is None:
            if part in all_channel_labels:
                excluded.append(part)
            continue
        for offset in (-1, 0, 1):
            candidate = f"{shaft_name}{contact_num + offset}"
            if candidate in all_channel_labels:
                excluded.append(candidate)
    return stable_unique(excluded)


def stable_unique(values):
    """按原始出现顺序去重。"""
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def benjamini_hochberg_mask(p_values, alpha):
    """执行 Benjamini-Hochberg FDR 校正。"""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p_values)
    ranked = p_values[order]
    thresholds = alpha * (np.arange(1, ranked.size + 1) / ranked.size)
    below = ranked <= thresholds
    if not np.any(below):
        return np.zeros_like(p_values, dtype=bool)
    cutoff_rank = np.max(np.flatnonzero(below))
    cutoff_value = ranked[cutoff_rank]
    return p_values <= cutoff_value


def enforce_min_consecutive(mask, min_points):
    """要求显著时间点至少连续出现指定采样点数。"""
    mask = np.asarray(mask, dtype=bool)
    if min_points <= 1 or not np.any(mask):
        return mask

    filtered = np.zeros_like(mask, dtype=bool)
    start = None
    for idx, is_sig in enumerate(mask):
        if is_sig and start is None:
            start = idx
        elif not is_sig and start is not None:
            if idx - start >= min_points:
                filtered[start:idx] = True
            start = None
    if start is not None and len(mask) - start >= min_points:
        filtered[start:] = True
    return filtered


def compute_channel_stats(trials, times_ms, config, is_excluded, exclusion_reason):
    """对单个电极做逐时间点单样本 t 检验，并提取主要响应指标。"""
    trials = np.asarray(trials, dtype=float)
    n_trials = int(trials.shape[0])
    mean_waveform = np.nanmean(trials, axis=0)
    if n_trials > 1:
        sem_waveform = np.nanstd(trials, axis=0, ddof=1) / math.sqrt(n_trials)
    else:
        sem_waveform = np.zeros_like(mean_waveform)

    p_values = np.ones_like(mean_waveform, dtype=float)
    t_values = np.zeros_like(mean_waveform, dtype=float)
    sig_mask = np.zeros_like(mean_waveform, dtype=bool)

    valid_time_mask = times_ms >= config["artifact_start_ms"]
    if not is_excluded:
        t_values, p_values = ttest_1samp(trials, popmean=0.0, axis=0, nan_policy="omit")
        t_values = np.nan_to_num(t_values, nan=0.0)
        p_values = np.nan_to_num(p_values, nan=1.0)

        post_p = p_values[valid_time_mask]
        if config["apply_fdr"]:
            post_sig = benjamini_hochberg_mask(post_p, config["alpha"])
        else:
            post_sig = post_p < config["alpha"]
        post_sig = enforce_min_consecutive(post_sig, config["min_consecutive_sig_points"])
        sig_mask[valid_time_mask] = post_sig

    sig_indices = np.flatnonzero(sig_mask)
    has_response = sig_indices.size > 0
    if has_response:
        peak_index = sig_indices[np.argmax(np.abs(mean_waveform[sig_indices]))]
        first_sig_ms = float(times_ms[sig_indices[0]])
        last_sig_ms = float(times_ms[sig_indices[-1]])
        peak_latency_ms = float(times_ms[peak_index])
        peak_amplitude = float(mean_waveform[peak_index])
        auc_abs = float(np.trapezoid(np.abs(mean_waveform[sig_indices]), times_ms[sig_indices] / 1000.0))
    else:
        first_sig_ms = np.nan
        last_sig_ms = np.nan
        peak_latency_ms = np.nan
        peak_amplitude = np.nan
        auc_abs = 0.0

    min_p_post = float(np.min(p_values[valid_time_mask])) if np.any(valid_time_mask) else np.nan
    return {
        "n_trials": n_trials,
        "is_excluded": bool(is_excluded),
        "exclusion_reason": exclusion_reason,
        "has_response": has_response,
        "n_sig_points": int(sig_indices.size),
        "first_sig_ms": first_sig_ms,
        "last_sig_ms": last_sig_ms,
        "peak_latency_ms": peak_latency_ms,
        "peak_amplitude": peak_amplitude,
        "peak_amplitude_abs": float(abs(peak_amplitude)) if has_response else np.nan,
        "auc_abs": auc_abs,
        "min_p_post": min_p_post,
        "mean_waveform": mean_waveform,
        "sem_waveform": sem_waveform,
        "sig_mask": sig_mask,
        "p_values": p_values,
        "t_values": t_values,
    }


def collect_site_channel_results(feature_files, config):
    """汇总所有 ROI 文件，构建以刺激 site 为索引的逐电极统计结果。"""
    loaded_features = [load_roi_feature(path) for path in feature_files]
    reference = loaded_features[0]
    site_labels = reference["site_labels"]
    times_ms = reference["time_ms"]
    all_channel_labels = stable_unique(
        channel
        for feature in loaded_features
        for channel in feature["channel_labels"]
    )

    excluded_channels_by_site = {
        site_label: set(derive_excluded_channels(site_label, all_channel_labels))
        for site_label in site_labels
    }

    site_channel_results = {site_label: {} for site_label in site_labels}
    roi_channel_map = {}
    channel_roi_membership = defaultdict(set)

    for feature in loaded_features:
        roi_name = feature["roi_name"]
        roi_channel_map[roi_name] = list(feature["channel_labels"])
        for site_index, site_label in enumerate(feature["site_labels"]):
            site_data = np.asarray(feature["site_data"][site_index], dtype=float)
            if site_data.ndim == 2:
                site_data = site_data[:, np.newaxis, :]
            if site_data.ndim != 3:
                raise ValueError(f"Unexpected site_data shape for {roi_name} / {site_label}: {site_data.shape}")

            for channel_index, channel_label in enumerate(feature["channel_labels"]):
                channel_roi_membership[(site_label, channel_label)].add(roi_name)
                if channel_label in site_channel_results[site_label]:
                    continue

                is_excluded = channel_label in excluded_channels_by_site[site_label]
                exclusion_reason = "stim_pair_or_neighbor" if is_excluded else ""
                stats = compute_channel_stats(
                    trials=site_data[:, channel_index, :],
                    times_ms=times_ms,
                    config=config,
                    is_excluded=is_excluded,
                    exclusion_reason=exclusion_reason,
                )
                stats["channel"] = channel_label
                site_channel_results[site_label][channel_label] = stats

    for site_label, channel_map in site_channel_results.items():
        for channel_label, stats in channel_map.items():
            stats["rois"] = sorted(channel_roi_membership[(site_label, channel_label)])

    return site_labels, times_ms, site_channel_results, roi_channel_map


def save_site_channel_tables(site_labels, site_channel_results, output_dir):
    """保存逐 site 的电极统计表和总表。"""
    all_rows = []
    per_site_dir = output_dir / "electrode_stats" / "by_site"
    per_site_dir.mkdir(parents=True, exist_ok=True)

    for site_label in site_labels:
        site_rows = []
        for channel_label, stats in sorted(site_channel_results[site_label].items()):
            row = {
                "site_label": site_label,
                "channel": channel_label,
                "roi_membership": ";".join(stats["rois"]),
                "n_trials": stats["n_trials"],
                "is_excluded": int(stats["is_excluded"]),
                "exclusion_reason": stats["exclusion_reason"],
                "has_response": int(stats["has_response"]),
                "n_sig_points": stats["n_sig_points"],
                "first_sig_ms": stats["first_sig_ms"],
                "last_sig_ms": stats["last_sig_ms"],
                "peak_latency_ms": stats["peak_latency_ms"],
                "peak_amplitude": stats["peak_amplitude"],
                "peak_amplitude_abs": stats["peak_amplitude_abs"],
                "auc_abs": stats["auc_abs"],
                "min_p_post": stats["min_p_post"],
            }
            site_rows.append(row)
            all_rows.append(row)

        site_df = pd.DataFrame(site_rows)
        site_df.to_csv(per_site_dir / f"{sanitize_token(site_label)}_channel_stats.csv", index=False)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(output_dir / "electrode_stats" / "all_channel_stats.csv", index=False)
    return all_df


def save_site_npz(site_label, site_results, times_ms, output_dir):
    """把单个刺激 site 的波形、显著掩码和 p 值导出为 npz。"""
    included_items = [(channel, stats) for channel, stats in site_results.items() if not stats["is_excluded"]]
    if not included_items:
        return

    channel_labels = np.array([channel for channel, _ in included_items], dtype=object)
    mean_waveforms = np.vstack([stats["mean_waveform"] for _, stats in included_items])
    sig_mask = np.vstack([stats["sig_mask"] for _, stats in included_items])
    p_values = np.vstack([stats["p_values"] for _, stats in included_items])

    npz_dir = output_dir / "electrode_stats" / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_dir / f"{sanitize_token(site_label)}.npz",
        times_ms=times_ms,
        channel_labels=channel_labels,
        mean_waveforms=mean_waveforms,
        sig_mask=sig_mask,
        p_values=p_values,
    )


def plot_site_heatmap(site_label, site_results, times_ms, output_dir, dpi, artifact_start_ms):
    """绘制单个刺激 site 的逐电极平均响应热图和显著性热图。"""
    included_items = [(channel, stats) for channel, stats in site_results.items() if not stats["is_excluded"]]
    if not included_items:
        return

    def sort_key(item):
        channel, stats = item
        first_sig = stats["first_sig_ms"]
        first_sig_key = first_sig if not np.isnan(first_sig) else np.inf
        peak_amp = stats["peak_amplitude_abs"] if not np.isnan(stats["peak_amplitude_abs"]) else -np.inf
        return (first_sig_key, -peak_amp, channel)

    included_items.sort(key=sort_key)
    channel_labels = [channel for channel, _ in included_items]
    mean_matrix = np.vstack([stats["mean_waveform"] for _, stats in included_items])
    sig_matrix = np.vstack([stats["sig_mask"] for _, stats in included_items]).astype(float)
    vmax = np.nanpercentile(np.abs(mean_matrix), 95)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig_height = max(6.0, 0.28 * len(channel_labels) + 2.5)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1.4]},
    )

    axes[0].imshow(
        mean_matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=[times_ms[0], times_ms[-1], len(channel_labels) - 0.5, -0.5],
    )
    axes[0].set_yticks(np.arange(len(channel_labels)))
    axes[0].set_yticklabels(channel_labels, fontsize=8)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1.0)
    axes[0].axvline(artifact_start_ms, color="#1f77b4", linestyle=":", linewidth=1.0)
    axes[0].set_title(f"{site_label} | Mean Response Heatmap")
    axes[0].set_ylabel("Channel")

    axes[1].imshow(
        sig_matrix,
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
        extent=[times_ms[0], times_ms[-1], len(channel_labels) - 0.5, -0.5],
    )
    axes[1].set_yticks(np.arange(len(channel_labels)))
    axes[1].set_yticklabels(channel_labels, fontsize=8)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].axvline(artifact_start_ms, color="#1f77b4", linestyle=":", linewidth=1.0)
    axes[1].set_title("Significant Time Points")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Channel")

    plt.tight_layout()
    save_dir = output_dir / "figures" / "site_heatmaps"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{sanitize_token(site_label)}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_roi_summary(site_labels, site_channel_results, roi_channel_map):
    """把逐电极结果进一步压缩成 ROI 层面的摘要指标。"""
    rows = []
    for site_label in site_labels:
        channel_map = site_channel_results[site_label]
        for roi_name, roi_channels in sorted(roi_channel_map.items()):
            roi_stats = [channel_map[channel] for channel in roi_channels if channel in channel_map]
            if not roi_stats:
                continue

            included_stats = [stats for stats in roi_stats if not stats["is_excluded"]]
            significant_stats = [stats for stats in included_stats if stats["has_response"]]
            rows.append(
                {
                    "site_label": site_label,
                    "roi": roi_name,
                    "n_channels_total": len(roi_stats),
                    "n_channels_included": len(included_stats),
                    "n_channels_significant": len(significant_stats),
                    "earliest_latency_ms": min((stats["first_sig_ms"] for stats in significant_stats), default=np.nan),
                    "max_abs_peak_amplitude": max((stats["peak_amplitude_abs"] for stats in significant_stats), default=np.nan),
                    "mean_abs_peak_amplitude": float(np.nanmean([stats["peak_amplitude_abs"] for stats in significant_stats])) if significant_stats else np.nan,
                    "mean_auc_abs": float(np.nanmean([stats["auc_abs"] for stats in significant_stats])) if significant_stats else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_roi_matrix(summary_df, value_column, title, cmap, output_path, dpi, fmt=".1f"):
    """把 ROI 摘要指标渲染成 ROI × 刺激 site 的矩阵图。"""
    matrix = summary_df.pivot(index="roi", columns="site_label", values=value_column)
    if matrix.empty:
        return

    values = matrix.to_numpy(dtype=float)
    fig_width = max(8.0, 1.0 * matrix.shape[1] + 2.5)
    fig_height = max(6.0, 0.45 * matrix.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = values[row, col]
            label = "" if np.isnan(value) else format(value, fmt)
            if label:
                ax.text(col, row, label, ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(image, ax=ax, shrink=0.9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_summary_artifacts(config, site_labels, times_ms, site_channel_results, roi_summary_df):
    """统一写出表格、图像、npz 和本次统计的配置快照。"""
    output_dir = config["output_dir"]
    (output_dir / "electrode_stats").mkdir(parents=True, exist_ok=True)
    (output_dir / "roi_summary").mkdir(parents=True, exist_ok=True)

    electrode_df = save_site_channel_tables(site_labels, site_channel_results, output_dir)
    electrode_summary = (
        electrode_df[electrode_df["is_excluded"] == 0]
        .groupby("site_label", as_index=False)
        .agg(
            n_channels=("channel", "count"),
            n_significant=("has_response", "sum"),
            earliest_latency_ms=("first_sig_ms", "min"),
            max_abs_peak_amplitude=("peak_amplitude_abs", "max"),
        )
    )
    electrode_summary.to_csv(output_dir / "electrode_stats" / "site_summary.csv", index=False)

    for site_label in site_labels:
        save_site_npz(site_label, site_channel_results[site_label], times_ms, output_dir)
        plot_site_heatmap(
            site_label,
            site_channel_results[site_label],
            times_ms,
            output_dir,
            config["dpi"],
            config["artifact_start_ms"],
        )

    roi_summary_df.to_csv(output_dir / "roi_summary" / "roi_summary.csv", index=False)
    plot_roi_matrix(
        roi_summary_df,
        value_column="n_channels_significant",
        title=f"{config['subject']} | {config['modality']} | Significant Channels per ROI",
        cmap="YlOrRd",
        output_path=output_dir / "figures" / "roi_matrices" / "significant_channel_count.png",
        dpi=config["dpi"],
        fmt=".0f",
    )
    plot_roi_matrix(
        roi_summary_df,
        value_column="earliest_latency_ms",
        title=f"{config['subject']} | {config['modality']} | Earliest ROI Latency (ms)",
        cmap="viridis",
        output_path=output_dir / "figures" / "roi_matrices" / "earliest_latency_ms.png",
        dpi=config["dpi"],
        fmt=".1f",
    )
    plot_roi_matrix(
        roi_summary_df,
        value_column="max_abs_peak_amplitude",
        title=f"{config['subject']} | {config['modality']} | Max Abs Peak Amplitude",
        cmap="magma",
        output_path=output_dir / "figures" / "roi_matrices" / "max_abs_peak_amplitude.png",
        dpi=config["dpi"],
        fmt=".2f",
    )

    config_dump = {key: (str(value) if isinstance(value, Path) else value) for key, value in config.items()}
    (output_dir / "config_used.json").write_text(json.dumps(config_dump, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    """统计脚本主入口。"""
    config = build_config()
    feature_dir = config["feature_dir"]
    feature_files = sorted(path for path in feature_dir.glob(config["roi_pattern"]) if path.is_file())
    if not feature_files:
        raise FileNotFoundError(f"No ROI feature files found in {feature_dir}")

    site_labels, times_ms, site_channel_results, roi_channel_map = collect_site_channel_results(feature_files, config)
    roi_summary_df = build_roi_summary(site_labels, site_channel_results, roi_channel_map)
    save_summary_artifacts(config, site_labels, times_ms, site_channel_results, roi_summary_df)

    print(f"Saved CCEP {config['modality']} response stats to {config['output_dir']}")


if __name__ == "__main__":
    script_start = time.time()
    try:
        main()
    finally:
        print(f"Total runtime: {time.time() - script_start:.2f} s")