"""Diagnostic plots for manually reviewing flagged channels."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import QC_PLOT_ROOT, TARGET_SFREQ, TASKS
from .io_seeg import load_set_metadata, open_fdt
from .signal_processing import filter_continuous


def plot_channel_diagnostics(
    subject: str,
    channel: str,
    output_path: Path,
    tasks: tuple[int, ...] = TASKS,
) -> None:
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(tasks), 3, figsize=(17, max(4, 3.2 * len(tasks))), squeeze=False)
    fig.suptitle(
        f"Channel review after 1-200 Hz + 50/100/150 Hz notch: {subject} {channel}",
        fontsize=13,
        fontweight="bold",
    )

    for row_index, task_num in enumerate(tasks):
        ax_raw, ax_filtered, ax_psd = axes[row_index]
        try:
            metadata = load_set_metadata(subject, task_num)
            labels = {label.upper(): idx for idx, label in enumerate(metadata.labels)}
            channel_index = labels.get(channel.upper())
            if channel_index is None:
                ax_raw.text(0.5, 0.5, "Channel absent", ha="center", va="center")
                ax_filtered.text(0.5, 0.5, "Channel absent", ha="center", va="center")
                ax_psd.text(0.5, 0.5, "Channel absent", ha="center", va="center")
                continue
            data = open_fdt(metadata)
            raw_full = np.asarray(data[channel_index, :], dtype=float)
            filtered_full = filter_continuous(raw_full, metadata.sfreq)
            raw = raw_full[: min(raw_full.size, int(metadata.sfreq * 20))]
            filtered = filtered_full[: min(filtered_full.size, int(TARGET_SFREQ * 20))]
            raw_time_s = np.arange(raw.size) / metadata.sfreq
            filtered_time_s = np.arange(filtered.size) / TARGET_SFREQ
            ax_raw.plot(raw_time_s, raw, linewidth=0.35, color="#555555")
            ax_raw.set_title(f"Task {task_num}: raw, first 20 s")
            ax_raw.set_xlabel("Time (s)")
            ax_raw.set_ylabel("Amplitude")

            ax_filtered.plot(filtered_time_s, filtered, linewidth=0.35, color="#1f77b4")
            ax_filtered.set_title("filtered, first 20 s")
            ax_filtered.set_xlabel("Time (s)")
            ax_filtered.set_ylabel("Amplitude")

            raw_nperseg = min(8192, raw_full.size)
            filtered_nperseg = min(8192, filtered_full.size)
            raw_freqs, raw_power = welch(raw_full, fs=metadata.sfreq, nperseg=raw_nperseg)
            filtered_freqs, filtered_power = welch(
                filtered_full,
                fs=TARGET_SFREQ,
                nperseg=filtered_nperseg,
            )
            raw_mask = (raw_freqs >= 1) & (raw_freqs <= 200)
            filtered_mask = (filtered_freqs >= 1) & (filtered_freqs <= 200)
            ax_psd.semilogy(
                raw_freqs[raw_mask],
                np.maximum(raw_power[raw_mask], 1e-20),
                color="#999999",
                linewidth=0.7,
                alpha=0.8,
                label="raw",
            )
            ax_psd.semilogy(
                filtered_freqs[filtered_mask],
                np.maximum(filtered_power[filtered_mask], 1e-20),
                color="#1f77b4",
                linewidth=0.9,
                label="filtered",
            )
            for frequency in (50, 100, 150):
                ax_psd.axvline(frequency, color="#d62728", linestyle=":", linewidth=0.7)
            ax_psd.set_title("PSD: raw vs filtered")
            ax_psd.set_xlabel("Frequency (Hz)")
            ax_psd.set_ylabel("Power")
            ax_psd.set_xlim(1, 200)
            ax_psd.legend(loc="best", fontsize=7)
        except Exception as exc:  # Keep the review set usable if one task is malformed.
            ax_raw.text(0.5, 0.5, f"Error: {exc}", ha="center", va="center", wrap=True)
            ax_filtered.axis("off")
            ax_psd.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
