"""Plot the S1/S2 time-resolved single-electrode decoding curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SET_NAMES = ("S1", "S2", "S1_or_S2", "CSC")
ANALYSES = ("task3_red_green", "task2_memory_color")
COLORS = {"task3_red_green": "#b2182b", "task2_memory_color": "#2166ac"}
LABELS = {
    "task3_red_green": "Task3 physical red/green",
    "task2_memory_color": "Task2 gray-fruit memory red/green",
}


def plot(out: Path, signal: str = "raw200", window: str = "100-400") -> Path:
    stage = out / "stage06_exploration"
    curves_path = stage / f"s1s2_timeresolved_electrode_curves_{window}_{signal}.npz"
    summary_path = stage / f"s1s2_timeresolved_electrode_summary_{window}_{signal}.csv"
    clusters_path = stage / f"s1s2_timeresolved_group_clusters_{window}_{signal}.csv"
    data = np.load(curves_path)
    table = pd.read_csv(summary_path)
    clusters = pd.read_csv(clusters_path)
    times = np.asarray(data["times_ms"], dtype=float)

    fig, axes = plt.subplots(4, 2, figsize=(15, 18), sharex=True, sharey=True)
    for row, set_name in enumerate(SET_NAMES):
        for col, analysis in enumerate(ANALYSES):
            ax = axes[row, col]
            subset = table[
                table.analysis.eq(analysis)
                & table.set_membership.str.split(";").apply(lambda names: set_name in names)
            ]
            curves = []
            for _, record in subset.iterrows():
                key = f"{record.subject}__{record.channel}__{analysis}"
                curve = np.asarray(data[key], dtype=float)
                curves.append(curve)
                ax.plot(times, curve, color=COLORS[analysis], alpha=0.18, linewidth=0.7)
            if curves:
                mean_curve = np.nanmean(np.asarray(curves), axis=0)
                ax.plot(times, mean_curve, color="black", linewidth=2.0, label="electrode mean")
            ax.axhline(0.5, color="0.45", linestyle="--", linewidth=0.8)
            ax.axvline(0, color="0.2", linewidth=0.8)
            sig = clusters[
                clusters.set_name.eq(set_name) & clusters.analysis.eq(analysis)
            ]
            sig = sig[sig.p.notna() & (sig.p < 0.05)]
            for _, cluster in sig.iterrows():
                ax.axvspan(cluster.start_ms, cluster.end_ms, color="#fdae61", alpha=0.20)
                ax.text(
                    (cluster.start_ms + cluster.end_ms) / 2,
                    0.985,
                    f"p={cluster.p:.3f}",
                    ha="center",
                    va="top",
                    fontsize=8,
                )
            ax.set_title(f"{set_name}: {LABELS[analysis]} (n={len(curves)})")
            ax.set_ylim(0.35, 1.0)
            ax.grid(axis="y", alpha=0.2)
            if col == 0:
                ax.set_ylabel("accuracy")
            if row == len(SET_NAMES) - 1:
                ax.set_xlabel("time (ms)")
    fig.suptitle(
        "Norm1 S1/S2 time-resolved single-electrode decoding\n"
        "thin lines = individual electrodes; black = electrode mean; orange = group cluster",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = stage / f"s1s2_timeresolved_single_electrode_curves_{window}_{signal}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--signal", choices=("lf30", "raw200"), default="raw200")
    parser.add_argument("--window", choices=("1-300", "100-400"), default="100-400")
    args = parser.parse_args()
    print(plot(args.out, args.signal, args.window))


if __name__ == "__main__":
    main()
