"""Count and plot the Norm2 ∩ Norm1-S1 electrode intersection in MNI space."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from nilearn import plotting


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "result/final_analysis_seeg_20260806_corrected/stage01_selection/electrode_sets_and_csc_100-400_lf30.csv"
OUT = ROOT / "result/final_analysis_seeg_20260806_corrected/stage01_selection/norm2_norm1_s1_intersection_glass_brain.png"


def main() -> None:
    df = pd.read_csv(CSV)
    for col in ("N2_union", "strategy1"):
        df[col] = df[col].astype(str).str.lower().eq("true")
    inter = df.loc[df["N2_union"] & df["strategy1"]].copy()
    coords = inter[["mni_x", "mni_y", "mni_z"]].to_numpy()

    fig = plt.figure(figsize=(13, 4.8), facecolor="white")
    display = plotting.plot_markers(
        coords,
        node_color="#d62728",
        node_size=75,
        alpha=0.95,
        display_mode="lyrz",
        title=f"Norm2 ∩ Norm1 S1: n={len(inter)} electrodes",
        colorbar=False,
        figure=fig,
        black_bg=False,
    )
    display.add_markers(coords, marker_color="#1f77b4", marker_size=18)
    fig.text(
        0.5,
        0.01,
        "Red: intersection electrodes; blue centers: MNI coordinates | 100–400 ms, lf30",
        ha="center",
        fontsize=9,
    )
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"intersection={len(inter)}")
    print(inter[["subject", "channel", "roi", "mni_x", "mni_y", "mni_z", "PC_hemisphere"]].to_string(index=False))
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
