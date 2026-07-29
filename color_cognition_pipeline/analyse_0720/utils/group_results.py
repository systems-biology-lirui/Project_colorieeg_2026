"""Group summaries and Nilearn visualisations for the all-channel branch."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from .plotting_science import set_science_style, save_figure, COLORS
from .wholebrain import location_table


def subject_accuracy_summary(electrode_set="union"):
    """Average subject decoding curves; subjects are the statistical units."""
    rows = []
    for subject in config.WHOLE_SUBJECTS:
        droot = config.all_subject_dir(subject) / "decoding"
        for path in sorted(droot.glob(f"*_{electrode_set}.csv")):
            d = pd.read_csv(path)
            if "accuracy" not in d or d.empty:
                continue
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    all_d = pd.concat(rows, ignore_index=True)
    group_cols = ["analysis", "modality", "electrode_set", "time_ms"]
    out = all_d.groupby(group_cols, as_index=False).agg(
        n_subjects=("subject", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        sd_accuracy=("accuracy", "std"),
    )
    out["sem_accuracy"] = out["sd_accuracy"] / np.sqrt(out["n_subjects"].clip(lower=1))
    out["ci_low"] = out["mean_accuracy"] - 1.96 * out["sem_accuracy"]
    out["ci_high"] = out["mean_accuracy"] + 1.96 * out["sem_accuracy"]
    outdir = config.ALL_RESULT_ROOT / "group" / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "subject_accuracy_mean.csv", index=False)
    return out


def plot_subject_accuracy_summary(electrode_set="union"):
    d = subject_accuracy_summary(electrode_set)
    if d.empty:
        return None
    outdir = config.ALL_RESULT_ROOT / "group" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    set_science_style()
    analyses = sorted(d.analysis.unique())
    fig, axes = plt.subplots(len(analyses), 2, figsize=(8.0, 2.4 * len(analyses)), squeeze=False,
                             constrained_layout=True)
    colors = {"erp": COLORS["blue"], "hg": COLORS["orange"]}
    for i, analysis in enumerate(analyses):
        for j, modality in enumerate(("erp", "hg")):
            ax = axes[i, j]
            x = d[(d.analysis == analysis) & (d.modality == modality)]
            if not x.empty:
                ax.plot(x.time_ms, x.mean_accuracy, color=colors[modality], lw=1.8)
                ax.fill_between(x.time_ms, x.ci_low, x.ci_high, color=colors[modality], alpha=.18, linewidth=0)
            ax.axhline(.5, color="#888", ls="--", lw=.7)
            ax.set_title(f"{analysis} · {modality.upper()}")
            ax.set(xlabel="Time (ms)", ylabel="Mean accuracy")
            ax.set_ylim(0.35, 1.02)
            ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, outdir / f"subject_accuracy_mean_{electrode_set}")


def nilearn_group_electrodes():
    """Overlay all localized electrodes and color-select electrodes in MNI glass brains."""
    from nilearn import plotting

    rows = []
    selected = []
    for subject in config.WHOLE_SUBJECTS:
        loc = location_table(subject).copy()
        if loc.empty:
            continue
        loc["subject"] = subject
        rows.append(loc)
        selpath = config.all_subject_dir(subject) / "color_select" / "color_select_electrodes.csv"
        if selpath.exists():
            s = pd.read_csv(selpath)[["channel", "color_select_evidence"]]
            selected.append(loc.merge(s, on="channel", how="inner"))
    all_loc = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    all_sel = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    outdir = config.ALL_RESULT_ROOT / "group" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    all_loc.to_csv(config.ALL_RESULT_ROOT / "group" / "tables" / "all_localized_electrodes.csv", index=False)
    all_sel.to_csv(config.ALL_RESULT_ROOT / "group" / "tables" / "all_color_select_electrodes.csv", index=False)

    def one(data, stem, title, include_gray=False):
        if data.empty:
            return None
        coords = data[["mni_x", "mni_y", "mni_z"]].to_numpy(float)
        keep = np.isfinite(coords).all(axis=1)
        coords = coords[keep]
        display = plotting.plot_glass_brain(None, display_mode="lyrz", colorbar=False,
                                            black_bg=False, plot_abs=False, annotate=True,
                                            title=title)
        if include_gray:
            display.add_markers(coords, marker_color="#BFC3C8", marker_size=4, alpha=.35)
            colored = data.loc[keep]
        else:
            colored = data.loc[keep]
        palette = {"ERP_only": "#2166AC", "HG_only": "#E08214", "ERP_and_HG": "#B2182B"}
        if "color_select_evidence" in colored:
            for evidence, color in palette.items():
                c = colored[colored.color_select_evidence == evidence][["mni_x", "mni_y", "mni_z"]].to_numpy(float)
                if len(c):
                    display.add_markers(c, marker_color=color, marker_size=8, alpha=.9)
        path = outdir / stem
        display.savefig(str(path.with_suffix(".png")), dpi=600)
        display.savefig(str(path.with_suffix(".pdf")))
        display.close()
        return path

    one(all_loc, "nilearn_all_localized_electrodes", "All localized electrodes · MNI", include_gray=True)
    one(all_sel, "nilearn_color_select_electrodes", "Color-select electrodes · MNI")
    return all_loc, all_sel
