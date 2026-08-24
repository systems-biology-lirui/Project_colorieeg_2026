"""Stage 2: spatial patch selection, CSC intersection, and CSC signal stats.

Norm 2 keeps common centers within ``RADIUS_MM`` of any bilateral PC/CC/AC
patch center (Talairach -> MNI via the Brett transform). CSC is defined as
(Strategy 1 union Strategy 2) intersect Norm 2 for each analysis variant.

Amplitude comparisons use only baseline-mean subtraction (no across-trial
standardization). Band-power comparisons use Welch log power baseline-z-scored
against the per-condition baseline-trial distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.spectral_features import (
    BAND_NAMES,
    FEATURE_BANDS,
    band_power_baseline_z,
    welch_band_power,
    window_mask,
)
from analysis.common import (
    BASELINE_MS,
    GRAY_FRUITS,
    PATCH_TALAIRACH,
    RADIUS_MM,
    SUBJECTS,
    AnalysisVariant,
    all_variants,
    baseline_subtract,
    bh_adjust,
    common_channels,
    load_conditions,
    natural_key,
    patch_table,
    read_localization,
)
from analysis.selection import prepare_signal


def run_spatial_selection(
    out: Path,
    functional: pd.DataFrame,
    variant: AnalysisVariant,
    subjects: Iterable[str] = SUBJECTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patches = patch_table()
    patch_centers = {
        patch: patches.loc[
            patches.patch == patch, ["hemisphere", "mni_x", "mni_y", "mni_z"]
        ].to_numpy()
        for patch in PATCH_TALAIRACH
    }
    func_idx = (
        functional.set_index(["subject", "channel"])
        if not functional.empty
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    for subject in subjects:
        loc = read_localization(subject).set_index("channel")
        for channel in common_channels(subject):
            if channel not in loc.index:
                continue
            coord_values = loc.loc[
                channel, ["mni_x", "mni_y", "mni_z"]
            ].astype(float)
            if not np.isfinite(coord_values).all():
                continue
            coord = coord_values.to_numpy()
            row: dict[str, object] = {
                "subject": subject,
                "channel": channel,
                "mni_x": coord[0],
                "mni_y": coord[1],
                "mni_z": coord[2],
                "roi": loc.loc[channel, "roi"],
                "color_with_sti": loc.loc[channel, "color_with_sti"],
                "functional_roi": bool(loc.loc[channel, "is_target_roi"]),
                "window": variant.window_label,
                "signal": variant.signal,
            }
            for patch, centers in patch_centers.items():
                distances = np.linalg.norm(
                    centers[:, 1:4].astype(float) - coord[None, :], axis=1
                )
                k = int(np.argmin(distances))
                row[f"{patch}_distance_mm"] = float(distances[k])
                row[f"{patch}_hemisphere"] = str(centers[k, 0])
                row[f"N2_{patch}"] = bool(distances[k] <= RADIUS_MM)
            row["N2_union"] = bool(
                any(row[f"N2_{p}"] for p in PATCH_TALAIRACH)
            )
            if not functional.empty and (subject, channel) in func_idx.index:
                func_row = func_idx.loc[(subject, channel)]
                row["strategy1"] = bool(func_row["strategy1"])
                row["strategy2"] = bool(func_row["strategy2"])
                row["strategy1_merged"] = bool(func_row["strategy1_merged"])
                row["strategy1_fdr"] = bool(func_row["strategy1_fdr"])
                row["strategy2_fdr"] = bool(func_row["strategy2_fdr"])
            else:
                row["strategy1"] = False
                row["strategy2"] = False
                row["strategy1_merged"] = False
                row["strategy1_fdr"] = False
                row["strategy2_fdr"] = False
            row["N1_union"] = bool(row["strategy1"] or row["strategy2"])
            row["CSC"] = bool(row["N1_union"] and row["N2_union"])
            # Historical merged Strategy 1 (pooled color vs gray) is kept as a
            # first-class CSC alternative so both prompt definitions are
            # browsable.
            row["CSC_merged"] = bool(
                (row["strategy1_merged"] or row["strategy2"]) and row["N2_union"]
            )
            row["CSC_strategy1"] = bool(row["strategy1"] and row["N2_union"])
            row["CSC_strategy2"] = bool(row["strategy2"] and row["N2_union"])
            row["CSC_fdr"] = bool(
                (row["strategy1_fdr"] or row["strategy2_fdr"]) and row["N2_union"]
            )
            rows.append(row)
    table = pd.DataFrame(rows)
    stage = out / "stage01_selection"
    stage.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        stage / f"electrode_sets_and_csc_{variant.suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows: list[dict[str, object]] = []
    if not table.empty:
        for subject, grp in table.groupby("subject"):
            summary_rows.append(
                {
                    "subject": subject,
                    "window": variant.window_label,
                    "signal": variant.signal,
                    "common_all_task_centers": int(len(grp)),
                    "strategy1": int(grp.strategy1.sum()),
                    "strategy2": int(grp.strategy2.sum()),
                    "N2_PC": int(grp.N2_PC.sum()),
                    "N2_CC": int(grp.N2_CC.sum()),
                    "N2_AC": int(grp.N2_AC.sum()),
                    "N2_union": int(grp.N2_union.sum()),
                    "CSC": int(grp.CSC.sum()),
                    "CSC_merged": int(grp.CSC_merged.sum()),
                    "CSC_strategy1": int(grp.CSC_strategy1.sum()),
                    "CSC_strategy2": int(grp.CSC_strategy2.sum()),
                    "color_with_sti_in_common": int(grp.color_with_sti.sum()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        stage / f"electrode_set_summary_by_subject_{variant.suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table, summary


def _pie_categories_overall(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    n2 = df.N2_union.astype(bool)
    n1 = df.N1_union.astype(bool)
    masks = [n2 & ~n1, n2 & n1, ~n2 & n1, ~n2 & ~n1]
    labels = ["N2 only", "CSC / overlap", "N1 only", "Neither"]
    return labels, [int(x.sum()) for x in masks]


def _pie_categories_three(
    df: pd.DataFrame, patch: str
) -> tuple[list[str], list[int]]:
    a, b, c = (
        df[f"N2_{patch}"].astype(bool),
        df.strategy1.astype(bool),
        df.strategy2.astype(bool),
    )
    masks = [
        a & ~b & ~c,
        ~a & b & ~c,
        ~a & ~b & c,
        a & b & ~c,
        a & ~b & c,
        ~a & b & c,
        a & b & c,
        ~a & ~b & ~c,
    ]
    labels = ["N2", "S1", "S2", "N2+S1", "N2+S2", "S1+S2", "CSC triple", "None"]
    return labels, [int(m.sum()) for m in masks]


def _three_set_region_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return the eight mutually exclusive regions of Norm1 S1/S2/Norm2."""
    s1 = df["strategy1"].astype(bool)
    s2 = df["strategy2"].astype(bool)
    n2 = df["N2_union"].astype(bool)
    masks = {
        "none": ~s1 & ~s2 & ~n2,
        "s1_only": s1 & ~s2 & ~n2,
        "s2_only": ~s1 & s2 & ~n2,
        "n2_only": ~s1 & ~s2 & n2,
        "s1_s2_only": s1 & s2 & ~n2,
        "s1_n2_only": s1 & ~s2 & n2,
        "s2_n2_only": ~s1 & s2 & n2,
        "triple": s1 & s2 & n2,
    }
    return {name: int(mask.sum()) for name, mask in masks.items()}


def _plot_overall_three_set_venn(
    figures: Path, table: pd.DataFrame, variant: AnalysisVariant
) -> None:
    """Draw one all-records three-set overlap summary for S1, S2 and Norm2."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    counts = _three_set_region_counts(table)
    total = int(len(table))
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    circles = (
        ((-0.42, 0.22), 0.78, "#355c7d", "Norm1 S1"),
        ((0.42, 0.22), 0.78, "#99b898", "Norm1 S2"),
        ((0.0, -0.38), 0.78, "#f67280", "Norm2"),
    )
    for center, radius, color, _ in circles:
        ax.add_patch(Circle(center, radius, facecolor=color, edgecolor=color, alpha=0.30, linewidth=2.5))

    # Labels are placed in the seven distinct regions and outside the three
    # circles.  The count is the number of subject-channel records in that
    # mutually exclusive region, not the number of unique channel names.
    labels = {
        "s1_only": (-0.75, 0.38, "S1 only"),
        "s2_only": (0.75, 0.38, "S2 only"),
        "n2_only": (0.0, -0.93, "Norm2 only"),
        "s1_s2_only": (0.0, 0.48, "S1 ∩ S2"),
        "s1_n2_only": (-0.36, -0.29, "S1 ∩ Norm2"),
        "s2_n2_only": (0.36, -0.29, "S2 ∩ Norm2"),
        "triple": (0.0, -0.02, "S1 ∩ S2 ∩ Norm2"),
        "none": (0.0, 1.18, "none"),
    }
    for region, (x, y, label) in labels.items():
        ax.text(
            x,
            y,
            f"{label}\n{counts[region]}",
            ha="center",
            va="center",
            fontsize=10 if region != "triple" else 9,
            fontweight="bold" if region == "triple" else "normal",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.5),
        )

    ax.text(-0.72, 1.03, "Norm1 S1", color="#355c7d", fontsize=12, fontweight="bold")
    ax.text(0.52, 1.03, "Norm1 S2", color="#4f8061", fontsize=12, fontweight="bold")
    ax.text(-0.14, -1.17, "Norm2", color="#c23b50", fontsize=12, fontweight="bold")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Overall overlap: Norm1 S1 × Norm1 S2 × Norm2\n"
        f"{variant.window_label} ms / {variant.signal_label}; N={total} common subject-channel records",
        fontsize=15,
        pad=18,
    )
    ax.text(
        0.5,
        -0.04,
        f"Set totals: S1={int((table.strategy1.astype(bool)).sum())}   "
        f"S2={int((table.strategy2.astype(bool)).sum())}   "
        f"Norm2={int((table.N2_union.astype(bool)).sum())}   "
        f"Union={int((table.strategy1.astype(bool) | table.strategy2.astype(bool) | table.N2_union.astype(bool)).sum())}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        figures / f"norm1s1_norm1s2_norm2_overall_venn_{variant.suffix}.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_set_figures(
    out: Path, table: pd.DataFrame, summary: pd.DataFrame, variant: AnalysisVariant
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    colors = {"N1": "#355c7d", "N2": "#f67280", "CSC": "#6c5b7b"}
    for ax, view in zip(axes, ("x-y", "x-z", "y-z")):
        if table.empty:
            ax.text(0.5, 0.5, "No spatial records", ha="center", va="center")
            continue
        for subject, grp in table.groupby("subject"):
            x = grp.mni_x if view in ("x-y", "x-z") else grp.mni_y
            y = grp.mni_y if view == "x-y" else grp.mni_z if view == "x-z" else grp.mni_z
            ax.scatter(x, y, s=16, alpha=0.22, label=subject)
            csc = grp[grp.CSC]
            if not csc.empty:
                cx = csc.mni_x if view in ("x-y", "x-z") else csc.mni_y
                cy = csc.mni_y if view == "x-y" else csc.mni_z if view == "x-z" else csc.mni_z
                ax.scatter(
                    cx,
                    cy,
                    s=55,
                    c="#6c5b7b",
                    edgecolor="black",
                    linewidth=0.4,
                )
        patches = patch_table()
        for patch, grp in patches.groupby("patch"):
            px = grp.mni_x if view in ("x-y", "x-z") else grp.mni_y
            py = grp.mni_y if view == "x-y" else grp.mni_z if view == "x-z" else grp.mni_z
            ax.scatter(px, py, s=120, marker="*", label=patch)
        ax.axhline(0, color="#aaaaaa", lw=0.5)
        ax.axvline(0, color="#aaaaaa", lw=0.5)
        ax.set_xlabel(view.split("-")[0] + " (MNI mm)")
        ax.set_ylabel(view.split("-")[1] + " (MNI mm)")
        ax.set_title(view.upper())
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle(
        f"{variant.window_label} ms / {variant.signal_label}: common bipolar "
        "centers and bilateral PC/CC/AC patch coordinates",
        fontsize=14,
    )
    figures = out / "stage01_selection" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        figures / f"patch_and_csc_mni_projections_{variant.suffix}.png", dpi=220
    )
    plt.close(fig)

    if not summary.empty:
        plot_cols = ["strategy1", "strategy2", "N2_union", "CSC"]
        ax = summary.set_index("subject")[plot_cols].plot(
            kind="bar", figsize=(12, 5)
        )
        ax.set_ylabel("Number of common centers")
        ax.set_title(
            f"Electrode set counts by subject ({variant.window_label} ms / "
            f"{variant.signal_label})"
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend(title="Set")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(
            figures / f"electrode_set_counts_{variant.suffix}.png", dpi=220
        )
        plt.close(fig)

    # Canonical all-records three-set summary requested by the user.  The
    # existing subject-level pie charts below remain available as historical
    # drill-downs; this Venn-style figure is the aggregate view.
    _plot_overall_three_set_venn(figures, table, variant)

    for subject, grp in table.groupby("subject"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        labels, values = _pie_categories_overall(grp)
        axes[0, 0].pie(
            values,
            labels=[f"{x}\n{v}" for x, v in zip(labels, values)],
            startangle=90,
            colors=["#f8b195", "#6c5b7b", "#355c7d", "#dddddd"],
        )
        axes[0, 0].set_title("Overall N1 vs N2\nCSC = overlap")
        for ax, patch in zip((axes[0, 1], axes[1, 0], axes[1, 1]), ("PC", "CC", "AC")):
            labels, values = _pie_categories_three(grp, patch)
            colors3 = [
                "#f8b195",
                "#355c7d",
                "#99b898",
                "#f67280",
                "#c06c84",
                "#6c5b7b",
                "#2a9d8f",
                "#dddddd",
            ]
            ax.pie(
                values,
                labels=[f"{x}\n{v}" if v else "" for x, v in zip(labels, values)],
                startangle=90,
                colors=colors3,
                textprops={"fontsize": 7},
            )
            ax.set_title(f"{patch}: N2 patch × Strategy 1 × Strategy 2")
        fig.suptitle(
            f"{subject} ({variant.window_label} ms / {variant.signal_label}): "
            "color-selective electrode overlap"
        )
        fig.savefig(
            figures / f"{subject}_cross_overlap_pies_{variant.suffix}.png", dpi=220
        )
        plt.close(fig)


def run_signal_stats(
    out: Path, csc_table: pd.DataFrame, variant: AnalysisVariant
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if csc_table.empty or not csc_table.CSC.any():
        return pd.DataFrame(), pd.DataFrame()
    amp_rows: list[dict[str, object]] = []
    spec_rows: list[dict[str, object]] = []
    subject_spec: list[dict[str, object]] = []
    for subject, grp in csc_table[csc_table.CSC].groupby("subject"):
        channels = list(grp.channel.astype(str))
        t2_raw, time_ms, _ = load_conditions(
            subject, 2, [f"{fruit}_gray" for fruit in GRAY_FRUITS], channels
        )
        t3_raw, _, _ = load_conditions(subject, 3, ["red", "green"], channels)
        t2 = {
            key: prepare_signal(values, time_ms, variant)
            for key, values in t2_raw.items()
        }
        t3 = {
            key: prepare_signal(values, time_ms, variant)
            for key, values in t3_raw.items()
        }
        amp_win = window_mask(time_ms, *variant.window)
        spec_win = window_mask(time_ms, *variant.window)
        base_win = window_mask(time_ms, *BASELINE_MS)
        subject_spec_record: dict[str, object] = {
            "subject": subject,
            "window": variant.window_label,
            "signal": variant.signal,
        }
        for j, channel in enumerate(channels):
            vals = []
            for fruit in GRAY_FRUITS:
                trial_vals = np.nanmean(t2[f"{fruit}_gray"][:, j, amp_win], axis=-1)
                vals.append(trial_vals[np.isfinite(trial_vals)])
                amp_rows.append(
                    {
                        "subject": subject,
                        "channel": channel,
                        "window": variant.window_label,
                        "signal": variant.signal,
                        "analysis": "task2_gray_fruit",
                        "condition": fruit,
                        "mean": float(np.nanmean(trial_vals)),
                        "sem": float(stats.sem(trial_vals, nan_policy="omit")),
                        "n_trials": int(np.isfinite(trial_vals).sum()),
                    }
                )
            if all(len(v) >= 3 for v in vals):
                f, p = stats.f_oneway(*vals)
            else:
                f, p = np.nan, np.nan
            amp_rows.append(
                {
                    "subject": subject,
                    "channel": channel,
                    "window": variant.window_label,
                    "signal": variant.signal,
                    "analysis": "task2_gray_fruit_anova",
                    "condition": "four_fruit",
                    "mean": float(np.nanmean([np.nanmean(v) for v in vals])),
                    "sem": np.nan,
                    "n_trials": int(sum(len(v) for v in vals)),
                    "F": float(f),
                    "p": float(p),
                }
            )
            r = np.nanmean(t3["red"][:, j, amp_win], axis=-1)
            g = np.nanmean(t3["green"][:, j, amp_win], axis=-1)
            r = r[np.isfinite(r)]
            g = g[np.isfinite(g)]
            if len(r) >= 3 and len(g) >= 3:
                p_rg = float(
                    stats.mannwhitneyu(r, g, alternative="two-sided").pvalue
                )
            else:
                p_rg = np.nan
            amp_rows.extend(
                [
                    {
                        "subject": subject,
                        "channel": channel,
                        "window": variant.window_label,
                        "signal": variant.signal,
                        "analysis": "task3_red_green",
                        "condition": "red",
                        "mean": float(np.nanmean(r)),
                        "sem": float(stats.sem(r, nan_policy="omit")),
                        "n_trials": int(len(r)),
                        "p_red_vs_green": p_rg,
                    },
                    {
                        "subject": subject,
                        "channel": channel,
                        "window": variant.window_label,
                        "signal": variant.signal,
                        "analysis": "task3_red_green",
                        "condition": "green",
                        "mean": float(np.nanmean(g)),
                        "sem": float(stats.sem(g, nan_policy="omit")),
                        "n_trials": int(len(g)),
                        "p_red_vs_green": p_rg,
                    },
                ]
            )

            # Band-power stats use the raw HDF5 epochs (1-200 Hz) with Welch
            # log power; the baseline z-score uses the per-condition baseline
            # trial distribution. No per-epoch band-pass is applied.
            fruit_z: dict[str, np.ndarray] = {}
            for fruit in GRAY_FRUITS:
                raw = t2_raw[f"{fruit}_gray"]
                base_power = welch_band_power(raw[:, j, base_win])
                analysis_power = welch_band_power(raw[:, j, spec_win])
                fruit_z[fruit] = band_power_baseline_z(analysis_power, base_power)
            for band_idx, band_name in enumerate(BAND_NAMES):
                values = [
                    fruit_z[f][:, band_idx][np.isfinite(fruit_z[f][:, band_idx])]
                    for f in GRAY_FRUITS
                ]
                if all(len(v) >= 3 for v in values):
                    f_stat, p_stat = stats.f_oneway(*values)
                else:
                    f_stat, p_stat = np.nan, np.nan
                for fruit in GRAY_FRUITS:
                    spec_rows.append(
                        {
                            "subject": subject,
                            "channel": channel,
                            "window": variant.window_label,
                            "signal": variant.signal,
                            "band": band_name,
                            "band_index": band_idx,
                            "condition": fruit,
                            "mean_logpower_z": float(
                                np.nanmean(fruit_z[fruit][:, band_idx])
                            ),
                            "F": float(f_stat) if np.isfinite(f_stat) else np.nan,
                            "p": float(p_stat) if np.isfinite(p_stat) else np.nan,
                        }
                    )
        # Subject-level per-band summary: average over channels and trials.
        for fruit in GRAY_FRUITS:
            raw = t2_raw[f"{fruit}_gray"]
            base_power = welch_band_power(raw[:, :, base_win])
            analysis_power = welch_band_power(raw[:, :, spec_win])
            z = band_power_baseline_z(analysis_power, base_power)
            subject_spec_record.update(
                {
                    f"{band_name}_{fruit}": float(
                        np.nanmean(z[..., b_idx])
                    )
                    for b_idx, band_name in enumerate(BAND_NAMES)
                }
            )
        subject_spec.append(subject_spec_record)

    amp = pd.DataFrame(amp_rows)
    spec = pd.DataFrame(spec_rows)
    subject_spec_df = pd.DataFrame(subject_spec)
    stage = out / "stage02_amplitude_spectral"
    stage.mkdir(parents=True, exist_ok=True)
    amp.to_csv(
        stage / f"csc_amplitude_statistics_{variant.suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    spec.to_csv(
        stage / f"csc_spectral_band_statistics_{variant.suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not spec.empty:
        rows = []
        for band_name in BAND_NAMES:
            arrays = [
                subject_spec_df[f"{band_name}_{fruit}"].to_numpy(dtype=float)
                for fruit in GRAY_FRUITS
            ]
            if len(arrays[0]) >= 3:
                try:
                    p = float(stats.friedmanchisquare(*arrays).pvalue)
                except Exception:
                    p = np.nan
            else:
                p = np.nan
            rows.append(
                {
                    "band": band_name,
                    "window": variant.window_label,
                    "signal": variant.signal,
                    "friedman_p_subject_level": p,
                }
            )
        pd.DataFrame(rows).to_csv(
            stage / f"csc_spectral_group_friedman_{variant.suffix}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return amp, spec


def plot_signal_stats(
    out: Path, amp: pd.DataFrame, spec: pd.DataFrame, variant: AnalysisVariant
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out / "stage02_amplitude_spectral" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if amp.empty:
        return
    rows = amp[(amp.analysis == "task2_gray_fruit") & (amp.channel != "CSC_mean")]
    if not rows.empty:
        agg = (
            rows.groupby("condition")["mean"]
            .agg(["mean", "sem"])
            .reindex(GRAY_FRUITS)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(GRAY_FRUITS))
        ax.bar(
            x,
            agg["mean"],
            yerr=agg["sem"],
            color=["#b7b7b7", "#76a65b", "#e15759", "#76b7b7"],
            capsize=4,
        )
        ax.set_xticks(x, ["cabbage", "kiwi", "strawberry", "watermelon"], rotation=20)
        ax.set_ylabel("Baseline-subtracted amplitude (a.u.)")
        ax.set_title(
            f"CSC Task 2 gray-fruit amplitude ({variant.window_label} ms, "
            f"{variant.signal_label})"
        )
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(
            figures / f"csc_task2_gray_fruit_amplitude_{variant.suffix}.png", dpi=220
        )
        plt.close(fig)
    if not spec.empty:
        summary = (
            spec.groupby(["band", "band_index", "condition"])["mean_logpower_z"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for condition in GRAY_FRUITS:
            s = summary[summary.condition == condition].sort_values("band_index")
            axes[0].plot(s.band_index, s["mean"], marker="o", label=condition)
            axes[0].fill_between(
                s.band_index,
                s["mean"] - s["sem"].fillna(0),
                s["mean"] + s["sem"].fillna(0),
                alpha=0.12,
            )
        axes[0].set_xticks(
            range(len(FEATURE_BANDS)), BAND_NAMES, rotation=60, fontsize=8
        )
        axes[0].set_ylabel("log-power baseline z")
        axes[0].set_title(f"CSC spectral profile ({variant.window_label} ms)")
        axes[0].legend(frameon=False, fontsize=8)
        fig.savefig(
            figures / f"csc_task2_spectral_{variant.window_label}_{variant.signal}.png",
            dpi=220,
        )
        plt.close(fig)
