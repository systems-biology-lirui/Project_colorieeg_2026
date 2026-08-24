"""Stage 1: functional color-vs-gray selection (Strategy 1 / Strategy 2).

Every analysis variant (time window x signal band) is evaluated with the same
rule: per category, a two-sided trial-wise Mann-Whitney U test on the
baseline-subtracted mean amplitude inside the window. Strategy 1 requires all
four categories (face/object/body/place) to reach raw p<0.05; Strategy 2
requires at least one. No direction consistency and no FDR gate are applied;
FDR-adjusted columns are informational only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.spectral_features import padded_bandpass, window_mask
from analysis.common import (
    BASELINE_MS,
    SUBJECTS,
    TASK1_PAIRS,
    AnalysisVariant,
    all_variants,
    baseline_subtract,
    bh_adjust,
    common_channels,
    load_conditions,
    read_localization,
)


def prepare_signal(
    data: np.ndarray,
    time_ms: np.ndarray,
    variant: AnalysisVariant,
) -> np.ndarray:
    """Apply the variant band-pass (padded, artifact-free) and baseline mean
    subtraction. Padded filtering avoids the per-epoch edge transient that the
    old direct ``sosfiltfilt`` introduced inside the analysis window.
    The ``raw200`` variant uses the HDF5 epochs as-is (already 1-200 Hz
    continuous-filtered) and only applies baseline mean subtraction.
    """
    if variant.signal_band is None:
        filtered = data
    else:
        filtered = padded_bandpass(data, *variant.signal_band)
    return baseline_subtract(filtered, time_ms, *BASELINE_MS)


def run_functional_selection(
    out: Path,
    variants: Iterable[AnalysisVariant],
    subjects: Iterable[str] = SUBJECTS,
) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    tables: list[pd.DataFrame] = []
    for variant in variants:
        rows: list[dict[str, object]] = []
        for subject in subjects:
            loc = read_localization(subject).set_index("channel")
            channels = [
                c
                for c in common_channels(subject)
                if c in loc.index and bool(loc.loc[c, "is_target_roi"])
            ]
            if not channels:
                continue
            condition_names = [name for _, name, _ in TASK1_PAIRS] + [
                name for _, _, name in TASK1_PAIRS
            ]
            data, time_ms, labels = load_conditions(
                subject, 1, condition_names, channels
            )
            data = {
                key: prepare_signal(values, time_ms, variant)
                for key, values in data.items()
            }
            win = window_mask(time_ms, *variant.window)
            for j, channel in enumerate(labels):
                record: dict[str, object] = {
                    "subject": subject,
                    "channel": channel,
                    "roi": loc.loc[channel, "roi"],
                    "mni_x": loc.loc[channel, "mni_x"],
                    "mni_y": loc.loc[channel, "mni_y"],
                    "mni_z": loc.loc[channel, "mni_z"],
                    "color_with_sti": loc.loc[channel, "color_with_sti"],
                    "window": variant.window_label,
                    "signal": variant.signal,
                }
                p_values: list[float] = []
                for category, c_name, g_name in TASK1_PAIRS:
                    c = np.nanmean(data[c_name][:, j, win], axis=-1)
                    g = np.nanmean(data[g_name][:, j, win], axis=-1)
                    c = c[np.isfinite(c)]
                    g = g[np.isfinite(g)]
                    if len(c) >= 3 and len(g) >= 3:
                        p = float(
                            stats.mannwhitneyu(
                                c, g, alternative="two-sided"
                            ).pvalue
                        )
                        p_t = float(
                            stats.ttest_ind(c, g, equal_var=False).pvalue
                        )
                        direction = float(np.nanmean(c) - np.nanmean(g))
                    else:
                        p, p_t, direction = 1.0, 1.0, np.nan
                    p_values.append(p)
                    record[f"{category}_p_raw"] = p
                    record[f"{category}_p_t"] = p_t
                    record[f"{category}_diff_uV"] = direction
                    record[f"{category}_n_color"] = int(len(c))
                    record[f"{category}_n_gray"] = int(len(g))
                fdr = bh_adjust(p_values)
                for category, p_fdr in zip((x[0] for x in TASK1_PAIRS), fdr):
                    record[f"{category}_p_fdr"] = float(p_fdr)
                # Historical Strategy 1 (pre-0727 prompt): merged / pooled
                # color trials across all four categories vs all gray trials,
                # Wilcoxon rank-sum on the window mean (see
                # color_cognition_pipeline/analyse_0617/code/
                # step1_1_select_channel_extended.py). The 0727 prompt
                # redefined Strategy 1 as all four categories individually
                # significant; both definitions are kept in the output.
                color_all = np.concatenate(
                    [
                        np.nanmean(data[c_name][:, j, win], axis=-1)
                        for _, c_name, _ in TASK1_PAIRS
                    ]
                )
                gray_all = np.concatenate(
                    [
                        np.nanmean(data[g_name][:, j, win], axis=-1)
                        for _, _, g_name in TASK1_PAIRS
                    ]
                )
                color_all = color_all[np.isfinite(color_all)]
                gray_all = gray_all[np.isfinite(gray_all)]
                if len(color_all) >= 3 and len(gray_all) >= 3:
                    merged_p = float(
                        stats.mannwhitneyu(
                            color_all, gray_all, alternative="two-sided"
                        ).pvalue
                    )
                    merged_diff = float(
                        np.nanmean(color_all) - np.nanmean(gray_all)
                    )
                else:
                    merged_p, merged_diff = 1.0, np.nan
                record["merged_p_raw"] = merged_p
                record["merged_diff_uV"] = merged_diff
                record["strategy1_merged"] = bool(merged_p < 0.05)
                # ANOVA-based standard (2026-08-05): two-way color x category,
                # type-II color main effect p<0.05. This is the parametric
                # pooled test and is the primary functional-selection rule.
                y = np.concatenate(
                    [
                        np.nanmean(data[c_name][:, j, win], axis=-1)
                        for _, c_name, _ in TASK1_PAIRS
                    ]
                    + [
                        np.nanmean(data[g_name][:, j, win], axis=-1)
                        for _, _, g_name in TASK1_PAIRS
                    ]
                )
                color_label = np.concatenate(
                    [
                        np.zeros_like(
                            np.nanmean(data[c_name][:, j, win], axis=-1)
                        )
                        for _, c_name, _ in TASK1_PAIRS
                    ]
                    + [
                        np.ones_like(
                            np.nanmean(data[g_name][:, j, win], axis=-1)
                        )
                        for _, _, g_name in TASK1_PAIRS
                    ]
                )
                category_label = np.concatenate(
                    [
                        np.full_like(
                            np.nanmean(data[c_name][:, j, win], axis=-1), k
                        )
                        for k, (_, c_name, _) in enumerate(TASK1_PAIRS)
                    ]
                    + [
                        np.full_like(
                            np.nanmean(data[g_name][:, j, win], axis=-1), k
                        )
                        for k, (_, _, g_name) in enumerate(TASK1_PAIRS)
                    ]
                )
                finite = np.isfinite(y)
                anova_df = pd.DataFrame(
                    {
                        "y": y[finite],
                        "color": color_label[finite].astype(int),
                        "cat": category_label[finite].astype(int),
                    }
                )
                try:
                    from statsmodels.formula.api import ols
                    from statsmodels.stats.anova import anova_lm

                    model = ols("y ~ C(color)*C(cat)", data=anova_df).fit()
                    aov = anova_lm(model, typ=2)
                    p_color = float(aov.loc["C(color)", "PR(>F)"])
                    p_interaction = float(
                        aov.loc["C(color):C(cat)", "PR(>F)"]
                    )
                except Exception:
                    p_color, p_interaction = np.nan, np.nan
                record["anova_color_main_p"] = p_color
                record["anova_interaction_p"] = p_interaction
                record["strategy1"] = bool(p_color < 0.05)
                record["n_categories_raw_p_lt_0.05"] = int(
                    sum(p < 0.05 for p in p_values)
                )
                record["n_categories_fdr_p_lt_0.05"] = int(
                    sum(p < 0.05 for p in fdr)
                )
                # Strategy 2: at least one single-category difference is
                # significant with a per-category Welch t-test (ANOVA with
                # two groups is equivalent to the t-test).
                p_t_values = [
                    float(record[f"{category}_p_t"])
                    for category, _, _ in TASK1_PAIRS
                ]
                record["strategy2"] = bool(any(p < 0.05 for p in p_t_values))
                record["strategy2_mwu"] = bool(
                    any(p < 0.05 for p in p_values)
                )
                record["strategy1_fdr"] = bool(all(p < 0.05 for p in fdr))
                record["strategy2_fdr"] = bool(any(p < 0.05 for p in fdr))
                rows.append(record)
        table = pd.DataFrame(rows)
        tables.append(table)
        path = (
            out
            / "stage01_selection"
            / f"functional_selection_{variant.suffix}.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, encoding="utf-8-sig")
    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    combined.to_csv(
        out / "stage01_selection" / "functional_selection_all_variants.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return combined
