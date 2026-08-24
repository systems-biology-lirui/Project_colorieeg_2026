#!/usr/bin/env python3
"""Time-continuous Task 1 electrode selection.

This is a new selection branch. It does not replace the completed 100-400 ms
windowed selection. For every target-ROI common electrode, it performs a
pooled color-versus-gray Welch t-test and a category-balanced color main-effect
test at each time point. The category-balanced test is based on the four
category-wise color-minus-gray contrasts and is the primary continuous
criterion; the pooled test is retained as a sensitivity column. The default
scan interval is 0-800 ms and the default duration is 200 ms.

The pointwise p threshold plus duration rule is intentionally labeled
exploratory. The output preserves pointwise p-values and the exact run
intervals so a later cluster-permutation or preregistered correction can be
added without losing provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = ROOT / "color_analyse_0727"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from analysis.common import (  # noqa: E402
    BASELINE_MS,
    SIGNAL_BANDS,
    SIGNAL_LABELS,
    SUBJECTS,
    TASK1_PAIRS,
    common_channels,
    df_to_markdown,
    load_conditions,
    read_localization,
)
from analysis.selection import prepare_signal  # noqa: E402
from analysis.common import AnalysisVariant  # noqa: E402


HISTORICAL_RESULT = (
    MODULE_ROOT
    / "result"
    / "final_analysis_seeg_20260806_corrected"
    / "stage01_selection"
    / "functional_selection_100-400_lf30.csv"
)


@dataclass(frozen=True)
class ContinuousRun:
    start_ms: float
    end_ms: float
    duration_ms: float
    n_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=MODULE_ROOT / "result" / "continuous_selection_200ms_20260807",
    )
    parser.add_argument("--signal", choices=tuple(SIGNAL_BANDS), default="lf30")
    parser.add_argument("--scan-start", type=float, default=0.0)
    parser.add_argument("--scan-end", type=float, default=800.0)
    parser.add_argument("--min-duration-ms", type=float, default=200.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    return parser.parse_args()


def contiguous_runs(
    significant: np.ndarray,
    time_ms: np.ndarray,
    min_duration_ms: float,
) -> list[ContinuousRun]:
    indices = np.flatnonzero(np.asarray(significant, dtype=bool))
    if indices.size == 0:
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, splits)
    if len(time_ms) > 1:
        dt = float(np.nanmedian(np.diff(time_ms)))
    else:
        dt = 1.0
    runs: list[ContinuousRun] = []
    for group in groups:
        start = float(time_ms[group[0]])
        end = float(time_ms[group[-1]] + dt)
        duration = end - start
        runs.append(
            ContinuousRun(
                start_ms=start,
                end_ms=end,
                duration_ms=duration,
                n_points=int(group.size),
            )
        )
    return sorted(runs, key=lambda item: (-item.duration_ms, item.start_ms))


def load_historical_selection() -> pd.DataFrame:
    if not HISTORICAL_RESULT.exists():
        return pd.DataFrame(columns=["subject", "channel", "strategy1"])
    table = pd.read_csv(HISTORICAL_RESULT)
    keep = [
        column
        for column in (
            "subject",
            "channel",
            "strategy1",
            "anova_color_main_p",
            "window",
            "signal",
        )
        if column in table.columns
    ]
    return table[keep].copy()


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def run_subject(
    subject: str,
    signal: str,
    scan_start: float,
    scan_end: float,
    min_duration_ms: float,
    alpha: float,
    pvalue_dir: Path,
) -> pd.DataFrame:
    localization = read_localization(subject).set_index("channel")
    channels = [
        channel
        for channel in common_channels(subject)
        if channel in localization.index
        and bool(localization.loc[channel, "is_target_roi"])
    ]
    if not channels:
        return pd.DataFrame()

    condition_names = [name for _, name, _ in TASK1_PAIRS] + [
        name for _, _, name in TASK1_PAIRS
    ]
    data, time_ms, labels = load_conditions(subject, 1, condition_names, channels)
    variant = AnalysisVariant((100.0, 400.0), signal)
    data = {
        key: prepare_signal(values, time_ms, variant)
        for key, values in data.items()
    }
    scan_mask = (time_ms >= scan_start) & (time_ms <= scan_end)
    scan_time = time_ms[scan_mask]
    if scan_time.size < 2:
        raise ValueError("The scan interval contains fewer than two time points")

    rows: list[dict[str, object]] = []
    pvalue_dir.mkdir(parents=True, exist_ok=True)
    pvalue_matrix = np.full((len(labels), scan_time.size), np.nan, dtype=np.float32)
    effect_matrix = np.full_like(pvalue_matrix, np.nan)
    category_balanced_pvalue_matrix = np.full(
        (len(labels), scan_time.size), np.nan, dtype=np.float32
    )
    category_balanced_effect_matrix = np.full_like(category_balanced_pvalue_matrix, np.nan)

    for channel_index, channel in enumerate(labels):
        color_trials = np.concatenate(
            [data[color_name][:, channel_index, scan_mask] for _, color_name, _ in TASK1_PAIRS],
            axis=0,
        )
        gray_trials = np.concatenate(
            [data[gray_name][:, channel_index, scan_mask] for _, _, gray_name in TASK1_PAIRS],
            axis=0,
        )
        with np.errstate(all="ignore"):
            test = stats.ttest_ind(
                color_trials,
                gray_trials,
                axis=0,
                equal_var=False,
                nan_policy="omit",
            )
        p_values = np.asarray(test.pvalue, dtype=float)
        effect = np.nanmean(color_trials, axis=0) - np.nanmean(gray_trials, axis=0)
        pvalue_matrix[channel_index] = p_values.astype(np.float32)
        effect_matrix[channel_index] = effect.astype(np.float32)
        significant = np.isfinite(p_values) & (p_values < alpha)
        all_runs = contiguous_runs(significant, scan_time, min_duration_ms)
        qualified = [run for run in all_runs if run.duration_ms >= min_duration_ms]
        best = qualified[0] if qualified else None
        category_differences = np.stack(
            [
                np.nanmean(data[color_name][:, channel_index, scan_mask], axis=0)
                - np.nanmean(data[gray_name][:, channel_index, scan_mask], axis=0)
                for _, color_name, gray_name in TASK1_PAIRS
            ],
            axis=0,
        )
        with np.errstate(all="ignore"):
            category_balanced_test = stats.ttest_1samp(
                category_differences,
                popmean=0.0,
                axis=0,
                nan_policy="omit",
            )
        category_balanced_p = np.asarray(category_balanced_test.pvalue, dtype=float)
        category_balanced_effect = np.nanmean(category_differences, axis=0)
        category_balanced_pvalue_matrix[channel_index] = category_balanced_p.astype(
            np.float32
        )
        category_balanced_effect_matrix[channel_index] = category_balanced_effect.astype(
            np.float32
        )
        balanced_mask = np.isfinite(category_balanced_p) & (category_balanced_p < alpha)
        balanced_runs = contiguous_runs(balanced_mask, scan_time, min_duration_ms)
        balanced_best = balanced_runs[0] if balanced_runs else None
        pooled_qualified = best is not None
        category_balanced_qualified = balanced_best is not None
        record: dict[str, object] = {
            "subject": subject,
            "channel": channel,
            "roi": localization.loc[channel, "roi"],
            "mni_x": localization.loc[channel, "mni_x"],
            "mni_y": localization.loc[channel, "mni_y"],
            "mni_z": localization.loc[channel, "mni_z"],
            "signal": signal,
            "signal_label": SIGNAL_LABELS[signal],
            "baseline_start_ms": BASELINE_MS[0],
            "baseline_end_ms": BASELINE_MS[1],
            "scan_start_ms": scan_start,
            "scan_end_ms": scan_end,
            "alpha": alpha,
            "min_duration_ms": min_duration_ms,
            "n_color_trials": int(color_trials.shape[0]),
            "n_gray_trials": int(gray_trials.shape[0]),
            "max_contiguous_duration_ms": float(all_runs[0].duration_ms) if all_runs else 0.0,
            "continuous_pooled": pooled_qualified,
            "continuous_category_balanced": category_balanced_qualified,
            "continuous_candidate": category_balanced_qualified,
            "significant_start_ms": best.start_ms if best else np.nan,
            "significant_end_ms": best.end_ms if best else np.nan,
            "significant_duration_ms": best.duration_ms if best else 0.0,
            "significant_n_points": best.n_points if best else 0,
            "category_balanced_significant_start_ms": balanced_best.start_ms if balanced_best else np.nan,
            "category_balanced_significant_end_ms": balanced_best.end_ms if balanced_best else np.nan,
            "category_balanced_significant_duration_ms": balanced_best.duration_ms if balanced_best else 0.0,
            "category_balanced_significant_n_points": balanced_best.n_points if balanced_best else 0,
            "peak_abs_effect_uV": float(np.nanmax(np.abs(effect))) if np.isfinite(effect).any() else np.nan,
            "mean_effect_uV_scan": _finite_mean(effect),
        }
        rows.append(record)

    np.savez_compressed(
        pvalue_dir / f"{subject}_{signal}.npz",
        channels=np.asarray(labels),
        time_ms=scan_time,
        p_values_pooled=pvalue_matrix,
        effect_uV_pooled=effect_matrix,
        p_values_category_balanced=category_balanced_pvalue_matrix,
        effect_uV_category_balanced=category_balanced_effect_matrix,
    )
    return pd.DataFrame(rows)


def add_historical_comparison(table: pd.DataFrame) -> pd.DataFrame:
    historical = load_historical_selection()
    if historical.empty:
        table["historical_100_400_strategy1"] = False
        table["historical_100_400_anova_p"] = np.nan
    else:
        historical = historical.rename(
            columns={
                "strategy1": "historical_100_400_strategy1",
                "anova_color_main_p": "historical_100_400_anova_p",
            }
        )
        historical["historical_100_400_strategy1"] = historical[
            "historical_100_400_strategy1"
        ].fillna(False).astype(bool)
        table = table.merge(
            historical[
                [
                    "subject",
                    "channel",
                    "historical_100_400_strategy1",
                    "historical_100_400_anova_p",
                ]
            ],
            on=["subject", "channel"],
            how="left",
        )
        table["historical_100_400_strategy1"] = table[
            "historical_100_400_strategy1"
        ].fillna(False).astype(bool)
    table["continuous_or_historical"] = table[
        "continuous_candidate"
    ] | table["historical_100_400_strategy1"]
    table["continuous_new_only"] = table["continuous_candidate"] & ~table[
        "historical_100_400_strategy1"
    ]
    return table


def make_summary(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    return (
        table.groupby("subject", as_index=False)
        .agg(
            common_roi_electrodes=("channel", "size"),
            historical_100_400=("historical_100_400_strategy1", "sum"),
            continuous_pooled=("continuous_pooled", "sum"),
            continuous_category_balanced=("continuous_category_balanced", "sum"),
            continuous_candidate=("continuous_candidate", "sum"),
            continuous_new_only=("continuous_new_only", "sum"),
            union=("continuous_or_historical", "sum"),
        )
    )


def make_figures(out: Path, table: pd.DataFrame, summary: pd.DataFrame, signal: str) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        ax = summary.set_index("subject")[[
            "historical_100_400",
            "continuous_pooled",
            "continuous_category_balanced",
            "union",
        ]].plot(
            kind="bar",
            figsize=(12, 5),
            color=["#9e9e9e", "#2c7fb8", "#41ab5d", "#756bb1"],
        )
        ax.set_title(
            f"Electrode selection comparison ({signal}; 100–400 ms scan, no merged mean)"
        )
        ax.set_xlabel("subject")
        ax.set_ylabel("number of target-ROI common electrodes")
        ax.legend(
            [
                "historical 100–400 ms",
                "pooled continuous (sensitivity)",
                "category-balanced continuous",
                "historical ∪ broad continuous",
            ]
        )
        ax.figure.tight_layout()
        ax.figure.savefig(figures / f"continuous_selection_summary_{signal}.png", dpi=180)
        plt.close(ax.figure)

    if table.empty:
        return
    plot = table[np.isfinite(table["mni_x"]) & np.isfinite(table["mni_y"])].copy()
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    historical = plot[plot["historical_100_400_strategy1"]]
    candidate = plot[plot["continuous_candidate"] & ~plot["historical_100_400_strategy1"]]
    both = plot[plot["continuous_candidate"] & plot["historical_100_400_strategy1"]]
    ax.scatter(historical["mni_x"], historical["mni_y"], s=35, c="#9e9e9e", label="historical 100–400 ms")
    ax.scatter(candidate["mni_x"], candidate["mni_y"], s=55, c="#d95f0e", label="continuous-only")
    ax.scatter(both["mni_x"], both["mni_y"], s=55, c="#2c7fb8", label="both")
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.axvline(0, color="#cccccc", linewidth=0.8)
    ax.set_title(f"MNI location of continuous-selection electrodes ({signal})")
    ax.set_xlabel("MNI x (mm)")
    ax.set_ylabel("MNI y (mm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / f"continuous_selection_mni_{signal}.png", dpi=180)
    plt.close(fig)


def write_readme(out: Path, args: argparse.Namespace, summary: pd.DataFrame) -> None:
    lines = [
        f"# Continuous {args.min_duration_ms:g} ms electrode selection",
        "",
        "This is a new candidate electrode-selection analysis. It does not overwrite or rerun the completed 100–400 ms selection.",
        "",
        "## Rule",
        "",
        f"- Signal: `{args.signal}` ({SIGNAL_LABELS[args.signal]})",
        f"- Baseline: `{BASELINE_MS[0]} to {BASELINE_MS[1]} ms`",
        f"- Scan interval: `{args.scan_start} to {args.scan_end} ms`",
        f"- Pointwise tests: pooled color-versus-gray Welch t-test and category-balanced one-sample t-test of four category contrasts",
        f"- Pointwise threshold: `p < {args.alpha}`",
        f"- Continuous criterion: at least `{args.min_duration_ms} ms` without a non-significant time point",
        "- `continuous_pooled`: pooled test satisfies the duration rule.",
        "- `continuous_category_balanced`: the four category-wise color-minus-gray contrasts jointly satisfy the duration rule; this is the primary continuous criterion.",
        "- `continuous_candidate`: currently equals the primary category-balanced criterion; the pooled criterion is sensitivity-only.",
        "- The historical 100–400 ms ANOVA selection is joined only for comparison.",
        "",
        "## Interpretation",
        "",
        "The duration rule reduces sensitivity to isolated pointwise fluctuations but is still based on uncorrected pointwise p-values. Treat the candidate set as exploratory until a cluster-permutation or preregistered multiple-comparison procedure is selected.",
        "",
        "## Summary",
        "",
        df_to_markdown(summary) if not summary.empty else "_No rows were generated._",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"continuous selection: signal={args.signal}, scan={args.scan_start}-{args.scan_end} ms")
    frames: list[pd.DataFrame] = []
    for subject in args.subjects:
        print(f"  {subject}", flush=True)
        frame = run_subject(
            subject=subject,
            signal=args.signal,
            scan_start=args.scan_start,
            scan_end=args.scan_end,
            min_duration_ms=args.min_duration_ms,
            alpha=args.alpha,
            pvalue_dir=out / "p_values",
        )
        if not frame.empty:
            frames.append(frame)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    table = add_historical_comparison(table) if not table.empty else table
    summary = make_summary(table)
    stage = out / "stage01_continuous_selection"
    stage.mkdir(parents=True, exist_ok=True)
    table.to_csv(stage / f"continuous_selection_{args.signal}.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(stage / f"continuous_selection_summary_{args.signal}.csv", index=False, encoding="utf-8-sig")
    make_figures(out, table, summary, args.signal)
    write_readme(out, args, summary)
    parameters = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal": args.signal,
        "signal_label": SIGNAL_LABELS[args.signal],
        "scan_start_ms": args.scan_start,
        "scan_end_ms": args.scan_end,
        "min_duration_ms": args.min_duration_ms,
        "alpha": args.alpha,
        "subjects": list(args.subjects),
        "baseline_ms": list(BASELINE_MS),
        "pointwise_tests": [
            "pooled color-versus-gray Welch t-test at each time point",
            "category-balanced one-sample t-test across four category color-minus-gray contrasts",
        ],
        "historical_comparison": str(HISTORICAL_RESULT),
        "outputs": {
            "electrode_table": str(stage / f"continuous_selection_{args.signal}.csv"),
            "summary_table": str(stage / f"continuous_selection_summary_{args.signal}.csv"),
            "p_values": str(out / "p_values"),
        },
    }
    (out / "analysis_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(table)} electrode rows to {out}")


if __name__ == "__main__":
    main()
