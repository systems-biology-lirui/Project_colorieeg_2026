"""Exploratory time-resolved decoding on Norm1 S1/S2 electrode sets.

The existing formal H2/H3 runner averages all ROI electrodes before the group
test and can be expensive.  This module answers the user's narrower question:
what do individual S1/S2 electrodes carry over time?  Electrode curves are
descriptive unless a permutation null is explicitly added later.  Group curves
are tested after averaging within subject, with sign-flip cluster permutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common import (
    GRAY_FRUITS,
    SUBJECTS,
    AnalysisVariant,
    common_channels,
    load_conditions,
    read_localization,
)
from analysis.decoding_timeresolved import (
    decode_single_electrode_curve,
    eval_frames,
    group_cluster_permutation,
    stft_band_power,
)
from analysis.hypotheses import _cross_fruit_curve
from analysis.selection import prepare_signal


def _selection_table(root: Path, selection_variant: AnalysisVariant) -> pd.DataFrame:
    path = root / "stage01_selection" / f"functional_selection_{selection_variant.suffix}.csv"
    table = pd.read_csv(path)
    table["strategy1"] = table["strategy1"].map(
        lambda value: str(value).strip().lower() == "true"
    )
    table["strategy2"] = table["strategy2"].map(
        lambda value: str(value).strip().lower() == "true"
    )
    return table


def _sets_for_subject(
    subject: str,
    selection: pd.DataFrame,
    out: Path,
    variant: AnalysisVariant,
    selection_variant: AnalysisVariant,
) -> dict[str, list[str]]:
    sub = selection[selection.subject == subject].copy()
    s1 = set(sub.loc[sub.strategy1, "channel"].astype(str))
    s2 = set(sub.loc[sub.strategy2, "channel"].astype(str))
    common_order = common_channels(subject)
    common = set(common_order)
    loc = read_localization(subject).set_index("channel")
    common_roi = {
        c for c in common if c in loc.index and bool(loc.loc[c, "is_target_roi"])
    }
    csc_path = out / "stage01_selection" / f"electrode_sets_and_csc_{selection_variant.suffix}.csv"
    csc = set()
    if csc_path.exists():
        csc_table = pd.read_csv(csc_path)
        csc_table["CSC_bool"] = csc_table["CSC"].map(
            lambda value: str(value).strip().lower() == "true"
        )
        csc = set(
            csc_table.loc[
                (csc_table.subject == subject) & csc_table.CSC_bool, "channel"
            ].astype(str)
        )
    return {
        "S1": [c for c in common_order if c in s1 and c in common_roi],
        "S2": [c for c in common_order if c in s2 and c in common_roi],
        "S1_or_S2": [c for c in common_order if c in (s1 | s2) and c in common_roi],
        "CSC": [c for c in common_order if c in csc and c in common_roi],
    }


def _prepare_task3(subject: str, channels: list[str], variant: AnalysisVariant):
    raw, time_ms, _ = load_conditions(subject, 3, ["red", "green"], channels)
    raw = {key: prepare_signal(value, time_ms, variant) for key, value in raw.items()}
    red, frame_times = stft_band_power(raw["red"])
    green, _ = stft_band_power(raw["green"])
    grid, frame_idx = eval_frames(frame_times)
    return red, green, grid, frame_idx


def _prepare_task2(subject: str, channels: list[str], variant: AnalysisVariant):
    names = [f"{fruit}_gray" for fruit in GRAY_FRUITS]
    raw, time_ms, _ = load_conditions(subject, 2, names, channels)
    raw = {key: prepare_signal(value, time_ms, variant) for key, value in raw.items()}
    feats = {}
    frame_times = None
    for fruit in GRAY_FRUITS:
        feat, frame_times = stft_band_power(raw[f"{fruit}_gray"])
        feats[fruit] = feat
    grid, frame_idx = eval_frames(frame_times)
    return feats, grid, frame_idx


def run(
    out: Path,
    variant: AnalysisVariant,
    selection_variant: AnalysisVariant | None = None,
    group_perms: int = 5000,
    seed: int = 20260806,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selection_variant is None:
        selection_variant = variant
    selection = _selection_table(out, selection_variant)
    set_names = ("S1", "S2", "S1_or_S2", "CSC")
    electrode_rows: list[dict[str, object]] = []
    seen_electrode_analysis: set[str] = set()
    curves: dict[str, np.ndarray] = {}
    group_rows: list[dict[str, object]] = []
    group_curves: dict[str, np.ndarray] = {}

    for subject_index, subject in enumerate(SUBJECTS):
        sets = _sets_for_subject(subject, selection, out, variant, selection_variant)
        common_order = common_channels(subject)
        union_set = set().union(*(set(sets[name]) for name in set_names))
        union = [channel for channel in common_order if channel in union_set]
        if not union:
            continue
        t3_red, t3_green, grid, frame_idx = _prepare_task3(subject, union, variant)
        t2_feats, grid2, frame_idx2 = _prepare_task2(subject, union, variant)
        if not np.allclose(grid, grid2):
            raise RuntimeError(f"Task time grids differ for {subject}")
        set_curves: dict[tuple[str, str], list[np.ndarray]] = {}
        for set_name in set_names:
            channels = sets[set_name]
            if not channels:
                continue
            subject_curves: dict[str, list[np.ndarray]] = {
                "task3_red_green": [],
                "task2_memory_color": [],
            }
            for channel in channels:
                j = union.index(channel)
                task3_curve, _ = decode_single_electrode_curve(
                    t3_red[:, j],
                    t3_green[:, j],
                    n_perms=0,
                    seed=seed + subject_index * 1000 + j,
                    workers=1,
                    frame_indices=frame_idx,
                )
                task2_curve, _ = _cross_fruit_curve(
                    t2_feats,
                    frame_idx2,
                    j,
                    n_perms=0,
                    seed=seed + subject_index * 1000 + j + 100,
                    workers=1,
                )
                membership = ";".join(name for name in set_names if channel in sets[name])
                for analysis, curve in (
                    ("task3_red_green", task3_curve),
                    ("task2_memory_color", task2_curve),
                ):
                    peak_index = int(np.nanargmax(curve)) if np.isfinite(curve).any() else -1
                    row_key = f"{subject}|{channel}|{analysis}"
                    if row_key not in seen_electrode_analysis:
                        electrode_rows.append(
                            {
                                "subject": subject,
                                "channel": channel,
                                "set_membership": membership,
                                "analysis": analysis,
                                "window": variant.window_label,
                                "signal": variant.signal,
                                "mean_accuracy": float(np.nanmean(curve)),
                                "peak_accuracy": float(curve[peak_index]) if peak_index >= 0 else np.nan,
                                "peak_time_ms": float(grid[peak_index]) if peak_index >= 0 else np.nan,
                                "electrode_p_value": "not_run",
                            }
                        )
                        seen_electrode_analysis.add(row_key)
                    curves[f"{subject}|{channel}|{analysis}"] = curve
                    subject_curves[analysis].append(curve)
            for analysis, values in subject_curves.items():
                if values:
                    set_curves[(set_name, analysis)] = [np.nanmean(values, axis=0)]

        # Save one subject-level curve per set after processing all electrodes.
        for set_name in set_names:
            channels = sets[set_name]
            if not channels:
                continue
            for analysis in ("task3_red_green", "task2_memory_color"):
                channel_curves = [curves[f"{subject}|{ch}|{analysis}"] for ch in channels]
                mean_curve = np.nanmean(channel_curves, axis=0)
                key = f"{set_name}|{analysis}"
                group_curves.setdefault(key, []).append(mean_curve)

    for key, subject_values in group_curves.items():
        set_name, analysis = key.split("|", 1)
        subject_array = np.asarray(subject_values, dtype=float)
        summary, clusters = group_cluster_permutation(
            subject_array, grid, n_perms=group_perms, seed=seed
        )
        if clusters:
            for cluster in clusters:
                group_rows.append(
                    {
                        "set_name": set_name,
                        "analysis": analysis,
                        "window": variant.window_label,
                        "signal": variant.signal,
                        "n_subjects": int(subject_array.shape[0]),
                        "n_electrode_subject_means": int(subject_array.shape[0]),
                        **cluster,
                    }
                )
        else:
            group_rows.append(
                {
                    "set_name": set_name,
                    "analysis": analysis,
                    "window": variant.window_label,
                    "signal": variant.signal,
                    "n_subjects": int(subject_array.shape[0]),
                    "n_electrode_subject_means": int(subject_array.shape[0]),
                    "start_ms": np.nan,
                    "end_ms": np.nan,
                    "mass": np.nan,
                    "p": np.nan,
                }
            )

    stage = out / "stage06_exploration"
    stage.mkdir(parents=True, exist_ok=True)
    electrode_table = pd.DataFrame(electrode_rows)
    group_table = pd.DataFrame(group_rows)
    electrode_table.to_csv(
        stage / f"s1s2_timeresolved_electrode_summary_{variant.suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_table.to_csv(
        stage / f"s1s2_timeresolved_group_clusters_{variant.suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        stage / f"s1s2_timeresolved_electrode_curves_{variant.suffix}.npz",
        times_ms=grid,
        **{key.replace("|", "__"): value for key, value in curves.items()},
    )
    (stage / f"s1s2_timeresolved_parameters_{variant.suffix}.json").write_text(
        json.dumps(
            {
                "variant": variant.suffix,
                "signal": variant.signal,
                "selection_variant": selection_variant.suffix,
                "window_ms": list(variant.window),
                "group_permutations": group_perms,
                "notes": [
                    "The decoding input uses the selected signal variant.",
                    "The electrode membership is held at the prior lf30 selection for direct signal-source comparison when selection_variant is 100-400_lf30.",
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return electrode_table, group_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--signal", choices=("lf30", "raw200"), default="raw200")
    parser.add_argument(
        "--selection-signal",
        choices=("lf30", "raw200"),
        default=None,
        help="Signal variant used only to select S1/S2/CSC; defaults to --signal.",
    )
    parser.add_argument(
        "--window",
        choices=("1-300", "100-400"),
        default="100-400",
        help="Time window used by the S1/S2 selection table.",
    )
    parser.add_argument("--group-perms", type=int, default=5000)
    args = parser.parse_args()
    window = {"1-300": (1.0, 300.0), "100-400": (100.0, 400.0)}[args.window]
    variant = AnalysisVariant(window, args.signal)
    selection_variant = AnalysisVariant(window, args.selection_signal or args.signal)
    electrode, group = run(
        args.out,
        variant,
        selection_variant=selection_variant,
        group_perms=args.group_perms,
    )
    print(f"electrode_rows={len(electrode)} group_rows={len(group)}")
    print(group.to_string(index=False))


if __name__ == "__main__":
    main()
