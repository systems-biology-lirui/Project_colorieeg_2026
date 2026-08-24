#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrator for the color_analyse_0727 v2 analysis pipeline.

Stages:
  selection  - functional color-vs-gray selection (Strategy 1/2)
  csc        - Norm 2 patch proximity, CSC intersection and overlap figures
  stats      - CSC amplitude and 16-band Welch power statistics
  decoding   - spectrum-level single-electrode decoding with 1000 permutations
  luminance  - stimulus luminance/contrast/colorfulness audit
  report     - markdown report, README index, output_index.csv and PPTX

Variants are time window x signal band; by default the standard window
100-400 ms runs for both signals: raw 1-200 Hz (HDF5 epochs) and low-frequency
1-30 Hz.

Example:
  python analysis/run_final_analysis.py --perms 1000 --workers 21
  python analysis/run_final_analysis.py --stages selection csc --windows 0-300
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.common import (
    DEFAULT_WINDOW_LABELS,
    MODULE_ROOT,
    AnalysisVariant,
    SIGNALS,
    all_variants,
)
from analysis.selection import run_functional_selection
from analysis.csc import (
    make_set_figures,
    plot_signal_stats,
    run_signal_stats,
    run_spatial_selection,
)
from analysis.decoding import run_spectrum_decoding
from analysis.hypotheses import (
    run_h1,
    run_h2_h3,
    run_h4,
    write_decision_summary,
)
from analysis.exploration import (
    run_exemplar_repeat_check,
    run_exploration_erp_amplitude,
    run_exploration_mvpa,
)
from analysis.luminance import run_luminance_audit
from analysis.reporting import (
    write_output_index,
    write_parameters,
    write_pptx,
    write_readme,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=MODULE_ROOT / "result" / f"final_analysis_{date.today():%Y%m%d}",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=(
            "selection",
            "csc",
            "stats",
            "decoding",
            "hypotheses",
            "exploration",
            "luminance",
            "report",
        ),
        default=(
            "selection",
            "csc",
            "stats",
            "decoding",
            "hypotheses",
            "exploration",
            "luminance",
            "report",
        ),
        help="Stages to run (default: all).",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        choices=(
            "0-300",
            "1-300",
            "50-350",
            "51-350",
            "100-300",
            "100-400",
            "150-450",
        ),
        default=list(DEFAULT_WINDOW_LABELS),
        help="Analysis time windows.",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        choices=SIGNALS,
        default=list(SIGNALS),
        help="Analysis signals.",
    )
    parser.add_argument("--perms", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=21)
    parser.add_argument(
        "--hypothesis-group-perms",
        type=int,
        default=5000,
        help="Group-level sign-flip permutations for H2/H3.",
    )
    parser.add_argument(
        "--hypothesis-electrode-perms",
        type=int,
        default=1000,
        help="Per-electrode label permutations for H2/H3; 0 disables them.",
    )
    parser.add_argument(
        "--tgm-perms",
        type=int,
        default=5000,
        help="Group-level TGM sign-flip permutations for H4.",
    )
    parser.add_argument("--nperseg", type=int, default=64)
    parser.add_argument("--noverlap", type=int, default=32)
    return parser.parse_args()


def _variants(args: argparse.Namespace) -> list[AnalysisVariant]:
    labels = {
        "0-300": (0.0, 300.0),
        "1-300": (1.0, 300.0),
        "50-350": (50.0, 350.0),
        "51-350": (51.0, 350.0),
        "100-300": (100.0, 300.0),
        "100-400": (100.0, 400.0),
        "150-450": (150.0, 450.0),
    }
    windows = [labels[label] for label in args.windows]
    return [
        AnalysisVariant(window, signal)
        for window in windows
        for signal in args.signals
    ]


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    variants = _variants(args)
    if not variants:
        raise SystemExit("No variant selected; check --windows/--signals")
    for folder in (
        out,
        out / "report",
        out / "stage01_selection",
        out / "stage02_amplitude_spectral",
        out / "stage03_decoding",
        out / "stage04_luminance",
        out / "cache",
    ):
        folder.mkdir(parents=True, exist_ok=True)

    write_parameters(
        out,
        variants=variants,
        perms=args.perms,
        workers=args.workers,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
    )
    print(f"Output: {out}")
    print(f"Variants: {[v.suffix for v in variants]}")

    variant_tables: dict[str, dict[str, Any]] = {
        variant.suffix: {} for variant in variants
    }

    if "selection" in args.stages:
        print("[selection] functional S1/S2 ...")
        functional = run_functional_selection(out, variants)
        for variant in variants:
            subset = functional[
                (functional.window == variant.window_label)
                & (functional.signal == variant.signal)
            ]
            variant_tables[variant.suffix]["functional"] = subset
            print(
                f"    {variant.suffix}: tested={len(subset)} "
                f"S1={int(subset.strategy1.sum()) if not subset.empty else 0} "
                f"S2={int(subset.strategy2.sum()) if not subset.empty else 0}"
            )

    if "csc" in args.stages:
        print("[csc] Norm 2 patch proximity and CSC ...")
        for variant in variants:
            functional = variant_tables[variant.suffix].get("functional")
            spatial, summary = run_spatial_selection(out, functional, variant)
            make_set_figures(out, spatial, summary, variant)
            variant_tables[variant.suffix]["spatial"] = spatial
            variant_tables[variant.suffix]["summary"] = summary
            print(
                f"    {variant.suffix}: spatial={len(spatial)} "
                f"CSC={int(spatial.CSC.sum()) if not spatial.empty else 0}"
            )

    if "stats" in args.stages:
        print("[stats] CSC amplitude and 16-band power statistics ...")
        for variant in variants:
            spatial = variant_tables[variant.suffix].get("spatial")
            if spatial is None:
                raise SystemExit("Run the csc stage before stats")
            amp, spec = run_signal_stats(out, spatial, variant)
            plot_signal_stats(out, amp, spec, variant)
            variant_tables[variant.suffix]["amp"] = amp
            variant_tables[variant.suffix]["spec"] = spec
            print(f"    {variant.suffix}: amp_rows={len(amp)} spec_rows={len(spec)}")

    if "decoding" in args.stages:
        print(f"[decoding] spectrum-level decoding with {args.perms} permutations ...")
        for variant in variants:
            spatial = variant_tables[variant.suffix].get("spatial")
            if spatial is None:
                raise SystemExit("Run the csc stage before decoding")
            decoding, _ = run_spectrum_decoding(
                out, spatial, variant, n_perms=args.perms, workers=args.workers
            )
            variant_tables[variant.suffix]["decoding"] = decoding
            print(f"    {variant.suffix}: decoding_rows={len(decoding)}")

    if "hypotheses" in args.stages:
        print("[hypotheses] H1-H4 pre-registered tests ...")
        h1_tables = []
        for variant in variants:
            h1 = run_h1(out, variant, n_perms=min(1000, args.perms), seed=0)
            h1_tables.append(h1)
            print(
                f"    H1 {variant.suffix}: "
                f"observed_S1={int(h1.observed_s1.sum()) if not h1.empty else 0}"
            )
        h1_combined = (
            pd.concat(h1_tables, ignore_index=True) if h1_tables else pd.DataFrame()
        )
        h2h3 = run_h2_h3(
            out,
            variants,
            n_perms_group=args.hypothesis_group_perms,
            n_perms_electrode=args.hypothesis_electrode_perms,
            workers=args.workers,
            seed=0,
        )
        h4 = run_h4(
            out, variants, n_perms_group=args.tgm_perms, workers=args.workers, seed=0
        )
        write_decision_summary(out, h1_combined, h2h3, h4, variants)
        print("    decision_summary.md written")

    if "exploration" in args.stages:
        print("[exploration] MVPA / ERP-amplitude / exemplar checks ...")
        run_exemplar_repeat_check(out)
        for variant in variants:
            run_exploration_mvpa(out, variant, n_perms_group=5000, seed=0)
            run_exploration_erp_amplitude(out, variant, n_perms_group=5000, seed=0)
            print(f"    {variant.suffix}: MVPA + ERP amplitude done")

    if "luminance" in args.stages:
        print("[luminance] stimulus audit ...")
        luminance = run_luminance_audit(out)
    else:
        luminance = None

    if "report" in args.stages:
        print("[report] markdown, README, index, PPTX ...")
        write_report(
            out,
            variant_tables,
            luminance,
            perms=args.perms,
            workers=args.workers,
        )
        write_readme(out, perms=args.perms, workers=args.workers)
        write_output_index(out)
        try:
            write_pptx(out, variant_tables)
        except Exception as exc:  # PPTX is a convenience artifact
            print(f"    PPTX generation skipped: {exc}")

    print(f"Completed: {out}")


if __name__ == "__main__":
    main()
