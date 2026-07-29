"""Reproduce analyse_0617 select-channel strategies for test004/005/006.

The statistical rules intentionally match step1_1_select_channel_extended.py:
rank-sum tests, 100--400 ms mean, or >=50 ms continuous point-wise effect,
using merged categories and category-specific tests.  Unlike the legacy script,
this adapter reads the restarted clean NPZ epochs and keeps test004's
functionally selected channels even though it has no localization table.
"""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ranksums

ROOT = Path("/home/lirui/liulab_project/ieeg/Project_colorieeg_2026")
PIPE = ROOT / "color_cognition_pipeline" / "analyse_0720"
import sys
sys.path.insert(0, str(PIPE))
import config
from utils.epochs import load_epochs
from utils.wholebrain import location_table
from utils.wholebrain import preprocess_subject

SUBJECTS = ("test004", "test005", "test006")
COND_PAIRS = (("11", "12"), ("21", "22"), ("31", "32"), ("41", "42"))
OUT = ROOT / "color_cognition_pipeline" / "analyse_0617" / "result" / "select_channel_batch_004_006"
DOC = ROOT / "color_cognition_pipeline" / "analyse_0617" / "doc"


def roi_category(label):
    s = str(label).lower()
    if any(k in s for k in ("calcarine", "occipital_inf", "occipital_mid", "lingual")):
        return "枕叶"
    if any(k in s for k in ("fusiform", "temporal_inf")):
        return "颞叶后/下部"
    if any(k in s for k in ("temporal_mid", "temporal_pole")):
        return "颞叶前/上部"
    return None


def parse_channel(name):
    m = re.match(r"^([A-Za-z]+)(\d+)$", str(name).strip())
    return (m.group(1).upper(), int(m.group(2))) if m else (None, None)


def neighbors(target, channels):
    p, n = parse_channel(target)
    return [c for c in channels if c != target and parse_channel(c)[0] == p and abs(parse_channel(c)[1] - n) == 1]


def continuous(mask, n):
    run = 0
    for value in mask:
        run = run + 1 if value else 0
        if run >= n:
            return True
    return False


def strategies(c_merged, g_merged, c_single, g_single, times):
    dt = float(np.median(np.diff(times)))
    n50 = int(np.ceil(50.0 / dt))
    w100 = (times >= 100) & (times <= 400)
    w50 = (times >= 50) & (times <= 400)
    p1 = ranksums(c_merged[:, w100].mean(1), g_merged[:, w100].mean(1)).pvalue < .05
    pm = [ranksums(c_merged[:, i], g_merged[:, i]).pvalue < .05 for i in np.flatnonzero(w50)]
    p2 = continuous(pm, n50)
    p3 = any(ranksums(c[:, w100].mean(1), g[:, w100].mean(1)).pvalue < .05 for c, g in zip(c_single, g_single))
    p4 = any(continuous([ranksums(c[:, i], g[:, i]).pvalue < .05 for i in np.flatnonzero(w50)], n50)
              for c, g in zip(c_single, g_single))
    return bool(p1), bool(p2), bool(p3), bool(p4)


def clean_trials(x):
    return x[~np.isnan(x).any(axis=1)]


def screen_subject(subject, ensure_preprocessed=False):
    if ensure_preprocessed:
        preprocess_subject(subject, include_hg=True)
    loc = location_table(subject)
    loc_map = loc.set_index("channel").to_dict("index") if not loc.empty else {}
    result_subject = OUT / "subjects" / subject
    (result_subject / "figures").mkdir(parents=True, exist_ok=True)
    modality_rows = []
    strategy_maps = {}
    for modality in ("erp", "hg"):
        ep = load_epochs(config.ALL_INTERMEDIATE_ROOT / subject / "preprocessing" / f"task1_{modality}_clean.npz", False)
        triggers = np.char.replace(ep["triggers"].astype(str), "Trigger-In:", "")
        channels = [str(x) for x in ep["channel_names"]]
        maps = {}
        for ci, ch in enumerate(channels):
            c = [clean_trials(ep["data"][triggers == cc, ci, :]) for cc, _ in COND_PAIRS]
            g = [clean_trials(ep["data"][triggers == gg, ci, :]) for _, gg in COND_PAIRS]
            cm = np.concatenate(c, axis=0); gm = np.concatenate(g, axis=0)
            maps[ch] = strategies(cm, gm, c, g, ep["times_ms"])
            row = {"subject": subject, "channel": ch, "modality": modality,
                   "strategy_1": maps[ch][0], "strategy_2": maps[ch][1],
                   "strategy_3": maps[ch][2], "strategy_4": maps[ch][3],
                   "strategies_matched": ",".join(str(i + 1) for i, v in enumerate(maps[ch]) if v),
                   "any_strategy": any(maps[ch])}
            row.update(loc_map.get(ch, {"mni_x": np.nan, "mni_y": np.nan, "mni_z": np.nan,
                                        "hemisphere": np.nan, "atlas_region": np.nan}))
            modality_rows.append(row)
        strategy_maps[modality] = (maps, channels)
    stats = pd.DataFrame(modality_rows)
    stats.to_csv(result_subject / "channel_strategy_statistics.csv", index=False)
    selected = []
    for ch in sorted(stats.channel.unique()):
        ss = stats[stats.channel == ch]
        main = ss.any_strategy.any()
        row = ss.iloc[0].to_dict()
        ev = ";".join(m + ":" + str(ss.loc[ss.modality == m, "strategies_matched"].iloc[0])
                       for m in ("erp", "hg") if ss.loc[ss.modality == m, "any_strategy"].iloc[0])
        row.update({"selection": "main_functional", "evidence": ev,
                    "has_localization": bool(np.isfinite(row["mni_x"]) and np.isfinite(row["mni_y"]) and np.isfinite(row["mni_z"])),
                    "roi_category": roi_category(row.get("atlas_region", ""))})
        if main:
            selected.append(row)
    selected = pd.DataFrame(selected)
    selected.to_csv(result_subject / "select_channel_summary.csv", index=False)
    # Exact legacy main-set equivalent: anatomical ROI must match one of the
    # three hard-coded categories. test004 therefore has no legacy ROI set.
    selected[selected.roi_category.notna()].to_csv(result_subject / "legacy_select_channel_summary.csv", index=False)
    # Legacy-style physical-neighbor extension: only unknown/unlabeled neighbors.
    extended = []
    for _, main in selected.iterrows():
        for neigh in neighbors(main.channel, sorted(stats.channel.unique())):
            if neigh in selected.channel.values or any(x["channel"] == neigh for x in extended):
                continue
            s = stats[stats.channel == neigh]
            atlas = str(s.iloc[0].get("atlas_region", ""))
            unknown = (not atlas) or atlas.lower() in ("nan", "unknown", "n/a") or "parahippocamp" in atlas.lower()
            if unknown and s.any_strategy.any():
                r = s.iloc[0].to_dict(); r.update({"selection": "neighbor_functional", "neighbor_of": main.channel})
                extended.append(r)
    pd.DataFrame(extended).to_csv(result_subject / "more_select_channel_summary.csv", index=False)
    return selected, pd.DataFrame(extended)


def run_batch():
    OUT.mkdir(parents=True, exist_ok=True); DOC.mkdir(parents=True, exist_ok=True)
    selected, extended = [], []
    for subject in SUBJECTS:
        # Task1 clean ERP/HG caches are prepared by the all-channel pipeline.
        # Do not regenerate unrelated Task2/Task3 features here.
        s, e = screen_subject(subject, ensure_preprocessed=False)
        selected.append(s.assign(subject=subject)); extended.append(e.assign(subject=subject))
    all_selected = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    all_extended = pd.concat(extended, ignore_index=True) if extended else pd.DataFrame()
    all_selected.to_csv(OUT / "multi_subject_select_channel_summary.csv", index=False)
    all_selected[all_selected.roi_category.notna()].to_csv(OUT / "multi_subject_legacy_select_channel_summary.csv", index=False)
    all_extended.to_csv(OUT / "multi_subject_more_select_channel_summary.csv", index=False)
    pd.DataFrame({"subject": list(SUBJECTS), "main_functional_n": [int((all_selected.subject == s).sum()) for s in SUBJECTS],
                  "legacy_roi_main_n": [int(((all_selected.subject == s) & all_selected.roi_category.notna()).sum()) for s in SUBJECTS],
                  "neighbor_functional_n": [int((all_extended.subject == s).sum()) for s in SUBJECTS]}).to_csv(OUT / "multi_subject_counts.csv", index=False)
    return all_selected, all_extended


if __name__ == "__main__":
    run_batch()
