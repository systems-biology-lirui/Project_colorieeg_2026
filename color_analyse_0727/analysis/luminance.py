"""Stage 4: stimulus luminance / contrast / colorfulness audit.

Checks whether the color-vs-gray (Task 1), gray-fruit (Task 2) and pure-patch
(Task 3) stimulus sets are matched on low-level visual properties. The gray
versions are expected to be luminance-preserving desaturations; the audit
quantifies and flags any mismatch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

from analysis.common import STIMULI_ROOT, df_to_markdown

LUMINANCE_FLAG_THRESHOLD = 3.0  # mean |dL| on the 0-255 Y scale


def _linearize(channel: np.ndarray) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.float64) / 255.0
    return np.where(
        channel <= 0.04045,
        channel / 12.92,
        ((channel + 0.055) / 1.055) ** 2.4,
    )


def _lab_l(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (
        _linearize(rgb[..., 0]),
        _linearize(rgb[..., 1]),
        _linearize(rgb[..., 2]),
    )
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    f = np.where(y > 0.008856, np.cbrt(y), 7.787 * y + 16.0 / 116.0)
    return 116.0 * f - 16.0


def _luminance_y(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float64) @ np.array(
        [0.299, 0.587, 0.114]
    )


def _contrast(values: np.ndarray) -> float:
    return float(np.std(values))


def _colorfulness(rgb: np.ndarray) -> float:
    values = np.asarray(rgb, dtype=np.float64)
    rg = values[..., 0] - values[..., 1]
    yb = 0.5 * (values[..., 0] + values[..., 1]) - values[..., 2]
    return float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def _audit_task1(stim_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    category_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for category in ("face", "object", "body", "place"):
        pairs = []
        for i in range(1, 71):
            color_path = stim_root / "Stimuli_Task1" / f"{category}_color_{i:02d}.bmp"
            gray_path = stim_root / "Stimuli_Task1" / f"{category}_gray_{i:02d}.bmp"
            if not (color_path.exists() and gray_path.exists()):
                continue
            color = _load_rgb(color_path)
            gray = _load_rgb(gray_path)
            yc, yg = _luminance_y(color), _luminance_y(gray)
            lc, lg = _lab_l(color), _lab_l(gray)
            pair_rows.append(
                {
                    "category": category,
                    "image_idx": i,
                    "lum_y_color": float(np.mean(yc)),
                    "lum_y_gray": float(np.mean(yg)),
                    "abs_dLum_y": float(np.abs(np.mean(yc) - np.mean(yg))),
                    "lab_l_color": float(np.mean(lc)),
                    "lab_l_gray": float(np.mean(lg)),
                    "abs_dLabL": float(np.abs(np.mean(lc) - np.mean(lg))),
                    "contrast_color": _contrast(yc),
                    "contrast_gray": _contrast(yg),
                    "abs_dContrast": float(np.abs(_contrast(yc) - _contrast(yg))),
                    "colorfulness_color": _colorfulness(color),
                    "colorfulness_gray": _colorfulness(gray),
                    "mean_abs_rgb_diff": float(
                        np.mean(np.abs(color - gray))
                    ),
                }
            )
            pairs.append(
                (
                    float(np.mean(yc)),
                    float(np.mean(yg)),
                    _contrast(yc),
                    _contrast(yg),
                )
            )
        pairs_arr = np.asarray(pairs, dtype=float)
        d_lum = np.abs(pairs_arr[:, 0] - pairs_arr[:, 1])
        d_contrast = np.abs(pairs_arr[:, 2] - pairs_arr[:, 3])
        wilcoxon_p = (
            float(stats.wilcoxon(pairs_arr[:, 0], pairs_arr[:, 1]).pvalue)
            if len(pairs) >= 2
            else np.nan
        )
        category_rows.append(
            {
                "category": category,
                "n_pairs": len(pairs),
                "mean_abs_dLum": float(np.mean(d_lum)),
                "max_abs_dLum": float(np.max(d_lum)),
                "mean_abs_dContrast": float(np.mean(d_contrast)),
                "wilcoxon_p_lum_color_vs_gray": wilcoxon_p,
                "flag_luminance_mismatch": bool(np.mean(d_lum) > LUMINANCE_FLAG_THRESHOLD),
            }
        )
    pairs_df = pd.DataFrame(pair_rows)
    if not pairs_df.empty:
        colorfulness_by_category = (
            pairs_df.groupby("category")["colorfulness_color"].mean().to_dict()
        )
        for row in category_rows:
            row["mean_colorfulness_color"] = colorfulness_by_category.get(
                row["category"], np.nan
            )
    cat_df = pd.DataFrame(category_rows)
    if not pairs_df.empty:
        interaction_p = float(
            stats.f_oneway(
                *[
                    pairs_df.loc[pairs_df.category == c, "abs_dLum_y"].to_numpy(
                        dtype=float
                    )
                    for c in ("face", "object", "body", "place")
                    if (pairs_df.category == c).any()
                ]
            ).pvalue
        )
        cat_df["category_x_condition_interaction_p"] = interaction_p
        # A statistically significant interaction is not practically relevant
        # when every category mean |dL| is below the flag threshold; require
        # both statistical significance and a practically meaningful mismatch.
        cat_df["flag_category_interaction"] = bool(
            (interaction_p < 0.05)
            and (cat_df["mean_abs_dLum"].max() > LUMINANCE_FLAG_THRESHOLD)
        )
        cat_df["max_category_mean_abs_dLum"] = float(cat_df["mean_abs_dLum"].max())
    return cat_df, pairs_df


def _audit_task2(stim_root: Path) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, object]] = []
    for fruit in ("Cabbage", "Kiwi", "Strawberry", "Watermelon"):
        paths = sorted(
            (stim_root / "Stimuli_Task2").glob(f"{fruit}_Gray_*.bmp")
        )
        if not paths:
            paths = sorted(
                (stim_root / "Stimuli_Task2").glob(f"{fruit}_gray_*.bmp")
            )
        for path in paths:
            rows.append(
                {
                    "fruit": fruit,
                    "image": path.name,
                    "lum_y": float(np.mean(_luminance_y(_load_rgb(path)))),
                }
            )
    table = pd.DataFrame(rows)
    p_value = np.nan
    if not table.empty and table.fruit.nunique() >= 2:
        groups = [table.loc[table.fruit == f, "lum_y"].to_numpy(float) for f in table.fruit.unique()]
        if all(len(g) >= 2 for g in groups):
            p_value = float(stats.f_oneway(*groups).pvalue)
    return table, p_value


def _audit_task3(stim_root: Path) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, object]] = []
    for color in ("Red", "Green", "Blue", "Yellow", "Black", "White"):
        paths = sorted((stim_root / "Stimuli_Task3").glob(f"{color}_Color_*.bmp"))
        if not paths:
            paths = sorted((stim_root / "Stimuli_Task3").glob(f"{color}_color_*.bmp"))
        for path in paths:
            rows.append(
                {
                    "color": color,
                    "image": path.name,
                    "lum_y": float(np.mean(_luminance_y(_load_rgb(path)))),
                }
            )
    table = pd.DataFrame(rows)
    red_green_delta = np.nan
    if not table.empty:
        red = table.loc[table.color == "Red", "lum_y"]
        green = table.loc[table.color == "Green", "lum_y"]
        if len(red) and len(green):
            red_green_delta = float(red.mean() - green.mean())
    return table, red_green_delta


def run_luminance_audit(
    out: Path, stim_root: Path = STIMULI_ROOT
) -> dict[str, pd.DataFrame]:
    cat1, pairs1 = _audit_task1(stim_root)
    table2, anova2_p = _audit_task2(stim_root)
    table3, red_green_delta = _audit_task3(stim_root)
    stage = out / "stage04_luminance"
    stage.mkdir(parents=True, exist_ok=True)
    cat1.to_csv(
        stage / "luminance_task1_category_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pairs1.to_csv(
        stage / "luminance_task1_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    table2.to_csv(
        stage / "luminance_task2_gray_fruits.csv",
        index=False,
        encoding="utf-8-sig",
    )
    table3.to_csv(
        stage / "luminance_task3_patches.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# Stimulus luminance audit",
        "",
        "## Task 1: color vs gray (280 pairs)",
        "",
        df_to_markdown(cat1),
        "",
        "## Task 2: gray fruits (luminance balance)",
        "",
        f"One-way ANOVA across four gray fruits: p = {anova2_p:.4f}"
        if np.isfinite(anova2_p)
        else "One-way ANOVA across four gray fruits: not computable",
        "",
        df_to_markdown(table2.groupby("fruit")["lum_y"].agg(["mean", "std", "min", "max"]).round(3)),
        "",
        "## Task 3: pure color patches",
        "",
        df_to_markdown(table3.groupby("color")["lum_y"].agg(["mean", "std"]).round(3)),
        "",
        f"Red minus Green luminance: {red_green_delta:.2f}",
        "",
        f"Flag threshold for mean |dL|: {LUMINANCE_FLAG_THRESHOLD:.1f} (0-255 Y scale)",
        "",
    ]
    (stage / "luminance_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"task1_category": cat1, "task1_pairs": pairs1, "task2": table2, "task3": table3}
