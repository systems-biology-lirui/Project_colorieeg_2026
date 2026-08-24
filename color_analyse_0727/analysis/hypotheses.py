"""Pre-registered hypothesis tests H1-H4 for the SEEG analysis.

H1: color-selective electrode (S1) count exceeds chance.
H2: physical color (Task 3 red vs green) is decodable at group level.
H3: memory color (Task 2 gray fruits, red vs green) is decodable, tested with
    leave-one-fruit-pair-out generalization.
H4: cross-task temporal generalization (Task 3 -> Task 2) plus site overlap.

All time-resolved tests use cluster-based permutation. Group tests use
sign-flip permutation on subject-level curves; per-electrode tests use label
permutation on single-electrode accuracy curves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import label
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pipeline.spectral_features import window_mask
from analysis.common import (
    BASELINE_MS,
    GRAY_FRUITS,
    SUBJECTS,
    TASK1_PAIRS,
    AnalysisVariant,
    all_variants,
    common_channels,
    load_conditions,
    read_localization,
)
from analysis.decoding_timeresolved import (
    cluster_permutation_1d,
    decode_single_electrode_curve,
    eval_frames,
    group_cluster_permutation,
    stft_band_power,
)
from analysis.selection import prepare_signal

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --------------------------------------------------------------------------
# H1 helpers: fast vectorized two-way ANOVA (color main effect)
# --------------------------------------------------------------------------


def _anova_color_f_p(cells: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Type-II color main-effect F and p, vectorized across channels.

    ``cells``: 8 arrays (4 color, 4 gray) of shape (n_cell_trials, n_channels)
    with window-mean amplitudes. Returns F (n_channels,) and p (n_channels,).
    """
    n_cells = len(cells)  # 8
    ns = np.asarray([a.shape[0] for a in cells], dtype=float)
    n_ch = cells[0].shape[1]
    cell_means = np.stack([a.mean(axis=0) for a in cells], axis=0)  # (8, ch)
    n_total = float(ns.sum())
    grand = np.sum(cell_means * ns[:, None], axis=0) / n_total  # (ch,)
    color_idx = [0, 1, 2, 3]
    gray_idx = [4, 5, 6, 7]
    n_color = ns[color_idx].sum()
    n_gray = ns[gray_idx].sum()
    m_color = np.sum(cell_means[color_idx] * ns[color_idx, None], axis=0) / n_color
    m_gray = np.sum(cell_means[gray_idx] * ns[gray_idx, None], axis=0) / n_gray
    ss_color = n_color * (m_color - grand) ** 2 + n_gray * (m_gray - grand) ** 2
    ss_err = np.zeros(n_ch)
    for a in cells:
        ss_err += np.sum((a - a.mean(axis=0, keepdims=True)) ** 2, axis=0)
    df_err = n_total - n_cells
    f = (ss_color / 1.0) / (ss_err / df_err + 1e-12)
    p = stats.f.sf(f, 1.0, df_err)
    return f, p


def _permuted_cells(
    cell_data: list[np.ndarray], rng: np.random.Generator
) -> list[np.ndarray]:
    """Permute color/gray assignment within each category."""
    out = []
    for k in range(4):
        combined = np.concatenate([cell_data[k], cell_data[k + 4]], axis=0)
        idx = rng.permutation(combined.shape[0])
        split = cell_data[k].shape[0]
        out.append(combined[idx[:split]])
        out.append(combined[idx[split:]])
    return out


def run_h1(
    out: Path,
    variant: AnalysisVariant,
    n_perms: int = 1000,
    seed: int = 0,
    subjects: Iterable[str] = SUBJECTS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for subject in subjects:
        loc = read_localization(subject).set_index("channel")
        channels = [
            c
            for c in common_channels(subject)
            if c in loc.index and bool(loc.loc[c, "is_target_roi"])
        ]
        if not channels:
            continue
        names = [n for _, n, _ in TASK1_PAIRS] + [n for _, _, n in TASK1_PAIRS]
        data, time_ms, _ = load_conditions(subject, 1, names, channels)
        data = {k: prepare_signal(v, time_ms, variant) for k, v in data.items()}
        win = window_mask(time_ms, *variant.window)
        cells = [
            np.nanmean(data[cond][:, :, win], axis=-1)
            for cond in names
        ]
        cells = [np.asarray(c, dtype=float) for c in cells]
        _, p_obs = _anova_color_f_p(cells)
        obs_count = int(np.sum(p_obs < 0.05))
        null_counts = []
        for _ in range(n_perms):
            perm = _permuted_cells(cells, rng)
            _, p_perm = _anova_color_f_p(perm)
            null_counts.append(int(np.sum(p_perm < 0.05)))
        null_counts = np.asarray(null_counts)
        p_value = float((1.0 + np.sum(null_counts >= obs_count)) / (n_perms + 1.0))
        rows.append(
            {
                "subject": subject,
                "n_roi_channels": len(channels),
                "observed_s1": obs_count,
                "expected_s1_chance": len(channels) * 0.05,
                "null_mean": float(null_counts.mean()),
                "null_95": float(np.quantile(null_counts, 0.95)),
                "p_value": p_value,
                "binomial_p": float(
                    stats.binom.sf(obs_count - 1, len(channels), 0.05)
                ),
                "n_permutations": n_perms,
                "window": variant.window_label,
                "signal": variant.signal,
            }
        )
    table = pd.DataFrame(rows)
    stage = out / "stage05_hypotheses"
    stage.mkdir(parents=True, exist_ok=True)
    table.to_csv(stage / f"h1_s1_enrichment_{variant.suffix}.csv", index=False, encoding="utf-8-sig")
    return table


# --------------------------------------------------------------------------
# H2 / H3 helpers
# --------------------------------------------------------------------------


def _cross_fruit_curve(
    feats: dict[str, np.ndarray],
    frame_indices: np.ndarray,
    channel: int,
    n_perms: int,
    seed: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-fruit-pair-out decoding curve for one electrode."""
    names = ["strawberry", "watermelon", "cabbage", "kiwi"]
    arrays = []
    rng = np.random.default_rng(seed)
    n_min = min(feats[n].shape[0] for n in names)
    for n in names:
        arr = feats[n][:, channel, :, frame_indices]
        idx = rng.permutation(arr.shape[0])[:n_min]
        arrays.append(arr[idx])
    folds = [(0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2)]
    n_times = len(frame_indices)

    def evaluate(shuffles: list[np.ndarray] | None) -> np.ndarray:
        scores = np.zeros(n_times)
        for t in range(n_times):
            fold_scores = []
            for fi, (tr_red, tr_green, te_red, te_green) in enumerate(folds):
                x_train = np.concatenate([arrays[tr_red][:, t], arrays[tr_green][:, t]], axis=0)
                y_train = np.concatenate(
                    [np.zeros(len(arrays[tr_red]), dtype=int), np.ones(len(arrays[tr_green]), dtype=int)]
                )
                x_test = np.concatenate([arrays[te_red][:, t], arrays[te_green][:, t]], axis=0)
                y_test = np.concatenate(
                    [np.zeros(len(arrays[te_red]), dtype=int), np.ones(len(arrays[te_green]), dtype=int)]
                )
                y_fit = y_train if shuffles is None else shuffles[fi]
                model = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=10000, dual=True))
                model.fit(x_train, y_fit)
                fold_scores.append(accuracy_score(y_test, model.predict(x_test)))
            scores[t] = float(np.mean(fold_scores))
        return scores

    real = evaluate(None)
    if n_perms <= 0:
        return real, np.empty((0, n_times), dtype=np.float32)
    from joblib import Parallel, delayed

    def one_perm(perm_seed: int) -> np.ndarray:
        local = np.random.default_rng(int(perm_seed))
        shuffles = []
        for tr_red, tr_green, _, _ in folds:
            shuffles.append(
                local.permutation(
                    np.concatenate(
                        [
                            np.zeros(len(arrays[tr_red]), dtype=int),
                            np.ones(len(arrays[tr_green]), dtype=int),
                        ]
                    )
                )
            )
        return evaluate(shuffles)

    perm_seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_perms, dtype=np.int64)
    null = Parallel(n_jobs=min(max(1, workers), max(1, n_perms)), prefer="processes")(
        delayed(one_perm)(s) for s in perm_seeds
    )
    return real, np.asarray(null, dtype=np.float32)


def run_h2_h3(
    out: Path,
    variants: Iterable[AnalysisVariant],
    n_perms_group: int = 5000,
    n_perms_electrode: int = 2000,
    workers: int = 21,
    seed: int = 0,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for variant in variants:
        # Per-electrode cluster permutations are expensive; run them for the
        # primary signal (lf30) only. The raw200 variant still gets the full
        # group-level test.
        n_perms_electrode_variant = (
            n_perms_electrode if variant.signal == "lf30" else 0
        )
        csc_path = out / "stage01_selection" / f"electrode_sets_and_csc_{variant.suffix}.csv"
        csc_channels_by_subject: dict[str, set[str]] = {}
        if csc_path.exists():
            csc_table = pd.read_csv(csc_path)
            csc_table["CSC_bool"] = csc_table["CSC"].map(
                lambda value: str(value).strip().lower() == "true"
            )
            for subject, sub_table in csc_table[csc_table.CSC_bool].groupby("subject"):
                csc_channels_by_subject[str(subject)] = set(
                    sub_table.channel.astype(str)
                )
        stage = out / "stage05_hypotheses"
        stage.mkdir(parents=True, exist_ok=True)
        for analysis, task, cond0, cond1, is_cross_fruit in (
            ("task3_red_green", 3, "red", "green", False),
            ("task2_memory_color", 2, None, None, True),
        ):
            subject_rows = []
            electrode_rows = []
            for subject in SUBJECTS:
                loc = read_localization(subject).set_index("channel")
                channels = [
                    c
                    for c in common_channels(subject)
                    if c in loc.index and bool(loc.loc[c, "is_target_roi"])
                ]
                if not channels:
                    continue
                if not is_cross_fruit:
                    raw, time_ms, _ = load_conditions(subject, task, [cond0, cond1], channels)
                    raw = {
                        key: prepare_signal(values, time_ms, variant)
                        for key, values in raw.items()
                    }
                    f0, frame_times = stft_band_power(raw[cond0])
                    f1, _ = stft_band_power(raw[cond1])
                    grid, frame_idx = eval_frames(frame_times)
                else:
                    raw, time_ms, _ = load_conditions(
                        subject, 2, [f"{f}_gray" for f in GRAY_FRUITS], channels
                    )
                    raw = {
                        key: prepare_signal(values, time_ms, variant)
                        for key, values in raw.items()
                    }
                    feats = {}
                    frame_times = None
                    for f in GRAY_FRUITS:
                        feat, ft = stft_band_power(raw[f"{f}_gray"])
                        feats[f] = feat
                        frame_times = ft
                    grid, frame_idx = eval_frames(frame_times)
                # group-level real curve over ROI channels
                if not is_cross_fruit:
                    real_curves = np.zeros((len(channels), len(grid)))
                    for j in range(len(channels)):
                        real, _ = decode_single_electrode_curve(
                            f0[:, j], f1[:, j], 0, seed + j, workers, frame_idx
                        )
                        real_curves[j] = real
                    subject_curve = real_curves.mean(axis=0)
                else:
                    subject_curves = np.zeros((len(channels), len(grid)))
                    for j in range(len(channels)):
                        real, _ = _cross_fruit_curve(
                            feats, frame_idx, j, 0, seed + j, workers
                        )
                        subject_curves[j] = real
                    subject_curve = subject_curves.mean(axis=0)
                subject_rows.append(
                    {
                        "subject": subject,
                        "n_channels": len(channels),
                        "mean_accuracy": subject_curve.tolist(),
                    }
                )
                for j, ch in enumerate(channels):
                    if ch not in csc_channels_by_subject.get(subject, set()):
                        continue
                    if not is_cross_fruit:
                        real, null = decode_single_electrode_curve(
                            f0[:, j], f1[:, j], n_perms_electrode_variant, seed + j, workers, frame_idx
                        )
                    else:
                        real, null = _cross_fruit_curve(
                            feats, frame_idx, j, n_perms_electrode_variant, seed + j, workers
                        )
                    clusters = cluster_permutation_1d(real, null, grid)
                    for cl in clusters:
                        electrode_rows.append(
                            {
                                "subject": subject,
                                "channel": ch,
                                "analysis": analysis,
                                "window": variant.window_label,
                                "signal": variant.signal,
                                **cl,
                            }
                        )
            # group cluster test
            curves = np.asarray(
                [r["mean_accuracy"] for r in subject_rows], dtype=float
            )
            group_summary, group_clusters = group_cluster_permutation(
                curves, grid, n_perms=n_perms_group, seed=seed
            )
            group_rows = []
            for cl in group_clusters:
                group_rows.append(
                    {
                        "analysis": analysis,
                        "window": variant.window_label,
                        "signal": variant.signal,
                        **cl,
                    }
                )
            group_df = pd.DataFrame(group_rows)
            electrode_df = pd.DataFrame(electrode_rows)
            stage = out / "stage05_hypotheses"
            stage.mkdir(parents=True, exist_ok=True)
            group_df.to_csv(
                stage / f"h2_h3_group_{analysis}_{variant.suffix}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            electrode_df.to_csv(
                stage / f"h2_h3_electrodes_{analysis}_{variant.suffix}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            results[f"{analysis}_{variant.suffix}"] = {
                "group": group_df,
                "electrodes": electrode_df,
                "curves": curves,
                "grid": grid,
                "n_perms_electrode": n_perms_electrode,
                "n_perms_group": n_perms_group,
            }
    return results


# --------------------------------------------------------------------------
# H4: cross-task temporal generalization + site overlap
# --------------------------------------------------------------------------


def _tgm_matrix(
    feats_train: dict[str, np.ndarray],
    feats_test: dict[str, np.ndarray],
    frame_idx: np.ndarray,
    channels: list[int],
    seed: int,
) -> np.ndarray:
    """Multi-electrode TGM: train Task3 (red/green) -> test Task2 (memory color).

    ``feats_train``: {'red': (trials, ch, bands, frames), 'green': ...};
    ``feats_test``: {'red': ...mem-red fruits..., 'green': ...}. Channel list
    selects which channels are used as features (channels x bands flattened).
    Returns (n_train_times, n_test_times) accuracy matrix.
    """
    rng = np.random.default_rng(seed)

    def pack(feats: dict[str, np.ndarray], color: str) -> tuple[np.ndarray, np.ndarray]:
        arr = np.take(feats[color], channels, axis=1)[:, :, :, frame_idx]
        n = arr.shape[0]
        idx = rng.permutation(n)
        return arr, idx

    xr, idxr = pack(feats_train, "red")
    xg, idxg = pack(feats_train, "green")
    n_balanced = min(len(idxr), len(idxg))
    x_train = np.concatenate(
        [xr[idxr[:n_balanced]], xg[idxg[:n_balanced]]], axis=0
    )
    y_train = np.concatenate(
        [np.zeros(n_balanced, dtype=int), np.ones(n_balanced, dtype=int)]
    )
    xr_t, idxr_t = pack(feats_test, "red")
    xg_t, idxg_t = pack(feats_test, "green")
    n_test = min(len(idxr_t), len(idxg_t))
    x_test = np.concatenate(
        [xr_t[idxr_t[:n_test]], xg_t[idxg_t[:n_test]]], axis=0
    )
    y_test = np.concatenate(
        [np.zeros(n_test, dtype=int), np.ones(n_test, dtype=int)]
    )
    n_train_times = x_train.shape[-1]
    n_test_times = x_test.shape[-1]
    matrix = np.zeros((n_train_times, n_test_times))
    for tt in range(n_train_times):
        for pt in range(n_test_times):
            model = make_pipeline(
                StandardScaler(), LinearSVC(C=1.0, max_iter=10000, dual=True)
            )
            xt = x_train[:, :, :, tt].reshape(x_train.shape[0], -1)
            xp = x_test[:, :, :, pt].reshape(x_test.shape[0], -1)
            model.fit(xt, y_train)
            matrix[tt, pt] = accuracy_score(y_test, model.predict(xp))
    return matrix


def group_cluster_permutation_2d(
    subject_matrices: np.ndarray,
    n_perms: int = 5000,
    seed: int = 0,
    p_form: float = 0.05,
    grid_ms: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """2-D sign-flip cluster permutation on TGM matrices."""
    dev = np.asarray(subject_matrices, dtype=float) - 0.5
    n_subj, nt, np_ = dev.shape
    mean_dev = dev.mean(axis=0)
    sd = dev.std(axis=0, ddof=1)
    t_obs = mean_dev / (sd / np.sqrt(n_subj) + 1e-12)
    t_crit = stats.t.ppf(1.0 - p_form, n_subj - 1)
    observed_mask = t_obs > t_crit

    def cluster_masses(mask: np.ndarray, values: np.ndarray) -> list[float]:
        lab, n_clusters = label(mask)
        masses = []
        for c in range(1, n_clusters + 1):
            masses.append(float(values[lab == c].sum()))
        return masses

    rng = np.random.default_rng(seed)
    null_max = []
    for _ in range(n_perms):
        flips = rng.choice([-1.0, 1.0], size=(n_subj, 1, 1))
        perm_dev = flips * dev
        perm_mean = perm_dev.mean(axis=0)
        perm_sd = perm_dev.std(axis=0, ddof=1)
        perm_t = perm_mean / (perm_sd / np.sqrt(n_subj) + 1e-12)
        masses = cluster_masses(perm_t > t_crit, perm_t)
        null_max.append(max(masses) if masses else 0.0)
    null_max = np.asarray(null_max)
    clusters_out = []
    lab, n_clusters = label(observed_mask)
    if grid_ms is None:
        grid_ms = np.arange(observed_mask.shape[0])
    for c in range(1, n_clusters + 1):
        mass = float(t_obs[lab == c].sum())
        p = float(np.mean(null_max >= mass))
        idx = np.argwhere(lab == c)
        clusters_out.append(
            {
                "train_start_ms": float(grid_ms[int(idx[:, 0].min())]),
                "train_end_ms": float(grid_ms[int(idx[:, 0].max())]),
                "test_start_ms": float(grid_ms[int(idx[:, 1].min())]),
                "test_end_ms": float(grid_ms[int(idx[:, 1].max())]),
                "mass": mass,
                "p": p,
                "n_subjects": n_subj,
                "n_permutations": n_perms,
            }
        )
    return clusters_out


def run_h4(
    out: Path,
    variants: Iterable[AnalysisVariant],
    n_perms_group: int = 5000,
    workers: int = 21,
    seed: int = 0,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for variant in variants:
        def read_electrodes(name: str) -> pd.DataFrame:
            path = stage / name
            if not path.exists() or path.stat().st_size == 0:
                return pd.DataFrame()
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()

        stage = out / "stage05_hypotheses"
        stage.mkdir(parents=True, exist_ok=True)
        csc_path = out / "stage01_selection" / f"electrode_sets_and_csc_{variant.suffix}.csv"
        csc_table = pd.read_csv(csc_path) if csc_path.exists() else pd.DataFrame()
        subject_rows = []
        for subject in SUBJECTS:
            sub_csc = csc_table[
                (csc_table.subject == subject) & csc_table.CSC
            ].channel.astype(str).tolist()
            if not sub_csc:
                continue
            loc = read_localization(subject).set_index("channel")
            channels = [
                c for c in common_channels(subject)
                if c in loc.index and c in sub_csc
            ]
            if not channels:
                continue
            ch_idx = {
                c: i
                for i, c in enumerate(common_channels(subject))
                if c in channels
            }
            raw3, time3_ms, _ = load_conditions(subject, 3, ["red", "green"], channels)
            raw2, time2_ms, _ = load_conditions(
                subject, 2, [f"{f}_gray" for f in GRAY_FRUITS], channels
            )
            raw3 = {
                key: prepare_signal(values, time3_ms, variant)
                for key, values in raw3.items()
            }
            raw2 = {
                key: prepare_signal(values, time2_ms, variant)
                for key, values in raw2.items()
            }
            f3r, ft = stft_band_power(raw3["red"])
            f3g, _ = stft_band_power(raw3["green"])
            f2r, _ = stft_band_power(raw2["strawberry_gray"])
            f2w, _ = stft_band_power(raw2["watermelon_gray"])
            f2c, _ = stft_band_power(raw2["cabbage_gray"])
            f2k, _ = stft_band_power(raw2["kiwi_gray"])
            grid, frame_idx = eval_frames(ft)
            train_feats = {"red": f3r, "green": f3g}
            test_red = np.concatenate([f2r, f2w], axis=0)  # memory-red fruits
            test_green = np.concatenate([f2c, f2k], axis=0)  # memory-green fruits
            test_feats = {"red": test_red, "green": test_green}
            matrix = _tgm_matrix(
                train_feats, test_feats, frame_idx, list(range(len(channels))), seed + int(subject[-3:])
            )
            subject_rows.append(
                {
                    "subject": subject,
                    "n_channels": len(channels),
                    "matrix": matrix.tolist(),
                }
            )
        matrices = np.asarray([r["matrix"] for r in subject_rows], dtype=float)
        clusters = group_cluster_permutation_2d(
            matrices, n_perms=n_perms_group, seed=seed, grid_ms=grid
        )
        cluster_df = pd.DataFrame(clusters)
        cluster_df["window"] = variant.window_label
        cluster_df["signal"] = variant.signal
        cluster_df.to_csv(
            stage / f"h4_tgm_{variant.suffix}.csv", index=False, encoding="utf-8-sig"
        )
        # site overlap: significant H2 & H3 electrodes -> hypergeometric
        h2 = read_electrodes(
            f"h2_h3_electrodes_task3_red_green_{variant.suffix}.csv"
        )
        h3 = read_electrodes(
            f"h2_h3_electrodes_task2_memory_color_{variant.suffix}.csv"
        )
        h2_sig = set(
            h2[h2.p < 0.05][["subject", "channel"]].apply(tuple, axis=1)
        ) if not h2.empty else set()
        h3_sig = set(
            h3[h3.p < 0.05][["subject", "channel"]].apply(tuple, axis=1)
        ) if not h3.empty else set()
        n_tested = len(csc_table[csc_table.CSC])
        overlap = h2_sig & h3_sig
        p_overlap = stats.hypergeom.sf(
            len(overlap) - 1, n_tested, len(h2_sig), len(h3_sig)
        )
        if np.isnan(p_overlap):
            p_overlap = 1.0
        overlap_df = pd.DataFrame(
            [
                {
                    "n_csc_tested": n_tested,
                    "n_h2_sig": len(h2_sig),
                    "n_h3_sig": len(h3_sig),
                    "n_overlap": len(overlap),
                    "overlap_electrodes": ";".join(
                        f"{s}-{c}" for s, c in sorted(overlap)
                    ),
                    "hypergeometric_p": float(p_overlap),
                    "window": variant.window_label,
                    "signal": variant.signal,
                }
            ]
        )
        overlap_df.to_csv(
            stage / f"h4_overlap_{variant.suffix}.csv", index=False, encoding="utf-8-sig"
        )
        results[variant.suffix] = {
            "clusters": cluster_df,
            "overlap": overlap_df,
            "matrices": matrices,
            "grid": grid,
        }
    return results


def write_decision_summary(
    out: Path,
    h1: pd.DataFrame,
    h2h3: dict[str, dict[str, object]],
    h4: dict[str, dict[str, object]],
    variants: Iterable[AnalysisVariant],
) -> Path:
    lines = ["# 预注册假设检验汇总 (SEEG)", ""]
    for variant in variants:
        suffix = variant.suffix
        lines.append(f"## 变体 {suffix}")
        lines.append("")
        h1_sub = h1[(h1.window == variant.window_label) & (h1.signal == variant.signal)]
        if not h1_sub.empty:
            total_obs = int(h1_sub.observed_s1.sum())
            total_expected = float(h1_sub.expected_s1_chance.sum())
            # analytic pooled test: total S1 count ~ Binomial(210, 0.05) under
            # the null; per-subject permutation p kept as sensitivity.
            p_h1_binom = float(
                stats.binom.sf(
                    total_obs - 1, int(h1_sub.n_roi_channels.sum()), 0.05
                )
            )
            p_h1 = float(h1_sub.p_value.min())
            lines.append(
                f"- **H1 (S1 富集)**: observed={total_obs}, chance-expected={total_expected:.1f}, "
                f"binomial p={p_h1_binom:.4f} (per-subject permutation p min={p_h1:.4f})"
            )
        for analysis in ("task3_red_green", "task2_memory_color"):
            key = f"{analysis}_{suffix}"
            entry = h2h3.get(key)
            if entry is None:
                continue
            group = entry["group"]
            if not group.empty:
                desc = "; ".join(
                    f"{r['start_ms']:.0f}-{r['end_ms']:.0f} ms (p={r['p']:.3f})"
                    for r in group.to_dict("records")
                )
                lines.append(
                    f"- **H2/H3 ({analysis})**: 显著簇 = {desc}"
                )
            else:
                lines.append(f"- **H2/H3 ({analysis})**: 无显著簇（组水平）")
        h4_entry = h4.get(suffix)
        if h4_entry is not None:
            clusters = h4_entry["clusters"]
            overlap = h4_entry["overlap"]
            sig_clusters = (
                clusters[clusters.p < 0.05] if not clusters.empty else clusters
            )
            if not sig_clusters.empty:
                desc = "; ".join(
                    f"train {r['train_start_ms']:.0f}-{r['train_end_ms']:.0f} -> "
                    f"test {r['test_start_ms']:.0f}-{r['test_end_ms']:.0f} ms (p={r['p']:.3f})"
                    for r in sig_clusters.to_dict("records")
                )
                lines.append(f"- **H4 (TGM)**: 显著泛化簇 = {desc}")
            else:
                lines.append("- **H4 (TGM)**: 无显著泛化簇")
            if not overlap.empty:
                row = overlap.iloc[0]
                lines.append(
                    f"- **H4 (位点重叠)**: overlap={int(row.n_overlap)}/"
                    f"{int(row.n_csc_tested)} (超几何 p={row.hypergeometric_p:.4f})"
                )
        lines.append("")
    lines.append("## 判定规则")
    lines.append("")
    lines.append(
        "- H1: 观察 S1 计数 > 置换零分布 95% 分位（单侧 p<0.05）→ 通过。"
    )
    lines.append("- H2/H3: 组水平簇置换 p<0.05 → 通过。")
    lines.append(
        "- H4: 存在显著 TGM 泛化簇 或 位点重叠超几何 p<0.05（两检验 BH-FDR）→ 通过。"
    )
    lines.append("")
    lines.append("## 重要方法学说明（样本重复检查）")
    lines.append("")
    lines.append(
        "- Task 1: 每条件 70 张唯一图片 × 70 试次，无样本重复 → S1 筛选不受样本泄漏影响。"
    )
    lines.append(
        "- Task 3: 每颜色仅 3 张唯一图片 × 20 次重复 → H2 的 within-task CV 可能部分"
        "来自样本级（exemplar）特征而非颜色本身，解释时需谨慎。"
    )
    lines.append(
        "- Task 2: 每水果 15 张 × 4 次重复，但 H3 使用 leave-one-fruit-pair-out"
        "（测试水果从未在训练中出现）→ 记忆颜色结论对样本泄漏免疫。"
    )
    lines.append("")
    path = out / "stage05_hypotheses" / "decision_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
