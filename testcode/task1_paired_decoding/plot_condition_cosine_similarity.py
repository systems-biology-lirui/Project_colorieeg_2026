from dataclasses import replace
from itertools import permutations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from scipy.stats import spearmanr

from common import (
    build_common_parser,
    build_output_dir,
    load_preprocessed_paired_dataset,
    normalize_common_args,
    resolve_selected_category_indices,
    select_paired_dataset_categories,
    smooth_time_axis,
)


DEFAULT_SMOOTH_WIN = 1
DEFAULT_SNAPSHOT_STEP_MS = 50.0



def build_parser():
    parser = build_common_parser('Time-resolved RSA on averaged color and gray condition patterns.')
    for action in parser._actions:
        if "--metric" in action.option_strings:
            parser._handle_conflict_resolve(None, [("--metric", action)])
    parser.set_defaults(metric='auc')

    parser.add_argument("--selected-categories", default="all", help="Comma-separated category names or indices (1-based). Use all for every category.")
    parser.add_argument("--smooth-win", type=int, default=DEFAULT_SMOOTH_WIN, help="Moving-average window on the time axis before building averaged condition patterns.")
    parser.add_argument("--snapshot-step-ms", type=float, default=DEFAULT_SNAPSHOT_STEP_MS, help="Snapshot spacing in ms for 8-condition similarity matrices.")
    return parser


def smooth_trial_data(trial_data, smooth_win):
    if smooth_win <= 1:
        return np.asarray(trial_data, dtype=float)
    smoothed = [smooth_time_axis(sample, smooth_win) for sample in np.asarray(trial_data, dtype=float)]
    return np.stack(smoothed, axis=0)


def compute_condition_statistics(paired_dataset, centered=False, smooth_win=1):
    color_data = paired_dataset.color
    gray_data = paired_dataset.gray
    if centered:
        pair_mean = 0.5 * (color_data + gray_data)
        color_data = color_data - pair_mean
        gray_data = gray_data - pair_mean

    color_data = smooth_trial_data(color_data, smooth_win)
    gray_data = smooth_trial_data(gray_data, smooth_win)

    color_means = []
    gray_means = []
    color_trials_by_category = []
    gray_trials_by_category = []
    for category_idx in range(len(paired_dataset.category_names)):
        mask = paired_dataset.category_ids == category_idx
        category_color_trials = color_data[mask]
        category_gray_trials = gray_data[mask]
        color_means.append(np.mean(category_color_trials, axis=0))
        gray_means.append(np.mean(category_gray_trials, axis=0))
        color_trials_by_category.append(category_color_trials)
        gray_trials_by_category.append(category_gray_trials)
    return (
        np.stack(color_means, axis=0),
        np.stack(gray_means, axis=0),
        color_trials_by_category,
        gray_trials_by_category,
    )


def correlation_matrix_from_patterns(patterns):
    centered = patterns - np.mean(patterns, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norms, 1e-12)
    corr = normalized @ normalized.T
    return np.clip(corr, -1.0, 1.0)


def extract_upper_triangle(matrix):
    upper_idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[upper_idx]


def safe_spearman(vector_a, vector_b):
    result = spearmanr(vector_a, vector_b)
    statistic = getattr(result, 'statistic', result[0])
    if np.isnan(statistic):
        if np.allclose(vector_a, vector_b, atol=1e-12, rtol=1e-12):
            return 1.0
        return 0.0
    return float(statistic)


def compute_cross_rsm(color_patterns, gray_patterns):
    color_centered = color_patterns - np.mean(color_patterns, axis=1, keepdims=True)
    gray_centered = gray_patterns - np.mean(gray_patterns, axis=1, keepdims=True)
    color_norm = color_centered / np.maximum(np.linalg.norm(color_centered, axis=1, keepdims=True), 1e-12)
    gray_norm = gray_centered / np.maximum(np.linalg.norm(gray_centered, axis=1, keepdims=True), 1e-12)
    return np.clip(color_norm @ gray_norm.T, -1.0, 1.0)


def compute_full_condition_rsm(color_patterns, gray_patterns):
    combined_patterns = []
    for category_idx in range(color_patterns.shape[0]):
        combined_patterns.append(color_patterns[category_idx])
        combined_patterns.append(gray_patterns[category_idx])
    combined_patterns = np.stack(combined_patterns, axis=0)
    return correlation_matrix_from_patterns(combined_patterns)


def compute_rsa_over_time(color_means, gray_means):
    n_categories, _, n_timepoints = color_means.shape
    color_rsms = np.zeros((n_timepoints, n_categories, n_categories), dtype=float)
    gray_rsms = np.zeros((n_timepoints, n_categories, n_categories), dtype=float)
    rsa_timecourse = np.zeros(n_timepoints, dtype=float)
    cross_diagonal = np.zeros((n_categories, n_timepoints), dtype=float)
    cross_offdiag_mean = np.zeros(n_timepoints, dtype=float)

    for time_idx in range(n_timepoints):
        color_patterns = color_means[:, :, time_idx]
        gray_patterns = gray_means[:, :, time_idx]
        color_rsm = correlation_matrix_from_patterns(color_patterns)
        gray_rsm = correlation_matrix_from_patterns(gray_patterns)

        color_rsms[time_idx] = color_rsm
        gray_rsms[time_idx] = gray_rsm
        rsa_timecourse[time_idx] = safe_spearman(
            extract_upper_triangle(color_rsm),
            extract_upper_triangle(gray_rsm),
        )

        cross_rsm = compute_cross_rsm(color_patterns, gray_patterns)
        cross_diagonal[:, time_idx] = np.diag(cross_rsm)
        cross_offdiag_mean[time_idx] = float(np.mean(extract_upper_triangle(0.5 * (cross_rsm + cross_rsm.T))))

    return color_rsms, gray_rsms, rsa_timecourse, cross_diagonal, cross_offdiag_mean


def generate_category_permutations(n_categories, n_perms, random_state):
    if n_categories <= 1 or n_perms <= 0:
        return np.empty((0, n_categories), dtype=int)

    identity = tuple(range(n_categories))
    all_perms = [perm for perm in permutations(range(n_categories)) if perm != identity]
    if len(all_perms) <= n_perms:
        return np.asarray(all_perms, dtype=int)

    rng = np.random.RandomState(random_state)
    selected_indices = rng.choice(len(all_perms), size=n_perms, replace=False)
    return np.asarray([all_perms[idx] for idx in selected_indices], dtype=int)


def compute_rsa_permutation_distribution(color_means, gray_means, n_perms, random_state):
    permutations_array = generate_category_permutations(color_means.shape[0], n_perms, random_state)
    if permutations_array.size == 0:
        return np.full((0, color_means.shape[2]), np.nan), permutations_array

    perm_curves = []
    for perm in permutations_array:
        _, perm_gray_rsms, perm_rsa_timecourse, _, _ = compute_rsa_over_time(color_means, gray_means[perm])
        perm_curves.append(perm_rsa_timecourse)
    return np.asarray(perm_curves, dtype=float), permutations_array


def compute_category_diagonal_permutation_distribution(color_means, gray_means, n_perms, random_state):
    permutations_array = generate_category_permutations(color_means.shape[0], n_perms, random_state)
    if permutations_array.size == 0:
        return np.full((0, color_means.shape[0], color_means.shape[2]), np.nan), permutations_array

    perm_curves = np.zeros((permutations_array.shape[0], color_means.shape[0], color_means.shape[2]), dtype=float)
    for perm_idx, perm in enumerate(permutations_array):
        for time_idx in range(color_means.shape[2]):
            cross_rsm = compute_cross_rsm(color_means[:, :, time_idx], gray_means[perm, :, time_idx])
            perm_curves[perm_idx, :, time_idx] = np.diag(cross_rsm)
    return perm_curves, permutations_array


def cluster_permutation_significance(observed_curve, perm_curves, time_vector, min_time_ms=0.0):
    if perm_curves is None or np.size(perm_curves) == 0:
        return np.full(observed_curve.shape, np.nan, dtype=float), np.zeros_like(observed_curve, dtype=bool), np.nan

    threshold_95 = np.percentile(perm_curves, 95, axis=0)
    valid_mask = np.asarray(time_vector, dtype=float) >= float(min_time_ms)
    supra_threshold = (observed_curve > threshold_95) & valid_mask
    clusters, n_clusters = label(supra_threshold.astype(int))
    cluster_masses = []
    for cluster_idx in range(1, n_clusters + 1):
        idx = clusters == cluster_idx
        cluster_masses.append(float(np.sum(observed_curve[idx] - threshold_95[idx])))

    null_cluster_masses = []
    for perm_curve in perm_curves:
        perm_binary = (perm_curve > threshold_95) & valid_mask
        perm_clusters, perm_count = label(perm_binary.astype(int))
        max_mass = 0.0
        for cluster_idx in range(1, perm_count + 1):
            idx = perm_clusters == cluster_idx
            max_mass = max(max_mass, float(np.sum(perm_curve[idx] - threshold_95[idx])))
        null_cluster_masses.append(max_mass)

    critical_mass = np.percentile(null_cluster_masses, 95) if null_cluster_masses else np.nan
    sig_indices = np.zeros_like(observed_curve, dtype=bool)
    for cluster_idx in range(1, n_clusters + 1):
        if cluster_masses[cluster_idx - 1] > critical_mass:
            sig_indices[clusters == cluster_idx] = True
    return threshold_95, sig_indices, critical_mass


def resolve_snapshot_indices(time_vector, snapshot_step_ms):
    time_vector = np.asarray(time_vector, dtype=float)
    requested = np.arange(time_vector[0], time_vector[-1] + 0.5 * snapshot_step_ms, snapshot_step_ms, dtype=float)
    indices = []
    actual_times = []
    for requested_time in requested:
        idx = int(np.argmin(np.abs(time_vector - requested_time)))
        if idx in indices:
            continue
        indices.append(idx)
        actual_times.append(float(time_vector[idx]))
    return np.asarray(indices, dtype=int), np.asarray(actual_times, dtype=float)


def plot_category_similarity_panel(ax, time_vector, category_curve, category_name, color, threshold_95=None, sig_indices=None):
    ax.plot(time_vector, category_curve, color=color, linewidth=2.0, label=category_name)
    if threshold_95 is not None and np.any(np.isfinite(threshold_95)):
        ax.plot(time_vector, threshold_95, color='#9467bd', linestyle='-.', linewidth=1.4, label='95% permutation threshold')
    if sig_indices is not None and np.any(sig_indices):
        ax.fill_between(time_vector, 0, 1, where=sig_indices, color='gray', alpha=0.18, transform=ax.get_xaxis_transform(), label='Significant cluster')
    ax.axhline(0.0, color='black', linestyle=':', linewidth=1.0)
    ax.axvline(0.0, color='#999999', linestyle='--', linewidth=1.0)
    ax.set_title(f'{category_name} color-gray similarity', fontsize=11)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Pattern correlation')
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best', fontsize=9)


def plot_condition_snapshot_grid(save_path, snapshot_times, condition_labels, condition_rsms, snapshot_indices, figure_title):
    n_cols = 5
    n_rows = int(np.ceil(len(snapshot_indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows), squeeze=False)
    flat_axes = axes.ravel()
    image = None
    for panel_idx, (time_idx, time_ms) in enumerate(zip(snapshot_indices, snapshot_times)):
        ax = flat_axes[panel_idx]
        image = ax.imshow(condition_rsms[time_idx], vmin=-1.0, vmax=1.0, cmap='coolwarm')
        ax.set_title(f'{time_ms:.0f} ms', fontsize=10)
        ax.set_xticks(np.arange(len(condition_labels)))
        ax.set_yticks(np.arange(len(condition_labels)))
        ax.set_xticklabels(condition_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(condition_labels, fontsize=7)
    for ax in flat_axes[len(snapshot_indices):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.92, wspace=0.45, hspace=0.55)
    cbar_ax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label('Pattern correlation', rotation=90)
    fig.suptitle(figure_title, fontsize=13)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = replace(normalize_common_args(args), metric_name='mean_pattern_rsa')
    roi_bundle, paired_dataset = load_preprocessed_paired_dataset(config)
    selected_category_indices = resolve_selected_category_indices(args.selected_categories, config.category_names)
    paired_dataset = select_paired_dataset_categories(paired_dataset, selected_category_indices)
    snapshot_indices, snapshot_times = resolve_snapshot_indices(roi_bundle.time_vector, args.snapshot_step_ms)

    raw_color_means, raw_gray_means, _, _ = compute_condition_statistics(
        paired_dataset,
        centered=False,
        smooth_win=args.smooth_win,
    )
    centered_color_means, centered_gray_means, _, _ = compute_condition_statistics(
        paired_dataset,
        centered=True,
        smooth_win=args.smooth_win,
    )

    raw_color_rsms, raw_gray_rsms, raw_rsa_timecourse, raw_cross_diagonal, raw_cross_offdiag_mean = compute_rsa_over_time(
        raw_color_means,
        raw_gray_means,
    )
    centered_color_rsms, centered_gray_rsms, centered_rsa_timecourse, centered_cross_diagonal, centered_cross_offdiag_mean = compute_rsa_over_time(
        centered_color_means,
        centered_gray_means,
    )
    raw_perm_curves, raw_permutations = compute_rsa_permutation_distribution(raw_color_means, raw_gray_means, config.n_perms, config.random_state)
    centered_perm_curves, centered_permutations = compute_rsa_permutation_distribution(centered_color_means, centered_gray_means, config.n_perms, config.random_state)
    raw_threshold_95, raw_sig_indices, raw_critical_mass = cluster_permutation_significance(raw_rsa_timecourse, raw_perm_curves, roi_bundle.time_vector)
    centered_threshold_95, centered_sig_indices, centered_critical_mass = cluster_permutation_significance(centered_rsa_timecourse, centered_perm_curves, roi_bundle.time_vector)
    raw_category_perm_curves, raw_category_permutations = compute_category_diagonal_permutation_distribution(
        raw_color_means,
        raw_gray_means,
        config.n_perms,
        config.random_state,
    )

    raw_condition_rsms = np.stack(
        [compute_full_condition_rsm(raw_color_means[:, :, time_idx], raw_gray_means[:, :, time_idx]) for time_idx in range(raw_color_means.shape[2])],
        axis=0,
    )
    centered_condition_rsms = np.stack(
        [compute_full_condition_rsm(centered_color_means[:, :, time_idx], centered_gray_means[:, :, time_idx]) for time_idx in range(centered_color_means.shape[2])],
        axis=0,
    )
    category_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf']
    raw_category_thresholds = []
    raw_category_sig_indices = []
    raw_category_critical_masses = []
    for category_idx in range(len(paired_dataset.category_names)):
        threshold_95, sig_indices, critical_mass = cluster_permutation_significance(
            raw_cross_diagonal[category_idx],
            raw_category_perm_curves[:, category_idx, :],
            roi_bundle.time_vector,
        )
        raw_category_thresholds.append(threshold_95)
        raw_category_sig_indices.append(sig_indices)
        raw_category_critical_masses.append(critical_mass)
    raw_category_thresholds = np.asarray(raw_category_thresholds, dtype=float)
    raw_category_sig_indices = np.asarray(raw_category_sig_indices, dtype=bool)
    raw_category_critical_masses = np.asarray(raw_category_critical_masses, dtype=float)

    out_dir = build_output_dir('condition_mean_rsa', config)
    fig_path = out_dir / 'condition_mean_rsa.png'
    raw_snapshot_fig_path = out_dir / 'condition_mean_rsa_snapshots_raw.png'
    centered_snapshot_fig_path = out_dir / 'condition_mean_rsa_snapshots_centered.png'
    npz_path = out_dir / 'condition_mean_rsa.npz'

    fig, axes = plt.subplots(len(paired_dataset.category_names), 1, figsize=(12, 2.8 * len(paired_dataset.category_names)), sharex=True, squeeze=False)
    for category_idx, category_name in enumerate(paired_dataset.category_names):
        plot_category_similarity_panel(
            axes[category_idx, 0],
            roi_bundle.time_vector,
            raw_cross_diagonal[category_idx],
            category_name,
            category_colors[category_idx % len(category_colors)],
            threshold_95=raw_category_thresholds[category_idx],
            sig_indices=raw_category_sig_indices[category_idx],
        )
    fig.suptitle(
        f'Task1 raw category-wise color-gray similarity | {config.subject} | {config.feature_kind} | {config.roi_name} | smooth={args.smooth_win}',
        fontsize=13,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    condition_labels = []
    for category_name in paired_dataset.category_names:
        condition_labels.extend([f'{category_name}-color', f'{category_name}-gray'])
    plot_condition_snapshot_grid(
        raw_snapshot_fig_path,
        snapshot_times,
        condition_labels,
        raw_condition_rsms,
        snapshot_indices,
        'Raw 8-condition similarity snapshots',
    )
    plot_condition_snapshot_grid(
        centered_snapshot_fig_path,
        snapshot_times,
        condition_labels,
        centered_condition_rsms,
        snapshot_indices,
        'Centered 8-condition similarity snapshots',
    )

    np.savez(
        npz_path,
        subject=np.array(config.subject),
        feature_kind=np.array(config.feature_kind),
        roi_name=np.array(config.roi_name),
        grouped_data_mat=np.array(str(config.grouped_data_mat)),
        roi_path=np.array(str(roi_bundle.roi_path)),
        task_field=np.array(roi_bundle.task_field),
        category_names=np.asarray(paired_dataset.category_names, dtype=object),
        selected_category_indices=np.asarray(selected_category_indices, dtype=int),
        analysis_name=np.array('mean_pattern_rsa'),
        rsm_similarity_metric=np.array('spearman'),
        pattern_similarity_metric=np.array('pearson_correlation'),
        smooth_win=np.array(args.smooth_win),
        time_ms=np.asarray(roi_bundle.time_vector, dtype=float),
        raw_color_means=raw_color_means,
        raw_gray_means=raw_gray_means,
        centered_color_means=centered_color_means,
        centered_gray_means=centered_gray_means,
        raw_color_rsms=raw_color_rsms,
        raw_gray_rsms=raw_gray_rsms,
        centered_color_rsms=centered_color_rsms,
        centered_gray_rsms=centered_gray_rsms,
        raw_condition_rsms=raw_condition_rsms,
        centered_condition_rsms=centered_condition_rsms,
        raw_rsa_timecourse=raw_rsa_timecourse,
        centered_rsa_timecourse=centered_rsa_timecourse,
        raw_perm_curves=raw_perm_curves,
        centered_perm_curves=centered_perm_curves,
        raw_threshold_95=raw_threshold_95,
        centered_threshold_95=centered_threshold_95,
        raw_sig_indices=raw_sig_indices,
        centered_sig_indices=centered_sig_indices,
        raw_critical_mass=np.array(raw_critical_mass),
        centered_critical_mass=np.array(centered_critical_mass),
        raw_permutations=raw_permutations,
        centered_permutations=centered_permutations,
        raw_cross_diagonal=raw_cross_diagonal,
        centered_cross_diagonal=centered_cross_diagonal,
        raw_category_perm_curves=raw_category_perm_curves,
        raw_category_permutations=raw_category_permutations,
        raw_category_thresholds=raw_category_thresholds,
        raw_category_sig_indices=raw_category_sig_indices,
        raw_category_critical_masses=raw_category_critical_masses,
        raw_cross_offdiag_mean=raw_cross_offdiag_mean,
        centered_cross_offdiag_mean=centered_cross_offdiag_mean,
        snapshot_indices=snapshot_indices,
        snapshot_times_ms=snapshot_times,
        snapshot_step_ms=np.array(args.snapshot_step_ms),
        condition_labels=np.asarray(condition_labels, dtype=object),
    )

    print(
        {
            'figure_path': str(fig_path),
            'raw_snapshot_figure_path': str(raw_snapshot_fig_path),
            'centered_snapshot_figure_path': str(centered_snapshot_fig_path),
            'summary_path': str(npz_path),
            'selected_category_names': paired_dataset.category_names,
            'analysis_name': 'mean_pattern_rsa',
            'n_perms_used': int(raw_perm_curves.shape[0]),
            'smooth_win': int(args.smooth_win),
        }
    )


if __name__ == '__main__':
    main()