from dataclasses import replace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import (
    PairedDataset,
    build_common_parser,
    build_output_dir,
    load_preprocessed_paired_dataset,
    normalize_common_args,
    paired_counts_by_category,
    resolve_selected_category_indices,
    select_paired_dataset_categories,
)


CURVE_COLORS = {
    'within_color': '#1f77b4',
    'within_gray': '#d62728',
    'between_color_gray': '#2ca02c',
}

DEFAULT_USE_SPLIT_HALF_CROSS = True


def parse_bool_flag(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if text in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'Invalid boolean flag: {value}')


def build_parser():
    parser = build_common_parser('Single-trial dissimilarity after averaging matched trials across categories by trial index.')
    for action in parser._actions:
        if '--time-smooth-win' in action.option_strings and '--smooth' not in action.option_strings:
            action.option_strings.append('--smooth')
            parser._option_string_actions['--smooth'] = action
        if '--decoding-step' in action.option_strings and '--step' not in action.option_strings:
            action.option_strings.append('--step')
            parser._option_string_actions['--step'] = action
    parser.add_argument('--selected-categories', default='all', help='Comma-separated category names or indices (1-based). Use all for every category.')
    parser.add_argument(
        '--use-split-half-cross',
        type=parse_bool_flag,
        default=DEFAULT_USE_SPLIT_HALF_CROSS,
        help='Whether raw curves use split-half cross comparisons.',
    )
    parser.add_argument(
        '--center-use-split-half',
        type=parse_bool_flag,
        default=None,
        help='Whether centered curves use split-half cross comparisons. Defaults to --use-split-half-cross when omitted.',
    )
    return parser


def correlation_matrix_from_patterns(patterns):
    centered = patterns - np.mean(patterns, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norms, 1e-12)
    corr = normalized @ normalized.T
    return np.clip(corr, -1.0, 1.0)


def extract_upper_triangle(matrix):
    upper_idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[upper_idx]


def mean_upper_triangle(matrix):
    values = extract_upper_triangle(matrix)
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def compute_cross_correlation(color_patterns, gray_patterns):
    color_centered = color_patterns - np.mean(color_patterns, axis=1, keepdims=True)
    gray_centered = gray_patterns - np.mean(gray_patterns, axis=1, keepdims=True)
    color_norm = color_centered / np.maximum(np.linalg.norm(color_centered, axis=1, keepdims=True), 1e-12)
    gray_norm = gray_centered / np.maximum(np.linalg.norm(gray_centered, axis=1, keepdims=True), 1e-12)
    return np.clip(color_norm @ gray_norm.T, -1.0, 1.0)


def split_two_folds(n_trials):
    if n_trials < 2:
        indices = np.arange(n_trials, dtype=int)
        return indices, indices

    fold1 = np.arange(0, n_trials, 2, dtype=int)
    fold2 = np.arange(1, n_trials, 2, dtype=int)
    if fold2.size == 0:
        fold2 = fold1.copy()
    return fold1, fold2


def mean_cross_fold_dissimilarity(patterns_a, patterns_b):
    cross_corr = compute_cross_correlation(patterns_a, patterns_b)
    return 1.0 - float(np.mean(cross_corr))


def mean_within_fold_dissimilarity(patterns):
    corr = correlation_matrix_from_patterns(patterns)
    return 1.0 - mean_upper_triangle(corr)


def compute_dissimilarity_curves(color_trials, gray_trials, time_vector, use_split_half_cross):
    n_timepoints = color_trials.shape[2]
    within_color = np.zeros((1, n_timepoints), dtype=float)
    within_gray = np.zeros((1, n_timepoints), dtype=float)
    between_color_gray = np.zeros((1, n_timepoints), dtype=float)

    if use_split_half_cross:
        fold1_idx, fold2_idx = split_two_folds(color_trials.shape[0])
        color_fold1 = color_trials[fold1_idx]
        gray_fold2 = gray_trials[fold2_idx]

    for time_idx in range(n_timepoints):
        if use_split_half_cross:
            color_patterns_fold1 = color_fold1[:, :, time_idx]
            gray_patterns_fold2 = gray_fold2[:, :, time_idx]
            within_color[0, time_idx] = mean_within_fold_dissimilarity(color_patterns_fold1)
            within_gray[0, time_idx] = mean_within_fold_dissimilarity(gray_patterns_fold2)
            between_color_gray[0, time_idx] = mean_cross_fold_dissimilarity(color_patterns_fold1, gray_patterns_fold2)
        else:
            color_patterns = color_trials[:, :, time_idx]
            gray_patterns = gray_trials[:, :, time_idx]
            color_corr = correlation_matrix_from_patterns(color_patterns)
            gray_corr = correlation_matrix_from_patterns(gray_patterns)
            cross_corr = compute_cross_correlation(color_patterns, gray_patterns)
            within_color[0, time_idx] = 1.0 - mean_upper_triangle(color_corr)
            within_gray[0, time_idx] = 1.0 - mean_upper_triangle(gray_corr)
            between_color_gray[0, time_idx] = 1.0 - float(np.mean(cross_corr))

    return {
        'within_color': within_color,
        'within_gray': within_gray,
        'between_color_gray': between_color_gray,
        'time_ms': np.asarray(time_vector, dtype=float),
    }


def center_color_gray_trials(color_trials, gray_trials):
    pair_mean = 0.5 * (color_trials + gray_trials)
    return color_trials - pair_mean, gray_trials - pair_mean


def compute_normalized_contrast_curves(curves):
    contrast = curves['between_color_gray'] - 0.5 * (curves['within_color'] + curves['within_gray'])
    normalized = np.zeros_like(contrast)
    for curve_idx in range(contrast.shape[0]):
        curve = contrast[curve_idx]
        curve_min = float(np.min(curve))
        curve_max = float(np.max(curve))
        if np.isclose(curve_max, curve_min):
            normalized[curve_idx] = 0.0
        else:
            normalized[curve_idx] = (curve - curve_min) / (curve_max - curve_min)
    return {
        'contrast': contrast,
        'normalized_contrast': normalized,
        'time_ms': curves['time_ms'],
    }


def subsample_curve_dict(curves, decoding_step):
    step = int(decoding_step)
    if step <= 1:
        return curves
    return {
        'within_color': curves['within_color'][:, ::step],
        'within_gray': curves['within_gray'][:, ::step],
        'between_color_gray': curves['between_color_gray'][:, ::step],
        'time_ms': curves['time_ms'][::step],
    }


def subsample_contrast_dict(contrast_curves, decoding_step):
    step = int(decoding_step)
    if step <= 1:
        return contrast_curves
    return {
        'contrast': contrast_curves['contrast'][:, ::step],
        'normalized_contrast': contrast_curves['normalized_contrast'][:, ::step],
        'time_ms': contrast_curves['time_ms'][::step],
    }


def build_cross_category_average_dataset(paired_dataset):
    category_counts = paired_counts_by_category(paired_dataset)
    category_color_trials = []
    category_gray_trials = []
    for category_idx in range(len(paired_dataset.category_names)):
        category_mask = paired_dataset.category_ids == category_idx
        category_color_trials.append(paired_dataset.color[category_mask])
        category_gray_trials.append(paired_dataset.gray[category_mask])

    min_count = min(trials.shape[0] for trials in category_color_trials)
    if min_count < 1:
        raise ValueError('At least one matched trial is required in every selected category.')

    truncated_color = np.stack([trials[:min_count] for trials in category_color_trials], axis=0)
    truncated_gray = np.stack([trials[:min_count] for trials in category_gray_trials], axis=0)
    averaged_color = np.mean(truncated_color, axis=0)
    averaged_gray = np.mean(truncated_gray, axis=0)

    aggregate_name = 'cross_category_average'
    aggregate_dataset = PairedDataset(
        color=averaged_color,
        gray=averaged_gray,
        pair_ids=np.arange(min_count, dtype=int),
        category_ids=np.zeros(min_count, dtype=int),
        sample_keys=[f'{aggregate_name}_trial_{trial_idx + 1}' for trial_idx in range(min_count)],
        color_repeat_indices=np.full(min_count, -1, dtype=int),
        gray_repeat_indices=np.full(min_count, -1, dtype=int),
        matched_counts=[min_count],
        category_names=[aggregate_name],
    )
    return aggregate_dataset, category_counts, int(min_count)


def plot_category_curves(save_path, category_names, curves, time_vector, subject, feature_kind, roi_name, suffix_note):
    fig, axes = plt.subplots(len(category_names), 1, figsize=(12, 2.8 * len(category_names)), sharex=True, squeeze=False)
    for category_idx, category_name in enumerate(category_names):
        ax = axes[category_idx, 0]
        ax.plot(time_vector, curves['within_color'][category_idx], color=CURVE_COLORS['within_color'], linewidth=2.0, label='Within-color dissimilarity')
        ax.plot(time_vector, curves['within_gray'][category_idx], color=CURVE_COLORS['within_gray'], linewidth=2.0, linestyle='--', label='Within-gray dissimilarity')
        ax.plot(time_vector, curves['between_color_gray'][category_idx], color=CURVE_COLORS['between_color_gray'], linewidth=2.0, linestyle='-.', label='Color-gray dissimilarity')
        ax.axvline(0.0, color='#999999', linestyle='--', linewidth=1.0)
        ax.set_title(f'{category_name} single-trial pattern dissimilarity', fontsize=11)
        ax.set_ylabel('1 - correlation')
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(loc='best', fontsize=8)
    axes[-1, 0].set_xlabel('Time (ms)')
    fig.suptitle(f'Task1 cross-category average dissimilarity | {subject} | {feature_kind} | {roi_name} | {suffix_note}', fontsize=13)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_normalized_contrast_grid(save_path, category_names, raw_contrast, centered_contrast, time_vector, subject, feature_kind, roi_name):
    fig, axes = plt.subplots(len(category_names), 2, figsize=(14, 2.8 * len(category_names)), sharex=True, squeeze=False)
    column_titles = ['Raw normalized contrast', 'Centered normalized contrast']
    for category_idx, category_name in enumerate(category_names):
        for col_idx, contrast_data in enumerate((raw_contrast, centered_contrast)):
            ax = axes[category_idx, col_idx]
            ax.plot(
                time_vector,
                contrast_data['normalized_contrast'][category_idx],
                color='#2ca02c',
                linewidth=2.2,
                label='Normalized green - mean(blue, red)',
            )
            ax.axvline(0.0, color='#999999', linestyle='--', linewidth=1.0)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.35)
            if category_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f'{category_name}\nNormalized value')
            if category_idx == len(category_names) - 1:
                ax.set_xlabel('Time (ms)')
            ax.legend(loc='best', fontsize=8)
    fig.suptitle(f'Task1 cross-category average normalized contrast | {subject} | {feature_kind} | {roi_name}', fontsize=13)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = replace(normalize_common_args(args), metric_name='cross_category_average_single_trial_dissimilarity')
    roi_bundle, paired_dataset = load_preprocessed_paired_dataset(config)
    selected_category_indices = resolve_selected_category_indices(args.selected_categories, config.category_names)
    paired_dataset = select_paired_dataset_categories(paired_dataset, selected_category_indices)

    aggregate_dataset, category_counts, aggregated_trial_count = build_cross_category_average_dataset(paired_dataset)
    raw_use_split_half = bool(args.use_split_half_cross)
    centered_use_split_half = raw_use_split_half if args.center_use_split_half is None else bool(args.center_use_split_half)

    print('Final matched trial counts by category:')
    for category_name, count in category_counts.items():
        print(f'  {category_name}: {count}')
    print(f'Cross-category averaged trial count: {aggregated_trial_count}')

    raw_curves = compute_dissimilarity_curves(
        aggregate_dataset.color,
        aggregate_dataset.gray,
        roi_bundle.time_vector,
        use_split_half_cross=raw_use_split_half,
    )
    centered_color, centered_gray = center_color_gray_trials(aggregate_dataset.color, aggregate_dataset.gray)
    centered_curves = compute_dissimilarity_curves(
        centered_color,
        centered_gray,
        roi_bundle.time_vector,
        use_split_half_cross=centered_use_split_half,
    )
    raw_contrast = compute_normalized_contrast_curves(raw_curves)
    centered_contrast = compute_normalized_contrast_curves(centered_curves)

    raw_curves = subsample_curve_dict(raw_curves, config.decoding_step)
    centered_curves = subsample_curve_dict(centered_curves, config.decoding_step)
    raw_contrast = subsample_contrast_dict(raw_contrast, config.decoding_step)
    centered_contrast = subsample_contrast_dict(centered_contrast, config.decoding_step)

    out_dir = build_output_dir('cross_category_average_single_trial_dissimilarity', config)
    raw_fig_path = out_dir / 'cross_category_average_single_trial_dissimilarity_raw.png'
    centered_fig_path = out_dir / 'cross_category_average_single_trial_dissimilarity_centered.png'
    contrast_fig_path = out_dir / 'cross_category_average_single_trial_dissimilarity_normalized_contrast.png'
    npz_path = out_dir / 'cross_category_average_single_trial_dissimilarity.npz'

    plot_category_curves(
        raw_fig_path,
        aggregate_dataset.category_names,
        raw_curves,
        raw_curves['time_ms'],
        config.subject,
        config.feature_kind,
        config.roi_name,
        f'raw | split-half={raw_use_split_half}',
    )
    plot_category_curves(
        centered_fig_path,
        aggregate_dataset.category_names,
        centered_curves,
        centered_curves['time_ms'],
        config.subject,
        config.feature_kind,
        config.roi_name,
        f'centered | split-half={centered_use_split_half}',
    )
    plot_normalized_contrast_grid(
        contrast_fig_path,
        aggregate_dataset.category_names,
        raw_contrast,
        centered_contrast,
        raw_curves['time_ms'],
        config.subject,
        config.feature_kind,
        config.roi_name,
    )

    np.savez(
        npz_path,
        subject=np.array(config.subject),
        feature_kind=np.array(config.feature_kind),
        roi_name=np.array(config.roi_name),
        grouped_data_mat=np.array(str(config.grouped_data_mat)),
        roi_path=np.array(str(roi_bundle.roi_path)),
        task_field=np.array(roi_bundle.task_field),
        time_smooth_win=np.array(config.time_smooth_win),
        decoding_step=np.array(config.decoding_step),
        category_names=np.asarray(aggregate_dataset.category_names, dtype=object),
        source_category_names=np.asarray(paired_dataset.category_names, dtype=object),
        selected_category_indices=np.asarray(selected_category_indices, dtype=int),
        source_pair_counts=np.asarray([category_counts[name] for name in paired_dataset.category_names], dtype=int),
        aggregated_trial_count=np.array(aggregated_trial_count),
        use_split_half_cross=np.array(raw_use_split_half),
        center_use_split_half=np.array(centered_use_split_half),
        time_ms=raw_curves['time_ms'],
        raw_within_color_dissimilarity=raw_curves['within_color'],
        raw_within_gray_dissimilarity=raw_curves['within_gray'],
        raw_between_color_gray_dissimilarity=raw_curves['between_color_gray'],
        raw_contrast_dissimilarity=raw_contrast['contrast'],
        raw_contrast_dissimilarity_normalized=raw_contrast['normalized_contrast'],
        centered_within_color_dissimilarity=centered_curves['within_color'],
        centered_within_gray_dissimilarity=centered_curves['within_gray'],
        centered_between_color_gray_dissimilarity=centered_curves['between_color_gray'],
        centered_contrast_dissimilarity=centered_contrast['contrast'],
        centered_contrast_dissimilarity_normalized=centered_contrast['normalized_contrast'],
    )

    print(
        {
            'raw_figure_path': str(raw_fig_path),
            'centered_figure_path': str(centered_fig_path),
            'contrast_figure_path': str(contrast_fig_path),
            'summary_path': str(npz_path),
            'source_category_names': paired_dataset.category_names,
            'source_pair_counts': category_counts,
            'aggregated_trial_count': aggregated_trial_count,
            'use_split_half_cross': raw_use_split_half,
            'center_use_split_half': centered_use_split_half,
            'time_smooth_win': int(config.time_smooth_win),
            'decoding_step': int(config.decoding_step),
        }
    )


if __name__ == '__main__':
    main()