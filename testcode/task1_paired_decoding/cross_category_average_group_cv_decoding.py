import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from common import (
    PairedDataset,
    build_output_dir,
    load_preprocessed_paired_dataset,
    normalize_common_args,
    paired_counts_by_category,
    resolve_selected_category_indices,
    run_decoding_result,
    save_curve_figure,
    select_paired_dataset_categories,
    stack_raw_samples,
)


SUBJECT = 'test001'
FEATURE_KIND = 'erp'
ROI_NAME = 'Color_with_sti'
GROUPED_DATA_MAT = str(Path(__file__).resolve().parents[2] / 'processed_data' / SUBJECT / 'groupedData.mat')
TASK_FIELD = None
METRIC = 'auc'
TIME_SMOOTH_WIN = 5
DECODING_STEP = 5
N_SPLITS = 5
N_REPEATS = 10
N_REPEATS_PERM = 10
N_PERMS = 100
RANDOM_STATE = 42
N_JOBS = -1
OUTPUT_TAG = ''
SELECTED_CATEGORIES = 'all'
ANALYSIS_MODE = 'center'
COLOR_CONDITION_INDICES = '0,4,6'
GRAY_CONDITION_INDICES = '1,5,7'
CATEGORY_NAMES = 'face,object,scene'


SCHEME_NAME = 'cross_category_average_group_cv'
SCHEME_TITLE = 'Task1 cross-category average color-vs-gray decoding'
SCHEME_NOTE = (
    'Match color and gray trials within each selected category first, then average the matched trials across categories '
    'at the same trial index to form a new paired color-gray dataset for grouped cross-validation decoding.'
)


def build_script_config():
    grouped_data_mat = GROUPED_DATA_MAT
    if not grouped_data_mat:
        grouped_data_mat = str(Path(__file__).resolve().parents[2] / 'processed_data' / SUBJECT / 'groupedData.mat')
    args = argparse.Namespace(
        subject=SUBJECT,
        feature_kind=FEATURE_KIND,
        roi_name=ROI_NAME,
        grouped_data_mat=grouped_data_mat,
        task_field=TASK_FIELD,
        metric=METRIC,
        time_smooth_win=TIME_SMOOTH_WIN,
        color_condition_indices=COLOR_CONDITION_INDICES,
        gray_condition_indices=GRAY_CONDITION_INDICES,
        category_names=CATEGORY_NAMES,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        n_repeats_perm=N_REPEATS_PERM,
        n_perms=N_PERMS,
        decoding_step=DECODING_STEP,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        output_tag=OUTPUT_TAG,
    )
    return normalize_common_args(args)


def center_aggregate_dataset(aggregate_dataset):
    pair_mean = 0.5 * (aggregate_dataset.color + aggregate_dataset.gray)
    return replace(
        aggregate_dataset,
        color=aggregate_dataset.color - pair_mean,
        gray=aggregate_dataset.gray - pair_mean,
    )


def resolve_analysis_mode():
    mode = str(ANALYSIS_MODE).strip().lower()
    if mode not in {'raw', 'center'}:
        raise ValueError(f'ANALYSIS_MODE must be raw or center, got: {ANALYSIS_MODE}')
    return mode


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


def save_cross_category_average_summary(
    save_path,
    config,
    roi_bundle,
    aggregate_dataset,
    plot_times,
    result,
    source_dataset,
    selected_category_indices,
    source_counts,
    aggregated_trial_count,
    analysis_mode,
):
    np.savez(
        save_path,
        subject=np.array(config.subject),
        feature_kind=np.array(config.feature_kind),
        roi_name=np.array(config.roi_name),
        grouped_data_mat=np.array(str(config.grouped_data_mat)),
        roi_path=np.array(str(roi_bundle.roi_path)),
        task_field=np.array(roi_bundle.task_field),
        metric_name=np.array(config.metric_name),
        analysis_mode=np.array(analysis_mode),
        scheme_name=np.array(SCHEME_NAME),
        scheme_note=np.array(SCHEME_NOTE),
        time_smooth_win=np.array(config.time_smooth_win),
        decoding_step=np.array(config.decoding_step),
        n_splits=np.array(config.n_splits),
        n_perms=np.array(config.n_perms),
        n_repeats_real=np.array(config.n_repeats),
        n_repeats_perm=np.array(config.n_repeats_perm),
        selected_category_indices=np.asarray(selected_category_indices, dtype=int),
        source_category_names=np.asarray(source_dataset.category_names, dtype=object),
        source_color_condition_indices=np.asarray([config.condition_pairs[idx][0] for idx in selected_category_indices], dtype=int),
        source_gray_condition_indices=np.asarray([config.condition_pairs[idx][1] for idx in selected_category_indices], dtype=int),
        source_matched_count_names=np.asarray(list(source_counts.keys()), dtype=object),
        source_matched_count_values=np.asarray(list(source_counts.values()), dtype=int),
        aggregate_category_name=np.asarray(aggregate_dataset.category_names, dtype=object),
        aggregated_trial_count=np.array(aggregated_trial_count),
        plot_times=plot_times,
        curve=result['curve'],
        sem=result['sem'],
        perm_dist=result['perm_dist'],
        threshold_95=result['threshold_95'],
        sig_mask=result['sig_mask'],
        chance_level=np.array(result['chance_level']),
        latency_earliest=np.array(result['latencies']['earliest']),
        latency_half_height=np.array(result['latencies']['half_height']),
        latency_peak=np.array(result['latencies']['peak']),
        pair_ids=aggregate_dataset.pair_ids,
        category_ids=aggregate_dataset.category_ids,
        sample_keys=np.asarray(aggregate_dataset.sample_keys, dtype=object),
        color_repeat_indices=aggregate_dataset.color_repeat_indices,
        gray_repeat_indices=aggregate_dataset.gray_repeat_indices,
    )


def execute_cross_category_average_decoding(config, selected_categories):
    start_time = time.time()
    analysis_mode = resolve_analysis_mode()
    roi_bundle, paired_dataset = load_preprocessed_paired_dataset(config)
    selected_category_indices = resolve_selected_category_indices(selected_categories, config.category_names)
    paired_dataset = select_paired_dataset_categories(paired_dataset, selected_category_indices)
    aggregate_dataset, source_counts, aggregated_trial_count = build_cross_category_average_dataset(paired_dataset)
    if analysis_mode == 'center':
        aggregate_dataset = center_aggregate_dataset(aggregate_dataset)

    print('Final matched trial counts by category:')
    for category_name, count in source_counts.items():
        print(f'  {category_name}: {count}')
    print(f'Cross-category averaged trial count: {aggregated_trial_count}')
    print(f'Analysis mode: {analysis_mode}')

    x, y, groups, split_labels = stack_raw_samples(aggregate_dataset)
    plot_times = roi_bundle.time_vector[::config.decoding_step][:len(np.arange(0, x.shape[2], config.decoding_step))]
    result = run_decoding_result(x, y, groups, split_labels, config, plot_times)
    plot_times = plot_times[:result['curve'].shape[0]]

    out_dir = build_output_dir(SCHEME_NAME, config)
    fig_path = out_dir / f'{SCHEME_NAME}_{analysis_mode}_curve.png'
    npz_path = out_dir / f'{SCHEME_NAME}_{analysis_mode}_curve.npz'
    subtitle_lines = [
        f'Subject: {config.subject}',
        f'Feature: {config.feature_kind}',
        f'ROI: {config.roi_name}',
        f'Analysis mode: {analysis_mode}',
        f'Source counts: {source_counts}',
        f'Aggregated trials: {aggregated_trial_count}',
        f'Permutations: {config.n_perms} | Perm repeats: {config.n_repeats_perm}',
    ]
    save_curve_figure(fig_path, plot_times, result, config.metric_name, f'{SCHEME_TITLE} | {analysis_mode}', subtitle_lines=subtitle_lines)
    save_cross_category_average_summary(
        npz_path,
        config,
        roi_bundle,
        aggregate_dataset,
        plot_times,
        result,
        paired_dataset,
        selected_category_indices,
        source_counts,
        aggregated_trial_count,
        analysis_mode,
    )

    return {
        'scheme_name': SCHEME_NAME,
        'scheme_note': SCHEME_NOTE,
        'subject': config.subject,
        'feature_kind': config.feature_kind,
        'roi_name': config.roi_name,
        'grouped_data_mat': str(config.grouped_data_mat),
        'roi_path': str(roi_bundle.roi_path),
        'task_field': roi_bundle.task_field,
        'metric_name': config.metric_name,
        'analysis_mode': analysis_mode,
        'selected_category_names': paired_dataset.category_names,
        'matched_counts_by_category': source_counts,
        'aggregated_trial_count': aggregated_trial_count,
        'earliest_sig_ms': result['latencies']['earliest'],
        'peak_sig_ms': result['latencies']['peak'],
        'figure_path': str(fig_path),
        'summary_path': str(npz_path),
        'runtime_s': time.time() - start_time,
    }


def main():
    config = build_script_config()
    summary = execute_cross_category_average_decoding(config, SELECTED_CATEGORIES)
    print(summary)


if __name__ == '__main__':
    main()