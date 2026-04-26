import glob
import math
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from joblib import Parallel, delayed
from scipy.ndimage import label

from groupeddata_pairing import (
    center_paired_trials,
    load_paired_category_trials,
    run_grouped_auc_over_time,
    stack_paired_binary_trials,
)
from newanalyse_paths import (
    get_feature_dir,
    get_task_groupeddata_path,
    get_within_decoding_batch_dir,
    get_within_decoding_task_dir,
    project_root,
)
from runtime_config import load_runtime_config


SUBJECT = 'test001'
BASE_PATH = str(project_root())
FEATURE_KIND = 'erp'
TASK_FIELD = None
GROUPEDDATA_MAT = ''

N_SPLITS = 5
N_REALS = 20
N_REPEATS_REAL = 5
N_REPEATS_PERM = 5
N_PERMS = 100
TIME_SMOOTH_WIN = 5
DECODING_STEP = 5
RANDOM_STATE = 42
N_JOBS = -1

FS = 500
T_START = -100
T_END = 1000
N_POINTS = 550
TIMES = np.linspace(T_START, T_END, N_POINTS)
PLOT_TIMES = TIMES[::DECODING_STEP]

BATCH_NAME = 'batch_within_decoding'
ROI_FILE_PATTERN = 'Color_with*.mat'
RUN_PERMUTATION_TEST = True
ANALYSIS_MODE = 'center'

DEFAULT_CATEGORY_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
DEFAULT_CATEGORY_NAMES = ['face', 'object', 'body', 'scene']

TASK_ID = 'task1_color_vs_gray_cross_category_average'
TASK_TITLE = 'Task 1 Color vs Gray Cross-Category Average Decoding'
TASK_DESCRIPTION = (
    'Match color and gray trials within each of the 4 task1 categories first, then average '
    'matched trials across categories at the same trial index before grouped color-vs-gray decoding.'
)

FEATURE_KIND_ALIASES = {
    'high_gamma': 'highgamma',
    'multi_gamma': 'gamma_multiband',
    'multigamma': 'gamma_multiband',
    'mulitgamma': 'gamma_multiband',
}

FEATURE_CONFIG = {
    'erp': {
        'feature_subdir': 'decoding_erp_features',
        'task_field': 'erp_task1',
        'default_smooth_win': 5,
    },
    'highgamma': {
        'feature_subdir': 'decoding_highgamma_features',
        'task_field': 'hg_task1',
        'default_smooth_win': 5,
    },
    'lowgamma': {
        'feature_subdir': 'decoding_lowgamma_features',
        'task_field': 'lg_task1',
        'default_smooth_win': 5,
    },
    'tfa': {
        'feature_subdir': 'decoding_tfa_features',
        'task_field': 'tfa_task1',
        'default_smooth_win': 5,
    },
    'gamma': {
        'feature_subdir': 'decoding_gamma_features',
        'task_field': 'g_task1',
        'default_smooth_win': 5,
    },
    'gamma_multiband': {
        'feature_subdir': 'decoding_gamma_multiband_features',
        'task_field': 'gmb_task1',
        'default_smooth_win': 20,
    },
}

FEATURE_SUBDIR = FEATURE_CONFIG['erp']['feature_subdir']
FEATURE_DIR = str(get_feature_dir(BASE_PATH, FEATURE_SUBDIR, SUBJECT))


def resolve_feature_kind(feature_kind):
    normalized = FEATURE_KIND_ALIASES.get(str(feature_kind), str(feature_kind))
    if normalized not in FEATURE_CONFIG:
        raise ValueError(f'Unsupported FEATURE_KIND: {feature_kind}')
    return normalized


def refresh_runtime_state():
    global FEATURE_KIND, FEATURE_SUBDIR, FEATURE_DIR, TASK_FIELD, GROUPEDDATA_MAT, TIMES, PLOT_TIMES

    FEATURE_KIND = resolve_feature_kind(FEATURE_KIND)
    feature_cfg = FEATURE_CONFIG[FEATURE_KIND]
    FEATURE_SUBDIR = feature_cfg['feature_subdir']
    if TASK_FIELD in {None, ''}:
        TASK_FIELD = feature_cfg['task_field']
    if not GROUPEDDATA_MAT:
        GROUPEDDATA_MAT = str(get_task_groupeddata_path(BASE_PATH, SUBJECT, 'task1'))

    FEATURE_DIR = str(get_feature_dir(BASE_PATH, FEATURE_SUBDIR, SUBJECT))
    TIMES = np.linspace(T_START, T_END, N_POINTS)
    PLOT_TIMES = TIMES[::DECODING_STEP]


_RUNTIME_CFG = load_runtime_config(__file__, sections=('python_defaults', 'sec3_defaults'))
if _RUNTIME_CFG:
    SUBJECT = str(_RUNTIME_CFG.get('subject', SUBJECT))
    BASE_PATH = str(_RUNTIME_CFG.get('base_path', BASE_PATH))
    FEATURE_KIND = str(_RUNTIME_CFG.get('feature_kind', FEATURE_KIND))
    TASK_FIELD = _RUNTIME_CFG.get('task_field', TASK_FIELD)
    GROUPEDDATA_MAT = str(_RUNTIME_CFG.get('groupeddata_mat', GROUPEDDATA_MAT or ''))
    N_SPLITS = int(_RUNTIME_CFG.get('n_splits', N_SPLITS))
    N_REALS = int(_RUNTIME_CFG.get('n_reals', N_REALS))
    N_REPEATS_REAL = int(_RUNTIME_CFG.get('n_repeats_real', N_REPEATS_REAL))
    N_REPEATS_PERM = int(_RUNTIME_CFG.get('n_repeats_perm', N_REPEATS_PERM))
    N_PERMS = int(_RUNTIME_CFG.get('n_perms', N_PERMS))
    TIME_SMOOTH_WIN = int(_RUNTIME_CFG.get('time_smooth_win', TIME_SMOOTH_WIN))
    DECODING_STEP = int(_RUNTIME_CFG.get('decoding_step', DECODING_STEP))
    RANDOM_STATE = int(_RUNTIME_CFG.get('random_state', RANDOM_STATE))
    N_JOBS = int(_RUNTIME_CFG.get('n_jobs', N_JOBS))
    BATCH_NAME = str(_RUNTIME_CFG.get('batch_name', BATCH_NAME))
    ROI_FILE_PATTERN = str(_RUNTIME_CFG.get('roi_pattern', ROI_FILE_PATTERN))
    RUN_PERMUTATION_TEST = bool(_RUNTIME_CFG.get('run_permutation_test', RUN_PERMUTATION_TEST))
    ANALYSIS_MODE = str(_RUNTIME_CFG.get('analysis_mode', ANALYSIS_MODE))

refresh_runtime_state()


def should_run_permutation_test():
    return bool(RUN_PERMUTATION_TEST and N_PERMS > 0)


def get_perm_tag():
    return f'perm{N_PERMS}' if should_run_permutation_test() else 'real_only'


def resolve_analysis_mode():
    mode = str(ANALYSIS_MODE).strip().lower()
    if mode not in {'raw', 'center'}:
        raise ValueError(f'ANALYSIS_MODE must be raw or center, got: {ANALYSIS_MODE}')
    return mode


def make_logger(log_path):
    def _log(msg):
        text = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(text)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
    return _log


def baseline_zscore(samples, baseline_end_idx):
    baseline = samples[:, :, :baseline_end_idx]
    mean = baseline.mean(axis=2, keepdims=True)
    std = baseline.std(axis=2, keepdims=True) + 1e-8
    return (samples - mean) / std


def smooth_data_causal(samples, win_size):
    if win_size <= 0:
        return samples.copy()
    smoothed = np.zeros_like(samples)
    for time_idx in range(samples.shape[2]):
        start_idx = max(0, time_idx - win_size)
        smoothed[:, :, time_idx] = np.mean(samples[:, :, start_idx:time_idx + 1], axis=2)
    return smoothed


def build_cross_category_average_dataset(paired_categories):
    source_counts = {paired.category_name: int(paired.matched_count) for paired in paired_categories}
    min_count = min(int(paired.matched_count) for paired in paired_categories)
    if min_count < 1:
        raise ValueError('At least one matched trial is required in every category.')

    color_stack = np.stack([paired.color[:min_count] for paired in paired_categories], axis=0)
    gray_stack = np.stack([paired.gray[:min_count] for paired in paired_categories], axis=0)
    averaged_color = np.mean(color_stack, axis=0)
    averaged_gray = np.mean(gray_stack, axis=0)
    pair_ids = np.arange(min_count, dtype=int)
    return averaged_color, averaged_gray, pair_ids, source_counts, int(min_count)


def prepare_aggregate_samples(data):
    paired_categories = load_paired_category_trials(
        BASE_PATH,
        SUBJECT,
        FEATURE_KIND,
        'task1',
        GROUPEDDATA_MAT,
        data,
        DEFAULT_CATEGORY_PAIRS,
        DEFAULT_CATEGORY_NAMES,
    )
    averaged_color, averaged_gray, pair_ids, source_counts, aggregated_trial_count = build_cross_category_average_dataset(paired_categories)

    baseline_end = np.searchsorted(TIMES, 0)
    stacked = np.concatenate([averaged_color, averaged_gray], axis=0)
    stacked = baseline_zscore(stacked, baseline_end)
    if TIME_SMOOTH_WIN > 0:
        stacked = smooth_data_causal(stacked, TIME_SMOOTH_WIN)

    pair_count = averaged_color.shape[0]
    color_samples = stacked[:pair_count]
    gray_samples = stacked[pair_count:]
    analysis_mode = resolve_analysis_mode()
    if analysis_mode == 'center':
        color_samples, gray_samples = center_paired_trials(color_samples, gray_samples)

    samples, labels, groups = stack_paired_binary_trials(color_samples, gray_samples, pair_ids)
    return samples, labels, groups, source_counts, aggregated_trial_count, analysis_mode


def run_real_distribution(samples, labels, groups):
    def one_real(real_seed):
        scores = run_grouped_auc_over_time(
            samples,
            labels,
            groups,
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS_REAL,
            decoding_step=DECODING_STEP,
            seed=real_seed,
            shuffle=False,
        )
        return np.mean(scores, axis=1)

    return np.asarray(
        Parallel(n_jobs=N_JOBS)(
            delayed(one_real)(RANDOM_STATE + run_idx) for run_idx in range(N_REALS)
        ),
        dtype=float,
    )


def run_permutation_distribution(samples, labels, groups, n_timepoints):
    if not should_run_permutation_test():
        return np.full((0, n_timepoints), np.nan)

    def one_perm(perm_seed):
        scores = run_grouped_auc_over_time(
            samples,
            labels,
            groups,
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS_PERM,
            decoding_step=DECODING_STEP,
            seed=perm_seed,
            shuffle=True,
        )
        return np.mean(scores, axis=1)

    return np.asarray(
        Parallel(n_jobs=N_JOBS)(
            delayed(one_perm)(RANDOM_STATE + 100000 + perm_idx) for perm_idx in range(N_PERMS)
        ),
        dtype=float,
    )


def cluster_permutation_significance(mean_auc, perm_dist):
    if perm_dist is None or np.size(perm_dist) == 0:
        return np.full(mean_auc.shape, np.nan, dtype=float), np.zeros_like(mean_auc, dtype=bool)

    threshold_95 = np.percentile(perm_dist, 95, axis=0)
    valid_mask = PLOT_TIMES[:len(mean_auc)] > 20
    binary_map = (mean_auc > threshold_95) & valid_mask
    clusters, n_clusters = label(binary_map.astype(int))

    cluster_masses = []
    for cluster_id in range(1, n_clusters + 1):
        indices = clusters == cluster_id
        cluster_masses.append(np.sum(mean_auc[indices] - threshold_95[indices]))

    null_cluster_masses = []
    for perm_curve in perm_dist:
        perm_binary = (perm_curve > threshold_95) & valid_mask
        perm_clusters, perm_count = label(perm_binary.astype(int))
        max_mass = 0.0
        for cluster_id in range(1, perm_count + 1):
            indices = perm_clusters == cluster_id
            mass = float(np.sum(perm_curve[indices] - threshold_95[indices]))
            if mass > max_mass:
                max_mass = mass
        null_cluster_masses.append(max_mass)

    critical_mass = np.percentile(null_cluster_masses, 95) if null_cluster_masses else np.inf
    sig_indices = np.zeros_like(mean_auc, dtype=bool)
    for cluster_id in range(1, n_clusters + 1):
        if cluster_masses[cluster_id - 1] > critical_mass:
            sig_indices[clusters == cluster_id] = True
    return threshold_95, sig_indices


def compute_latency_points(mean_auc, sig_indices):
    latencies = {'earliest': np.nan, 'half_height': np.nan, 'peak': np.nan}
    if not np.any(sig_indices):
        return latencies

    sig_idx = np.where(sig_indices)[0]
    earliest_idx = int(sig_idx[0])
    masked_auc = np.where(sig_indices, mean_auc, -np.inf)
    peak_idx = int(np.argmax(masked_auc))

    clusters, _ = label(sig_indices.astype(int))
    first_cluster_id = clusters[earliest_idx]
    first_cluster_indices = np.where(clusters == first_cluster_id)[0]
    first_peak_idx = int(first_cluster_indices[np.argmax(mean_auc[first_cluster_indices])])
    first_peak_val = float(mean_auc[first_peak_idx])
    half_level = 0.5 + (first_peak_val - 0.5) / 2.0
    pre_peak_indices = first_cluster_indices[first_cluster_indices <= first_peak_idx]
    half_candidates = pre_peak_indices[mean_auc[pre_peak_indices] >= half_level]
    half_idx = int(half_candidates[0]) if half_candidates.size > 0 else first_peak_idx

    latencies['earliest'] = float(PLOT_TIMES[min(earliest_idx, len(PLOT_TIMES) - 1)])
    latencies['half_height'] = float(PLOT_TIMES[min(half_idx, len(PLOT_TIMES) - 1)])
    latencies['peak'] = float(PLOT_TIMES[min(peak_idx, len(PLOT_TIMES) - 1)])
    return latencies


def plot_single_roi_result(roi_name, mean_auc, sem_auc, threshold_95, sig_indices, figure_title, save_path):
    plot_times = PLOT_TIMES[:len(mean_auc)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(plot_times, mean_auc - sem_auc, mean_auc + sem_auc, color='#1f77b4', alpha=0.25)
    ax.plot(plot_times, mean_auc, color='#1f77b4', linewidth=1.8, label='Mean ROC AUC')
    if np.any(np.isfinite(threshold_95)):
        ax.plot(plot_times, threshold_95, color='#d62728', linestyle='--', linewidth=1.2, label='95% permutation threshold')
    if np.any(sig_indices):
        ax.fill_between(plot_times, 0, 1, where=sig_indices[:len(plot_times)], color='gray', alpha=0.25, transform=ax.get_xaxis_transform(), label='Significant cluster')
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1.0, label='Chance')
    ax.set_title(f'{figure_title}\nROI: {roi_name}', fontsize=11)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('ROC AUC')
    ax.set_xlim(T_START, T_END)
    ax.set_ylim(0.35, 1.0)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_grid_figures(results_dict, title_text, save_path):
    n_rois = len(results_dict)
    if n_rois == 0:
        return
    n_cols = min(5, n_rois)
    n_rows = math.ceil(n_rois / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.8, n_rows * 2.8), squeeze=False)
    fig.suptitle(f'{SUBJECT} | {title_text}', fontsize=14, y=1.02)
    flat_axes = axes.flatten()
    for idx, (roi, res) in enumerate(results_dict.items()):
        ax = flat_axes[idx]
        mean_auc, sem_auc, threshold_95, sig_indices = res
        plot_times = PLOT_TIMES[:len(mean_auc)]
        ax.fill_between(plot_times, mean_auc - sem_auc, mean_auc + sem_auc, color='#1f77b4', alpha=0.25)
        ax.plot(plot_times, mean_auc, color='#1f77b4', linewidth=1.5)
        if np.any(np.isfinite(threshold_95)):
            ax.plot(plot_times, threshold_95, color='#d62728', linestyle='--', linewidth=1.0)
        if np.any(sig_indices):
            ax.fill_between(plot_times, 0, 1, where=sig_indices[:len(plot_times)], color='gray', alpha=0.25, transform=ax.get_xaxis_transform())
        ax.axhline(0.5, color='black', linestyle=':', linewidth=0.8)
        ax.set_title(roi, fontsize=9)
        ax.set_xlim(T_START, T_END)
        ax.set_ylim(0.35, 1.0)
        ax.grid(True, linestyle='--', alpha=0.35)
        if idx % n_cols == 0:
            ax.set_ylabel('ROC AUC', fontsize=8)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Time (ms)', fontsize=8)
    for ax in flat_axes[n_rois:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_latency_bar_scatter(latency_dict, title_text, y_label, save_path):
    clean_items = [(k, v) for k, v in latency_dict.items() if not np.isnan(v)]
    if not clean_items:
        return
    clean_items.sort(key=lambda x: x[1])
    rois = [x[0] for x in clean_items]
    latencies = [x[1] for x in clean_items]
    fig, ax = plt.subplots(figsize=(max(9, len(rois) * 0.7), 6))
    x_pos = np.arange(len(rois))
    bars = ax.bar(x_pos, latencies, color='#9ecae1', edgecolor='black', linewidth=1.0, alpha=0.9, width=0.62)
    ax.scatter(x_pos, latencies, color='#d62728', s=55, zorder=3, edgecolor='white', linewidth=1.0)
    for bar, latency in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, latency + (T_END - T_START) * 0.01, f'{latency:.0f}ms', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rois, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f'{SUBJECT} | {title_text}', fontsize=12)
    ax.set_ylim(T_START, max(latencies) * 1.15 if max(latencies) > 0 else T_END)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_task_for_roi(fpath, roi_name, save_path, roi_plot_dir, logger):
    mat = sio.loadmat(fpath)
    if TASK_FIELD not in mat:
        raise ValueError(f'Missing matrix: {TASK_FIELD}')
    data = np.asarray(mat[TASK_FIELD], dtype=float)

    samples, labels, groups, source_counts, aggregated_trial_count, analysis_mode = prepare_aggregate_samples(data)
    real_auc_runs = run_real_distribution(samples, labels, groups)
    mean_auc = np.mean(real_auc_runs, axis=0)
    sem_auc = np.std(real_auc_runs, axis=0, ddof=0) / np.sqrt(real_auc_runs.shape[0])
    perm_dist = run_permutation_distribution(samples, labels, groups, mean_auc.shape[0])
    threshold_95, sig_indices = cluster_permutation_significance(mean_auc, perm_dist)
    latencies = compute_latency_points(mean_auc, sig_indices)

    np.savez(
        save_path,
        mean_auc=mean_auc,
        sem_auc=sem_auc,
        threshold_95=threshold_95,
        sig_indices=sig_indices,
        real_auc_runs=real_auc_runs,
        perm_dist=perm_dist,
        latency_earliest=latencies['earliest'],
        latency_half_height=latencies['half_height'],
        latency_peak=latencies['peak'],
        task_id=np.array(TASK_ID),
        task_title=np.array(TASK_TITLE),
        task_description=np.array(TASK_DESCRIPTION),
        task_name=np.array('task1'),
        feature_kind=np.array(FEATURE_KIND),
        task_field=np.array(TASK_FIELD),
        analysis_mode=np.array(analysis_mode),
        n_real=np.array(N_REALS),
        n_repeats_real=np.array(N_REPEATS_REAL),
        n_repeats_perm=np.array(N_REPEATS_PERM),
        n_perm=np.array(N_PERMS),
        groupeddata_mat=np.array(str(GROUPEDDATA_MAT)),
        category_names=np.asarray(DEFAULT_CATEGORY_NAMES, dtype=object),
        matched_pair_count_names=np.asarray(list(source_counts.keys()), dtype=object),
        matched_pair_count_values=np.asarray(list(source_counts.values()), dtype=int),
        aggregated_trial_count=np.array(aggregated_trial_count),
    )

    plot_single_roi_result(
        roi_name=roi_name,
        mean_auc=mean_auc,
        sem_auc=sem_auc,
        threshold_95=threshold_95,
        sig_indices=sig_indices,
        figure_title=f'{TASK_TITLE} ({analysis_mode})',
        save_path=os.path.join(roi_plot_dir, f'{roi_name}_curve.png'),
    )
    logger(
        f'ROI saved: {roi_name} | mode={analysis_mode} | aggregated_trial_count={aggregated_trial_count} '
        f'| source_counts={source_counts}'
    )


def generate_summary_figures(cache_dir, output_dir, logger):
    result_files = sorted(glob.glob(os.path.join(cache_dir, '*_results.npz')))
    if not result_files:
        logger(f'No result files in {cache_dir}')
        return {'all': {}, 'significant': {}}

    all_results = {}
    sig_results = {}
    latency_earliest = {}
    latency_half = {}
    latency_peak = {}

    for fpath in result_files:
        roi_name = os.path.basename(fpath).replace('_results.npz', '')
        data = np.load(fpath)
        mean_auc = data['mean_auc']
        sem_auc = data['sem_auc']
        threshold_95 = data['threshold_95']
        sig_indices = data['sig_indices']
        all_results[roi_name] = (mean_auc, sem_auc, threshold_95, sig_indices)
        if np.any(sig_indices):
            sig_results[roi_name] = (mean_auc, sem_auc, threshold_95, sig_indices)
            latency_earliest[roi_name] = float(data['latency_earliest'])
            latency_half[roi_name] = float(data['latency_half_height'])
            latency_peak[roi_name] = float(data['latency_peak'])

    plot_grid_figures(
        results_dict=all_results,
        title_text=f'{TASK_TITLE} | {TASK_DESCRIPTION}',
        save_path=os.path.join(output_dir, 'Fig_All_ROIs_Overview.png'),
    )
    if sig_results:
        plot_grid_figures(
            results_dict=sig_results,
            title_text=f'{TASK_TITLE} | Significant ROIs',
            save_path=os.path.join(output_dir, 'Fig_Significant_ROIs_Only.png'),
        )
        plot_latency_bar_scatter(
            latency_dict=latency_earliest,
            title_text=f'{TASK_TITLE} | Earliest Significant Latency',
            y_label='Earliest Significant Latency (ms)',
            save_path=os.path.join(output_dir, 'Fig_Latency_Earliest.png'),
        )
        plot_latency_bar_scatter(
            latency_dict=latency_half,
            title_text=f'{TASK_TITLE} | First-Peak Half-Height Latency',
            y_label='First-Peak Half-Height Latency (ms)',
            save_path=os.path.join(output_dir, 'Fig_Latency_FirstPeakHalfHeight.png'),
        )
        plot_latency_bar_scatter(
            latency_dict=latency_peak,
            title_text=f'{TASK_TITLE} | Peak Latency',
            y_label='Peak Latency (ms)',
            save_path=os.path.join(output_dir, 'Fig_Latency_Peak.png'),
        )
        logger(f'Significant ROI count: {len(sig_results)}')
    else:
        logger(f'No significant ROI found for task: {TASK_ID}')

    return {'all': all_results, 'significant': sig_results}


def main():
    refresh_runtime_state()
    analysis_mode = resolve_analysis_mode()

    mat_files = sorted(glob.glob(os.path.join(FEATURE_DIR, ROI_FILE_PATTERN)))
    if not mat_files:
        print(f'No ROI file found: {FEATURE_DIR}')
        return

    variant = f'with_sti_{analysis_mode}'
    batch_root = str(get_within_decoding_batch_dir(BASE_PATH, FEATURE_KIND, SUBJECT, batch_name=BATCH_NAME))
    os.makedirs(batch_root, exist_ok=True)
    log_path = os.path.join(batch_root, f'{TASK_ID}_{analysis_mode}_batch_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logger = make_logger(log_path)

    output_dir = str(get_within_decoding_task_dir(BASE_PATH, TASK_ID, FEATURE_KIND, SUBJECT, get_perm_tag(), variant=variant, batch_name=BATCH_NAME))
    cache_dir = os.path.join(output_dir, 'computed_results')
    roi_plot_dir = os.path.join(output_dir, 'roi_curves')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(roi_plot_dir, exist_ok=True)

    logger(f'Start cross-category average decoding for {SUBJECT}')
    logger(f'Feature kind: {FEATURE_KIND}')
    logger(f'Feature dir: {FEATURE_DIR}')
    logger(f'GroupedData: {GROUPEDDATA_MAT}')
    logger(f'Analysis mode: {analysis_mode}')
    logger(f'ROI count: {len(mat_files)}')
    logger(f'n_real={N_REALS}, n_repeats_real={N_REPEATS_REAL}, n_repeats_perm={N_REPEATS_PERM}, n_perm={N_PERMS}')

    for fpath in mat_files:
        roi_name = os.path.splitext(os.path.basename(fpath))[0]
        save_path = os.path.join(cache_dir, f'{roi_name}_results.npz')
        logger(f'Processing ROI: {roi_name}')
        try:
            run_task_for_roi(fpath, roi_name, save_path, roi_plot_dir, logger)
        except Exception as exc:
            logger(f'ROI failed: {roi_name} | {exc}')

    generate_summary_figures(cache_dir, output_dir, logger)
    logger('Cross-category average decoding completed')


if __name__ == '__main__':
    _script_start_time = time.time()
    try:
        main()
    finally:
        print(f'Total runtime: {time.time() - _script_start_time:.2f} s')