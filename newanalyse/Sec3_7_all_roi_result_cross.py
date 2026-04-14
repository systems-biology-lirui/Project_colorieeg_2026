import os
import glob
import math
from datetime import datetime
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from scipy.ndimage import label

from newanalyse_paths import (
    get_cross_decoding_batch_dir,
    get_cross_decoding_task_dir,
    get_feature_dir,
    project_root,
)

SUBJECT = 'test001'
BASE_PATH = str(project_root())
FEATURE_DIR = str(get_feature_dir(BASE_PATH, 'decoding_lowgamma_features', SUBJECT))

N_REPEATS_REAL = 10
N_REPEATS_PERM = N_REPEATS_REAL
N_PERMS = 200
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

BATCH_NAME = 'batch_cross_decoding'

CROSS_TASKS = [
    {
        'id': 'task3_to_task2_gray_memory',
        'title': 'Task 3 Pure Color to Task 2 Gray Memory-Color Decoding',
        'description': 'Train on Task 3 condition 1 vs 4, test on Task 2 gray fruits with memory color green vs red.',
        'train_key': 'lg_task3',
        'train_class0': [0],
        'train_class1': [3],
        'test_key': 'lg_task2',
        'test_class0': [2, 5],
        'test_class1': [8, 11],
        'folder': 'task3_to_task2_gray_memory_smooth20'
    }
]


def main():
    mat_files = sorted(glob.glob(os.path.join(FEATURE_DIR, '*.mat')))
    if not mat_files:
        print(f'No ROI file found: {FEATURE_DIR}')
        return

    batch_root = str(get_cross_decoding_batch_dir(BASE_PATH, 'lowgamma', SUBJECT, batch_name=BATCH_NAME))
    os.makedirs(batch_root, exist_ok=True)
    log_path = os.path.join(batch_root, f'batch_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logger = make_logger(log_path)

    logger(f'Start cross-decoding batch for {SUBJECT}')
    logger(f'Feature dir: {FEATURE_DIR}')
    logger(f'ROI count: {len(mat_files)}')

    for task in CROSS_TASKS:
        output_dir = str(get_cross_decoding_task_dir(BASE_PATH, task['id'], 'lowgamma', SUBJECT, 'perm1000', batch_name=BATCH_NAME))
        cache_dir = os.path.join(output_dir, 'computed_results')
        roi_plot_dir = os.path.join(output_dir, 'roi_curves')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(roi_plot_dir, exist_ok=True)
        logger(f'===== {task["id"]} =====')
        logger(task['title'])
        logger(task['description'])

        for fpath in mat_files:
            roi_name = os.path.splitext(os.path.basename(fpath))[0]
            save_path = os.path.join(cache_dir, f'{roi_name}_results.npz')
            logger(f'Processing ROI: {roi_name}')
            try:
                process_cross_task_roi(fpath, roi_name, task, save_path, roi_plot_dir, logger)
            except Exception as exc:
                logger(f'ROI failed: {roi_name} | {exc}')

        generate_summary_figures(task, cache_dir, output_dir, logger)

    logger('Cross-decoding batch completed')


def make_logger(log_path):
    def _log(msg):
        text = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(text)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
    return _log


def process_cross_task_roi(fpath, roi_name, task, save_path, roi_plot_dir, logger):
    X_train_raw, y_train, X_test_raw, y_test = extract_cross_task_data(fpath, task)
    
    # ✅ 在这里加，平滑之前，train和test分别做各自的基线校正
    baseline_end = np.searchsorted(TIMES, 0)   # t=0 对应索引 = 50
    X_train_raw = baseline_zscore(X_train_raw, baseline_end)
    X_test_raw  = baseline_zscore(X_test_raw,  baseline_end)
    
    if TIME_SMOOTH_WIN > 0:
        X_train = smooth_data_causal(X_train_raw, TIME_SMOOTH_WIN)
        X_test = smooth_data_causal(X_test_raw, TIME_SMOOTH_WIN)
    else:
        X_train, X_test = X_train_raw, X_test_raw

    real_auc_matrix = run_cross_decoding_over_time(X_train, y_train, X_test, y_test, n_repeats=N_REPEATS_REAL, shuffle=False, seed=RANDOM_STATE)
    perm_dist = Parallel(n_jobs=N_JOBS)(
        delayed(run_cross_decoding_over_time_mean)(
            X_train, y_train, X_test, y_test,
            n_repeats=N_REPEATS_PERM,
            shuffle=True,
            seed=i
        ) for i in range(N_PERMS)
    )
    perm_dist = np.array(perm_dist)

    mean_auc = np.mean(real_auc_matrix, axis=1)
    sem_auc = np.std(real_auc_matrix, axis=1, ddof=0) / np.sqrt(real_auc_matrix.shape[1])
    threshold_95, sig_indices = cluster_permutation_significance(mean_auc, perm_dist)
    latencies = compute_latency_points(mean_auc, sig_indices)

    np.savez(
        save_path,
        mean_auc=mean_auc,
        sem_auc=sem_auc,
        threshold_95=threshold_95,
        sig_indices=sig_indices,
        real_auc_matrix=real_auc_matrix,
        perm_dist=perm_dist,
        latency_earliest=latencies['earliest'],
        latency_half_height=latencies['half_height'],
        latency_peak=latencies['peak'],
        task_id=np.array(task['id']),
        task_title=np.array(task['title'])
    )
    plot_single_roi_result(
        roi_name=roi_name,
        mean_auc=mean_auc,
        sem_auc=sem_auc,
        threshold_95=threshold_95,
        sig_indices=sig_indices,
        figure_title=f'{task["title"]} | {task["description"]}',
        save_path=os.path.join(roi_plot_dir, f'{roi_name}_curve.png')
    )
    logger(f'ROI saved: {roi_name}')


def extract_cross_task_data(fpath, task):
    mat = sio.loadmat(fpath)
    train_key = resolve_feature_key(mat, task['train_key'])
    test_key = resolve_feature_key(mat, task['test_key'])
    train_data = mat[train_key]
    test_data = mat[test_key]
    X_train_0 = np.concatenate([train_data[idx, :, :, :] for idx in task['train_class0']], axis=0)
    X_train_1 = np.concatenate([train_data[idx, :, :, :] for idx in task['train_class1']], axis=0)
    X_test_0 = np.concatenate([test_data[idx, :, :, :] for idx in task['test_class0']], axis=0)
    X_test_1 = np.concatenate([test_data[idx, :, :, :] for idx in task['test_class1']], axis=0)

    X_train = np.concatenate([X_train_0, X_train_1], axis=0)
    y_train = np.concatenate([np.zeros(X_train_0.shape[0]), np.ones(X_train_1.shape[0])])
    X_test = np.concatenate([X_test_0, X_test_1], axis=0)
    y_test = np.concatenate([np.zeros(X_test_0.shape[0]), np.ones(X_test_1.shape[0])])
    return X_train, y_train, X_test, y_test


def resolve_feature_key(mat, preferred_key):
    candidates = [preferred_key]
    if preferred_key.startswith('lg_'):
        candidates.append(preferred_key.replace('lg_', 'hg_', 1))
    elif preferred_key.startswith('hg_'):
        candidates.append(preferred_key.replace('hg_', 'lg_', 1))

    for key in candidates:
        if key in mat:
            return key
    raise ValueError(f'Missing data key in file: {preferred_key}')


def run_cross_decoding_over_time(X_train, y_train, X_test, y_test, n_repeats=1, shuffle=False, seed=None):
    rng = np.random.RandomState(seed if seed is not None else RANDOM_STATE)
    y_train_use = rng.permutation(y_train) if shuffle else y_train.copy()
    n_time = X_train.shape[2]
    time_indices = np.arange(0, n_time, DECODING_STEP)
    scores = np.zeros((len(time_indices), n_repeats))
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    n_test_samples = len(y_test)

    for ti, t in enumerate(time_indices):
        clf.fit(X_train[:, :, t], y_train_use)
        y_prob = clf.predict_proba(X_test[:, :, t])[:, 1]
        if n_repeats == 1:
            scores[ti, 0] = safe_auc(y_test, y_prob)
        else:
            for ri in range(n_repeats):
                boot_idx = rng.randint(0, n_test_samples, n_test_samples)
                y_boot = y_test[boot_idx]
                p_boot = y_prob[boot_idx]
                scores[ti, ri] = safe_auc(y_boot, p_boot)
    return scores


def run_cross_decoding_over_time_mean(X_train, y_train, X_test, y_test, n_repeats=1, shuffle=False, seed=None):
    return np.mean(run_cross_decoding_over_time(X_train, y_train, X_test, y_test, n_repeats=n_repeats, shuffle=shuffle, seed=seed), axis=1)


def safe_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5


def smooth_data_causal(X, win_size):
    n_samples, n_ch, n_time = X.shape
    X_smooth = np.zeros_like(X)
    for t in range(n_time):
        t_start = max(0, t - win_size)
        X_smooth[:, :, t] = np.mean(X[:, :, t_start:t + 1], axis=2)
    return X_smooth


def cluster_permutation_significance(mean_auc, perm_dist):
    threshold_95 = np.percentile(perm_dist, 95, axis=0)
    binary_map = mean_auc > threshold_95
    clusters, n_clusters = label(binary_map.astype(int))
    cluster_masses = []
    for ci in range(1, n_clusters + 1):
        idx = clusters == ci
        cluster_masses.append(np.sum(mean_auc[idx] - threshold_95[idx]))

    null_cluster_masses = []
    for pi in range(perm_dist.shape[0]):
        curve = perm_dist[pi]
        p_binary = curve > threshold_95
        p_clusters, p_count = label(p_binary.astype(int))
        max_mass = 0.0
        for ci in range(1, p_count + 1):
            idx = p_clusters == ci
            mass = float(np.sum(curve[idx] - threshold_95[idx]))
            if mass > max_mass:
                max_mass = mass
        null_cluster_masses.append(max_mass)

    critical_mass = np.percentile(null_cluster_masses, 95) if len(null_cluster_masses) > 0 else np.inf
    sig_indices = np.zeros_like(mean_auc, dtype=bool)
    for ci in range(1, n_clusters + 1):
        if cluster_masses[ci - 1] > critical_mass:
            sig_indices[clusters == ci] = True
    return threshold_95, sig_indices


def compute_latency_points(mean_auc, sig_indices):
    latencies = {'earliest': np.nan, 'half_height': np.nan, 'peak': np.nan}
    if not np.any(sig_indices):
        return latencies

    sig_idx = np.where(sig_indices)[0]
    earliest_idx = sig_idx[0]
    masked_auc = np.where(sig_indices, mean_auc, -np.inf)
    peak_idx = int(np.argmax(masked_auc))

    clusters, _ = label(sig_indices.astype(int))
    first_cluster_id = clusters[earliest_idx]
    first_cluster_indices = np.where(clusters == first_cluster_id)[0]
    first_peak_idx = first_cluster_indices[np.argmax(mean_auc[first_cluster_indices])]
    first_peak_val = mean_auc[first_peak_idx]
    half_level = 0.5 + (first_peak_val - 0.5) / 2.0
    pre_peak_indices = first_cluster_indices[first_cluster_indices <= first_peak_idx]
    half_candidates = pre_peak_indices[mean_auc[pre_peak_indices] >= half_level]
    half_idx = int(half_candidates[0]) if half_candidates.size > 0 else int(first_peak_idx)

    latencies['earliest'] = float(PLOT_TIMES[min(earliest_idx, len(PLOT_TIMES) - 1)])
    latencies['half_height'] = float(PLOT_TIMES[min(half_idx, len(PLOT_TIMES) - 1)])
    latencies['peak'] = float(PLOT_TIMES[min(peak_idx, len(PLOT_TIMES) - 1)])
    return latencies


def generate_summary_figures(task, cache_dir, output_dir, logger):
    result_files = sorted(glob.glob(os.path.join(cache_dir, '*_results.npz')))
    if not result_files:
        logger(f'No result files in {cache_dir}')
        return

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
        title_text=f'{task["title"]} | {task["description"]}',
        save_path=os.path.join(output_dir, 'Fig_All_ROIs_Overview.png')
    )
    if sig_results:
        plot_grid_figures(
            results_dict=sig_results,
            title_text=f'{task["title"]} | Significant ROIs',
            save_path=os.path.join(output_dir, 'Fig_Significant_ROIs_Only.png')
        )
        plot_latency_bar_scatter(
            latency_dict=latency_earliest,
            title_text=f'{task["title"]} | Earliest Significant Latency',
            y_label='Earliest Significant Latency (ms)',
            save_path=os.path.join(output_dir, 'Fig_Latency_Earliest.png')
        )
        plot_latency_bar_scatter(
            latency_dict=latency_half,
            title_text=f'{task["title"]} | First-Peak Half-Height Latency',
            y_label='First-Peak Half-Height Latency (ms)',
            save_path=os.path.join(output_dir, 'Fig_Latency_FirstPeakHalfHeight.png')
        )
        plot_latency_bar_scatter(
            latency_dict=latency_peak,
            title_text=f'{task["title"]} | Peak Latency',
            y_label='Peak Latency (ms)',
            save_path=os.path.join(output_dir, 'Fig_Latency_Peak.png')
        )
        logger(f'Significant ROI count: {len(sig_results)}')
    else:
        logger(f'No significant ROI found for task: {task["id"]}')


def plot_single_roi_result(roi_name, mean_auc, sem_auc, threshold_95, sig_indices, figure_title, save_path):
    plot_times = PLOT_TIMES[:len(mean_auc)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(plot_times, mean_auc - sem_auc, mean_auc + sem_auc, color='#1f77b4', alpha=0.25)
    ax.plot(plot_times, mean_auc, color='#1f77b4', linewidth=1.8, label='Mean ROC AUC')
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
    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, lat + (T_END - T_START) * 0.01, f'{lat:.0f}ms', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rois, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f'{SUBJECT} | {title_text}', fontsize=12)
    ax.set_ylim(T_START, max(latencies) * 1.15 if max(latencies) > 0 else T_END)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def baseline_zscore(X, baseline_end_idx):
    # X: (n_trials, n_channels, n_time)
    baseline = X[:, :, :baseline_end_idx]           # 取t<0的部分
    mu = baseline.mean(axis=2, keepdims=True)        # 每trial每channel的均值
    sd = baseline.std(axis=2, keepdims=True) + 1e-8  # 避免除以0
    return (X - mu) / sd

if __name__ == '__main__':
    main()
