import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from joblib import Parallel, delayed
from scipy.ndimage import label
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEWANALYSE_ROOT = PROJECT_ROOT / 'newanalyse'
if str(NEWANALYSE_ROOT) not in sys.path:
    sys.path.insert(0, str(NEWANALYSE_ROOT))

from newanalyse_paths import get_feature_dir, project_root, result_root


# =========================
# User Config
# =========================
# 1) SUBJECT: which subject to analyze.
# 2) FEATURE_KIND: which modality feature file to use.
# 3) ROI_NAME: which ROI feature file to decode. Example: 'Color_with_sti'.
# 4) TASK_FIELD: optional manual override of the task1 field inside the ROI file.
#    Leave as None to follow the modality defaults from newanalyse.
# 5) COLOR_CONDITION_INDICES / GRAY_CONDITION_INDICES: task1 condition indices used for 4-way category decoding.
#    The default mapping assumes [face, object, body, scene].
# 6) CATEGORY_NAMES: the class labels corresponding to those four conditions.
# 7) METRIC_NAME: 'acc' is the main 4-class decoding metric. 'auc_ovr' is optional macro one-vs-rest AUC.
# 8) N_PERMS / N_REPEATS_PERM: set to 0 if you only want the real decoding curve without permutation significance.
SUBJECT = 'test001'
FEATURE_KIND = 'erp'  # 'erp' | 'highgamma' | 'lowgamma' | 'tfa' | 'gamma' | 'gamma_multiband' |
ROI_NAME = 'Color_with_sti'
TASK_FIELD = None

COLOR_CONDITION_INDICES = [0, 2, 4, 6]
GRAY_CONDITION_INDICES = [1, 3, 5, 7]
CATEGORY_NAMES = ['face', 'object', 'body', 'scene']

METRIC_NAME = 'auc_ovr'  # 'acc' | 'auc_ovr'
N_SPLITS = 5
N_REPEATS_REAL = 10
N_REPEATS_PERM = 10
N_PERMS = 10
DECODING_STEP = 5
RANDOM_STATE = 42
N_JOBS = -1

FS = 500
T_START = -100
T_END = 1000
N_POINTS = 550

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
        'time_field': None,
        'default_smooth_win': 5,
    },
    'highgamma': {
        'feature_subdir': 'decoding_highgamma_features',
        'task_field': 'hg_task1',
        'time_field': None,
        'default_smooth_win': 5,
    },
    'lowgamma': {
        'feature_subdir': 'decoding_lowgamma_features',
        'task_field': 'lg_task1',
        'time_field': None,
        'default_smooth_win': 5,
    },
    'tfa': {
        'feature_subdir': 'decoding_tfa_features',
        'task_field': 'tfa_task1',
        'time_field': 'tfa_time_ms',
        'default_smooth_win': 5,
    },
    'gamma': {
        'feature_subdir': 'decoding_gamma_features',
        'task_field': 'g_task1',
        'time_field': None,
        'default_smooth_win': 5,
    },
    'gamma_multiband': {
        'feature_subdir': 'decoding_gamma_multiband_features',
        'task_field': 'gmb_task1',
        'time_field': 'gmb_time_ms',
        'default_smooth_win': 20,
    },
}

FEATURE_KIND = FEATURE_KIND_ALIASES.get(FEATURE_KIND, FEATURE_KIND)
if FEATURE_KIND not in FEATURE_CONFIG:
    raise ValueError(f'Unsupported FEATURE_KIND: {FEATURE_KIND}')

TIME_SMOOTH_WIN = FEATURE_CONFIG[FEATURE_KIND]['default_smooth_win']
if TASK_FIELD is None:
    TASK_FIELD = FEATURE_CONFIG[FEATURE_KIND]['task_field']

CHANCE_LEVEL_BY_METRIC = {
    'acc': 0.25,
    'auc_ovr': 0.5,
}


def get_roi_path():
    feature_dir = get_feature_dir(project_root(), FEATURE_CONFIG[FEATURE_KIND]['feature_subdir'], SUBJECT)
    return feature_dir / f'{ROI_NAME}.mat'


def get_time_vector(mat, n_time):
    time_field = FEATURE_CONFIG[FEATURE_KIND]['time_field']
    if time_field and time_field in mat:
        times = np.asarray(mat[time_field], dtype=float).reshape(-1)
        if times.size == n_time:
            return times
    return np.linspace(T_START, T_END, N_POINTS)[:n_time]


def score_predictions(metric_name, y_true, y_prob, y_pred):
    if metric_name == 'acc':
        return float(accuracy_score(y_true, y_pred))
    if metric_name == 'auc_ovr':
        if len(np.unique(y_true)) < len(CATEGORY_NAMES):
            return np.nan
        return float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
    raise ValueError(f'Unsupported metric: {metric_name}')


def baseline_zscore(data, baseline_end_idx):
    baseline = data[:, :, :baseline_end_idx]
    mean = baseline.mean(axis=2, keepdims=True)
    std = baseline.std(axis=2, keepdims=True) + 1e-8
    return (data - mean) / std


def smooth_data_causal(data, win_size):
    n_samples, n_features, n_time = data.shape
    smoothed = np.zeros_like(data)
    for time_idx in range(n_time):
        start_idx = max(0, time_idx - win_size)
        smoothed[:, :, time_idx] = np.mean(data[:, :, start_idx:time_idx + 1], axis=2)
    return smoothed


def build_multiclass_dataset(data, condition_indices):
    samples = []
    labels = []
    for class_idx, condition_idx in enumerate(condition_indices):
        condition_trials = data[condition_idx, :, :, :]
        samples.append(condition_trials)
        labels.append(np.full(condition_trials.shape[0], class_idx, dtype=int))
    x = np.concatenate(samples, axis=0)
    y = np.concatenate(labels, axis=0)

    baseline_end = np.searchsorted(np.linspace(T_START, T_END, x.shape[2]), 0)
    x = baseline_zscore(x, baseline_end)
    if TIME_SMOOTH_WIN > 0:
        x = smooth_data_causal(x, TIME_SMOOTH_WIN)
    return x, y


def run_multiclass_curve(x, y, metric_name, n_repeats, shuffle, seed):
    rng = np.random.RandomState(seed)
    y_use = rng.permutation(y) if shuffle else y.copy()
    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=rng)
    else:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=rng)
    splits = list(cv.split(x[:, 0, 0], y_use))
    time_indices = np.arange(0, x.shape[2], DECODING_STEP)
    curve = np.zeros(len(time_indices), dtype=float)
    sem = np.zeros(len(time_indices), dtype=float)
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))

    for time_pos, time_idx in enumerate(time_indices):
        x_t = x[:, :, time_idx]
        fold_scores = []
        for train_idx, test_idx in splits:
            clf.fit(x_t[train_idx], y_use[train_idx])
            y_pred = clf.predict(x_t[test_idx])
            y_prob = clf.predict_proba(x_t[test_idx])
            score = score_predictions(metric_name, y_use[test_idx], y_prob, y_pred)
            fold_scores.append(score)
        fold_scores = np.asarray(fold_scores, dtype=float)
        curve[time_pos] = float(np.nanmean(fold_scores))
        sem[time_pos] = float(np.nanstd(fold_scores, ddof=0) / np.sqrt(np.sum(np.isfinite(fold_scores))))
    return curve, sem


def run_multiclass_curve_mean_only(x, y, metric_name, n_repeats, shuffle, seed):
    curve, _ = run_multiclass_curve(x, y, metric_name, n_repeats, shuffle, seed)
    return curve


def cluster_permutation_significance(real_curve, perm_dist, chance_level, plot_times):
    if perm_dist is None or np.size(perm_dist) == 0:
        return np.full(real_curve.shape, np.nan, dtype=float), np.zeros_like(real_curve, dtype=bool)

    threshold_95 = np.nanpercentile(perm_dist, 95, axis=0)
    valid_mask = plot_times[:len(real_curve)] > 20
    binary_map = (real_curve > threshold_95) & valid_mask
    clusters, n_clusters = label(binary_map.astype(int))

    cluster_masses = []
    for cluster_id in range(1, n_clusters + 1):
        indices = clusters == cluster_id
        cluster_masses.append(float(np.sum(real_curve[indices] - np.maximum(threshold_95[indices], chance_level))))

    null_cluster_masses = []
    for perm_curve in perm_dist:
        perm_binary = (perm_curve > threshold_95) & valid_mask
        perm_clusters, perm_count = label(perm_binary.astype(int))
        max_mass = 0.0
        for cluster_id in range(1, perm_count + 1):
            indices = perm_clusters == cluster_id
            mass = float(np.sum(perm_curve[indices] - np.maximum(threshold_95[indices], chance_level)))
            if mass > max_mass:
                max_mass = mass
        null_cluster_masses.append(max_mass)

    critical_mass = np.percentile(null_cluster_masses, 95) if null_cluster_masses else np.inf
    sig_mask = np.zeros_like(real_curve, dtype=bool)
    for cluster_id in range(1, n_clusters + 1):
        if cluster_masses[cluster_id - 1] > critical_mass:
            sig_mask[clusters == cluster_id] = True
    return threshold_95, sig_mask


def compute_latency_points(curve, sig_mask, chance_level, plot_times):
    latencies = {'earliest': np.nan, 'half_height': np.nan, 'peak': np.nan}
    if not np.any(sig_mask):
        return latencies

    sig_idx = np.where(sig_mask)[0]
    earliest_idx = int(sig_idx[0])
    masked_curve = np.where(sig_mask, curve, -np.inf)
    peak_idx = int(np.argmax(masked_curve))

    clusters, _ = label(sig_mask.astype(int))
    first_cluster_id = clusters[earliest_idx]
    first_cluster_indices = np.where(clusters == first_cluster_id)[0]
    first_peak_idx = int(first_cluster_indices[np.argmax(curve[first_cluster_indices])])
    first_peak_val = float(curve[first_peak_idx])
    half_level = chance_level + (first_peak_val - chance_level) / 2.0
    pre_peak_indices = first_cluster_indices[first_cluster_indices <= first_peak_idx]
    half_candidates = pre_peak_indices[curve[pre_peak_indices] >= half_level]
    half_idx = int(half_candidates[0]) if half_candidates.size > 0 else first_peak_idx

    latencies['earliest'] = float(plot_times[min(earliest_idx, len(plot_times) - 1)])
    latencies['half_height'] = float(plot_times[min(half_idx, len(plot_times) - 1)])
    latencies['peak'] = float(plot_times[min(peak_idx, len(plot_times) - 1)])
    return latencies


def run_condition_set(x, y, label_name, seed_offset, plot_times):
    real_curve, real_sem = run_multiclass_curve(
        x,
        y,
        metric_name=METRIC_NAME,
        n_repeats=N_REPEATS_REAL,
        shuffle=False,
        seed=RANDOM_STATE + seed_offset,
    )

    if N_PERMS > 0:
        perm_dist = np.asarray(
            Parallel(n_jobs=N_JOBS)(
                delayed(run_multiclass_curve_mean_only)(
                    x,
                    y,
                    METRIC_NAME,
                    N_REPEATS_PERM,
                    True,
                    RANDOM_STATE + seed_offset * 100000 + perm_idx,
                )
                for perm_idx in range(N_PERMS)
            )
        )
    else:
        perm_dist = np.full((0, real_curve.shape[0]), np.nan)

    chance_level = CHANCE_LEVEL_BY_METRIC[METRIC_NAME]
    threshold_95, sig_mask = cluster_permutation_significance(real_curve, perm_dist, chance_level, plot_times)
    latencies = compute_latency_points(real_curve, sig_mask, chance_level, plot_times)
    return {
        'label': label_name,
        'curve': real_curve,
        'sem': real_sem,
        'perm_dist': perm_dist,
        'threshold_95': threshold_95,
        'sig_mask': sig_mask,
        'latencies': latencies,
        'chance_level': chance_level,
    }


def save_figure(save_path, plot_times, color_result, gray_result):
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

    ax = axes[0]
    for result, color in ((color_result, '#d62728'), (gray_result, '#4c78a8')):
        ax.fill_between(plot_times, result['curve'] - result['sem'], result['curve'] + result['sem'], color=color, alpha=0.18)
        ax.plot(plot_times, result['curve'], color=color, linewidth=2.0, label=f"{result['label']} real")
        if np.any(np.isfinite(result['threshold_95'])):
            ax.plot(plot_times, result['threshold_95'], color=color, linestyle='--', linewidth=1.5, label=f"{result['label']} 95% threshold")
    ax.axhline(color_result['chance_level'], color='black', linestyle=':', linewidth=1.0)
    ax.set_ylabel(METRIC_NAME.upper())
    ax.set_title(f'{SUBJECT} | {ROI_NAME} | Task1 color vs gray 4-class decoding')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best', fontsize=9)

    ax = axes[1]
    for row_idx, (result, color) in enumerate(((color_result, '#d62728'), (gray_result, '#4c78a8'))):
        mask = result['sig_mask'].astype(float)
        if np.any(mask):
            ax.fill_between(plot_times, row_idx, row_idx + mask, step='mid', color=color, alpha=0.85)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels([color_result['label'], gray_result['label']])
    ax.set_ylabel('Significant mask')
    ax.set_ylim(0, 2)
    ax.grid(True, linestyle='--', alpha=0.35)

    ax = axes[2]
    diff_curve = color_result['curve'] - gray_result['curve']
    ax.plot(plot_times, diff_curve, color='#2ca02c', linewidth=2.0, label='Color - Gray')
    ax.axhline(0.0, color='black', linestyle=':', linewidth=1.0)
    for result, color in ((color_result, '#d62728'), (gray_result, '#4c78a8')):
        earliest = result['latencies']['earliest']
        if np.isfinite(earliest):
            ax.axvline(earliest, color=color, linestyle='--', linewidth=1.2, label=f"{result['label']} earliest = {earliest:.1f} ms")
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(f'{METRIC_NAME.upper()} difference')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_summary(save_path, plot_times, color_result, gray_result):
    np.savez(
        save_path,
        subject=np.array(SUBJECT),
        feature_kind=np.array(FEATURE_KIND),
        roi_name=np.array(ROI_NAME),
        task_field=np.array(TASK_FIELD),
        metric_name=np.array(METRIC_NAME),
        category_names=np.array(CATEGORY_NAMES),
        color_condition_indices=np.array(COLOR_CONDITION_INDICES),
        gray_condition_indices=np.array(GRAY_CONDITION_INDICES),
        plot_times=plot_times,
        color_curve=color_result['curve'],
        color_sem=color_result['sem'],
        color_perm_dist=color_result['perm_dist'],
        color_threshold_95=color_result['threshold_95'],
        color_sig_mask=color_result['sig_mask'],
        color_latency_earliest=np.array(color_result['latencies']['earliest']),
        color_latency_half_height=np.array(color_result['latencies']['half_height']),
        color_latency_peak=np.array(color_result['latencies']['peak']),
        gray_curve=gray_result['curve'],
        gray_sem=gray_result['sem'],
        gray_perm_dist=gray_result['perm_dist'],
        gray_threshold_95=gray_result['threshold_95'],
        gray_sig_mask=gray_result['sig_mask'],
        gray_latency_earliest=np.array(gray_result['latencies']['earliest']),
        gray_latency_half_height=np.array(gray_result['latencies']['half_height']),
        gray_latency_peak=np.array(gray_result['latencies']['peak']),
    )


def main():
    start_time = time.time()
    if TASK_FIELD != FEATURE_CONFIG[FEATURE_KIND]['task_field']:
        raise ValueError(f"TASK_FIELD should be {FEATURE_CONFIG[FEATURE_KIND]['task_field']} for FEATURE_KIND={FEATURE_KIND}")
    if len(COLOR_CONDITION_INDICES) != 4 or len(GRAY_CONDITION_INDICES) != 4:
        raise ValueError('Both COLOR_CONDITION_INDICES and GRAY_CONDITION_INDICES must have 4 entries for 4-class decoding.')
    if len(CATEGORY_NAMES) != 4:
        raise ValueError('CATEGORY_NAMES must contain exactly 4 class names.')

    roi_path = get_roi_path()
    if not roi_path.is_file():
        raise FileNotFoundError(f'ROI file not found: {roi_path}')

    out_dir = result_root(project_root()) / 'reports' / 'task1_fourclass_compare' / FEATURE_KIND / SUBJECT / ROI_NAME / METRIC_NAME / f'perm{N_PERMS}_repeat{N_REPEATS_PERM}'
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = sio.loadmat(roi_path)
    if TASK_FIELD not in mat:
        raise ValueError(f'Missing matrix: {TASK_FIELD}')
    data = np.asarray(mat[TASK_FIELD], dtype=float)
    if data.shape[0] < 8:
        raise ValueError(f'{TASK_FIELD} requires at least 8 conditions for task1 color/gray category decoding.')
    time_vector = get_time_vector(mat, data.shape[-1])
    plot_times = time_vector[::DECODING_STEP][: len(np.arange(0, data.shape[-1], DECODING_STEP))]

    x_color, y_color = build_multiclass_dataset(data, COLOR_CONDITION_INDICES)
    x_gray, y_gray = build_multiclass_dataset(data, GRAY_CONDITION_INDICES)

    color_result = run_condition_set(x_color, y_color, 'Color 4-class', seed_offset=1, plot_times=plot_times)
    gray_result = run_condition_set(x_gray, y_gray, 'Gray 4-class', seed_offset=2, plot_times=plot_times)

    plot_times = plot_times[:len(color_result['curve'])]
    fig_path = out_dir / 'task1_color_vs_gray_fourclass.png'
    npz_path = out_dir / 'task1_color_vs_gray_fourclass.npz'
    save_figure(fig_path, plot_times, color_result, gray_result)
    save_summary(npz_path, plot_times, color_result, gray_result)

    summary = {
        'subject': SUBJECT,
        'feature_kind': FEATURE_KIND,
        'roi_name': ROI_NAME,
        'task_field': TASK_FIELD,
        'metric_name': METRIC_NAME,
        'color_conditions': COLOR_CONDITION_INDICES,
        'gray_conditions': GRAY_CONDITION_INDICES,
        'color_earliest_ms': color_result['latencies']['earliest'],
        'gray_earliest_ms': gray_result['latencies']['earliest'],
        'color_peak_ms': color_result['latencies']['peak'],
        'gray_peak_ms': gray_result['latencies']['peak'],
        'figure_path': str(fig_path),
        'summary_path': str(npz_path),
        'runtime_s': time.time() - start_time,
    }
    print('Configuration:')
    print(f'  Data source: ROI feature mat file = {roi_path}')
    print(f"  Feature subdir: {FEATURE_CONFIG[FEATURE_KIND]['feature_subdir']}")
    print(f'  TASK_FIELD: {TASK_FIELD}')
    print(f"  Time field: {FEATURE_CONFIG[FEATURE_KIND]['time_field']}")
    print(f'  Time smooth win: {TIME_SMOOTH_WIN}')
    print(f'  Color conditions: {COLOR_CONDITION_INDICES} -> {CATEGORY_NAMES}')
    print(f'  Gray conditions: {GRAY_CONDITION_INDICES} -> {CATEGORY_NAMES}')
    print(f'  Metric: {METRIC_NAME}')
    print(summary)


if __name__ == '__main__':
    main()