import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEWANALYSE_ROOT = PROJECT_ROOT / 'newanalyse'
if str(NEWANALYSE_ROOT) not in sys.path:
    sys.path.insert(0, str(NEWANALYSE_ROOT))

import Sec3_1_all_roi_result_erp as erp_decoder
from newanalyse_paths import project_root, result_root


SUBJECT = 'test001'
ROI_NAME = 'Color_with_sti'
TASK_ID = 'task1_color_vs_gray_per_category'
N_PERMS = 100
N_REPEATS_PERM = 10
METRICS = [
    {'name': 'auc', 'color': '#1f77b4'},
    {'name': 'acc', 'color': '#d62728'},
]


def main():
    start_time = time.time()
    base_path = str(project_root())
    roi_path = project_root() / 'feature' / 'erp' / SUBJECT / f'{ROI_NAME}.mat'
    if not roi_path.is_file():
        raise FileNotFoundError(f'ROI file not found: {roi_path}')

    out_dir = result_root(base_path) / 'reports' / 'acc_auc_compare' / 'erp' / SUBJECT / ROI_NAME / TASK_ID / f'perm{N_PERMS}_repeat{N_REPEATS_PERM}'
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = sio.loadmat(roi_path)
    category_data = build_category_data(mat)
    plot_times = erp_decoder.PLOT_TIMES[:len(np.arange(0, category_data[0][0].shape[2], erp_decoder.DECODING_STEP))]

    metric_results = {}
    for metric_idx, metric in enumerate(METRICS):
        print(f'Running metric: {metric["name"]}', flush=True)
        real_curve, perm_dist = run_metric_full_time(category_data, metric['name'], metric_idx)
        threshold, sig_mask = erp_decoder.cluster_permutation_significance(real_curve, perm_dist)
        latencies = erp_decoder.compute_latency_points(real_curve, sig_mask)
        metric_results[metric['name']] = {
            'color': metric['color'],
            'real_curve': real_curve,
            'perm_dist': perm_dist,
            'threshold': threshold,
            'sig_mask': sig_mask,
            'latencies': latencies,
        }

    save_figure(out_dir / 'acc_vs_auc_full_timecourse.png', plot_times, metric_results)
    save_summary(out_dir / 'acc_vs_auc_full_timecourse.npz', plot_times, metric_results)

    summary = {
        'auc_earliest_ms': metric_results['auc']['latencies']['earliest'],
        'acc_earliest_ms': metric_results['acc']['latencies']['earliest'],
        'auc_peak_ms': metric_results['auc']['latencies']['peak'],
        'acc_peak_ms': metric_results['acc']['latencies']['peak'],
        'runtime_s': time.time() - start_time,
        'figure_path': str(out_dir / 'acc_vs_auc_full_timecourse.png'),
        'summary_path': str(out_dir / 'acc_vs_auc_full_timecourse.npz'),
    }
    print(summary)


def build_category_data(mat):
    if 'erp_task1' not in mat:
        raise ValueError('Missing matrix: erp_task1')
    data = mat['erp_task1']
    baseline_end = np.searchsorted(erp_decoder.TIMES, 0)
    category_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    category_data = []
    for color_idx, gray_idx in category_pairs:
        x_color = data[color_idx, :, :, :]
        x_gray = data[gray_idx, :, :, :]
        x_cat = np.concatenate([x_color, x_gray], axis=0)
        y_cat = np.concatenate([np.zeros(x_color.shape[0]), np.ones(x_gray.shape[0])])
        x_cat = erp_decoder.baseline_zscore(x_cat, baseline_end)
        if erp_decoder.TIME_SMOOTH_WIN > 0:
            x_cat = erp_decoder.smooth_data_causal(x_cat, erp_decoder.TIME_SMOOTH_WIN)
        category_data.append((x_cat, y_cat))
    return category_data


def score_fold(metric_name, y_true, y_prob):
    if metric_name == 'auc':
        return erp_decoder.safe_auc(y_true, y_prob)
    if metric_name == 'acc':
        y_pred = (y_prob >= 0.5).astype(float)
        return float(np.mean(y_pred == y_true)) if y_true.size else 0.5
    raise ValueError(f'Unsupported metric: {metric_name}')


def run_metric_curve_single_category(x_cat, y_cat, metric_name, n_repeats, shuffle, seed):
    rng = np.random.RandomState(seed)
    y_use = rng.permutation(y_cat) if shuffle else y_cat.copy()
    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=erp_decoder.N_SPLITS, n_repeats=n_repeats, random_state=rng)
    else:
        cv = StratifiedKFold(n_splits=erp_decoder.N_SPLITS, shuffle=True, random_state=rng)
    splits = list(cv.split(x_cat[:, 0, 0], y_use))
    time_indices = np.arange(0, x_cat.shape[2], erp_decoder.DECODING_STEP)
    curve = np.zeros(len(time_indices), dtype=float)
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    for time_pos, time_idx in enumerate(time_indices):
        x_t = x_cat[:, :, time_idx]
        fold_scores = []
        for train_idx, test_idx in splits:
            clf.fit(x_t[train_idx], y_use[train_idx])
            y_prob = clf.predict_proba(x_t[test_idx])[:, 1]
            fold_scores.append(score_fold(metric_name, y_use[test_idx], y_prob))
        curve[time_pos] = float(np.mean(fold_scores))
    return curve


def run_metric_full_time(category_data, metric_name, metric_seed_offset):
    real_curves = []
    for cat_idx, (x_cat, y_cat) in enumerate(category_data):
        real_curve = run_metric_curve_single_category(
            x_cat,
            y_cat,
            metric_name,
            n_repeats=erp_decoder.N_REPEATS_REAL,
            shuffle=False,
            seed=erp_decoder.RANDOM_STATE + metric_seed_offset * 1000 + cat_idx,
        )
        real_curves.append(real_curve)
    real_curve = np.mean(np.stack(real_curves, axis=1), axis=1)

    def one_perm_curve(perm_idx):
        perm_curves = []
        for cat_idx, (x_cat, y_cat) in enumerate(category_data):
            perm_curve = run_metric_curve_single_category(
                x_cat,
                y_cat,
                metric_name,
                n_repeats=N_REPEATS_PERM,
                shuffle=True,
                seed=erp_decoder.RANDOM_STATE + metric_seed_offset * 100000 + perm_idx * 10 + cat_idx,
            )
            perm_curves.append(perm_curve)
        return np.mean(np.stack(perm_curves, axis=1), axis=1)

    perm_dist = np.array(
        Parallel(n_jobs=erp_decoder.N_JOBS)(
            delayed(one_perm_curve)(perm_idx) for perm_idx in range(N_PERMS)
        )
    )
    return real_curve, perm_dist


def save_figure(save_path, plot_times, metric_results):
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

    ax = axes[0]
    for metric_name, result in metric_results.items():
        ax.plot(plot_times, result['real_curve'], color=result['color'], linewidth=2.0, label=f'{metric_name.upper()} real')
        ax.plot(plot_times, result['threshold'], color=result['color'], linestyle='--', linewidth=1.5, label=f'{metric_name.upper()} 95% threshold')
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1.0)
    ax.set_ylabel('Score')
    ax.set_title(f'{SUBJECT} | {ROI_NAME} | ACC vs AUC significance comparison')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best', fontsize=9)

    ax = axes[1]
    for row_idx, metric_name in enumerate(['auc', 'acc']):
        result = metric_results[metric_name]
        mask = result['sig_mask'].astype(float)
        ax.fill_between(plot_times, row_idx, row_idx + mask, step='mid', color=result['color'], alpha=0.85)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['AUC', 'ACC'])
    ax.set_ylabel('Significant mask')
    ax.set_ylim(0, 2)
    ax.grid(True, linestyle='--', alpha=0.35)

    ax = axes[2]
    auc_curve = metric_results['auc']['real_curve']
    acc_curve = metric_results['acc']['real_curve']
    ax.plot(plot_times, acc_curve - auc_curve, color='#2ca02c', linewidth=2.0, label='ACC - AUC')
    ax.axhline(0.0, color='black', linestyle=':', linewidth=1.0)
    for metric_name, result in metric_results.items():
        earliest = result['latencies']['earliest']
        if np.isfinite(earliest):
            ax.axvline(earliest, color=result['color'], linestyle='--', linewidth=1.2, label=f'{metric_name.upper()} earliest = {earliest:.1f} ms')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Score difference')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_summary(save_path, plot_times, metric_results):
    payload = {'plot_times': plot_times}
    for metric_name, result in metric_results.items():
        payload[f'{metric_name}_real_curve'] = result['real_curve']
        payload[f'{metric_name}_perm_dist'] = result['perm_dist']
        payload[f'{metric_name}_threshold'] = result['threshold']
        payload[f'{metric_name}_sig_mask'] = result['sig_mask']
        payload[f'{metric_name}_latency_earliest'] = np.array(result['latencies']['earliest'])
        payload[f'{metric_name}_latency_half_height'] = np.array(result['latencies']['half_height'])
        payload[f'{metric_name}_latency_peak'] = np.array(result['latencies']['peak'])
    np.savez(save_path, **payload)


if __name__ == '__main__':
    main()