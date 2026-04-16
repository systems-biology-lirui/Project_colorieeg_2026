import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
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
REGIMES = [
    {'label': '10x100', 'n_repeats_perm': 10, 'n_perms': 100, 'color': '#1f77b4'},
    {'label': '1x1000', 'n_repeats_perm': 1, 'n_perms': 1000, 'color': '#d62728'},
]


def main():
    start_time = time.time()
    base_path = str(project_root())
    roi_path = project_root() / 'feature' / 'erp' / SUBJECT / f'{ROI_NAME}.mat'
    if not roi_path.is_file():
        raise FileNotFoundError(f'ROI file not found: {roi_path}')

    out_dir = result_root(base_path) / 'reports' / 'perm_regime_compare' / 'erp' / SUBJECT / ROI_NAME / TASK_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_times, real_auc, peak_idx, peak_time, peak_real_auc, regime_results = run_comparison(roi_path)
    save_figure(out_dir / 'peak_timepoint_comparison.png', plot_times, real_auc, peak_time, peak_real_auc, regime_results)
    save_summary(out_dir / 'peak_timepoint_comparison.npz', plot_times, real_auc, peak_idx, peak_time, peak_real_auc, regime_results)

    print({
        'peak_time_ms': peak_time,
        'peak_real_auc': peak_real_auc,
        'threshold_10x100': regime_results[0]['threshold'],
        'threshold_1x1000': regime_results[1]['threshold'],
        'delta_threshold': regime_results[1]['threshold'] - regime_results[0]['threshold'],
        'empirical_p_10x100': regime_results[0]['empirical_p'],
        'empirical_p_1x1000': regime_results[1]['empirical_p'],
        'runtime_s': time.time() - start_time,
        'figure_path': str(out_dir / 'peak_timepoint_comparison.png'),
        'summary_path': str(out_dir / 'peak_timepoint_comparison.npz'),
    })


def run_comparison(roi_path):
    mat = sio.loadmat(roi_path)
    category_data = build_category_data(mat)
    plot_times, real_auc = compute_real_auc(category_data)
    valid_idx = np.where(plot_times > 20)[0]
    peak_idx = int(valid_idx[np.argmax(real_auc[valid_idx])]) if valid_idx.size > 0 else int(np.argmax(real_auc))
    peak_time = float(plot_times[peak_idx])
    peak_real_auc = float(real_auc[peak_idx])

    regime_results = []
    for regime_idx, regime in enumerate(REGIMES):
        print(f'Start regime {regime["label"]}', flush=True)
        perm_values = run_perm_dist_at_time(
            category_data,
            peak_idx,
            regime['n_repeats_perm'],
            regime['n_perms'],
            base_seed=erp_decoder.RANDOM_STATE + regime_idx * 100000,
        )
        threshold = float(np.percentile(perm_values, 95))
        empirical_p = float((np.sum(perm_values >= peak_real_auc) + 1) / (len(perm_values) + 1))
        regime_results.append({
            **regime,
            'perm_values': perm_values,
            'threshold': threshold,
            'empirical_p': empirical_p,
            'mean_perm': float(np.mean(perm_values)),
            'std_perm': float(np.std(perm_values, ddof=0)),
        })
    return plot_times, real_auc, peak_idx, peak_time, peak_real_auc, regime_results


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


def compute_real_auc(category_data):
    auc_curves = []
    for x_cat, y_cat in category_data:
        scores = erp_decoder.run_decoding_over_time_cv(
            x_cat,
            y_cat,
            n_repeats=erp_decoder.N_REPEATS_REAL,
            shuffle=False,
            seed=erp_decoder.RANDOM_STATE,
        )
        auc_curves.append(np.mean(scores, axis=1))
    real_auc = np.mean(np.stack(auc_curves, axis=1), axis=1)
    plot_times = erp_decoder.PLOT_TIMES[:len(real_auc)]
    return plot_times, real_auc


def safe_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5


def run_single_timepoint_cv(x_cat, y_cat, time_idx, n_repeats, shuffle, seed):
    rng = np.random.RandomState(seed)
    y_use = rng.permutation(y_cat) if shuffle else y_cat.copy()
    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=erp_decoder.N_SPLITS, n_repeats=n_repeats, random_state=rng)
    else:
        cv = StratifiedKFold(n_splits=erp_decoder.N_SPLITS, shuffle=True, random_state=rng)
    x_t = x_cat[:, :, time_idx]
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    scores = []
    for train_idx, test_idx in cv.split(x_cat[:, 0, 0], y_use):
        clf.fit(x_t[train_idx], y_use[train_idx])
        y_prob = clf.predict_proba(x_t[test_idx])[:, 1]
        scores.append(safe_auc(y_use[test_idx], y_prob))
    return float(np.mean(scores))


def run_perm_dist_at_time(category_data, time_idx, n_repeats_perm, n_perms, base_seed):
    perm_values = np.zeros(n_perms, dtype=float)
    for perm_idx in range(n_perms):
        category_scores = []
        for cat_idx, (x_cat, y_cat) in enumerate(category_data):
            seed = base_seed + perm_idx * 10 + cat_idx
            score = run_single_timepoint_cv(x_cat, y_cat, time_idx, n_repeats_perm, True, seed)
            category_scores.append(score)
        perm_values[perm_idx] = float(np.mean(category_scores))
    return perm_values


def save_figure(save_path, plot_times, real_auc, peak_time, peak_real_auc, regime_results):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    ax = axes[0]
    ax.plot(plot_times, real_auc, color='black', linewidth=2.0, label='Real mean AUC')
    ax.axvline(peak_time, color='gray', linestyle='--', linewidth=1.0, label=f'Peak time = {peak_time:.1f} ms')
    for regime in regime_results:
        ax.axhline(regime['threshold'], color=regime['color'], linestyle='--', linewidth=1.8, label=f'{regime["label"]} threshold @ peak')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.0)
    ax.scatter([peak_time], [peak_real_auc], color='black', s=60, zorder=3)
    ax.set_xlim(erp_decoder.T_START, erp_decoder.T_END)
    ax.set_ylim(0.35, max(0.7, peak_real_auc + 0.05))
    ax.set_ylabel('ROC AUC')
    ax.set_title(f'{SUBJECT} | {ROI_NAME} | permutation regime comparison at decoding peak')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best', fontsize=9)

    ax = axes[1]
    box = ax.boxplot([regime['perm_values'] for regime in regime_results], patch_artist=True, labels=[regime['label'] for regime in regime_results])
    for patch, regime in zip(box['boxes'], regime_results):
        patch.set_facecolor(regime['color'])
        patch.set_alpha(0.45)
    for pos, regime in enumerate(regime_results, start=1):
        rng = np.random.RandomState(100 + pos)
        x_jitter = rng.normal(loc=pos, scale=0.03, size=regime['n_perms'])
        ax.scatter(x_jitter, regime['perm_values'], s=8, alpha=0.10, color=regime['color'])
        ax.scatter([pos], [regime['threshold']], s=90, color=regime['color'], edgecolor='white', linewidth=1.0, zorder=4)
    ax.axhline(peak_real_auc, color='black', linewidth=1.8, label=f'Real AUC = {peak_real_auc:.4f}')
    ax.set_ylabel(f'Permutation mean AUC at {peak_time:.1f} ms')
    ax.set_xlabel('Permutation regime')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_summary(save_path, plot_times, real_auc, peak_idx, peak_time, peak_real_auc, regime_results):
    payload = {
        'plot_times': plot_times,
        'real_auc': real_auc,
        'peak_idx': np.array(peak_idx),
        'peak_time': np.array(peak_time),
        'peak_real_auc': np.array(peak_real_auc),
    }
    for regime in regime_results:
        payload[f'{regime["label"]}_perm_values'] = regime['perm_values']
        payload[f'{regime["label"]}_threshold'] = np.array(regime['threshold'])
        payload[f'{regime["label"]}_empirical_p'] = np.array(regime['empirical_p'])
        payload[f'{regime["label"]}_mean_perm'] = np.array(regime['mean_perm'])
        payload[f'{regime["label"]}_std_perm'] = np.array(regime['std_perm'])
    np.savez(save_path, **payload)


if __name__ == '__main__':
    main()