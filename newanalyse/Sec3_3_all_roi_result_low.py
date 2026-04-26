import os
import glob
import math
import time
from datetime import datetime
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from scipy.ndimage import label

from runtime_config import load_runtime_config
from groupeddata_pairing import (
    center_paired_trials,
    load_paired_category_trials,
    resolve_within_category_task,
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




SUBJECT = 'test001'
BASE_PATH = str(project_root())
FEATURE_SUBDIR = 'decoding_lowgamma_features'
FEATURE_DIR = str(get_feature_dir(BASE_PATH, FEATURE_SUBDIR, SUBJECT))

N_SPLITS = 5
N_REALS = 100
N_REPEATS_REAL = 10
N_REPEATS_PERM = N_REPEATS_REAL
N_PERMS = 1000
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
FEATURE_KIND = 'lowgamma'
FIELD_PREFIX = 'lg'
USE_GROUPEDDATA_PAIRING = False
USE_GROUPEDDATA_PAIR_CENTERING = False
GROUPEDDATA_FILES = {}
DEFAULT_CATEGORY_PAIRS_BY_TASK = {
    'task1': [(0, 1), (2, 3), (4, 5), (6, 7)],
}
DEFAULT_CATEGORY_NAMES_BY_TASK = {
    'task1': ['face', 'object', 'body', 'scene'],
}

TASKS = [
    # {
    #     'id': 'task3_1vs4_self_1000',
    #     'title': 'Task 3 Pure Color Self-Decoding: Condition 1 vs Condition 4',
        'description': 'Decode color vs gray within each of 4 categories (face/object/body/scene), then average AUC across categories.',
    #     'mode': 'within_cv',
        'task_name': 'task1',
    #     'data_key': 'lg_task3',
    #     'class0': [0],
    #     'class1': [3],
    #     'folder': 'task3_1vs4_self'
    # },
    # {
    #     'id': 'task3_2vs3_self',
    #     'title': 'Task 3 Pure Color Self-Decoding: Condition 2 vs Condition 3',
    #     'description': 'Train and test within Task 3 using condition 2 vs condition 3.',
    #     'mode': 'within_cv',
    #     'data_key': 'lg_task3',
    #     'class0': [1],
    #     'class1': [2],
    #     'folder': 'task3_2vs3_self'
    # },
    # {
    #     'id': 'task1_color_vs_gray_pair_cv',
    #     'title': 'Task 1 Color vs Gray Pair Holdout Decoding',
    #     'description': 'Four-fold pair holdout: three odd-even pairs for train, one pair for test.',
    #     'mode': 'pair_holdout_task1',
    #     'folder': 'task1_color_vs_gray_pair_cv3'
    # },
    {
        'id': 'task1_color_vs_gray_per_category',
        'title': 'Task 1 Color vs Gray Per-Category Decoding',
        'description': 'Decode color vs gray within each of 4 categories (face/object/body/scene), then average AUC across categories.',
        'mode': 'within_category_color_gray',
        'folder': 'task1_color_vs_gray_per_category1'
    },
    # {
    #     'id': 'task2_gray_memory_color_cross',
    #     'title': 'Task 2 Gray Fruit Memory-Color Decoding',
    #     'description': 'Cross-object decoding on gray fruits for memory color red vs green with four combinations.',
    #     'mode': 'cross_combo_task2_gray',
    #     'folder': 'task2_gray_memory_color_cross'
    # },
    # {
    #     'id': 'task2_true_vs_false',
    #     'title': 'Task 2 True vs False Fruit Color Decoding',
    #     'description': 'Binary decoding between true-color fruits and false-color fruits.',
    #     'mode': 'within_cv',
    #     'data_key': 'lg_task2',
    #     'class0': [0, 3, 6, 9],
    #     'class1': [1, 4, 7, 10],
    #     'folder': 'task2_true_vs_false'
    # }
]


_RUNTIME_CFG = load_runtime_config(__file__, sections=('python_defaults', 'sec3_defaults'))
GROUPEDDATA_FILES = {
    'task1': str(get_task_groupeddata_path(BASE_PATH, SUBJECT, 'task1')),
    'task2': '',
    'task3': '',
}
if _RUNTIME_CFG:
    SUBJECT = str(_RUNTIME_CFG.get('subject', SUBJECT))
    BASE_PATH = str(_RUNTIME_CFG.get('base_path', BASE_PATH))
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
    USE_GROUPEDDATA_PAIRING = bool(_RUNTIME_CFG.get('use_groupeddata_pairing', USE_GROUPEDDATA_PAIRING))
    USE_GROUPEDDATA_PAIR_CENTERING = bool(_RUNTIME_CFG.get('use_groupeddata_pair_centering', USE_GROUPEDDATA_PAIR_CENTERING))
    GROUPEDDATA_FILES = {
        'task1': str(get_task_groupeddata_path(BASE_PATH, SUBJECT, 'task1')),
        'task2': '',
        'task3': '',
    }
    if 'groupeddata_files' in _RUNTIME_CFG:
        GROUPEDDATA_FILES = dict(_RUNTIME_CFG['groupeddata_files'])
    if 'tasks' in _RUNTIME_CFG:
        TASKS = _RUNTIME_CFG['tasks']

FEATURE_DIR = str(get_feature_dir(BASE_PATH, FEATURE_SUBDIR, SUBJECT))
TIMES = np.linspace(T_START, T_END, N_POINTS)
PLOT_TIMES = TIMES[::DECODING_STEP]


def should_run_permutation_test():
    return bool(RUN_PERMUTATION_TEST and N_PERMS > 0)


def get_perm_tag():
    return f'perm{N_PERMS}' if should_run_permutation_test() else 'real_only'


def resolve_within_category_spec(task):
    return resolve_within_category_task(
        task,
        field_prefix=FIELD_PREFIX,
        default_task_name='task1',
        default_category_pairs=DEFAULT_CATEGORY_PAIRS_BY_TASK,
        default_category_names=DEFAULT_CATEGORY_NAMES_BY_TASK,
        use_groupeddata_pairing=USE_GROUPEDDATA_PAIRING,
        use_groupeddata_pair_centering=USE_GROUPEDDATA_PAIR_CENTERING,
        groupeddata_files=GROUPEDDATA_FILES,
    )


def main():
    mat_files = sorted(glob.glob(os.path.join(FEATURE_DIR, ROI_FILE_PATTERN)))
    if not mat_files:
        print(f'No ROI file found: {FEATURE_DIR}')
        return

    batch_root = str(get_within_decoding_batch_dir(BASE_PATH, 'lowgamma', SUBJECT, batch_name=BATCH_NAME))
    os.makedirs(batch_root, exist_ok=True)
    log_path = os.path.join(batch_root, f'batch_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logger = make_logger(log_path)

    logger(f'Start batch decoding for {SUBJECT}')
    logger(f'Feature dir: {FEATURE_DIR}')
    logger(f'ROI count: {len(mat_files)}')
    logger(f'n_real={N_REALS}, n_repeats_real={N_REPEATS_REAL}, n_repeats_perm={N_REPEATS_PERM}, n_perm={N_PERMS}')

    task_outputs = {}
    for task in TASKS:
        output_dir = str(get_within_decoding_task_dir(BASE_PATH, task['id'], 'lowgamma', SUBJECT, get_perm_tag(), variant='with_sti', batch_name=BATCH_NAME))
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
                run_task_for_roi(fpath, roi_name, task, save_path, roi_plot_dir, logger)
            except Exception as exc:
                logger(f'ROI failed: {roi_name} | {exc}')

        summary = generate_summary_figures(task, cache_dir, output_dir, logger)
        task_outputs[task['id']] = summary

    plot_task3_dual_overlay(task_outputs, batch_root, logger)
    logger('Batch decoding completed')


def make_logger(log_path):
    def _log(msg):
        text = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(text)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
    return _log


def run_task_for_roi(fpath, roi_name, task, save_path, roi_plot_dir, logger):
    mat = sio.loadmat(fpath)
    mode = task['mode']

    if mode == 'within_cv':
        X_raw, y = build_within_cv_data(mat, task['data_key'], task['class0'], task['class1'])
        baseline_end = np.searchsorted(TIMES, 0)
        X_raw = baseline_zscore(X_raw, baseline_end)
        X_use = smooth_data_causal(X_raw, TIME_SMOOTH_WIN) if TIME_SMOOTH_WIN > 0 else X_raw
        
        real_auc_matrix = run_decoding_over_time_cv(X_use, y, n_repeats=N_REPEATS_REAL, shuffle=False, seed=RANDOM_STATE)
        real_auc_runs = run_real_curve_distribution(
            np.mean(real_auc_matrix, axis=1),
            lambda real_seed: run_decoding_over_time_cv_mean(X_use, y, N_REPEATS_REAL, False, real_seed),
        )
        perm_dist = run_permutation_distribution(
            lambda shuffle, seed: run_decoding_over_time_cv_mean(X_use, y, N_REPEATS_PERM, shuffle, seed),
            real_auc_matrix.shape[0],
        )
    elif mode == 'pair_holdout_task1':
        X_raw, y, groups = build_task1_pair_holdout_data(mat)
        baseline_end = np.searchsorted(TIMES, 0)
        X_raw = baseline_zscore(X_raw, baseline_end)
        X_use = smooth_data_causal(X_raw, TIME_SMOOTH_WIN) if TIME_SMOOTH_WIN > 0 else X_raw
        real_auc_matrix = run_decoding_over_time_group_holdout(X_use, y, groups, shuffle=False, seed=RANDOM_STATE)
        real_auc_runs = run_real_curve_distribution(
            np.mean(real_auc_matrix, axis=1),
            lambda real_seed: run_decoding_over_time_group_holdout_mean(X_use, y, groups, False, real_seed),
        )
        perm_dist = run_permutation_distribution(
            lambda shuffle, seed: run_decoding_over_time_group_holdout_mean(X_use, y, groups, shuffle, seed),
            real_auc_matrix.shape[0],
        )
    elif mode == 'cross_combo_task2_gray':
        combo_data = build_task2_gray_memory_combos(mat)
        real_auc_matrix = run_decoding_over_time_task2_gray_combos(combo_data, shuffle=False, seed=RANDOM_STATE)
        real_auc_runs = run_real_curve_distribution(
            np.mean(real_auc_matrix, axis=1),
            lambda real_seed: run_decoding_over_time_task2_gray_combos_mean(combo_data, False, real_seed),
        )
        perm_dist = run_permutation_distribution(
            lambda shuffle, seed: run_decoding_over_time_task2_gray_combos_mean(combo_data, shuffle, seed),
            real_auc_matrix.shape[0],
        )
    elif mode == 'within_category_color_gray':
        within_category_spec = resolve_within_category_spec(task)
        real_auc_matrix, real_sem_matrix, real_auc_runs, perm_dist_mean, perm_dist_all, matched_pair_counts = run_decoding_per_category(
            mat,
            within_category_spec,
            shuffle=False,
            seed=RANDOM_STATE,
        )
        category_names = within_category_spec['category_names']
        real_auc_mean_runs = np.mean(real_auc_runs, axis=2)

        mean_auc = np.mean(real_auc_mean_runs, axis=0)
        sem_auc = np.std(real_auc_mean_runs, axis=0, ddof=0) / np.sqrt(real_auc_mean_runs.shape[0])
        threshold_95, sig_indices = cluster_permutation_significance(mean_auc, perm_dist_mean)
        latencies = compute_latency_points(mean_auc, sig_indices)

        np.savez(
            save_path,
            mean_auc=mean_auc,
            sem_auc=sem_auc,
            threshold_95=threshold_95,
            sig_indices=sig_indices,
            real_auc_matrix=real_auc_matrix,
            real_auc_runs=real_auc_runs,
            perm_dist_mean=perm_dist_mean,
            perm_dist_all=perm_dist_all,
            latency_earliest=latencies['earliest'],
            latency_half_height=latencies['half_height'],
            latency_peak=latencies['peak'],
            task_id=np.array(task['id']),
            task_title=np.array(task['title']),
            task_name=np.array(within_category_spec['task_name']),
            category_names=np.array(category_names),
            n_real=np.array(N_REALS),
            n_repeats_real=np.array(N_REPEATS_REAL),
            n_repeats_perm=np.array(N_REPEATS_PERM),
            n_perm=np.array(N_PERMS),
            use_groupeddata_pairing=np.array(within_category_spec['use_groupeddata_pairing']),
            use_groupeddata_pair_centering=np.array(within_category_spec['use_groupeddata_pair_centering']),
            groupeddata_mat=np.array(str(within_category_spec['groupeddata_mat'] or '')),
            matched_pair_counts=np.asarray(matched_pair_counts, dtype=int),
        )
        
        plot_single_roi_result(
            roi_name=f"{roi_name}_Mean",
            mean_auc=mean_auc,
            sem_auc=sem_auc,
            threshold_95=threshold_95,
            sig_indices=sig_indices,
            figure_title=f'{task["title"]} (Mean) | {task["description"]}',
            save_path=os.path.join(roi_plot_dir, f'{roi_name}_mean_curve.png')
        )

        for i, cat_name in enumerate(category_names):
            cat_auc = real_auc_matrix[:, i]
            cat_sem = real_sem_matrix[:, i]
            cat_perm_dist = perm_dist_all[:, :, i]
            
            # 单独计算显著性
            cat_threshold_95, cat_sig_indices = cluster_permutation_significance(cat_auc, cat_perm_dist)
            cat_latencies = compute_latency_points(cat_auc, cat_sig_indices)
            
            # 为该类别生成专属的保存路径 (例如: test001_face_results.npz)
            cat_save_path = save_path.replace('_results.npz', f'_{cat_name}_results.npz')
            
            np.savez(
                cat_save_path,
                mean_auc=cat_auc,
                sem_auc=cat_sem,
                threshold_95=cat_threshold_95,
                sig_indices=cat_sig_indices,
                real_auc_runs=real_auc_runs[:, :, i],
                perm_dist=cat_perm_dist,
                latency_earliest=cat_latencies['earliest'],
                latency_half_height=cat_latencies['half_height'],
                latency_peak=cat_latencies['peak'],
                task_name=np.array(within_category_spec['task_name']),
                category_name=np.array(cat_name),
                n_real=np.array(N_REALS),
                n_repeats_real=np.array(N_REPEATS_REAL),
                n_repeats_perm=np.array(N_REPEATS_PERM),
                n_perm=np.array(N_PERMS),
                use_groupeddata_pairing=np.array(within_category_spec['use_groupeddata_pairing']),
                use_groupeddata_pair_centering=np.array(within_category_spec['use_groupeddata_pair_centering']),
                groupeddata_mat=np.array(str(within_category_spec['groupeddata_mat'] or '')),
                matched_pair_count=np.array(matched_pair_counts[i]),
                task_id=np.array(f"{task['id']}_{cat_name}"),
                task_title=np.array(f"{task['title']} - {cat_name}")
            )
            
            # 绘制单个类别的曲线图
            plot_single_roi_result(
                roi_name=f"{roi_name} ({cat_name})",
                mean_auc=cat_auc,
                sem_auc=cat_sem,
                threshold_95=cat_threshold_95,
                sig_indices=cat_sig_indices,
                figure_title=f'{task["title"]} | {cat_name}',
                save_path=os.path.join(roi_plot_dir, f'{roi_name}_{cat_name}_curve.png')
            )

        logger(f'ROI saved (mean + 4 categories): {roi_name}')
        return
    else:
        raise ValueError(f'Unsupported mode: {mode}')

    mean_auc = np.mean(real_auc_runs, axis=0)
    sem_auc = np.std(real_auc_runs, axis=0, ddof=0) / np.sqrt(real_auc_runs.shape[0])
    threshold_95, sig_indices = cluster_permutation_significance(mean_auc, perm_dist)
    latencies = compute_latency_points(mean_auc, sig_indices)

    np.savez(
        save_path,
        mean_auc=mean_auc,
        sem_auc=sem_auc,
        threshold_95=threshold_95,
        sig_indices=sig_indices,
        real_auc_matrix=real_auc_matrix,
        real_auc_runs=real_auc_runs,
        perm_dist=perm_dist,
        latency_earliest=latencies['earliest'],
        latency_half_height=latencies['half_height'],
        latency_peak=latencies['peak'],
        task_id=np.array(task['id']),
        task_title=np.array(task['title']),
        n_real=np.array(N_REALS),
        n_repeats_real=np.array(N_REPEATS_REAL),
        n_repeats_perm=np.array(N_REPEATS_PERM),
        n_perm=np.array(N_PERMS),
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


def build_within_cv_data(mat, data_key, class0_indices, class1_indices):
    if data_key not in mat:
        raise ValueError(f'Missing matrix: {data_key}')
    data = mat[data_key]
    class0 = np.concatenate([data[idx, :, :, :] for idx in class0_indices], axis=0)
    class1 = np.concatenate([data[idx, :, :, :] for idx in class1_indices], axis=0)
    X = np.concatenate([class0, class1], axis=0)
    y = np.concatenate([np.zeros(class0.shape[0]), np.ones(class1.shape[0])])
    return X, y


def build_task1_pair_holdout_data(mat):
    if 'lg_task1' not in mat:
        raise ValueError('Missing matrix: lg_task1')
    data = mat['lg_task1']
    if data.shape[0] < 8:
        raise ValueError('Task1 requires at least 8 conditions')

    samples = []
    labels = []
    groups = []
    for pair in [0,1,3]:
        color_idx = pair * 2
        gray_idx = pair * 2 + 1
        color_trials = data[color_idx, :, :, :]
        gray_trials = data[gray_idx, :, :, :]
        for rep in range(color_trials.shape[0]):
            samples.append(color_trials[rep])
            labels.append(0.0)
            groups.append(pair)
        for rep in range(gray_trials.shape[0]):
            samples.append(gray_trials[rep])
            labels.append(1.0)
            groups.append(pair)
    X = np.stack(samples, axis=0)
    y = np.array(labels)
    g = np.array(groups)
    return X, y, g


def build_task2_gray_memory_combos(mat):
    if 'lg_task2' not in mat:
        raise ValueError('Missing matrix: lg_task2')
    data = mat['lg_task2']
    gray_green = [2, 5]
    gray_red = [8, 11]
    for idx in gray_green + gray_red:
        if idx >= data.shape[0]:
            raise ValueError('Task2 gray condition index out of range')

    green_a = data[gray_green[0], :, :, :]
    green_b = data[gray_green[1], :, :, :]
    red_a = data[gray_red[0], :, :, :]
    red_b = data[gray_red[1], :, :, :]

    combos = [
        (green_a, red_a, green_b, red_b),
        (green_a, red_b, green_b, red_a),
        (green_b, red_a, green_a, red_b),
        (green_b, red_b, green_a, red_a)
    ]
    combo_data = []
    baseline_end = np.searchsorted(TIMES, 0)
    for g_train, r_train, g_test, r_test in combos:
        X_train = np.concatenate([g_train, r_train], axis=0)
        y_train = np.concatenate([np.zeros(g_train.shape[0]), np.ones(r_train.shape[0])])
        X_test = np.concatenate([g_test, r_test], axis=0)
        y_test = np.concatenate([np.zeros(g_test.shape[0]), np.ones(r_test.shape[0])])
        X_train = baseline_zscore(X_train, baseline_end)
        X_test = baseline_zscore(X_test, baseline_end)
        if TIME_SMOOTH_WIN > 0:
            X_train = smooth_data_causal(X_train, TIME_SMOOTH_WIN)
            X_test = smooth_data_causal(X_test, TIME_SMOOTH_WIN)
        combo_data.append((X_train, y_train, X_test, y_test))
    return combo_data


def run_permutation_distribution(run_mean_curve_fn, n_timepoints):
    if not should_run_permutation_test():
        return np.full((0, n_timepoints), np.nan)
    perm_results = Parallel(n_jobs=N_JOBS)(delayed(run_mean_curve_fn)(True, i) for i in range(N_PERMS))
    return np.array(perm_results)


def run_real_curve_distribution(first_curve, run_mean_curve_fn):
    first_curve = np.asarray(first_curve, dtype=float)
    if N_REALS <= 1:
        return first_curve[None, :]
    extra_results = Parallel(n_jobs=N_JOBS)(
        delayed(run_mean_curve_fn)(RANDOM_STATE + i) for i in range(1, N_REALS)
    )
    return np.vstack([first_curve[None, :], np.asarray(extra_results, dtype=float)])


def run_decoding_over_time_cv(X, y, n_repeats=1, shuffle=False, seed=None):
    rng = np.random.RandomState(seed if seed is not None else RANDOM_STATE)
    y_use = rng.permutation(y) if shuffle else y.copy()
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=rng) if n_repeats > 1 else StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=rng)
    splits = list(cv.split(X[:, 0, 0], y_use))
    time_indices = np.arange(0, X.shape[2], DECODING_STEP)
    scores = np.zeros((len(time_indices), len(splits)))
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    for ti, t in enumerate(time_indices):
        X_t = X[:, :, t]
        for si, (train_idx, test_idx) in enumerate(splits):
            clf.fit(X_t[train_idx], y_use[train_idx])
            y_prob = clf.predict_proba(X_t[test_idx])[:, 1]
            scores[ti, si] = safe_auc(y_use[test_idx], y_prob)
    return scores


def run_decoding_over_time_cv_mean(X, y, n_repeats=1, shuffle=False, seed=None):
    return np.mean(run_decoding_over_time_cv(X, y, n_repeats=n_repeats, shuffle=shuffle, seed=seed), axis=1)


def run_decoding_over_time_group_holdout(X, y, groups, shuffle=False, seed=None):
    rng = np.random.RandomState(seed if seed is not None else RANDOM_STATE)
    y_use = rng.permutation(y) if shuffle else y.copy()
    uniq_groups = np.unique(groups)
    time_indices = np.arange(0, X.shape[2], DECODING_STEP)
    scores = np.zeros((len(time_indices), len(uniq_groups)))
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    for ti, t in enumerate(time_indices):
        X_t = X[:, :, t]
        for gi, g in enumerate(uniq_groups):
            test_idx = groups == g
            train_idx = ~test_idx
            if len(np.unique(y_use[train_idx])) < 2 or len(np.unique(y_use[test_idx])) < 2:
                scores[ti, gi] = 0.5
                continue
            clf.fit(X_t[train_idx], y_use[train_idx])
            y_prob = clf.predict_proba(X_t[test_idx])[:, 1]
            scores[ti, gi] = safe_auc(y_use[test_idx], y_prob)
    return scores


def run_decoding_over_time_group_holdout_mean(X, y, groups, shuffle=False, seed=None):
    return np.mean(run_decoding_over_time_group_holdout(X, y, groups, shuffle=shuffle, seed=seed), axis=1)


def run_decoding_over_time_task2_gray_combos(combo_data, shuffle=False, seed=None):
    rng = np.random.RandomState(seed if seed is not None else RANDOM_STATE)
    n_time = combo_data[0][0].shape[2]
    time_indices = np.arange(0, n_time, DECODING_STEP)
    scores = np.zeros((len(time_indices), len(combo_data)))
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    for ti, t in enumerate(time_indices):
        for ci, (X_train, y_train, X_test, y_test) in enumerate(combo_data):
            y_train_use = rng.permutation(y_train) if shuffle else y_train
            if len(np.unique(y_train_use)) < 2 or len(np.unique(y_test)) < 2:
                scores[ti, ci] = 0.5
                continue
            clf.fit(X_train[:, :, t], y_train_use)
            y_prob = clf.predict_proba(X_test[:, :, t])[:, 1]
            scores[ti, ci] = safe_auc(y_test, y_prob)
    return scores


def run_decoding_over_time_task2_gray_combos_mean(combo_data, shuffle=False, seed=None):
    return np.mean(run_decoding_over_time_task2_gray_combos(combo_data, shuffle=shuffle, seed=seed), axis=1)


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
    if perm_dist is None or np.size(perm_dist) == 0:
        return np.full(mean_auc.shape, np.nan, dtype=float), np.zeros_like(mean_auc, dtype=bool)
    threshold_95 = np.percentile(perm_dist, 95, axis=0)
    binary_map = mean_auc > threshold_95
    clusters, n_clusters = label(binary_map.astype(int))
    cluster_masses = []
    for ci in range(1, n_clusters + 1):
        idx = clusters == ci
        cluster_masses.append(np.sum(mean_auc[idx] - threshold_95[idx]))

    null_cluster_masses = []
    for pi in range(perm_dist.shape[0]):
        perm_curve = perm_dist[pi]
        p_binary = perm_curve > threshold_95
        p_clusters, p_count = label(p_binary.astype(int))
        max_mass = 0.0
        for ci in range(1, p_count + 1):
            idx = p_clusters == ci
            mass = float(np.sum(perm_curve[idx] - threshold_95[idx]))
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
    peak_value = mean_auc[peak_idx]

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

    return {'all': all_results, 'significant': sig_results}


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


def plot_task3_dual_overlay(task_outputs, batch_root, logger):
    key_a = 'task3_1vs4_self'
    key_b = 'task3_2vs3_self'
    if key_a not in task_outputs or key_b not in task_outputs:
        return
    sig_a = task_outputs[key_a].get('significant', {})
    sig_b = task_outputs[key_b].get('significant', {})
    all_a = task_outputs[key_a].get('all', {})
    all_b = task_outputs[key_b].get('all', {})
    union_rois = sorted(set(sig_a.keys()).union(set(sig_b.keys())))
    if not union_rois:
        logger('No ROI is significant in Task3 1vs4 or 2vs3')
        return

    n_cols = min(4, len(union_rois))
    n_rows = math.ceil(len(union_rois) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.0), squeeze=False)
    fig.suptitle(f'{SUBJECT} | Task3 Pure Color: ROI curves for ROIs significant in either decoding', fontsize=14, y=1.02)
    flat_axes = axes.flatten()
    for idx, roi in enumerate(union_rois):
        ax = flat_axes[idx]
        if roi in all_a:
            mean_auc_a, sem_auc_a, _, _ = all_a[roi]
            t_a = PLOT_TIMES[:len(mean_auc_a)]
            ax.fill_between(t_a, mean_auc_a - sem_auc_a, mean_auc_a + sem_auc_a, color='#1f77b4', alpha=0.18)
            ax.plot(t_a, mean_auc_a, color='#1f77b4', linewidth=1.6, label='Task3 1vs4')
        if roi in all_b:
            mean_auc_b, sem_auc_b, _, _ = all_b[roi]
            t_b = PLOT_TIMES[:len(mean_auc_b)]
            ax.fill_between(t_b, mean_auc_b - sem_auc_b, mean_auc_b + sem_auc_b, color='#ff7f0e', alpha=0.18)
            ax.plot(t_b, mean_auc_b, color='#ff7f0e', linewidth=1.6, label='Task3 2vs3')
        ax.axhline(0.5, color='black', linestyle=':', linewidth=0.9)
        ax.set_title(roi, fontsize=9)
        ax.set_xlim(T_START, T_END)
        ax.set_ylim(0.35, 1.0)
        ax.grid(True, linestyle='--', alpha=0.35)
        if idx % n_cols == 0:
            ax.set_ylabel('ROC AUC', fontsize=8)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Time (ms)', fontsize=8)
        ax.legend(loc='lower right', fontsize=7)
    for ax in flat_axes[len(union_rois):]:
        ax.set_visible(False)
    plt.tight_layout()
    save_path = os.path.join(batch_root, 'Fig_Task3_1vs4_and_2vs3_SignificantUnionROIs.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger(f'Dual-task overlay figure saved: {save_path}')

def baseline_zscore(X, baseline_end_idx):
    # X: (n_trials, n_channels, n_time)
    baseline = X[:, :, :baseline_end_idx]           # 取t<0的部分
    mu = baseline.mean(axis=2, keepdims=True)        # 每trial每channel的均值
    sd = baseline.std(axis=2, keepdims=True) + 1e-8  # 避免除以0
    return (X - mu) / sd

def run_decoding_per_category(mat, within_category_spec, shuffle=False, seed=None):
    data_key = within_category_spec['data_key']
    if data_key not in mat:
        raise ValueError(f'Missing matrix: {data_key}')
    data = np.asarray(mat[data_key], dtype=float)
    category_pairs = within_category_spec['category_pairs']
    category_names = within_category_spec['category_names']
    use_groupeddata_pairing = within_category_spec['use_groupeddata_pairing']
    use_groupeddata_pair_centering = within_category_spec['use_groupeddata_pair_centering']

    baseline_end = np.searchsorted(TIMES, 0)
    category_data = []
    matched_pair_counts = []

    if use_groupeddata_pairing:
        paired_categories = load_paired_category_trials(
            BASE_PATH,
            SUBJECT,
            FEATURE_KIND,
            within_category_spec['task_name'],
            within_category_spec['groupeddata_mat'],
            data,
            category_pairs,
            category_names,
        )
        for paired_category in paired_categories:
            pair_count = paired_category.color.shape[0]
            samples = np.concatenate([paired_category.color, paired_category.gray], axis=0)
            samples = baseline_zscore(samples, baseline_end)
            if TIME_SMOOTH_WIN > 0:
                samples = smooth_data_causal(samples, TIME_SMOOTH_WIN)
            color_samples = samples[:pair_count]
            gray_samples = samples[pair_count:]
            if use_groupeddata_pair_centering:
                color_samples, gray_samples = center_paired_trials(color_samples, gray_samples)
            samples, labels, groups = stack_paired_binary_trials(
                color_samples,
                gray_samples,
                paired_category.pair_ids,
            )
            category_data.append((samples, labels, groups))
            matched_pair_counts.append(paired_category.matched_count)
    else:
        for color_idx, gray_idx in category_pairs:
            X_color = data[color_idx, :, :, :]
            X_gray = data[gray_idx, :, :, :]
            X_cat = np.concatenate([X_color, X_gray], axis=0)
            y_cat = np.concatenate([
                np.zeros(X_color.shape[0]),
                np.ones(X_gray.shape[0])
            ])
            X_cat = baseline_zscore(X_cat, baseline_end)
            if TIME_SMOOTH_WIN > 0:
                X_cat = smooth_data_causal(X_cat, TIME_SMOOTH_WIN)
            category_data.append((X_cat, y_cat, None))
            matched_pair_counts.append(int(min(X_color.shape[0], X_gray.shape[0])))

    n_categories = len(category_data)
    n_time_indices = len(np.arange(0, category_data[0][0].shape[2], DECODING_STEP))

    def one_real(real_seed):
        real_curves = []
        for X_cat, y_cat, groups_cat in category_data:
            if groups_cat is None:
                scores = run_decoding_over_time_cv(
                    X_cat,
                    y_cat,
                    n_repeats=N_REPEATS_REAL,
                    shuffle=False,
                    seed=real_seed,
                )
            else:
                scores = run_grouped_auc_over_time(
                    X_cat,
                    y_cat,
                    groups_cat,
                    n_splits=N_SPLITS,
                    n_repeats=N_REPEATS_REAL,
                    decoding_step=DECODING_STEP,
                    seed=real_seed,
                    shuffle=False,
                )
            real_curves.append(np.mean(scores, axis=1))
        return np.stack(real_curves, axis=1)

    real_auc_runs = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(one_real)(RANDOM_STATE + i) for i in range(N_REALS)
        )
    )
    real_auc_matrix = np.mean(real_auc_runs, axis=0)
    real_sem_matrix = np.std(real_auc_runs, axis=0, ddof=0) / np.sqrt(real_auc_runs.shape[0])

    # 置换分布
    def one_perm(perm_seed):
        perm_aucs = []
        for X_cat, y_cat, groups_cat in category_data:
            if groups_cat is None:
                scores = run_decoding_over_time_cv(
                    X_cat,
                    y_cat,
                    n_repeats=N_REPEATS_PERM,
                    shuffle=True,
                    seed=perm_seed,
                )
            else:
                scores = run_grouped_auc_over_time(
                    X_cat,
                    y_cat,
                    groups_cat,
                    n_splits=N_SPLITS,
                    n_repeats=N_REPEATS_PERM,
                    decoding_step=DECODING_STEP,
                    seed=perm_seed,
                    shuffle=True,
                )
            perm_aucs.append(np.mean(scores, axis=1))
        return np.stack(perm_aucs, axis=1)

    # perm_dist_all 形状: (N_PERMS, n_timepoints, n_categories)
    if not should_run_permutation_test():
        perm_dist_mean = np.full((0, n_time_indices), np.nan)
        perm_dist_all = np.full((0, n_time_indices, n_categories), np.nan)
        return real_auc_matrix, real_sem_matrix, real_auc_runs, perm_dist_mean, perm_dist_all, matched_pair_counts

    perm_dist_all = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(one_perm)(i) for i in range(N_PERMS)
        )
    )

    perm_dist_mean = np.mean(perm_dist_all, axis=2)

    return real_auc_matrix, real_sem_matrix, real_auc_runs, perm_dist_mean, perm_dist_all, matched_pair_counts

if __name__ == '__main__':
    _script_start_time = time.time()
    try:
        main()
    finally:
        print(f'Total runtime: {time.time() - _script_start_time:.2f} s')
