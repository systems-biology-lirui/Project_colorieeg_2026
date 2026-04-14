import csv
import math
import os
import re
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from joblib import Parallel, delayed
from scipy.ndimage import label
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from runtime_config import load_runtime_config
from newanalyse_paths import (
    get_all_electrode_summary_path,
    get_all_electrode_task_dir,
    get_feature_dir,
    project_root,
)


BASE_PATH = project_root()

# =========================
# User Config
# =========================
SUBJECT = 'test001'
FEATURE_KIND = 'erp'  # 'erp' | 'highgamma' | 'lowgamma' | 'tfa' | 'gamma' | 'gamma_multiband'

ROI_PATTERN = '*.mat'
SKIP_ROIS = {'Unknown', 'N_A'}
MAX_ELECTRODES = None

N_SPLITS = 5
N_REPEATS_REAL = 10
N_REPEATS_PERM = N_REPEATS_REAL
N_PERMS = 0
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

RUN_PERMUTATION_TEST = False
IMPORTANCE_TOP_N = 20
BATCH_NAME = 'all_electrode_decoding_importance'

TASKS = [
    {
        'id': 'task1_color_vs_gray_per_category_all_electrodes',
        'title': 'Task 1 Color vs Gray Per-Category All-Electrode Decoding',
        'description': 'Use all available electrode sites directly without ROI grouping.',
        'mode': 'within_category_color_gray',
        'folder': 'task1_color_vs_gray_per_category',
    },
]


FEATURE_CONFIG = {
    'erp': {
        'feature_subdir': 'decoding_erp_features',
        'field_prefix': 'erp',
        'channel_mode': 'roi_map',
        'time_field': None,
        'channel_field': None,
    },
    'highgamma': {
        'feature_subdir': 'decoding_highgamma_features',
        'field_prefix': 'hg',
        'channel_mode': 'roi_map',
        'time_field': None,
        'channel_field': None,
    },
    'lowgamma': {
        'feature_subdir': 'decoding_lowgamma_features',
        'field_prefix': 'lg',
        'channel_mode': 'roi_map',
        'time_field': None,
        'channel_field': None,
    },
    'tfa': {
        'feature_subdir': 'decoding_tfa_features',
        'field_prefix': 'tfa',
        'channel_mode': 'metadata',
        'time_field': 'tfa_time_ms',
        'channel_field': 'tfa_roi_channels',
    },
    'gamma': {
        'feature_subdir': 'decoding_gamma_features',
        'field_prefix': 'g',
        'channel_mode': 'roi_map',
        'time_field': None,
        'channel_field': None,
    },
    'gamma_multiband': {
        'feature_subdir': 'decoding_gamma_multiband_features',
        'field_prefix': 'gmb',
        'channel_mode': 'multiband',
        'time_field': 'gmb_time_ms',
        'channel_field': 'gmb_roi_channels',
        'feature_channel_index_field': 'gmb_feature_channel_index',
    },
}


_RUNTIME_CFG = load_runtime_config(__file__, sections=('python_defaults', 'sec3_defaults'))
if _RUNTIME_CFG:
    SUBJECT = str(_RUNTIME_CFG.get('subject', SUBJECT))
    FEATURE_KIND = str(_RUNTIME_CFG.get('feature_kind', FEATURE_KIND))
    ROI_PATTERN = str(_RUNTIME_CFG.get('roi_pattern', ROI_PATTERN))
    MAX_ELECTRODES = _RUNTIME_CFG.get('max_electrodes', MAX_ELECTRODES)
    N_SPLITS = int(_RUNTIME_CFG.get('n_splits', N_SPLITS))
    N_REPEATS_REAL = int(_RUNTIME_CFG.get('n_repeats_real', N_REPEATS_REAL))
    N_REPEATS_PERM = int(_RUNTIME_CFG.get('n_repeats_perm', N_REPEATS_PERM))
    N_PERMS = int(_RUNTIME_CFG.get('n_perms', N_PERMS))
    TIME_SMOOTH_WIN = int(_RUNTIME_CFG.get('time_smooth_win', TIME_SMOOTH_WIN))
    DECODING_STEP = int(_RUNTIME_CFG.get('decoding_step', DECODING_STEP))
    RANDOM_STATE = int(_RUNTIME_CFG.get('random_state', RANDOM_STATE))
    N_JOBS = int(_RUNTIME_CFG.get('n_jobs', N_JOBS))
    RUN_PERMUTATION_TEST = bool(_RUNTIME_CFG.get('run_permutation_test', RUN_PERMUTATION_TEST))
    IMPORTANCE_TOP_N = int(_RUNTIME_CFG.get('importance_top_n', IMPORTANCE_TOP_N))
    BATCH_NAME = str(_RUNTIME_CFG.get('batch_name', BATCH_NAME))
    if 'tasks' in _RUNTIME_CFG:
        TASKS = _RUNTIME_CFG['tasks']

TIMES = np.linspace(T_START, T_END, N_POINTS)
PLOT_TIMES = TIMES[::DECODING_STEP]


def validate_config():
    if FEATURE_KIND not in FEATURE_CONFIG:
        raise ValueError(f'Unsupported FEATURE_KIND: {FEATURE_KIND}')
    if not TASKS:
        raise ValueError('TASKS cannot be empty.')


def should_run_permutation_test():
    return bool(RUN_PERMUTATION_TEST and N_PERMS > 0)


def get_perm_tag():
    return f'perm{N_PERMS}' if should_run_permutation_test() else 'real_only'


def sanitize_name(name):
    name = re.sub(r'[^a-zA-Z0-9_]+', '_', str(name))
    name = re.sub(r'^_+|_+$', '', name)
    return name or 'Unknown'


def matlab_cellstr_to_list(value):
    array = np.asarray(value)
    if array.size == 0:
        return []
    result = []
    for item in np.ravel(array, order='F'):
        if isinstance(item, np.ndarray):
            flattened = np.ravel(item, order='F')
            if flattened.size == 1:
                result.append(str(flattened[0]).strip())
            else:
                result.append(''.join(str(v) for v in flattened.tolist()).strip())
        else:
            result.append(str(item).strip())
    return result


def extract_channel_labels(epoch_struct):
    channels = epoch_struct.ch
    if isinstance(channels, np.ndarray):
        return [str(item.labels) for item in channels.flat]
    return [str(channels.labels)]


def build_subject_paths(subject):
    data_dir = BASE_PATH / 'processed_data' / subject
    feature_dir = get_feature_dir(BASE_PATH, FEATURE_CONFIG[FEATURE_KIND]['feature_subdir'], subject)
    loc_file = data_dir / f'{subject}_ieegloc.xlsx'
    task_to_file = {
        'task1': data_dir / 'task1_ERP_epoched.mat',
        'task2': data_dir / 'task2_ERP_epoched.mat',
        'task3': data_dir / 'task3_ERP_epoched.mat',
    }
    return data_dir, feature_dir, loc_file, task_to_file


def load_common_channels(task_to_file):
    channel_sets = []
    for task_name in ('task1', 'task2', 'task3'):
        mat = sio.loadmat(
            task_to_file[task_name],
            variable_names=['epoch'],
            squeeze_me=True,
            struct_as_record=False,
        )
        labels = extract_channel_labels(mat['epoch'])
        channel_sets.append(set(labels))
    common = sorted(channel_sets[0] & channel_sets[1] & channel_sets[2])
    if not common:
        raise RuntimeError('No common channels found across task1/task2/task3.')
    return common


def get_roi_map(loc_file, channel_labels):
    loc_table = pd.read_excel(loc_file)
    columns = list(loc_table.columns)
    lower_columns = [str(col).strip().lower() for col in columns]

    name_idx = None
    for candidate in ('name', 'channel', 'electrode', 'label'):
        if candidate in lower_columns:
            name_idx = lower_columns.index(candidate)
            break
    if name_idx is None:
        name_idx = 0
    name_col = columns[name_idx]

    roi_idx = None
    for candidate in ('aal3', 'aal3_mni_linear_', 'aal3_label', 'aal3 (mni-linear)'):
        if candidate in lower_columns:
            roi_idx = lower_columns.index(candidate)
            break
    if roi_idx is None:
        for candidate in ('roi', 'region', 'anatomy', 'dk_lobe', 'lobe'):
            if candidate in lower_columns:
                roi_idx = lower_columns.index(candidate)
                break
    if roi_idx is None:
        raise RuntimeError('Could not identify ROI column in location file.')
    roi_col = columns[roi_idx]

    roi_map = {}
    table_names = loc_table[name_col].astype(str).str.strip().str.lower()
    for channel_name in channel_labels:
        row_indices = np.flatnonzero(table_names == channel_name.strip().lower())
        if len(row_indices) == 0:
            target_rois = ['Unknown']
        else:
            target_rois = []
            for index in row_indices:
                value = loc_table.iloc[index][roi_col]
                roi_name = sanitize_name(value) if not pd.isna(value) else 'Unknown'
                target_rois.append(roi_name)
            target_rois = sorted(set(target_rois))

        for roi_name in target_rois:
            roi_map.setdefault(roi_name, []).append(channel_name)
    return roi_map


def get_time_vector(mat, n_time):
    time_field = FEATURE_CONFIG[FEATURE_KIND].get('time_field')
    if time_field and time_field in mat:
        times = np.asarray(mat[time_field], dtype=float).reshape(-1)
        if times.size == n_time:
            return times
    return T_START + np.arange(n_time, dtype=float) * (1000.0 / FS)


def build_task_field(task_name):
    return f"{FEATURE_CONFIG[FEATURE_KIND]['field_prefix']}_{task_name}"


def infer_channel_names(mat, roi_name, roi_map, n_channels):
    config = FEATURE_CONFIG[FEATURE_KIND]
    if config['channel_mode'] in {'metadata', 'multiband'}:
        channel_field = config.get('channel_field')
        names = matlab_cellstr_to_list(mat.get(channel_field, np.array([])))
        if len(names) == n_channels:
            return names
    names = list(roi_map.get(roi_name, []))
    if len(names) == n_channels:
        return names
    return [f'{roi_name}__Ch{idx + 1}' for idx in range(n_channels)]


def select_channel_blocks(task_arrays, channel_index, n_channels_total):
    blocks = {}
    for task_name, task_array in task_arrays.items():
        blocks[task_name] = task_array[:, :, channel_index:channel_index + 1, :]
    return blocks


def select_multiband_blocks(task_arrays, channel_position, channel_index_vector):
    blocks = {}
    feature_indices = np.flatnonzero(channel_index_vector == (channel_position + 1))
    for task_name, task_array in task_arrays.items():
        blocks[task_name] = task_array[:, :, feature_indices, :]
    return blocks, feature_indices


def collect_all_electrode_data(subject):
    _, feature_dir, loc_file, task_to_file = build_subject_paths(subject)
    if not feature_dir.exists():
        raise FileNotFoundError(f'Feature directory not found: {feature_dir}')

    roi_map = {}
    if FEATURE_CONFIG[FEATURE_KIND]['channel_mode'] == 'roi_map':
        common_channels = load_common_channels(task_to_file)
        roi_map = get_roi_map(loc_file, common_channels)

    mat_files = sorted(feature_dir.glob(ROI_PATTERN))
    mat_files = [path for path in mat_files if path.is_file() and path.suffix == '.mat']
    task_names = ('task1', 'task2', 'task3')
    task_blocks = {task_name: [] for task_name in task_names}
    expected_shapes = {}
    time_vector = None
    electrode_records = []
    source_rois = {}
    seen_channels = set()

    for mat_file in mat_files:
        roi_name = mat_file.stem
        if roi_name in SKIP_ROIS:
            continue
        try:
            mat = sio.loadmat(mat_file)
        except Exception as exc:
            print(f'Skip unreadable ROI file: {mat_file.name} ({type(exc).__name__}: {exc})')
            continue
        task_arrays = {}
        skip_file = False
        for task_name in task_names:
            field = build_task_field(task_name)
            if field not in mat:
                skip_file = True
                break
            task_array = np.asarray(mat[field], dtype=float)
            expected = expected_shapes.get(task_name)
            current_shape = (task_array.shape[0], task_array.shape[1], task_array.shape[-1])
            if expected is None:
                expected_shapes[task_name] = current_shape
            elif expected != current_shape:
                raise ValueError(
                    f'Inconsistent shape for {task_name} in {mat_file.name}: {current_shape} vs {expected}'
                )
            task_arrays[task_name] = task_array
        if skip_file:
            continue

        time_vector = get_time_vector(mat, task_arrays['task1'].shape[-1])

        if FEATURE_CONFIG[FEATURE_KIND]['channel_mode'] == 'multiband':
            channel_index_field = FEATURE_CONFIG[FEATURE_KIND]['feature_channel_index_field']
            channel_index_vector = np.asarray(mat[channel_index_field]).reshape(-1).astype(int)
            n_channels = int(np.max(channel_index_vector))
            channel_names = infer_channel_names(mat, roi_name, roi_map, n_channels)
            for channel_position, channel_name in enumerate(channel_names):
                source_rois.setdefault(channel_name, set()).add(roi_name)
                if channel_name in seen_channels:
                    continue
                seen_channels.add(channel_name)
                channel_blocks, feature_indices = select_multiband_blocks(
                    task_arrays,
                    channel_position,
                    channel_index_vector,
                )
                start_index = sum(block.shape[2] for block in task_blocks['task1'])
                for task_name in task_names:
                    task_blocks[task_name].append(channel_blocks[task_name])
                stop_index = start_index + len(feature_indices)
                electrode_records.append(
                    {
                        'channel': channel_name,
                        'feature_start': start_index,
                        'feature_stop': stop_index,
                        'feature_count': len(feature_indices),
                    }
                )
        else:
            n_channels = task_arrays['task1'].shape[2]
            channel_names = infer_channel_names(mat, roi_name, roi_map, n_channels)
            for channel_index, channel_name in enumerate(channel_names):
                source_rois.setdefault(channel_name, set()).add(roi_name)
                if channel_name in seen_channels:
                    continue
                seen_channels.add(channel_name)
                start_index = len(task_blocks['task1'])
                channel_blocks = select_channel_blocks(task_arrays, channel_index, n_channels)
                for task_name in task_names:
                    task_blocks[task_name].append(channel_blocks[task_name])
                electrode_records.append(
                    {
                        'channel': channel_name,
                        'feature_start': start_index,
                        'feature_stop': start_index + 1,
                        'feature_count': 1,
                    }
                )

    if not electrode_records:
        raise RuntimeError(f'No electrode data found in {feature_dir}')

    if MAX_ELECTRODES is not None:
        electrode_records = electrode_records[:MAX_ELECTRODES]
        max_feature_stop = electrode_records[-1]['feature_stop']
    else:
        max_feature_stop = None

    combined = {}
    for task_name in task_names:
        data = np.concatenate(task_blocks[task_name], axis=2)
        if max_feature_stop is not None:
            data = data[:, :, :max_feature_stop, :]
        combined[task_name] = data

    for record in electrode_records:
        record['source_rois'] = ';'.join(sorted(source_rois.get(record['channel'], [])))

    return combined, np.asarray(time_vector, dtype=float), electrode_records


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


def build_within_cv_prepared(data_bank, task_name, class0_indices, class1_indices):
    data = data_bank[task_name]
    class0 = np.concatenate([data[idx, :, :, :] for idx in class0_indices], axis=0)
    class1 = np.concatenate([data[idx, :, :, :] for idx in class1_indices], axis=0)
    samples = np.concatenate([class0, class1], axis=0)
    labels = np.concatenate([np.zeros(class0.shape[0]), np.ones(class1.shape[0])])
    baseline_end = np.searchsorted(TIMES, 0)
    samples = baseline_zscore(samples, baseline_end)
    if TIME_SMOOTH_WIN > 0:
        samples = smooth_data_causal(samples, TIME_SMOOTH_WIN)
    return {'mode': 'within_cv', 'X': samples, 'y': labels}


def build_task1_pair_holdout_prepared(data_bank):
    data = data_bank['task1']
    samples = []
    labels = []
    groups = []
    for pair in [0, 1, 3]:
        color_index = pair * 2
        gray_index = pair * 2 + 1
        color_trials = data[color_index, :, :, :]
        gray_trials = data[gray_index, :, :, :]
        for repeat_index in range(color_trials.shape[0]):
            samples.append(color_trials[repeat_index])
            labels.append(0.0)
            groups.append(pair)
        for repeat_index in range(gray_trials.shape[0]):
            samples.append(gray_trials[repeat_index])
            labels.append(1.0)
            groups.append(pair)
    samples = np.stack(samples, axis=0)
    baseline_end = np.searchsorted(TIMES, 0)
    samples = baseline_zscore(samples, baseline_end)
    if TIME_SMOOTH_WIN > 0:
        samples = smooth_data_causal(samples, TIME_SMOOTH_WIN)
    return {'mode': 'pair_holdout_task1', 'X': samples, 'y': np.array(labels), 'groups': np.array(groups)}


def build_task2_gray_memory_combos_prepared(data_bank):
    data = data_bank['task2']
    gray_green = [2, 5]
    gray_red = [8, 11]
    combos = [
        (data[gray_green[0]], data[gray_red[0]], data[gray_green[1]], data[gray_red[1]]),
        (data[gray_green[0]], data[gray_red[1]], data[gray_green[1]], data[gray_red[0]]),
        (data[gray_green[1]], data[gray_red[0]], data[gray_green[0]], data[gray_red[1]]),
        (data[gray_green[1]], data[gray_red[1]], data[gray_green[0]], data[gray_red[0]]),
    ]
    baseline_end = np.searchsorted(TIMES, 0)
    prepared = []
    for green_train, red_train, green_test, red_test in combos:
        train_x = np.concatenate([green_train, red_train], axis=0)
        train_y = np.concatenate([np.zeros(green_train.shape[0]), np.ones(red_train.shape[0])])
        test_x = np.concatenate([green_test, red_test], axis=0)
        test_y = np.concatenate([np.zeros(green_test.shape[0]), np.ones(red_test.shape[0])])
        train_x = baseline_zscore(train_x, baseline_end)
        test_x = baseline_zscore(test_x, baseline_end)
        if TIME_SMOOTH_WIN > 0:
            train_x = smooth_data_causal(train_x, TIME_SMOOTH_WIN)
            test_x = smooth_data_causal(test_x, TIME_SMOOTH_WIN)
        prepared.append((train_x, train_y, test_x, test_y))
    return {'mode': 'cross_combo_task2_gray', 'combo_data': prepared}


def build_within_category_prepared(data_bank):
    data = data_bank['task1']
    baseline_end = np.searchsorted(TIMES, 0)
    category_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    category_names = ['face', 'body', 'object', 'scene']
    category_data = []
    for color_index, gray_index in category_pairs:
        color_samples = data[color_index, :, :, :]
        gray_samples = data[gray_index, :, :, :]
        samples = np.concatenate([color_samples, gray_samples], axis=0)
        labels = np.concatenate([np.zeros(color_samples.shape[0]), np.ones(gray_samples.shape[0])])
        samples = baseline_zscore(samples, baseline_end)
        if TIME_SMOOTH_WIN > 0:
            samples = smooth_data_causal(samples, TIME_SMOOTH_WIN)
        category_data.append((samples, labels))
    return {'mode': 'within_category_color_gray', 'category_data': category_data, 'category_names': category_names}


def prepare_task_data(data_bank, task):
    mode = task['mode']
    if mode == 'within_cv':
        return build_within_cv_prepared(data_bank, task['task_name'], task['class0'], task['class1'])
    if mode == 'pair_holdout_task1':
        return build_task1_pair_holdout_prepared(data_bank)
    if mode == 'cross_combo_task2_gray':
        return build_task2_gray_memory_combos_prepared(data_bank)
    if mode == 'within_category_color_gray':
        return build_within_category_prepared(data_bank)
    raise ValueError(f"Unsupported mode: {mode}")


def infer_n_timepoints(prepared):
    mode = prepared['mode']
    if mode in {'within_cv', 'pair_holdout_task1'}:
        return len(np.arange(0, prepared['X'].shape[2], DECODING_STEP))
    if mode == 'cross_combo_task2_gray':
        return len(np.arange(0, prepared['combo_data'][0][0].shape[2], DECODING_STEP))
    if mode == 'within_category_color_gray':
        return len(np.arange(0, prepared['category_data'][0][0].shape[2], DECODING_STEP))
    raise ValueError(f"Unsupported mode: {mode}")


def select_prepared_features(prepared, feature_indices):
    mode = prepared['mode']
    if mode == 'within_cv':
        return {'mode': mode, 'X': prepared['X'][:, feature_indices, :], 'y': prepared['y']}
    if mode == 'pair_holdout_task1':
        return {
            'mode': mode,
            'X': prepared['X'][:, feature_indices, :],
            'y': prepared['y'],
            'groups': prepared['groups'],
        }
    if mode == 'cross_combo_task2_gray':
        combo_data = []
        for train_x, train_y, test_x, test_y in prepared['combo_data']:
            combo_data.append((train_x[:, feature_indices, :], train_y, test_x[:, feature_indices, :], test_y))
        return {'mode': mode, 'combo_data': combo_data}
    if mode == 'within_category_color_gray':
        category_data = []
        for samples, labels in prepared['category_data']:
            category_data.append((samples[:, feature_indices, :], labels))
        return {'mode': mode, 'category_data': category_data, 'category_names': prepared['category_names']}
    raise ValueError(f"Unsupported mode: {mode}")


def safe_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5


def run_decoding_over_time_cv(samples, labels, n_repeats=1, shuffle=False, seed=None):
    rng = np.random.RandomState(seed if seed is not None else RANDOM_STATE)
    labels_use = rng.permutation(labels) if shuffle else labels.copy()
    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=rng)
    else:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=rng)
    splits = list(cv.split(samples[:, 0, 0], labels_use))
    time_indices = np.arange(0, samples.shape[2], DECODING_STEP)
    scores = np.zeros((len(time_indices), len(splits)))
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    for time_offset, time_index in enumerate(time_indices):
        samples_at_time = samples[:, :, time_index]
        for split_index, (train_idx, test_idx) in enumerate(splits):
            clf.fit(samples_at_time[train_idx], labels_use[train_idx])
            probabilities = clf.predict_proba(samples_at_time[test_idx])[:, 1]
            scores[time_offset, split_index] = safe_auc(labels_use[test_idx], probabilities)
    return scores


def run_decoding_over_time_group_holdout(samples, labels, groups, shuffle=False, seed=None):
    rng = np.random.RandomState(seed if seed is not None else RANDOM_STATE)
    labels_use = rng.permutation(labels) if shuffle else labels.copy()
    unique_groups = np.unique(groups)
    time_indices = np.arange(0, samples.shape[2], DECODING_STEP)
    scores = np.zeros((len(time_indices), len(unique_groups)))
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    for time_offset, time_index in enumerate(time_indices):
        samples_at_time = samples[:, :, time_index]
        for group_index, group_id in enumerate(unique_groups):
            test_idx = groups == group_id
            train_idx = ~test_idx
            if len(np.unique(labels_use[train_idx])) < 2 or len(np.unique(labels_use[test_idx])) < 2:
                scores[time_offset, group_index] = 0.5
                continue
            clf.fit(samples_at_time[train_idx], labels_use[train_idx])
            probabilities = clf.predict_proba(samples_at_time[test_idx])[:, 1]
            scores[time_offset, group_index] = safe_auc(labels_use[test_idx], probabilities)
    return scores


def run_decoding_over_time_task2_gray_combos(combo_data, shuffle=False, seed=None):
    rng = np.random.RandomState(seed if seed is not None else RANDOM_STATE)
    n_time = combo_data[0][0].shape[2]
    time_indices = np.arange(0, n_time, DECODING_STEP)
    scores = np.zeros((len(time_indices), len(combo_data)))
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    for time_offset, time_index in enumerate(time_indices):
        for combo_index, (train_x, train_y, test_x, test_y) in enumerate(combo_data):
            train_labels = rng.permutation(train_y) if shuffle else train_y
            if len(np.unique(train_labels)) < 2 or len(np.unique(test_y)) < 2:
                scores[time_offset, combo_index] = 0.5
                continue
            clf.fit(train_x[:, :, time_index], train_labels)
            probabilities = clf.predict_proba(test_x[:, :, time_index])[:, 1]
            scores[time_offset, combo_index] = safe_auc(test_y, probabilities)
    return scores


def run_decoding_per_category(category_data, shuffle=False, seed=None, n_repeats=1):
    n_categories = len(category_data)
    n_timepoints = len(np.arange(0, category_data[0][0].shape[2], DECODING_STEP))
    scores = np.zeros((n_timepoints, n_categories))
    for category_index, (samples, labels) in enumerate(category_data):
        category_scores = run_decoding_over_time_cv(samples, labels, n_repeats=n_repeats, shuffle=shuffle, seed=seed)
        scores[:, category_index] = np.mean(category_scores, axis=1)
    return scores


def compute_mean_and_sem(score_matrix):
    mean_auc = np.mean(score_matrix, axis=1)
    sem_auc = np.std(score_matrix, axis=1, ddof=0) / np.sqrt(score_matrix.shape[1])
    return mean_auc, sem_auc


def run_real_decoding(prepared):
    mode = prepared['mode']
    if mode == 'within_cv':
        return run_decoding_over_time_cv(prepared['X'], prepared['y'], n_repeats=N_REPEATS_REAL, shuffle=False, seed=RANDOM_STATE)
    if mode == 'pair_holdout_task1':
        return run_decoding_over_time_group_holdout(prepared['X'], prepared['y'], prepared['groups'], shuffle=False, seed=RANDOM_STATE)
    if mode == 'cross_combo_task2_gray':
        return run_decoding_over_time_task2_gray_combos(prepared['combo_data'], shuffle=False, seed=RANDOM_STATE)
    if mode == 'within_category_color_gray':
        return run_decoding_per_category(prepared['category_data'], shuffle=False, seed=RANDOM_STATE, n_repeats=N_REPEATS_REAL)
    raise ValueError(f"Unsupported mode: {mode}")


def run_perm_mean_curve(prepared, seed):
    mode = prepared['mode']
    if mode == 'within_cv':
        scores = run_decoding_over_time_cv(prepared['X'], prepared['y'], n_repeats=N_REPEATS_PERM, shuffle=True, seed=seed)
        return np.mean(scores, axis=1)
    if mode == 'pair_holdout_task1':
        scores = run_decoding_over_time_group_holdout(prepared['X'], prepared['y'], prepared['groups'], shuffle=True, seed=seed)
        return np.mean(scores, axis=1)
    if mode == 'cross_combo_task2_gray':
        scores = run_decoding_over_time_task2_gray_combos(prepared['combo_data'], shuffle=True, seed=seed)
        return np.mean(scores, axis=1)
    if mode == 'within_category_color_gray':
        scores = run_decoding_per_category(prepared['category_data'], shuffle=True, seed=seed, n_repeats=N_REPEATS_PERM)
        return np.mean(scores, axis=1)
    raise ValueError(f"Unsupported mode: {mode}")


def run_permutation_distribution(prepared, n_timepoints):
    if not should_run_permutation_test():
        return np.full((0, n_timepoints), np.nan)
    perm_results = Parallel(n_jobs=N_JOBS)(delayed(run_perm_mean_curve)(prepared, perm_seed) for perm_seed in range(N_PERMS))
    return np.asarray(perm_results)


def cluster_permutation_significance(mean_auc, perm_dist):
    if perm_dist is None or np.size(perm_dist) == 0:
        return np.full(mean_auc.shape, np.nan, dtype=float), np.zeros_like(mean_auc, dtype=bool)
    threshold_95 = np.percentile(perm_dist, 95, axis=0)
    binary_map = mean_auc > threshold_95
    clusters, n_clusters = label(binary_map.astype(int))
    cluster_masses = []
    for cluster_id in range(1, n_clusters + 1):
        indices = clusters == cluster_id
        cluster_masses.append(np.sum(mean_auc[indices] - threshold_95[indices]))

    null_cluster_masses = []
    for perm_curve in perm_dist:
        perm_binary = perm_curve > threshold_95
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


def plot_full_curve(mean_auc, sem_auc, threshold_95, sig_indices, title, save_path):
    plot_times = PLOT_TIMES[:len(mean_auc)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(plot_times, mean_auc - sem_auc, mean_auc + sem_auc, color='#1f77b4', alpha=0.25)
    ax.plot(plot_times, mean_auc, color='#1f77b4', linewidth=1.8, label='Mean ROC AUC')
    if np.any(np.isfinite(threshold_95)):
        ax.plot(plot_times, threshold_95, color='#d62728', linestyle='--', linewidth=1.2, label='95% permutation threshold')
    if np.any(sig_indices):
        ax.fill_between(plot_times, 0, 1, where=sig_indices[:len(plot_times)], color='gray', alpha=0.25, transform=ax.get_xaxis_transform(), label='Significant cluster')
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1.0, label='Chance')
    ax.axvline(0.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('ROC AUC')
    ax.set_xlim(T_START, T_END)
    ax.set_ylim(0.35, 1.0)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_importance_heatmap(summary_rows, delta_matrix, save_path):
    if not summary_rows:
        return
    top_rows = summary_rows[:min(IMPORTANCE_TOP_N, len(summary_rows))]
    matrix = np.stack([row['delta_auc'] for row in top_rows], axis=0)
    labels = [row['channel'] for row in top_rows]
    plot_times = PLOT_TIMES[:matrix.shape[1]]
    vmax = max(float(np.percentile(np.abs(matrix), 95)), 1e-6)

    fig_height = max(4.0, 0.42 * len(labels) + 2.2)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    image = ax.imshow(
        matrix,
        aspect='auto',
        cmap='RdBu_r',
        vmin=-vmax,
        vmax=vmax,
        extent=[plot_times[0], plot_times[-1], len(labels) - 0.5, -0.5],
    )
    ax.axvline(0.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Time (ms)')
    ax.set_title('Leave-One-Electrode-Out Delta AUC')
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label('Full model AUC - leave-one-out AUC')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_importance_bar(summary_rows, save_path):
    if not summary_rows:
        return
    top_rows = summary_rows[:min(IMPORTANCE_TOP_N, len(summary_rows))]
    labels = [row['channel'] for row in top_rows]
    values = [row['peak_delta_auc'] for row in top_rows]
    fig, ax = plt.subplots(figsize=(11, max(4.0, 0.45 * len(labels) + 1.8)))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color='#4c78a8', edgecolor='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Peak Delta AUC')
    ax.set_title('Top Electrode Contributions')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def write_summary_csv(rows, save_path):
    if not rows:
        return
    fieldnames = [
        'rank',
        'channel',
        'source_rois',
        'feature_count',
        'peak_delta_auc',
        'peak_delta_time_ms',
        'mean_delta_auc',
        'peak_single_auc',
        'peak_single_time_ms',
        'mean_single_auc',
    ]
    with open(save_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def compute_importance(prepared, electrode_records):
    full_score_matrix = run_real_decoding(prepared)
    full_mean_auc, full_sem_auc = compute_mean_and_sem(full_score_matrix)
    n_timepoints = full_mean_auc.shape[0]
    perm_dist = run_permutation_distribution(prepared, n_timepoints)
    threshold_95, sig_indices = cluster_permutation_significance(full_mean_auc, perm_dist)
    latencies = compute_latency_points(full_mean_auc, sig_indices)

    all_feature_indices = np.arange(prepared['X'].shape[1] if prepared['mode'] in {'within_cv', 'pair_holdout_task1'} else infer_total_features(prepared))

    def evaluate_one(record_index, record):
        feature_indices = np.arange(record['feature_start'], record['feature_stop'])
        keep_indices = np.setdiff1d(all_feature_indices, feature_indices, assume_unique=True)

        if keep_indices.size == 0:
            loo_mean_auc = np.full_like(full_mean_auc, 0.5)
        else:
            loo_prepared = select_prepared_features(prepared, keep_indices)
            loo_scores = run_real_decoding(loo_prepared)
            loo_mean_auc, _ = compute_mean_and_sem(loo_scores)

        single_prepared = select_prepared_features(prepared, feature_indices)
        single_scores = run_real_decoding(single_prepared)
        single_mean_auc, _ = compute_mean_and_sem(single_scores)

        delta_auc = full_mean_auc - loo_mean_auc
        peak_delta_index = int(np.argmax(delta_auc))
        peak_single_index = int(np.argmax(single_mean_auc))
        return {
            'electrode_index': record_index,
            'channel': record['channel'],
            'source_rois': record['source_rois'],
            'feature_count': record['feature_count'],
            'delta_auc': delta_auc,
            'single_auc': single_mean_auc,
            'peak_delta_auc': float(delta_auc[peak_delta_index]),
            'peak_delta_time_ms': float(PLOT_TIMES[min(peak_delta_index, len(PLOT_TIMES) - 1)]),
            'mean_delta_auc': float(np.mean(delta_auc)),
            'peak_single_auc': float(single_mean_auc[peak_single_index]),
            'peak_single_time_ms': float(PLOT_TIMES[min(peak_single_index, len(PLOT_TIMES) - 1)]),
            'mean_single_auc': float(np.mean(single_mean_auc)),
        }

    rows = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_one)(index, record) for index, record in enumerate(electrode_records)
    )

    rows.sort(key=lambda item: item['peak_delta_auc'], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row['rank'] = rank

    delta_matrix = np.stack([row['delta_auc'] for row in rows], axis=0)
    single_matrix = np.stack([row['single_auc'] for row in rows], axis=0)
    return {
        'full_score_matrix': full_score_matrix,
        'full_mean_auc': full_mean_auc,
        'full_sem_auc': full_sem_auc,
        'perm_dist': perm_dist,
        'threshold_95': threshold_95,
        'sig_indices': sig_indices,
        'latencies': latencies,
        'rows': rows,
        'delta_matrix': delta_matrix,
        'single_matrix': single_matrix,
    }


def infer_total_features(prepared):
    mode = prepared['mode']
    if mode == 'cross_combo_task2_gray':
        return prepared['combo_data'][0][0].shape[1]
    if mode == 'within_category_color_gray':
        return prepared['category_data'][0][0].shape[1]
    return prepared['X'].shape[1]


def build_output_dirs(task):
    base_dir = get_all_electrode_task_dir(
        BASE_PATH,
        task_id=task['id'],
        data_type=FEATURE_KIND,
        subject=SUBJECT,
        perm_tag=get_perm_tag(),
        batch_name=BATCH_NAME,
    )
    dirs = {
        'base': base_dir,
        'figures': base_dir / 'figures',
        'tables': base_dir / 'tables',
        'mat': base_dir / 'mat',
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main():
    validate_config()
    data_bank, times_ms, electrode_records = collect_all_electrode_data(SUBJECT)
    print(f'Collected {len(electrode_records)} unique electrodes for {SUBJECT} | {FEATURE_KIND}')

    summary_records = []
    for task in TASKS:
        prepared = prepare_task_data(data_bank, task)
        results = compute_importance(prepared, electrode_records)
        output_dirs = build_output_dirs(task)

        full_curve_path = output_dirs['figures'] / 'full_model_curve.png'
        heatmap_path = output_dirs['figures'] / 'top_electrode_importance_heatmap.png'
        bar_path = output_dirs['figures'] / 'top_electrode_importance_bar.png'
        table_path = output_dirs['tables'] / 'electrode_importance_summary.csv'
        mat_path = output_dirs['mat'] / 'all_electrode_results.npz'

        plot_full_curve(
            results['full_mean_auc'],
            results['full_sem_auc'],
            results['threshold_95'],
            results['sig_indices'],
            f"{SUBJECT} | {FEATURE_KIND} | {task['title']}",
            full_curve_path,
        )
        plot_importance_heatmap(results['rows'], results['delta_matrix'], heatmap_path)
        plot_top_importance_bar(results['rows'], bar_path)
        write_summary_csv(results['rows'], table_path)

        np.savez_compressed(
            mat_path,
            full_score_matrix=results['full_score_matrix'],
            full_mean_auc=results['full_mean_auc'],
            full_sem_auc=results['full_sem_auc'],
            threshold_95=results['threshold_95'],
            sig_indices=results['sig_indices'],
            perm_dist=results['perm_dist'],
            delta_matrix=results['delta_matrix'],
            single_matrix=results['single_matrix'],
            channel_names=np.array([row['channel'] for row in results['rows']], dtype=object),
            source_rois=np.array([row['source_rois'] for row in results['rows']], dtype=object),
            times_ms=np.asarray(times_ms, dtype=float),
            plot_times_ms=np.asarray(PLOT_TIMES[:results['full_mean_auc'].shape[0]], dtype=float),
            generated_at=np.array(datetime.now().isoformat()),
        )

        print(f'Saved all-electrode results: {mat_path}')
        summary_records.append(
            {
                'task_id': task['id'],
                'feature_kind': FEATURE_KIND,
                'subject': SUBJECT,
                'n_electrodes': len(electrode_records),
                'full_peak_auc': float(np.max(results['full_mean_auc'])),
                'top_channel': results['rows'][0]['channel'] if results['rows'] else '',
                'top_peak_delta_auc': results['rows'][0]['peak_delta_auc'] if results['rows'] else np.nan,
                'table_path': str(table_path),
                'curve_path': str(full_curve_path),
            }
        )

    summary_path = get_all_electrode_summary_path(
        BASE_PATH,
        data_type=FEATURE_KIND,
        subject=SUBJECT,
        perm_tag=get_perm_tag(),
        batch_name=BATCH_NAME,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['task_id', 'feature_kind', 'subject', 'n_electrodes', 'full_peak_auc', 'top_channel', 'top_peak_delta_auc', 'table_path', 'curve_path'],
        )
        writer.writeheader()
        writer.writerows(summary_records)
    print(f'Saved summary: {summary_path}')


if __name__ == '__main__':
    main()