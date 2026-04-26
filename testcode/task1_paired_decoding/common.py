import argparse
import sys
import time
from dataclasses import dataclass, replace
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEWANALYSE_ROOT = PROJECT_ROOT / 'newanalyse'
if str(NEWANALYSE_ROOT) not in sys.path:
    sys.path.insert(0, str(NEWANALYSE_ROOT))

from newanalyse_paths import get_feature_dir, get_task_groupeddata_path, project_root, result_root, sanitize_token


T_START = -100.0
T_END = 1000.0

DEFAULT_SUBJECT = 'test001'
DEFAULT_FEATURE_KIND = 'erp'
DEFAULT_ROI_NAME = 'Color_with_sti'
DEFAULT_TASK_FIELD = None
DEFAULT_GROUPED_DATA_MAT = None
DEFAULT_COLOR_CONDITION_INDICES = [0, 2, 4, 6]
DEFAULT_GRAY_CONDITION_INDICES = [1, 3, 5, 7]
DEFAULT_CATEGORY_NAMES = ['face', 'object', 'body', 'scene']
DEFAULT_METRIC_NAME = 'acc'
DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 10
DEFAULT_N_REPEATS_PERM = 10
DEFAULT_N_PERMS = 10
DEFAULT_DECODING_STEP = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_JOBS = -1
DEFAULT_OUTPUT_TAG = ''

CHANCE_LEVEL_BY_METRIC = {
    'acc': 0.5,
    'auc': 0.5,
}

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

SOURCE_EPOCH_KIND_BY_FEATURE_KIND = {
    'erp': 'ERP',
    'highgamma': 'TFA',
    'lowgamma': 'TFA',
    'tfa': 'TFA',
    'gamma': 'TFA',
    'gamma_multiband': 'TFA',
}


@dataclass
class CommonConfig:
    subject: str
    feature_kind: str
    roi_name: str
    grouped_data_mat: Path
    task_field: str
    metric_name: str
    time_smooth_win: int
    condition_pairs: list
    category_names: list
    n_splits: int
    n_repeats: int
    n_repeats_perm: int
    n_perms: int
    decoding_step: int
    random_state: int
    n_jobs: int
    output_tag: str


@dataclass
class RoiDataBundle:
    roi_path: Path
    task_field: str
    feature_kind: str
    data: np.ndarray
    time_vector: np.ndarray


@dataclass
class PairedDataset:
    color: np.ndarray
    gray: np.ndarray
    pair_ids: np.ndarray
    category_ids: np.ndarray
    sample_keys: list
    color_repeat_indices: np.ndarray
    gray_repeat_indices: np.ndarray
    matched_counts: list
    category_names: list


@dataclass
class RepeatSelectionMetadata:
    source_epoch_path: Path
    source_epoch_kind: str
    bad_epoch_indices: np.ndarray
    condition_original_repeat_index: np.ndarray
    condition_original_count: np.ndarray | None
    condition_kept_count_before_trim: np.ndarray | None


def resolve_feature_kind(feature_kind):
    normalized = FEATURE_KIND_ALIASES.get(str(feature_kind), str(feature_kind))
    if normalized not in FEATURE_CONFIG:
        raise ValueError(f'Unsupported FEATURE_KIND: {feature_kind}')
    return normalized


def parse_int_list(text):
    return [int(item.strip()) for item in str(text).split(',') if item.strip()]


def parse_name_list(text):
    return [item.strip() for item in str(text).split(',') if item.strip()]


def build_common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--subject', default=DEFAULT_SUBJECT)
    parser.add_argument('--feature-kind', default=DEFAULT_FEATURE_KIND)
    parser.add_argument('--roi-name', default=DEFAULT_ROI_NAME)
    parser.add_argument('--grouped-data-mat', default=DEFAULT_GROUPED_DATA_MAT)
    parser.add_argument('--task-field', default=DEFAULT_TASK_FIELD)
    parser.add_argument('--metric', default=DEFAULT_METRIC_NAME, choices=['acc', 'auc'])
    parser.add_argument('--time-smooth-win', type=int, default=None)
    parser.add_argument('--color-condition-indices', default=','.join(map(str, DEFAULT_COLOR_CONDITION_INDICES)))
    parser.add_argument('--gray-condition-indices', default=','.join(map(str, DEFAULT_GRAY_CONDITION_INDICES)))
    parser.add_argument('--category-names', default=','.join(DEFAULT_CATEGORY_NAMES))
    parser.add_argument('--n-splits', type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument('--n-repeats', type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument('--n-repeats-perm', type=int, default=DEFAULT_N_REPEATS_PERM)
    parser.add_argument('--n-perms', type=int, default=DEFAULT_N_PERMS)
    parser.add_argument('--decoding-step', type=int, default=DEFAULT_DECODING_STEP)
    parser.add_argument('--random-state', type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument('--n-jobs', type=int, default=DEFAULT_N_JOBS)
    parser.add_argument('--output-tag', default=DEFAULT_OUTPUT_TAG)
    return parser


def normalize_common_args(args):
    feature_kind = resolve_feature_kind(args.feature_kind)
    task_field = args.task_field or FEATURE_CONFIG[feature_kind]['task_field']
    time_smooth_win = FEATURE_CONFIG[feature_kind]['default_smooth_win'] if args.time_smooth_win is None else int(args.time_smooth_win)
    color_condition_indices = parse_int_list(args.color_condition_indices)
    gray_condition_indices = parse_int_list(args.gray_condition_indices)
    category_names = parse_name_list(args.category_names)

    grouped_data_arg = args.grouped_data_mat
    if not grouped_data_arg:
        grouped_data_arg = str(get_task_groupeddata_path(project_root(), args.subject, 'task1'))
    if len(color_condition_indices) != len(gray_condition_indices):
        raise ValueError('Color and gray condition index lists must have the same length.')
    if len(category_names) != len(color_condition_indices):
        raise ValueError('CATEGORY_NAMES must match the number of condition pairs.')
    if args.n_splits < 2:
        raise ValueError('N_SPLITS must be at least 2.')
    if args.n_repeats < 1:
        raise ValueError('N_REPEATS must be at least 1.')
    if args.n_repeats_perm < 1:
        raise ValueError('N_REPEATS_PERM must be at least 1.')
    if args.n_perms < 0:
        raise ValueError('N_PERMS must be >= 0.')
    if args.decoding_step < 1:
        raise ValueError('DECODING_STEP must be at least 1.')

    grouped_data_mat = Path(grouped_data_arg).expanduser()
    if not grouped_data_mat.is_absolute():
        grouped_data_mat = (Path.cwd() / grouped_data_mat).resolve()
    if not grouped_data_mat.is_file():
        raise FileNotFoundError(f'groupedData mat file not found: {grouped_data_mat}')

    return CommonConfig(
        subject=str(args.subject),
        feature_kind=feature_kind,
        roi_name=str(args.roi_name),
        grouped_data_mat=grouped_data_mat,
        task_field=task_field,
        metric_name=str(args.metric),
        time_smooth_win=int(time_smooth_win),
        condition_pairs=list(zip(color_condition_indices, gray_condition_indices)),
        category_names=category_names,
        n_splits=int(args.n_splits),
        n_repeats=int(args.n_repeats),
        n_repeats_perm=int(args.n_repeats_perm),
        n_perms=int(args.n_perms),
        decoding_step=int(args.decoding_step),
        random_state=int(args.random_state),
        n_jobs=int(args.n_jobs),
        output_tag=str(args.output_tag),
    )


def get_roi_path(subject, feature_kind, roi_name):
    feature_dir = get_feature_dir(project_root(), FEATURE_CONFIG[feature_kind]['feature_subdir'], subject)
    return feature_dir / f'{roi_name}.mat'


def get_time_vector(mat, feature_kind, n_time):
    time_field = FEATURE_CONFIG[feature_kind]['time_field']
    if time_field and time_field in mat:
        times = np.asarray(mat[time_field], dtype=float).reshape(-1)
        if times.size == n_time:
            return times
    return np.linspace(T_START, T_END, int(n_time), dtype=float)


def load_roi_task_data(config):
    roi_path = get_roi_path(config.subject, config.feature_kind, config.roi_name)
    if not roi_path.is_file():
        raise FileNotFoundError(f'ROI feature file not found: {roi_path}')

    mat = sio.loadmat(roi_path)
    if config.task_field not in mat:
        raise ValueError(f'Missing matrix: {config.task_field}')

    data = np.asarray(mat[config.task_field], dtype=float)
    if data.ndim != 4:
        raise ValueError(f'{config.task_field} must be a 4D array [Cond, Rep, Feature, Time], got shape {data.shape}')

    time_vector = get_time_vector(mat, config.feature_kind, data.shape[-1])
    return RoiDataBundle(
        roi_path=roi_path,
        task_field=config.task_field,
        feature_kind=config.feature_kind,
        data=data,
        time_vector=time_vector,
    )


def _normalize_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return value


def _sample_key(value):
    value = _normalize_scalar(value)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        if np.isclose(value, round(value)):
            return int(round(value))
        return float(value)
    return str(value)


def _cell_to_id_list(entry):
    array = np.asarray(entry)
    if array.size == 0:
        return []
    if array.dtype == object:
        values = []
        for item in array.reshape(-1):
            values.extend(_cell_to_id_list(item))
        return values

    values = []
    for value in array.reshape(-1):
        key = _sample_key(value)
        if key is not None:
            values.append(key)
    return values


def load_grouped_data(grouped_data_mat_path):
    mat = sio.loadmat(grouped_data_mat_path)
    if 'groupedData' not in mat:
        raise ValueError(f'groupedData variable not found in {grouped_data_mat_path}')

    grouped = np.asarray(mat['groupedData'], dtype=object)
    if grouped.shape == (2, 4):
        grouped = grouped.T
    if grouped.shape != (4, 2):
        raise ValueError(f'groupedData must be 4x2, got shape {grouped.shape}')

    nested = []
    for row_idx in range(grouped.shape[0]):
        nested_row = []
        for col_idx in range(grouped.shape[1]):
            nested_row.append(_cell_to_id_list(grouped[row_idx, col_idx]))
        nested.append(nested_row)
    return nested


def get_source_epoch_path(subject, feature_kind, task_name='task1'):
    source_epoch_kind = SOURCE_EPOCH_KIND_BY_FEATURE_KIND[feature_kind]
    return project_root() / 'processed_data' / str(subject) / f'{task_name}_{source_epoch_kind}_epoched.mat'


def _unwrap_singleton_object(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.reshape(-1)[0]
    return value


def _extract_struct_field(struct_obj, field_name):
    struct_obj = _unwrap_singleton_object(struct_obj)
    if hasattr(struct_obj, field_name):
        return getattr(struct_obj, field_name)
    if isinstance(struct_obj, np.ndarray) and struct_obj.dtype.names and field_name in struct_obj.dtype.names:
        return struct_obj[field_name]
    raise AttributeError(field_name)


def _maybe_int_vector(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return np.zeros(0, dtype=int)
    return np.asarray(array, dtype=int).reshape(-1)


def load_repeat_selection_metadata(config, task_name='task1'):
    source_epoch_path = get_source_epoch_path(config.subject, config.feature_kind, task_name=task_name)
    if not source_epoch_path.is_file():
        raise FileNotFoundError(f'Source epoch file not found: {source_epoch_path}')

    mat = sio.loadmat(source_epoch_path, struct_as_record=False, squeeze_me=True)
    if 'epoch' not in mat:
        raise ValueError(f'Missing epoch struct in {source_epoch_path}')

    epoch_struct = _unwrap_singleton_object(mat['epoch'])
    try:
        condition_original_repeat_index = np.asarray(
            _extract_struct_field(epoch_struct, 'condition_original_repeat_index'),
            dtype=int,
        )
    except AttributeError as exc:
        raise ValueError(
            'Missing repeat-selection metadata in '
            f'{source_epoch_path}. Please rerun newanalyse/Sec1_preanalyse.m to regenerate task1 epoched files '
            'with bad-trial index metadata before using paired decoding.'
        ) from exc

    if condition_original_repeat_index.ndim == 1:
        condition_original_repeat_index = condition_original_repeat_index.reshape(1, -1)

    bad_epoch_indices = _maybe_int_vector(
        _extract_struct_field(epoch_struct, 'bad_epoch_indices') if hasattr(epoch_struct, 'bad_epoch_indices') else None
    )
    condition_original_count = None
    condition_kept_count_before_trim = None
    if hasattr(epoch_struct, 'condition_original_count'):
        condition_original_count = _maybe_int_vector(_extract_struct_field(epoch_struct, 'condition_original_count'))
    if hasattr(epoch_struct, 'condition_kept_count_before_trim'):
        condition_kept_count_before_trim = _maybe_int_vector(_extract_struct_field(epoch_struct, 'condition_kept_count_before_trim'))

    return RepeatSelectionMetadata(
        source_epoch_path=source_epoch_path,
        source_epoch_kind=SOURCE_EPOCH_KIND_BY_FEATURE_KIND[config.feature_kind],
        bad_epoch_indices=bad_epoch_indices,
        condition_original_repeat_index=condition_original_repeat_index,
        condition_original_count=condition_original_count,
        condition_kept_count_before_trim=condition_kept_count_before_trim,
    )


def _align_id_list_to_selected_repeats(id_list, selected_repeat_indices, expected_repeats, label_text):
    if len(id_list) == expected_repeats:
        return list(id_list)

    selected_repeat_indices = np.asarray(selected_repeat_indices, dtype=int).reshape(-1)
    if selected_repeat_indices.size != expected_repeats:
        raise ValueError(
            f'{label_text} repeat selection length mismatch: selected={selected_repeat_indices.size}, expected={expected_repeats}.'
        )
    if selected_repeat_indices.size == 0:
        return []

    max_required = int(np.max(selected_repeat_indices))
    if len(id_list) < max_required:
        raise ValueError(
            f'{label_text} groupedData has {len(id_list)} ids, but needs either exactly {expected_repeats} '
            f'already-aligned ids or at least {max_required} original ids to apply bad-trial reindexing.'
        )
    return [id_list[repeat_idx - 1] for repeat_idx in selected_repeat_indices]


def align_grouped_data_to_saved_repeats(grouped_data, repeat_metadata, condition_pairs, expected_repeats):
    condition_original_repeat_index = repeat_metadata.condition_original_repeat_index
    n_conditions = condition_original_repeat_index.shape[0]
    aligned = []
    for row_idx, (color_condition_idx, gray_condition_idx) in enumerate(condition_pairs):
        if color_condition_idx >= n_conditions or gray_condition_idx >= n_conditions:
            raise IndexError(
                f'Condition pair {(color_condition_idx, gray_condition_idx)} exceeds metadata rows {n_conditions} '
                f'in {repeat_metadata.source_epoch_path}.'
            )
        color_ids = _align_id_list_to_selected_repeats(
            grouped_data[row_idx][0],
            condition_original_repeat_index[color_condition_idx],
            expected_repeats,
            f'groupedData row {row_idx} color condition {color_condition_idx}',
        )
        gray_ids = _align_id_list_to_selected_repeats(
            grouped_data[row_idx][1],
            condition_original_repeat_index[gray_condition_idx],
            expected_repeats,
            f'groupedData row {row_idx} gray condition {gray_condition_idx}',
        )
        aligned.append([color_ids, gray_ids])
    return aligned


def _match_pair_positions(color_ids, gray_ids):
    gray_positions = {}
    for gray_idx, sample_id in enumerate(gray_ids):
        gray_positions.setdefault(sample_id, []).append(gray_idx)

    color_seen = {}
    matches = []
    for color_idx, sample_id in enumerate(color_ids):
        occurrence_rank = color_seen.get(sample_id, 0)
        color_seen[sample_id] = occurrence_rank + 1
        gray_candidates = gray_positions.get(sample_id, [])
        if occurrence_rank < len(gray_candidates):
            matches.append((sample_id, occurrence_rank, color_idx, gray_candidates[occurrence_rank]))
    return matches


def build_paired_dataset(data, grouped_data, condition_pairs, category_names):
    if len(grouped_data) != len(condition_pairs):
        raise ValueError('groupedData rows must match the number of condition pairs.')

    n_conditions, n_repeats = data.shape[0], data.shape[1]
    color_trials = []
    gray_trials = []
    pair_ids = []
    category_ids = []
    sample_keys = []
    color_repeat_indices = []
    gray_repeat_indices = []
    matched_counts = []
    pair_id = 0

    for category_idx, (condition_pair, category_name) in enumerate(zip(condition_pairs, category_names)):
        color_condition_idx, gray_condition_idx = condition_pair
        if color_condition_idx >= n_conditions or gray_condition_idx >= n_conditions:
            raise IndexError(f'Condition pair {condition_pair} exceeds available conditions {n_conditions}.')

        color_ids = grouped_data[category_idx][0]
        gray_ids = grouped_data[category_idx][1]
        if len(color_ids) != n_repeats or len(gray_ids) != n_repeats:
            raise ValueError(
                f'groupedData row {category_idx} must each have {n_repeats} ids to match trial count, '
                f'got color={len(color_ids)}, gray={len(gray_ids)}.'
            )

        matches = _match_pair_positions(color_ids, gray_ids)
        if not matches:
            raise ValueError(f'No matched color-gray pairs found for category {category_name}.')

        matched_counts.append(len(matches))
        for sample_id, occurrence_rank, color_rep_idx, gray_rep_idx in matches:
            color_trials.append(np.asarray(data[color_condition_idx, color_rep_idx], dtype=float))
            gray_trials.append(np.asarray(data[gray_condition_idx, gray_rep_idx], dtype=float))
            pair_ids.append(pair_id)
            category_ids.append(category_idx)
            sample_keys.append(f'{category_name}:{sample_id}:{occurrence_rank}')
            color_repeat_indices.append(color_rep_idx)
            gray_repeat_indices.append(gray_rep_idx)
            pair_id += 1

    if not color_trials:
        raise ValueError('No paired samples could be constructed from groupedData.')

    return PairedDataset(
        color=np.stack(color_trials, axis=0),
        gray=np.stack(gray_trials, axis=0),
        pair_ids=np.asarray(pair_ids, dtype=int),
        category_ids=np.asarray(category_ids, dtype=int),
        sample_keys=sample_keys,
        color_repeat_indices=np.asarray(color_repeat_indices, dtype=int),
        gray_repeat_indices=np.asarray(gray_repeat_indices, dtype=int),
        matched_counts=matched_counts,
        category_names=category_names,
    )


def baseline_zscore(data, time_vector, baseline_end_ms=0.0):
    baseline_mask = np.asarray(time_vector, dtype=float) < float(baseline_end_ms)
    if not np.any(baseline_mask):
        return data.copy()

    baseline = data[:, :, baseline_mask]
    mean = baseline.mean(axis=2, keepdims=True)
    std = baseline.std(axis=2, keepdims=True) + 1e-8
    return (data - mean) / std


def smooth_data_causal(data, win_size):
    if win_size <= 0:
        return data.copy()
    n_samples, n_features, n_time = data.shape
    smoothed = np.zeros_like(data)
    for time_idx in range(n_time):
        start_idx = max(0, time_idx - win_size)
        smoothed[:, :, time_idx] = np.mean(data[:, :, start_idx:time_idx + 1], axis=2)
    return smoothed


def preprocess_paired_dataset(paired_dataset, time_vector, time_smooth_win):
    stacked = np.concatenate([paired_dataset.color, paired_dataset.gray], axis=0)
    stacked = baseline_zscore(stacked, time_vector)
    stacked = smooth_data_causal(stacked, time_smooth_win)
    n_pairs = paired_dataset.color.shape[0]
    return replace(
        paired_dataset,
        color=stacked[:n_pairs],
        gray=stacked[n_pairs:],
    )


def load_preprocessed_paired_dataset(config):
    roi_bundle = load_roi_task_data(config)
    grouped_data = load_grouped_data(config.grouped_data_mat)
    repeat_metadata = load_repeat_selection_metadata(config, task_name='task1')
    grouped_data = align_grouped_data_to_saved_repeats(
        grouped_data,
        repeat_metadata,
        config.condition_pairs,
        expected_repeats=roi_bundle.data.shape[1],
    )
    paired_dataset = build_paired_dataset(roi_bundle.data, grouped_data, config.condition_pairs, config.category_names)
    paired_dataset = preprocess_paired_dataset(paired_dataset, roi_bundle.time_vector, config.time_smooth_win)
    return roi_bundle, paired_dataset


def stack_raw_samples(paired_dataset):
    n_pairs = paired_dataset.color.shape[0]
    x = np.concatenate([paired_dataset.color, paired_dataset.gray], axis=0)
    y = np.concatenate([np.ones(n_pairs, dtype=int), np.zeros(n_pairs, dtype=int)], axis=0)
    groups = np.concatenate([paired_dataset.pair_ids, paired_dataset.pair_ids], axis=0)
    split_labels = np.concatenate([
        paired_dataset.category_ids * 2 + 1,
        paired_dataset.category_ids * 2,
    ], axis=0)
    return x, y, groups, split_labels


def stack_centered_samples(paired_dataset):
    pair_mean = 0.5 * (paired_dataset.color + paired_dataset.gray)
    color_centered = paired_dataset.color - pair_mean
    gray_centered = paired_dataset.gray - pair_mean
    centered_dataset = replace(paired_dataset, color=color_centered, gray=gray_centered)
    return stack_raw_samples(centered_dataset)


def stack_difference_samples(paired_dataset):
    difference = paired_dataset.color - paired_dataset.gray
    x = np.concatenate([difference, -difference], axis=0)
    y = np.concatenate([
        np.ones(paired_dataset.color.shape[0], dtype=int),
        np.zeros(paired_dataset.color.shape[0], dtype=int),
    ], axis=0)
    groups = np.concatenate([paired_dataset.pair_ids, paired_dataset.pair_ids], axis=0)
    split_labels = np.concatenate([
        paired_dataset.category_ids * 2 + 1,
        paired_dataset.category_ids * 2,
    ], axis=0)
    return x, y, groups, split_labels


def _iter_group_splits(split_labels, groups, n_splits, n_repeats, seed):
    unique_groups = np.unique(groups)
    if unique_groups.size < n_splits:
        raise ValueError(f'Need at least {n_splits} matched pairs, got {unique_groups.size}.')

    if StratifiedGroupKFold is None:
        if n_repeats > 1:
            raise ImportError('StratifiedGroupKFold is unavailable in this scikit-learn version, so repeated grouped CV is not supported.')
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(np.zeros(len(groups)), groups=groups))

    splits = []
    for repeat_idx in range(n_repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat_idx)
        splits.extend(list(splitter.split(np.zeros(len(split_labels)), split_labels, groups)))
    return splits


def _score_binary_predictions(metric_name, y_true, y_prob, y_pred):
    if metric_name == 'acc':
        return float(accuracy_score(y_true, y_pred))
    if metric_name == 'auc':
        if len(np.unique(y_true)) < 2:
            return np.nan
        positive_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        return float(roc_auc_score(y_true, positive_prob))
    raise ValueError(f'Unsupported metric: {metric_name}')


def _permute_binary_labels_within_groups(y, groups, seed):
    rng = np.random.RandomState(seed)
    y_perm = np.asarray(y, dtype=int).copy()
    groups = np.asarray(groups)
    for group_id in np.unique(groups):
        indices = np.flatnonzero(groups == group_id)
        if indices.size <= 1:
            continue
        if indices.size == 2:
            if rng.rand() < 0.5:
                y_perm[indices] = y_perm[indices[::-1]]
        else:
            y_perm[indices] = y_perm[rng.permutation(indices)]
    return y_perm


def run_binary_curve(x, y, groups, split_labels, metric_name, n_splits, n_repeats, decoding_step, seed, shuffle=False):
    if x.ndim != 3:
        raise ValueError(f'x must be [Sample, Feature, Time], got shape {x.shape}')

    y_use = _permute_binary_labels_within_groups(y, groups, seed) if shuffle else np.asarray(y, dtype=int).copy()
    splits = _iter_group_splits(split_labels, groups, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    time_indices = np.arange(0, x.shape[2], decoding_step)
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
            fold_scores.append(_score_binary_predictions(metric_name, y_use[test_idx], y_prob, y_pred))

        fold_scores = np.asarray(fold_scores, dtype=float)
        finite_mask = np.isfinite(fold_scores)
        curve[time_pos] = float(np.nanmean(fold_scores))
        if np.any(finite_mask):
            sem[time_pos] = float(np.nanstd(fold_scores, ddof=0) / np.sqrt(np.sum(finite_mask)))
        else:
            sem[time_pos] = np.nan
    return curve, sem


def run_binary_curve_mean_only(x, y, groups, split_labels, metric_name, n_splits, n_repeats, decoding_step, seed):
    curve, _ = run_binary_curve(
        x,
        y,
        groups,
        split_labels,
        metric_name=metric_name,
        n_splits=n_splits,
        n_repeats=n_repeats,
        decoding_step=decoding_step,
        seed=seed,
        shuffle=True,
    )
    return curve


def cluster_permutation_significance(real_curve, perm_dist, chance_level, plot_times):
    if perm_dist is None or np.size(perm_dist) == 0:
        return np.full(real_curve.shape, np.nan, dtype=float), np.zeros_like(real_curve, dtype=bool)

    threshold_95 = np.nanpercentile(perm_dist, 95, axis=0)
    valid_mask = np.asarray(plot_times[:len(real_curve)], dtype=float) > 20
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


def run_decoding_result(x, y, groups, split_labels, config, plot_times):
    real_curve, real_sem = run_binary_curve(
        x,
        y,
        groups,
        split_labels,
        metric_name=config.metric_name,
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        decoding_step=config.decoding_step,
        seed=config.random_state,
        shuffle=False,
    )

    if config.n_perms > 0:
        perm_dist = np.asarray(
            Parallel(n_jobs=config.n_jobs)(
                delayed(run_binary_curve_mean_only)(
                    x,
                    y,
                    groups,
                    split_labels,
                    config.metric_name,
                    config.n_splits,
                    config.n_repeats_perm,
                    config.decoding_step,
                    config.random_state + 100000 + perm_idx,
                )
                for perm_idx in range(config.n_perms)
            )
        )
    else:
        perm_dist = np.full((0, real_curve.shape[0]), np.nan)

    chance_level = CHANCE_LEVEL_BY_METRIC[config.metric_name]
    threshold_95, sig_mask = cluster_permutation_significance(real_curve, perm_dist, chance_level, plot_times)
    latencies = compute_latency_points(real_curve, sig_mask, chance_level, plot_times)
    return {
        'curve': real_curve,
        'sem': real_sem,
        'perm_dist': perm_dist,
        'threshold_95': threshold_95,
        'sig_mask': sig_mask,
        'latencies': latencies,
        'chance_level': chance_level,
    }


def paired_counts_by_category(paired_dataset):
    return {
        category_name: int(count)
        for category_name, count in zip(paired_dataset.category_names, paired_dataset.matched_counts)
    }


def build_output_dir(scheme_name, config):
    out_dir = (
        result_root(project_root())
        / 'reports'
        / 'task1_paired_decoding'
        / sanitize_token(scheme_name)
        / sanitize_token(config.feature_kind)
        / sanitize_token(config.subject)
        / sanitize_token(config.roi_name)
        / sanitize_token(config.metric_name)
    )
    if config.output_tag:
        out_dir /= sanitize_token(config.output_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_curve_figure(save_path, plot_times, result, metric_name, title, subtitle_lines=None):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    ax.fill_between(plot_times, result['curve'] - result['sem'], result['curve'] + result['sem'], color='#4c78a8', alpha=0.18)
    ax.plot(plot_times, result['curve'], color='#4c78a8', linewidth=2.0, label='Real decoding')
    if np.any(np.isfinite(result['threshold_95'])):
        ax.plot(plot_times, result['threshold_95'], color='#e45756', linestyle='--', linewidth=1.5, label='Permutation 95% threshold')
    ax.axhline(result['chance_level'], color='black', linestyle=':', linewidth=1.0, label='Chance')
    ax.axvline(0.0, color='#999999', linestyle='--', linewidth=1.0)
    earliest = result['latencies']['earliest']
    if np.isfinite(earliest):
        ax.axvline(earliest, color='#72b7b2', linestyle='--', linewidth=1.2, label=f'Earliest = {earliest:.1f} ms')
    ax.set_ylabel(metric_name.upper())
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best', fontsize=9)

    if subtitle_lines:
        text = '\n'.join(subtitle_lines)
        ax.text(
            0.99,
            0.02,
            text,
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=9,
            bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '#cccccc'},
        )

    ax = axes[1]
    sig_mask = result['sig_mask'].astype(float)
    if np.any(sig_mask):
        ax.fill_between(plot_times, np.zeros_like(sig_mask), sig_mask, step='mid', color='#4c78a8', alpha=0.85)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.set_ylabel('Cluster sig.')
    ax.set_xlabel('Time (ms)')
    ax.grid(True, linestyle='--', alpha=0.35)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_decoding_summary(save_path, config, roi_bundle, paired_dataset, plot_times, result, scheme_name, scheme_note):
    counts = paired_counts_by_category(paired_dataset)
    np.savez(
        save_path,
        subject=np.array(config.subject),
        feature_kind=np.array(config.feature_kind),
        roi_name=np.array(config.roi_name),
        grouped_data_mat=np.array(str(config.grouped_data_mat)),
        roi_path=np.array(str(roi_bundle.roi_path)),
        task_field=np.array(roi_bundle.task_field),
        metric_name=np.array(config.metric_name),
        scheme_name=np.array(scheme_name),
        scheme_note=np.array(scheme_note),
        n_perms=np.array(config.n_perms),
        n_repeats_real=np.array(config.n_repeats),
        n_repeats_perm=np.array(config.n_repeats_perm),
        category_names=np.asarray(config.category_names, dtype=object),
        color_condition_indices=np.asarray([pair[0] for pair in config.condition_pairs], dtype=int),
        gray_condition_indices=np.asarray([pair[1] for pair in config.condition_pairs], dtype=int),
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
        pair_ids=paired_dataset.pair_ids,
        category_ids=paired_dataset.category_ids,
        sample_keys=np.asarray(paired_dataset.sample_keys, dtype=object),
        color_repeat_indices=paired_dataset.color_repeat_indices,
        gray_repeat_indices=paired_dataset.gray_repeat_indices,
        matched_count_names=np.asarray(list(counts.keys()), dtype=object),
        matched_count_values=np.asarray(list(counts.values()), dtype=int),
    )


def execute_decoding_scheme(config, scheme_name, scheme_title, scheme_note, dataset_builder):
    start_time = time.time()
    roi_bundle, paired_dataset = load_preprocessed_paired_dataset(config)
    x, y, groups, split_labels = dataset_builder(paired_dataset)
    plot_times = roi_bundle.time_vector[::config.decoding_step][:len(np.arange(0, x.shape[2], config.decoding_step))]
    result = run_decoding_result(x, y, groups, split_labels, config, plot_times)
    plot_times = plot_times[:result['curve'].shape[0]]

    out_dir = build_output_dir(scheme_name, config)
    fig_path = out_dir / f'{scheme_name}_curve.png'
    npz_path = out_dir / f'{scheme_name}_curve.npz'
    subtitle_lines = [
        f'Subject: {config.subject}',
        f'Feature: {config.feature_kind}',
        f'ROI: {config.roi_name}',
        f'Matched pairs: {paired_dataset.color.shape[0]}',
        f'Counts: {paired_counts_by_category(paired_dataset)}',
        f'Permutations: {config.n_perms} | Perm repeats: {config.n_repeats_perm}',
    ]
    save_curve_figure(fig_path, plot_times, result, config.metric_name, scheme_title, subtitle_lines=subtitle_lines)
    save_decoding_summary(npz_path, config, roi_bundle, paired_dataset, plot_times, result, scheme_name, scheme_note)

    return {
        'scheme_name': scheme_name,
        'scheme_note': scheme_note,
        'subject': config.subject,
        'feature_kind': config.feature_kind,
        'roi_name': config.roi_name,
        'grouped_data_mat': str(config.grouped_data_mat),
        'roi_path': str(roi_bundle.roi_path),
        'task_field': roi_bundle.task_field,
        'metric_name': config.metric_name,
        'n_perms': config.n_perms,
        'n_repeats_perm': config.n_repeats_perm,
        'matched_pairs': int(paired_dataset.color.shape[0]),
        'matched_counts_by_category': paired_counts_by_category(paired_dataset),
        'earliest_sig_ms': result['latencies']['earliest'],
        'peak_sig_ms': result['latencies']['peak'],
        'figure_path': str(fig_path),
        'summary_path': str(npz_path),
        'runtime_s': time.time() - start_time,
    }


def reduce_time_window(tensor, time_vector, window_start_ms, window_end_ms):
    time_vector = np.asarray(time_vector, dtype=float)
    mask = (time_vector >= float(window_start_ms)) & (time_vector <= float(window_end_ms))
    if not np.any(mask):
        center = 0.5 * (float(window_start_ms) + float(window_end_ms))
        nearest_idx = int(np.argmin(np.abs(time_vector - center)))
        mask = np.zeros_like(time_vector, dtype=bool)
        mask[nearest_idx] = True
    return np.mean(tensor[:, :, mask], axis=2)


def resolve_selected_category_indices(selection_text, category_names):
    if selection_text is None:
        return list(range(len(category_names)))

    text = str(selection_text).strip()
    if not text or text.lower() == 'all':
        return list(range(len(category_names)))

    selected = []
    for token in text.split(','):
        item = token.strip()
        if not item:
            continue
        if item.isdigit():
            idx = int(item)
            if idx >= 1 and idx <= len(category_names):
                idx -= 1
            elif idx < 0 or idx >= len(category_names):
                raise ValueError(f'Category index out of range: {item}')
        else:
            try:
                idx = next(i for i, name in enumerate(category_names) if name.lower() == item.lower())
            except StopIteration as exc:
                raise ValueError(f'Unknown category name: {item}') from exc
        if idx not in selected:
            selected.append(idx)

    if not selected:
        raise ValueError('No valid categories were selected.')
    return selected


def select_paired_dataset_categories(paired_dataset, selected_category_indices):
    selected_category_indices = list(selected_category_indices)
    mask = np.isin(paired_dataset.category_ids, selected_category_indices)
    if not np.any(mask):
        raise ValueError('Selected categories do not contain any paired samples.')

    remapped_category_ids = np.asarray(
        [selected_category_indices.index(int(category_id)) for category_id in paired_dataset.category_ids[mask]],
        dtype=int,
    )
    selected_names = [paired_dataset.category_names[idx] for idx in selected_category_indices]
    matched_counts = [int(np.sum(remapped_category_ids == idx)) for idx in range(len(selected_names))]

    return replace(
        paired_dataset,
        color=paired_dataset.color[mask],
        gray=paired_dataset.gray[mask],
        pair_ids=np.arange(np.sum(mask), dtype=int),
        category_ids=remapped_category_ids,
        sample_keys=[key for keep, key in zip(mask, paired_dataset.sample_keys) if keep],
        color_repeat_indices=paired_dataset.color_repeat_indices[mask],
        gray_repeat_indices=paired_dataset.gray_repeat_indices[mask],
        matched_counts=matched_counts,
        category_names=selected_names,
    )


def smooth_time_axis(data, win_size):
    array = np.asarray(data, dtype=float)
    if win_size is None or int(win_size) <= 1:
        return array.copy()

    win_size = int(win_size)
    kernel = np.ones(win_size, dtype=float) / float(win_size)
    return np.apply_along_axis(lambda values: np.convolve(values, kernel, mode='same'), -1, array)