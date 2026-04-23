from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None


SOURCE_EPOCH_KIND_BY_FEATURE_KIND = {
    'erp': 'ERP',
    'highgamma': 'TFA',
    'lowgamma': 'TFA',
    'tfa': 'TFA',
    'gamma': 'TFA',
    'gamma_multiband': 'TFA',
}


@dataclass
class RepeatSelectionMetadata:
    source_epoch_path: Path
    source_epoch_kind: str
    condition_original_repeat_index: np.ndarray


@dataclass
class PairedCategoryTrials:
    category_name: str
    color: np.ndarray
    gray: np.ndarray
    pair_ids: np.ndarray
    sample_keys: list[str]
    color_repeat_indices: np.ndarray
    gray_repeat_indices: np.ndarray
    matched_count: int


def normalize_category_pairs(category_pairs):
    normalized = []
    for pair in category_pairs:
        if len(pair) != 2:
            raise ValueError(f'Each category pair must have two condition indices, got: {pair}')
        normalized.append((int(pair[0]), int(pair[1])))
    return normalized


def resolve_existing_path(base_path, raw_path):
    if raw_path in {None, ''}:
        return None

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    base_candidate = Path(base_path) / path
    if base_candidate.exists():
        return base_candidate.resolve()
    return (Path.cwd() / path).resolve()


def resolve_within_category_task(
    task,
    field_prefix,
    default_task_name='task1',
    default_category_pairs=None,
    default_category_names=None,
    use_groupeddata_pairing=False,
    use_groupeddata_pair_centering=False,
    groupeddata_files=None,
):
    task_name = str(task.get('task_name', default_task_name))
    data_key = str(task.get('data_key') or f'{field_prefix}_{task_name}')

    category_pairs = task.get('category_pairs')
    if category_pairs is None:
        if isinstance(default_category_pairs, dict):
            category_pairs = default_category_pairs.get(task_name)
        else:
            category_pairs = default_category_pairs
    if category_pairs is None:
        raise ValueError(
            f'Missing category_pairs for task {task.get("id", "<unknown>")} and no default exists for {task_name}.'
        )
    category_pairs = normalize_category_pairs(category_pairs)

    category_names = task.get('category_names')
    if category_names is None:
        if isinstance(default_category_names, dict):
            category_names = default_category_names.get(task_name)
        else:
            category_names = default_category_names
    if category_names is None:
        raise ValueError(
            f'Missing category_names for task {task.get("id", "<unknown>")} and no default exists for {task_name}.'
        )
    category_names = [str(name) for name in category_names]
    if len(category_names) != len(category_pairs):
        raise ValueError(
            f'category_names length {len(category_names)} must match category_pairs length {len(category_pairs)} '
            f'for task {task.get("id", "<unknown>")}.'
        )

    groupeddata_mat = task.get('groupeddata_mat')
    if groupeddata_mat in {None, ''} and isinstance(groupeddata_files, dict):
        groupeddata_mat = groupeddata_files.get(task_name)
    if groupeddata_mat in {None, ''}:
        groupeddata_mat = None

    use_pairing = bool(task.get('use_groupeddata_pairing', use_groupeddata_pairing))
    use_pair_centering = bool(task.get('use_groupeddata_pair_centering', use_groupeddata_pair_centering))
    if use_pairing and groupeddata_mat is None:
        raise ValueError(
            f'Task {task.get("id", "<unknown>")} enabled groupedData pairing, but no groupedData path was provided '
            f'for {task_name}. Set task["groupeddata_mat"] or GROUPEDDATA_FILES[{task_name!r}].'
        )
    if use_pair_centering and not use_pairing:
        raise ValueError(
            f'Task {task.get("id", "<unknown>")} enabled groupedData pair centering, but groupedData pairing is disabled. '
            'Enable use_groupeddata_pairing before use_groupeddata_pair_centering.'
        )

    return {
        'task_name': task_name,
        'data_key': data_key,
        'category_pairs': category_pairs,
        'category_names': category_names,
        'use_groupeddata_pairing': use_pairing,
        'use_groupeddata_pair_centering': use_pair_centering,
        'groupeddata_mat': groupeddata_mat,
    }


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


def load_grouped_data(groupeddata_mat_path, variable_name='groupedData'):
    mat = sio.loadmat(groupeddata_mat_path)
    if variable_name not in mat:
        raise ValueError(f'{variable_name} variable not found in {groupeddata_mat_path}')

    grouped = np.asarray(mat[variable_name], dtype=object)
    if grouped.shape == (2, 4):
        grouped = grouped.T
    if grouped.shape[1] != 2:
        raise ValueError(f'{variable_name} must have exactly 2 columns [color, gray], got shape {grouped.shape}')

    nested = []
    for row_idx in range(grouped.shape[0]):
        nested_row = []
        for col_idx in range(grouped.shape[1]):
            nested_row.append(_cell_to_id_list(grouped[row_idx, col_idx]))
        nested.append(nested_row)
    return nested


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


def load_repeat_selection_metadata(base_path, subject, feature_kind, task_name):
    if feature_kind not in SOURCE_EPOCH_KIND_BY_FEATURE_KIND:
        raise ValueError(f'Unsupported feature_kind for groupedData pairing: {feature_kind}')

    source_epoch_kind = SOURCE_EPOCH_KIND_BY_FEATURE_KIND[feature_kind]
    source_epoch_path = Path(base_path) / 'processed_data' / str(subject) / f'{task_name}_{source_epoch_kind}_epoched.mat'
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
            f'{source_epoch_path}. Please rerun newanalyse/Sec1_preanalyse.m before using groupedData pairing.'
        ) from exc

    if condition_original_repeat_index.ndim == 1:
        condition_original_repeat_index = condition_original_repeat_index.reshape(1, -1)

    return RepeatSelectionMetadata(
        source_epoch_path=source_epoch_path,
        source_epoch_kind=source_epoch_kind,
        condition_original_repeat_index=condition_original_repeat_index,
    )


def _align_id_list_to_selected_repeats(id_list, selected_repeat_indices, expected_repeats, label_text):
    if len(id_list) == expected_repeats:
        return list(id_list)

    selected_repeat_indices = np.asarray(selected_repeat_indices, dtype=int).reshape(-1)
    if selected_repeat_indices.size != expected_repeats:
        raise ValueError(
            f'{label_text} repeat selection length mismatch: metadata keeps {selected_repeat_indices.size} repeats, '
            f'but the ROI feature tensor has {expected_repeats} repeats. This usually means the ROI features for this '
            'modality were generated from an older preprocessing run than the current processed_data metadata. '
            'Regenerate the corresponding Sec2_* ROI features so they match the current task epoched file before '
            'enabling groupedData pairing.'
        )
    if selected_repeat_indices.size == 0:
        return []

    max_required = int(np.max(selected_repeat_indices))
    if len(id_list) < max_required:
        raise ValueError(
            f'{label_text} groupedData has {len(id_list)} ids, but needs at least {max_required} original ids '
            f'to apply bad-trial reindexing.'
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
        if row_idx >= len(grouped_data):
            raise IndexError(
                f'groupedData row count {len(grouped_data)} is insufficient for requested row {row_idx}. '
                'Please provide category_pairs matching the groupedData rows.'
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


def build_paired_category_trials(data, grouped_data, condition_pairs, category_names):
    if data.ndim != 4:
        raise ValueError(f'data must be [Cond, Rep, Feature, Time], got shape {data.shape}')
    if len(grouped_data) < len(condition_pairs):
        raise ValueError('groupedData rows must cover every requested condition pair.')

    n_conditions, n_repeats = data.shape[0], data.shape[1]
    paired_categories = []

    for category_idx, ((color_condition_idx, gray_condition_idx), category_name) in enumerate(zip(condition_pairs, category_names)):
        if color_condition_idx >= n_conditions or gray_condition_idx >= n_conditions:
            raise IndexError(f'Condition pair {(color_condition_idx, gray_condition_idx)} exceeds available conditions {n_conditions}.')

        color_ids = grouped_data[category_idx][0]
        gray_ids = grouped_data[category_idx][1]
        if len(color_ids) != n_repeats or len(gray_ids) != n_repeats:
            raise ValueError(
                f'groupedData row {category_idx} must have {n_repeats} ids per column after repeat alignment, '
                f'got color={len(color_ids)}, gray={len(gray_ids)}.'
            )

        matches = _match_pair_positions(color_ids, gray_ids)
        if not matches:
            raise ValueError(f'No matched color-gray pairs found for category {category_name}.')

        color_trials = []
        gray_trials = []
        sample_keys = []
        color_repeat_indices = []
        gray_repeat_indices = []
        pair_ids = []
        for pair_index, (sample_id, occurrence_rank, color_rep_idx, gray_rep_idx) in enumerate(matches):
            color_trials.append(np.asarray(data[color_condition_idx, color_rep_idx], dtype=float))
            gray_trials.append(np.asarray(data[gray_condition_idx, gray_rep_idx], dtype=float))
            pair_ids.append(pair_index)
            sample_keys.append(f'{category_name}:{sample_id}:{occurrence_rank}')
            color_repeat_indices.append(color_rep_idx)
            gray_repeat_indices.append(gray_rep_idx)

        paired_categories.append(
            PairedCategoryTrials(
                category_name=str(category_name),
                color=np.stack(color_trials, axis=0),
                gray=np.stack(gray_trials, axis=0),
                pair_ids=np.asarray(pair_ids, dtype=int),
                sample_keys=sample_keys,
                color_repeat_indices=np.asarray(color_repeat_indices, dtype=int),
                gray_repeat_indices=np.asarray(gray_repeat_indices, dtype=int),
                matched_count=len(matches),
            )
        )

    return paired_categories


def load_paired_category_trials(base_path, subject, feature_kind, task_name, groupeddata_mat, data, condition_pairs, category_names):
    groupeddata_path = resolve_existing_path(base_path, groupeddata_mat)
    if groupeddata_path is None or not groupeddata_path.is_file():
        raise FileNotFoundError(f'groupedData file not found: {groupeddata_mat}')

    grouped_data = load_grouped_data(groupeddata_path)
    repeat_metadata = load_repeat_selection_metadata(base_path, subject, feature_kind, task_name)
    grouped_data = align_grouped_data_to_saved_repeats(
        grouped_data,
        repeat_metadata,
        normalize_category_pairs(condition_pairs),
        expected_repeats=int(data.shape[1]),
    )
    return build_paired_category_trials(data, grouped_data, condition_pairs, category_names)


def stack_paired_binary_trials(color_trials, gray_trials, pair_ids):
    color_trials = np.asarray(color_trials, dtype=float)
    gray_trials = np.asarray(gray_trials, dtype=float)
    pair_ids = np.asarray(pair_ids, dtype=int).reshape(-1)
    if color_trials.shape != gray_trials.shape:
        raise ValueError(f'color_trials and gray_trials must share shape, got {color_trials.shape} vs {gray_trials.shape}')
    if color_trials.shape[0] != pair_ids.size:
        raise ValueError(f'pair_ids length {pair_ids.size} must match paired sample count {color_trials.shape[0]}')

    samples = np.concatenate([color_trials, gray_trials], axis=0)
    labels = np.concatenate([
        np.zeros(color_trials.shape[0], dtype=float),
        np.ones(gray_trials.shape[0], dtype=float),
    ])
    groups = np.concatenate([pair_ids, pair_ids], axis=0)
    return samples, labels, groups


def center_paired_trials(color_trials, gray_trials):
    color_trials = np.asarray(color_trials, dtype=float)
    gray_trials = np.asarray(gray_trials, dtype=float)
    if color_trials.shape != gray_trials.shape:
        raise ValueError(f'color_trials and gray_trials must share shape, got {color_trials.shape} vs {gray_trials.shape}')

    pair_mean = 0.5 * (color_trials + gray_trials)
    return color_trials - pair_mean, gray_trials - pair_mean


def safe_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5


def _permute_binary_labels_within_groups(labels, groups, seed):
    rng = np.random.RandomState(seed)
    labels_perm = np.asarray(labels).copy()
    groups = np.asarray(groups)
    for group_id in np.unique(groups):
        indices = np.flatnonzero(groups == group_id)
        if indices.size <= 1:
            continue
        if indices.size == 2:
            if rng.rand() < 0.5:
                labels_perm[indices] = labels_perm[indices[::-1]]
        else:
            labels_perm[indices] = labels_perm[rng.permutation(indices)]
    return labels_perm


def _iter_group_splits(labels, groups, n_splits, n_repeats, seed):
    unique_groups = np.unique(groups)
    if unique_groups.size < n_splits:
        raise ValueError(f'Need at least {n_splits} matched pairs, got {unique_groups.size}.')

    if StratifiedGroupKFold is None:
        if n_repeats > 1:
            raise ImportError(
                'StratifiedGroupKFold is unavailable in this scikit-learn version, so repeated grouped CV is not supported.'
            )
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(np.zeros(len(groups)), groups=groups))

    splits = []
    for repeat_idx in range(n_repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat_idx)
        splits.extend(list(splitter.split(np.zeros(len(labels)), labels, groups)))
    return splits


def run_grouped_auc_over_time(samples, labels, groups, n_splits, n_repeats, decoding_step, seed, shuffle=False):
    samples = np.asarray(samples, dtype=float)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    labels_use = _permute_binary_labels_within_groups(labels, groups, seed) if shuffle else labels.copy()

    splits = _iter_group_splits(labels_use, groups, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    time_indices = np.arange(0, samples.shape[2], decoding_step)
    scores = np.zeros((len(time_indices), len(splits)))
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))

    for time_offset, time_index in enumerate(time_indices):
        samples_at_time = samples[:, :, time_index]
        for split_index, (train_idx, test_idx) in enumerate(splits):
            if len(np.unique(labels_use[train_idx])) < 2 or len(np.unique(labels_use[test_idx])) < 2:
                scores[time_offset, split_index] = 0.5
                continue
            clf.fit(samples_at_time[train_idx], labels_use[train_idx])
            probabilities = clf.predict_proba(samples_at_time[test_idx])[:, 1]
            scores[time_offset, split_index] = safe_auc(labels_use[test_idx], probabilities)
    return scores