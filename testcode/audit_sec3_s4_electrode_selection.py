import csv
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEWANALYSE_ROOT = PROJECT_ROOT / 'newanalyse'
if str(NEWANALYSE_ROOT) not in sys.path:
    sys.path.insert(0, str(NEWANALYSE_ROOT))

import Sec3_s4_all_electrode_decoding_importance as sec3_s4
from newanalyse_paths import get_report_dir, project_root


def invert_roi_map(roi_map):
    channel_to_rois = {}
    for roi_name, channels in roi_map.items():
        for channel_name in channels:
            channel_to_rois.setdefault(channel_name, []).append(roi_name)
    for channel_name in channel_to_rois:
        channel_to_rois[channel_name] = sorted(set(channel_to_rois[channel_name]))
    return channel_to_rois


def inspect_feature_files(feature_dir, roi_map):
    selected_channels = []
    selected_channel_set = set()
    selected_from_roi = {}
    duplicate_rows = []
    skipped_rows = []
    valid_rois = []

    mat_files = sorted(path for path in feature_dir.glob(sec3_s4.ROI_PATTERN) if path.is_file() and path.suffix == '.mat')
    for mat_file in mat_files:
        roi_name = mat_file.stem
        if roi_name in sec3_s4.SKIP_ROIS:
            skipped_rows.append(
                {
                    'roi_name': roi_name,
                    'status': 'skipped_by_config',
                    'detail': 'roi_name_in_skip_list',
                }
            )
            continue

        try:
            mat = sio.loadmat(mat_file)
        except Exception as exc:
            skipped_rows.append(
                {
                    'roi_name': roi_name,
                    'status': 'unreadable_feature_file',
                    'detail': f'{type(exc).__name__}: {exc}',
                }
            )
            continue

        task_arrays = {}
        missing_fields = []
        for task_name in ('task1', 'task2', 'task3'):
            field_name = sec3_s4.build_task_field(task_name)
            if field_name not in mat:
                missing_fields.append(field_name)
                continue
            task_arrays[task_name] = np.asarray(mat[field_name], dtype=float)

        if missing_fields:
            skipped_rows.append(
                {
                    'roi_name': roi_name,
                    'status': 'missing_required_fields',
                    'detail': ','.join(missing_fields),
                }
            )
            continue

        valid_rois.append(roi_name)
        n_channels = task_arrays['task1'].shape[2]
        channel_names = sec3_s4.infer_channel_names(mat, roi_name, roi_map, n_channels)
        for channel_name in channel_names:
            if channel_name in selected_channel_set:
                duplicate_rows.append(
                    {
                        'channel': channel_name,
                        'kept_from_roi': selected_from_roi[channel_name],
                        'duplicate_roi': roi_name,
                    }
                )
                continue
            selected_channel_set.add(channel_name)
            selected_channels.append(channel_name)
            selected_from_roi[channel_name] = roi_name

    skipped_by_roi = {row['roi_name']: row for row in skipped_rows}
    return {
        'selected_channels': selected_channels,
        'selected_from_roi': selected_from_roi,
        'duplicate_rows': duplicate_rows,
        'skipped_rows': skipped_rows,
        'skipped_by_roi': skipped_by_roi,
        'valid_rois': valid_rois,
    }


def classify_channel(channel_name, selected_channel_set, selected_from_roi, channel_to_rois, feature_info):
    candidate_rois = channel_to_rois.get(channel_name, [])
    if channel_name in selected_channel_set:
        return {
            'channel': channel_name,
            'status': 'kept',
            'selected_from_roi': selected_from_roi[channel_name],
            'candidate_rois': ';'.join(candidate_rois),
            'reason': 'selected',
        }

    if not candidate_rois:
        return {
            'channel': channel_name,
            'status': 'removed',
            'selected_from_roi': '',
            'candidate_rois': '',
            'reason': 'no_roi_mapping',
        }

    skipped_labels = []
    valid_labels = []
    unresolved_labels = []
    for roi_name in candidate_rois:
        if roi_name in feature_info['skipped_by_roi']:
            row = feature_info['skipped_by_roi'][roi_name]
            skipped_labels.append(f"{roi_name}:{row['status']}")
        elif roi_name in feature_info['valid_rois']:
            valid_labels.append(roi_name)
        else:
            unresolved_labels.append(f'{roi_name}:no_feature_file')

    if skipped_labels and not valid_labels:
        reason = ';'.join(skipped_labels)
    elif unresolved_labels and not valid_labels:
        reason = ';'.join(unresolved_labels)
    elif skipped_labels or unresolved_labels:
        pieces = skipped_labels + unresolved_labels + [f'valid_rois={";".join(valid_labels)}']
        reason = ';'.join(pieces)
    else:
        reason = 'not_selected_by_sec3_s4_rules'

    return {
        'channel': channel_name,
        'status': 'removed',
        'selected_from_roi': '',
        'candidate_rois': ';'.join(candidate_rois),
        'reason': reason,
    }


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    start_time = time.time()
    subject = sec3_s4.SUBJECT
    feature_kind = sec3_s4.FEATURE_KIND
    base_path = project_root()

    _, feature_dir, loc_file, task_to_file = sec3_s4.build_subject_paths(subject)
    common_channels = sec3_s4.load_common_channels(task_to_file)
    common_channel_set = set(common_channels)

    roi_map = {}
    if sec3_s4.FEATURE_CONFIG[feature_kind]['channel_mode'] == 'roi_map':
        roi_map = sec3_s4.get_roi_map(loc_file, common_channels)
    channel_to_rois = invert_roi_map(roi_map)

    feature_info = inspect_feature_files(feature_dir, roi_map)
    selected_channel_set = set(feature_info['selected_channels'])
    audit_rows = [
        classify_channel(
            channel_name,
            selected_channel_set,
            feature_info['selected_from_roi'],
            channel_to_rois,
            feature_info,
        )
        for channel_name in common_channels
    ]

    removed_rows = [row for row in audit_rows if row['status'] == 'removed']
    report_dir = get_report_dir(base_path, 'sec3_s4_electrode_audit') / feature_kind / subject
    audit_csv = report_dir / 'electrode_audit.csv'
    removed_csv = report_dir / 'removed_electrodes.csv'
    duplicate_csv = report_dir / 'duplicate_channel_assignments.csv'
    skipped_csv = report_dir / 'skipped_feature_rois.csv'

    write_csv(
        audit_csv,
        ['channel', 'status', 'selected_from_roi', 'candidate_rois', 'reason'],
        audit_rows,
    )
    write_csv(
        removed_csv,
        ['channel', 'status', 'selected_from_roi', 'candidate_rois', 'reason'],
        removed_rows,
    )
    write_csv(
        duplicate_csv,
        ['channel', 'kept_from_roi', 'duplicate_roi'],
        feature_info['duplicate_rows'],
    )
    write_csv(
        skipped_csv,
        ['roi_name', 'status', 'detail'],
        feature_info['skipped_rows'],
    )

    print(f'Common channels: {len(common_channel_set)}')
    print(f'Selected channels: {len(selected_channel_set)}')
    print(f'Removed channels: {len(removed_rows)}')
    if removed_rows:
        print('Removed electrode names:')
        print(', '.join(row['channel'] for row in removed_rows))
    print(f'Duplicate channel assignments: {len(feature_info["duplicate_rows"])}')
    print(f'Skipped ROI files: {len(feature_info["skipped_rows"])}')
    print(f'Audit CSV: {audit_csv}')
    print(f'Removed CSV: {removed_csv}')
    print(f'Duplicate CSV: {duplicate_csv}')
    print(f'Skipped ROI CSV: {skipped_csv}')
    print(f'Total runtime: {time.time() - start_time:.2f} s')


if __name__ == '__main__':
    main()