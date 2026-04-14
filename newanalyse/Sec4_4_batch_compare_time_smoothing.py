import csv
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from batch_runner_utils import apply_python_overrides, load_python_module
from newanalyse_paths import get_smoothing_compare_summary_dir, get_within_decoding_batch_dir, get_within_decoding_task_dir, project_root


BASE_PATH = project_root()
NEWANALYSE_DIR = BASE_PATH / 'newanalyse'

SUBJECTS = ['test001','test003']
SMOOTH_WINDOWS = [0, 2, 5, 10, 20]

STOP_ON_ERROR = True
DRY_RUN = False

PLOT_DPI = 240
PLOT_YLIM = (0.35, 1.0)
CATEGORY_SUFFIXES = {'face', 'body', 'object', 'scene'}
SMOOTH_LINEWIDTH_RANGE = (2.0, 4.8)

BATCH_NAME_ROOT = 'batch_smooth_compare_real_only'
SUMMARY_DIR = get_smoothing_compare_summary_dir(BASE_PATH)
FIGURE_DIR = SUMMARY_DIR / 'figures'

MODALITY_STEPS = [
    {
        'enabled': True,
        'name': 'ERP',
        'script': NEWANALYSE_DIR / 'Sec3_1_all_roi_result_erp.py',
        'feature_subdir': 'decoding_erp_features',
        'output_folder': 'erp',
    },
    {
        'enabled': True,
        'name': 'High-Gamma',
        'script': NEWANALYSE_DIR / 'Sec3_2_all_roi_result_high.py',
        'feature_subdir': 'decoding_highgamma_features',
        'output_folder': 'highgamma',
    },
    {
        'enabled': True,
        'name': 'Low-Gamma',
        'script': NEWANALYSE_DIR / 'Sec3_3_all_roi_result_low.py',
        'feature_subdir': 'decoding_lowgamma_features',
        'output_folder': 'lowgamma',
    },
    {
        'enabled': True,
        'name': 'TFA',
        'script': NEWANALYSE_DIR / 'Sec3_4_all_roi_result_tfa.py',
        'feature_subdir': 'decoding_tfa_features',
        'output_folder': 'tfa',
    },
    {
        'enabled': True,
        'name': 'Gamma Multiband',
        'script': NEWANALYSE_DIR / 'Sec3_6_all_roi_result_gamma_multiband.py',
        'feature_subdir': 'decoding_gamma_multiband_features',
        'output_folder': 'gamma_multiband',
    },
]


def format_seconds(seconds):
    return f'{seconds:.1f}s'


def smooth_tag(smooth_win):
    return f'smooth_{int(smooth_win)}'


def build_batch_name(smooth_win):
    return str(Path(BATCH_NAME_ROOT) / smooth_tag(smooth_win))


def build_task_output_dir(subject, smooth_win, output_folder, task_id):
    return get_within_decoding_task_dir(
        BASE_PATH,
        task_id=task_id,
        data_type=output_folder,
        subject=subject,
        perm_tag='real_only',
        variant='with_sti',
        batch_name=build_batch_name(smooth_win),
    )


def sanitize_name(name):
    text = ''.join(ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in str(name))
    return text.strip('_') or 'Unknown'


def should_include_result_file(npz_path, data):
    if 'category_names' in data.files:
        return True
    stem = npz_path.stem
    if not stem.endswith('_results'):
        return False
    base_name = stem[:-len('_results')]
    suffix = base_name.rsplit('_', 1)[-1].lower()
    if suffix in CATEGORY_SUFFIXES:
        return False
    return True


def load_task_curve_summary(task_dir, plot_times):
    cache_dir = task_dir / 'computed_results'
    if not cache_dir.exists():
        return None

    curve_list = []
    for npz_path in sorted(cache_dir.glob('*.npz')):
        with np.load(npz_path, allow_pickle=True) as data:
            if not should_include_result_file(npz_path, data):
                continue
            if 'mean_auc' not in data.files:
                continue
            curve = np.asarray(data['mean_auc'], dtype=float).reshape(-1)
            if curve.size == 0:
                continue
            curve_list.append(curve)

    if not curve_list:
        return None

    min_len = min(curve.size for curve in curve_list)
    aligned = np.stack([curve[:min_len] for curve in curve_list], axis=0)
    return {
        'plot_times': np.asarray(plot_times[:min_len], dtype=float),
        'mean_curve': np.mean(aligned, axis=0),
        'sem_curve': np.std(aligned, axis=0, ddof=0) / np.sqrt(aligned.shape[0]),
        'n_rois': aligned.shape[0],
    }


def collect_output_curves(subject, smooth_win, output_folder, tasks, plot_times):
    task_curves = {}
    for task in tasks:
        task_dir = build_task_output_dir(subject, smooth_win, output_folder, task['id'])
        if not task_dir.exists():
            continue
        summary = load_task_curve_summary(task_dir, plot_times)
        if summary is not None:
            task_curves[task['id']] = summary
    return task_curves


def ensure_curve_store_entry(curve_store, subject, modality, task_name):
    return curve_store.setdefault(subject, {}).setdefault(modality, {}).setdefault(task_name, {})


def build_smoothing_style_map(smooth_values):
    ordered_values = sorted(int(value) for value in smooth_values)
    if not ordered_values:
        return {}

    cmap = plt.get_cmap('coolwarm')
    min_linewidth, max_linewidth = SMOOTH_LINEWIDTH_RANGE
    if len(ordered_values) == 1:
        return {
            ordered_values[0]: {
                'color': cmap(0.0),
                'linewidth': max_linewidth,
            }
        }

    style_map = {}
    for index, smooth_win in enumerate(ordered_values):
        ratio = index / (len(ordered_values) - 1)
        style_map[smooth_win] = {
            'color': cmap(ratio),
            'linewidth': min_linewidth + ratio * (max_linewidth - min_linewidth),
        }
    return style_map


def plot_modality_smoothing_figure(subject, modality, task_name, smooth_curves, save_path):
    ordered = sorted(smooth_curves.items(), key=lambda item: item[0])
    style_map = build_smoothing_style_map(smooth_curves.keys())
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for smooth_win, payload in ordered:
        style = style_map.get(int(smooth_win), {'color': None, 'linewidth': SMOOTH_LINEWIDTH_RANGE[0]})
        label = f'smooth={smooth_win} | n_roi={payload["n_rois"]}'
        ax.plot(
            payload['plot_times'],
            payload['mean_curve'],
            color=style['color'],
            linewidth=style['linewidth'],
            label=label,
        )

    ax.axhline(0.5, color='black', linestyle=':', linewidth=1.0)
    ax.axvline(0.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title(f'{subject} | {modality} | {task_name} | smoothing comparison')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Mean ROC AUC across ROIs')
    ax.set_ylim(*PLOT_YLIM)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def write_curve_summary_csv(records, save_path):
    if not records:
        return None
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['subject', 'modality', 'task_name', 'smooth_win', 'n_rois', 'peak_auc', 'peak_time_ms', 'mean_auc', 'figure_path'],
        )
        writer.writeheader()
        writer.writerows(records)
    return save_path


def build_visualizations(curve_store):
    figure_records = []
    generated_paths = []
    for subject, modality_dict in curve_store.items():
        for modality, task_dict in modality_dict.items():
            for task_name, smooth_curves in task_dict.items():
                if not smooth_curves:
                    continue
                figure_path = FIGURE_DIR / subject / f'{sanitize_name(modality)}__{sanitize_name(task_name)}.png'
                plot_modality_smoothing_figure(subject, modality, task_name, smooth_curves, figure_path)
                generated_paths.append(figure_path)

                for smooth_win, payload in sorted(smooth_curves.items(), key=lambda item: item[0]):
                    peak_idx = int(np.argmax(payload['mean_curve']))
                    figure_records.append(
                        {
                            'subject': subject,
                            'modality': modality,
                            'task_name': task_name,
                            'smooth_win': int(smooth_win),
                            'n_rois': int(payload['n_rois']),
                            'peak_auc': float(payload['mean_curve'][peak_idx]),
                            'peak_time_ms': float(payload['plot_times'][peak_idx]),
                            'mean_auc': float(np.mean(payload['mean_curve'])),
                            'figure_path': str(figure_path),
                        }
                    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = FIGURE_DIR / f'smoothing_curve_summary_{timestamp}.csv'
    summary_path = write_curve_summary_csv(figure_records, csv_path)
    return generated_paths, summary_path


def run_one(subject, smooth_win, step):
    print(f"Run | subject={subject} | smooth={smooth_win} | modality={step['name']}")
    if DRY_RUN:
        return 0.0, build_expected_output_root(subject, smooth_win, step['output_folder']), None

    module = load_python_module(step['script'], module_name_prefix='smooth_compare')
    apply_python_overrides(
        module,
        subject=subject,
        feature_subdir=step['feature_subdir'],
        overrides={
            'time_smooth_win': smooth_win,
            'run_permutation_test': False,
            'n_perms': 0,
            'n_repeats_perm': 0,
            'batch_name': build_batch_name(smooth_win),
        },
    )

    start_time = time.time()
    module.main()
    duration = time.time() - start_time
    batch_root = get_within_decoding_batch_dir(
        BASE_PATH,
        data_type=step['output_folder'],
        subject=subject,
        batch_name=build_batch_name(smooth_win),
    )
    return duration, batch_root, np.asarray(module.PLOT_TIMES, dtype=float), list(module.TASKS)


def write_summary(records):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = SUMMARY_DIR / f'smooth_compare_summary_{timestamp}.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['subject', 'smooth_win', 'modality', 'duration_sec', 'output_root'],
        )
        writer.writeheader()
        writer.writerows(records)
    return csv_path


def main():
    total_start = time.time()
    records = []
    curve_store = {}
    for subject in SUBJECTS:
        for smooth_win in SMOOTH_WINDOWS:
            for step in MODALITY_STEPS:
                if not step['enabled']:
                    continue
                try:
                    duration, output_root, plot_times, tasks = run_one(subject, smooth_win, step)
                    print(f"Finished in {format_seconds(duration)}")
                    records.append(
                        {
                            'subject': subject,
                            'smooth_win': int(smooth_win),
                            'modality': step['name'],
                            'duration_sec': round(duration, 3),
                            'output_root': str(output_root),
                        }
                    )
                    if not DRY_RUN and plot_times is not None:
                        task_curves = collect_output_curves(subject, smooth_win, step['output_folder'], tasks, plot_times)
                        for task_name, payload in task_curves.items():
                            smooth_map = ensure_curve_store_entry(curve_store, subject, step['name'], task_name)
                            smooth_map[int(smooth_win)] = payload
                except Exception as exc:
                    print(f"[ERROR] subject={subject} | smooth={smooth_win} | modality={step['name']} | {exc}")
                    if STOP_ON_ERROR:
                        raise

    summary_path = write_summary(records)
    generated_figures, figure_summary_path = build_visualizations(curve_store) if not DRY_RUN else ([], None)
    total_duration = time.time() - total_start
    print(f'Summary saved: {summary_path}')
    if generated_figures:
        print(f'Generated smoothing figures: {len(generated_figures)}')
        for figure_path in generated_figures:
            print(f'Figure: {figure_path}')
    if figure_summary_path is not None:
        print(f'Figure summary saved: {figure_summary_path}')
    print(f'All smoothing comparisons finished in {format_seconds(total_duration)}')


if __name__ == '__main__':
    main()