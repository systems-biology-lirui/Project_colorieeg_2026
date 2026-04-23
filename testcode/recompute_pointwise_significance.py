import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_DIR = PROJECT_ROOT / 'result' / 'decoding' / 'task1_color_vs_gray_per_category' / 'erp' / 'test001' / 'batch_within_decoding' / 'perm20' / 'with_sti' / 'computed_results'
DEFAULT_ALPHA = 0.05
DEFAULT_MIN_TIME_MS = 20.0
DEFAULT_T_START = -100.0
DEFAULT_T_END = 1000.0
DEFAULT_N_POINTS = 550
DEFAULT_DECODING_STEP = 5
DEFAULT_CHANCE_LEVEL = 0.5

REAL_KEY_CANDIDATES = ('mean_auc', 'real_curve')
PERM_KEY_CANDIDATES = ('perm_dist', 'perm_dist_mean')
SEM_KEY_CANDIDATES = ('sem_auc',)
TIME_KEY_CANDIDATES = ('time_ms', 'plot_times')


def build_parser():
    parser = argparse.ArgumentParser(
        description='Recompute pointwise significance for saved decoding results without cluster correction.'
    )
    parser.add_argument('--results-dir', default=str(DEFAULT_RESULTS_DIR), help='Directory containing *_results.npz files.')
    parser.add_argument('--output-dir', default=None, help='Optional output directory. Defaults to a sibling pointwise_significance_no_cluster folder.')
    parser.add_argument('--pattern', default='*_results.npz', help='Glob pattern used inside results-dir.')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, help='Pointwise significance threshold.')
    parser.add_argument('--min-time-ms', type=float, default=DEFAULT_MIN_TIME_MS, help='Only mark timepoints at or after this time as significant.')
    parser.add_argument('--chance-level', type=float, default=DEFAULT_CHANCE_LEVEL, help='Chance-level reference line for plots.')
    parser.add_argument('--real-key', default=None, help='Optional override for the observed curve key inside the npz.')
    parser.add_argument('--perm-key', default=None, help='Optional override for the permutation distribution key inside the npz.')
    parser.add_argument('--time-key', default=None, help='Optional override for the time vector key inside the npz.')
    parser.add_argument('--t-start', type=float, default=DEFAULT_T_START, help='Fallback time vector start when no time field exists.')
    parser.add_argument('--t-end', type=float, default=DEFAULT_T_END, help='Fallback time vector end when no time field exists.')
    parser.add_argument('--n-points', type=int, default=DEFAULT_N_POINTS, help='Fallback full-resolution time point count.')
    parser.add_argument('--decoding-step', type=int, default=DEFAULT_DECODING_STEP, help='Fallback decoding step for subsampling the fallback time vector.')
    return parser


def find_first_existing_key(npz_data, requested_key, candidates, label):
    if requested_key:
        if requested_key not in npz_data:
            raise KeyError(f'Requested {label} key not found: {requested_key}')
        return requested_key
    for key in candidates:
        if key in npz_data:
            return key
    raise KeyError(f'Could not find any {label} key in result file. Tried: {candidates}')


def build_fallback_time_vector(args, n_timepoints):
    full_time = np.linspace(args.t_start, args.t_end, args.n_points, dtype=float)
    time_vector = full_time[::args.decoding_step]
    if time_vector.size < n_timepoints:
        raise ValueError(
            f'Fallback time vector has only {time_vector.size} points, but observed curve has {n_timepoints} points. '
            'Adjust --n-points or --decoding-step.'
        )
    return time_vector[:n_timepoints]


def resolve_time_vector(npz_data, args, n_timepoints):
    time_key = find_first_existing_key(npz_data, args.time_key, TIME_KEY_CANDIDATES, 'time') if (args.time_key or any(key in npz_data for key in TIME_KEY_CANDIDATES)) else None
    if time_key is None:
        return build_fallback_time_vector(args, n_timepoints), 'fallback'

    time_vector = np.asarray(npz_data[time_key], dtype=float).reshape(-1)
    if time_vector.size == n_timepoints:
        return time_vector, time_key
    if time_vector.size > n_timepoints:
        full_time = time_vector
        subsampled = full_time[::args.decoding_step]
        if subsampled.size >= n_timepoints:
            return subsampled[:n_timepoints], time_key
    raise ValueError(
        f'Time vector key {time_key} has {time_vector.size} points, but observed curve has {n_timepoints} points.'
    )


def compute_pointwise_significance(observed_curve, perm_dist, alpha, min_time_ms, time_vector):
    observed_curve = np.asarray(observed_curve, dtype=float).reshape(-1)
    perm_dist = np.asarray(perm_dist, dtype=float)
    if perm_dist.ndim != 2:
        raise ValueError(f'Permutation distribution must be 2D [n_perm, n_time], got shape {perm_dist.shape}')
    if perm_dist.shape[1] != observed_curve.size:
        raise ValueError(
            f'Permutation distribution time dimension {perm_dist.shape[1]} does not match observed curve length {observed_curve.size}.'
        )
    valid_mask = np.asarray(time_vector, dtype=float) >= float(min_time_ms)
    threshold_95 = np.nanpercentile(perm_dist, 100 * (1.0 - alpha), axis=0)
    exceed_count = np.sum(perm_dist >= observed_curve[None, :], axis=0)
    p_values = (exceed_count + 1.0) / (perm_dist.shape[0] + 1.0)
    sig_mask = (p_values <= alpha) & valid_mask
    sig_times = np.asarray(time_vector, dtype=float)[sig_mask]
    return threshold_95, p_values, sig_mask, sig_times


def summarize_sig_times(sig_times):
    if sig_times.size == 0:
        return '', np.nan, np.nan
    rounded = [f'{value:.1f}' for value in sig_times]
    return ';'.join(rounded), float(sig_times[0]), float(sig_times[-1])


def plot_pointwise_result(save_path, roi_name, observed_curve, sem_curve, threshold_95, p_values, sig_mask, time_vector, chance_level, alpha):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    if sem_curve is not None:
        ax.fill_between(time_vector, observed_curve - sem_curve, observed_curve + sem_curve, color='#1f77b4', alpha=0.22)
    ax.plot(time_vector, observed_curve, color='#1f77b4', linewidth=2.0, label='Observed curve')
    ax.plot(time_vector, threshold_95, color='#d62728', linestyle='--', linewidth=1.4, label=f'{100 * (1.0 - alpha):.0f}% permutation threshold')
    if np.any(sig_mask):
        ax.scatter(time_vector[sig_mask], observed_curve[sig_mask], color='#ff7f0e', s=26, zorder=4, label='Significant timepoints')
    ax.axhline(chance_level, color='black', linestyle=':', linewidth=1.0, label='Chance')
    ax.set_ylabel('Score')
    ax.set_title(f'{roi_name} | Pointwise significance without cluster correction')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best', fontsize=9)

    ax = axes[1]
    ax.plot(time_vector, p_values, color='#2ca02c', linewidth=1.8, label='Pointwise p-value')
    ax.axhline(alpha, color='#d62728', linestyle='--', linewidth=1.2, label=f'alpha = {alpha:.3f}')
    if np.any(sig_mask):
        ax.fill_between(time_vector, 0.0, 1.0, where=sig_mask, color='gray', alpha=0.20, transform=ax.get_xaxis_transform(), label='Significant mask')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('p-value')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f'Results directory not found: {results_dir}')

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else results_dir.parent / 'pointwise_significance_no_cluster'
    figures_dir = output_dir / 'figures'
    summaries_dir = output_dir / 'npz'
    figures_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(results_dir.glob(args.pattern))
    if not result_files:
        raise FileNotFoundError(f'No files matched {args.pattern} in {results_dir}')

    rows = []
    for result_path in result_files:
        with np.load(result_path, allow_pickle=True) as data:
            real_key = find_first_existing_key(data, args.real_key, REAL_KEY_CANDIDATES, 'observed')
            perm_key = find_first_existing_key(data, args.perm_key, PERM_KEY_CANDIDATES, 'permutation')
            observed_curve = np.asarray(data[real_key], dtype=float).reshape(-1)
            perm_dist = np.asarray(data[perm_key], dtype=float)
            sem_curve = None
            for sem_key in SEM_KEY_CANDIDATES:
                if sem_key in data:
                    candidate = np.asarray(data[sem_key], dtype=float).reshape(-1)
                    if candidate.size == observed_curve.size:
                        sem_curve = candidate
                        break
            time_vector, time_key = resolve_time_vector(data, args, observed_curve.size)

            threshold_95, p_values, sig_mask, sig_times = compute_pointwise_significance(
                observed_curve,
                perm_dist,
                args.alpha,
                args.min_time_ms,
                time_vector,
            )

            roi_name = result_path.stem.replace('_results', '')
            plot_path = figures_dir / f'{roi_name}_pointwise_significance.png'
            summary_npz_path = summaries_dir / f'{roi_name}_pointwise_significance.npz'
            sig_times_text, first_sig_ms, last_sig_ms = summarize_sig_times(sig_times)

            plot_pointwise_result(
                plot_path,
                roi_name,
                observed_curve,
                sem_curve,
                threshold_95,
                p_values,
                sig_mask,
                time_vector,
                args.chance_level,
                args.alpha,
            )

            np.savez(
                summary_npz_path,
                source_path=np.array(str(result_path)),
                roi_name=np.array(roi_name),
                real_key=np.array(real_key),
                perm_key=np.array(perm_key),
                time_key=np.array(time_key),
                alpha=np.array(args.alpha),
                min_time_ms=np.array(args.min_time_ms),
                chance_level=np.array(args.chance_level),
                time_ms=np.asarray(time_vector, dtype=float),
                observed_curve=observed_curve,
                sem_curve=np.asarray(sem_curve, dtype=float) if sem_curve is not None else np.full(observed_curve.shape, np.nan),
                perm_dist=perm_dist,
                threshold_95=threshold_95,
                p_values=p_values,
                sig_mask=sig_mask,
                sig_times_ms=sig_times,
            )

            rows.append({
                'roi_name': roi_name,
                'source_path': str(result_path),
                'real_key': real_key,
                'perm_key': perm_key,
                'time_key': time_key,
                'n_sig_timepoints': int(np.sum(sig_mask)),
                'first_sig_ms': '' if np.isnan(first_sig_ms) else f'{first_sig_ms:.1f}',
                'last_sig_ms': '' if np.isnan(last_sig_ms) else f'{last_sig_ms:.1f}',
                'sig_times_ms': sig_times_text,
                'figure_path': str(plot_path),
                'summary_npz_path': str(summary_npz_path),
            })

    csv_path = output_dir / 'pointwise_significance_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'roi_name',
                'source_path',
                'real_key',
                'perm_key',
                'time_key',
                'n_sig_timepoints',
                'first_sig_ms',
                'last_sig_ms',
                'sig_times_ms',
                'figure_path',
                'summary_npz_path',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print({
        'results_dir': str(results_dir),
        'output_dir': str(output_dir),
        'n_files': len(result_files),
        'summary_csv': str(csv_path),
    })


if __name__ == '__main__':
    main()