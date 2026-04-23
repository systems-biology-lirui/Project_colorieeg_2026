import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import (
    build_common_parser,
    build_output_dir,
    load_preprocessed_paired_dataset,
    normalize_common_args,
    reduce_time_window,
    resolve_selected_category_indices,
    select_paired_dataset_categories,
)


WINDOW_START_MS = 20
WINDOW_END_MS = 100

COLOR_MAP = {
    'color': '#d62728',
    'gray': '#4c78a8',
}
MARKERS = ['o', 's', '^', 'D']


def build_parser():
    parser = build_common_parser('PCA visualization for paired Task1 color-gray samples.')
    parser.add_argument('--window-start-ms', type=float, default=WINDOW_START_MS)
    parser.add_argument('--window-end-ms', type=float, default=WINDOW_END_MS)
    parser.add_argument('--selected-categories', default='all', help='Comma-separated category names or indices (1-based). Use all for every category.')
    parser.add_argument('--plot-3d', action='store_true', help='Also save a 3D PCA figure using the first three principal components.')
    parser.add_argument('--plot-bars', action='store_true', help='Also save bar+scatter summaries of raw PCA scores.')
    parser.add_argument('--bar-n-components', type=int, default=3, help='Number of leading PCs to include in the bar+scatter summaries.')
    return parser


def fit_pca_projection(features, n_components=2):
    standardized = StandardScaler().fit_transform(features)
    max_components = min(standardized.shape[0], standardized.shape[1])
    if max_components < n_components:
        raise ValueError(
            f'Cannot fit PCA with {n_components} components from shape {standardized.shape}. '
            f'Max available components: {max_components}.'
        )
    pca = PCA(n_components=n_components, random_state=0)
    projection = pca.fit_transform(standardized)
    return projection, pca


def dedupe_legend(handles, labels):
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            unique_handles.append(handle)
            unique_labels.append(label)
            seen.add(label)
    return unique_handles, unique_labels


def compute_bar_positions(n_components, n_series, group_width=0.82):
    centers = np.arange(n_components, dtype=float)
    if n_series <= 0:
        return centers, np.zeros((0, n_components), dtype=float), 0.0
    bar_width = group_width / float(n_series)
    offsets = (np.arange(n_series, dtype=float) - 0.5 * (n_series - 1)) * bar_width
    return centers, centers[None, :] + offsets[:, None], bar_width


def _make_rng(seed=0):
    return np.random.RandomState(seed)


def _sem(values):
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _pvalue_to_stars(pvalue):
    if not np.isfinite(pvalue):
        return 'n.s.'
    if pvalue < 0.001:
        return '***'
    if pvalue < 0.01:
        return '**'
    if pvalue < 0.05:
        return '*'
    return 'n.s.'


def plot_condition_bar_panel(ax, color_projection, gray_projection, paired_dataset, pca_model):
    n_components = pca_model.n_components_
    series = []
    for category_idx, category_name in enumerate(paired_dataset.category_names):
        mask = paired_dataset.category_ids == category_idx
        series.append(
            {
                'label': f'{category_name} color',
                'values': color_projection[mask],
                'facecolor': COLOR_MAP['color'],
                'edgecolor': COLOR_MAP['color'],
                'marker': MARKERS[category_idx % len(MARKERS)],
                'alpha': 0.55,
                'filled': True,
            }
        )
        series.append(
            {
                'label': f'{category_name} gray',
                'values': gray_projection[mask],
                'facecolor': 'white',
                'edgecolor': COLOR_MAP['gray'],
                'marker': MARKERS[category_idx % len(MARKERS)],
                'alpha': 0.95,
                'filled': False,
            }
        )

    centers, positions, bar_width = compute_bar_positions(n_components, len(series))
    rng = _make_rng(0)

    for series_idx, series_item in enumerate(series):
        values = np.asarray(series_item['values'], dtype=float)
        means = values.mean(axis=0)
        sems = np.asarray([_sem(values[:, component_idx]) for component_idx in range(n_components)], dtype=float)
        ax.bar(
            positions[series_idx],
            means,
            width=bar_width * 0.92,
            color=series_item['facecolor'],
            edgecolor=series_item['edgecolor'],
            linewidth=1.2,
            alpha=series_item['alpha'],
            label=series_item['label'],
            yerr=sems,
            error_kw={'elinewidth': 1.0, 'ecolor': series_item['edgecolor'], 'capsize': 2},
            zorder=2,
        )
        for component_idx in range(n_components):
            x_values = positions[series_idx, component_idx] + rng.uniform(-bar_width * 0.18, bar_width * 0.18, size=values.shape[0])
            if series_item['filled']:
                ax.scatter(x_values, values[:, component_idx], s=18, color=series_item['edgecolor'], marker=series_item['marker'], alpha=0.65, zorder=3)
            else:
                ax.scatter(x_values, values[:, component_idx], s=18, facecolors='none', edgecolors=series_item['edgecolor'], linewidths=1.0, marker=series_item['marker'], alpha=0.9, zorder=3)

    ax.axhline(0.0, color='#999999', linestyle=':', linewidth=1.0)
    ax.set_xticks(centers)
    ax.set_xticklabels([f'PC{idx + 1}' for idx in range(n_components)])
    ax.set_ylabel('PCA score')
    ax.set_title('Raw PCA scores by condition')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    variance_text = ' | '.join([f'PC{idx + 1}={ratio * 100:.1f}%' for idx, ratio in enumerate(pca_model.explained_variance_ratio_)])
    ax.text(0.99, 0.99, variance_text, transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '#cccccc'})


def plot_group_bar_panel(ax, color_projection, gray_projection, pca_model):
    n_components = pca_model.n_components_
    centers, positions, bar_width = compute_bar_positions(n_components, 2)
    rng = _make_rng(1)

    color_means = color_projection.mean(axis=0)
    gray_means = gray_projection.mean(axis=0)
    color_sems = np.asarray([_sem(color_projection[:, component_idx]) for component_idx in range(n_components)], dtype=float)
    gray_sems = np.asarray([_sem(gray_projection[:, component_idx]) for component_idx in range(n_components)], dtype=float)

    ax.bar(
        positions[0], color_means, width=bar_width * 0.9, color=COLOR_MAP['color'], edgecolor=COLOR_MAP['color'],
        linewidth=1.2, alpha=0.55, label='color', yerr=color_sems,
        error_kw={'elinewidth': 1.0, 'ecolor': COLOR_MAP['color'], 'capsize': 2}, zorder=2,
    )
    ax.bar(
        positions[1], gray_means, width=bar_width * 0.9, color='white', edgecolor=COLOR_MAP['gray'],
        linewidth=1.2, alpha=0.95, label='gray', yerr=gray_sems,
        error_kw={'elinewidth': 1.0, 'ecolor': COLOR_MAP['gray'], 'capsize': 2}, zorder=2,
    )

    pvalues = []
    for component_idx in range(n_components):
        stat = ttest_rel(color_projection[:, component_idx], gray_projection[:, component_idx], nan_policy='omit')
        pvalue = float(stat.pvalue) if np.isfinite(stat.pvalue) else np.nan
        pvalues.append(pvalue)
        color_x = positions[0, component_idx] + rng.uniform(-bar_width * 0.14, bar_width * 0.14, size=color_projection.shape[0])
        gray_x = positions[1, component_idx] + rng.uniform(-bar_width * 0.14, bar_width * 0.14, size=gray_projection.shape[0])
        ax.scatter(color_x, color_projection[:, component_idx], s=16, color=COLOR_MAP['color'], alpha=0.55, zorder=3)
        ax.scatter(gray_x, gray_projection[:, component_idx], s=16, facecolors='none', edgecolors=COLOR_MAP['gray'], linewidths=1.0, alpha=0.9, zorder=3)

    y_min = min(np.min(color_projection), np.min(gray_projection))
    y_max = max(np.max(color_projection), np.max(gray_projection))
    y_range = max(y_max - y_min, 1e-6)
    for component_idx, pvalue in enumerate(pvalues):
        bracket_y = max(color_means[component_idx] + color_sems[component_idx], gray_means[component_idx] + gray_sems[component_idx]) + 0.12 * y_range
        x_left = positions[0, component_idx]
        x_right = positions[1, component_idx]
        ax.plot([x_left, x_left, x_right, x_right], [bracket_y - 0.02 * y_range, bracket_y, bracket_y, bracket_y - 0.02 * y_range], color='#444444', linewidth=1.0, zorder=4)
        ax.text(0.5 * (x_left + x_right), bracket_y + 0.02 * y_range, _pvalue_to_stars(pvalue), ha='center', va='bottom', fontsize=10)

    ax.axhline(0.0, color='#999999', linestyle=':', linewidth=1.0)
    ax.set_xticks(centers)
    ax.set_xticklabels([f'PC{idx + 1}' for idx in range(n_components)])
    ax.set_ylabel('PCA score')
    ax.set_title('Raw PCA scores: color vs gray')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    variance_text = ' | '.join([f'PC{idx + 1}={ratio * 100:.1f}%' for idx, ratio in enumerate(pca_model.explained_variance_ratio_)])
    ax.text(0.99, 0.99, variance_text, transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '#cccccc'})
    return np.asarray(pvalues, dtype=float)


def plot_raw_panel(ax, raw_color, raw_gray, paired_dataset, pca_model):
    for pair_idx in range(raw_color.shape[0]):
        ax.plot(
            [raw_color[pair_idx, 0], raw_gray[pair_idx, 0]],
            [raw_color[pair_idx, 1], raw_gray[pair_idx, 1]],
            color='#bbbbbb',
            linewidth=0.8,
            alpha=0.5,
            zorder=1,
        )

    for category_idx, category_name in enumerate(paired_dataset.category_names):
        mask = paired_dataset.category_ids == category_idx
        marker = MARKERS[category_idx % len(MARKERS)]
        ax.scatter(raw_color[mask, 0], raw_color[mask, 1], s=45, marker=marker, color=COLOR_MAP['color'], alpha=0.85, label=f'{category_name} color')
        ax.scatter(raw_gray[mask, 0], raw_gray[mask, 1], s=45, marker=marker, facecolors='none', edgecolors=COLOR_MAP['gray'], linewidths=1.2, alpha=0.9, label=f'{category_name} gray')

    ax.set_title('Raw paired samples')
    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% var)')
    ax.grid(True, linestyle='--', alpha=0.35)


def plot_centered_panel(ax, centered_color, centered_gray, paired_dataset, pca_model):
    for pair_idx in range(centered_color.shape[0]):
        ax.plot(
            [centered_color[pair_idx, 0], centered_gray[pair_idx, 0]],
            [centered_color[pair_idx, 1], centered_gray[pair_idx, 1]],
            color='#cccccc',
            linewidth=0.8,
            alpha=0.5,
            zorder=1,
        )

    for category_idx, category_name in enumerate(paired_dataset.category_names):
        mask = paired_dataset.category_ids == category_idx
        marker = MARKERS[category_idx % len(MARKERS)]
        ax.scatter(centered_color[mask, 0], centered_color[mask, 1], s=40, marker=marker, color=COLOR_MAP['color'], alpha=0.8, label=f'{category_name} color centered')
        ax.scatter(centered_gray[mask, 0], centered_gray[mask, 1], s=40, marker=marker, facecolors='none', edgecolors=COLOR_MAP['gray'], linewidths=1.2, alpha=0.9, label=f'{category_name} gray centered')

    ax.axhline(0.0, color='#999999', linestyle=':', linewidth=1.0)
    ax.axvline(0.0, color='#999999', linestyle=':', linewidth=1.0)
    ax.set_title('Within-pair centered samples')
    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% var)')
    ax.grid(True, linestyle='--', alpha=0.35)


def plot_difference_panel(ax, difference_projection, paired_dataset, pca_model):
    for category_idx, category_name in enumerate(paired_dataset.category_names):
        mask = paired_dataset.category_ids == category_idx
        marker = MARKERS[category_idx % len(MARKERS)]
        ax.scatter(difference_projection[mask, 0], difference_projection[mask, 1], s=45, marker=marker, alpha=0.85, label=category_name)

    ax.axhline(0.0, color='#999999', linestyle=':', linewidth=1.0)
    ax.axvline(0.0, color='#999999', linestyle=':', linewidth=1.0)
    ax.set_title('Pair differences (color - gray)')
    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% var)')
    ax.grid(True, linestyle='--', alpha=0.35)


def plot_raw_panel_3d(ax, raw_color, raw_gray, paired_dataset, pca_model):
    for pair_idx in range(raw_color.shape[0]):
        ax.plot(
            [raw_color[pair_idx, 0], raw_gray[pair_idx, 0]],
            [raw_color[pair_idx, 1], raw_gray[pair_idx, 1]],
            [raw_color[pair_idx, 2], raw_gray[pair_idx, 2]],
            color='#bbbbbb',
            linewidth=0.8,
            alpha=0.5,
            zorder=1,
        )

    for category_idx, category_name in enumerate(paired_dataset.category_names):
        mask = paired_dataset.category_ids == category_idx
        marker = MARKERS[category_idx % len(MARKERS)]
        ax.scatter(raw_color[mask, 0], raw_color[mask, 1], raw_color[mask, 2], s=45, marker=marker, color=COLOR_MAP['color'], alpha=0.85, label=f'{category_name} color')
        ax.scatter(raw_gray[mask, 0], raw_gray[mask, 1], raw_gray[mask, 2], s=45, marker=marker, facecolors='none', edgecolors=COLOR_MAP['gray'], linewidths=1.2, alpha=0.9, label=f'{category_name} gray')

    ax.set_title('Raw paired samples')
    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% var)')
    ax.set_zlabel(f'PC3 ({pca_model.explained_variance_ratio_[2] * 100:.1f}% var)')


def plot_centered_panel_3d(ax, centered_color, centered_gray, paired_dataset, pca_model):
    for pair_idx in range(centered_color.shape[0]):
        ax.plot(
            [centered_color[pair_idx, 0], centered_gray[pair_idx, 0]],
            [centered_color[pair_idx, 1], centered_gray[pair_idx, 1]],
            [centered_color[pair_idx, 2], centered_gray[pair_idx, 2]],
            color='#cccccc',
            linewidth=0.8,
            alpha=0.5,
            zorder=1,
        )

    for category_idx, category_name in enumerate(paired_dataset.category_names):
        mask = paired_dataset.category_ids == category_idx
        marker = MARKERS[category_idx % len(MARKERS)]
        ax.scatter(centered_color[mask, 0], centered_color[mask, 1], centered_color[mask, 2], s=40, marker=marker, color=COLOR_MAP['color'], alpha=0.8, label=f'{category_name} color centered')
        ax.scatter(centered_gray[mask, 0], centered_gray[mask, 1], centered_gray[mask, 2], s=40, marker=marker, facecolors='none', edgecolors=COLOR_MAP['gray'], linewidths=1.2, alpha=0.9, label=f'{category_name} gray centered')

    ax.set_title('Within-pair centered samples')
    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% var)')
    ax.set_zlabel(f'PC3 ({pca_model.explained_variance_ratio_[2] * 100:.1f}% var)')


def plot_difference_panel_3d(ax, difference_projection, paired_dataset, pca_model):
    for category_idx, category_name in enumerate(paired_dataset.category_names):
        mask = paired_dataset.category_ids == category_idx
        marker = MARKERS[category_idx % len(MARKERS)]
        ax.scatter(difference_projection[mask, 0], difference_projection[mask, 1], difference_projection[mask, 2], s=45, marker=marker, alpha=0.85, label=category_name)

    ax.set_title('Pair differences (color - gray)')
    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% var)')
    ax.set_zlabel(f'PC3 ({pca_model.explained_variance_ratio_[2] * 100:.1f}% var)')


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.bar_n_components < 1:
        raise ValueError('--bar-n-components must be >= 1.')
    config = normalize_common_args(args)
    roi_bundle, paired_dataset = load_preprocessed_paired_dataset(config)
    selected_category_indices = resolve_selected_category_indices(args.selected_categories, config.category_names)
    paired_dataset = select_paired_dataset_categories(paired_dataset, selected_category_indices)

    raw_color_matrix = reduce_time_window(paired_dataset.color, roi_bundle.time_vector, args.window_start_ms, args.window_end_ms)
    raw_gray_matrix = reduce_time_window(paired_dataset.gray, roi_bundle.time_vector, args.window_start_ms, args.window_end_ms)

    raw_projection, raw_pca = fit_pca_projection(np.concatenate([raw_color_matrix, raw_gray_matrix], axis=0), n_components=2)
    raw_color_projection = raw_projection[:raw_color_matrix.shape[0]]
    raw_gray_projection = raw_projection[raw_color_matrix.shape[0]:]

    pair_mean_matrix = 0.5 * (raw_color_matrix + raw_gray_matrix)
    centered_projection, centered_pca = fit_pca_projection(np.concatenate([raw_color_matrix - pair_mean_matrix, raw_gray_matrix - pair_mean_matrix], axis=0), n_components=2)
    centered_color_projection = centered_projection[:raw_color_matrix.shape[0]]
    centered_gray_projection = centered_projection[raw_color_matrix.shape[0]:]

    difference_projection, difference_pca = fit_pca_projection(raw_color_matrix - raw_gray_matrix, n_components=2)

    raw_bar_projection = None
    raw_bar_pca = None
    raw_bar_color_projection = None
    raw_bar_gray_projection = None
    group_bar_pvalues = None
    if args.plot_bars:
        raw_bar_projection, raw_bar_pca = fit_pca_projection(np.concatenate([raw_color_matrix, raw_gray_matrix], axis=0), n_components=args.bar_n_components)
        raw_bar_color_projection = raw_bar_projection[:raw_color_matrix.shape[0]]
        raw_bar_gray_projection = raw_bar_projection[raw_color_matrix.shape[0]:]

    raw_color_projection_3d = None
    raw_gray_projection_3d = None
    centered_color_projection_3d = None
    centered_gray_projection_3d = None
    difference_projection_3d = None
    raw_pca_3d = None
    centered_pca_3d = None
    difference_pca_3d = None
    if args.plot_3d:
        raw_projection_3d, raw_pca_3d = fit_pca_projection(np.concatenate([raw_color_matrix, raw_gray_matrix], axis=0), n_components=3)
        raw_color_projection_3d = raw_projection_3d[:raw_color_matrix.shape[0]]
        raw_gray_projection_3d = raw_projection_3d[raw_color_matrix.shape[0]:]

        centered_projection_3d, centered_pca_3d = fit_pca_projection(
            np.concatenate([raw_color_matrix - pair_mean_matrix, raw_gray_matrix - pair_mean_matrix], axis=0),
            n_components=3,
        )
        centered_color_projection_3d = centered_projection_3d[:raw_color_matrix.shape[0]]
        centered_gray_projection_3d = centered_projection_3d[raw_color_matrix.shape[0]:]

        difference_projection_3d, difference_pca_3d = fit_pca_projection(raw_color_matrix - raw_gray_matrix, n_components=3)

    out_dir = build_output_dir('pca_pairwise', config)
    fig_path = out_dir / 'pairwise_pca.png'
    fig_3d_path = out_dir / 'pairwise_pca_3d.png'
    bar_fig_path = out_dir / 'pairwise_pca_condition_bars.png'
    group_bar_fig_path = out_dir / 'pairwise_pca_color_gray_bars.png'
    npz_path = out_dir / 'pairwise_pca.npz'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plot_raw_panel(axes[0], raw_color_projection, raw_gray_projection, paired_dataset, raw_pca)
    plot_centered_panel(axes[1], centered_color_projection, centered_gray_projection, paired_dataset, centered_pca)
    plot_difference_panel(axes[2], difference_projection, paired_dataset, difference_pca)
    handles, labels = axes[0].get_legend_handles_labels()
    unique_handles, unique_labels = dedupe_legend(handles, labels)
    fig.legend(unique_handles, unique_labels, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f'Task1 paired PCA | {config.subject} | {config.feature_kind} | {config.roi_name} | '
        f'{args.window_start_ms:.0f}-{args.window_end_ms:.0f} ms',
        fontsize=13,
    )
    plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.96])
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if args.plot_3d:
        fig = plt.figure(figsize=(18, 5.5))
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        plot_raw_panel_3d(ax1, raw_color_projection_3d, raw_gray_projection_3d, paired_dataset, raw_pca_3d)
        plot_centered_panel_3d(ax2, centered_color_projection_3d, centered_gray_projection_3d, paired_dataset, centered_pca_3d)
        plot_difference_panel_3d(ax3, difference_projection_3d, paired_dataset, difference_pca_3d)
        handles, labels = ax1.get_legend_handles_labels()
        unique_handles, unique_labels = dedupe_legend(handles, labels)
        fig.legend(unique_handles, unique_labels, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            f'Task1 paired PCA (3D) | {config.subject} | {config.feature_kind} | {config.roi_name} | '
            f'{args.window_start_ms:.0f}-{args.window_end_ms:.0f} ms',
            fontsize=13,
        )
        plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.96])
        plt.savefig(fig_3d_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    if args.plot_bars:
        fig, ax = plt.subplots(figsize=(max(10, 2.2 * raw_bar_pca.n_components_ + 4), 6.2))
        plot_condition_bar_panel(ax, raw_bar_color_projection, raw_bar_gray_projection, paired_dataset, raw_bar_pca)
        handles, labels = ax.get_legend_handles_labels()
        unique_handles, unique_labels = dedupe_legend(handles, labels)
        fig.legend(unique_handles, unique_labels, loc='lower center', ncol=min(len(unique_labels), 4), fontsize=8, bbox_to_anchor=(0.5, -0.03))
        fig.suptitle(
            f'Task1 paired PCA condition bars | {config.subject} | {config.feature_kind} | {config.roi_name} | '
            f'{args.window_start_ms:.0f}-{args.window_end_ms:.0f} ms',
            fontsize=13,
        )
        plt.tight_layout(rect=[0.0, 0.08, 1.0, 0.94])
        plt.savefig(bar_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8, 1.8 * raw_bar_pca.n_components_ + 4), 6.2))
        group_bar_pvalues = plot_group_bar_panel(ax, raw_bar_color_projection, raw_bar_gray_projection, raw_bar_pca)
        fig.legend(loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.03))
        fig.suptitle(
            f'Task1 paired PCA color-gray bars | {config.subject} | {config.feature_kind} | {config.roi_name} | '
            f'{args.window_start_ms:.0f}-{args.window_end_ms:.0f} ms',
            fontsize=13,
        )
        plt.tight_layout(rect=[0.0, 0.08, 1.0, 0.94])
        plt.savefig(group_bar_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    np.savez(
        npz_path,
        subject=np.array(config.subject),
        feature_kind=np.array(config.feature_kind),
        roi_name=np.array(config.roi_name),
        grouped_data_mat=np.array(str(config.grouped_data_mat)),
        roi_path=np.array(str(roi_bundle.roi_path)),
        task_field=np.array(roi_bundle.task_field),
        category_names=np.asarray(config.category_names, dtype=object),
        selected_category_indices=np.asarray(selected_category_indices, dtype=int),
        selected_category_names=np.asarray(paired_dataset.category_names, dtype=object),
        plot_3d=np.array(bool(args.plot_3d)),
        plot_bars=np.array(bool(args.plot_bars)),
        bar_n_components=np.array(int(args.bar_n_components)),
        window_start_ms=np.array(args.window_start_ms),
        window_end_ms=np.array(args.window_end_ms),
        sample_keys=np.asarray(paired_dataset.sample_keys, dtype=object),
        category_ids=paired_dataset.category_ids,
        raw_explained_variance_ratio=raw_pca.explained_variance_ratio_,
        centered_explained_variance_ratio=centered_pca.explained_variance_ratio_,
        difference_explained_variance_ratio=difference_pca.explained_variance_ratio_,
        raw_color_projection=raw_color_projection,
        raw_gray_projection=raw_gray_projection,
        centered_color_projection=centered_color_projection,
        centered_gray_projection=centered_gray_projection,
        difference_projection=difference_projection,
        raw_explained_variance_ratio_3d=np.asarray(raw_pca_3d.explained_variance_ratio_ if raw_pca_3d is not None else [], dtype=float),
        centered_explained_variance_ratio_3d=np.asarray(centered_pca_3d.explained_variance_ratio_ if centered_pca_3d is not None else [], dtype=float),
        difference_explained_variance_ratio_3d=np.asarray(difference_pca_3d.explained_variance_ratio_ if difference_pca_3d is not None else [], dtype=float),
        raw_color_projection_3d=np.asarray(raw_color_projection_3d if raw_color_projection_3d is not None else [], dtype=float),
        raw_gray_projection_3d=np.asarray(raw_gray_projection_3d if raw_gray_projection_3d is not None else [], dtype=float),
        centered_color_projection_3d=np.asarray(centered_color_projection_3d if centered_color_projection_3d is not None else [], dtype=float),
        centered_gray_projection_3d=np.asarray(centered_gray_projection_3d if centered_gray_projection_3d is not None else [], dtype=float),
        difference_projection_3d=np.asarray(difference_projection_3d if difference_projection_3d is not None else [], dtype=float),
        raw_bar_explained_variance_ratio=np.asarray(raw_bar_pca.explained_variance_ratio_ if raw_bar_pca is not None else [], dtype=float),
        raw_bar_color_projection=np.asarray(raw_bar_color_projection if raw_bar_color_projection is not None else [], dtype=float),
        raw_bar_gray_projection=np.asarray(raw_bar_gray_projection if raw_bar_gray_projection is not None else [], dtype=float),
        raw_bar_group_pvalues=np.asarray(group_bar_pvalues if group_bar_pvalues is not None else [], dtype=float),
    )

    print(
        {
            'figure_path': str(fig_path),
            'figure_3d_path': str(fig_3d_path) if args.plot_3d else None,
            'condition_bar_figure_path': str(bar_fig_path) if args.plot_bars else None,
            'group_bar_figure_path': str(group_bar_fig_path) if args.plot_bars else None,
            'summary_path': str(npz_path),
            'matched_pairs': int(paired_dataset.color.shape[0]),
            'selected_category_names': paired_dataset.category_names,
            'window_ms': [float(args.window_start_ms), float(args.window_end_ms)],
        }
    )


if __name__ == '__main__':
    main()