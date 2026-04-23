import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import (
    build_common_parser,
    build_output_dir,
    load_preprocessed_paired_dataset,
    normalize_common_args,
    resolve_selected_category_indices,
    select_paired_dataset_categories,
    smooth_time_axis,
)


DEFAULT_SMOOTH_WIN = 1

COLOR_MAP = {
    'color': '#d62728',
    'gray': '#4c78a8',
}
MARKERS = ['o', 's', '^', 'D']
PLOTLY_MARKERS = ['circle', 'square', 'cross', 'diamond']
PLOTLY_OPEN_MARKERS = {
    'circle': 'circle-open',
    'square': 'square-open',
    'cross': 'x',
    'diamond': 'diamond-open',
}


def build_parser():
    parser = build_common_parser('Dynamic PCA trajectories for paired Task1 color-gray condition means.')
    parser.add_argument('--selected-categories', default='all', help='Comma-separated category names or indices (1-based). Use all for every category.')
    parser.add_argument('--smooth-win', type=int, default=DEFAULT_SMOOTH_WIN, help='Moving-average window on the time axis for condition-mean trajectories.')
    parser.add_argument('--point-step', type=int, default=0, help='Plot an additional marker every N time points along each trajectory. Use 0 to disable intermediate markers.')
    parser.add_argument('--hide-start-end', action='store_true', help='Disable start and end point markers on the PCA trajectories.')
    parser.add_argument('--plot-start-ms', type=float, default=None, help='Start time of the trajectory window to plot and project. Default uses the first available time point.')
    parser.add_argument('--plot-end-ms', type=float, default=None, help='End time of the trajectory window to plot and project. Default uses the last available time point.')
    parser.add_argument('--interactive-3d', action='store_true', help='Also save an interactive 3D HTML figure that can be rotated in a browser.')
    return parser


def fit_pca_model(samples, n_components):
    standardized = StandardScaler().fit_transform(samples)
    pca = PCA(n_components=n_components, random_state=0)
    projection = pca.fit_transform(standardized)
    return projection, pca


def compute_condition_mean_signals(paired_dataset, centered=False, smooth_win=1):
    color_data = paired_dataset.color
    gray_data = paired_dataset.gray
    if centered:
        pair_mean = 0.5 * (color_data + gray_data)
        color_data = color_data - pair_mean
        gray_data = gray_data - pair_mean

    n_categories = len(paired_dataset.category_names)
    color_means = []
    gray_means = []
    for category_idx in range(n_categories):
        mask = paired_dataset.category_ids == category_idx
        color_mean = np.mean(color_data[mask], axis=0)
        gray_mean = np.mean(gray_data[mask], axis=0)
        color_means.append(smooth_time_axis(color_mean, smooth_win))
        gray_means.append(smooth_time_axis(gray_mean, smooth_win))
    return np.stack(color_means, axis=0), np.stack(gray_means, axis=0)


def project_condition_trajectories(color_means, gray_means, n_components):
    n_categories, _, n_time = color_means.shape
    stacked = []
    for category_idx in range(n_categories):
        stacked.append(color_means[category_idx].T)
        stacked.append(gray_means[category_idx].T)
    stacked = np.concatenate(stacked, axis=0)
    projection, pca_model = fit_pca_model(stacked, n_components=n_components)

    color_proj = np.zeros((n_categories, n_time, n_components), dtype=float)
    gray_proj = np.zeros((n_categories, n_time, n_components), dtype=float)
    cursor = 0
    for category_idx in range(n_categories):
        color_proj[category_idx] = projection[cursor:cursor + n_time]
        cursor += n_time
        gray_proj[category_idx] = projection[cursor:cursor + n_time]
        cursor += n_time
    return color_proj, gray_proj, pca_model


def _trajectory_marker_indices(n_time, point_step):
    if point_step is None or int(point_step) <= 0:
        return np.zeros(0, dtype=int)
    return np.arange(0, n_time, int(point_step), dtype=int)


def select_time_range(color_means, gray_means, time_vector, start_ms=None, end_ms=None):
    time_vector = np.asarray(time_vector, dtype=float)
    start_value = float(time_vector[0]) if start_ms is None else float(start_ms)
    end_value = float(time_vector[-1]) if end_ms is None else float(end_ms)
    if start_value > end_value:
        raise ValueError('--plot-start-ms must be <= --plot-end-ms.')

    mask = (time_vector >= start_value) & (time_vector <= end_value)
    if not np.any(mask):
        raise ValueError(
            f'No time points fall inside the selected trajectory window [{start_value}, {end_value}] ms.'
        )

    return color_means[:, :, mask], gray_means[:, :, mask], time_vector[mask], start_value, end_value


def plot_2d_trajectories(ax, color_proj, gray_proj, paired_dataset, time_vector, pca_model, title, show_start_end=True, point_step=0):
    marker_indices = _trajectory_marker_indices(color_proj.shape[1], point_step)
    for category_idx, category_name in enumerate(paired_dataset.category_names):
        marker = MARKERS[category_idx % len(MARKERS)]
        ax.plot(color_proj[category_idx, :, 0], color_proj[category_idx, :, 1], color=COLOR_MAP['color'], linewidth=2.0, alpha=0.9)
        ax.plot(gray_proj[category_idx, :, 0], gray_proj[category_idx, :, 1], color=COLOR_MAP['gray'], linewidth=2.0, linestyle='--', alpha=0.9)
        if marker_indices.size:
            ax.scatter(color_proj[category_idx, marker_indices, 0], color_proj[category_idx, marker_indices, 1], color=COLOR_MAP['color'], marker='.', s=18, alpha=0.7)
            ax.scatter(gray_proj[category_idx, marker_indices, 0], gray_proj[category_idx, marker_indices, 1], color=COLOR_MAP['gray'], marker='.', s=18, alpha=0.7)
        if show_start_end:
            ax.scatter(color_proj[category_idx, 0, 0], color_proj[category_idx, 0, 1], color=COLOR_MAP['color'], marker=marker, s=40, label=f'{category_name} color')
            ax.scatter(gray_proj[category_idx, 0, 0], gray_proj[category_idx, 0, 1], facecolors='none', edgecolors=COLOR_MAP['gray'], marker=marker, s=40, linewidths=1.2, label=f'{category_name} gray')
            ax.scatter(color_proj[category_idx, -1, 0], color_proj[category_idx, -1, 1], color=COLOR_MAP['color'], marker='>', s=36)
            ax.scatter(gray_proj[category_idx, -1, 0], gray_proj[category_idx, -1, 1], color=COLOR_MAP['gray'], marker='>', s=36)
        else:
            ax.scatter(color_proj[category_idx, 0, 0], color_proj[category_idx, 0, 1], color=COLOR_MAP['color'], marker=marker, s=40, alpha=0.001, label=f'{category_name} color')
            ax.scatter(gray_proj[category_idx, 0, 0], gray_proj[category_idx, 0, 1], facecolors='none', edgecolors=COLOR_MAP['gray'], marker=marker, s=40, linewidths=1.2, alpha=0.001, label=f'{category_name} gray')

    ax.set_title(title)
    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% var)')
    ax.grid(True, linestyle='--', alpha=0.35)


def plot_3d_trajectories(ax, color_proj, gray_proj, paired_dataset, pca_model, title, show_start_end=True, point_step=0):
    marker_indices = _trajectory_marker_indices(color_proj.shape[1], point_step)
    for category_idx, category_name in enumerate(paired_dataset.category_names):
        marker = MARKERS[category_idx % len(MARKERS)]
        ax.plot(color_proj[category_idx, :, 0], color_proj[category_idx, :, 1], color_proj[category_idx, :, 2], color=COLOR_MAP['color'], linewidth=2.0, alpha=0.9)
        ax.plot(gray_proj[category_idx, :, 0], gray_proj[category_idx, :, 1], gray_proj[category_idx, :, 2], color=COLOR_MAP['gray'], linewidth=2.0, linestyle='--', alpha=0.9)
        if marker_indices.size:
            ax.scatter(color_proj[category_idx, marker_indices, 0], color_proj[category_idx, marker_indices, 1], color_proj[category_idx, marker_indices, 2], color=COLOR_MAP['color'], marker='.', s=12, alpha=0.7)
            ax.scatter(gray_proj[category_idx, marker_indices, 0], gray_proj[category_idx, marker_indices, 1], gray_proj[category_idx, marker_indices, 2], color=COLOR_MAP['gray'], marker='.', s=12, alpha=0.7)
        if show_start_end:
            ax.scatter(color_proj[category_idx, 0, 0], color_proj[category_idx, 0, 1], color_proj[category_idx, 0, 2], color=COLOR_MAP['color'], marker=marker, s=30)
            ax.scatter(gray_proj[category_idx, 0, 0], gray_proj[category_idx, 0, 1], gray_proj[category_idx, 0, 2], facecolors='none', edgecolors=COLOR_MAP['gray'], marker=marker, s=30)
            ax.scatter(color_proj[category_idx, -1, 0], color_proj[category_idx, -1, 1], color_proj[category_idx, -1, 2], color=COLOR_MAP['color'], marker='>', s=28)
            ax.scatter(gray_proj[category_idx, -1, 0], gray_proj[category_idx, -1, 1], gray_proj[category_idx, -1, 2], color=COLOR_MAP['gray'], marker='>', s=28)

    ax.set_title(title)
    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% var)')
    ax.set_zlabel(f'PC3 ({pca_model.explained_variance_ratio_[2] * 100:.1f}% var)')


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


def _plotly_hover_text(category_name, condition_name, time_vector):
    return [f'{category_name} {condition_name}<br>time={time_ms:.1f} ms' for time_ms in time_vector]


def save_interactive_3d_html(
    save_path,
    raw_color_3d,
    raw_gray_3d,
    centered_color_3d,
    centered_gray_3d,
    paired_dataset,
    raw_pca_3d,
    centered_pca_3d,
    time_vector,
    show_start_end,
    point_step,
    title,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            'plotly is required for --interactive-3d. Install it with `pip install plotly` or rerun dependency setup.'
        ) from exc

    marker_indices = _trajectory_marker_indices(raw_color_3d.shape[1], point_step)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Raw condition-mean trajectories', 'Centered condition-mean trajectories'),
        horizontal_spacing=0.04,
    )

    panels = [
        ('scene', 1, raw_color_3d, raw_gray_3d, raw_pca_3d),
        ('scene2', 2, centered_color_3d, centered_gray_3d, centered_pca_3d),
    ]

    for scene_name, col_idx, color_proj, gray_proj, pca_model in panels:
        for category_idx, category_name in enumerate(paired_dataset.category_names):
            marker_symbol = PLOTLY_MARKERS[category_idx % len(PLOTLY_MARKERS)]
            open_marker_symbol = PLOTLY_OPEN_MARKERS[marker_symbol]
            color_hover = _plotly_hover_text(category_name, 'color', time_vector)
            gray_hover = _plotly_hover_text(category_name, 'gray', time_vector)

            fig.add_trace(
                go.Scatter3d(
                    x=color_proj[category_idx, :, 0],
                    y=color_proj[category_idx, :, 1],
                    z=color_proj[category_idx, :, 2],
                    mode='lines',
                    line={'color': COLOR_MAP['color'], 'width': 5},
                    name=f'{category_name} color',
                    legendgroup=f'{category_name}-color',
                    showlegend=(col_idx == 1),
                    hovertext=color_hover,
                    hovertemplate='%{hovertext}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>',
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=gray_proj[category_idx, :, 0],
                    y=gray_proj[category_idx, :, 1],
                    z=gray_proj[category_idx, :, 2],
                    mode='lines',
                    line={'color': COLOR_MAP['gray'], 'width': 5, 'dash': 'dash'},
                    name=f'{category_name} gray',
                    legendgroup=f'{category_name}-gray',
                    showlegend=(col_idx == 1),
                    hovertext=gray_hover,
                    hovertemplate='%{hovertext}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>',
                ),
                row=1,
                col=col_idx,
            )

            if marker_indices.size:
                fig.add_trace(
                    go.Scatter3d(
                        x=color_proj[category_idx, marker_indices, 0],
                        y=color_proj[category_idx, marker_indices, 1],
                        z=color_proj[category_idx, marker_indices, 2],
                        mode='markers',
                        marker={'size': 3, 'color': COLOR_MAP['color'], 'symbol': 'circle'},
                        legendgroup=f'{category_name}-color',
                        showlegend=False,
                        hovertext=[color_hover[idx] for idx in marker_indices],
                        hovertemplate='%{hovertext}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>',
                    ),
                    row=1,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=gray_proj[category_idx, marker_indices, 0],
                        y=gray_proj[category_idx, marker_indices, 1],
                        z=gray_proj[category_idx, marker_indices, 2],
                        mode='markers',
                        marker={'size': 3, 'color': COLOR_MAP['gray'], 'symbol': 'circle-open'},
                        legendgroup=f'{category_name}-gray',
                        showlegend=False,
                        hovertext=[gray_hover[idx] for idx in marker_indices],
                        hovertemplate='%{hovertext}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>',
                    ),
                    row=1,
                    col=col_idx,
                )

            if show_start_end:
                fig.add_trace(
                    go.Scatter3d(
                        x=[color_proj[category_idx, 0, 0]],
                        y=[color_proj[category_idx, 0, 1]],
                        z=[color_proj[category_idx, 0, 2]],
                        mode='markers',
                        marker={'size': 6, 'color': COLOR_MAP['color'], 'symbol': marker_symbol},
                        legendgroup=f'{category_name}-color',
                        showlegend=False,
                        hovertext=[color_hover[0]],
                        hovertemplate='start<br>%{hovertext}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>',
                    ),
                    row=1,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=[gray_proj[category_idx, 0, 0]],
                        y=[gray_proj[category_idx, 0, 1]],
                        z=[gray_proj[category_idx, 0, 2]],
                        mode='markers',
                        marker={'size': 6, 'color': COLOR_MAP['gray'], 'symbol': open_marker_symbol},
                        legendgroup=f'{category_name}-gray',
                        showlegend=False,
                        hovertext=[gray_hover[0]],
                        hovertemplate='start<br>%{hovertext}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>',
                    ),
                    row=1,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=[color_proj[category_idx, -1, 0]],
                        y=[color_proj[category_idx, -1, 1]],
                        z=[color_proj[category_idx, -1, 2]],
                        mode='markers',
                        marker={'size': 6, 'color': COLOR_MAP['color'], 'symbol': 'diamond'},
                        legendgroup=f'{category_name}-color',
                        showlegend=False,
                        hovertext=[color_hover[-1]],
                        hovertemplate='end<br>%{hovertext}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>',
                    ),
                    row=1,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=[gray_proj[category_idx, -1, 0]],
                        y=[gray_proj[category_idx, -1, 1]],
                        z=[gray_proj[category_idx, -1, 2]],
                        mode='markers',
                        marker={'size': 6, 'color': COLOR_MAP['gray'], 'symbol': 'diamond-open'},
                        legendgroup=f'{category_name}-gray',
                        showlegend=False,
                        hovertext=[gray_hover[-1]],
                        hovertemplate='end<br>%{hovertext}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>',
                    ),
                    row=1,
                    col=col_idx,
                )

        fig.update_layout(
            {
                scene_name: {
                    'xaxis_title': f'PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% var)',
                    'yaxis_title': f'PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% var)',
                    'zaxis_title': f'PC3 ({pca_model.explained_variance_ratio_[2] * 100:.1f}% var)',
                }
            }
        )

    fig.update_layout(
        title=title,
        width=1500,
        height=700,
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': -0.08, 'xanchor': 'center', 'x': 0.5},
        margin={'l': 20, 'r': 20, 't': 70, 'b': 70},
    )
    fig.write_html(str(save_path), include_plotlyjs=True)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.point_step < 0:
        raise ValueError('--point-step must be >= 0.')
    config = normalize_common_args(args)
    roi_bundle, paired_dataset = load_preprocessed_paired_dataset(config)
    selected_category_indices = resolve_selected_category_indices(args.selected_categories, config.category_names)
    paired_dataset = select_paired_dataset_categories(paired_dataset, selected_category_indices)
    show_start_end = not args.hide_start_end

    raw_color_means, raw_gray_means = compute_condition_mean_signals(paired_dataset, centered=False, smooth_win=args.smooth_win)
    centered_color_means, centered_gray_means = compute_condition_mean_signals(paired_dataset, centered=True, smooth_win=args.smooth_win)
    raw_color_means, raw_gray_means, plot_time_vector, plot_start_ms, plot_end_ms = select_time_range(
        raw_color_means,
        raw_gray_means,
        roi_bundle.time_vector,
        start_ms=args.plot_start_ms,
        end_ms=args.plot_end_ms,
    )
    centered_color_means, centered_gray_means, _, _, _ = select_time_range(
        centered_color_means,
        centered_gray_means,
        roi_bundle.time_vector,
        start_ms=args.plot_start_ms,
        end_ms=args.plot_end_ms,
    )

    raw_color_2d, raw_gray_2d, raw_pca_2d = project_condition_trajectories(raw_color_means, raw_gray_means, n_components=2)
    centered_color_2d, centered_gray_2d, centered_pca_2d = project_condition_trajectories(centered_color_means, centered_gray_means, n_components=2)
    raw_color_3d, raw_gray_3d, raw_pca_3d = project_condition_trajectories(raw_color_means, raw_gray_means, n_components=3)
    centered_color_3d, centered_gray_3d, centered_pca_3d = project_condition_trajectories(centered_color_means, centered_gray_means, n_components=3)

    out_dir = build_output_dir('pca_condition_trajectories', config)
    fig_2d_path = out_dir / 'condition_pca_trajectories_2d.png'
    fig_3d_path = out_dir / 'condition_pca_trajectories_3d.png'
    interactive_fig_3d_path = out_dir / 'condition_pca_trajectories_3d_interactive.html'
    npz_path = out_dir / 'condition_pca_trajectories.npz'

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_2d_trajectories(axes[0], raw_color_2d, raw_gray_2d, paired_dataset, plot_time_vector, raw_pca_2d, 'Raw condition-mean trajectories', show_start_end=show_start_end, point_step=args.point_step)
    plot_2d_trajectories(axes[1], centered_color_2d, centered_gray_2d, paired_dataset, plot_time_vector, centered_pca_2d, 'Centered condition-mean trajectories', show_start_end=show_start_end, point_step=args.point_step)
    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = dedupe_legend(handles, labels)
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f'Task1 condition PCA trajectories (2D) | {config.subject} | {config.feature_kind} | {config.roi_name} | '
        f'smooth={args.smooth_win} | {plot_start_ms:.0f}-{plot_end_ms:.0f} ms',
        fontsize=13,
    )
    plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.94])
    plt.savefig(fig_2d_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_3d_trajectories(ax1, raw_color_3d, raw_gray_3d, paired_dataset, raw_pca_3d, 'Raw condition-mean trajectories', show_start_end=show_start_end, point_step=args.point_step)
    plot_3d_trajectories(ax2, centered_color_3d, centered_gray_3d, paired_dataset, centered_pca_3d, 'Centered condition-mean trajectories', show_start_end=show_start_end, point_step=args.point_step)
    fig.suptitle(
        f'Task1 condition PCA trajectories (3D) | {config.subject} | {config.feature_kind} | {config.roi_name} | '
        f'smooth={args.smooth_win} | {plot_start_ms:.0f}-{plot_end_ms:.0f} ms',
        fontsize=13,
    )
    plt.tight_layout(rect=[0.0, 0.02, 1.0, 0.94])
    plt.savefig(fig_3d_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if args.interactive_3d:
        save_interactive_3d_html(
            interactive_fig_3d_path,
            raw_color_3d,
            raw_gray_3d,
            centered_color_3d,
            centered_gray_3d,
            paired_dataset,
            raw_pca_3d,
            centered_pca_3d,
            plot_time_vector,
            show_start_end=show_start_end,
            point_step=args.point_step,
            title=(
                f'Task1 condition PCA trajectories (Interactive 3D) | {config.subject} | {config.feature_kind} | '
                f'{config.roi_name} | smooth={args.smooth_win} | {plot_start_ms:.0f}-{plot_end_ms:.0f} ms'
            ),
        )

    np.savez(
        npz_path,
        subject=np.array(config.subject),
        feature_kind=np.array(config.feature_kind),
        roi_name=np.array(config.roi_name),
        grouped_data_mat=np.array(str(config.grouped_data_mat)),
        roi_path=np.array(str(roi_bundle.roi_path)),
        task_field=np.array(roi_bundle.task_field),
        category_names=np.asarray(paired_dataset.category_names, dtype=object),
        selected_category_indices=np.asarray(selected_category_indices, dtype=int),
        smooth_win=np.array(args.smooth_win),
        point_step=np.array(args.point_step),
        show_start_end=np.array(show_start_end),
        interactive_3d=np.array(bool(args.interactive_3d)),
        plot_start_ms=np.array(plot_start_ms),
        plot_end_ms=np.array(plot_end_ms),
        time_ms=np.asarray(plot_time_vector, dtype=float),
        raw_explained_variance_ratio_2d=raw_pca_2d.explained_variance_ratio_,
        centered_explained_variance_ratio_2d=centered_pca_2d.explained_variance_ratio_,
        raw_explained_variance_ratio_3d=raw_pca_3d.explained_variance_ratio_,
        centered_explained_variance_ratio_3d=centered_pca_3d.explained_variance_ratio_,
        raw_color_2d=raw_color_2d,
        raw_gray_2d=raw_gray_2d,
        centered_color_2d=centered_color_2d,
        centered_gray_2d=centered_gray_2d,
        raw_color_3d=raw_color_3d,
        raw_gray_3d=raw_gray_3d,
        centered_color_3d=centered_color_3d,
        centered_gray_3d=centered_gray_3d,
    )

    print(
        {
            'figure_2d_path': str(fig_2d_path),
            'figure_3d_path': str(fig_3d_path),
            'interactive_3d_path': str(interactive_fig_3d_path) if args.interactive_3d else None,
            'summary_path': str(npz_path),
            'selected_category_names': paired_dataset.category_names,
            'smooth_win': int(args.smooth_win),
            'point_step': int(args.point_step),
            'show_start_end': bool(show_start_end),
            'plot_window_ms': [float(plot_start_ms), float(plot_end_ms)],
        }
    )


if __name__ == '__main__':
    main()