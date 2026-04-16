import os
import glob
import math
import time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label

from newanalyse_paths import get_within_decoding_task_dir, project_root

SUBJECT = 'test002'
BASE_PATH = str(project_root())
BATCH_NAME = 'batch_within_decoding'

FS = 500
T_START = -100
T_END = 1000
N_POINTS = 550
TIMES = np.linspace(T_START, T_END, N_POINTS)
DECODING_STEP = 5
PLOT_TIMES = TIMES[::DECODING_STEP]

# Exclude ROIs
EXCLUDE_ROIS = ['unknown', 'preceneus', 'precuneus', 'n_a', 'calcarine']

TASKS = [
    {
        'id': 'task3_1vs4_self',
        'title': 'Task 3 Pure Color Self-Decoding: Condition 1 vs Condition 4',
        'description': 'Train and test within Task 3 using condition 1 vs condition 4.',
        'folder': 'task3_1vs4_self'
    },
    {
        'id': 'task3_2vs3_self',
        'title': 'Task 3 Pure Color Self-Decoding: Condition 2 vs Condition 3',
        'description': 'Train and test within Task 3 using condition 2 vs condition 3.',
        'folder': 'task3_2vs3_self'
    },
    {
        'id': 'task1_color_vs_gray_pair_cv',
        'title': 'Task 1 Color vs Gray Pair Holdout Decoding',
        'description': 'Four-fold pair holdout: three odd-even pairs for train, one pair for test.',
        'folder': 'task1_color_vs_gray_pair_cv'
    },
    {
        'id': 'task1_color_vs_gray_per_category',
        'title': 'Task 1 Color vs Gray Per-Category Decoding',
        'description': 'Decode color vs gray within each of 4 categories and average AUC across categories.',
        'folder': 'task1_color_vs_gray_per_category'
    },
    {
        'id': 'task2_gray_memory_color_cross',
        'title': 'Task 2 Gray Fruit Memory-Color Decoding',
        'description': 'Cross-object decoding on gray fruits for memory color red vs green with four combinations.',
        'folder': 'task2_gray_memory_color_cross'
    },
    {
        'id': 'task2_true_vs_false',
        'title': 'Task 2 True vs False Fruit Color Decoding',
        'description': 'Binary decoding between true-color fruits and false-color fruits.',
        'folder': 'task2_true_vs_false'
    }
]

def make_logger(log_path):
    def _log(msg):
        text = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(text)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
    return _log

def is_excluded(roi_name):
    roi_lower = roi_name.lower()
    for ex in EXCLUDE_ROIS:
        if ex in roi_lower:
            return True
    return False

def cluster_permutation_significance(mean_auc, perm_dist):
    threshold_95 = np.percentile(perm_dist, 95, axis=0)
    # Ignore before 20ms
    valid_mask = PLOT_TIMES[:len(mean_auc)] >= 20
    binary_map = (mean_auc > threshold_95) & valid_mask
    clusters, n_clusters = label(binary_map.astype(int))
    cluster_masses = []
    for ci in range(1, n_clusters + 1):
        idx = clusters == ci
        cluster_masses.append(np.sum(mean_auc[idx] - threshold_95[idx]))

    null_cluster_masses = []
    for pi in range(perm_dist.shape[0]):
        perm_curve = perm_dist[pi]
        p_binary = (perm_curve > threshold_95) & valid_mask
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

def plot_single_roi_result(roi_name, mean_auc, sem_auc, threshold_95, sig_indices, figure_title, save_path):
    plot_times = PLOT_TIMES[:len(mean_auc)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(plot_times, mean_auc - sem_auc, mean_auc + sem_auc, color='#1f77b4', alpha=0.25)
    ax.plot(plot_times, mean_auc, color='#1f77b4', linewidth=1.8, label='Mean ROC AUC')
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
    save_path = os.path.join(batch_root, 'Fig_Task3_1vs4_and_2vs3_SignificantUnionROIs_replot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger(f'Dual-task overlay figure saved: {save_path}')

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

    roi_plot_dir = os.path.join(output_dir, 'roi_curves_replot')
    os.makedirs(roi_plot_dir, exist_ok=True)

    for fpath in result_files:
        roi_name = os.path.basename(fpath).replace('_results.npz', '')
        
        if is_excluded(roi_name):
            logger(f'Excluded ROI: {roi_name}')
            continue
            
        data = np.load(fpath)
        mean_auc = data['mean_auc']
        sem_auc = data['sem_auc']
        perm_dist = data['perm_dist']
        
        threshold_95, sig_indices = cluster_permutation_significance(mean_auc, perm_dist)
        latencies = compute_latency_points(mean_auc, sig_indices)
        
        all_results[roi_name] = (mean_auc, sem_auc, threshold_95, sig_indices)
        if np.any(sig_indices):
            sig_results[roi_name] = (mean_auc, sem_auc, threshold_95, sig_indices)
            latency_earliest[roi_name] = float(latencies['earliest'])
            latency_half[roi_name] = float(latencies['half_height'])
            latency_peak[roi_name] = float(latencies['peak'])
            
        plot_single_roi_result(
            roi_name=roi_name,
            mean_auc=mean_auc,
            sem_auc=sem_auc,
            threshold_95=threshold_95,
            sig_indices=sig_indices,
            figure_title=f'{task["title"]} | {task["description"]}',
            save_path=os.path.join(roi_plot_dir, f'{roi_name}_curve_replot.png')
        )

    plot_grid_figures(
        results_dict=all_results,
        title_text=f'{task["title"]} | {task["description"]}',
        save_path=os.path.join(output_dir, 'Fig_All_ROIs_Overview_replot.png')
    )
    if sig_results:
        plot_grid_figures(
            results_dict=sig_results,
            title_text=f'{task["title"]} | Significant ROIs',
            save_path=os.path.join(output_dir, 'Fig_Significant_ROIs_Only_replot.png')
        )
        plot_latency_bar_scatter(
            latency_dict=latency_earliest,
            title_text=f'{task["title"]} | Earliest Significant Latency',
            y_label='Earliest Significant Latency (ms)',
            save_path=os.path.join(output_dir, 'Fig_Latency_Earliest_replot.png')
        )
        plot_latency_bar_scatter(
            latency_dict=latency_half,
            title_text=f'{task["title"]} | First-Peak Half-Height Latency',
            y_label='First-Peak Half-Height Latency (ms)',
            save_path=os.path.join(output_dir, 'Fig_Latency_FirstPeakHalfHeight_replot.png')
        )
        plot_latency_bar_scatter(
            latency_dict=latency_peak,
            title_text=f'{task["title"]} | Peak Latency',
            y_label='Peak Latency (ms)',
            save_path=os.path.join(output_dir, 'Fig_Latency_Peak_replot.png')
        )
        logger(f'Significant ROI count: {len(sig_results)}')
    else:
        logger(f'No significant ROI found for task: {task["id"]}')

    return {'all': all_results, 'significant': sig_results, 'latency_earliest': latency_earliest}

def main():
    report_root = os.path.join(BASE_PATH, 'result', 'reports', 'replot_within_decoding', SUBJECT)
    os.makedirs(report_root, exist_ok=True)
    log_path = os.path.join(report_root, f'replot_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logger = make_logger(log_path)

    logger(f'Start re-plotting batch decoding for {SUBJECT}')
    logger(f'Excluding ROIs containing: {EXCLUDE_ROIS}')
    
    # --- 定义需要循环的特征类型 ---
    FEATURES = ['erp', 'highgamma', 'lowgamma']
    
    for feature in FEATURES:
        logger(f'\n\n{"#"*20} Processing Feature: {feature} {"#"*20}')
        
        # 根据特征类型更新基础路径
        new_base_dir = os.path.join(report_root, feature, 'perm1000', 'NEW')
        os.makedirs(new_base_dir, exist_ok=True)
        md_file_path = os.path.join(new_base_dir, 'significant_rois_summary.md')
        
        task_outputs = {}
        for task in TASKS:
            # 输入路径随 feature 变化
            input_dir = str(
                get_within_decoding_task_dir(
                    BASE_PATH,
                    task['id'],
                    feature,
                    SUBJECT,
                    'perm1000',
                    variant='with_sti',
                    batch_name=BATCH_NAME,
                )
            )
            cache_dir = os.path.join(input_dir, 'computed_results')
            
            # 输出路径到 NEW 文件夹
            output_dir = os.path.join(new_base_dir, task['folder'])
            os.makedirs(output_dir, exist_ok=True) # 确保子任务文件夹存在

            if not os.path.exists(cache_dir):
                logger(f'[{feature}] Cache directory not found: {cache_dir}')
                continue
                
            logger(f'===== {task["id"]} ({feature.upper()} REPLOT) =====')
            logger(task['title'])
            
            # 生成图片和统计结果
            summary = generate_summary_figures(task, cache_dir, output_dir, logger)
            task_outputs[task['id']] = summary

        # --- 针对当前 feature 写入 Markdown 总结 ---
        if task_outputs:
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Significant ROIs Summary for {SUBJECT} ({feature})\n\n")
                for task in TASKS:
                    task_id = task['id']
                    if task_id not in task_outputs:
                        continue
                    
                    summary = task_outputs[task_id]
                    sig_rois = summary.get('significant', {})
                    latencies = summary.get('latency_earliest', {})
                    
                    f.write(f"## {task['title']}\n")
                    if not sig_rois:
                        f.write("No significant ROIs found.\n\n")
                    else:
                        f.write("| ROI Name | Earliest Significant Latency (ms) |\n")
                        f.write("|----------|-----------------------------------|\n")
                        sorted_rois = sorted(latencies.items(), key=lambda x: x[1])
                        for roi, lat in sorted_rois:
                            f.write(f"| {roi} | {lat:.0f} |\n")
                        f.write("\n")
            
            logger(f'[{feature}] Summary markdown saved to {md_file_path}')

            # --- 针对当前 feature 执行 dual overlay 绘图 ---
            # 如果这个函数支持传递 feature 相关的路径，请确保参数正确
            try:
                plot_task3_dual_overlay(task_outputs, new_base_dir, logger)
            except Exception as e:
                logger(f"Error plotting dual overlay for {feature}: {e}")

    logger('\nAll re-plotting tasks completed')

if __name__ == '__main__':
    _script_start_time = time.time()
    try:
        main()
    finally:
        print(f'Total runtime: {time.time() - _script_start_time:.2f} s')