import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import ranksums
from pymatreader import read_mat
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import re
import ast
import warnings
import copy

# 忽略绘图和计算过程中的警告
warnings.filterwarnings('ignore')
# 设置 matplotlib 后端为 Agg，防止没有 GUI 的服务器环境报错
import matplotlib
matplotlib.use('Agg')

# 基础路径配置
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
analyse_dir = os.path.join(pipeline_dir, 'analyse_0617')
feature_dir = os.path.join(analyse_dir, 'feature')
doc_dir = os.path.join(analyse_dir, 'doc')
result_dir = os.path.join(analyse_dir, 'result')

subjects = ['test001', 'test002', 'test003']
categories = ['Face', 'Object', 'Body', 'Place', 'Merged_All']
cond_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

# 确保文件夹存在
os.makedirs(doc_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# ----------------- ROI 与解剖辅助判定 -----------------
def get_roi_category(label):
    """
    根据解剖标签判断其所属的 ROI 大类。
    """
    if not isinstance(label, str) or label.strip() == '':
        return None
    label_lower = label.lower()
    if any(kw in label_lower for kw in ['calcarine', 'occipital_inf', 'occipital_mid', 'lingual']):
        return '枕叶'
    elif any(kw in label_lower for kw in ['fusiform', 'temporal_inf']):
        return '颞叶后/下部'
    elif any(kw in label_lower for kw in ['temporal_mid', 'temporal_pole']):
        return '颞叶前/上部'
    return None

def is_neighbor_valid_label(label):
    """
    判断扩展电极的解剖标签是否为 unknown, N/A 或旁海马区
    """
    if pd.isna(label) or not isinstance(label, str) or label.strip() == '':
        return True
    label_lower = label.lower().strip()
    if 'unknown' in label_lower or 'n/a' in label_lower or label_lower == 'nan':
        return True
    if 'parahippocampal' in label_lower or 'parahippocampus' in label_lower:
        return True
    return False

def parse_channel_name(ch_name):
    """
    解析 'G11' -> ('G', 11)
    """
    if not isinstance(ch_name, str):
        return None, None
    match = re.match(r'^([a-zA-Z]+)(\d+)$', ch_name.strip())
    if match:
        prefix = match.group(1).upper()
        num = int(match.group(2))
        return prefix, num
    return None, None

def find_neighbors(target_ch, all_channels):
    """
    寻找同一 shaft 且序号相差为 +/- 1 的邻近通道
    """
    target_prefix, target_num = parse_channel_name(target_ch)
    if target_prefix is None:
        return []
    neighbors = []
    for ch in all_channels:
        if ch == target_ch:
            continue
        pref, num = parse_channel_name(ch)
        if pref == target_prefix and abs(num - target_num) == 1:
            neighbors.append(ch)
    return neighbors

# ----------------- 统计筛选核心逻辑 -----------------
def has_continuous_sig(sig_bool, consec_pts):
    count = 0
    for val in sig_bool:
        if val:
            count += 1
            if count >= consec_pts:
                return True
        else:
            count = 0
    return False

def check_channel_strategies(c_merged, g_merged, c_singles, g_singles, time_ms):
    """
    检测输入通道在 4 种统计策略下的显著情况
    """
    dt = time_ms[1] - time_ms[0]
    consec_pts = int(np.ceil(50.0 / dt))
    
    idx_100_400 = np.where((time_ms >= 100) & (time_ms <= 400))[0]
    idx_50_400 = np.where((time_ms >= 50) & (time_ms <= 400))[0]
    
    # 策略1: 混合类别 100-400ms 平均值显著
    c_merged_100_400 = np.mean(c_merged[:, idx_100_400], axis=1)
    g_merged_100_400 = np.mean(g_merged[:, idx_100_400], axis=1)
    _, p_s1 = ranksums(c_merged_100_400, g_merged_100_400)
    s1 = p_s1 < 0.05
    
    # 策略2: 混合类别在 50-400ms 点对点连续显著 >= 50ms
    sig_m = []
    for t_idx in idx_50_400:
        _, p_t = ranksums(c_merged[:, t_idx], g_merged[:, t_idx])
        sig_m.append(p_t < 0.05)
    s2 = has_continuous_sig(sig_m, consec_pts)
    
    # 策略3: 至少一个单一类别 100-400ms 平均值显著
    s3 = False
    for c_s, g_s in zip(c_singles, g_singles):
        c_s_100_400 = np.mean(c_s[:, idx_100_400], axis=1)
        g_s_100_400 = np.mean(g_s[:, idx_100_400], axis=1)
        _, p_s = ranksums(c_s_100_400, g_s_100_400)
        if p_s < 0.05:
            s3 = True
            break
            
    # 策略4: 至少一个单一类别在 50-400ms 下点对点连续显著 >= 50ms
    s4 = False
    for c_s, g_s in zip(c_singles, g_singles):
        sig_s = []
        for t_idx in idx_50_400:
            _, p_t = ranksums(c_s[:, t_idx], g_s[:, t_idx])
            sig_s.append(p_t < 0.05)
        if has_continuous_sig(sig_s, consec_pts):
            s4 = True
            break
            
    return s1, s2, s3, s4

# ----------------- 数据裁剪另存 -----------------
def crop_erp_mat(mat, target_labels):
    new_mat = copy.deepcopy(mat)
    original_labels = list(mat['epoch']['ch']['labels'])
    indices = [original_labels.index(label) for label in target_labels if label in original_labels]
    
    # 裁剪数据
    new_mat['epoch']['data'] = mat['epoch']['data'][:, :, indices, :]
    
    # 裁剪 ch 属性
    n_ch_orig = len(original_labels)
    for k, v in mat['epoch']['ch'].items():
        if isinstance(v, list) and len(v) == n_ch_orig:
            new_mat['epoch']['ch'][k] = [v[idx] for idx in indices]
        elif isinstance(v, np.ndarray) and len(v) == n_ch_orig:
            new_mat['epoch']['ch'][k] = v[indices]
            
    return new_mat

def crop_hg_mat(mat, target_labels):
    new_mat = copy.deepcopy(mat)
    original_labels = list(mat['epoch']['ch']['labels'])
    indices = [original_labels.index(label) for label in target_labels if label in original_labels]
    
    # 裁剪 data_cell 里的每一个条件矩阵
    new_mat['epoch']['data_cell'] = [
        cell[:, indices, :] for cell in mat['epoch']['data_cell']
    ]
    
    # 裁剪 ch 属性
    n_ch_orig = len(original_labels)
    for k, v in mat['epoch']['ch'].items():
        if isinstance(v, list) and len(v) == n_ch_orig:
            new_mat['epoch']['ch'][k] = [v[idx] for idx in indices]
        elif isinstance(v, np.ndarray) and len(v) == n_ch_orig:
            new_mat['epoch']['ch'][k] = v[indices]
            
    return new_mat

# ----------------- ERP & HG 信号差异图绘制 -----------------
def plot_signal_erp_or_hg(subject, elec, ch_idx, data, time_ms, out_path, is_hg=False):
    """
    绘制指定电极的时程信号图和 100-400ms 幅值条形散点图
    """
    fig, axes = plt.subplots(5, 2, figsize=(12, 20), gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(f"Subject: {subject} | Electrode: {elec} - {'High Gamma' if is_hg else 'ERP'}", fontsize=16, fontweight='bold', y=0.98)
    
    idx_100_400 = np.where((time_ms >= 100) & (time_ms <= 400))[0]
    
    for i, cat_name in enumerate(categories):
        ax_time = axes[i, 0]
        ax_bar = axes[i, 1]
        
        # 数据提取
        if is_hg:
            # data 为 data_cell
            if cat_name == 'Merged_All':
                c_data = np.concatenate([data[idx][:, ch_idx, :] for idx in [0, 2, 4, 6]], axis=0)
                g_data = np.concatenate([data[idx][:, ch_idx, :] for idx in [1, 3, 5, 7]], axis=0)
            else:
                c_idx, g_idx = cond_pairs[i]
                c_data = data[c_idx][:, ch_idx, :]
                g_data = data[g_idx][:, ch_idx, :]
        else:
            # data 为 4D array [Cond, Rep, Ch, Time]
            if cat_name == 'Merged_All':
                c_data = np.concatenate([data[idx, :, ch_idx, :] for idx in [0, 2, 4, 6]], axis=0)
                g_data = np.concatenate([data[idx, :, ch_idx, :] for idx in [1, 3, 5, 7]], axis=0)
            else:
                c_idx, g_idx = cond_pairs[i]
                c_data = data[c_idx, :, ch_idx, :]
                g_data = data[g_idx, :, ch_idx, :]
                
        # 过滤 NaN 的 trials
        c_data = c_data[~np.isnan(c_data).any(axis=1)]
        g_data = g_data[~np.isnan(g_data).any(axis=1)]
        
        c_mean = np.mean(c_data, axis=0)
        c_sem = np.std(c_data, axis=0) / np.sqrt(c_data.shape[0])
        g_mean = np.mean(g_data, axis=0)
        g_sem = np.std(g_data, axis=0) / np.sqrt(g_data.shape[0])
        
        # 1. 左侧时程图
        ax_time.plot(time_ms, c_mean, color='#d32f2f', lw=2.2, label='Color')
        ax_time.fill_between(time_ms, c_mean - c_sem, c_mean + c_sem, color='#d32f2f', alpha=0.15)
        
        ax_time.plot(time_ms, g_mean, color='#212121', lw=2.2, label='Gray')
        ax_time.fill_between(time_ms, g_mean - g_sem, g_mean + g_sem, color='#212121', alpha=0.15)
        
        ax_time.axvline(0, color='#9E9E9E', linestyle='--', alpha=0.6)
        
        # 点对点显著性标记
        ymin, ymax = ax_time.get_ylim()
        sig_y = ymin + (ymax - ymin) * 0.05
        for t_idx in range(len(time_ms)):
            stat, p = ranksums(c_data[:, t_idx], g_data[:, t_idx])
            if p < 0.05:
                color = 'yellow' if stat > 0 else 'cyan'
                ax_time.plot(time_ms[t_idx], sig_y, marker='s', color=color, markersize=3, alpha=0.7)
                
        ax_time.set_title(f"{cat_name} (Time Course)", fontsize=11, fontweight='bold')
        ax_time.set_xlabel("Time (ms)", fontsize=9.5)
        ax_time.set_ylabel("Amplitude (z-score)" if is_hg else "Amplitude (μV)", fontsize=9.5)
        ax_time.set_xlim([-200, 800])
        ax_time.grid(False)
        for spine in ax_time.spines.values():
            spine.set_visible(True)
            spine.set_color('#757575')
            
        if i == 0:
            ax_time.legend(loc='upper right', frameon=True, fontsize=8)
            
        # 2. 右侧 100-400ms 条形散点图
        c_vals = np.mean(c_data[:, idx_100_400], axis=1)
        g_vals = np.mean(g_data[:, idx_100_400], axis=1)
        
        bar_c_mean = np.mean(c_vals)
        bar_c_sem = np.std(c_vals) / np.sqrt(len(c_vals))
        bar_g_mean = np.mean(g_vals)
        bar_g_sem = np.std(g_vals) / np.sqrt(len(g_vals))
        
        ax_bar.bar([1], [bar_c_mean], yerr=[bar_c_sem], color='#d32f2f', alpha=0.7, capsize=5, width=0.4, error_kw={'elinewidth':1.5, 'capthick':1.5})
        ax_bar.bar([2], [bar_g_mean], yerr=[bar_g_sem], color='#212121', alpha=0.7, capsize=5, width=0.4, error_kw={'elinewidth':1.5, 'capthick':1.5})
        
        # 散点
        ax_bar.scatter(np.random.normal(1, 0.05, len(c_vals)), c_vals, color='darkred', alpha=0.2, s=8)
        ax_bar.scatter(np.random.normal(2, 0.05, len(g_vals)), g_vals, color='gray', alpha=0.2, s=8)
        
        stat_bar, p_val_bar = ranksums(c_vals, g_vals)
        
        ax_bar.set_xticks([1, 2])
        ax_bar.set_xticklabels(['Color', 'Gray'], fontsize=8.5)
        ax_bar.set_title(f"100-400ms\np={p_val_bar:.3f}", fontsize=9.5)
        ax_bar.grid(False)
        for spine in ax_bar.spines.values():
            spine.set_visible(True)
            spine.set_color('#757575')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

# ----------------- 绘制数量对比柱状图 -----------------
def plot_electrode_selection_comparison(data_df, title_prefix, out_path):
    if title_prefix.startswith("Group"):
        title_prefix = title_prefix.replace("Group (test001+test002+test003)", "Group (5 Subjects)")
        
    stats_data = {
        'erp_target': [0, 0, 0, 0],
        'hg_target': [0, 0, 0, 0]
    }
    
    for s_idx in range(1, 5):
        s_col = f'Strategy_{s_idx}'
        if s_col in data_df.columns:
            stats_data['erp_target'][s_idx-1] = data_df[(data_df['Signal'] == 'ERP') & (data_df[s_col] == True) & (data_df['In_Target'] == True)]['Electrode_ID'].nunique()
            stats_data['hg_target'][s_idx-1] = data_df[(data_df['Signal'] == 'HG') & (data_df[s_col] == True) & (data_df['In_Target'] == True)]['Electrode_ID'].nunique()
        
    x = np.arange(4)
    width = 0.45
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True, dpi=300)
    fig.suptitle(f"{title_prefix} - Target Area Electrode Selection Across Strategies", fontsize=14, fontweight='bold', y=0.98)
    
    strategies_labels = [
        "Strategy 1\n(Merged 100-400ms)",
        "Strategy 2\n(Merged Cont. 50ms)",
        "Strategy 3\n(Single 100-400ms)",
        "Strategy 4\n(Single Cont. 50ms)"
    ]
    
    # 子图 A: ERP Signals
    ax_erp = axes[0]
    rects1_t = ax_erp.bar(x, stats_data['erp_target'], width, label='In Target Area', color='#0072B2')
    ax_erp.set_title("A) ERP Signals (Target Area)", fontsize=12, fontweight='bold')
    ax_erp.set_xticks(x)
    ax_erp.set_xticklabels(strategies_labels, fontsize=9.5)
    ax_erp.set_ylabel("Number of Selected Electrodes", fontsize=11)
    ax_erp.grid(True, linestyle='--', alpha=0.3)
    ax_erp.set_facecolor('#fcfcfc')
    
    # 子图 B: HG Signals
    ax_hg = axes[1]
    rects2_t = ax_hg.bar(x, stats_data['hg_target'], width, label='In Target Area', color='#D55E00')
    ax_hg.set_title("B) High Gamma Signals (Target Area)", fontsize=12, fontweight='bold')
    ax_hg.set_xticks(x)
    ax_hg.set_xticklabels(strategies_labels, fontsize=9.5)
    ax_hg.set_ylabel("Number of Selected Electrodes", fontsize=11)
    ax_hg.grid(True, linestyle='--', alpha=0.3)
    ax_hg.set_facecolor('#fcfcfc')
    
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            if height >= 0:
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    autolabel(rects1_t, ax_erp)
    autolabel(rects2_t, ax_hg)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[FIGURE] Saved target-only comparison plot to: {out_path}")
    ax_hg.legend(loc='upper right', frameon=True)
    
    autolabel(rects2_w, ax_hg)
    autolabel(rects2_t, ax_hg)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

# ----------------- 绘制 Nilearn 2D 玻璃脑电极图 -----------------
def plot_nilearn_glass_brain_electrodes(erp_list, hg_list, title_str, out_path):
    """
    使用 Nilearn 库绘制 2D 玻璃脑电极分布图。
    erp_list/hg_list 元素结构为: {'coords': [X, Y, Z], 'strategy': 1~4}
    """
    from nilearn import plotting
    
    strategy_colors = {
        1: '#2ca02c',  # 🟢 绿色 (Strategy 1)
        2: '#1f77b4',  # 🔵 蓝色 (Strategy 2)
        3: '#9467bd',  # 🟣 紫色 (Strategy 3)
        4: '#ff7f0e'   # 🟡 橙/黄色 (Strategy 4)
    }
    
    fig = plt.figure(figsize=(15, 10))
    display = plotting.plot_glass_brain(None, display_mode='ortho', figure=fig, title=title_str)
    
    # 1. 绘制 ERP 电极 (实心圆)
    if erp_list:
        coords = np.array([item['coords'] for item in erp_list])
        colors = [strategy_colors[item['strategy']] for item in erp_list]
        display.add_markers(
            marker_coords=coords,
            marker_color=colors,
            marker_size=120,
            marker='o',
            alpha=0.95
        )
        
    # 2. 绘制 HG 电极 (空心圆圈，marker_color='none')
    if hg_list:
        coords = np.array([item['coords'] for item in hg_list])
        colors = [strategy_colors[item['strategy']] for item in hg_list]
        display.add_markers(
            marker_coords=coords,
            marker_color='none',
            edgecolors=colors,
            marker_size=120,
            linewidths=2.5,
            marker='o',
            alpha=0.95
        )
        
    # 绘制图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=10, label='Strategy 1 (Merged Mean)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='Strategy 2 (Merged Cont 50ms)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9467bd', markersize=10, label='Strategy 3 (Single Mean)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=10, label='Strategy 4 (Single Cont 50ms)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='ERP (Solid Circle)'),
        Line2D([0], [0], marker='o', color='gray', markerfacecolor='none', markeredgewidth=2, markersize=10, label='High Gamma (Hollow Circle)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05), frameon=True)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

# ----------------- 数据分析主逻辑 -----------------
def run_selection_extended():
    print("="*60)
    print("Step 1_1: Starting Double Screening (ERP & HG) and Visualization")
    print("="*60)
    
    # 缓存用于全脑对比柱状图统计的数据
    # columns: ['Subject', 'Signal', 'Electrode_ID', 'Strategy_1', 'Strategy_2', 'Strategy_3', 'Strategy_4', 'In_Target']
    records_for_plot_main = []
    records_for_plot_more = []
    
    # 缓存用于 Nilearn 玻璃脑绘制的数据
    # 各元素为 {'coords': [X,Y,Z], 'strategy': 1~4, 'subject': subj}
    glass_brain_main_erp = []
    glass_brain_main_hg = []
    glass_brain_more_erp = []
    glass_brain_more_hg = []
    
    # 缓存最终导出的 Excel 表格记录
    final_selected_excel_records = {} # key: (subj, elec), value: dict
    final_more_excel_records = {}
    
    for subj in subjects:
        print(f"\nProcessing Subject: {subj}...")
        
        # 1. 加载解剖定位表与 MNI 坐标
        loc_path = os.path.join(base_dir, 'processed_data', subj, f'{subj}_ieegloc.xlsx')
        df_loc = pd.read_excel(loc_path)
        ch_col = 'Channel' if 'Channel' in df_loc.columns else df_loc.columns[0]
        aal_col = 'AAL3 (MNI-linear)' if 'AAL3 (MNI-linear)' in df_loc.columns else 'AAL3 (MNI-segment)'
        
        elec_aal_map = {}
        elec_mni_map = {}
        for idx, row in df_loc.iterrows():
            ch = str(row[ch_col]).strip()
            # AAL3
            aal = row[aal_col]
            elec_aal_map[ch] = str(aal) if not pd.isna(aal) else ''
            # MNI
            mni_val = row.get('MNI', None)
            if pd.notna(mni_val):
                try:
                    coords = ast.literal_eval(str(mni_val))
                    if isinstance(coords, (list, tuple)) and len(coords) == 3:
                        elec_mni_map[ch] = [float(c) for c in coords]
                except:
                    pass
            if ch not in elec_mni_map:
                elec_mni_map[ch] = [np.nan, np.nan, np.nan]
                
        # 2. 读取原始的 ERP 和 HG 数据
        src_erp_path = os.path.join(feature_dir, subj, 'task1_ERP_epoched.mat')
        erp_mat = read_mat(src_erp_path)
        erp_epoch = erp_mat['epoch']
        erp_data = erp_epoch['data']
        erp_ch_labels = list(erp_epoch['ch']['labels'])
        # 兼容 time_ms
        if 'time_ms' in erp_epoch:
            erp_time_ms = erp_epoch['time_ms']
        else:
            erp_time_ms = np.linspace(-500, 998, erp_data.shape[-1])
            
        src_hg_path = os.path.join(feature_dir, subj, 'task1_hg_subband.mat')
        hg_mat = read_mat(src_hg_path)
        hg_epoch = hg_mat['epoch']
        hg_data_cell = hg_epoch['data_cell']
        hg_ch_labels = list(hg_epoch['ch']['labels'])
        hg_time_ms = hg_epoch['time_ms']
        
        # 3. 对所有通道分别计算 ERP 和 HG 在四种策略下的满足情况
        # ERP 策略计算
        erp_strategies = {}
        for ch_idx, elec in enumerate(erp_ch_labels):
            c_merged = np.concatenate([erp_data[idx, :, ch_idx, :] for idx in [0, 2, 4, 6]], axis=0)
            g_merged = np.concatenate([erp_data[idx, :, ch_idx, :] for idx in [1, 3, 5, 7]], axis=0)
            c_merged = c_merged[~np.isnan(c_merged).any(axis=1)]
            g_merged = g_merged[~np.isnan(g_merged).any(axis=1)]
            
            c_singles, g_singles = [], []
            for c_idx, g_idx in cond_pairs:
                c_s = erp_data[c_idx, :, ch_idx, :]
                g_s = erp_data[g_idx, :, ch_idx, :]
                c_singles.append(c_s[~np.isnan(c_s).any(axis=1)])
                g_singles.append(g_s[~np.isnan(g_s).any(axis=1)])
                
            erp_strategies[elec] = check_channel_strategies(c_merged, g_merged, c_singles, g_singles, erp_time_ms)
            
        # HG 策略计算
        hg_strategies = {}
        for ch_idx, elec in enumerate(hg_ch_labels):
            c_merged = np.concatenate([hg_data_cell[idx][:, ch_idx, :] for idx in [0, 2, 4, 6]], axis=0)
            g_merged = np.concatenate([hg_data_cell[idx][:, ch_idx, :] for idx in [1, 3, 5, 7]], axis=0)
            c_merged = c_merged[~np.isnan(c_merged).any(axis=1)]
            g_merged = g_merged[~np.isnan(g_merged).any(axis=1)]
            
            c_singles, g_singles = [], []
            for c_idx, g_idx in cond_pairs:
                c_s = hg_data_cell[c_idx][:, ch_idx, :]
                g_s = hg_data_cell[g_idx][:, ch_idx, :]
                c_singles.append(c_s[~np.isnan(c_s).any(axis=1)])
                g_singles.append(g_s[~np.isnan(g_s).any(axis=1)])
                
            hg_strategies[elec] = check_channel_strategies(c_merged, g_merged, c_singles, g_singles, hg_time_ms)
            
        # 4. 统计缓存用于主要通道柱状图和筛选的通道数据
        # 并集通道 (凡是在 ERP 或 HG 中有定义的通道)
        union_all_channels = list(set(erp_ch_labels) | set(hg_ch_labels))
        
        # 主要电极的并集列表
        selected_channels_subj = []
        
        for elec in union_all_channels:
            aal = elec_aal_map.get(elec, '')
            roi_cat = get_roi_category(aal)
            in_target = roi_cat is not None
            
            # ERP 缓存
            if elec in erp_ch_labels:
                s1, s2, s3, s4 = erp_strategies[elec]
                records_for_plot_main.append({
                    'Subject': subj, 'Signal': 'ERP', 'Electrode_ID': elec,
                    'Strategy_1': s1, 'Strategy_2': s2, 'Strategy_3': s3, 'Strategy_4': s4,
                    'In_Target': in_target
                })
            else:
                s1, s2, s3, s4 = False, False, False, False
                
            # HG 缓存
            if elec in hg_ch_labels:
                h1, h2, h3, h4 = hg_strategies[elec]
                records_for_plot_main.append({
                    'Subject': subj, 'Signal': 'HG', 'Electrode_ID': elec,
                    'Strategy_1': h1, 'Strategy_2': h2, 'Strategy_3': h3, 'Strategy_4': h4,
                    'In_Target': in_target
                })
            else:
                h1, h2, h3, h4 = False, False, False, False
                
            # 如果符合靶区要求，且 ERP 或 HG 至少有一种策略显著，则是主要筛选电极！
            if in_target:
                erp_sig = (s1 or s2 or s3 or s4)
                hg_sig = (h1 or h2 or h3 or h4)
                if erp_sig or hg_sig:
                    selected_channels_subj.append(elec)
                    
                    # 确定 MNI 坐标
                    mni_coords = elec_mni_map.get(elec, [np.nan, np.nan, np.nan])
                    
                    # 汇总策略列表
                    erp_matched = [str(i) for i, val in enumerate([s1, s2, s3, s4], 1) if val]
                    hg_matched = [str(i) for i, val in enumerate([h1, h2, h3, h4], 1) if val]
                    
                    final_selected_excel_records[(subj, elec)] = {
                        'Subject': subj,
                        'Electrode': elec,
                        'AAL3': aal,
                        'AAL3_ROI': roi_cat,
                        'MNI_X': mni_coords[0],
                        'MNI_Y': mni_coords[1],
                        'MNI_Z': mni_coords[2],
                        'ERP_Selected': erp_sig,
                        'ERP_Strategies_Matched': ",".join(erp_matched) if erp_matched else "None",
                        'HG_Selected': hg_sig,
                        'HG_Strategies_Matched': ",".join(hg_matched) if hg_matched else "None"
                    }
                    
                    # 确定 Nilearn 颜色优先级 (1 > 2 > 3 > 4)
                    if erp_sig:
                        for strat in [1, 2, 3, 4]:
                            if [s1, s2, s3, s4][strat-1]:
                                glass_brain_main_erp.append({'coords': mni_coords, 'strategy': strat, 'subject': subj})
                                break
                    if hg_sig:
                        for strat in [1, 2, 3, 4]:
                            if [h1, h2, h3, h4][strat-1]:
                                glass_brain_main_hg.append({'coords': mni_coords, 'strategy': strat, 'subject': subj})
                                break
                                
        print(f"  Selected main channels count: {len(selected_channels_subj)}: {selected_channels_subj}")
        
        # 5. 扩展物理邻近电极筛选
        more_selected_channels_subj = []
        for main_ch in selected_channels_subj:
            neighbors = find_neighbors(main_ch, union_all_channels)
            for neigh in neighbors:
                if neigh in selected_channels_subj or neigh in more_selected_channels_subj:
                    continue
                    
                aal_neigh = elec_aal_map.get(neigh, '')
                # 是否为 unknown/NA/parahippocampus
                in_more_target = is_neighbor_valid_label(aal_neigh)
                
                # ERP 策略
                if neigh in erp_ch_labels:
                    s1, s2, s3, s4 = erp_strategies[neigh]
                    records_for_plot_more.append({
                        'Subject': subj, 'Signal': 'ERP', 'Electrode_ID': neigh,
                        'Strategy_1': s1, 'Strategy_2': s2, 'Strategy_3': s3, 'Strategy_4': s4,
                        'In_Target': in_more_target
                    })
                else:
                    s1, s2, s3, s4 = False, False, False, False
                    
                # HG 策略
                if neigh in hg_ch_labels:
                    h1, h2, h3, h4 = hg_strategies[neigh]
                    records_for_plot_more.append({
                        'Subject': subj, 'Signal': 'HG', 'Electrode_ID': neigh,
                        'Strategy_1': h1, 'Strategy_2': h2, 'Strategy_3': h3, 'Strategy_4': h4,
                        'In_Target': in_more_target
                    })
                else:
                    h1, h2, h3, h4 = False, False, False, False
                    
                # 如果解剖匹配，且 ERP/HG 有一个显著，则是扩展电极！
                if in_more_target:
                    erp_sig = (s1 or s2 or s3 or s4)
                    hg_sig = (h1 or h2 or h3 or h4)
                    if erp_sig or hg_sig:
                        more_selected_channels_subj.append(neigh)
                        mni_coords = elec_mni_map.get(neigh, [np.nan, np.nan, np.nan])
                        
                        erp_matched = [str(i) for i, val in enumerate([s1, s2, s3, s4], 1) if val]
                        hg_matched = [str(i) for i, val in enumerate([h1, h2, h3, h4], 1) if val]
                        
                        final_more_excel_records[(subj, neigh)] = {
                            'Subject': subj,
                            'Electrode': neigh,
                            'Neighbor_Of': main_ch,
                            'AAL3': aal_neigh,
                            'AAL3_ROI': '旁海马或未知/NA' if 'parahippocamp' in aal_neigh.lower() else '未知/NA',
                            'MNI_X': mni_coords[0],
                            'MNI_Y': mni_coords[1],
                            'MNI_Z': mni_coords[2],
                            'ERP_Selected': erp_sig,
                            'ERP_Strategies_Matched': ",".join(erp_matched) if erp_matched else "None",
                            'HG_Selected': hg_sig,
                            'HG_Strategies_Matched': ",".join(hg_matched) if hg_matched else "None"
                        }
                        
                        # Nilearn
                        if erp_sig:
                            for strat in [1, 2, 3, 4]:
                                if [s1, s2, s3, s4][strat-1]:
                                    glass_brain_more_erp.append({'coords': mni_coords, 'strategy': strat, 'subject': subj})
                                    break
                        if hg_sig:
                            for strat in [1, 2, 3, 4]:
                                if [h1, h2, h3, h4][strat-1]:
                                    glass_brain_more_hg.append({'coords': mni_coords, 'strategy': strat, 'subject': subj})
                                    break
                                    
        print(f"  Selected extended (more) channels count: {len(more_selected_channels_subj)}: {more_selected_channels_subj}")
        
        # 6. 为选中的通道绘制 ERP 和 HG 信号图，并保存到新路径
        # A. 主要电极绘图
        print(f"  Plotting ERP & HG signals for main channels of {subj}...")
        for elec in selected_channels_subj:
            rec = final_selected_excel_records[(subj, elec)]
            # ERP 信号图
            if rec['ERP_Selected'] and elec in erp_ch_labels:
                ch_idx = erp_ch_labels.index(elec)
                stras_str = rec['ERP_Strategies_Matched'].replace(",", "_")
                out_path = os.path.join(result_dir, 'select_channel', 'erp', subj, f"stra{stras_str}_{elec}.png")
                plot_signal_erp_or_hg(subj, elec, ch_idx, erp_data, erp_time_ms, out_path, is_hg=False)
            # HG 信号图
            if rec['HG_Selected'] and elec in hg_ch_labels:
                ch_idx = hg_ch_labels.index(elec)
                stras_str = rec['HG_Strategies_Matched'].replace(",", "_")
                out_path = os.path.join(result_dir, 'select_channel', 'hg', subj, f"stra{stras_str}_{elec}.png")
                plot_signal_erp_or_hg(subj, elec, ch_idx, hg_data_cell, hg_time_ms, out_path, is_hg=True)
                
        # B. 扩展电极绘图
        print(f"  Plotting ERP & HG signals for extended channels of {subj}...")
        for elec in more_selected_channels_subj:
            rec = final_more_excel_records[(subj, elec)]
            # ERP
            if rec['ERP_Selected'] and elec in erp_ch_labels:
                ch_idx = erp_ch_labels.index(elec)
                stras_str = rec['ERP_Strategies_Matched'].replace(",", "_")
                out_path = os.path.join(result_dir, 'more_select_channel', 'erp', subj, f"stra{stras_str}_{elec}.png")
                plot_signal_erp_or_hg(subj, elec, ch_idx, erp_data, erp_time_ms, out_path, is_hg=False)
            # HG
            if rec['HG_Selected'] and elec in hg_ch_labels:
                ch_idx = hg_ch_labels.index(elec)
                stras_str = rec['HG_Strategies_Matched'].replace(",", "_")
                out_path = os.path.join(result_dir, 'more_select_channel', 'hg', subj, f"stra{stras_str}_{elec}.png")
                plot_signal_erp_or_hg(subj, elec, ch_idx, hg_data_cell, hg_time_ms, out_path, is_hg=True)
                
        # 7. 裁剪原 mat 特征文件通道并另存
        print(f"  Cropping feature mat files for {subj}...")
        # A. 主要通道裁剪并保存
        if selected_channels_subj:
            # ERP
            selected_erp_ch = [e for e in selected_channels_subj if e in erp_ch_labels]
            cropped_erp = crop_erp_mat(erp_mat, selected_erp_ch)
            out_erp_dir = os.path.join(feature_dir, 'select_channel', subj)
            os.makedirs(out_erp_dir, exist_ok=True)
            sio.savemat(os.path.join(out_erp_dir, 'task1_ERP_epoched.mat'), cropped_erp, long_field_names=True)
            
            # HG
            selected_hg_ch = [e for e in selected_channels_subj if e in hg_ch_labels]
            cropped_hg = crop_hg_mat(hg_mat, selected_hg_ch)
            out_hg_dir = os.path.join(feature_dir, 'select_channel', subj)
            os.makedirs(out_hg_dir, exist_ok=True)
            sio.savemat(os.path.join(out_hg_dir, 'task1_hg_subband.mat'), cropped_hg, long_field_names=True)
            
        # B. 扩展通道裁剪并保存
        if more_selected_channels_subj:
            # ERP
            more_erp_ch = [e for e in more_selected_channels_subj if e in erp_ch_labels]
            cropped_erp = crop_erp_mat(erp_mat, more_erp_ch)
            out_erp_dir = os.path.join(feature_dir, 'more_select_channel', subj)
            os.makedirs(out_erp_dir, exist_ok=True)
            sio.savemat(os.path.join(out_erp_dir, 'task1_ERP_epoched.mat'), cropped_erp, long_field_names=True)
            
            # HG
            more_hg_ch = [e for e in more_selected_channels_subj if e in hg_ch_labels]
            cropped_hg = crop_hg_mat(hg_mat, more_hg_ch)
            out_hg_dir = os.path.join(feature_dir, 'more_select_channel', subj)
            os.makedirs(out_hg_dir, exist_ok=True)
            sio.savemat(os.path.join(out_hg_dir, 'task1_hg_subband.mat'), cropped_hg, long_field_names=True)
            
    # 8. 保存 Excel 汇总文件
    print("\nExporting XLSX spreadsheets...")
    df_selected = pd.DataFrame(list(final_selected_excel_records.values()))
    selected_xlsx_path = os.path.join(doc_dir, 'select_channel_summary.xlsx')
    selected_csv_path = os.path.join(doc_dir, 'select_channel_summary.csv')
    df_selected.to_excel(selected_xlsx_path, index=False)
    df_selected.to_csv(selected_csv_path, index=False)
    
    # 备份到结果文件夹中
    df_selected.to_excel(os.path.join(result_dir, 'select_channel', 'select_channel_summary.xlsx'), index=False)
    df_selected.to_csv(os.path.join(result_dir, 'select_channel', 'select_channel_summary.csv'), index=False)
    
    df_more = pd.DataFrame(list(final_more_excel_records.values()))
    more_xlsx_path = os.path.join(doc_dir, 'more_select_channel_summary.xlsx')
    more_csv_path = os.path.join(doc_dir, 'more_select_channel_summary.csv')
    df_more.to_excel(more_xlsx_path, index=False)
    df_more.to_csv(more_csv_path, index=False)
    
    # 备份到结果文件夹中
    df_more.to_excel(os.path.join(result_dir, 'more_select_channel', 'more_select_channel_summary.xlsx'), index=False)
    df_more.to_csv(os.path.join(result_dir, 'more_select_channel', 'more_select_channel_summary.csv'), index=False)
    
    # 9. 绘制电极数量对比柱状图
    print("\nPlotting selection comparison bar charts...")
    # A. 主要电极
    df_plot_main = pd.DataFrame(records_for_plot_main)
    # 总体
    plot_electrode_selection_comparison(df_plot_main, "Group (test001+test002+test003)", os.path.join(result_dir, 'select_channel', 'electrode_selection_comparison.png'))
    # 被试各自
    for subj in subjects:
        plot_electrode_selection_comparison(df_plot_main[df_plot_main['Subject'] == subj], subj, os.path.join(result_dir, 'select_channel', f'{subj}_electrode_selection_comparison.png'))
        
    # B. 扩展电极
    df_plot_more = pd.DataFrame(records_for_plot_more)
    if not df_plot_more.empty:
        # 总体
        plot_electrode_selection_comparison(df_plot_more, "Group (test001+test002+test003) - Extended", os.path.join(result_dir, 'more_select_channel', 'electrode_selection_comparison.png'))
        # 被试各自
        for subj in subjects:
            plot_electrode_selection_comparison(df_plot_more[df_plot_more['Subject'] == subj], f'{subj} - Extended', os.path.join(result_dir, 'more_select_channel', f'{subj}_electrode_selection_comparison.png'))
            
    # 10. 绘制 Nilearn 2D 玻璃脑电极图
    print("\nPlotting 2D glass brain electrode distributions using Nilearn...")
    # A. 主要电极
    # 总体
    plot_nilearn_glass_brain_electrodes(glass_brain_main_erp, glass_brain_main_hg, 'Electrode Selection - Group (ERP: Solid, HG: Hollow)', os.path.join(result_dir, 'select_channel', 'nilearn_glass_brain_electrodes.png'))
    # 被试各自
    for subj in subjects:
        g_erp = [item for item in glass_brain_main_erp if item['subject'] == subj]
        g_hg = [item for item in glass_brain_main_hg if item['subject'] == subj]
        plot_nilearn_glass_brain_electrodes(g_erp, g_hg, f'Electrode Selection - {subj} (ERP: Solid, HG: Hollow)', os.path.join(result_dir, 'select_channel', f'{subj}_nilearn_glass_brain_electrodes.png'))
        
    # B. 扩展电极
    # 总体
    plot_nilearn_glass_brain_electrodes(glass_brain_more_erp, glass_brain_more_hg, 'Extended Electrode Selection - Group (ERP: Solid, HG: Hollow)', os.path.join(result_dir, 'more_select_channel', 'nilearn_glass_brain_electrodes.png'))
    # 被试各自
    for subj in subjects:
        g_erp = [item for item in glass_brain_more_erp if item['subject'] == subj]
        g_hg = [item for item in glass_brain_more_hg if item['subject'] == subj]
        plot_nilearn_glass_brain_electrodes(g_erp, g_hg, f'Extended Electrode Selection - {subj} (ERP: Solid, HG: Hollow)', os.path.join(result_dir, 'more_select_channel', f'{subj}_nilearn_glass_brain_electrodes.png'))
        
    print("\n" + "="*60)
    print("Step 1_1 Electrode Selection and Visualization Complete!")
    print("="*60)

if __name__ == '__main__':
    run_selection_extended()
