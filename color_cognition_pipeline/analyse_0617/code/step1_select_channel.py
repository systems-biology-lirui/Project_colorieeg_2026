import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import ranksums
from pymatreader import read_mat
import matplotlib.pyplot as plt
import os
import re
import warnings

# 忽略绘图和计算过程中的警告
warnings.filterwarnings('ignore')

# 基础路径配置
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
analyse_dir = os.path.join(pipeline_dir, 'analyse_0617')
feature_dir = os.path.join(analyse_dir, 'feature')
doc_dir = os.path.join(analyse_dir, 'doc')
result_dir = os.path.join(analyse_dir, 'result')

subjects = ['test001', 'test002', 'test003']
categories = ['Face', 'Object', 'Body', 'Place', 'Merged_All']
# 单一类别的 Color 和 Gray 触发器索引对
# Face: 0 vs 1; Object: 2 vs 3; Body: 4 vs 5; Place: 6 vs 7
cond_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

# 创建必要的文件夹
os.makedirs(doc_dir, exist_ok=True)

# ----------------- 脑区匹配辅助函数 -----------------
def get_roi_category(label):
    """
    根据解剖标签判断其所属的 ROI 大类。
    返回: '枕叶', '颞叶后/下部', '颞叶前/上部' 或 None
    """
    if not isinstance(label, str) or label.strip() == '':
        return None
    label_lower = label.lower()
    
    # 枕叶: Calcarine, Occipital_Inf, Occipital_Mid, Lingual
    if any(kw in label_lower for kw in ['calcarine', 'occipital_inf', 'occipital_mid', 'lingual']):
        return '枕叶'
    # 颞叶后/下部: Fusiform, Temporal_Inf
    elif any(kw in label_lower for kw in ['fusiform', 'temporal_inf']):
        return '颞叶后/下部'
    # 颞叶前/上部: Temporal_Mid, Temporal_Pole
    elif any(kw in label_lower for kw in ['temporal_mid', 'temporal_pole']):
        return '颞叶前/上部'
    return None

def is_neighbor_valid_label(label):
    """
    判断邻近电极的解剖标注是否属于 unknown, N/A 或旁海马
    """
    if pd.isna(label) or not isinstance(label, str) or label.strip() == '':
        return True
    label_lower = label.lower().strip()
    if 'unknown' in label_lower or 'n/a' in label_lower or label_lower == 'nan':
        return True
    if 'parahippocampal' in label_lower or 'parahippocampus' in label_lower:
        return True
    return False

# ----------------- 通道前缀与序号解析 -----------------
def parse_channel_name(ch_name):
    """
    解析例如 G11 -> ('G', 11), FP2 -> ('FP', 2)
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
    在所有可用通道中，寻找 target_ch 的物理邻近电极（同一轴且序号相差为 +/- 1）
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

# ----------------- 统计学筛选核心逻辑 -----------------
def has_continuous_sig(sig_bool, consec_pts):
    """
    检查布尔数组中是否存在至少连续 consec_pts 个 True
    """
    count = 0
    for val in sig_bool:
        if val:
            count += 1
            if count >= consec_pts:
                return True
        else:
            count = 0
    return False

def check_strategies(c_merged, g_merged, c_singles, g_singles, time_ms):
    """
    计算该电极是否满足四种统计学策略。
    c_merged, g_merged: 混合类别的 Color/Gray trials, shape [N_merged_trials, N_time]
    c_singles, g_singles: 单一类别的 Color/Gray trials 列表, 每个元素为 [N_trials, N_time]
    time_ms: 时间轴
    返回: (stra1, stra2, stra3, stra4)
    """
    # 动态计算连续显著的点数，即持续时间 >= 50ms
    dt = time_ms[1] - time_ms[0]
    consec_pts = int(np.ceil(50.0 / dt))
    
    # 提取时间窗口索引
    idx_100_400 = np.where((time_ms >= 100) & (time_ms <= 400))[0]
    idx_50_400 = np.where((time_ms >= 50) & (time_ms <= 400))[0]
    
    # 1. 策略1: 混合类别的 100-400ms 平均值显著
    c_merged_100_400 = np.mean(c_merged[:, idx_100_400], axis=1)
    g_merged_100_400 = np.mean(g_merged[:, idx_100_400], axis=1)
    _, p_stra1 = ranksums(c_merged_100_400, g_merged_100_400)
    stra1 = p_stra1 < 0.05
    
    # 2. 策略2: 混合类别在 50-400ms 点对点连续显著 >= 50ms
    sig_merged = []
    for t_idx in idx_50_400:
        c_vals = c_merged[:, t_idx]
        g_vals = g_merged[:, t_idx]
        _, p_t = ranksums(c_vals, g_vals)
        sig_merged.append(p_t < 0.05)
    stra2 = has_continuous_sig(sig_merged, consec_pts)
    
    # 3. 策略3: 至少一个单一类别的 100-400ms 平均值显著
    stra3 = False
    for c_s, g_s in zip(c_singles, g_singles):
        c_s_100_400 = np.mean(c_s[:, idx_100_400], axis=1)
        g_s_100_400 = np.mean(g_s[:, idx_100_400], axis=1)
        _, p_s = ranksums(c_s_100_400, g_s_100_400)
        if p_s < 0.05:
            stra3 = True
            break
            
    # 4. 策略4: 至少一个单一类别在 50-400ms 下点对点连续显著 >= 50ms
    stra4 = False
    for c_s, g_s in zip(c_singles, g_singles):
        sig_single = []
        for t_idx in idx_50_400:
            c_vals = c_s[:, t_idx]
            g_vals = g_s[:, t_idx]
            _, p_t = ranksums(c_vals, g_vals)
            sig_single.append(p_t < 0.05)
        if has_continuous_sig(sig_single, consec_pts):
            stra4 = True
            break
            
    return stra1, stra2, stra3, stra4

# ----------------- 绘图辅助函数 -----------------
def plot_electrode_erp(subject, elec, ch_idx, erp_data, time_ms, out_path):
    """
    绘制并保存电极在不同类别下的 ERP 信号对比图。
    结构为 5行2列，左边为时程图，右边为100-400ms条形散点图。
    """
    fig, axes = plt.subplots(5, 2, figsize=(12, 20), gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(f"Subject: {subject} | Electrode: {elec} - ERP", fontsize=16, fontweight='bold', y=0.98)
    
    idx_100_400 = np.where((time_ms >= 100) & (time_ms <= 400))[0]
    
    for i, cat_name in enumerate(categories):
        ax_time = axes[i, 0]
        ax_bar = axes[i, 1]
        
        # 提取当前类别下的 Color 和 Gray trials 数据
        if cat_name == 'Merged_All':
            c_data = np.concatenate([erp_data[idx, :, ch_idx, :] for idx in [0, 2, 4, 6]], axis=0)
            g_data = np.concatenate([erp_data[idx, :, ch_idx, :] for idx in [1, 3, 5, 7]], axis=0)
        else:
            c_idx, g_idx = cond_pairs[i]
            c_data = erp_data[c_idx, :, ch_idx, :]
            g_data = erp_data[g_idx, :, ch_idx, :]
            
        # 过滤 NaN 的 trials
        c_data = c_data[~np.isnan(c_data).any(axis=1)]
        g_data = g_data[~np.isnan(g_data).any(axis=1)]
        
        # 计算均值和 SEM
        c_mean = np.mean(c_data, axis=0)
        c_sem = np.std(c_data, axis=0) / np.sqrt(c_data.shape[0])
        g_mean = np.mean(g_data, axis=0)
        g_sem = np.std(g_data, axis=0) / np.sqrt(g_data.shape[0])
        
        # 1. 绘制左侧 ERP 波形时程
        ax_time.plot(time_ms, c_mean, color='#d32f2f', lw=2.2, label='Color')
        ax_time.fill_between(time_ms, c_mean - c_sem, c_mean + c_sem, color='#d32f2f', alpha=0.15)
        
        ax_time.plot(time_ms, g_mean, color='#212121', lw=2.2, label='Gray')
        ax_time.fill_between(time_ms, g_mean - g_sem, g_mean + g_sem, color='#212121', alpha=0.15)
        
        # 辅助虚线
        ax_time.axvline(0, color='#9E9E9E', linestyle='--', alpha=0.6)
        
        # 计算并绘制点对点显著性标记
        ymin, ymax = ax_time.get_ylim()
        sig_y = ymin + (ymax - ymin) * 0.05
        
        for t_idx in range(len(time_ms)):
            stat, p = ranksums(c_data[:, t_idx], g_data[:, t_idx])
            if p < 0.05:
                color = 'yellow' if stat > 0 else 'cyan'
                ax_time.plot(time_ms[t_idx], sig_y, marker='s', color=color, markersize=3, alpha=0.7)
                
        ax_time.set_title(f"{cat_name} (Time Course)", fontsize=11, fontweight='bold')
        ax_time.set_xlabel("Time (ms)", fontsize=9.5)
        ax_time.set_ylabel("Amplitude (μV)", fontsize=9.5)
        ax_time.set_xlim([-200, 800])
        ax_time.grid(False)
        for spine in ax_time.spines.values():
            spine.set_visible(True)
            spine.set_color('#757575')
            
        if i == 0:
            ax_time.legend(loc='upper right', frameon=True, fontsize=8)
            
        # 2. 绘制右侧 100-400ms 平均幅值条形散点图
        c_vals = np.mean(c_data[:, idx_100_400], axis=1)
        g_vals = np.mean(g_data[:, idx_100_400], axis=1)
        
        bar_c_mean = np.mean(c_vals)
        bar_c_sem = np.std(c_vals) / np.sqrt(len(c_vals))
        bar_g_mean = np.mean(g_vals)
        bar_g_sem = np.std(g_vals) / np.sqrt(len(g_vals))
        
        ax_bar.bar([1], [bar_c_mean], yerr=[bar_c_sem], color='#d32f2f', alpha=0.7, capsize=5, width=0.4, error_kw={'elinewidth':1.5, 'capthick':1.5})
        ax_bar.bar([2], [bar_g_mean], yerr=[bar_g_sem], color='#212121', alpha=0.7, capsize=5, width=0.4, error_kw={'elinewidth':1.5, 'capthick':1.5})
        
        # 绘制 trial 个体散点
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

# ----------------- 数据分析主流程 -----------------
def run_selection():
    print("="*60)
    print("Starting Electrode Screening and Extended Identification")
    print("="*60)
    
    all_selected_records = []
    all_more_records = []
    
    for subj in subjects:
        print(f"\nProcessing Subject: {subj}")
        
        # 1. 加载解剖定位表
        loc_path = os.path.join(base_dir, 'processed_data', subj, f'{subj}_ieegloc.xlsx')
        df_loc = pd.read_excel(loc_path)
        ch_col = 'Channel' if 'Channel' in df_loc.columns else df_loc.columns[0]
        aal_col = 'AAL3 (MNI-linear)' if 'AAL3 (MNI-linear)' in df_loc.columns else 'AAL3 (MNI-segment)'
        
        elec_aal_map = {}
        for idx, row in df_loc.iterrows():
            ch = str(row[ch_col]).strip()
            aal = row[aal_col]
            elec_aal_map[ch] = str(aal) if not pd.isna(aal) else ''
            
        # 2. 加载 ERP feature
        erp_path = os.path.join(feature_dir, subj, 'task1_ERP_epoched.mat')
        mat = read_mat(erp_path)
        epoch = mat['epoch']
        erp_data = epoch['data']  # (8, 70, N_ch, 750)
        ch_labels = list(epoch['ch']['labels'])
        if 'time_ms' in epoch:
            time_ms = epoch['time_ms']
        else:
            time_ms = np.linspace(-500, 998, erp_data.shape[-1])
        
        n_ch = len(ch_labels)
        print(f"  Loaded ERP data: channels count = {n_ch}, time range = {time_ms.min()} to {time_ms.max()}ms")
        
        # 3. 对所有通道进行统计学策略计算
        ch_strategies = {}
        for ch_idx, elec in enumerate(ch_labels):
            # 获取混合类别数据
            c_merged = np.concatenate([erp_data[idx, :, ch_idx, :] for idx in [0, 2, 4, 6]], axis=0)
            g_merged = np.concatenate([erp_data[idx, :, ch_idx, :] for idx in [1, 3, 5, 7]], axis=0)
            c_merged = c_merged[~np.isnan(c_merged).any(axis=1)]
            g_merged = g_merged[~np.isnan(g_merged).any(axis=1)]
            
            # 获取单一类别数据
            c_singles = []
            g_singles = []
            for c_idx, g_idx in cond_pairs:
                c_s = erp_data[c_idx, :, ch_idx, :]
                g_s = erp_data[g_idx, :, ch_idx, :]
                c_singles.append(c_s[~np.isnan(c_s).any(axis=1)])
                g_singles.append(g_s[~np.isnan(g_s).any(axis=1)])
                
            # 计算四种策略的满足情况
            s1, s2, s3, s4 = check_strategies(c_merged, g_merged, c_singles, g_singles, time_ms)
            ch_strategies[elec] = (s1, s2, s3, s4)
            
        # 4. 执行主要通道的筛选 (必须满足 ROI 分类 + 任意一种策略)
        selected_channels_subj = []
        for elec in ch_labels:
            aal = elec_aal_map.get(elec, '')
            roi_cat = get_roi_category(aal)
            
            if roi_cat is not None:
                s1, s2, s3, s4 = ch_strategies[elec]
                if s1 or s2 or s3 or s4:
                    matched_stras = []
                    if s1: matched_stras.append("1")
                    if s2: matched_stras.append("2")
                    if s3: matched_stras.append("3")
                    if s4: matched_stras.append("4")
                    
                    selected_channels_subj.append(elec)
                    
                    all_selected_records.append({
                        'Subject': subj,
                        'Electrode': elec,
                        'AAL3': aal,
                        'ROI_Category': roi_cat,
                        'Strategy_1': s1,
                        'Strategy_2': s2,
                        'Strategy_3': s3,
                        'Strategy_4': s4,
                        'Strategies_Matched': ",".join(matched_stras)
                    })
                    
        print(f"  Selected main channels count: {len(selected_channels_subj)}: {selected_channels_subj}")
        
        # 5. 执行扩展物理邻近通道筛选 (主要通道的相邻物理通道且符合解剖和策略)
        more_selected_channels_subj = []
        for main_ch in selected_channels_subj:
            # 找到 main_ch 在所有通道里的邻居
            neighbors = find_neighbors(main_ch, ch_labels)
            for neigh in neighbors:
                # 排除已经是主要电极的通道，以及已经包含在 more_selected_channels_subj 中的通道
                if neigh in selected_channels_subj or neigh in more_selected_channels_subj:
                    continue
                    
                # 检查解剖标签是否属于 unknown, N/A 或者是 parahippocampus
                aal_neigh = elec_aal_map.get(neigh, '')
                if is_neighbor_valid_label(aal_neigh):
                    s1, s2, s3, s4 = ch_strategies[neigh]
                    if s1 or s2 or s3 or s4:
                        matched_stras = []
                        if s1: matched_stras.append("1")
                        if s2: matched_stras.append("2")
                        if s3: matched_stras.append("3")
                        if s4: matched_stras.append("4")
                        
                        more_selected_channels_subj.append(neigh)
                        
                        all_more_records.append({
                            'Subject': subj,
                            'Electrode': neigh,
                            'Neighbor_Of': main_ch,
                            'AAL3': aal_neigh,
                            'Strategy_1': s1,
                            'Strategy_2': s2,
                            'Strategy_3': s3,
                            'Strategy_4': s4,
                            'Strategies_Matched': ",".join(matched_stras)
                        })
                        
        print(f"  Selected extended (more) channels count: {len(more_selected_channels_subj)}: {more_selected_channels_subj}")
        
        # 6. 为选中的电极绘制 ERP 信号图
        # 主要通道绘图
        print(f"  Plotting ERP signals for main channels of {subj}...")
        for rec in [r for r in all_selected_records if r['Subject'] == subj]:
            elec = rec['Electrode']
            ch_idx = ch_labels.index(elec)
            stras_str = rec['Strategies_Matched'].replace(",", "_")
            out_img_name = f"stra{stras_str}_{elec}.png"
            out_path = os.path.join(result_dir, 'select_channel', subj, out_img_name)
            plot_electrode_erp(subj, elec, ch_idx, erp_data, time_ms, out_path)
            
        # 扩展通道绘图
        print(f"  Plotting ERP signals for extended channels of {subj}...")
        for rec in [r for r in all_more_records if r['Subject'] == subj]:
            elec = rec['Electrode']
            ch_idx = ch_labels.index(elec)
            stras_str = rec['Strategies_Matched'].replace(",", "_")
            out_img_name = f"stra{stras_str}_{elec}.png"
            out_path = os.path.join(result_dir, 'more_select_channel', subj, out_img_name)
            plot_electrode_erp(subj, elec, ch_idx, erp_data, time_ms, out_path)

    # 7. 保存两张汇总表格文件
    print("\nSaving spreadsheet summary tables...")
    df_selected = pd.DataFrame(all_selected_records)
    selected_xlsx_path = os.path.join(doc_dir, 'select_channel_summary.xlsx')
    selected_csv_path = os.path.join(doc_dir, 'select_channel_summary.csv')
    df_selected.to_excel(selected_xlsx_path, index=False)
    df_selected.to_csv(selected_csv_path, index=False)
    print(f"  Main electrodes summary saved to: {selected_xlsx_path} and .csv")
    
    df_more = pd.DataFrame(all_more_records)
    more_xlsx_path = os.path.join(doc_dir, 'more_select_channel_summary.xlsx')
    more_csv_path = os.path.join(doc_dir, 'more_select_channel_summary.csv')
    df_more.to_excel(more_xlsx_path, index=False)
    df_more.to_csv(more_csv_path, index=False)
    print(f"  Extended electrodes summary saved to: {more_xlsx_path} and .csv")
    
    print("\n" + "="*60)
    print("Electrode Screening and Plotting Process Completed Successfully!")
    print("="*60)

if __name__ == '__main__':
    run_selection()
