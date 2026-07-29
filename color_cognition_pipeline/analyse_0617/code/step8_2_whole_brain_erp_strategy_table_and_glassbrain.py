"""
Step 8_2 (修正版):
  1) 重新读取各被试电极定位表，将包含 color_patch 或 color_with_sti (拼写及空格/下划线不限) 的电极标记为特殊电极并在主分析中排除。
     保留只有 V2 和 V4 脑区标注的电极。
     解决多行 ROI 脑区映射问题，每个通道只映射到唯一的 ROI 标签。
  2) 在排除特殊电极的全脑电极上，运行 ERP 和 HG 的 4 策略检测 (使用 Task1)。
  3) 重新生成全脑 ERP 策略表格 `doc/whole_brain_erp_strategy_summary.xlsx` (仅含非特殊电极)。
  4) 重新生成全脑 ERP 玻璃脑 `result/select_channel/whole_brain_erp_glass_brain.png` (Group & Subj)。
  5) 重新生成主筛选对比柱状图 `result/select_channel/electrode_selection_comparison.png` (Group & Subj)。
  6) 新增: 针对所有包含 color_patch 的电极运行 ERP 4 策略检测，绘制玻璃脑图 `result/select_channel/color_patch_erp_glass_brain.png`，
     并输出策略表格 `doc/color_patch_erp_strategy_check.xlsx` & csv。
"""
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import ranksums
from pymatreader import read_mat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, ast, warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 路径 =====================
base_dir   = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline   = os.path.join(base_dir, 'color_cognition_pipeline')
analyse    = os.path.join(pipeline, 'analyse_0617')
feature_dir= os.path.join(analyse, 'feature')
doc_dir    = os.path.join(analyse, 'doc')
result_dir = os.path.join(analyse, 'result')
out_dir    = os.path.join(result_dir, 'select_channel')
os.makedirs(out_dir, exist_ok=True)

subjects = ['test001', 'test002', 'test003']
cond_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

# ===================== 策略检测 (复用核心逻辑) =====================
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
    dt = time_ms[1] - time_ms[0]
    consec_pts = int(np.ceil(50.0 / dt))
    idx_100_400 = np.where((time_ms >= 100) & (time_ms <= 400))[0]
    idx_50_400  = np.where((time_ms >=  50) & (time_ms <= 400))[0]
    
    # Strategy 1
    c_m = np.mean(c_merged[:, idx_100_400], axis=1)
    g_m = np.mean(g_merged[:, idx_100_400], axis=1)
    _, p_s1 = ranksums(c_m, g_m)
    s1 = p_s1 < 0.05
    
    # Strategy 2
    sig_m = []
    for t_idx in idx_50_400:
        _, p_t = ranksums(c_merged[:, t_idx], g_merged[:, t_idx])
        sig_m.append(p_t < 0.05)
    s2 = has_continuous_sig(sig_m, consec_pts)
    
    # Strategy 3
    s3 = False
    for c_s, g_s in zip(c_singles, g_singles):
        c_s_m = np.mean(c_s[:, idx_100_400], axis=1)
        g_s_m = np.mean(g_s[:, idx_100_400], axis=1)
        _, p_s = ranksums(c_s_m, g_s_m)
        if p_s < 0.05:
            s3 = True; break
            
    # Strategy 4
    s4 = False
    for c_s, g_s in zip(c_singles, g_singles):
        sig_s = []
        for t_idx in idx_50_400:
            _, p_t = ranksums(c_s[:, t_idx], g_s[:, t_idx])
            sig_s.append(p_t < 0.05)
        if has_continuous_sig(sig_s, consec_pts):
            s4 = True; break
            
    return s1, s2, s3, s4

# ===================== ROI 分类 =====================
def get_roi_category(label):
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

# ===================== 辅助匹配与清理 =====================
def clean_aal_label(aal):
    if pd.isna(aal):
        return ''
    return str(aal).strip()

def is_special_match(aal, pattern):
    s = aal.lower().replace('_', '').replace(' ', '')
    return pattern in s

# ===================== 绘制主要电极筛选对比柱状图 =====================
def plot_electrode_selection_comparison(data_df, title_prefix, out_path):
    if title_prefix.startswith("Group"):
        title_prefix = "Group (5 Subjects)"
        
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
    fig.suptitle(f"{title_prefix} - Target Area Electrode Selection Across Strategies\n(Excluding color_patch & color_with_sti)", fontsize=14, fontweight='bold', y=0.98)
    
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

# ===================== 主程序 =====================
def main():
    print("="*70)
    print("Step 8_2 (修正版): Whole Brain ERP & HG Analysis and Visualizations")
    print("="*70)
    
    # 临时缓存
    records_for_plot_main = []  # 主柱状图输入
    all_erp_records = []        # 全脑 ERP 汇总表记录
    glass_target_erp = []       # 靶区内 ERP 显著通道
    glass_outside_erp = []      # 靶区外 ERP 显著通道
    
    color_patch_results = []    # color_patch 电极的 ERP 4 策略检验
    glass_color_patch = []      # color_patch 玻璃脑打点记录
    
    for subj in subjects:
        print(f"\nProcessing {subj}...")
        
        # 1. 读入脑电定位并提取信息
        loc_path = os.path.join(base_dir, 'processed_data', subj, f'{subj}_ieegloc.xlsx')
        if not os.path.exists(loc_path):
            print(f"  [ERROR] 定位文件不存在: {loc_path}")
            continue
        df_loc = pd.read_excel(loc_path)
        
        ch_col = 'Channel' if 'Channel' in df_loc.columns else df_loc.columns[0]
        aal_col = 'AAL3 (MNI-linear)' if 'AAL3 (MNI-linear)' in df_loc.columns else 'AAL3 (MNI-segment)'
        
        # 按通道聚合并过滤
        channel_groups = df_loc.groupby(ch_col)
        elec_info_map = {}
        
        for ch, group in channel_groups:
            ch_str = str(ch).strip()
            aal_labels = [clean_aal_label(val) for val in group[aal_col] if pd.notna(val)]
            
            # 查找 MNI 坐标
            mni_coords = [np.nan, np.nan, np.nan]
            for mni_val in group['MNI']:
                if pd.notna(mni_val):
                    try:
                        coords = ast.literal_eval(str(mni_val))
                        if isinstance(coords, (list, tuple)) and len(coords) == 3:
                            mni_coords = [float(c) for c in coords]
                            break
                    except:
                        pass
                        
            # 判断是否为特殊电极
            has_cp = any(is_special_match(lbl, 'colorpatch') for lbl in aal_labels)
            has_cws = any(is_special_match(lbl, 'colorwithsti') for lbl in aal_labels)
            
            is_special = has_cp or has_cws
            
            # 解析唯一非特殊脑区作为标准脑区
            remain_labels = [lbl for lbl in aal_labels if lbl != '' and not is_special_match(lbl, 'colorpatch') and not is_special_match(lbl, 'colorwithsti')]
            resolved_aal = remain_labels[0] if remain_labels else ''
            
            roi = get_roi_category(resolved_aal)
            in_target = roi is not None
            
            elec_info_map[ch_str] = {
                'AAL3': resolved_aal,
                'AAL3_ROI': roi if roi else '',
                'In_Target_ROI': in_target,
                'MNI': mni_coords,
                'is_special': is_special,
                'is_color_patch': has_cp,
                'is_color_with_sti': has_cws
            }
            
        # 2. 读取 Task1 ERP 数据
        erp_path = os.path.join(feature_dir, subj, 'task1_ERP_epoched.mat')
        if not os.path.exists(erp_path):
            print(f"  [WARNING] {subj} Task1 ERP 数据不存在")
            continue
        erp_mat = read_mat(erp_path)
        erp_epoch = erp_mat['epoch']
        erp_data = erp_epoch['data']  # (Cond, Rep, Ch, Time)
        erp_ch_labels = [str(x).strip() for x in erp_epoch['ch']['labels']]
        erp_time_ms = erp_epoch['time_ms'] if 'time_ms' in erp_epoch else np.linspace(-500, 998, erp_data.shape[-1])
        n_cond_erp = erp_data.shape[0]
        
        # 3. 读取 Task1 HG 数据
        hg_path = os.path.join(feature_dir, subj, 'task1_hg_subband.mat')
        if not os.path.exists(hg_path):
            print(f"  [WARNING] {subj} Task1 HG 数据不存在")
            continue
        hg_mat = read_mat(hg_path)
        hg_epoch = hg_mat['epoch']
        hg_data_cell = hg_epoch['data_cell']  # List of arrays, shape (Rep, Ch, Time)
        hg_ch_labels = [str(x).strip() for x in hg_epoch['ch']['labels']]
        hg_time_ms = hg_epoch['time_ms']
        n_cond_hg = len(hg_data_cell)
        
        # 4. 逐通道进行 4 策略检验 (主分析仅含非特殊电极; color_patch 分支计算 color_patch 电极)
        all_channels_union = list(set(erp_ch_labels) | set(hg_ch_labels))
        
        for elec in all_channels_union:
            info = elec_info_map.get(elec, {
                'AAL3': '', 'AAL3_ROI': '', 'In_Target_ROI': False,
                'MNI': [np.nan, np.nan, np.nan], 'is_special': False,
                'is_color_patch': False, 'is_color_with_sti': False
            })
            
            # --- ERP 策略计算 ---
            s1, s2, s3, s4 = False, False, False, False
            if elec in erp_ch_labels:
                ch_idx = erp_ch_labels.index(elec)
                c_indices = [i for i in [0, 2, 4, 6] if i < n_cond_erp]
                g_indices = [i for i in [1, 3, 5, 7] if i < n_cond_erp]
                c_merged = np.concatenate([erp_data[idx, :, ch_idx, :] for idx in c_indices], axis=0)
                g_merged = np.concatenate([erp_data[idx, :, ch_idx, :] for idx in g_indices], axis=0)
                c_merged = c_merged[~np.isnan(c_merged).any(axis=1)]
                g_merged = g_merged[~np.isnan(g_merged).any(axis=1)]
                
                c_singles, g_singles = [], []
                for c_i, g_i in cond_pairs:
                    if c_i < n_cond_erp and g_i < n_cond_erp:
                        c_s = erp_data[c_i, :, ch_idx, :]
                        g_s = erp_data[g_i, :, ch_idx, :]
                        c_singles.append(c_s[~np.isnan(c_s).any(axis=1)])
                        g_singles.append(g_s[~np.isnan(g_s).any(axis=1)])
                
                if c_merged.shape[0] >= 3 and g_merged.shape[0] >= 3:
                    s1, s2, s3, s4 = check_channel_strategies(c_merged, g_merged, c_singles, g_singles, erp_time_ms)
            
            # --- HG 策略计算 ---
            h1, h2, h3, h4 = False, False, False, False
            if elec in hg_ch_labels:
                ch_idx = hg_ch_labels.index(elec)
                c_indices = [i for i in [0, 2, 4, 6] if i < n_cond_hg]
                g_indices = [i for i in [1, 3, 5, 7] if i < n_cond_hg]
                c_merged = np.concatenate([hg_data_cell[idx][:, ch_idx, :] for idx in c_indices], axis=0)
                g_merged = np.concatenate([hg_data_cell[idx][:, ch_idx, :] for idx in g_indices], axis=0)
                c_merged = c_merged[~np.isnan(c_merged).any(axis=1)]
                g_merged = g_merged[~np.isnan(g_merged).any(axis=1)]
                
                c_singles, g_singles = [], []
                for c_i, g_i in cond_pairs:
                    if c_i < n_cond_hg and g_i < n_cond_hg:
                        c_s = hg_data_cell[c_i][:, ch_idx, :]
                        g_s = hg_data_cell[g_i][:, ch_idx, :]
                        c_singles.append(c_s[~np.isnan(c_s).any(axis=1)])
                        g_singles.append(g_s[~np.isnan(g_s).any(axis=1)])
                
                if c_merged.shape[0] >= 3 and g_merged.shape[0] >= 3:
                    h1, h2, h3, h4 = check_channel_strategies(c_merged, g_merged, c_singles, g_singles, hg_time_ms)
            
            # A. 主分析过滤
            if not info['is_special']:
                # 填充 records_for_plot_main 用于重新绘制柱状图
                if elec in erp_ch_labels:
                    records_for_plot_main.append({
                        'Subject': subj, 'Signal': 'ERP', 'Electrode_ID': f"{subj}_{elec}",
                        'Strategy_1': s1, 'Strategy_2': s2, 'Strategy_3': s3, 'Strategy_4': s4,
                        'In_Target': info['In_Target_ROI']
                    })
                if elec in hg_ch_labels:
                    records_for_plot_main.append({
                        'Subject': subj, 'Signal': 'HG', 'Electrode_ID': f"{subj}_{elec}",
                        'Strategy_1': h1, 'Strategy_2': h2, 'Strategy_3': h3, 'Strategy_4': h4,
                        'In_Target': info['In_Target_ROI']
                    })
                    
                # 填充 ERP 的全脑电极表格记录
                if elec in erp_ch_labels:
                    strategies = []
                    if s1: strategies.append(1)
                    if s2: strategies.append(2)
                    if s3: strategies.append(3)
                    if s4: strategies.append(4)
                    strategies_str = ','.join(map(str, strategies)) if strategies else 'None'
                    any_sig = len(strategies) > 0
                    
                    all_erp_records.append({
                        'Subject': subj,
                        'Electrode': elec,
                        'AAL3': info['AAL3'],
                        'AAL3_ROI': info['AAL3_ROI'],
                        'In_Target_ROI': info['In_Target_ROI'],
                        'MNI_X': info['MNI'][0],
                        'MNI_Y': info['MNI'][1],
                        'MNI_Z': info['MNI'][2],
                        'ERP_Strategy_1': s1,
                        'ERP_Strategy_2': s2,
                        'ERP_Strategy_3': s3,
                        'ERP_Strategy_4': s4,
                        'ERP_Any_Sig': any_sig,
                        'ERP_Strategies_Matched': strategies_str
                    })
                    
                    # 缓存 ERP 显著电极用于玻璃脑绘图
                    if any_sig and not np.isnan(info['MNI'][0]):
                        best_strat = strategies[0]
                        item = {'coords': info['MNI'], 'strategy': best_strat, 'subject': subj, 'electrode': elec}
                        if info['In_Target_ROI']:
                            glass_target_erp.append(item)
                        else:
                            glass_outside_erp.append(item)
                            
            # B. Color patch 电极单独分析 (必须包含在 color_patch 中，用 ERP 数据)
            if info['is_color_patch']:
                if elec in erp_ch_labels:
                    strategies = []
                    if s1: strategies.append(1)
                    if s2: strategies.append(2)
                    if s3: strategies.append(3)
                    if s4: strategies.append(4)
                    strategies_str = ','.join(map(str, strategies)) if strategies else 'None'
                    best_strat = strategies[0] if strategies else 0
                    
                    color_patch_results.append({
                        'Subject': subj,
                        'Electrode': elec,
                        'MNI_X': info['MNI'][0],
                        'MNI_Y': info['MNI'][1],
                        'MNI_Z': info['MNI'][2],
                        'Strategy_1': s1,
                        'Strategy_2': s2,
                        'Strategy_3': s3,
                        'Strategy_4': s4,
                        'Strategies_Matched': strategies_str,
                        'Best_Strategy': best_strat
                    })
                    
                    if not np.isnan(info['MNI'][0]):
                        glass_color_patch.append({
                            'coords': info['MNI'],
                            'strategy': best_strat,
                            'subject': subj,
                            'electrode': elec,
                            'strategies': strategies_str
                        })
                        
    # ===================== 保存主分析 ERP 表格 =====================
    df_all_erp = pd.DataFrame(all_erp_records)
    n_total = len(df_all_erp)
    n_sig = df_all_erp['ERP_Any_Sig'].sum()
    n_target_sig = df_all_erp[df_all_erp['In_Target_ROI'] & df_all_erp['ERP_Any_Sig']].shape[0]
    n_outside_sig = df_all_erp[~df_all_erp['In_Target_ROI'] & df_all_erp['ERP_Any_Sig']].shape[0]
    
    print(f"\n{'='*55}")
    print("【主分析结果统计】(已排除 color_patch & color_with_sti)")
    print(f"全脑 ERP 电极总数: {n_total}")
    print(f"至少通过一种策略: {n_sig} ({n_sig/n_total*100:.1f}%)")
    print(f"  靶区内显著: {n_target_sig}")
    print(f"  靶区外显著: {n_outside_sig}")
    for s in range(1, 5):
        n_s = df_all_erp[f'ERP_Strategy_{s}'].sum()
        print(f"  Strategy {s}: {n_s}")
    print(f"{'='*55}")
    
    xlsx_path = os.path.join(doc_dir, 'whole_brain_erp_strategy_summary.xlsx')
    csv_path  = os.path.join(doc_dir, 'whole_brain_erp_strategy_summary.csv')
    df_all_erp.to_excel(xlsx_path, index=False)
    df_all_erp.to_csv(csv_path, index=False)
    print(f"[SAVED] ERP 策略表格: {xlsx_path}")
    
    # ===================== 绘制全脑 ERP 玻璃脑 =====================
    strategy_colors = {
        1: '#2ca02c',  # 绿
        2: '#1f77b4',  # 蓝
        3: '#9467bd',  # 紫
        4: '#ff7f0e'   # 橙
    }
    
    from nilearn import plotting
    
    # Group 全脑 ERP 玻璃脑
    fig = plt.figure(figsize=(16, 11))
    display = plotting.plot_glass_brain(None, display_mode='ortho', figure=fig, 
                                         title='Whole Brain ERP Color-Selective Electrodes (Excluding special electrodse)\n(Solid: In Target ROI, Hollow: Outside Target ROI)')
    
    if glass_target_erp:
        coords_t = np.array([item['coords'] for item in glass_target_erp])
        colors_t = [strategy_colors[item['strategy']] for item in glass_target_erp]
        display.add_markers(marker_coords=coords_t, marker_color=colors_t, marker_size=130, marker='o', alpha=0.92)
    if glass_outside_erp:
        coords_o = np.array([item['coords'] for item in glass_outside_erp])
        colors_o = [strategy_colors[item['strategy']] for item in glass_outside_erp]
        display.add_markers(marker_coords=coords_o, marker_color='none', edgecolors=colors_o, marker_size=130, linewidths=2.5, marker='o', alpha=0.92)
        
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=11, label='Strategy 1 (Merged Mean)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=11, label='Strategy 2 (Merged Cont 50ms)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9467bd', markersize=11, label='Strategy 3 (Single Mean)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=11, label='Strategy 4 (Single Cont 50ms)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=11, markeredgecolor='gray', label='In Target ROI (Solid)'),
        Line2D([0], [0], marker='o', color='gray', markerfacecolor='none', markeredgewidth=2, markersize=11, label='Outside Target ROI (Hollow)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.04), frameon=True, fontsize=10)
    
    out_glass_path = os.path.join(out_dir, 'whole_brain_erp_glass_brain.png')
    plt.savefig(out_glass_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[FIGURE] Saved whole brain ERP glass brain to: {out_glass_path}")
    
    # Subj 全脑 ERP 玻璃脑
    for subj in subjects:
        fig_s = plt.figure(figsize=(16, 11))
        display_s = plotting.plot_glass_brain(None, display_mode='ortho', figure=fig_s, 
                                               title=f'Whole Brain ERP Color-Selective Electrodes - {subj}\n(Solid: In Target ROI, Hollow: Outside Target ROI)')
        target_s = [item for item in glass_target_erp if item['subject'] == subj]
        outside_s = [item for item in glass_outside_erp if item['subject'] == subj]
        
        if target_s:
            coords_ts = np.array([item['coords'] for item in target_s])
            colors_ts = [strategy_colors[item['strategy']] for item in target_s]
            display_s.add_markers(marker_coords=coords_ts, marker_color=colors_ts, marker_size=130, marker='o', alpha=0.92)
        if outside_s:
            coords_os = np.array([item['coords'] for item in outside_s])
            colors_os = [strategy_colors[item['strategy']] for item in outside_s]
            display_s.add_markers(marker_coords=coords_os, marker_color='none', edgecolors=colors_os, marker_size=130, linewidths=2.5, marker='o', alpha=0.92)
            
        fig_s.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.04), frameon=True, fontsize=10)
        out_subj = os.path.join(out_dir, f'{subj}_whole_brain_erp_glass_brain.png')
        plt.savefig(out_subj, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[FIGURE] Saved {subj} ERP glass brain to: {out_subj}")
        
    # ===================== 绘制主要电极柱状对比图 =====================
    df_plot_main = pd.DataFrame(records_for_plot_main)
    # Group 柱状图
    plot_electrode_selection_comparison(df_plot_main, "Group (5 Subjects)", os.path.join(out_dir, 'electrode_selection_comparison.png'))
    # Subj 柱状图
    for subj in subjects:
        plot_electrode_selection_comparison(df_plot_main[df_plot_main['Subject'] == subj], subj, os.path.join(out_dir, f'{subj}_electrode_selection_comparison.png'))
        
    # ===================== 保存 & 绘制 color_patch 玻璃脑 =====================
    if color_patch_results:
        df_cp = pd.DataFrame(color_patch_results)
        cp_xlsx = os.path.join(doc_dir, 'color_patch_erp_strategy_check.xlsx')
        cp_csv = os.path.join(doc_dir, 'color_patch_erp_strategy_check.csv')
        df_cp.to_excel(cp_xlsx, index=False)
        df_cp.to_csv(cp_csv, index=False)
        print(f"[SAVED] color_patch ERP 策略表格: {cp_xlsx}")
        
        # 绘制 Group color_patch 玻璃脑
        fig_cp = plt.figure(figsize=(16, 10))
        display_cp = plotting.plot_glass_brain(None, display_mode='ortho', figure=fig_cp, 
                                               title='Color Patch ERP Electrodes - Strategy Color Mapping')
        
        strategy_colors_cp = {
            1: '#2ca02c',  # 绿
            2: '#1f77b4',  # 蓝
            3: '#9467bd',  # 紫
            4: '#ff7f0e',  # 橙
            0: '#888888'   # 未通过
        }
        
        if glass_color_patch:
            coords_cp = np.array([item['coords'] for item in glass_color_patch])
            colors_cp = [strategy_colors_cp[item['strategy']] for item in glass_color_patch]
            display_cp.add_markers(marker_coords=coords_cp, marker_color=colors_cp, marker_size=140, marker='o', alpha=0.92)
            
        legend_elements_cp = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=11, label='Strategy 1 (Merged Mean)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=11, label='Strategy 2 (Merged Cont 50ms)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#9467bd', markersize=11, label='Strategy 3 (Single Mean)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=11, label='Strategy 4 (Single Cont 50ms)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888', markersize=11, label='No Strategy Match (Gray)')
        ]
        fig_cp.legend(handles=legend_elements_cp, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.04), frameon=True, fontsize=10)
        
        out_cp_path = os.path.join(out_dir, 'color_patch_erp_glass_brain.png')
        plt.savefig(out_cp_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[FIGURE] Saved color_patch glass brain to: {out_cp_path}")
        
        # 顺便绘制被试各自的 color_patch 玻璃脑
        for subj in subjects:
            subj_cp = [item for item in glass_color_patch if item['subject'] == subj]
            if not subj_cp:
                continue
            fig_cps = plt.figure(figsize=(16, 10))
            display_cps = plotting.plot_glass_brain(None, display_mode='ortho', figure=fig_cps, 
                                                   title=f'Color Patch ERP Electrodes - {subj} - Strategy Color Mapping')
            coords_cps = np.array([item['coords'] for item in subj_cp])
            colors_cps = [strategy_colors_cp[item['strategy']] for item in subj_cp]
            display_cps.add_markers(marker_coords=coords_cps, marker_color=colors_cps, marker_size=140, marker='o', alpha=0.92)
            fig_cps.legend(handles=legend_elements_cp, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.04), frameon=True, fontsize=10)
            
            out_cps_path = os.path.join(out_dir, f'{subj}_color_patch_erp_glass_brain.png')
            plt.savefig(out_cps_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[FIGURE] Saved {subj} color_patch glass brain to: {out_cps_path}")
            
    print("\n" + "="*70)
    print("Step 8_2 修正数据表和可视化生成任务圆满完成！")
    print("="*70)

if __name__ == '__main__':
    main()
