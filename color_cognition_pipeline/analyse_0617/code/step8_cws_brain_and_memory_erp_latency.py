"""
Step 8 — Two analyses:
  8_1: 将 color_with_sti 电极绘制到 nilearn 2D 玻璃脑图上，
       颜色按它在纯色选择性 4 策略筛选中匹配到的最高优先级策略着色（全不符合则灰色）。
  8_2: 对 memory_color ERP 显著电极进行单电极红绿记忆颜色 ERP 信号差异分析，
       统计最早显著差异时间点 (ESTP)，并与 MNI_Y 进行相关性分析。
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
from scipy.stats import ranksums
from pymatreader import read_mat
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, re, ast, warnings, copy

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
out_cws    = os.path.join(result_dir, 'select_channel', 'decoding', 'color_with_sti')
out_single = os.path.join(result_dir, 'select_channel', 'decoding', 'single_electrode')
os.makedirs(out_cws, exist_ok=True)
os.makedirs(out_single, exist_ok=True)

subjects = ['test001', 'test002', 'test003']

# ===================== 策略筛选核心逻辑（复用 step1_1 的方法） =====================
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
    检测输入通道在 4 种统计策略下的显著情况。
    c/g_merged: 混合类别的 (Rep, Time)
    c/g_singles: list of (Rep, Time)，4 个单一类别
    """
    dt = time_ms[1] - time_ms[0]
    consec_pts = int(np.ceil(50.0 / dt))
    idx_100_400 = np.where((time_ms >= 100) & (time_ms <= 400))[0]
    idx_50_400  = np.where((time_ms >=  50) & (time_ms <= 400))[0]
    
    # 策略1: 混合类别 100-400ms 平均值显著
    c_m = np.mean(c_merged[:, idx_100_400], axis=1)
    g_m = np.mean(g_merged[:, idx_100_400], axis=1)
    _, p_s1 = ranksums(c_m, g_m)
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
        c_s_m = np.mean(c_s[:, idx_100_400], axis=1)
        g_s_m = np.mean(g_s[:, idx_100_400], axis=1)
        _, p_s = ranksums(c_s_m, g_s_m)
        if p_s < 0.05:
            s3 = True; break
            
    # 策略4: 至少一个单一类别在 50-400ms 下点对点连续显著 >= 50ms
    s4 = False
    for c_s, g_s in zip(c_singles, g_singles):
        sig_s = []
        for t_idx in idx_50_400:
            _, p_t = ranksums(c_s[:, t_idx], g_s[:, t_idx])
            sig_s.append(p_t < 0.05)
        if has_continuous_sig(sig_s, consec_pts):
            s4 = True; break
    return s1, s2, s3, s4

# ===================== 辅助函数 =====================
def get_color_with_sti_electrodes():
    subj_elecs = {}
    for subj in subjects:
        p = os.path.join(base_dir, 'processed_data', subj, f'{subj}_ieegloc.xlsx')
        if not os.path.exists(p):
            subj_elecs[subj] = []
            continue
        df_loc = pd.read_excel(p)
        aal_col = 'AAL3 (MNI-linear)' if 'AAL3 (MNI-linear)' in df_loc.columns else 'AAL3 (MNI-segment)'
        is_cs = df_loc[aal_col].astype(str).str.lower().str.replace('-','_').str.replace(' ','_') == 'color_with_sti'
        elecs = df_loc[is_cs]['Channel'].tolist()
        seen = set(); unique = []
        for e in elecs:
            ec = str(e).strip()
            if ec not in seen: seen.add(ec); unique.append(ec)
        subj_elecs[subj] = unique
    return subj_elecs

def get_mni_coords(subj, elec):
    p = os.path.join(base_dir, 'processed_data', subj, f'{subj}_ieegloc.xlsx')
    df_loc = pd.read_excel(p)
    ch_col = 'Channel' if 'Channel' in df_loc.columns else df_loc.columns[0]
    row = df_loc[df_loc[ch_col].astype(str).str.strip() == elec]
    if row.empty:
        return None
    mni_val = row.iloc[0].get('MNI', None)
    if pd.notna(mni_val):
        try:
            coords = ast.literal_eval(str(mni_val))
            if isinstance(coords, (list, tuple)) and len(coords) == 3:
                return [float(c) for c in coords]
        except:
            pass
    return None

def clean_data(x):
    if x is None: return None
    return x[~np.isnan(x).any(axis=1)] if x.ndim == 2 else x[~np.isnan(x).any(axis=tuple(range(1, x.ndim)))]

# ===============================================================
#                   TASK 8_1: color_with_sti 脑图绘制
# ===============================================================
def run_task_8_1():
    print("="*70)
    print("Step 8_1: color_with_sti 电极策略检验 + 2D 玻璃脑图绘制")
    print("="*70)
    
    cws_elecs = get_color_with_sti_electrodes()
    
    # 对 Task3 的纯色 vs 灰色数据进行策略检验
    # Task3 triggers: 51-56, 3 categories × 2 (color, gray) = 6 条件
    # pairs: (0,1), (2,3), (4,5)
    cond_pairs_task3 = [(0,1), (2,3), (4,5)]
    
    strategy_colors = {
        1: '#2ca02c',  # 绿
        2: '#1f77b4',  # 蓝
        3: '#9467bd',  # 紫
        4: '#ff7f0e',  # 橙
        0: '#888888'   # 灰色 (不匹配任何策略)
    }
    
    glass_items = []  # [{'coords', 'strategy', 'subject', 'electrode', 'strategies_matched'}]
    all_results = []
    
    for subj in subjects:
        elecs = cws_elecs.get(subj, [])
        if not elecs:
            print(f"  {subj}: 无 color_with_sti 电极")
            continue
        
        # 加载 Task3 ERP 数据
        erp_path = os.path.join(feature_dir, subj, 'task3_ERP_epoched.mat')
        if not os.path.exists(erp_path):
            print(f"  [WARNING] {subj} Task3 ERP 文件不存在: {erp_path}")
            continue
        mat = read_mat(erp_path)
        epoch = mat['epoch']
        ch_names = list(epoch['ch']['labels'])
        time_ms = epoch['time_ms'] if 'time_ms' in epoch else np.linspace(-500, 998, epoch['data'].shape[-1])
        data = epoch['data']  # (Cond, Rep, Ch, Time)
        
        for elec in elecs:
            if elec not in ch_names:
                print(f"  [WARNING] {subj}-{elec} 不在 Task3 通道列表中")
                continue
            ch_idx = ch_names.index(elec)
            
            # 提取该通道的 Color/Gray merged 和 singles
            c_singles, g_singles = [], []
            for c_i, g_i in cond_pairs_task3:
                c_d = data[c_i, :, ch_idx, :]  # (Rep, Time)
                g_d = data[g_i, :, ch_idx, :]
                c_d = c_d[~np.isnan(c_d).any(axis=1)]
                g_d = g_d[~np.isnan(g_d).any(axis=1)]
                c_singles.append(c_d)
                g_singles.append(g_d)
            
            c_merged = np.concatenate(c_singles, axis=0)
            g_merged = np.concatenate(g_singles, axis=0)
            
            s1, s2, s3, s4 = check_channel_strategies(c_merged, g_merged, c_singles, g_singles, time_ms)
            
            strategies = []
            if s1: strategies.append(1)
            if s2: strategies.append(2)
            if s3: strategies.append(3)
            if s4: strategies.append(4)
            
            # 取最高优先级策略（1>2>3>4），如都无则0（灰色）
            best_strategy = strategies[0] if strategies else 0
            strategies_str = ','.join(map(str, strategies)) if strategies else 'None'
            
            coords = get_mni_coords(subj, elec)
            if coords is None:
                print(f"  [WARNING] {subj}-{elec} 无 MNI 坐标，跳过")
                continue
            
            glass_items.append({
                'coords': coords,
                'strategy': best_strategy,
                'subject': subj,
                'electrode': elec,
                'strategies': strategies_str
            })
            all_results.append({
                'Subject': subj,
                'Electrode': elec,
                'MNI_X': coords[0],
                'MNI_Y': coords[1],
                'MNI_Z': coords[2],
                'Strategy_1': s1,
                'Strategy_2': s2,
                'Strategy_3': s3,
                'Strategy_4': s4,
                'Strategies_Matched': strategies_str,
                'Best_Strategy': best_strategy
            })
            print(f"  {subj}-{elec}: S1={s1} S2={s2} S3={s3} S4={s4} → Best={best_strategy} ({strategies_str})")
    
    # 保存统计结果
    df_res = pd.DataFrame(all_results)
    df_res.to_excel(os.path.join(doc_dir, 'color_with_sti_strategy_check.xlsx'), index=False)
    df_res.to_csv(os.path.join(doc_dir, 'color_with_sti_strategy_check.csv'), index=False)
    print(f"\n[SAVED] 策略检验结果导出至 doc/ 目录")
    
    # ---- 绘制玻璃脑图 ----
    from nilearn import plotting
    
    fig = plt.figure(figsize=(16, 10))
    display = plotting.plot_glass_brain(None, display_mode='ortho', figure=fig, 
                                         title='color_with_sti Electrodes - Strategy Color Mapping')
    
    if glass_items:
        coords_arr = np.array([item['coords'] for item in glass_items])
        colors = [strategy_colors[item['strategy']] for item in glass_items]
        
        display.add_markers(
            marker_coords=coords_arr,
            marker_color=colors,
            marker_size=140,
            marker='o',
            alpha=0.92
        )
    
    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=11, label='Strategy 1 (Merged Mean)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=11, label='Strategy 2 (Merged Cont 50ms)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9467bd', markersize=11, label='Strategy 3 (Single Mean)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=11, label='Strategy 4 (Single Cont 50ms)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888', markersize=11, label='No Strategy Match (Gray)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, 0.04), frameon=True, fontsize=10)
    
    out_path = os.path.join(out_cws, 'color_with_sti_glass_brain_strategies.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[FIGURE] Saved glass brain to: {out_path}")

# ===============================================================
#       TASK 8_2: memory_color 单电极 ERP 信号差异 ESTP & 相关性
# ===============================================================
def run_task_8_2():
    print("\n" + "="*70)
    print("Step 8_2: memory_color 单电极 ERP 信号差异 & MNI_Y 相关性")
    print("="*70)
    
    # 读取 memory_color 显著电极
    sig_path = os.path.join(doc_dir, 'select_channel_memory_significance_erp.csv')
    if not os.path.exists(sig_path):
        print("[ERROR] 显著电极文件不存在！")
        return
    df_sig = pd.read_csv(sig_path)
    df_mc = df_sig[df_sig['Sig_Category'] != 'Non_Sig'].copy()
    print(f"Total memory-selective ERP electrodes: {len(df_mc)}")
    
    # Task2 灰色水果 trigger 映射
    r1_trigs = ['Trigger-In:123']  # 灰色草莓(红)
    r2_trigs = ['Trigger-In:133']  # 灰色西瓜(红)
    g1_trigs = ['Trigger-In:103']  # 灰色卷心菜(绿)
    g2_trigs = ['Trigger-In:113']  # 灰色猕猴桃(绿)
    
    def get_single_ch_erp(mat_path, trig_list, elec):
        """提取单电极的 ERP 数据 -> (Rep, Time)"""
        if not os.path.exists(mat_path):
            return None, None
        try:
            mat = read_mat(mat_path)
            epoch = mat['epoch']
            ch_names = list(epoch['ch']['labels'])
            time_ms = epoch['time_ms'] if 'time_ms' in epoch else np.linspace(-500, 998, epoch['data'].shape[-1])
            all_trigs = list(epoch['trigger'])
            if elec not in ch_names:
                return None, None
            ch_idx = ch_names.index(elec)
            idx_list = [all_trigs.index(t) for t in trig_list if t in all_trigs]
            if not idx_list:
                return None, None
            data_list = []
            for idx in idx_list:
                # (Cond, Rep, Ch, Time) -> 取 cond idx, 所有 rep, 单通道
                td = epoch['data'][idx, :, ch_idx, :]  # (Rep, Time)
                data_list.append(td)
            merged = np.concatenate(data_list, axis=0)
            # 基线校正
            bl_mask = time_ms < 0
            bl_idx = np.where(bl_mask)[0]
            if len(bl_idx) > 0:
                merged = merged - np.mean(merged[:, bl_idx], axis=1, keepdims=True)
            return merged, time_ms
        except Exception as e:
            print(f"  [ERROR] {elec}: {e}")
            return None, None
    
    results = []
    electrode_curves = {}
    time_ms_ref = None
    
    for idx, row in df_mc.iterrows():
        subj = str(row['Subject']).strip()
        elec = str(row['Electrode']).strip()
        mat_path = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
        
        # 提取红色记忆的灰色水果和绿色记忆的灰色水果
        d_r1, t_arr = get_single_ch_erp(mat_path, r1_trigs, elec)
        d_r2, _     = get_single_ch_erp(mat_path, r2_trigs, elec)
        d_g1, _     = get_single_ch_erp(mat_path, g1_trigs, elec)
        d_g2, _     = get_single_ch_erp(mat_path, g2_trigs, elec)
        
        if any(d is None for d in [d_r1, d_r2, d_g1, d_g2]):
            print(f"  [WARNING] {subj}-{elec}: 灰色水果数据缺失，跳过")
            continue
        if time_ms_ref is None:
            time_ms_ref = t_arr
        
        # 清理 NaN
        d_r1 = d_r1[~np.isnan(d_r1).any(axis=1)]
        d_r2 = d_r2[~np.isnan(d_r2).any(axis=1)]
        d_g1 = d_g1[~np.isnan(d_g1).any(axis=1)]
        d_g2 = d_g2[~np.isnan(d_g2).any(axis=1)]
        
        # 合并红记忆和绿记忆的灰色水果 ERP 信号
        red_erp   = np.concatenate([d_r1, d_r2], axis=0)    # (n_red, Time)
        green_erp = np.concatenate([d_g1, d_g2], axis=0)    # (n_green, Time)
        
        # 计算每个时间点的差异幅度（红-绿均值差的绝对值曲线）
        diff_curve = np.mean(red_erp, axis=0) - np.mean(green_erp, axis=0)
        electrode_curves[f"{subj}_{elec}"] = diff_curve
        
        # 逐时间点 Wilcoxon/ranksums 检验 红 vs 绿
        n_time = len(time_ms_ref)
        p_vals = np.ones(n_time)
        for t in range(n_time):
            _, p_vals[t] = ranksums(red_erp[:, t], green_erp[:, t])
        
        # 计算 ESTP：80ms 之后最早的连续显著 (p<0.05) 的第一个时间点
        # 使用"第一个连续显著段的起始点"
        search_idx = np.where(time_ms_ref >= 80)[0]
        sig_mask = p_vals[search_idx] < 0.05
        estp = np.nan
        
        # 找第一个显著的时间点
        for si, is_sig in enumerate(sig_mask):
            if is_sig:
                estp = time_ms_ref[search_idx[si]]
                break
        
        print(f"  {subj}-{elec} (Y={row['MNI_Y']:.1f}): ESTP={estp:.0f}ms, n_red={red_erp.shape[0]}, n_green={green_erp.shape[0]}")
        
        results.append({
            'Subject': subj,
            'Electrode': elec,
            'MNI_X': float(row['MNI_X']),
            'MNI_Y': float(row['MNI_Y']),
            'MNI_Z': float(row['MNI_Z']),
            'AAL3_ROI': str(row.get('AAL3_ROI', '')),
            'Strategies_Matched': str(row.get('Strategies_Matched', '')),
            'ESTP_ERP_Diff': estp,
            'N_Red': red_erp.shape[0],
            'N_Green': green_erp.shape[0]
        })
    
    df_estp = pd.DataFrame(results)
    df_estp.to_excel(os.path.join(doc_dir, 'memory_color_erp_signal_diff_estp.xlsx'), index=False)
    df_estp.to_csv(os.path.join(doc_dir, 'memory_color_erp_signal_diff_estp.csv'), index=False)
    print(f"\n[SAVED] ERP 信号差异 ESTP 结果导出至 doc/ 目录")
    
    # ---- 绘制组水平与被试水平大图 ----
    time_ms = time_ms_ref
    
    def plot_diff_and_correlation(df, curves, time_ms, panel_name, out_path):
        """
        1行2列：
        左图：所有电极的红-绿 ERP 差异曲线（按 MNI_Y coolwarm 着色），黑色粗线为平均。
        右图：ESTP vs MNI_Y 散点 + 相关性回归。
        """
        if df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, f"No memory-selective ERP electrodes\nfor {panel_name}", 
                    ha='center', va='center', fontsize=14, color='red')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            return
        
        fig, (ax_line, ax_scatter) = plt.subplots(1, 2, figsize=(19, 8), 
              gridspec_kw={'width_ratios': [1.3, 1]}, dpi=300)
        fig.suptitle(f"ERP Single-Electrode Memory Color Signal Difference & Latency: {panel_name}",
                     fontsize=15, fontweight='bold', y=0.97)
        
        y_vals_all = df['MNI_Y'].values
        y_min, y_max = y_vals_all.min(), y_vals_all.max()
        y_range = (y_max - y_min) if (y_max - y_min) > 0 else 1.0
        
        t_idx_plot = np.where((time_ms >= -200) & (time_ms <= 800))[0]
        time_plot = time_ms[t_idx_plot]
        
        # ---- 左图：差异曲线 ----
        all_curves = []
        for _, row in df.iterrows():
            key = f"{row['Subject']}_{row['Electrode']}"
            if key in curves:
                curve = curves[key][t_idx_plot]
                all_curves.append(curve)
                norm_y = (row['MNI_Y'] - y_min) / y_range
                color = plt.cm.coolwarm(norm_y)
                ax_line.plot(time_plot, curve, color=color, lw=1.2, alpha=0.55, zorder=2)
        
        if all_curves:
            mean_curve = np.mean(all_curves, axis=0)
            ax_line.plot(time_plot, mean_curve, color='black', lw=3.6, 
                        label='Average Difference (Red - Green)', zorder=3)
        
        ax_line.axhline(0, color='#757575', linestyle=':', lw=1.5, label='Zero Line (No Difference)')
        ax_line.axvline(0, color='gray', linestyle='-', lw=1.2)
        ax_line.axvline(80, color='#d62728', linestyle='-.', lw=1.2, alpha=0.7, label='Search Boundary (80ms)')
        
        ax_line.set_title("Single-Electrode ERP Difference Curves (Red - Green Memory)", 
                         fontsize=12.5, fontweight='bold')
        ax_line.set_xlabel("Time relative to stimulus onset (ms)", fontsize=10.5)
        ax_line.set_ylabel("Amplitude Difference (μV)", fontsize=10.5)
        ax_line.set_xlim([-200, 800])
        ax_line.grid(True, linestyle=':', alpha=0.45)
        ax_line.legend(loc='upper left', fontsize=9)
        ax_line.set_facecolor('#fafafa')
        
        # ---- 右图：ESTP vs MNI_Y 相关性 ----
        df_valid = df[~df['ESTP_ERP_Diff'].isna()]
        
        if len(df_valid) < 2:
            ax_scatter.text(0.5, 0.5, "Insufficient channels with\nsignificant latency >= 80ms\n(N < 2)", 
                           ha='center', va='center', fontsize=12.5, color='gray', fontweight='semibold')
            ax_scatter.set_title("Latency vs. Anterior-Posterior Location", fontsize=12.5, fontweight='bold')
            ax_scatter.grid(True, linestyle=':', alpha=0.3)
            ax_scatter.set_facecolor('#fafafa')
        else:
            mni_y = df_valid['MNI_Y'].values
            estp = df_valid['ESTP_ERP_Diff'].values
            
            colors_scatter = []
            for yc in mni_y:
                norm = (yc - y_min) / y_range
                colors_scatter.append(plt.cm.coolwarm(norm))
            
            ax_scatter.scatter(mni_y, estp, color=colors_scatter, s=70, edgecolor='#555555', lw=1.0, zorder=3)
            
            for _, rv in df_valid.iterrows():
                ax_scatter.annotate(rv['Electrode'], (rv['MNI_Y'], rv['ESTP_ERP_Diff']),
                                   xytext=(0, 6), textcoords='offset points', 
                                   ha='center', fontsize=8.5, alpha=0.85)
            
            s_r, s_p = stats.spearmanr(mni_y, estp)
            p_r, p_p = stats.pearsonr(mni_y, estp)
            
            slope, intercept, r_val, p_val, std_err = stats.linregress(mni_y, estp)
            x_fit = np.linspace(mni_y.min() - 3, mni_y.max() + 3, 100)
            y_fit = slope * x_fit + intercept
            ax_scatter.plot(x_fit, y_fit, color='#d62728', lw=2.2, label='Linear Trend Line', zorder=2)
            
            y_fit_err = std_err * x_fit
            ax_scatter.fill_between(x_fit, y_fit - np.abs(y_fit_err) * 2.0, y_fit + np.abs(y_fit_err) * 2.0,
                                   color='#d62728', alpha=0.1)
            
            def star(p):
                if p < 0.001: return "***"
                elif p < 0.01: return "**"
                elif p < 0.05: return "*"
                else: return "(n.s.)"
            
            text = (f"Active Channels N = {len(df_valid)}\n\n"
                    f"Spearman $r_s$ = {s_r:.3f}{star(s_p)}\n(p = {s_p:.2e})\n"
                    f"Pearson $r_p$ = {p_r:.3f}{star(p_p)}\n(p = {p_p:.2e})")
            ax_scatter.text(0.05, 0.95, text, transform=ax_scatter.transAxes, fontsize=10,
                           fontweight='semibold', verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.45', facecolor='#fbfbfb', alpha=0.9, edgecolor='#cccccc'))
            
            ax_scatter.set_title("ERP Diff Latency vs. Anterior-Posterior Location", fontsize=12.5, fontweight='bold')
            ax_scatter.set_xlabel("Electrode MNI Y Coordinate\n(Posterior/后脑 <-- 0 --> Anterior/前脑)", fontsize=10.5)
            ax_scatter.set_ylabel("Earliest Significant Difference Time (ms)", fontsize=10.5)
            ax_scatter.grid(True, linestyle=':', alpha=0.45)
            ax_scatter.set_facecolor('#fafafa')
            ax_scatter.legend(loc='lower right', fontsize=9.5)
        
        # 底部色标
        cax = fig.add_axes([0.38, 0.03, 0.28, 0.025])
        norm_obj = plt.Normalize(vmin=y_min, vmax=y_max)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_obj, cmap=plt.cm.coolwarm), cax=cax, orientation='horizontal')
        cb.set_label('Electrode Position (MNI Y)', fontsize=9)
        cb.ax.tick_params(labelsize=8)
        
        plt.tight_layout(rect=[0, 0.07, 1, 0.94])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  [FIGURE] Saved to: {out_path}")
    
    # 组水平大图
    plot_diff_and_correlation(df_estp, electrode_curves, time_ms, "Group",
                              os.path.join(out_single, 'erp_group_memory_erp_diff_estp.png'))
    
    # 各被试个体大图
    for subj in subjects:
        df_subj = df_estp[df_estp['Subject'] == subj]
        if df_subj.empty:
            continue
        plot_diff_and_correlation(df_subj, electrode_curves, time_ms, subj,
                                  os.path.join(out_single, f'erp_{subj}_memory_erp_diff_estp.png'))

# ===================== 主入口 =====================
if __name__ == '__main__':
    run_task_8_1()
    run_task_8_2()
    print("\n" + "="*70)
    print("Step 8 完成！所有图表和数据已导出。")
    print("="*70)
