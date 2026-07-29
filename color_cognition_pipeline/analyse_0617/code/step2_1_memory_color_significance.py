import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import ranksums
from pymatreader import read_mat
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import warnings

# 忽略绘图警告
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
analyse_dir = os.path.join(pipeline_dir, 'analyse_0617')
feature_dir = os.path.join(analyse_dir, 'feature')
doc_dir = os.path.join(analyse_dir, 'doc')
result_dir = os.path.join(analyse_dir, 'result')

# 记忆颜色触发器定义
red_memory_trigs = ['Trigger-In:121', 'Trigger-In:122', 'Trigger-In:123', 'Trigger-In:131', 'Trigger-In:132', 'Trigger-In:133']
green_memory_trigs = ['Trigger-In:101', 'Trigger-In:102', 'Trigger-In:103', 'Trigger-In:111', 'Trigger-In:112', 'Trigger-In:113']

subjects = ['test001', 'test002', 'test003']

# ----------------- 数据提取函数 -----------------
def extract_memory_trials(epoch, is_hg, trigs_to_extract, ch_idx):
    """
    自 epoch 数据结构中提取特定触发器的 trials
    """
    all_trigs = list(epoch['trigger'])
    # 查找匹配的触发器索引
    idx_list = [all_trigs.index(t) for t in trigs_to_extract if t in all_trigs]
    if not idx_list:
        return None
        
    data_list = []
    for idx in idx_list:
        if is_hg:
            # data_cell list of (Rep, Ch, Time) -> 取指定通道 -> (Rep, Time)
            trial_data = epoch['data_cell'][idx][:, ch_idx, :]
        else:
            # data shape: (Cond, Rep, Ch, Time) -> 取指定条件与通道 -> (Rep, Time)
            trial_data = epoch['data'][idx, :, ch_idx, :]
        data_list.append(trial_data)
        
    # 合并所有 trial 并剔除含有 NaN 的行
    merged_data = np.concatenate(data_list, axis=0)
    merged_data = merged_data[~np.isnan(merged_data).any(axis=1)]
    return merged_data

# ----------------- 显著性分析核心流程 -----------------
def run_memory_significance_analysis(is_hg=False):
    feature_label = "HG" if is_hg else "ERP"
    print(f"\n[INFO] Starting Memory Color Significance Analysis for {feature_label}...")
    
    # 1. 载入先前筛选的 75 个主要电极名单
    summary_path = os.path.join(doc_dir, 'select_channel_summary.xlsx')
    if not os.path.exists(summary_path):
        print(f"[ERROR] select_channel_summary.xlsx 不存在，请确保运行了 step1_1！")
        return None
        
    df_summary = pd.read_excel(summary_path)
    
    # 单通道图像保存目录
    single_out_dir = os.path.join(result_dir, 'select_channel', 'memory_color', f"{feature_label.lower()}_single")
    os.makedirs(single_out_dir, exist_ok=True)
    
    results = []
    
    # 缓存各被试的数据文件，避免重复读取
    subj_epochs = {}
    for subj in subjects:
        if is_hg:
            mat_path = os.path.join(feature_dir, subj, 'task2_hg_subband.mat')
        else:
            mat_path = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
            
        if os.path.exists(mat_path):
            try:
                mat = read_mat(mat_path)
                subj_epochs[subj] = mat['epoch']
            except Exception as e:
                print(f"  [ERROR] 读取被试 {subj} 的 mat 文件失败: {e}")
        else:
            print(f"  [WARNING] 被试 {subj} 的 mat 文件不存在: {mat_path}")
            
    # 遍历主要电极列表中的每一个通道
    for idx, row in df_summary.iterrows():
        subj = str(row['Subject']).strip()
        elec = str(row['Electrode']).strip()
        
        # 提取策略标签
        strat_matched = str(row[f'{feature_label}_Strategies_Matched']) if f'{feature_label}_Strategies_Matched' in df_summary.columns else ""
        
        if subj not in subj_epochs:
            continue
            
        epoch = subj_epochs[subj]
        ch_labels = list(epoch['ch']['labels'])
        
        if elec not in ch_labels:
            print(f"  [WARNING] 电极 {subj}-{elec} 未在 task2 {feature_label} 脑电通道中找到，跳过")
            continue
            
        ch_idx = ch_labels.index(elec)
        
        # 获取时间轴与步长
        if 'time_ms' in epoch:
            time_ms = epoch['time_ms']
        else:
            time_ms = np.linspace(-500, 998, epoch['data'].shape[-1] if 'data' in epoch else epoch['data_cell'][0].shape[-1])
            
        dt = time_ms[1] - time_ms[0]
        
        # A. 提取红、绿记忆颜色 trials
        data_r = extract_memory_trials(epoch, is_hg, red_memory_trigs, ch_idx)
        data_g = extract_memory_trials(epoch, is_hg, green_memory_trigs, ch_idx)
        
        if data_r is None or data_g is None or len(data_r) == 0 or len(data_g) == 0:
            print(f"  [WARNING] 电极 {subj}-{elec} 的 task2 红/绿记忆 trial 数据为空，跳过")
            continue
            
        # B. 100-400ms 窗口平均检验
        t_idx_100_400 = np.where((time_ms >= 100) & (time_ms <= 400))[0]
        mean_r = np.nanmean(data_r[:, t_idx_100_400], axis=1)
        mean_g = np.nanmean(data_g[:, t_idx_100_400], axis=1)
        
        _, p_mean = ranksums(mean_r, mean_g)
        is_mean_sig = p_mean < 0.05
        
        # C. 100-400ms 逐时间点时序检验
        p_times_100_400 = []
        for t in t_idx_100_400:
            _, p_t = ranksums(data_r[:, t], data_g[:, t])
            p_times_100_400.append(p_t)
        p_times_100_400 = np.array(p_times_100_400)
        
        # 计算 100-400ms 范围内的最长连续显著时长
        max_consecutive = 0
        current_consecutive = 0
        for p in p_times_100_400:
            if p < 0.05:
                current_consecutive += 1
                if current_consecutive > max_consecutive:
                    max_consecutive = current_consecutive
        else:
            current_consecutive = 0
            
        max_duration = max_consecutive * dt
        is_cont_sig = max_duration >= 50.0
        
        # D. 显著性综合分类 (Both, Mean_Only, Cont_Only, Non_Sig)
        if is_mean_sig and is_cont_sig:
            sig_category = "Both_Sig"
        elif is_mean_sig:
            sig_category = "Mean_Sig_Only"
        elif is_cont_sig:
            sig_category = "Cont_Sig_Only"
        else:
            sig_category = "Non_Sig"
            
        # 记录结果
        res_dict = {
            'Subject': subj,
            'Electrode': elec,
            'MNI_X': float(row['MNI_X']),
            'MNI_Y': float(row['MNI_Y']),
            'MNI_Z': float(row['MNI_Z']),
            'AAL3_ROI': str(row['AAL3_ROI']),
            'Strategies_Matched': strat_matched,
            'Mean_P': p_mean,
            'Is_Mean_Sig': is_mean_sig,
            'Max_Cont_Duration_ms': max_duration,
            'Is_Cont_Sig': is_cont_sig,
            'Sig_Category': sig_category
        }
        results.append(res_dict)
        
        # E. 绘制单电极的左右对比主图
        plot_single_electrode_memory_significance(
            data_r, data_g, time_ms, subj, elec, row['AAL3_ROI'], strat_matched,
            p_mean, is_mean_sig, max_duration, is_cont_sig, feature_label, single_out_dir
        )
        
    # 保存明细表数据
    df_res = pd.DataFrame(results)
    excel_path = os.path.join(doc_dir, f'select_channel_memory_significance_{feature_label.lower()}.xlsx')
    csv_path = os.path.join(doc_dir, f'select_channel_memory_significance_{feature_label.lower()}.csv')
    df_res.to_excel(excel_path, index=False)
    df_res.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Saved {feature_label} statistical tables to:\n  - {excel_path}\n  - {csv_path}")
    
    return df_res

# ----------------- 单电极对比图绘制 -----------------
def plot_single_electrode_memory_significance(
    data_r, data_g, time_ms, subj, elec, roi, strat_matched,
    p_mean, is_mean_sig, max_duration, is_cont_sig, feature_label, out_dir
):
    fig, (ax_time, ax_bar) = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={'width_ratios': [2.2, 1]}, dpi=150)
    fig.suptitle(f"{feature_label} Memory Color Selectivity: {subj} - {elec} ({roi})\nOriginal Selection Strategies: {strat_matched}", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # ---------------- 1. 左子图: 时程曲线 ----------------
    # 截取 [-200, 800] ms 绘图区间
    t_idx_plot = np.where((time_ms >= -200) & (time_ms <= 800))[0]
    time_plot = time_ms[t_idx_plot]
    r_plot = data_r[:, t_idx_plot]
    g_plot = data_g[:, t_idx_plot]
    
    mean_r_plot = np.mean(r_plot, axis=0)
    sem_r_plot = np.std(r_plot, axis=0) / np.sqrt(r_plot.shape[0])
    
    mean_g_plot = np.mean(g_plot, axis=0)
    sem_g_plot = np.std(g_plot, axis=0) / np.sqrt(g_plot.shape[0])
    
    # 绘制曲线和 SEM 阴影
    ax_time.plot(time_plot, mean_r_plot, color='#d62728', lw=2.2, label='Red Memory (Strawberry/Watermelon)')
    ax_time.fill_between(time_plot, mean_r_plot - sem_r_plot, mean_r_plot + sem_r_plot, color='#d62728', alpha=0.15)
    
    ax_time.plot(time_plot, mean_g_plot, color='#2ca02c', lw=2.2, label='Green Memory (Kiwi/Cabbage)')
    ax_time.fill_between(time_plot, mean_g_plot - sem_g_plot, mean_g_plot + sem_g_plot, color='#2ca02c', alpha=0.15)
    
    # 突出标记测试窗口 [100, 400] ms
    ax_time.axvspan(100, 400, color='#ffffcc', alpha=0.25, label='Window of Interest (100-400ms)')
    ax_time.axvline(0, color='gray', linestyle='--', lw=1.2)
    
    # 时程上逐点检验并标示显著时间点
    ymin, ymax = ax_time.get_ylim()
    y_range = ymax - ymin
    sig_y = ymin + y_range * 0.03 # 放在底端 3% 处
    
    sig_red_x, sig_green_x = [], []
    for idx_t in t_idx_plot:
        t_val = time_ms[idx_t]
        _, p_val = ranksums(data_r[:, idx_t], data_g[:, idx_t])
        if p_val < 0.05:
            # 区分强弱色
            if mean_r_plot[np.where(time_plot == t_val)[0][0]] > mean_g_plot[np.where(time_plot == t_val)[0][0]]:
                sig_red_x.append(t_val)
            else:
                sig_green_x.append(t_val)
                
    if sig_red_x:
        ax_time.scatter(sig_red_x, [sig_y]*len(sig_red_x), color='#d62728', marker='s', s=8, alpha=0.7, label='Red > Green (p < 0.05)')
    if sig_green_x:
        ax_time.scatter(sig_green_x, [sig_y]*len(sig_green_x), color='#2ca02c', marker='s', s=8, alpha=0.7, label='Green > Red (p < 0.05)')
        
    ax_time.set_title("Time Course Response", fontsize=11.5, fontweight='bold')
    ax_time.set_xlabel("Time (ms)", fontsize=10)
    ax_time.set_ylabel("Z-score / Amplitude", fontsize=10)
    ax_time.set_xlim([-200, 800])
    ax_time.grid(True, linestyle=':', alpha=0.5)
    ax_time.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    
    # ---------------- 2. 右子图: 100-400ms 均值差异 ----------------
    t_idx_100_400 = np.where((time_ms >= 100) & (time_ms <= 400))[0]
    mean_r_win = np.nanmean(data_r[:, t_idx_100_400], axis=1)
    mean_g_win = np.nanmean(data_g[:, t_idx_100_400], axis=1)
    
    bar_m_r = np.mean(mean_r_win)
    bar_sem_r = np.std(mean_r_win) / np.sqrt(mean_r_win.shape[0])
    
    bar_m_g = np.mean(mean_g_win)
    bar_sem_g = np.std(mean_g_win) / np.sqrt(mean_g_win.shape[0])
    
    bars = ax_bar.bar(['Red Memory', 'Green Memory'], [bar_m_r, bar_m_g], 
                      yerr=[bar_sem_r, bar_sem_g], color=['#e57373', '#81c784'], 
                      edgecolor=['#d32f2f', '#388e3c'], capsize=7, width=0.55, error_kw={'elinewidth':1.8, 'ecolor':'#333333'})
                      
    # 显著性星号标注
    # 确定最大高度
    max_h = max(bar_m_r + bar_sem_r, bar_m_g + bar_sem_g)
    y_line = max_h + abs(max_h)*0.1 if max_h != 0 else 0.5
    h_tick = abs(y_line)*0.03
    
    # 画联结线
    ax_bar.plot([0, 0, 1, 1], [y_line - h_tick, y_line, y_line, y_line - h_tick], color='#444444', lw=1.2)
    p_star = f"p = {p_mean:.4f} *" if is_mean_sig else f"p = {p_mean:.4f} (n.s.)"
    if p_mean < 0.001:
        p_star = f"p = {p_mean:.2e} ***"
    elif p_mean < 0.01:
        p_star = f"p = {p_mean:.4f} **"
        
    ax_bar.text(0.5, y_line + h_tick*0.8, p_star, ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    
    # 标注连续50ms的信息
    info_text = (
        f"100-400ms Stats:\n"
        f"  - Mean Diff Sig: {is_mean_sig}\n"
        f"  - Max Cont Segment: {max_duration:.1f} ms\n"
        f"  - Cont >= 50ms Sig: {is_cont_sig}"
    )
    ax_bar.text(0.5, -0.25, info_text, transform=ax_bar.transAxes, ha='center', va='top', 
                fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#fafafa', alpha=0.9, edgecolor='#cccccc'))
                
    ax_bar.set_title("100-400ms Window Mean", fontsize=11.5, fontweight='bold')
    ax_bar.set_ylabel("Amplitude Mean (50-400ms)", fontsize=10)
    ax_bar.grid(True, linestyle=':', alpha=0.35, axis='y')
    
    # 适当拉伸 Y 轴高度以容纳显著性标注
    bar_ymin, bar_ymax = ax_bar.get_ylim()
    ax_bar.set_ylim([bar_ymin, y_line + h_tick*4])
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    out_img = os.path.join(out_dir, f"{subj}_{elec}_{feature_label}_Memory_Significance.png")
    plt.savefig(out_img, dpi=150)
    plt.close()

# ----------------- 多饼图绘制 -----------------
def plot_strategies_pie_charts(df, is_hg=False):
    """
    绘制 1行4列 的多饼图，展现属于策略 1-4 的电极子集在记忆颜色中的显著分布比例
    """
    feature_label = "HG" if is_hg else "ERP"
    print(f"[INFO] Plotting selection strategies pie charts for {feature_label}...")
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.5), dpi=300)
    fig.suptitle(f"{feature_label} Memory-Selective Electrodes Distribution Across Initial Screening Strategies", 
                 fontsize=15, fontweight='bold', y=0.98)
                 
    # 过滤出在此特征下显著的电极判定 (Mean_Sig 或 Cont_Sig 中任一满足)
    # 计算显著性标志
    df['Is_Memory_Significant'] = df['Is_Mean_Sig'] | df['Is_Cont_Sig']
    
    # 策略配色
    colors = ['#ff8a80', '#cfd8dc']  # 显著(柔红)，不显著(浅灰蓝)
    
    for strategy_idx in range(1, 5):
        ax = axes[strategy_idx - 1]
        
        # 寻找原本匹配策略 strategy_idx 的子集
        # 匹配格式为逗号分隔的字符串，包含字符 '1' ~ '4'
        sub_df = df[df['Strategies_Matched'].astype(str).str.contains(str(strategy_idx))]
        
        if sub_df.empty:
            ax.text(0.5, 0.5, f"No Electrodes\nMatched Strategy {strategy_idx}", 
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(f"Strategy {strategy_idx}", fontsize=13, fontweight='bold')
            ax.axis('off')
            continue
            
        n_total = len(sub_df)
        n_sig = sub_df['Is_Memory_Significant'].sum()
        n_nonsig = n_total - n_sig
        
        sizes = [n_sig, n_nonsig]
        labels = ['Sig', 'Non-Sig']
        
        # 如果显著数量为0，只画一瓣以防画图报错
        if n_sig == 0:
            sizes = [n_total]
            labels = ['Non-Sig Only']
            pie_colors = ['#cfd8dc']
        else:
            pie_colors = colors
            
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%', startangle=140,
            colors=pie_colors, textprops=dict(fontsize=10.5, fontweight='semibold'),
            wedgeprops=dict(edgecolor='#9e9e9e', linewidth=1.2, alpha=0.9)
        )
        
        # 装饰
        plt.setp(autotexts, size=11, weight="bold")
        ax.set_title(f"Strategy {strategy_idx} Channels\n(N = {n_total})", fontsize=12.5, fontweight='bold', pad=8)
        
        # 底部添加数据摘要框
        summary_txt = f"Significant: {n_sig}\nNon-Significant: {n_nonsig}"
        ax.text(0.5, -0.15, summary_txt, transform=ax.transAxes, ha='center', va='top', 
                fontsize=10, bbox=dict(boxstyle='round,pad=0.35', facecolor='#ffffff', alpha=0.85, edgecolor='#dcdcdc'))
                
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    out_pie = os.path.join(result_dir, 'select_channel', 'memory_color', f"memory_color_strategy_pie_{feature_label.lower()}.png")
    plt.savefig(out_pie, dpi=300)
    plt.close()
    print(f"  Saved Pie Chart Figure to: {out_pie}")

# ----------------- Nilearn 2D 脑图绘制 -----------------
def plot_nilearn_glass_brain_memory(df, is_hg=False):
    """
    使用 Nilearn 库绘制 2D 玻璃脑电极分布，基于四分类标记染色：
    - Both_Sig (红色)
    - Mean_Sig_Only (橙黄色)
    - Cont_Sig_Only (蓝色)
    - Non_Sig (灰色)
    """
    from nilearn import plotting
    
    feature_label = "HG" if is_hg else "ERP"
    print(f"[INFO] Plotting Nilearn 2D glass brain for {feature_label}...")
    
    # 染色映射
    colors_map = {
        'Both_Sig': '#d62728',       # 🔴 红色 (两者皆显著)
        'Mean_Sig_Only': '#ff7f0e',  # 🟡 橙色/黄色 (仅均值显著)
        'Cont_Sig_Only': '#1f77b4',  # 🔵 蓝色 (仅连续显著)
        'Non_Sig': '#c0c0c0'         # ⚪ 灰色 (不显著主要电极)
    }
    
    fig = plt.figure(figsize=(15, 10))
    display = plotting.plot_glass_brain(None, display_mode='ortho', figure=fig, 
                                        title=f"{feature_label} Memory-Selective Channel Spatial Distribution in MNI Space")
                                        
    # 分类提取坐标并绘制
    # 为保证不显著的灰色点不遮挡显著点，我们需要先绘制灰色点，再绘制显著点 (调高 zorder)
    categories_order = ['Non_Sig', 'Cont_Sig_Only', 'Mean_Sig_Only', 'Both_Sig']
    
    for cat in categories_order:
        cat_df = df[df['Sig_Category'] == cat]
        if cat_df.empty:
            continue
            
        coords = cat_df[['MNI_X', 'MNI_Y', 'MNI_Z']].values
        color = colors_map[cat]
        
        # 尺寸与虚实心设计
        if cat == 'Non_Sig':
            marker_size = 70
            alpha_val = 0.55
        else:
            marker_size = 130
            alpha_val = 0.95
            
        if is_hg:
            # High Gamma: 空心圆 (利用 edgecolors，marker_color设为'none')
            if cat == 'Non_Sig':
                # 灰色通道用淡灰色空心
                display.add_markers(
                    marker_coords=coords,
                    marker_color='none',
                    edgecolors=color,
                    marker_size=marker_size,
                    linewidths=1.2,
                    marker='o',
                    alpha=0.6
                )
            else:
                display.add_markers(
                    marker_coords=coords,
                    marker_color='none',
                    edgecolors=color,
                    marker_size=marker_size,
                    linewidths=2.5,
                    marker='o',
                    alpha=alpha_val
                )
        else:
            # ERP: 实心圆
            display.add_markers(
                marker_coords=coords,
                marker_color=color,
                marker_size=marker_size,
                marker='o',
                alpha=alpha_val
            )
            
    # 自定义图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10, label='Both Significant (Mean & Cont >= 50ms)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=10, label='Mean Window Significant Only (100-400ms)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='Continuous Significant Only (>= 50ms)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#c0c0c0', markersize=8, label='Other Select Electrodes (Non-Significant)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='ERP Channels (Solid Circles)'),
        Line2D([0], [0], marker='o', color='gray', markerfacecolor='none', markeredgewidth=2, markersize=10, label='High Gamma Channels (Hollow Circles)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05), frameon=True, fontsize=10)
    
    out_brain = os.path.join(result_dir, 'select_channel', 'memory_color', f"memory_color_glass_brain_{feature_label.lower()}.png")
    os.makedirs(os.path.dirname(out_brain), exist_ok=True)
    plt.savefig(out_brain, dpi=300)
    plt.close()
    print(f"  Saved Nilearn 2D Brain Figure to: {out_brain}")

# ----------------- 主入口 -----------------
def run_all_analysis():
    print("="*70)
    print("Step 2_1: Running Memory Color Significance (Red vs Green) on Select Electrodes")
    print("="*70)
    
    # 1. 运行 ERP
    df_erp = run_memory_significance_analysis(is_hg=False)
    if df_erp is not None:
        plot_strategies_pie_charts(df_erp, is_hg=False)
        plot_nilearn_glass_brain_memory(df_erp, is_hg=False)
        
    # 2. 运行 HG
    df_hg = run_memory_significance_analysis(is_hg=True)
    if df_hg is not None:
        plot_strategies_pie_charts(df_hg, is_hg=True)
        plot_nilearn_glass_brain_memory(df_hg, is_hg=True)
        
    print("\n" + "="*70)
    print("Step 2_1 Memory Color Significance Analysis Successfully Completed!")
    print("="*70)

if __name__ == '__main__':
    run_all_analysis()
