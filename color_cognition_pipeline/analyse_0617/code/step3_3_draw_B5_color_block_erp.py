"""
Step 3_3 (新增):
  绘制 test001 被试 B5 电极在 Task 3 红/绿纯色色块刺激下的 ERP 信号图。
  画图风格和排版布局完全对齐 test001_B5_ERP_Memory_Significance.png。
  包含:
    - 左子图: 时程响应曲线 (-200 到 800ms)，带 SEM 阴影，100-400ms 黄金色背景，底部显著点标记 (Red vs Green Wilcoxon, p < 0.05)。
    - 右子图: 100-400ms 窗口的均值对比条形图 (带误差线、显著性联结线及 p 值)，底部放置统计详细信息文本框。
"""
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import ranksums
from pymatreader import read_mat
import matplotlib.pyplot as plt
import os, warnings

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 路径 =====================
base_dir   = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline   = os.path.join(base_dir, 'color_cognition_pipeline')
analyse    = os.path.join(pipeline, 'analyse_0617')
feature_dir= os.path.join(analyse, 'feature')
doc_dir    = os.path.join(analyse, 'doc')
result_dir = os.path.join(analyse, 'result')
out_dir    = os.path.join(result_dir, 'select_channel', 'color_block_sig')
os.makedirs(out_dir, exist_ok=True)

def main():
    print("="*65)
    print("Step 3_3: Plotting test001 - B5 ERP Color Block Significance")
    print("="*65)
    
    subj = 'test001'
    elec = 'B5'
    
    # 1. 载入先前筛选电极表以获取 AAL3 脑区和策略信息
    summary_path = os.path.join(doc_dir, 'select_channel_summary.xlsx')
    roi = "Fusiform R"  # 默认备用值
    strat_matched = "1,2,3,4"  # 默认备用值
    
    if os.path.exists(summary_path):
        df_summary = pd.read_excel(summary_path)
        row = df_summary[(df_summary['Subject'] == subj) & (df_summary['Electrode'] == elec)]
        if not row.empty:
            roi = str(row.iloc[0].get('AAL3_ROI', 'Fusiform R'))
            strat_matched = str(row.iloc[0].get('ERP_Strategies_Matched', '1,2,3,4'))
            print(f"Loaded info from summary sheet: ROI={roi}, Strategies={strat_matched}")
            
    # 2. 读取 task3 ERP 数据
    mat_path = os.path.join(feature_dir, subj, 'task3_ERP_epoched.mat')
    if not os.path.exists(mat_path):
        print(f"[ERROR] Task3 mat 文件不存在: {mat_path}")
        return
        
    mat = read_mat(mat_path)
    epoch = mat['epoch']
    trigs = list(epoch['trigger'])
    ch_labels = [str(x).strip() for x in epoch['ch']['labels']]
    
    if elec not in ch_labels:
        print(f"[ERROR] 通道 {elec} 不在数据通道中！")
        return
    ch_idx = ch_labels.index(elec)
    
    time_ms = epoch['time_ms'] if 'time_ms' in epoch else np.linspace(-500, 998, epoch['data'].shape[-1])
    dt = time_ms[1] - time_ms[0]
    data = epoch['data']  # (Cond, Rep, Ch, Time)
    
    # 提取 Red(51) 和 Green(54)
    if 'Trigger-In:51' not in trigs or 'Trigger-In:54' not in trigs:
        print(f"[ERROR] 找不到 Red (51) 或 Green (54) 的 Trigger！")
        return
    idx_r = trigs.index('Trigger-In:51')
    idx_g = trigs.index('Trigger-In:54')
    
    data_r = data[idx_r, :, ch_idx, :]
    data_g = data[idx_g, :, ch_idx, :]
    
    # 过滤 NaN trials
    data_r = data_r[~np.isnan(data_r).any(axis=1)]
    data_g = data_g[~np.isnan(data_g).any(axis=1)]
    
    # 基线校正 (减去 time < 0 的均值)
    bl_mask = time_ms < 0
    bl_idx = np.where(bl_mask)[0]
    if len(bl_idx) > 0:
        data_r = data_r - np.mean(data_r[:, bl_idx], axis=1, keepdims=True)
        data_g = data_g - np.mean(data_g[:, bl_idx], axis=1, keepdims=True)
        
    # 3. 统计计算
    # 100-400ms 均值检验
    t_idx_100_400 = np.where((time_ms >= 100) & (time_ms <= 400))[0]
    mean_r_win = np.nanmean(data_r[:, t_idx_100_400], axis=1)
    mean_g_win = np.nanmean(data_g[:, t_idx_100_400], axis=1)
    
    _, p_mean = ranksums(mean_r_win, mean_g_win)
    is_mean_sig = p_mean < 0.05
    
    # 100-400ms 逐点检验，计算最长连续显著时长
    p_times_100_400 = []
    for t in t_idx_100_400:
        _, p_t = ranksums(data_r[:, t], data_g[:, t])
        p_times_100_400.append(p_t)
    p_times_100_400 = np.array(p_times_100_400)
    
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
    
    # 4. 绘图 (对齐原有风格)
    fig, (ax_time, ax_bar) = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={'width_ratios': [2.2, 1]}, dpi=150)
    fig.suptitle(f"ERP Color Block Selectivity: {subj} - {elec} ({roi})\nOriginal Selection Strategies: {strat_matched}", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # ---- A. 左子图: 时程曲线 ----
    t_idx_plot = np.where((time_ms >= -200) & (time_ms <= 800))[0]
    time_plot = time_ms[t_idx_plot]
    r_plot = data_r[:, t_idx_plot]
    g_plot = data_g[:, t_idx_plot]
    
    mean_r_plot = np.mean(r_plot, axis=0)
    sem_r_plot = np.std(r_plot, axis=0) / np.sqrt(r_plot.shape[0])
    
    mean_g_plot = np.mean(g_plot, axis=0)
    sem_g_plot = np.std(g_plot, axis=0) / np.sqrt(g_plot.shape[0])
    
    ax_time.plot(time_plot, mean_r_plot, color='#d62728', lw=2.2, label='Red Color Block (Trigger-In:51)')
    ax_time.fill_between(time_plot, mean_r_plot - sem_r_plot, mean_r_plot + sem_r_plot, color='#d62728', alpha=0.15)
    
    ax_time.plot(time_plot, mean_g_plot, color='#2ca02c', lw=2.2, label='Green Color Block (Trigger-In:54)')
    ax_time.fill_between(time_plot, mean_g_plot - sem_g_plot, mean_g_plot + sem_g_plot, color='#2ca02c', alpha=0.15)
    
    ax_time.axvspan(100, 400, color='#ffffcc', alpha=0.25, label='Window of Interest (100-400ms)')
    ax_time.axvline(0, color='gray', linestyle='--', lw=1.2)
    
    # 逐点检验并标示显著时间点
    ymin, ymax = ax_time.get_ylim()
    y_range = ymax - ymin
    sig_y = ymin + y_range * 0.03
    
    sig_red_x, sig_green_x = [], []
    for idx_t in t_idx_plot:
        t_val = time_ms[idx_t]
        _, p_val = ranksums(data_r[:, idx_t], data_g[:, idx_t])
        if p_val < 0.05:
            # 区分哪种响应强
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
    ax_time.set_ylabel("Amplitude (μV)", fontsize=10)
    ax_time.set_xlim([-200, 800])
    ax_time.grid(True, linestyle=':', alpha=0.5)
    ax_time.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    
    # ---- B. 右子图: 100-400ms 均值条形图 ----
    bar_m_r = np.mean(mean_r_win)
    bar_sem_r = np.std(mean_r_win) / np.sqrt(mean_r_win.shape[0])
    
    bar_m_g = np.mean(mean_g_win)
    bar_sem_g = np.std(mean_g_win) / np.sqrt(mean_g_win.shape[0])
    
    bars = ax_bar.bar(['Red Block', 'Green Block'], [bar_m_r, bar_m_g], 
                      yerr=[bar_sem_r, bar_sem_g], color=['#e57373', '#81c784'], 
                      edgecolor=['#d32f2f', '#388e3c'], capsize=7, width=0.55, error_kw={'elinewidth':1.8, 'ecolor':'#333333'})
                      
    max_h = max(bar_m_r + bar_sem_r, bar_m_g + bar_sem_g)
    y_line = max_h + abs(max_h)*0.1 if max_h != 0 else 0.5
    h_tick = abs(y_line)*0.03
    
    ax_bar.plot([0, 0, 1, 1], [y_line - h_tick, y_line, y_line, y_line - h_tick], color='#444444', lw=1.2)
    p_star = f"p = {p_mean:.4f} *" if is_mean_sig else f"p = {p_mean:.4f} (n.s.)"
    if p_mean < 0.001:
        p_star = f"p = {p_mean:.2e} ***"
    elif p_mean < 0.01:
        p_star = f"p = {p_mean:.4f} **"
        
    ax_bar.text(0.5, y_line + h_tick*0.8, p_star, ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    
    info_text = (
        f"100-400ms Stats:\n"
        f"  - Mean Diff Sig: {is_mean_sig}\n"
        f"  - Max Cont Segment: {max_duration:.1f} ms\n"
        f"  - Cont >= 50ms Sig: {is_cont_sig}"
    )
    ax_bar.text(0.5, -0.25, info_text, transform=ax_bar.transAxes, ha='center', va='top', 
                fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#fafafa', alpha=0.9, edgecolor='#cccccc'))
                
    ax_bar.set_title("100-400ms Window Mean", fontsize=11.5, fontweight='bold')
    ax_bar.set_ylabel("Amplitude Mean (100-400ms)", fontsize=10)
    ax_bar.grid(True, linestyle=':', alpha=0.35, axis='y')
    
    bar_ymin, bar_ymax = ax_bar.get_ylim()
    ax_bar.set_ylim([bar_ymin, y_line + h_tick*4])
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    
    out_img = os.path.join(out_dir, f"{subj}_{elec}_ERP_Color_Block_Significance.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved color block significance curve to: {out_img}")
    
    print("\n" + "="*65)
    print("Step 3_3 绘图任务圆满完成！")
    print("="*65)

if __name__ == '__main__':
    main()
