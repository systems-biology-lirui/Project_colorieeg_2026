#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intracranial LFP/ERP color-gray comparison analysis.
Includes MNI coordinates nearest electrode search, 20ms sliding window significance test,
100-400ms mean value t-test, and premium visualization.
"""

import os
import shutil
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import ttest_ind
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 目标坐标
TARGET_ABS = np.array([-33.34, -80.21, 2.82])      # 绝对坐标
TARGET_SYM_R = np.array([33.34, -80.21, 2.82])    # 右半脑对称坐标

# 被试数据映射
SUBJECTS = ['test001', 'test002', 'test003', 'test006']
PROCESSED_DIR = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026/processed_data'
ARTIFACTS_DIR = '/home/lirui/.gemini/antigravity-ide/brain/1870c00b-e214-4064-b47f-91fb6f22337f'

# 调和配色
COLOR_MAP = {
    'color': '#e15759',  # 柔和红色
    'gray': '#4e79a7',   # 柔和蓝色/灰色
    'sig': '#f28e2b'     # 显著区间橘色
}

def parse_mni(val):
    if pd.isna(val):
        return None
    try:
        if isinstance(val, str):
            val = val.strip()
            if val.startswith('[') and val.endswith(']'):
                val = val[1:-1]
            parts = val.split(',')
            if len(parts) == 3:
                return np.array([float(x) for x in parts])
    except Exception:
        pass
    return None

def get_electrode_loc(subject):
    xlsx_path = os.path.join(PROCESSED_DIR, subject, f'{subject}_ieegloc.xlsx')
    tsv_path = os.path.join(PROCESSED_DIR, subject, f'{subject}.tsv')
    
    if os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    elif os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep='\t')
    else:
        raise FileNotFoundError(f'Electrode location file not found for {subject}')
    
    electrodes = []
    for _, row in df.iterrows():
        coord = parse_mni(row['MNI'])
        if coord is not None:
            electrodes.append({
                'Channel': str(row['Channel']),
                'Coord': coord
            })
    return electrodes

def find_nearest_electrode(electrodes, target):
    best_ch = None
    best_dist = np.inf
    best_coord = None
    for item in electrodes:
        dist = np.linalg.norm(item['Coord'] - target)
        if dist < best_dist:
            best_dist = dist
            best_ch = item['Channel']
            best_coord = item['Coord']
    return best_ch, best_coord, best_dist

def extract_channel_data(subject, channel_name):
    mat_path = os.path.join(PROCESSED_DIR, subject, 'task1_ERP_epoched.mat')
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f'task1_ERP_epoched.mat not found for {subject}')
    
    mat = sio.loadmat(mat_path)
    epoch = mat['epoch']
    
    # epoch.data 形状: [Cond, Rep, Ch, Time]
    data = epoch['data'][0, 0]
    
    # 提取时间轴
    if 'time_ms' in epoch.dtype.names:
        time_ms = epoch['time_ms'][0, 0].flatten()
    else:
        time_len = data.shape[3]
        time_ms = np.arange(-500, -500 + time_len * 2, 2)
    
    # 提取通道列表
    ch_list = [str(x[0]) for x in epoch['ch'][0, 0]['labels'].flatten()]
    
    if channel_name not in ch_list:
        raise ValueError(f'Channel {channel_name} not found in the EEG epochs of {subject}')
    
    ch_idx = ch_list.index(channel_name)
    
    # 提取出目标通道的数据，形状为 [Cond, Rep, Time]
    ch_data = data[:, :, ch_idx, :]
    
    return ch_data, time_ms

def compute_sliding_window_p(color_trials, gray_trials, time_ms):
    """
    计算 10ms 间隔 20ms 滑窗差异的 p 值
    """
    t_min = time_ms.min()
    t_max = time_ms.max()
    
    # 窗口中心点从 t_min+10ms 到 t_max-10ms，步长 10ms
    centers = np.arange(t_min + 10, t_max - 10, 10)
    p_values = []
    
    for center in centers:
        # 获取落在 [center - 10, center + 10) ms 范围内的采样点掩码
        win_mask = (time_ms >= center - 10) & (time_ms < center + 10)
        
        # 将各 trial 在窗口内的时间点求平均，得到一维的 trial 数据
        col_win = color_trials[:, win_mask].mean(axis=1)
        gray_win = gray_trials[:, win_mask].mean(axis=1)
        
        # 独立双样本 t 检验
        stat, p = ttest_ind(col_win, gray_win, equal_var=False, nan_policy='omit')
        p_values.append(p)
        
    return centers, np.array(p_values)

def analyze_100_400ms(color_trials, gray_trials, time_ms):
    win_mask = (time_ms >= 100) & (time_ms <= 400)
    col_mean = color_trials[:, win_mask].mean(axis=1)
    gray_mean = gray_trials[:, win_mask].mean(axis=1)
    
    stat, p = ttest_ind(col_mean, gray_mean, equal_var=False, nan_policy='omit')
    
    return {
        'col_mean': float(np.nanmean(col_mean)),
        'col_sem': float(np.nanstd(col_mean, ddof=1) / np.sqrt(len(col_mean))),
        'gray_mean': float(np.nanmean(gray_mean)),
        'gray_sem': float(np.nanstd(gray_mean, ddof=1) / np.sqrt(len(gray_mean))),
        't_stat': float(stat),
        'p_val': float(p)
    }

def plot_erp_comparison(subject, ch_name, ch_data, time_ms, mode_name, best_dist):
    # 定义 5 个条件对比
    conditions = [
        {'name': 'Face (Color vs Gray)', 'color_idx': 0, 'gray_idx': 1},
        {'name': 'Object (Color vs Gray)', 'color_idx': 2, 'gray_idx': 3},
        {'name': 'Body (Color vs Gray)', 'color_idx': 4, 'gray_idx': 5},
        {'name': 'Place (Color vs Gray)', 'color_idx': 6, 'gray_idx': 7},
        {'name': 'All Combined (Color vs Gray)', 'color_idx': [0, 2, 4, 6], 'gray_idx': [1, 3, 5, 7]}
    ]
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 16), sharex=True)
    stats_results = []
    
    for i, cond in enumerate(conditions):
        ax = axes[i]
        
        # 提取 Color 与 Gray 的 trials
        if isinstance(cond['color_idx'], list):
            # Combined 条件：在 Rep (axis 0) 拼接所有的 trials
            color_trials = np.concatenate([ch_data[idx] for idx in cond['color_idx']], axis=0)
            gray_trials = np.concatenate([ch_data[idx] for idx in cond['gray_idx']], axis=0)
        else:
            color_trials = ch_data[cond['color_idx']]
            gray_trials = ch_data[cond['gray_idx']]
            
        # 均值与标准误
        col_mean = color_trials.mean(axis=0)
        col_sem = color_trials.std(axis=0, ddof=1) / np.sqrt(color_trials.shape[0])
        gray_mean = gray_trials.mean(axis=0)
        gray_sem = gray_trials.std(axis=0, ddof=1) / np.sqrt(gray_trials.shape[0])
        
        # 100-400ms 平均值显著性
        win_stats = analyze_100_400ms(color_trials, gray_trials, time_ms)
        win_stats['condition'] = cond['name'].split(' ')[0]
        stats_results.append(win_stats)
        
        # 滑窗 p 值
        centers, p_vals = compute_sliding_window_p(color_trials, gray_trials, time_ms)
        
        # 绘图
        ax.plot(time_ms, col_mean, label='Color', color=COLOR_MAP['color'], linewidth=1.5)
        ax.fill_between(time_ms, col_mean - col_sem, col_mean + col_sem, color=COLOR_MAP['color'], alpha=0.15)
        
        ax.plot(time_ms, gray_mean, label='Gray', color=COLOR_MAP['gray'], linewidth=1.5)
        ax.fill_between(time_ms, gray_mean - gray_sem, gray_mean + gray_sem, color=COLOR_MAP['gray'], alpha=0.15)
        
        # 显著性滑窗区域标记 (p < 0.05)
        sig_mask = p_vals < 0.05
        # 绘制滑窗显著性背景
        for j, center in enumerate(centers):
            if sig_mask[j]:
                # 绘制宽度为 10ms 的条带覆盖该滑窗
                ax.axvspan(center - 5, center + 5, color=COLOR_MAP['sig'], alpha=0.1, ymin=0.02, ymax=0.08)
        
        # 细节美化
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
        # 虚线标出 100ms 和 400ms 时间段
        ax.axvline(100, color='purple', linestyle=':', linewidth=1.0, alpha=0.6)
        ax.axvline(400, color='purple', linestyle=':', linewidth=1.0, alpha=0.6)
        
        ax.set_title(f"{cond['name']} (100-400ms p = {win_stats['p_val']:.4f})", fontsize=11, fontweight='bold')
        ax.set_ylabel('Amplitude (μV)', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        if i == 0:
            ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
            
    axes[-1].set_xlabel('Time (ms)', fontsize=11)
    
    # 顶部总标题
    fig.suptitle(f"{subject} | Channel: {ch_name} | Mode: {mode_name}\n"
                 f"MNI Distance to Target: {best_dist:.2f} mm",
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # 保存图像
    img_name = f'erp_color_gray_{mode_name.lower()}.png'
    save_path = os.path.join(PROCESSED_DIR, subject, img_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # 复制一份到 artifacts 文件夹
    artifact_img_name = f'{subject}_erp_color_gray_{mode_name.lower()}.png'
    shutil.copy(save_path, os.path.join(ARTIFACTS_DIR, artifact_img_name))
    
    return stats_results, artifact_img_name

def main():
    print("Starting LFP Color-Gray analysis batch pipeline...")
    
    summary_data = []
    
    for subj in SUBJECTS:
        print(f"\nProcessing {subj}...")
        
        # 获取所有电极
        electrodes = get_electrode_loc(subj)
        is_left_hemi = any(item['Coord'][0] < 0 for item in electrodes)
        
        # 对两个方案进行跑批
        for mode in ['Symmetric', 'Absolute']:
            if mode == 'Symmetric' and (not is_left_hemi):
                # 如果是右半脑被试，方案 A 用对称坐标匹配
                target = TARGET_SYM_R
            else:
                target = TARGET_ABS
                
            ch_name, ch_coord, dist = find_nearest_electrode(electrodes, target)
            print(f"  [{mode} Mode] Nearest channel: {ch_name}, Coord: {ch_coord}, Distance: {dist:.4f} mm")
            
            # 提取数据
            ch_data, time_ms = extract_channel_data(subj, ch_name)
            
            # 分析与绘图
            stats_list, art_img = plot_erp_comparison(subj, ch_name, ch_data, time_ms, mode, dist)
            
            # 保存统计数据用于汇总
            for stat in stats_list:
                summary_data.append({
                    'Subject': subj,
                    'Mode': mode,
                    'Channel': ch_name,
                    'Distance_mm': dist,
                    'MNI_Coord': f"[{ch_coord[0]:.2f},{ch_coord[1]:.2f},{ch_coord[2]:.2f}]",
                    'Condition': stat['condition'],
                    'Color_Mean_uV': stat['col_mean'],
                    'Color_SEM': stat['col_sem'],
                    'Gray_Mean_uV': stat['gray_mean'],
                    'Gray_SEM': stat['gray_sem'],
                    't_stat': stat['t_stat'],
                    'p_val': stat['p_val']
                })
                
    # 创建汇总 DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # 格式化输出
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)
    print("\n=== LFP Color-Gray 100-400ms Stats Summary ===")
    print(df_summary.to_string(index=False))
    
    # 输出为 markdown 表格存入 artifacts 目录
    md_path = os.path.join(ARTIFACTS_DIR, 'lfp_color_gray_stats_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 100-400ms ERP 平均值显著性统计汇总\n\n")
        f.write("此表格汇总了各被试在 **方案 A (Symmetric)** 与 **方案 B (Absolute)** 下，距离目标坐标最近通道在 100-400ms 的平均振幅独立样本 t 检验结果。\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n")
        
    print(f"\nStats summary saved to: {md_path}")
    print("Batch pipeline completed successfully!")

if __name__ == '__main__':
    main()
