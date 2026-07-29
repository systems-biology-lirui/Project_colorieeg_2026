#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Subdirectory Plotting Pipeline for PC, CC, and AC Color Patches.
Generates ERP/HG Color vs Gray (5x2) and Task 2 Gray Fruits ANOVA (1x2) plots
and saves them to: kaojin/{patch}/{subject}/
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy.stats import ranksums, f_oneway
from pymatreader import read_mat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 基础路径
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
kaojin_base = os.path.join(base_dir, 'color_cognition_pipeline', 'analyse_0617', 'result', 'kaojin')
artifacts_base = '/home/lirui/.gemini/antigravity-ide/brain/1870c00b-e214-4064-b47f-91fb6f22337f'

# 3 个 Patch 的通道配置（方案 A）
PATCH_ELECTRODES = {
    'PC': {
        'test001': 'D13',
        'test002': 'C6',
        'test003': 'H11',
        'test006': 'A10'
    },
    'CC': {
        'test001': 'H1',
        'test002': 'C1',
        'test003': 'H4',
        'test006': 'G5'
    },
    'AC': {
        'test001': 'F5',
        'test002': 'F1',
        'test003': 'B5',
        'test006': 'G2'
    }
}

categories = ['Face', 'Object', 'Body', 'Place', 'Merged_All']
cond_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

# ==================== 寻址与加载 ====================

def load_data(subj, is_hg, task_num=1):
    t_prefix = f"task{task_num}"
    if is_hg:
        p1 = os.path.join(base_dir, 'color_cognition_pipeline', 'analyse_0617', 'feature', subj, f'{t_prefix}_hg_subband.mat')
        p2 = os.path.join(base_dir, 'processed_data', subj, f'{t_prefix}_HG_epoched.mat')
        if os.path.exists(p1):
            mat = read_mat(p1)
            epoch = mat['epoch']
            data = epoch['data_cell']
        elif os.path.exists(p2):
            mat = read_mat(p2)
            epoch = mat['epoch']
            data = epoch['data']
        else:
            raise FileNotFoundError(f"{t_prefix} HG file not found for {subj}")
    else:
        p1 = os.path.join(base_dir, 'color_cognition_pipeline', 'analyse_0617', 'feature', subj, f'{t_prefix}_ERP_epoched.mat')
        p2 = os.path.join(base_dir, 'processed_data', subj, f'{t_prefix}_ERP_epoched.mat')
        if os.path.exists(p1):
            mat = read_mat(p1)
        elif os.path.exists(p2):
            mat = read_mat(p2)
        else:
            raise FileNotFoundError(f"{t_prefix} ERP file not found for {subj}")
        epoch = mat['epoch']
        data = epoch['data']
        
    ch_list = list(epoch['ch']['labels'])
    
    if 'time_ms' in epoch:
        time_ms = np.array(epoch['time_ms']).flatten()
    else:
        if is_hg and isinstance(data, list):
            time_len = data[0].shape[-1]
        else:
            time_len = data.shape[-1]
        time_ms = np.arange(-500, -500 + time_len * 2, 2)
        
    return data, ch_list, time_ms

# ==================== 绘图 1: Color vs Gray (5x2) ====================

def plot_signal_erp_or_hg(subject, elec, ch_idx, data, time_ms, out_path, is_hg=False):
    fig, axes = plt.subplots(5, 2, figsize=(12, 20), gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(f"Subject: {subject} | Electrode: {elec} - {'High Gamma' if is_hg else 'ERP'}", fontsize=16, fontweight='bold', y=0.98)
    
    idx_100_400 = np.where((time_ms >= 100) & (time_ms <= 400))[0]
    
    for i, cat_name in enumerate(categories):
        ax_time = axes[i, 0]
        ax_bar = axes[i, 1]
        
        # 数据提取
        if is_hg and isinstance(data, list):
            if cat_name == 'Merged_All':
                c_data = np.concatenate([data[idx][:, ch_idx, :] for idx in [0, 2, 4, 6]], axis=0)
                g_data = np.concatenate([data[idx][:, ch_idx, :] for idx in [1, 3, 5, 7]], axis=0)
            else:
                c_idx, g_idx = cond_pairs[i]
                c_data = data[c_idx][:, ch_idx, :]
                g_data = data[g_idx][:, ch_idx, :]
        else:
            if cat_name == 'Merged_All':
                c_data = np.concatenate([data[idx, :, ch_idx, :] for idx in [0, 2, 4, 6]], axis=0)
                g_data = np.concatenate([data[idx, :, ch_idx, :] for idx in [1, 3, 5, 7]], axis=0)
            else:
                c_idx, g_idx = cond_pairs[i]
                c_data = data[c_idx, :, ch_idx, :]
                g_data = data[g_idx, :, ch_idx, :]
                
        c_data = c_data[~np.isnan(c_data).any(axis=1)]
        g_data = g_data[~np.isnan(g_data).any(axis=1)]
        
        c_mean = np.mean(c_data, axis=0)
        c_sem = np.std(c_data, axis=0) / np.sqrt(c_data.shape[0])
        g_mean = np.mean(g_data, axis=0)
        g_sem = np.std(g_data, axis=0) / np.sqrt(g_data.shape[0])
        
        ax_time.plot(time_ms, c_mean, color='#d32f2f', lw=2.2, label='Color')
        ax_time.fill_between(time_ms, c_mean - c_sem, c_mean + c_sem, color='#d32f2f', alpha=0.15)
        
        ax_time.plot(time_ms, g_mean, color='#212121', lw=2.2, label='Gray')
        ax_time.fill_between(time_ms, g_mean - g_sem, g_mean + g_sem, color='#212121', alpha=0.15)
        
        ax_time.axvline(0, color='#9E9E9E', linestyle='--', alpha=0.6)
        
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
            
        c_vals = np.mean(c_data[:, idx_100_400], axis=1)
        g_vals = np.mean(g_data[:, idx_100_400], axis=1)
        
        bar_c_mean = np.mean(c_vals)
        bar_c_sem = np.std(c_vals) / np.sqrt(len(c_vals))
        bar_g_mean = np.mean(g_vals)
        bar_g_sem = np.std(g_vals) / np.sqrt(len(g_vals))
        
        ax_bar.bar([1], [bar_c_mean], yerr=[bar_c_sem], color='#d32f2f', alpha=0.7, capsize=5, width=0.4, error_kw={'elinewidth':1.5, 'capthick':1.5})
        ax_bar.bar([2], [bar_g_mean], yerr=[bar_g_sem], color='#212121', alpha=0.7, capsize=5, width=0.4, error_kw={'elinewidth':1.5, 'capthick':1.5})
        
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

# ==================== 绘图 2: 4-Way Gray Fruit Comparison (1x2) ====================

def plot_gray_fruits_diff(subject, elec, ch_idx, data, time_ms, out_path, is_hg=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(f"Subject: {subject} | Electrode: {elec} - Gray Fruits Comparison ({'High Gamma' if is_hg else 'ERP'})", fontsize=15, fontweight='bold', y=0.96)
    
    ax_time = axes[0]
    ax_bar = axes[1]
    
    idx_100_400 = np.where((time_ms >= 100) & (time_ms <= 400))[0]
    
    # 提取 4 组条件数据
    if is_hg and isinstance(data, list):
        g4_data = data[4][:, ch_idx, :]
        g5_data = data[5][:, ch_idx, :]
        g6_data = data[6][:, ch_idx, :]
        g7_data = data[7][:, ch_idx, :]
    else:
        g4_data = data[4, :, ch_idx, :]
        g5_data = data[5, :, ch_idx, :]
        g6_data = data[6, :, ch_idx, :]
        g7_data = data[7, :, ch_idx, :]
        
    g4_data = g4_data[~np.isnan(g4_data).any(axis=1)]
    g5_data = g5_data[~np.isnan(g5_data).any(axis=1)]
    g6_data = g6_data[~np.isnan(g6_data).any(axis=1)]
    g7_data = g7_data[~np.isnan(g7_data).any(axis=1)]
    
    conditions = [
        ('Strawberry (Gray)', g4_data, '#e15759', 'darkred'),
        ('Kiwi (Gray)', g5_data, '#59a14f', 'darkgreen'),
        ('Cabbage (Gray)', g6_data, '#bab0ac', '#666666'),
        ('Watermelon (Gray)', g7_data, '#4e79a7', 'navy')
    ]
    
    for label, g_data, curve_color, scatter_color in conditions:
        mean_curve = np.mean(g_data, axis=0)
        sem_curve = np.std(g_data, axis=0) / np.sqrt(g_data.shape[0])
        ax_time.plot(time_ms, mean_curve, color=curve_color, lw=2.2, label=label)
        ax_time.fill_between(time_ms, mean_curve - sem_curve, mean_curve + sem_curve, color=curve_color, alpha=0.12)
        
    ax_time.axvline(0, color='#9E9E9E', linestyle='--', alpha=0.6)
    
    # 逐点 One-way ANOVA
    ymin, ymax = ax_time.get_ylim()
    sig_y = ymin + (ymax - ymin) * 0.05
    for t_idx in range(len(time_ms)):
        v4 = g4_data[:, t_idx]
        v5 = g5_data[:, t_idx]
        v6 = g6_data[:, t_idx]
        v7 = g7_data[:, t_idx]
        stat, p = f_oneway(v4, v5, v6, v7)
        if p < 0.05:
            ax_time.plot(time_ms[t_idx], sig_y, marker='s', color='#f28e2b', markersize=3, alpha=0.7)
            
    ax_time.set_title("Time Course Comparison (Achromatic Fruits)", fontsize=11, fontweight='bold')
    ax_time.set_xlabel("Time (ms)", fontsize=9.5)
    ax_time.set_ylabel("Amplitude (z-score)" if is_hg else "Amplitude (μV)", fontsize=9.5)
    ax_time.set_xlim([-200, 800])
    ax_time.grid(False)
    ax_time.legend(loc='upper right', frameon=True, fontsize=8.5)
    for spine in ax_time.spines.values():
        spine.set_visible(True)
        spine.set_color('#757575')
        
    bar_positions = [1, 2, 3, 4]
    val_means = []
    val_sems = []
    vals_all = []
    
    for label, g_data, curve_color, scatter_color in conditions:
        vals = np.mean(g_data[:, idx_100_400], axis=1)
        vals_all.append(vals)
        val_means.append(np.mean(vals))
        val_sems.append(np.std(vals) / np.sqrt(len(vals)))
        
    for pos, m, sem, cond_info in zip(bar_positions, val_means, val_sems, conditions):
        label, _, curve_color, scatter_color = cond_info
        ax_bar.bar([pos], [m], yerr=[sem], color=curve_color, alpha=0.7, capsize=5, width=0.5, error_kw={'elinewidth':1.5, 'capthick':1.5})
        
    for pos, vals, cond_info in zip(bar_positions, vals_all, conditions):
        label, _, curve_color, scatter_color = cond_info
        ax_bar.scatter(np.random.normal(pos, 0.06, len(vals)), vals, color=scatter_color, alpha=0.2, s=8)
        
    stat_bar, p_val_bar = f_oneway(*vals_all)
    
    ax_bar.set_xticks(bar_positions)
    ax_bar.set_xticklabels(['Strawb', 'Kiwi', 'Cabg', 'Waterm'], fontsize=8.5, rotation=15, ha='right')
    ax_bar.set_title(f"100-400ms Mean\nANOVA p={p_val_bar:.4f}", fontsize=10, fontweight='bold')
    ax_bar.grid(False)
    for spine in ax_bar.spines.values():
        spine.set_visible(True)
        spine.set_color('#757575')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==================== 主控运行 ====================

def main():
    print("="*60)
    print("Starting Hierarchical Subdirectory Plotting for PC, CC, AC")
    print("="*60)
    
    for patch, subjects_dict in PATCH_ELECTRODES.items():
        print(f"\n>>>> Patch: {patch} <<<<")
        for subj, elec in subjects_dict.items():
            print(f"  Subject: {subj}, Best Channel: {elec}")
            
            # 定位输出目录
            out_subdir = os.path.join(kaojin_base, patch, subj)
            art_subdir = os.path.join(artifacts_base, patch, subj)
            os.makedirs(out_subdir, exist_ok=True)
            os.makedirs(art_subdir, exist_ok=True)
            
            # ----------------- 1. TASK 1: Color vs Gray (5x2) -----------------
            # ERP
            try:
                data, ch_list, time_ms = load_data(subj, is_hg=False, task_num=1)
                ch_idx = ch_list.index(elec)
                out_name = f"erp_{elec}.png"
                out_path = os.path.join(out_subdir, out_name)
                plot_signal_erp_or_hg(subj, elec, ch_idx, data, time_ms, out_path, is_hg=False)
                shutil.copy(out_path, os.path.join(art_subdir, out_name))
                print(f"    Saved task1 ERP to: {out_path}")
            except Exception as e:
                print(f"    [Error] task1 ERP failed: {e}")
                
            # HG
            try:
                data, ch_list, time_ms = load_data(subj, is_hg=True, task_num=1)
                ch_idx = ch_list.index(elec)
                out_name = f"hg_{elec}.png"
                out_path = os.path.join(out_subdir, out_name)
                plot_signal_erp_or_hg(subj, elec, ch_idx, data, time_ms, out_path, is_hg=True)
                shutil.copy(out_path, os.path.join(art_subdir, out_name))
                print(f"    Saved task1 HG to: {out_path}")
            except Exception as e:
                print(f"    [Error] task1 HG failed: {e}")
                
            # ----------------- 2. TASK 2: Gray Fruits ANOVA (1x2) -----------------
            # ERP
            try:
                data, ch_list, time_ms = load_data(subj, is_hg=False, task_num=2)
                ch_idx = ch_list.index(elec)
                out_name = f"erp_{elec}_gray_fruits.png"
                out_path = os.path.join(out_subdir, out_name)
                plot_gray_fruits_diff(subj, elec, ch_idx, data, time_ms, out_path, is_hg=False)
                shutil.copy(out_path, os.path.join(art_subdir, out_name))
                print(f"    Saved task2 ERP to: {out_path}")
            except Exception as e:
                print(f"    [Error] task2 ERP failed: {e}")
                
            # HG
            try:
                data, ch_list, time_ms = load_data(subj, is_hg=True, task_num=2)
                ch_idx = ch_list.index(elec)
                out_name = f"hg_{elec}_gray_fruits.png"
                out_path = os.path.join(out_subdir, out_name)
                plot_gray_fruits_diff(subj, elec, ch_idx, data, time_ms, out_path, is_hg=True)
                shutil.copy(out_path, os.path.join(art_subdir, out_name))
                print(f"    Saved task2 HG to: {out_path}")
            except Exception as e:
                print(f"    [Error] task2 HG failed: {e}")
                
    print("\n" + "="*60)
    print("Hierarchical Subdirectory Plotting Complete!")
    print("="*60)

if __name__ == '__main__':
    main()
