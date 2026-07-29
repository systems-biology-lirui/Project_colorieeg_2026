#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot LFP/ERP & HG time-course and bar-scatter differences for target electrodes.
Style identical to B5 target channel plots (stra1_2_3_4_B5.png style).
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from pymatreader import read_mat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 基础路径
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
kaojin_dir = os.path.join(base_dir, 'color_cognition_pipeline', 'analyse_0617', 'result', 'kaojin')
artifacts_dir = '/home/lirui/.gemini/antigravity-ide/brain/1870c00b-e214-4064-b47f-91fb6f22337f'

# 目标电极与被试映射（方案 A）
TARGET_ELECTRODES = {
    'test001': 'D13',
    'test002': 'C6',
    'test003': 'H11',
    'test006': 'A10'
}

categories = ['Face', 'Object', 'Body', 'Place', 'Merged_All']
cond_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

def load_data(subj, is_hg):
    if is_hg:
        p1 = os.path.join(base_dir, 'color_cognition_pipeline', 'analyse_0617', 'feature', subj, 'task1_hg_subband.mat')
        p2 = os.path.join(base_dir, 'processed_data', subj, 'task1_HG_epoched.mat')
        if os.path.exists(p1):
            mat = read_mat(p1)
            epoch = mat['epoch']
            data = epoch['data_cell']
        elif os.path.exists(p2):
            mat = read_mat(p2)
            epoch = mat['epoch']
            data = epoch['data']
        else:
            raise FileNotFoundError(f"HG file not found for {subj}")
    else:
        p1 = os.path.join(base_dir, 'color_cognition_pipeline', 'analyse_0617', 'feature', subj, 'task1_ERP_epoched.mat')
        p2 = os.path.join(base_dir, 'processed_data', subj, 'task1_ERP_epoched.mat')
        if os.path.exists(p1):
            mat = read_mat(p1)
        elif os.path.exists(p2):
            mat = read_mat(p2)
        else:
            raise FileNotFoundError(f"ERP file not found for {subj}")
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
        if is_hg and isinstance(data, list):
            # data_cell 格式 (test001-003)
            if cat_name == 'Merged_All':
                c_data = np.concatenate([data[idx][:, ch_idx, :] for idx in [0, 2, 4, 6]], axis=0)
                g_data = np.concatenate([data[idx][:, ch_idx, :] for idx in [1, 3, 5, 7]], axis=0)
            else:
                c_idx, g_idx = cond_pairs[i]
                c_data = data[c_idx][:, ch_idx, :]
                g_data = data[g_idx][:, ch_idx, :]
        else:
            # 4D array [Cond, Rep, Ch, Time] (对 ERP 或者 test006 的 HG)
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

def main():
    print("="*60)
    print("Starting Target Electrodes Time Course & Bar-Scatter Plotting (pymatreader based)")
    print("="*60)
    
    os.makedirs(kaojin_dir, exist_ok=True)
    
    for subj, elec in TARGET_ELECTRODES.items():
        print(f"\nProcessing Subject: {subj}, Target Electrode: {elec}...")
        
        # 1. 绘制 ERP
        print("  Plotting ERP...")
        try:
            data, ch_list, time_ms = load_data(subj, is_hg=False)
            if elec not in ch_list:
                print(f"  [Warning] Channel {elec} not found in ERP labels of {subj}. Available labels count: {len(ch_list)}")
                continue
            ch_idx = ch_list.index(elec)
            out_name = f"{subj}_erp_{elec}.png"
            out_path = os.path.join(kaojin_dir, out_name)
            plot_signal_erp_or_hg(subj, elec, ch_idx, data, time_ms, out_path, is_hg=False)
            print(f"  Saved ERP plot to: {out_path}")
            
            # 复制一份到 artifacts
            shutil.copy(out_path, os.path.join(artifacts_dir, out_name))
        except Exception as e:
            print(f"  [Error] Failed to process ERP for {subj}: {e}")
            import traceback; traceback.print_exc()
            
        # 2. 绘制 HG
        print("  Plotting HG...")
        try:
            data, ch_list, time_ms = load_data(subj, is_hg=True)
            if elec not in ch_list:
                print(f"  [Warning] Channel {elec} not found in HG labels of {subj}. Available labels count: {len(ch_list)}")
                continue
            ch_idx = ch_list.index(elec)
            out_name = f"{subj}_hg_{elec}.png"
            out_path = os.path.join(kaojin_dir, out_name)
            plot_signal_erp_or_hg(subj, elec, ch_idx, data, time_ms, out_path, is_hg=True)
            print(f"  Saved HG plot to: {out_path}")
            
            # 复制一份到 artifacts
            shutil.copy(out_path, os.path.join(artifacts_dir, out_name))
        except Exception as e:
            print(f"  [Error] Failed to process HG for {subj}: {e}")
            import traceback; traceback.print_exc()
            
    print("\n" + "="*60)
    print("Plotting Pipeline for Kaojin Target Electrodes Complete!")
    print("="*60)

if __name__ == '__main__':
    main()
