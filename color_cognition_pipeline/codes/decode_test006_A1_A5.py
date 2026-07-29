#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory color (red vs green) decoding on test006 using A1-A5 electrodes.
Includes a 50-permutation test per timepoint for single-subject significance thresholding.
Optimized via pre-slicing and environment thread locking to eliminate BLAS deadlocks.
"""

import os
# 强制限制底层的 BLAS/MKL 多线程，防止多进程下死锁与过度上下文切换开销
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import shutil
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from pymatreader import read_mat
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 基础路径
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
kaojin_dir = os.path.join(base_dir, 'color_cognition_pipeline', 'analyse_0617', 'result', 'kaojin')
artifacts_dir = '/home/lirui/.gemini/antigravity-ide/brain/1870c00b-e214-4064-b47f-91fb6f22337f'

subj = 'test006'
elecs = ['A1', 'A2', 'A3', 'A4', 'A5']

# ==================== 1. 数据读取与校正 ====================

def load_task2_data(is_hg):
    t_prefix = "task2"
    if is_hg:
        p2 = os.path.join(base_dir, 'processed_data', subj, f'{t_prefix}_HG_epoched.mat')
        if os.path.exists(p2):
            mat = read_mat(p2)
            epoch = mat['epoch']
            data = epoch['data']
        else:
            raise FileNotFoundError(f"HG file not found at {p2}")
    else:
        p2 = os.path.join(base_dir, 'processed_data', subj, f'{t_prefix}_ERP_epoched.mat')
        if os.path.exists(p2):
            mat = read_mat(p2)
            epoch = mat['epoch']
            data = epoch['data']
        else:
            raise FileNotFoundError(f"ERP file not found at {p2}")
            
    ch_list = list(epoch['ch']['labels'])
    
    if 'time_ms' in epoch:
        time_ms = np.array(epoch['time_ms']).flatten()
    else:
        time_len = data.shape[-1]
        time_ms = np.arange(-500, -500 + time_len * 2, 2)
        
    # 定位电极索引
    ch_indices = [ch_list.index(e) for e in elecs if e in ch_list]
    if len(ch_indices) != len(elecs):
        raise ValueError(f"Some electrodes of A1-A5 not found in channels: {ch_list}")
        
    # 提取 A1-A5 电极的数据 [Cond, Rep, Ch, Time] -> [Cond, Rep, 5, Time]
    data_extracted = data[:, :, ch_indices, :]
    
    # 提取 4 种灰色水果
    g4 = data_extracted[4]
    g5 = data_extracted[5]
    g6 = data_extracted[6]
    g7 = data_extracted[7]
    
    # 过滤 NaN 的 trials
    g4 = g4[~np.isnan(g4).any(axis=(1, 2))]
    g5 = g5[~np.isnan(g5).any(axis=(1, 2))]
    g6 = g6[~np.isnan(g6).any(axis=(1, 2))]
    g7 = g7[~np.isnan(g7).any(axis=(1, 2))]
    
    # 基线校正 (Baseline correction, 时间 < 0ms)
    baseline_mask = time_ms < 0
    baseline_indices = np.where(baseline_mask)[0]
    
    if len(baseline_indices) > 0:
        g4 = g4 - np.mean(g4[:, :, baseline_indices], axis=2, keepdims=True)
        g5 = g5 - np.mean(g5[:, :, baseline_indices], axis=2, keepdims=True)
        g6 = g6 - np.mean(g6[:, :, baseline_indices], axis=2, keepdims=True)
        g7 = g7 - np.mean(g7[:, :, baseline_indices], axis=2, keepdims=True)
        
    return g4, g5, g6, g7, time_ms

# ==================== 2. SVM 拟合与 Permutation ====================

def fit_eval_single_t_with_perm(t_pair_slices, n_perms=50):
    """
    接收预先切好的 2D 矩阵片进行解码与置换检验，大幅降低 IPC 开销
    """
    # 1. 计算真实 4 折交叉解码
    real_accs = []
    for tr_r, tr_g, te_r, te_g in t_pair_slices:
        X_tr = np.vstack([tr_r, tr_g])
        y_tr = np.hstack([np.zeros(tr_r.shape[0]), np.ones(tr_g.shape[0])])
        X_te = np.vstack([te_r, te_g])
        y_te = np.hstack([np.zeros(te_r.shape[0]), np.ones(te_g.shape[0])])
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        
        clf = SVC(kernel='linear', C=0.1)
        clf.fit(X_tr_scaled, y_tr)
        real_accs.append(np.mean(clf.predict(X_te_scaled) == y_te))
        
    real_mean = np.mean(real_accs)
    
    # 2. 置换检验 (打乱训练标签)
    perm_means = []
    for _ in range(n_perms):
        perm_accs = []
        for tr_r, tr_g, te_r, te_g in t_pair_slices:
            X_tr = np.vstack([tr_r, tr_g])
            y_tr = np.hstack([np.zeros(tr_r.shape[0]), np.ones(tr_g.shape[0])])
            np.random.shuffle(y_tr) # 打乱训练集标签关系
            
            X_te = np.vstack([te_r, te_g])
            y_te = np.hstack([np.zeros(te_r.shape[0]), np.ones(te_g.shape[0])])
            
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)
            
            clf = SVC(kernel='linear', C=0.1)
            clf.fit(X_tr_scaled, y_tr)
            perm_accs.append(np.mean(clf.predict(X_te_scaled) == y_te))
        perm_means.append(np.mean(perm_accs))
        
    perm_means = np.array(perm_means)
    p_val = (np.sum(perm_means >= real_mean) + 1.0) / (n_perms + 1.0)
    
    return real_mean, p_val, real_accs

# ==================== 3. 统计显著段寻找 ====================

def find_significant_windows(p_vals, time_ms, p_thresh=0.05, min_duration=20):
    dt = time_ms[1] - time_ms[0]
    sig_mask = p_vals < p_thresh
    windows = []
    
    in_window = False
    start_idx = None
    
    for idx, is_sig in enumerate(sig_mask):
        if is_sig and not in_window:
            in_window = True
            start_idx = idx
        elif not is_sig and in_window:
            in_window = False
            end_idx = idx - 1
            duration = (end_idx - start_idx + 1) * dt
            if duration >= min_duration:
                windows.append((time_ms[start_idx], time_ms[end_idx]))
    if in_window:
        end_idx = len(sig_mask) - 1
        duration = (end_idx - start_idx + 1) * dt
        if duration >= min_duration:
            windows.append((time_ms[start_idx], time_ms[end_idx]))
            
    return windows

# ==================== 4. 绘图函数 ====================

def plot_decoding_result(time_ms, real_means, perm_ps, sig_windows, pair_history, feature_label, out_path, n_perms=50):
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    
    t_idx_plot = np.where((time_ms >= -200) & (time_ms <= 800))[0]
    time_plot = time_ms[t_idx_plot]
    
    pair_colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
    for idx in range(4):
        p_plot = pair_history[idx][t_idx_plot]
        ax.plot(time_plot, p_plot, color=pair_colors[idx], lw=1.0, linestyle='--', alpha=0.6, label=f"Fold {idx+1}")
        
    mean_plot = real_means[t_idx_plot]
    ax.plot(time_plot, mean_plot, color='#6f2da8', lw=3.0, label='Average Decoding Accuracy')
    
    ax.axhline(0.5, color='#9e9e9e', linestyle=':', lw=1.5, label='Chance Level (50%)')
    ax.axvline(0, color='#757575', linestyle='-', lw=1.2)
    
    ymin, ymax = 0.35, 0.75
    sig_y = ymin + (ymax - ymin) * 0.05
    
    has_sig = False
    for start, end in sig_windows:
        if end < -200 or start > 800:
            continue
        s_plot = max(start, -200)
        e_plot = min(end, 800)
        
        ax.axvspan(s_plot, e_plot, color='#d62728', alpha=0.1, zorder=1)
        
        t_sig_range = time_plot[(time_plot >= s_plot) & (time_plot <= e_plot)]
        label_sig = 'Permutation Significant (p < 0.05, >20ms)' if not has_sig else ""
        ax.plot(t_sig_range, [sig_y] * len(t_sig_range), marker='s', color='#d62728', markersize=3, alpha=0.8, linestyle='none', label=label_sig)
        has_sig = True
        
    ax.set_title(f"{feature_label} Memory Color Decoding on {subj} (A1-A5 Electrodes)\n(Permutation Test: N={n_perms} per timepoint)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Time relative to stimulus onset (ms)", fontsize=10.5)
    ax.set_ylabel("Decoding Accuracy", fontsize=10.5)
    ax.set_xlim([-200, 800])
    ax.set_ylim([ymin, ymax])
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.set_facecolor('#fafafa')
    ax.legend(loc='lower left', framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# ==================== 5. 主流程 ====================

def run_pipeline(is_hg):
    label = "HG" if is_hg else "ERP"
    print(f"\n>>>> Running decoding for: {label} <<<<")
    
    # 1. 读取数据
    g4, g5, g6, g7, time_ms = load_task2_data(is_hg)
    print(f"  Data Loaded. Trial counts: G4={g4.shape[0]}, G5={g5.shape[0]}, G6={g6.shape[0]}, G7={g7.shape[0]}")
    
    pairs = [
        (g4, g6, g7, g5), # 4 & 6 -> 7 & 5
        (g4, g5, g7, g6), # 4 & 5 -> 7 & 6
        (g7, g6, g4, g5), # 7 & 6 -> 4 & 5
        (g7, g5, g4, g6)  # 7 & 5 -> 4 & 6
    ]
    
    n_time = len(time_ms)
    n_perms = 50
    
    # 预先分切数据 (2D)，减小进程间反序列化开销 750 倍！
    t_slices = []
    for t in range(n_time):
        t_pair_slices = []
        for train_r, train_g, test_r, test_g in pairs:
            t_pair_slices.append((
                train_r[:, :, t], # (Rep, 5)
                train_g[:, :, t], # (Rep, 5)
                test_r[:, :, t],  # (Rep, 5)
                test_g[:, :, t]   # (Rep, 5)
            ))
        t_slices.append(t_pair_slices)
    
    # 2. SVM 解码加 50 次置换检验
    print(f"  Fitting SVM and Permutations on {n_time} timepoints (Joblib Parallel)...")
    results = Parallel(n_jobs=-1)(
        delayed(fit_eval_single_t_with_perm)(t_slices[t], n_perms=n_perms)
        for t in range(n_time)
    )
    
    real_means = np.array([r[0] for r in results])
    perm_ps = np.array([r[1] for r in results])
    
    # 提取各时间点 4 折各自的解码值 (维度: [4, Time])
    pair_history = np.zeros((4, n_time))
    for t in range(n_time):
        pair_history[:, t] = results[t][2]
        
    # 3. 统计显著段
    sig_windows = find_significant_windows(perm_ps, time_ms, p_thresh=0.05, min_duration=20)
    print(f"  Significant memory decoding windows (>20ms): {sig_windows}")
    
    # 4. 导出 Excel 表格
    df_export = pd.DataFrame({
        'Time_ms': time_ms,
        'Mean_Decoding_Accuracy': real_means,
        'Permutation_P': perm_ps,
        'Fold_1_Acc': pair_history[0],
        'Fold_2_Acc': pair_history[1],
        'Fold_3_Acc': pair_history[2],
        'Fold_4_Acc': pair_history[3]
    })
    
    out_xlsx = os.path.join(kaojin_dir, f"{subj}_{label.lower()}_A1_A5_decoding_summary.xlsx")
    df_export.to_excel(out_xlsx, index=False)
    print(f"  Saved stats table to: {out_xlsx}")
    
    # 5. 绘制时程图
    out_fig = os.path.join(kaojin_dir, f"{subj}_{label.lower()}_A1_A5_decoding.png")
    plot_decoding_result(time_ms, real_means, perm_ps, sig_windows, pair_history, label, out_fig, n_perms=n_perms)
    print(f"  Saved figure to: {out_fig}")
    
    # 拷贝一份到 artifacts
    shutil.copy(out_xlsx, os.path.join(artifacts_dir, f"{subj}_{label.lower()}_A1_A5_decoding_summary.xlsx"))
    shutil.copy(out_fig, os.path.join(artifacts_dir, f"{subj}_{label.lower()}_A1_A5_decoding.png"))
    
def main():
    print("="*75)
    print("Starting Memory Color Decoding Pipeline for test006 (A1-A5) [Optimized]")
    print("="*75)
    
    os.makedirs(kaojin_dir, exist_ok=True)
    
    # ERP 解码
    run_pipeline(is_hg=False)
    
    # HG 解码
    run_pipeline(is_hg=True)
    
    print("\n" + "="*75)
    print("test006 (A1-A5) Memory Color Decoding Successfully Completed!")
    print("="*75)

if __name__ == '__main__':
    main()
