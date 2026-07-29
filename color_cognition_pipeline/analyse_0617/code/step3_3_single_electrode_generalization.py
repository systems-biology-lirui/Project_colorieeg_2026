import numpy as np
import pandas as pd
import scipy.io as sio
from pymatreader import read_mat
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os
import warnings
import time

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
analyse_dir = os.path.join(pipeline_dir, 'analyse_0617')
feature_dir = os.path.join(analyse_dir, 'feature')
doc_dir = os.path.join(analyse_dir, 'doc')
result_dir = os.path.join(analyse_dir, 'result')
out_fig_dir = os.path.join(result_dir, 'select_channel', 'decoding', 'single_electrode', 'cross_decoding')
os.makedirs(out_fig_dir, exist_ok=True)

subjects = ['test001', 'test002', 'test003']

# Task 3 纯色触发器
red_color_trigs = ['Trigger-In:51']
green_color_trigs = ['Trigger-In:54']

# Task 2 灰色水果触发器
r1_trigs = ['Trigger-In:123'] # 灰色草莓
r2_trigs = ['Trigger-In:133'] # 灰色西瓜
g1_trigs = ['Trigger-In:103'] # 灰色卷心菜
g2_trigs = ['Trigger-In:113'] # 灰色猕猴桃

# ----------------- 1. 数据读取、基线减法 -----------------
def get_data(mat_path, trigs_to_extract, elec):
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
        
        idx_list = [all_trigs.index(t) for t in trigs_to_extract if t in all_trigs]
        if not idx_list:
            return None, None
            
        data_list = []
        for idx in idx_list:
            # ERP: data shape (Cond, Rep, Ch, Time) -> (Rep, 1, Time)
            # 使用整数索引配合 np.expand_dims 避免 numpy 高级索引维度换位
            trial_data = epoch['data'][idx, :, ch_idx, :]
            trial_data = np.expand_dims(trial_data, axis=1)
            data_list.append(trial_data)
            
        merged_data = np.concatenate(data_list, axis=0)
        
        # Trial-wise 基线减法
        baseline_mask = time_ms < 0
        baseline_indices = np.where(baseline_mask)[0]
        if len(baseline_indices) > 0:
            mean_bl = np.mean(merged_data[:, :, baseline_indices], axis=2, keepdims=True)
            merged_data = merged_data - mean_bl
        return merged_data, time_ms
    except Exception as e:
        return None, None

def clean_data(x):
    if x is None:
        return None
    return x[~np.isnan(x).any(axis=(1,2))]

# ----------------- 2. 10ms 时间步长均值重采样 -----------------
def resample_10ms_bins(data, time_ms, start_time=-100.0, end_time=700.0):
    t_indices = np.where((time_ms >= start_time) & (time_ms <= end_time))[0]
    bin_size = 5
    n_bins = len(t_indices) // bin_size
    
    resampled_data = np.zeros((data.shape[0], data.shape[1], n_bins))
    resampled_time = np.zeros(n_bins)
    
    for b in range(n_bins):
        bin_idx = t_indices[b*bin_size : (b+1)*bin_size]
        resampled_time[b] = np.mean(time_ms[bin_idx])
        resampled_data[:, :, b] = np.mean(data[:, :, bin_idx], axis=2)
        
    return resampled_data, resampled_time

# ----------------- 3. 单时间点一维 SVM 拟合 -----------------
def fit_eval_tg_row_strategy1_1d(t_tr, X_tr_r, X_tr_g, test_pairs_resampled):
    x_train_r = X_tr_r[:, :, t_tr]
    x_train_g = X_tr_g[:, :, t_tr]
    X_tr = np.vstack([x_train_r, x_train_g])
    y_tr = np.hstack([np.zeros(x_train_r.shape[0]), np.ones(x_train_g.shape[0])])
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    
    clf = SVC(kernel='linear', C=0.1)
    clf.fit(X_tr_scaled, y_tr)
    
    n_bins = X_tr_r.shape[2]
    row_accs = np.zeros((4, n_bins))
    
    for p_idx, (te_r_data, te_g_data) in enumerate(test_pairs_resampled):
        for t_te in range(n_bins):
            x_test_r = te_r_data[:, :, t_te]
            x_test_g = te_g_data[:, :, t_te]
            X_te = np.vstack([x_test_r, x_test_g])
            y_te = np.hstack([np.zeros(x_test_r.shape[0]), np.ones(x_test_g.shape[0])])
            
            X_te_scaled = scaler.transform(X_te)
            y_pred = clf.predict(X_te_scaled)
            row_accs[p_idx, t_te] = np.mean(y_pred == y_te)
            
    return np.mean(row_accs, axis=0)

def fit_eval_tg_row_strategy2_1d(t_tr, train_pairs_resampled, X_te_r, X_te_g):
    n_bins = X_te_r.shape[2]
    row_accs = np.zeros((4, n_bins))
    
    for p_idx, (tr_r_data, tr_g_data) in enumerate(train_pairs_resampled):
        x_train_r = tr_r_data[:, :, t_tr]
        x_train_g = tr_g_data[:, :, t_tr]
        X_tr = np.vstack([x_train_r, x_train_g])
        y_tr = np.hstack([np.zeros(x_train_r.shape[0]), np.ones(x_train_g.shape[0])])
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        
        clf = SVC(kernel='linear', C=0.1)
        clf.fit(X_tr_scaled, y_tr)
        
        for t_te in range(n_bins):
            x_test_r = X_te_r[:, :, t_te]
            x_test_g = X_te_g[:, :, t_te]
            X_te = np.vstack([x_test_r, x_test_g])
            y_te = np.hstack([np.zeros(x_test_r.shape[0]), np.ones(x_test_g.shape[0])])
            
            X_te_scaled = scaler.transform(X_te)
            y_pred = clf.predict(X_te_scaled)
            row_accs[p_idx, t_te] = np.mean(y_pred == y_te)
            
    return np.mean(row_accs, axis=0)

# ----------------- 4. 1行2列 双策略热图绘制 -----------------
def plot_single_electrode_tg_combined(matrix_s1, matrix_s2, time_bins, subj, elec, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    fig.suptitle(f"Single-Channel temporal generalization cross-decoding: {subj} - {elec}", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 策略 1 热图
    im1 = ax1.imshow(
        matrix_s1, origin='lower', cmap='RdBu_r', 
        extent=[time_bins[0], time_bins[-1], time_bins[0], time_bins[-1]],
        vmin=0.42, vmax=0.58
    )
    ax1.axhline(0, color='black', linestyle='--', alpha=0.6, lw=1.2)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.6, lw=1.2)
    ax1.plot([time_bins[0], time_bins[-1]], [time_bins[0], time_bins[-1]], color='#555555', linestyle=':', alpha=0.5, lw=1.0)
    ax1.set_title("Strategy 1: Color -> Gray-Memory", fontsize=11.5, fontweight='bold')
    ax1.set_xlabel("Testing Time (ms)", fontsize=10)
    ax1.set_ylabel("Training Time (ms)", fontsize=10)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 策略 2 热图
    im2 = ax2.imshow(
        matrix_s2, origin='lower', cmap='RdBu_r', 
        extent=[time_bins[0], time_bins[-1], time_bins[0], time_bins[-1]],
        vmin=0.42, vmax=0.58
    )
    ax2.axhline(0, color='black', linestyle='--', alpha=0.6, lw=1.2)
    ax2.axvline(0, color='black', linestyle='--', alpha=0.6, lw=1.2)
    ax2.plot([time_bins[0], time_bins[-1]], [time_bins[0], time_bins[-1]], color='#555555', linestyle=':', alpha=0.5, lw=1.0)
    ax2.set_title("Strategy 2: Gray-Memory -> Color", fontsize=11.5, fontweight='bold')
    ax2.set_xlabel("Testing Time (ms)", fontsize=10)
    ax2.set_ylabel("Training Time (ms)", fontsize=10)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ----------------- 主程序 -----------------
def main():
    print("="*85)
    print("Step 3_3: Single-Electrode Temporal Generalization Analysis")
    print("="*85)
    
    # 1. 加载显著通道表
    sig_path = os.path.join(doc_dir, 'select_channel_memory_significance_erp.csv')
    if not os.path.exists(sig_path):
        print("[ERROR] 显著通道表不存在，退出")
        return
        
    df_sig = pd.read_csv(sig_path)
    df_memory_color = df_sig[df_sig['Sig_Category'] != 'Non_Sig']
    
    print(f"Total memory-selective ERP electrodes to run: {len(df_memory_color)}")
    
    save_dict = {}
    time_bins = None
    
    # 2. 逐电极计算
    for idx, row in df_memory_color.iterrows():
        subj = str(row['Subject']).strip()
        elec = str(row['Electrode']).strip()
        
        print(f"\nProcessing Electrode: {subj} - {elec}...")
        
        path3 = os.path.join(feature_dir, subj, 'task3_ERP_epoched.mat')
        d_color, t_arr3 = get_data(path3, red_color_trigs, elec)
        d_gray, _ = get_data(path3, green_color_trigs, elec)
        
        path2 = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
        d_r1, t_arr2 = get_data(path2, r1_trigs, elec)
        d_r2, _ = get_data(path2, r2_trigs, elec)
        d_g1, _ = get_data(path2, g1_trigs, elec)
        d_g2, _ = get_data(path2, g2_trigs, elec)
        
        if any(d is None for d in [d_color, d_gray, d_r1, d_r2, d_g1, d_g2]):
            print(f"  [WARNING] 数据不完整，跳过此电极")
            continue
            
        # 清理
        d_color, d_gray, d_r1, d_r2, d_g1, d_g2 = map(clean_data, [d_color, d_gray, d_r1, d_r2, d_g1, d_g2])
        
        # 重采样
        d_color_res, time_b = resample_10ms_bins(d_color, t_arr3)
        d_gray_res, _ = resample_10ms_bins(d_gray, t_arr3)
        d_r1_res, _ = resample_10ms_bins(d_r1, t_arr2)
        d_r2_res, _ = resample_10ms_bins(d_r2, t_arr2)
        d_g1_res, _ = resample_10ms_bins(d_g1, t_arr2)
        d_g2_res, _ = resample_10ms_bins(d_g2, t_arr2)
        
        if time_bins is None:
            time_bins = time_b
            
        n_bins = len(time_bins)
        
        pairs_res = [
            (d_r1_res, d_g1_res),
            (d_r1_res, d_g2_res),
            (d_r2_res, d_g1_res),
            (d_r2_res, d_g2_res)
        ]
        
        # 3. 计算策略 1
        results_s1 = Parallel(n_jobs=-1)(
            delayed(fit_eval_tg_row_strategy1_1d)(t_tr, d_color_res, d_gray_res, pairs_res)
            for t_tr in range(n_bins)
        )
        matrix_s1 = np.vstack(results_s1)
        
        # 4. 计算策略 2
        results_s2 = Parallel(n_jobs=-1)(
            delayed(fit_eval_tg_row_strategy2_1d)(t_tr, pairs_res, d_color_res, d_gray_res)
            for t_tr in range(n_bins)
        )
        matrix_s2 = np.vstack(results_s2)
        
        # 缓存数据用于打包导出
        save_dict[f"{subj}_{elec}_s1"] = matrix_s1
        save_dict[f"{subj}_{elec}_s2"] = matrix_s2
        
        # 5. 绘图
        out_img = os.path.join(out_fig_dir, f"{subj}_{elec}_cross_decoding_generalization.png")
        plot_single_electrode_tg_combined(matrix_s1, matrix_s2, time_bins, subj, elec, out_img)
        print(f"  Saved heatmap to: {out_img}")
        
    # 保存所有矩阵数据到压缩包
    npz_path = os.path.join(doc_dir, 'single_electrode_tg_data.npz')
    np.savez_compressed(npz_path, time_bins=time_bins, **save_dict)
    print(f"\n[SUCCESS] Compressed and saved all individual matrices to: {npz_path}")
    print("="*85)

if __name__ == '__main__':
    main()
