"""
Step 3_2_union (新增):
  在进行 Cross-Decoding 时间泛化时，使用所有主要筛选电极 (select电极的并集，来自 select_channel_summary.xlsx)
  而不是仅使用 memory color 显著电极。
  输出的新热图及数据表将加上 `_union` 后缀，避免覆盖原有结果。
"""
import numpy as np
import pandas as pd
import scipy.io as sio
from pymatreader import read_mat
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os, warnings, time

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
out_fig_dir = os.path.join(result_dir, 'select_channel', 'decoding', 'cross_decoding')
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
def get_data(mat_path, trigs_to_extract, elecs):
    if not os.path.exists(mat_path):
        return None, None
    try:
        mat = read_mat(mat_path)
        epoch = mat['epoch']
        ch_names = list(epoch['ch']['labels'])
        time_ms = epoch['time_ms'] if 'time_ms' in epoch else np.linspace(-500, 998, epoch['data'].shape[-1])
        all_trigs = list(epoch['trigger'])
        
        ch_indices = [ch_names.index(e) for e in elecs if e in ch_names]
        if not ch_indices:
            return None, None
            
        idx_list = [all_trigs.index(t) for t in trigs_to_extract if t in all_trigs]
        if not idx_list:
            return None, None
            
        data_list = []
        for idx in idx_list:
            trial_data = epoch['data'][idx, :, :, :]
            trial_data = trial_data[:, ch_indices, :]
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
        print(f"  [ERROR] Loading {mat_path} failed: {e}")
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

# ----------------- 3. 单训练时间点 TG 分类拟合 -----------------
def fit_eval_tg_row_strategy1(t_tr, X_tr_r, X_tr_g, test_pairs_resampled):
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

def fit_eval_tg_row_strategy2(t_tr, train_pairs_resampled, X_te_r, X_te_g):
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

# ----------------- 4. 绘制时间泛化二维热图 -----------------
def plot_tg_heatmap(matrix, time_bins, title, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    im = ax.imshow(
        matrix, origin='lower', cmap='RdBu_r', 
        extent=[time_bins[0], time_bins[-1], time_bins[0], time_bins[-1]],
        vmin=0.42, vmax=0.58
    )
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.6, lw=1.2)
    ax.axvline(0, color='black', linestyle='--', alpha=0.6, lw=1.2)
    ax.plot([time_bins[0], time_bins[-1]], [time_bins[0], time_bins[-1]], color='#555555', linestyle=':', alpha=0.5, lw=1.0)
    
    ax.set_title(title, fontsize=12.5, fontweight='bold', pad=12)
    ax.set_xlabel("Testing Time (ms)", fontsize=10.5)
    ax.set_ylabel("Training Time (ms)", fontsize=10.5)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Decoding Accuracy', rotation=270, labelpad=15, fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [HEATMAP SAVED] Saved temporal generalization map to: {out_path}")

# ----------------- 主程序 -----------------
def main():
    print("="*85)
    print("Step 3_2_union: Cross-Decoding Temporal Generalization ERP (Union of Select Electrodes)")
    print("="*85)
    
    # 1. 读取 select_channel_summary.xlsx 作为 select电极的并集
    summary_path = os.path.join(doc_dir, 'select_channel_summary.xlsx')
    if not os.path.exists(summary_path):
        print("[ERROR] select_channel_summary.xlsx 丢失，跳过计算")
        return
        
    df_summary = pd.read_excel(summary_path)
    
    subj_elecs = {}
    for subj in subjects:
        df_subj = df_summary[df_summary['Subject'] == subj]
        subj_elecs[subj] = df_subj['Electrode'].astype(str).tolist()
        print(f"  {subj}: select电极数量 = {len(subj_elecs[subj])}")
        
    active_subjs = [s for s, e in subj_elecs.items() if len(e) > 0]
    if not active_subjs:
        print("[WARNING] 无任何被试包含主要筛选电极！")
        return
        
    tg_matrices_strategy1 = {}
    tg_matrices_strategy2 = {}
    time_bins = None
    
    # 2. 逐被试计算时间泛化矩阵
    for subj in active_subjs:
        print(f"\n>>> Computing temporal generalization for {subj} using union of select electrodes...")
        elecs = subj_elecs[subj]
        
        # A. 加载 Task 3 (纯色色块)
        path3 = os.path.join(feature_dir, subj, 'task3_ERP_epoched.mat')
        d_color, t_arr3 = get_data(path3, red_color_trigs, elecs)
        d_gray, _ = get_data(path3, green_color_trigs, elecs)
        
        # B. 加载 Task 2 (灰色水果)
        path2 = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
        d_r1, t_arr2 = get_data(path2, r1_trigs, elecs)
        d_r2, _ = get_data(path2, r2_trigs, elecs)
        d_g1, _ = get_data(path2, g1_trigs, elecs)
        d_g2, _ = get_data(path2, g2_trigs, elecs)
        
        if any(d is None for d in [d_color, d_gray, d_r1, d_r2, d_g1, d_g2]):
            print(f"  [ERROR] {subj} 缺少某种解码所需的数据，跳过")
            continue
            
        # 清理
        d_color, d_gray, d_r1, d_r2, d_g1, d_g2 = map(clean_data, [d_color, d_gray, d_r1, d_r2, d_g1, d_g2])
        
        # C. 10ms 均值下采样与 [-100, 700]ms 提取
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
            (d_r1_res, d_g1_res), # Strawberry, Cabbage
            (d_r1_res, d_g2_res), # Strawberry, Kiwi
            (d_r2_res, d_g1_res), # Watermelon, Cabbage
            (d_r2_res, d_g2_res)  # Watermelon, Kiwi
        ]
        
        # 3. 计算策略 1 (Color-to-Gray-Memory)
        print("  Running Strategy 1: Color-to-Gray-Memory...")
        t_start = time.time()
        results_strat1 = Parallel(n_jobs=-1)(
            delayed(fit_eval_tg_row_strategy1)(t_tr, d_color_res, d_gray_res, pairs_res)
            for t_tr in range(n_bins)
        )
        matrix_strat1 = np.vstack(results_strat1)
        tg_matrices_strategy1[subj] = matrix_strat1
        print(f"  Strategy 1 matrix calculated in {time.time() - t_start:.2f}s.")
        
        # 4. 计算策略 2 (Gray-Memory-to-Color)
        print("  Running Strategy 2: Gray-Memory-to-Color...")
        t_start = time.time()
        results_strat2 = Parallel(n_jobs=-1)(
            delayed(fit_eval_tg_row_strategy2)(t_tr, pairs_res, d_color_res, d_gray_res)
            for t_tr in range(n_bins)
        )
        matrix_strat2 = np.vstack(results_strat2)
        tg_matrices_strategy2[subj] = matrix_strat2
        print(f"  Strategy 2 matrix calculated in {time.time() - t_start:.2f}s.")
        
        # D. 绘制被试独立热图 (输出带 _union 后缀)
        plot_tg_heatmap(
            matrix_strat1, time_bins, 
            f"TG Strategy 1 (Color-to-Gray-Memory): {subj} (Union)",
            os.path.join(out_fig_dir, f"strategy1_{subj}_temporal_generalization_union.png")
        )
        plot_tg_heatmap(
            matrix_strat2, time_bins, 
            f"TG Strategy 2 (Gray-Memory-to-Color): {subj} (Union)",
            os.path.join(out_fig_dir, f"strategy2_{subj}_temporal_generalization_union.png")
        )
        
    # 3. 计算 Group 平均时间泛化热图并绘制 (输出带 _union 后缀)
    print("\nComputing Group level average temporal generalization maps...")
    group_strat1 = np.mean([tg_matrices_strategy1[s] for s in active_subjs], axis=0)
    group_strat2 = np.mean([tg_matrices_strategy2[s] for s in active_subjs], axis=0)
    
    plot_tg_heatmap(
        group_strat1, time_bins,
        "Group TG Strategy 1 (Color-to-Gray-Memory) (Union)",
        os.path.join(out_fig_dir, "strategy1_group_temporal_generalization_union.png")
    )
    plot_tg_heatmap(
        group_strat2, time_bins,
        "Group TG Strategy 2 (Gray-Memory-to-Color) (Union)",
        os.path.join(out_fig_dir, "strategy2_group_temporal_generalization_union.png")
    )
    
    # 4. 导出矩阵明细为 CSV / Excel
    export_rows_strat1 = []
    export_rows_strat2 = []
    
    for tr_idx in range(n_bins):
        for te_idx in range(n_bins):
            export_rows_strat1.append({
                'TrainTime_ms': time_bins[tr_idx],
                'TestTime_ms': time_bins[te_idx],
                'Group_Acc': group_strat1[tr_idx, te_idx],
                'test001_Acc': tg_matrices_strategy1['test001'][tr_idx, te_idx] if 'test001' in tg_matrices_strategy1 else np.nan,
                'test002_Acc': tg_matrices_strategy1['test002'][tr_idx, te_idx] if 'test002' in tg_matrices_strategy1 else np.nan,
                'test003_Acc': tg_matrices_strategy1['test003'][tr_idx, te_idx] if 'test003' in tg_matrices_strategy1 else np.nan,
            })
            export_rows_strat2.append({
                'TrainTime_ms': time_bins[tr_idx],
                'TestTime_ms': time_bins[te_idx],
                'Group_Acc': group_strat2[tr_idx, te_idx],
                'test001_Acc': tg_matrices_strategy2['test001'][tr_idx, te_idx] if 'test001' in tg_matrices_strategy2 else np.nan,
                'test002_Acc': tg_matrices_strategy2['test002'][tr_idx, te_idx] if 'test002' in tg_matrices_strategy2 else np.nan,
                'test003_Acc': tg_matrices_strategy2['test003'][tr_idx, te_idx] if 'test003' in tg_matrices_strategy2 else np.nan,
            })
            
    df_exp_s1 = pd.DataFrame(export_rows_strat1)
    df_exp_s2 = pd.DataFrame(export_rows_strat2)
    
    xlsx_s1 = os.path.join(doc_dir, 'cross_decoding_tg_strategy1_union.xlsx')
    csv_s1 = os.path.join(doc_dir, 'cross_decoding_tg_strategy1_union.csv')
    xlsx_s2 = os.path.join(doc_dir, 'cross_decoding_tg_strategy2_union.xlsx')
    csv_s2 = os.path.join(doc_dir, 'cross_decoding_tg_strategy2_union.csv')
    
    df_exp_s1.to_excel(xlsx_s1, index=False)
    df_exp_s1.to_csv(csv_s1, index=False)
    df_exp_s2.to_excel(xlsx_s2, index=False)
    df_exp_s2.to_csv(csv_s2, index=False)
    
    print(f"\n[SUCCESS] Exported cross decoding temporal generalization tables (Union) to:\n  - {xlsx_s1}\n  - {xlsx_s2}")
    print("="*85)

if __name__ == '__main__':
    main()
