import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
from pymatreader import read_mat
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import statsmodels.genmod.bayes_mixed_glm as bmg
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
out_fig_dir = os.path.join(result_dir, 'select_channel', 'decoding')
os.makedirs(out_fig_dir, exist_ok=True)

subjects = ['test001', 'test002', 'test003']
# 纯色刺激触发器 (Task 3)
red_trigs = ['Trigger-In:51']
green_trigs = ['Trigger-In:54']

# ----------------- 1. 数据读取与向量化基线减除 -----------------
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
            # ERP: data shape (Cond, Rep, Ch, Time) -> 取指定条件与通道 -> (Rep, n_ch, Time)
            trial_data = epoch['data'][idx, :, :, :]
            trial_data = trial_data[:, ch_indices, :]
            data_list.append(trial_data)
            
        merged_data = np.concatenate(data_list, axis=0)
        
        # 向量化 Trial-wise 基线减法校正
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

# ----------------- 2. 5 折交叉验证 SVM 拟合 -----------------
def fit_eval_kfold_single_t(t, data_r, data_g, n_splits=5):
    X_r = data_r[:, :, t]
    X_g = data_g[:, :, t]
    
    X = np.vstack([X_r, X_g])
    y = np.hstack([np.zeros(X_r.shape[0]), np.ones(X_g.shape[0])])
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    correct_all = np.zeros(len(y), dtype=int)
    
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        clf = SVC(kernel='linear', C=0.1)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        
        correct_all[test_idx] = (y_pred == y_te).astype(int)
        
    acc = np.mean(correct_all)
    return acc, correct_all

# ----------------- 3. 连续显著时间窗定位 -----------------
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
        else:
            pass
            
    if in_window:
        end_idx = len(sig_mask) - 1
        duration = (end_idx - start_idx + 1) * dt
        if duration >= min_duration:
            windows.append((time_ms[start_idx], time_ms[end_idx]))
            
    return windows

# ----------------- 主程序 -----------------
def main():
    print("="*85)
    print("Step 3_1: Pure Color Block SVM Decoding with memory_color Electrodes on ERP")
    print("="*85)
    
    # 1. 载入 ERP 记忆显著通道表
    sig_path = os.path.join(doc_dir, 'select_channel_memory_significance_erp.csv')
    if not os.path.exists(sig_path):
        print("[ERROR] ERP 记忆显著通道表不存在，请确保完成了 Step 2_1！")
        return
        
    df_sig = pd.read_csv(sig_path)
    df_memory_color = df_sig[df_sig['Sig_Category'] != 'Non_Sig']
    
    if df_memory_color.empty:
        print("[WARNING] 无任何 memory_color 显著的 ERP 电极！跳过计算")
        return
        
    print(f"Total memory-selective ERP electrodes: {len(df_memory_color)}")
    
    # 确定各被试的电极名单
    subj_elecs = {}
    for subj in subjects:
        df_subj = df_memory_color[df_memory_color['Subject'] == subj]
        subj_elecs[subj] = df_subj['Electrode'].astype(str).tolist()
        print(f"  Subject {subj}: {len(subj_elecs[subj])} electrodes -> {subj_elecs[subj]}")
        
    active_subjs = [s for s, e in subj_elecs.items() if len(e) > 0]
    if not active_subjs:
        print("[WARNING] 无任何被试包含有效的 memory_color 显著 ERP 电极！")
        return
        
    # 2. 读取各被试的 task3 ERP 数据并运行解码
    correct_data = {}
    subj_accs = {}
    time_ms = None
    
    for subj in active_subjs:
        mat_path = os.path.join(feature_dir, subj, 'task3_ERP_epoched.mat')
        elecs = subj_elecs[subj]
        
        d_r, t_arr = get_data(mat_path, red_trigs, elecs)
        d_g, _ = get_data(mat_path, green_trigs, elecs)
        
        if d_r is None or d_g is None:
            print(f"  [ERROR] {subj} 缺少 task3 红/绿刺激数据，跳过该被试")
            continue
            
        if time_ms is None:
            time_ms = t_arr
            
        d_r, d_g = map(clean_data, [d_r, d_g])
        
        n_time = time_ms.shape[0]
        print(f"  Decoding {subj} (Red: {len(d_r)} trials, Green: {len(d_g)} trials)...")
        
        # 并行解码
        results = Parallel(n_jobs=-1)(
            delayed(fit_eval_kfold_single_t)(t, d_r, d_g, n_splits=5)
            for t in range(n_time)
        )
        
        accs_t = np.array([r[0] for r in results])
        corrects_t = [r[1] for r in results]
        
        subj_accs[subj] = accs_t
        correct_data[subj] = corrects_t
        
    # 3. 拟合多被试 GLMM 显著时间窗
    n_time = time_ms.shape[0]
    glmm_p_vals = np.zeros(n_time)
    glmm_est_accs = np.zeros(n_time)
    glmm_z_stats = np.zeros(n_time)
    
    print(f"\nFitting GLMM (random intercept by Subject) across {n_time} timepoints...")
    
    for t in range(n_time):
        y_list = []
        subj_list = []
        for subj in active_subjs:
            y_vector = correct_data[subj][t]
            y_list.append(y_vector)
            subj_list.extend([subj] * len(y_vector))
            
        df_t = pd.DataFrame({
            'Y': np.concatenate(y_list),
            'Subject': subj_list
        })
        
        try:
            model = bmg.BinomialBayesMixedGLM.from_formula(
                'Y ~ 1',
                {'Subject': '0 + C(Subject)'},
                data=df_t
            )
            res = model.fit_vb()
            fe_mean = res.fe_mean[0]
            fe_sd = res.fe_sd[0]
            z_stat = fe_mean / fe_sd
            
            p_val = 1.0 - stats.norm.cdf(z_stat)
            est_acc = 1.0 / (1.0 + np.exp(-fe_mean))
            
            glmm_p_vals[t] = p_val
            glmm_est_accs[t] = est_acc
            glmm_z_stats[t] = z_stat
        except Exception as e:
            glmm_p_vals[t] = 1.0
            glmm_est_accs[t] = df_t['Y'].mean()
            glmm_z_stats[t] = 0.0
            
    sig_windows = find_significant_windows(glmm_p_vals, time_ms, p_thresh=0.05, min_duration=20)
    print(f"  GLMM Significant Windows (>20ms): {sig_windows}")
    
    # 4. 保存统计明细
    group_acc_list = [subj_accs[s] for s in active_subjs]
    group_mean_acc = np.mean(group_acc_list, axis=0)
    
    export_dict = {
        'Time_ms': time_ms,
        'Group_Mean_Acc': group_mean_acc,
        'GLMM_Est_Acc': glmm_est_accs,
        'GLMM_Z': glmm_z_stats,
        'GLMM_P': glmm_p_vals
    }
    for s in subjects:
        if s in active_subjs:
            export_dict[f'{s}_Acc'] = subj_accs[s]
        else:
            export_dict[f'{s}_Acc'] = np.nan
            
    df_export = pd.DataFrame(export_dict)
    xlsx_path = os.path.join(doc_dir, 'decoding_data_erp_color_block.xlsx')
    csv_path = os.path.join(doc_dir, 'decoding_data_erp_color_block.csv')
    df_export.to_excel(xlsx_path, index=False)
    df_export.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Saved decoding data to:\n  - {xlsx_path}\n  - {csv_path}")
    
    # 5. 绘制多被试曲线与 GLMM 显著窗图
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    t_idx_plot = np.where((time_ms >= -200) & (time_ms <= 800))[0]
    time_plot = time_ms[t_idx_plot]
    
    subj_colors = {
        'test001': '#ff7f0e',
        'test002': '#2ca02c',
        'test003': '#1f77b4',
        'test005': '#9467bd',
        'test006': '#8c564b'
    }
    for subj in active_subjs:
        acc_plot = subj_accs[subj][t_idx_plot]
        ax.plot(time_plot, acc_plot, color=subj_colors[subj], lw=1.3, linestyle='--', alpha=0.55, label=f"Subj: {subj}")
        
    mean_plot = group_mean_acc[t_idx_plot]
    ax.plot(time_plot, mean_plot, color='#6f2da8', lw=3.5, label='Group Average')
    
    ax.axhline(0.5, color='#9e9e9e', linestyle=':', lw=1.5, label='Chance Level (50%)')
    ax.axvline(0, color='#757575', linestyle='-', lw=1.2)
    
    # 显著时区绘制
    y_line_val = 0.73
    has_shaded = False
    for start, end in sig_windows:
        if end < -200 or start > 800:
            continue
        s_plot = max(start, -200)
        e_plot = min(end, 800)
        
        ax.axvspan(s_plot, e_plot, color='#d62728', alpha=0.12, zorder=1)
        
        label_line = 'GLMM Significant (p < 0.05, >20ms)' if not has_shaded else ""
        ax.plot([s_plot, e_plot], [y_line_val, y_line_val], color='#d62728', lw=4.5, solid_capstyle='butt', label=label_line, zorder=4)
        has_shaded = True
        
    ax.set_title("ERP Pure Color Block SVM Decoding Performance\nUsing memory_color Electrodes (Active Subjects N = {})".format(len(active_subjs)), 
                 fontsize=13.5, fontweight='bold', pad=12)
    ax.set_xlabel("Time relative to stimulus onset (ms)", fontsize=11)
    ax.set_ylabel("Decoding Accuracy", fontsize=11)
    ax.set_xlim([-200, 800])
    ax.set_ylim([0.35, 0.76])
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.set_facecolor('#fafafa')
    ax.legend(loc='lower left', framealpha=0.9, fontsize=9.5)
    
    plt.tight_layout()
    out_fig = os.path.join(out_fig_dir, 'erp_color_block_decoding.png')
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved decoding figure to: {out_fig}")
    print("="*85)

if __name__ == '__main__':
    main()
