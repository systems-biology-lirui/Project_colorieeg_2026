import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
from pymatreader import read_mat
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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

# 图像保存主目录
out_fig_dir = os.path.join(result_dir, 'select_channel', 'decoding', 'color_with_sti')
os.makedirs(out_fig_dir, exist_ok=True)
os.makedirs(os.path.join(out_fig_dir, 'single_electrode'), exist_ok=True)

subjects = ['test001', 'test002', 'test003']

# 8个真假水果 Trigger 定义 (Task 2)
strawberry_trigs = ['Trigger-In:121', 'Trigger-In:122']
watermelon_trigs = ['Trigger-In:131', 'Trigger-In:132']
cabbage_trigs = ['Trigger-In:102', 'Trigger-In:101']
kiwi_trigs = ['Trigger-In:112', 'Trigger-In:111']

# 灰色水果 Triggers (Task 2)
r1_trigs = ['Trigger-In:123'] # 灰色草莓
r2_trigs = ['Trigger-In:133'] # 灰色西瓜
g1_trigs = ['Trigger-In:103'] # 灰色卷心菜
g2_trigs = ['Trigger-In:113'] # 灰色猕猴桃

# 纯色刺激 Triggers (Task 3)
task3_red = ['Trigger-In:51']
task3_green = ['Trigger-In:54']

# ----------------- 1. 获取 color_with_sti 电极 -----------------
def get_color_with_sti_electrodes():
    subj_elecs = {}
    for subj in subjects:
        p = os.path.join(base_dir, 'processed_data', subj, f'{subj}_ieegloc.xlsx')
        if not os.path.exists(p):
            subj_elecs[subj] = []
            continue
        df_loc = pd.read_excel(p)
        aal_col = 'AAL3 (MNI-linear)' if 'AAL3 (MNI-linear)' in df_loc.columns else 'AAL3 (MNI-segment)'
        ch_col = 'Channel' if 'Channel' in df_loc.columns else df_loc.columns[0]
        
        is_color_sti = df_loc[aal_col].astype(str).str.lower().str.replace('-', '_').str.replace(' ', '_') == 'color_with_sti'
        elecs = df_loc[is_color_sti]['Channel'].tolist()
        
        # 去重并保持顺序
        seen = set()
        unique_elecs = []
        for e in elecs:
            e_clean = str(e).strip()
            if e_clean not in seen:
                seen.add(e_clean)
                unique_elecs.append(e_clean)
        subj_elecs[subj] = unique_elecs
    return subj_elecs

# ----------------- 2. 数据提取核心方法 -----------------
def clean_data(x):
    if x is None:
        return None
    return x[~np.isnan(x).any(axis=(1,2))]

def get_data_task2(mat_path, trigs, elecs):
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
            
        data_list = []
        for t in trigs:
            if t in all_trigs:
                idx = all_trigs.index(t)
                trial_data = epoch['data'][idx, :, :, :]
                trial_data = trial_data[:, ch_indices, :]
                baseline_mask = time_ms < 0
                baseline_indices = np.where(baseline_mask)[0]
                if len(baseline_indices) > 0:
                    mean_bl = np.mean(trial_data[:, :, baseline_indices], axis=2, keepdims=True)
                    trial_data = trial_data - mean_bl
                data_list.append(trial_data)
        if not data_list:
            return None, None
        return np.concatenate(data_list, axis=0), time_ms
    except Exception as e:
        print(f"  [ERROR] Loading task2 data failed: {e}")
        return None, None

def get_data_task3(mat_path, trigs, elecs):
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
            
        data_list = []
        for t in trigs:
            if t in all_trigs:
                idx = all_trigs.index(t)
                trial_data = epoch['data'][idx, :, :, :]
                trial_data = trial_data[:, ch_indices, :]
                baseline_mask = time_ms < 0
                baseline_indices = np.where(baseline_mask)[0]
                if len(baseline_indices) > 0:
                    mean_bl = np.mean(trial_data[:, :, baseline_indices], axis=2, keepdims=True)
                    trial_data = trial_data - mean_bl
                data_list.append(trial_data)
        if not data_list:
            return None, None
        return np.concatenate(data_list, axis=0), time_ms
    except Exception as e:
        print(f"  [ERROR] Loading task3 data failed: {e}")
        return None, None

# ----------------- 3. 解码方法 -----------------
def fit_eval_cross_pairs_t(t, r1, r2, g1, g2):
    pairs = [(r1, g1, r2, g2), (r1, g2, r2, g1), (r2, g1, r1, g2), (r2, g2, r1, g1)]
    correct_all_list = []
    for tr_r, tr_g, te_r, te_g in pairs:
        X_tr = np.vstack([tr_r[:, :, t], tr_g[:, :, t]])
        y_tr = np.hstack([np.zeros(tr_r.shape[0]), np.ones(tr_g.shape[0])])
        X_te = np.vstack([te_r[:, :, t], te_g[:, :, t]])
        y_te = np.hstack([np.zeros(te_r.shape[0]), np.ones(te_g.shape[0])])
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        clf = SVC(kernel='linear', C=0.1)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        correct_all_list.append((y_pred == y_te).astype(int))
    correct_all = np.concatenate(correct_all_list)
    return np.mean(correct_all), correct_all

def fit_eval_kfold_single_t(t, data_r, data_g, n_splits=5):
    X = np.vstack([data_r[:, :, t], data_g[:, :, t]])
    y = np.hstack([np.zeros(data_r.shape[0]), np.ones(data_g.shape[0])])
    from sklearn.model_selection import KFold
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
    return np.mean(correct_all), correct_all

def fit_eval_logo_single_t(t, group_trials):
    group_names = ['strawberry', 'watermelon', 'cabbage', 'kiwi']
    g_features = {}
    g_labels = {}
    for g in group_names:
        trigs = {'strawberry': strawberry_trigs, 'watermelon': watermelon_trigs, 'cabbage': cabbage_trigs, 'kiwi': kiwi_trigs}[g]
        t_red, t_green = trigs[0], trigs[1]
        d_red = group_trials[g][t_red][:, :, t]
        d_green = group_trials[g][t_green][:, :, t]
        feat = np.vstack([d_red, d_green])
        lbl = np.hstack([np.zeros(d_red.shape[0]), np.ones(d_green.shape[0])])
        g_features[g] = feat
        g_labels[g] = lbl
    correct_all_list = []
    for test_g in group_names:
        train_groups = [g for g in group_names if g != test_g]
        X_tr = np.vstack([g_features[g] for g in train_groups])
        y_tr = np.concatenate([g_labels[g] for g in train_groups])
        X_te = g_features[test_g]
        y_te = g_labels[test_g]
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        clf = SVC(kernel='linear', C=0.1)
        clf.fit(X_tr_scaled, y_tr)
        y_pred = clf.predict(X_te_scaled)
        correct_all_list.append((y_pred == y_te).astype(int))
    correct_all = np.concatenate(correct_all_list)
    return np.mean(correct_all), correct_all

def fit_eval_cross_task_1d_t(t, train_r, train_g, test_r1, test_r2, test_g1, test_g2):
    """红绿色块训练，灰色水果测试的一维时程"""
    X_tr = np.vstack([train_r[:, :, t], train_g[:, :, t]])
    y_tr = np.hstack([np.zeros(train_r.shape[0]), np.ones(train_g.shape[0])])
    
    pairs = [
        (test_r1[:, :, t], test_g1[:, :, t]),
        (test_r1[:, :, t], test_g2[:, :, t]),
        (test_r2[:, :, t], test_g1[:, :, t]),
        (test_r2[:, :, t], test_g2[:, :, t])
    ]
    correct_all_list = []
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    
    clf = SVC(kernel='linear', C=0.1)
    clf.fit(X_tr, y_tr)
    
    for te_r, te_g in pairs:
        X_te = np.vstack([te_r, te_g])
        y_te = np.hstack([np.zeros(te_r.shape[0]), np.ones(te_g.shape[0])])
        X_te = scaler.transform(X_te)
        y_pred = clf.predict(X_te)
        correct_all_list.append((y_pred == y_te).astype(int))
    correct_all = np.concatenate(correct_all_list)
    return np.mean(correct_all), correct_all

# ----------------- 4. 统计工具 -----------------
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

def run_glmm(correct_data, active_subjs, time_ms):
    n_time = time_ms.shape[0]
    p_vals = np.zeros(n_time)
    est_accs = np.zeros(n_time)
    z_stats = np.zeros(n_time)
    
    if len(active_subjs) >= 2:
        for t in range(n_time):
            y_list = []
            subj_list = []
            for subj in active_subjs:
                y_vector = correct_data[subj][t]
                y_list.append(y_vector)
                subj_list.extend([subj] * len(y_vector))
            df_t = pd.DataFrame({'Y': np.concatenate(y_list), 'Subject': subj_list})
            try:
                model = bmg.BinomialBayesMixedGLM.from_formula('Y ~ 1', {'Subject': '0 + C(Subject)'}, data=df_t)
                res = model.fit_vb()
                fe_mean = res.fe_mean[0]
                fe_sd = res.fe_sd[0]
                z_stat = fe_mean / fe_sd
                p_val = 1.0 - stats.norm.cdf(z_stat)
                est_acc = 1.0 / (1.0 + np.exp(-fe_mean))
                p_vals[t] = p_val
                est_accs[t] = est_acc
                z_stats[t] = z_stat
            except Exception:
                p_vals[t] = 1.0
                est_accs[t] = df_t['Y'].mean()
                z_stats[t] = 0.0
    else:
        single_subj = active_subjs[0]
        for t in range(n_time):
            y_vector = correct_data[single_subj][t]
            k = np.sum(y_vector)
            n = len(y_vector)
            p_val = stats.binomtest(k, n, p=0.5, alternative='greater').pvalue
            p_vals[t] = p_val
            est_accs[t] = k / n if n > 0 else 0.5
            z_stats[t] = 0.0
    return p_vals, est_accs, z_stats

# ----------------- 5. 绘图模板 -----------------
def plot_decoding_1d(time_ms, acc_dict, group_mean_acc, sig_windows, active_subjs, title, filename):
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    t_idx = np.where((time_ms >= -200) & (time_ms <= 800))[0]
    t_plot = time_ms[t_idx]
    
    colors = {'test001': '#ff7f0e', 'test002': '#2ca02c', 'test003': '#1f77b4', 'test005': '#9467bd', 'test006': '#8c564b'}
    for subj in active_subjs:
        ax.plot(t_plot, acc_dict[subj][t_idx], color=colors[subj], lw=1.2, linestyle='--', alpha=0.5, label=f"Subj: {subj}")
    ax.plot(t_plot, group_mean_acc[t_idx], color='#2c3e50', lw=3.2, label='Group Average')
    ax.axhline(0.5, color='#95a5a6', linestyle=':', lw=1.5, label='Chance (50%)')
    ax.axvline(0, color='#7f8c8d', linestyle='-', lw=1.2)
    
    has_shaded = False
    y_line = 0.73
    sig_label = 'GLMM Significant (p<0.05, >20ms)' if len(active_subjs) >= 2 else 'Binomial Significant (p<0.05, >20ms)'
    for start, end in sig_windows:
        if end < -200 or start > 800:
            continue
        s_plot = max(start, -200)
        e_plot = min(end, 800)
        ax.axvspan(s_plot, e_plot, color='#d62728', alpha=0.1, zorder=1)
        lbl = sig_label if not has_shaded else ""
        ax.plot([s_plot, e_plot], [y_line, y_line], color='#d62728', lw=4.0, label=lbl, zorder=4)
        has_shaded = True
        
    ax.set_title(title, fontsize=12.5, fontweight='bold', pad=10)
    ax.set_xlabel("Time relative to stimulus onset (ms)", fontsize=10.5)
    ax.set_ylabel("Decoding Accuracy", fontsize=10.5)
    ax.set_xlim([-200, 800])
    ax.set_ylim([0.35, 0.76])
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_facecolor('#fafafa')
    ax.legend(loc='lower left', framealpha=0.9, fontsize=9.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ----------------- 主程序 -----------------
def main():
    print("="*90)
    print("Step 7: Prior Electrodes color_with_sti Comprehensive Pipeline")
    print("="*90)
    
    # 1. 提取电极名单
    subj_elecs = get_color_with_sti_electrodes()
    print("color_with_sti Prior Electrode Coverage:")
    for subj in subjects:
        print(f"  - {subj}: count = {len(subj_elecs[subj])} | {subj_elecs[subj]}")
        
    active_subjs = [s for s in subjects if len(subj_elecs[s]) > 0]
    if len(active_subjs) == 0:
        print("[ERROR] 没有被试覆盖 color_with_sti 电极！程序退出")
        return
        
    # ##################################################################
    # 7_1. Color vs Gray 信号差异分析
    # ##################################################################
    print("\n" + "-"*50)
    print("7_1. Single-Electrode Color vs. Gray Signal Difference Analysis")
    print("-"*50)
    
    results_7_1 = []
    # 有色和灰色刺激对应条件 (Task 2)
    color_trigs = ['Trigger-In:101', 'Trigger-In:102', 'Trigger-In:111', 'Trigger-In:112', 
                   'Trigger-In:121', 'Trigger-In:122', 'Trigger-In:131', 'Trigger-In:132']
    gray_trigs = ['Trigger-In:103', 'Trigger-In:113', 'Trigger-In:123', 'Trigger-In:133']
    
    for subj in active_subjs:
        mat_path = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
        for elec in subj_elecs[subj]:
            print(f"  Analysing {subj} - {elec} for Color vs. Gray...")
            
            c_data, time_ms = get_data_task2(mat_path, color_trigs, [elec])
            g_data, _ = get_data_task2(mat_path, gray_trigs, [elec])
            
            if c_data is None or g_data is None:
                continue
                
            # 去除 (Rep, 1, Time) 中的第二维，变成 (Rep, Time)
            c_data = clean_data(c_data)[:, 0, :]
            g_data = clean_data(g_data)[:, 0, :]
            
            n_time = time_ms.shape[0]
            p_vals_t = np.zeros(n_time)
            for t in range(n_time):
                _, p = stats.ranksums(c_data[:, t], g_data[:, t])
                p_vals_t[t] = p
                
            sig_windows = find_cont_sig_windows(p_vals_t, time_ms, p_thresh=0.05, min_duration=50)
            
            # 200-500ms 均值差异
            t_idx_200_500 = np.where((time_ms >= 200) & (time_ms <= 500))[0]
            mean_c_200_500 = np.mean(c_data[:, t_idx_200_500], axis=1)
            mean_g_200_500 = np.mean(g_data[:, t_idx_200_500], axis=1)
            stat_m, p_m = stats.ranksums(mean_c_200_500, mean_g_200_500)
            
            results_7_1.append({
                'Subject': subj,
                'Electrode': elec,
                'Mean_Color_200_500': np.mean(mean_c_200_500),
                'Mean_Gray_200_500': np.mean(mean_g_200_500),
                'Mean_Diff_200_500': np.mean(mean_c_200_500) - np.mean(mean_g_200_500),
                'Wilcoxon_Z': stat_m,
                'Wilcoxon_P': p_m,
                'Is_Mean_Sig_200_500': p_m < 0.05,
                'Has_Cont_50ms_Sig': len(sig_windows) > 0,
                'Cont_Sig_Windows': str(sig_windows)
            })
            
            # 1行2列 信号差异图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [2.5, 1]}, dpi=300)
            t_plot_idx = np.where((time_ms >= -200) & (time_ms <= 800))[0]
            t_plot = time_ms[t_plot_idx]
            
            m_c_plot = np.mean(c_data[:, t_plot_idx], axis=0)
            m_g_plot = np.mean(g_data[:, t_plot_idx], axis=0)
            sem_c = stats.sem(c_data[:, t_plot_idx], axis=0)
            sem_g = stats.sem(g_data[:, t_plot_idx], axis=0)
            
            ax1.plot(t_plot, m_c_plot, color='#e74c3c', lw=2.5, label=f'Color Fruits (N={len(c_data)})')
            ax1.fill_between(t_plot, m_c_plot - sem_c, m_c_plot + sem_c, color='#e74c3c', alpha=0.15)
            
            ax1.plot(t_plot, m_g_plot, color='#7f8c8d', lw=2.5, label=f'Gray Fruits (N={len(g_data)})')
            ax1.fill_between(t_plot, m_g_plot - sem_g, m_g_plot + sem_g, color='#7f8c8d', alpha=0.15)
            
            ax1.axhline(0, color='#7f8c8d', linestyle=':', lw=1.2)
            ax1.axvline(0, color='#7f8c8d', linestyle='-', lw=1.2)
            
            has_shading = False
            for start, end in sig_windows:
                if end < -200 or start > 800:
                    continue
                s_p = max(start, -200)
                e_p = min(end, 800)
                ax1.axvspan(s_p, e_p, color='#d35400', alpha=0.1, zorder=1)
                lbl = 'Cont. Sig. (p<0.05, >50ms)' if not has_shading else ''
                ax1.plot([s_p, e_p], [ax1.get_ylim()[0] + 0.05 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])] * 2, 
                         color='#d35400', lw=4.0, label=lbl, zorder=4)
                has_shading = True
                
            ax1.set_title(f"ERP Evoked Signal Color vs. Gray: {subj} - {elec}", fontsize=11, fontweight='bold')
            ax1.set_xlabel("Time relative to stimulus onset (ms)", fontsize=10)
            ax1.set_ylabel("Amplitude (μV)", fontsize=10)
            ax1.set_xlim([-200, 800])
            ax1.grid(True, linestyle=':', alpha=0.45)
            ax1.set_facecolor('#fafafa')
            ax1.legend(loc='upper right', framealpha=0.9, fontsize=9.0)
            
            # 箱线图
            bp = ax2.boxplot([mean_c_200_500, mean_g_200_500], widths=0.45, patch_artist=True,
                             boxprops=dict(facecolor='#fafafa', color='#2c3e50', linewidth=1.2),
                             medianprops=dict(color='#e74c3c', linewidth=1.8))
            bp['boxes'][0].set_facecolor('#e74c3c')
            bp['boxes'][0].set_alpha(0.6)
            bp['boxes'][1].set_facecolor('#7f8c8d')
            bp['boxes'][1].set_alpha(0.6)
            
            ax2.set_xticklabels(['Color', 'Gray'], fontsize=9.5)
            ax2.set_ylabel("Mean Amplitude (200-500ms, μV)", fontsize=10)
            ax2.grid(True, linestyle=':', alpha=0.3)
            ax2.set_facecolor('#fafafa')
            
            y_max = max(np.max(mean_c_200_500), np.max(mean_g_200_500))
            y_min = min(np.min(mean_c_200_500), np.min(mean_g_200_500))
            h = (y_max - y_min) * 0.05
            sig_str = '***' if p_m < 0.001 else ('**' if p_m < 0.01 else ('*' if p_m < 0.05 else 'n.s.'))
            ax2.plot([1, 1, 2, 2], [y_max + h, y_max + 2*h, y_max + 2*h, y_max + h], color='#2c3e50', lw=1.2)
            ax2.text(1.5, y_max + 2.2*h, sig_str, ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax2.set_title("200-500ms Average", fontsize=10)
            
            plt.suptitle(f"Color vs. Gray Evoked Signal Difference | {subj} - {elec}", fontsize=13, fontweight='bold', y=0.98)
            plt.tight_layout()
            out_img = os.path.join(out_fig_dir, 'single_electrode', f"{subj}_{elec}_color_vs_gray_difference.png")
            plt.savefig(out_img, dpi=300)
            plt.close()
            
    df_7_1 = pd.DataFrame(results_7_1)
    df_7_1.to_excel(os.path.join(doc_dir, 'color_with_sti_color_vs_gray_erp_stats.xlsx'), index=False)
    df_7_1.to_csv(os.path.join(doc_dir, 'color_with_sti_color_vs_gray_erp_stats.csv'), index=False)
    
    # ##################################################################
    # 7_2. Memory Color Decoding
    # ##################################################################
    print("\n" + "-"*50)
    print("7_2. Memory Color Decoding (Red Memory vs. Green Memory in Gray Fruits)")
    print("-"*50)
    
    correct_data_7_2 = {}
    acc_data_7_2 = {}
    time_ms = None
    
    for subj in active_subjs:
        mat_path = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
        elecs = subj_elecs[subj]
        trigs_list = {'r1': r1_trigs, 'r2': r2_trigs, 'g1': g1_trigs, 'g2': g2_trigs}
        data_dict, t_arr = get_data_task2_gray(mat_path, trigs_list, elecs)
        
        if data_dict is None or any(v is None for v in data_dict.values()):
            continue
        if time_ms is None:
            time_ms = t_arr
            
        r1_arr, r2_arr, g1_arr, g2_arr = map(clean_data, [data_dict['r1'], data_dict['r2'], data_dict['g1'], data_dict['g2']])
        n_time = time_ms.shape[0]
        
        print(f"  Decoding {subj}...")
        results = Parallel(n_jobs=-1)(
            delayed(fit_eval_cross_pairs_t)(t, r1_arr, r2_arr, g1_arr, g2_arr) for t in range(n_time)
        )
        acc_data_7_2[subj] = np.array([r[0] for r in results])
        correct_data_7_2[subj] = [r[1] for r in results]
        
    p_vals_7_2, est_accs_7_2, z_stats_7_2 = run_glmm(correct_data_7_2, active_subjs, time_ms)
    sig_windows_7_2 = find_significant_windows(p_vals_7_2, time_ms, p_thresh=0.05, min_duration=20)
    print(f"  GLMM Significant Windows: {sig_windows_7_2}")
    
    group_mean_acc_7_2 = np.mean([acc_data_7_2[s] for s in active_subjs], axis=0)
    export_7_2 = {'Time_ms': time_ms, 'Group_Mean_Acc': group_mean_acc_7_2, 'GLMM_Est_Acc': est_accs_7_2, 'Stats_P': p_vals_7_2, 'Stats_Z': z_stats_7_2}
    for s in subjects:
        export_7_2[f'{s}_Acc'] = acc_data_7_2[s] if s in active_subjs else np.nan
    pd.DataFrame(export_7_2).to_excel(os.path.join(doc_dir, 'decoding_data_erp_color_with_sti_memory_color.xlsx'), index=False)
    
    plot_decoding_1d(time_ms, acc_data_7_2, group_mean_acc_7_2, sig_windows_7_2, active_subjs,
                     "ERP Memory Color Decoding: ROI [color_with_sti]\n(Active Subjects N = {})".format(len(active_subjs)),
                     os.path.join(out_fig_dir, 'erp_color_with_sti_memory_color_decoding.png'))
                     
    # ##################################################################
    # 7_3. Color Block Decoding
    # ##################################################################
    print("\n" + "-"*50)
    print("7_3. Pure Color Block Decoding (Task 3 Red vs. Green)")
    print("-"*50)
    
    correct_data_7_3 = {}
    acc_data_7_3 = {}
    
    for subj in active_subjs:
        mat_path = os.path.join(feature_dir, subj, 'task3_ERP_epoched.mat')
        elecs = subj_elecs[subj]
        d_red, _ = get_data_task3(mat_path, task3_red, elecs)
        d_green, _ = get_data_task3(mat_path, task3_green, elecs)
        
        if d_red is None or d_green is None:
            continue
        d_red, d_green = map(clean_data, [d_red, d_green])
        n_time = time_ms.shape[0]
        
        print(f"  Decoding {subj}...")
        results = Parallel(n_jobs=-1)(
            delayed(fit_eval_kfold_single_t)(t, d_red, d_green, n_splits=5) for t in range(n_time)
        )
        acc_data_7_3[subj] = np.array([r[0] for r in results])
        correct_data_7_3[subj] = [r[1] for r in results]
        
    p_vals_7_3, est_accs_7_3, z_stats_7_3 = run_glmm(correct_data_7_3, active_subjs, time_ms)
    sig_windows_7_3 = find_significant_windows(p_vals_7_3, time_ms, p_thresh=0.05, min_duration=20)
    print(f"  GLMM Significant Windows: {sig_windows_7_3}")
    
    group_mean_acc_7_3 = np.mean([acc_data_7_3[s] for s in active_subjs], axis=0)
    export_7_3 = {'Time_ms': time_ms, 'Group_Mean_Acc': group_mean_acc_7_3, 'GLMM_Est_Acc': est_accs_7_3, 'Stats_P': p_vals_7_3, 'Stats_Z': z_stats_7_3}
    for s in subjects:
        export_7_3[f'{s}_Acc'] = acc_data_7_3[s] if s in active_subjs else np.nan
    pd.DataFrame(export_7_3).to_excel(os.path.join(doc_dir, 'decoding_data_erp_color_with_sti_color_block.xlsx'), index=False)
    
    plot_decoding_1d(time_ms, acc_data_7_3, group_mean_acc_7_3, sig_windows_7_3, active_subjs,
                     "ERP Pure Color Block Decoding: ROI [color_with_sti]\n(Active Subjects N = {})".format(len(active_subjs)),
                     os.path.join(out_fig_dir, 'erp_color_with_sti_color_block_decoding.png'))

    # ##################################################################
    # 7_4. Cross Decoding (Task 3 Train, Task 2 Test)
    # ##################################################################
    print("\n" + "-"*50)
    print("7_4. Cross Task Decoding (Color Block Train, Gray Fruits Test)")
    print("-"*50)
    
    # A. 1D对角线时程解码
    correct_data_7_4 = {}
    acc_data_7_4 = {}
    
    for subj in active_subjs:
        m3_path = os.path.join(feature_dir, subj, 'task3_ERP_epoched.mat')
        m2_path = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
        elecs = subj_elecs[subj]
        
        train_r, _ = get_data_task3(m3_path, task3_red, elecs)
        train_g, _ = get_data_task3(m3_path, task3_green, elecs)
        
        trigs_list = {'r1': r1_trigs, 'r2': r2_trigs, 'g1': g1_trigs, 'g2': g2_trigs}
        gray_dict, _ = get_data_task2_gray(m2_path, trigs_list, elecs)
        
        if train_r is None or train_g is None or gray_dict is None or any(v is None for v in gray_dict.values()):
            continue
            
        train_r, train_g = map(clean_data, [train_r, train_g])
        test_r1, test_r2, test_g1, test_g2 = map(clean_data, [gray_dict['r1'], gray_dict['r2'], gray_dict['g1'], gray_dict['g2']])
        n_time = time_ms.shape[0]
        
        print(f"  1D Cross Decoding {subj}...")
        results = Parallel(n_jobs=-1)(
            delayed(fit_eval_cross_task_1d_t)(t, train_r, train_g, test_r1, test_r2, test_g1, test_g2)
            for t in range(n_time)
        )
        acc_data_7_4[subj] = np.array([r[0] for r in results])
        correct_data_7_4[subj] = [r[1] for r in results]
        
    p_vals_7_4, est_accs_7_4, z_stats_7_4 = run_glmm(correct_data_7_4, active_subjs, time_ms)
    sig_windows_7_4 = find_significant_windows(p_vals_7_4, time_ms, p_thresh=0.05, min_duration=20)
    print(f"  GLMM Significant Windows: {sig_windows_7_4}")
    
    group_mean_acc_7_4 = np.mean([acc_data_7_4[s] for s in active_subjs], axis=0)
    export_7_4 = {'Time_ms': time_ms, 'Group_Mean_Acc': group_mean_acc_7_4, 'GLMM_Est_Acc': est_accs_7_4, 'Stats_P': p_vals_7_4, 'Stats_Z': z_stats_7_4}
    for s in subjects:
        export_7_4[f'{s}_Acc'] = acc_data_7_4[s] if s in active_subjs else np.nan
    pd.DataFrame(export_7_4).to_excel(os.path.join(doc_dir, 'decoding_data_erp_color_with_sti_cross_decoding_1d.xlsx'), index=False)
    
    plot_decoding_1d(time_ms, acc_data_7_4, group_mean_acc_7_4, sig_windows_7_4, active_subjs,
                     "ERP Cross-Task 1D Decoding (Diag): ROI [color_with_sti]\n(Active Subjects N = {})".format(len(active_subjs)),
                     os.path.join(out_fig_dir, 'erp_color_with_sti_cross_decoding_1d.png'))

    # B. 2D时间泛化 (TG)
    print("  Calculating 2D Temporal Generalization (重采样到 10ms)...")
    # 下采样映射
    t_idx_tg = np.where((time_ms >= -100) & (time_ms <= 700))[0]
    time_tg_orig = time_ms[t_idx_tg]
    
    # 均值下采样成 10ms (5个数据点取均值)
    step = 5
    n_bins = len(time_tg_orig) // step
    time_tg_ds = np.array([np.mean(time_tg_orig[i*step : (i+1)*step]) for i in range(n_bins)])
    
    def get_ds_data(data):
        # data: (Rep, Ch, Time) -> 下采样后为 (Rep, Ch, n_bins)
        rep, ch, _ = data.shape
        ds_arr = np.zeros((rep, ch, n_bins))
        for i in range(n_bins):
            ds_arr[:, :, i] = np.mean(data[:, :, t_idx_tg[i*step : (i+1)*step]], axis=2)
        return ds_arr
        
    tg_results = {}
    for subj in active_subjs:
        m3_path = os.path.join(feature_dir, subj, 'task3_ERP_epoched.mat')
        m2_path = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
        elecs = subj_elecs[subj]
        
        train_r, _ = get_data_task3(m3_path, task3_red, elecs)
        train_g, _ = get_data_task3(m3_path, task3_green, elecs)
        
        trigs_list = {'r1': r1_trigs, 'r2': r2_trigs, 'g1': g1_trigs, 'g2': g2_trigs}
        gray_dict, _ = get_data_task2_gray(m2_path, trigs_list, elecs)
        
        if train_r is None or train_g is None or gray_dict is None:
            continue
            
        train_r, train_g = map(clean_data, [train_r, train_g])
        test_r1, test_r2, test_g1, test_g2 = map(clean_data, [gray_dict['r1'], gray_dict['r2'], gray_dict['g1'], gray_dict['g2']])
        
        tr_r_ds, tr_g_ds = map(get_ds_data, [train_r, train_g])
        te_r1_ds, te_r2_ds, te_g1_ds, te_g2_ds = map(get_ds_data, [test_r1, test_r2, test_g1, test_g2])
        
        print(f"    Calculating TG 2D Matrix for {subj}...")
        
        def fit_tg_row(t_tr):
            row_accs = np.zeros(n_bins)
            X_tr = np.vstack([tr_r_ds[:, :, t_tr], tr_g_ds[:, :, t_tr]])
            y_tr = np.hstack([np.zeros(tr_r_ds.shape[0]), np.ones(tr_g_ds.shape[0])])
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            clf = SVC(kernel='linear', C=0.1)
            clf.fit(X_tr, y_tr)
            
            for t_te in range(n_bins):
                pairs_te = [
                    (te_r1_ds[:, :, t_te], te_g1_ds[:, :, t_te]),
                    (te_r1_ds[:, :, t_te], te_g2_ds[:, :, t_te]),
                    (te_r2_ds[:, :, t_te], te_g1_ds[:, :, t_te]),
                    (te_r2_ds[:, :, t_te], te_g2_ds[:, :, t_te])
                ]
                acc_sum = 0
                for te_r, te_g in pairs_te:
                    X_te = np.vstack([te_r, te_g])
                    y_te = np.hstack([np.zeros(te_r.shape[0]), np.ones(te_g.shape[0])])
                    X_te = scaler.transform(X_te)
                    acc_sum += clf.score(X_te, y_te)
                row_accs[t_te] = acc_sum / 4.0
            return row_accs
            
        res_matrix = Parallel(n_jobs=-1)(
            delayed(fit_tg_row)(t_tr) for t_tr in range(n_bins)
        )
        tg_results[subj] = np.array(res_matrix)
        
    # 计算 Group 平均
    group_tg = np.mean([tg_results[s] for s in active_subjs], axis=0)
    
    # 导出 2D 数据为 Excel (平铺格式)
    tg_rows = []
    for i in range(n_bins):
        for j in range(n_bins):
            row_dict = {
                'TrainTime_ms': time_tg_ds[i],
                'TestTime_ms': time_tg_ds[j],
                'Group_Acc': group_tg[i, j]
            }
            for s in active_subjs:
                row_dict[f'{s}_Acc'] = tg_results[s][i, j]
            tg_rows.append(row_dict)
    pd.DataFrame(tg_rows).to_excel(os.path.join(doc_dir, 'cross_decoding_tg_color_with_sti.xlsx'), index=False)
    
    # 绘制 2D 热图
    heatmaps = {'Group': group_tg}
    for s in active_subjs:
        heatmaps[s] = tg_results[s]
        
    for name, mat in heatmaps.items():
        fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
        im = ax.imshow(mat, cmap='jet', origin='lower', extent=[-100, 700, -100, 700], vmin=0.45, vmax=0.58)
        ax.set_title(f"TG 2D Heatmap: {name} Level [color_with_sti]\n(Train: Task3 Pure Color | Test: Task2 Gray Fruits)", fontsize=11.5, fontweight='bold')
        ax.set_xlabel("Testing Time (Task 2, ms)", fontsize=10)
        ax.set_ylabel("Training Time (Task 3, ms)", fontsize=10)
        ax.axhline(0, color='white', linestyle='--', alpha=0.6)
        ax.axvline(0, color='white', linestyle='--', alpha=0.6)
        
        # 加上对角线
        ax.plot([-100, 700], [-100, 700], color='white', linestyle=':', lw=1.2, alpha=0.7)
        fig.colorbar(im, ax=ax, label='Decoding Accuracy')
        plt.tight_layout()
        out_heatmap = os.path.join(out_fig_dir, f"erp_color_with_sti_cross_decoding_tg_heatmap_{name.lower()}.png")
        plt.savefig(out_heatmap, dpi=300)
        plt.close()
        print(f"    [SAVED HEATMAP] -> {out_heatmap}")

    # ##################################################################
    # 7_5. True vs Fake Decoding
    # ##################################################################
    print("\n" + "-"*50)
    print("7_5. True vs. Fake Color Present Decoding (LOGO 4-fold)")
    print("-"*50)
    
    correct_data_7_5 = {}
    acc_data_7_5 = {}
    
    for subj in active_subjs:
        mat_path = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
        elecs = subj_elecs[subj]
        all_t_trigs = strawberry_trigs + watermelon_trigs + cabbage_trigs + kiwi_trigs
        data_dict, _ = get_data_task2_real_fake(mat_path, all_t_trigs, elecs)
        
        if data_dict is None or any(v is None for v in data_dict.values()):
            continue
        for k in data_dict.keys():
            data_dict[k] = clean_data(data_dict[k])
            
        group_trials = {
            'strawberry': {t: data_dict[t] for t in strawberry_trigs},
            'watermelon': {t: data_dict[t] for t in watermelon_trigs},
            'cabbage': {t: data_dict[t] for t in cabbage_trigs},
            'kiwi': {t: data_dict[t] for t in kiwi_trigs}
        }
        n_time = time_ms.shape[0]
        
        print(f"  Decoding {subj}...")
        results = Parallel(n_jobs=-1)(
            delayed(fit_eval_logo_single_t)(t, group_trials) for t in range(n_time)
        )
        acc_data_7_5[subj] = np.array([r[0] for r in results])
        correct_data_7_5[subj] = [r[1] for r in results]
        
    p_vals_7_5, est_accs_7_5, z_stats_7_5 = run_glmm(correct_data_7_5, active_subjs, time_ms)
    sig_windows_7_5 = find_significant_windows(p_vals_7_5, time_ms, p_thresh=0.05, min_duration=20)
    print(f"  GLMM Significant Windows: {sig_windows_7_5}")
    
    group_mean_acc_7_5 = np.mean([acc_data_7_5[s] for s in active_subjs], axis=0)
    export_7_5 = {'Time_ms': time_ms, 'Group_Mean_Acc': group_mean_acc_7_5, 'GLMM_Est_Acc': est_accs_7_5, 'Stats_P': p_vals_7_5, 'Stats_Z': z_stats_7_5}
    for s in subjects:
        export_7_5[f'{s}_Acc'] = acc_data_7_5[s] if s in active_subjs else np.nan
    pd.DataFrame(export_7_5).to_excel(os.path.join(doc_dir, 'decoding_data_erp_color_with_sti_real_fake.xlsx'), index=False)
    
    plot_decoding_1d(time_ms, acc_data_7_5, group_mean_acc_7_5, sig_windows_7_5, active_subjs,
                     "ERP True vs. Fake Fruit Color Decoding: ROI [color_with_sti]\n(Active Subjects N = {})".format(len(active_subjs)),
                     os.path.join(out_fig_dir, 'erp_color_with_sti_real_fake_decoding.png'))
                     
    print("\n" + "="*90)
    print("Step 7 successfully completed!")
    print("="*90)

# ----------------- 副本函数定义 -----------------
# 复制灰色水果读取以自洽
def get_data_task2_gray(mat_path, trigs_list, elecs):
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
            
        data_dict = {}
        for name, trigs in trigs_list.items():
            idx_list = [all_trigs.index(t) for t in trigs if t in all_trigs]
            if not idx_list:
                data_dict[name] = None
                continue
            trial_list = []
            for idx in idx_list:
                trial_data = epoch['data'][idx, :, :, :]
                trial_data = trial_data[:, ch_indices, :]
                baseline_mask = time_ms < 0
                baseline_indices = np.where(baseline_mask)[0]
                if len(baseline_indices) > 0:
                    mean_bl = np.mean(trial_data[:, :, baseline_indices], axis=2, keepdims=True)
                    trial_data = trial_data - mean_bl
                trial_list.append(trial_data)
            data_dict[name] = np.concatenate(trial_list, axis=0)
        return data_dict, time_ms
    except Exception as e:
        print(f"  [ERROR] Loading gray task2 data failed: {e}")
        return None, None

def get_data_task2_real_fake(mat_path, trigs_to_extract, elecs):
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
            
        data_dict = {}
        for t in trigs_to_extract:
            if t in all_trigs:
                idx = all_trigs.index(t)
                trial_data = epoch['data'][idx, :, :, :]
                trial_data = trial_data[:, ch_indices, :]
                baseline_mask = time_ms < 0
                baseline_indices = np.where(baseline_mask)[0]
                if len(baseline_indices) > 0:
                    mean_bl = np.mean(trial_data[:, :, baseline_indices], axis=2, keepdims=True)
                    trial_data = trial_data - mean_bl
                data_dict[t] = trial_data
            else:
                data_dict[t] = None
        return data_dict, time_ms
    except Exception as e:
        print(f"  [ERROR] Loading real_fake task2 data failed: {e}")
        return None, None

def find_cont_sig_windows(p_vals, time_ms, p_thresh=0.05, min_duration=50):
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

if __name__ == '__main__':
    main()
