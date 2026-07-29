import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
from pymatreader import read_mat
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
out_fig_dir = os.path.join(result_dir, 'select_channel', 'decoding', 'clusters')
os.makedirs(out_fig_dir, exist_ok=True)

subjects = ['test001', 'test002', 'test003']

# 8 个真假水果 Trigger 定义 (Task 2)
strawberry_trigs = ['Trigger-In:121', 'Trigger-In:122'] # 真红, 假绿
watermelon_trigs = ['Trigger-In:131', 'Trigger-In:132'] # 真红, 假绿
cabbage_trigs = ['Trigger-In:102', 'Trigger-In:101']    # 假红, 真绿
kiwi_trigs = ['Trigger-In:112', 'Trigger-In:111']       # 假红, 真绿

# 区分红绿标签 (红=0, 绿=1)
red_labels = ['Trigger-In:121', 'Trigger-In:131', 'Trigger-In:102', 'Trigger-In:112']
green_labels = ['Trigger-In:122', 'Trigger-In:132', 'Trigger-In:101', 'Trigger-In:111']

# 交叉配对灰色水果 Triggers (Task 2)
r1_trigs = ['Trigger-In:123'] # 灰色草莓
r2_trigs = ['Trigger-In:133'] # 灰色西瓜
g1_trigs = ['Trigger-In:103'] # 灰色卷心菜
g2_trigs = ['Trigger-In:113'] # 灰色猕猴桃

# ----------------- 1. 数据读取与清理 -----------------
def get_data_task2_gray(mat_path, trigs_list, elecs):
    """读取灰色水果4个Triggers的试次"""
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
                trial_data = trial_data[:, ch_indices, :] # (Rep, n_ch, Time)
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
    """读取真假颜色水果的试次"""
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

def clean_data(x):
    if x is None:
        return None
    return x[~np.isnan(x).any(axis=(1,2))]

# ----------------- 2. 解码核心逻辑 -----------------
def fit_eval_cross_pairs_t(t, r1, r2, g1, g2):
    """
    4个灰色水果配对条件解码记忆颜色在第 t 时间步
    正类: 红色记忆, 负类: 绿色记忆
    """
    pairs = [
        (r1, g1, r2, g2),
        (r1, g2, r2, g1),
        (r2, g1, r1, g2),
        (r2, g2, r1, g1)
    ]
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
    acc = np.mean(correct_all)
    return acc, correct_all

def fit_eval_logo_single_t(t, group_trials):
    """
    4折 Leave-One-Group-Out 跨物体真/假呈现分类解码在第 t 时间步
    正类: 呈现绿色(1), 负类: 呈现红色(0)
    """
    group_names = ['strawberry', 'watermelon', 'cabbage', 'kiwi']
    g_features = {}
    g_labels = {}
    
    for g in group_names:
        trigs = {
            'strawberry': strawberry_trigs,
            'watermelon': watermelon_trigs,
            'cabbage': cabbage_trigs,
            'kiwi': kiwi_trigs
        }[g]
        
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
    acc = np.mean(correct_all)
    return acc, correct_all

# ----------------- 3. 统计工具 -----------------
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
    print("="*90)
    print("Step 5: Memory Color Electrodes Clustering and SVM Decoding on ERP")
    print("="*90)
    
    # 1. 载入 ERP 记忆显著通道表并聚类
    sig_path = os.path.join(doc_dir, 'select_channel_memory_significance_erp.csv')
    if not os.path.exists(sig_path):
        print("[ERROR] ERP 显著通道表不存在，请确保 Step 2_1 已跑完！")
        return
        
    df_sig = pd.read_csv(sig_path)
    df_memory = df_sig[df_sig['Sig_Category'] != 'Non_Sig'].copy()
    
    if df_memory.empty:
        print("[WARNING] 无任何 memory_color 显著 ERP 电极！跳过")
        return
        
    print(f"Total memory-selective ERP electrodes for clustering: {len(df_memory)}")
    
    # K-Means 按 MNI_Y 轴坐标聚类为 2 类
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    df_memory['cluster'] = kmeans.fit_predict(df_memory[['MNI_Y']])
    
    # 根据 Y 均值大小，划分 posterior (更负) 与 anterior (更正)
    c0_mean_y = df_memory[df_memory['cluster'] == 0]['MNI_Y'].mean()
    c1_mean_y = df_memory[df_memory['cluster'] == 1]['MNI_Y'].mean()
    
    if c0_mean_y < c1_mean_y:
        post_label = 0
        ant_label = 1
    else:
        post_label = 1
        ant_label = 0
        
    df_memory['cluster_name'] = df_memory['cluster'].map({post_label: 'posterior', ant_label: 'anterior'})
    
    print("\n--- Electrode Clustering Results (KMeans 2 clusters) ---")
    for name in ['posterior', 'anterior']:
        sub_df = df_memory[df_memory['cluster_name'] == name]
        print(f"Cluster: {name:10s} | Y range: [{sub_df['MNI_Y'].min():.1f}, {sub_df['MNI_Y'].max():.1f}] mm | count: {len(sub_df)}")
        for subj in subjects:
            s_elecs = sub_df[sub_df['Subject'] == subj]['Electrode'].tolist()
            print(f"  - {subj}: {len(s_elecs)} electrodes -> {s_elecs}")
            
    # 2. 依次在后部和前部进行两种解码分析
    for cluster_name in ['posterior', 'anterior']:
        print(f"\n" + "#"*70)
        print(f"Running SVM decodings for Cluster: {cluster_name.upper()}...")
        print("#"*70)
        
        sub_df = df_memory[df_memory['cluster_name'] == cluster_name]
        cluster_elecs = {s: sub_df[sub_df['Subject'] == s]['Electrode'].tolist() for s in subjects}
        
        # 确定在此 cluster 内有有效电极的被试
        active_subjs = [s for s in subjects if len(cluster_elecs[s]) > 0]
        if len(active_subjs) == 0:
            print(f"  [WARNING] Cluster {cluster_name} 无任何被试有电极覆盖，跳过")
            continue
            
        print(f"  Active subjects: {active_subjs}")
        
        # ----------------- 2.1 Memory Color 解码 -----------------
        print(f"\n>>> 1) Memory Color Decoding (Red Memory vs Green Memory in Gray Fruits)...")
        correct_data_mem = {}
        acc_data_mem = {}
        time_ms = None
        
        for subj in active_subjs:
            mat_path = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
            elecs = cluster_elecs[subj]
            
            trigs_list = {'r1': r1_trigs, 'r2': r2_trigs, 'g1': g1_trigs, 'g2': g2_trigs}
            data_dict, t_arr = get_data_task2_gray(mat_path, trigs_list, elecs)
            
            if data_dict is None or any(v is None for v in data_dict.values()):
                print(f"    - [ERROR] {subj} 数据读取失败，跳过")
                continue
                
            if time_ms is None:
                time_ms = t_arr
                
            r1_arr, r2_arr, g1_arr, g2_arr = map(clean_data, [data_dict['r1'], data_dict['r2'], data_dict['g1'], data_dict['g2']])
            
            n_time = time_ms.shape[0]
            print(f"    Decoding {subj} (Pairs CV with {len(elecs)} elecs)...")
            
            results = Parallel(n_jobs=-1)(
                delayed(fit_eval_cross_pairs_t)(t, r1_arr, r2_arr, g1_arr, g2_arr)
                for t in range(n_time)
            )
            
            accs_t = np.array([r[0] for r in results])
            corrects_t = [r[1] for r in results]
            
            acc_data_mem[subj] = accs_t
            correct_data_mem[subj] = corrects_t
            
        # 拟合 GLMM (若被试数 >= 2)
        n_time = time_ms.shape[0]
        p_vals_mem = np.zeros(n_time)
        est_accs_mem = np.zeros(n_time)
        z_stats_mem = np.zeros(n_time)
        
        if len(active_subjs) >= 2:
            print(f"    Fitting GLMM across {n_time} timepoints...")
            for t in range(n_time):
                y_list = []
                subj_list = []
                for subj in active_subjs:
                    y_vector = correct_data_mem[subj][t]
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
                    
                    p_vals_mem[t] = p_val
                    est_accs_mem[t] = est_acc
                    z_stats_mem[t] = z_stat
                except Exception as e:
                    p_vals_mem[t] = 1.0
                    est_accs_mem[t] = df_t['Y'].mean()
                    z_stats_mem[t] = 0.0
        else:
            # 降级为单被试二项检验
            single_subj = active_subjs[0]
            print(f"    [DEGRADED] Single Subject Binomial Test for {single_subj}...")
            for t in range(n_time):
                y_vector = correct_data_mem[single_subj][t]
                k = np.sum(y_vector)
                n = len(y_vector)
                p_val = stats.binomtest(k, n, p=0.5, alternative='greater').pvalue
                p_vals_mem[t] = p_val
                est_accs_mem[t] = k / n if n > 0 else 0.5
                z_stats_mem[t] = 0.0
                
        sig_windows_mem = find_significant_windows(p_vals_mem, time_ms, p_thresh=0.05, min_duration=20)
        print(f"    Significant Windows (>20ms): {sig_windows_mem}")
        
        # 导出 Memory Color 数据
        group_mean_acc_mem = np.mean([acc_data_mem[s] for s in active_subjs], axis=0)
        export_mem = {
            'Time_ms': time_ms,
            'Group_Mean_Acc': group_mean_acc_mem,
            'GLMM_Est_Acc': est_accs_mem,
            'Stats_P': p_vals_mem,
            'Stats_Z': z_stats_mem
        }
        for s in subjects:
            export_mem[f'{s}_Acc'] = acc_data_mem[s] if s in active_subjs else np.nan
        df_export_mem = pd.DataFrame(export_mem)
        xlsx_mem = os.path.join(doc_dir, f'decoding_data_erp_cluster_{cluster_name}_memory_color.xlsx')
        csv_mem = os.path.join(doc_dir, f'decoding_data_erp_cluster_{cluster_name}_memory_color.csv')
        df_export_mem.to_excel(xlsx_mem, index=False)
        df_export_mem.to_csv(csv_mem, index=False)
        
        # ----------------- 2.2 True vs Fake Color 解码 -----------------
        print(f"\n>>> 2) True vs. Fake Color Decoding (Green Present vs Red Present, LOGO 4-fold)...")
        correct_data_tf = {}
        acc_data_tf = {}
        
        for subj in active_subjs:
            mat_path = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
            elecs = cluster_elecs[subj]
            
            all_t_trigs = strawberry_trigs + watermelon_trigs + cabbage_trigs + kiwi_trigs
            data_dict, _ = get_data_task2_real_fake(mat_path, all_t_trigs, elecs)
            
            if data_dict is None or any(v is None for v in data_dict.values()):
                print(f"    - [ERROR] {subj} 真假颜色数据读取失败，跳过")
                continue
                
            # 清理
            for k in data_dict.keys():
                data_dict[k] = clean_data(data_dict[k])
                
            group_trials = {
                'strawberry': {t: data_dict[t] for t in strawberry_trigs},
                'watermelon': {t: data_dict[t] for t in watermelon_trigs},
                'cabbage': {t: data_dict[t] for t in cabbage_trigs},
                'kiwi': {t: data_dict[t] for t in kiwi_trigs}
            }
            
            print(f"    Decoding {subj} (LOGO CV with {len(elecs)} elecs)...")
            results = Parallel(n_jobs=-1)(
                delayed(fit_eval_logo_single_t)(t, group_trials)
                for t in range(n_time)
            )
            
            accs_t = np.array([r[0] for r in results])
            corrects_t = [r[1] for r in results]
            
            acc_data_tf[subj] = accs_t
            correct_data_tf[subj] = corrects_t
            
        # 拟合 GLMM
        p_vals_tf = np.zeros(n_time)
        est_accs_tf = np.zeros(n_time)
        z_stats_tf = np.zeros(n_time)
        
        if len(active_subjs) >= 2:
            print(f"    Fitting GLMM across {n_time} timepoints...")
            for t in range(n_time):
                y_list = []
                subj_list = []
                for subj in active_subjs:
                    y_vector = correct_data_tf[subj][t]
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
                    
                    p_vals_tf[t] = p_val
                    est_accs_tf[t] = est_acc
                    z_stats_tf[t] = z_stat
                except Exception as e:
                    p_vals_tf[t] = 1.0
                    est_accs_tf[t] = df_t['Y'].mean()
                    z_stats_tf[t] = 0.0
        else:
            single_subj = active_subjs[0]
            print(f"    [DEGRADED] Single Subject Binomial Test for {single_subj}...")
            for t in range(n_time):
                y_vector = correct_data_tf[single_subj][t]
                k = np.sum(y_vector)
                n = len(y_vector)
                p_val = stats.binomtest(k, n, p=0.5, alternative='greater').pvalue
                p_vals_tf[t] = p_val
                est_accs_tf[t] = k / n if n > 0 else 0.5
                z_stats_tf[t] = 0.0
                
        sig_windows_tf = find_significant_windows(p_vals_tf, time_ms, p_thresh=0.05, min_duration=20)
        print(f"    Significant Windows (>20ms): {sig_windows_tf}")
        
        # 导出 True vs Fake 数据
        group_mean_acc_tf = np.mean([acc_data_tf[s] for s in active_subjs], axis=0)
        export_tf = {
            'Time_ms': time_ms,
            'Group_Mean_Acc': group_mean_acc_tf,
            'GLMM_Est_Acc': est_accs_tf,
            'Stats_P': p_vals_tf,
            'Stats_Z': z_stats_tf
        }
        for s in subjects:
            export_tf[f'{s}_Acc'] = acc_data_tf[s] if s in active_subjs else np.nan
        df_export_tf = pd.DataFrame(export_tf)
        xlsx_tf = os.path.join(doc_dir, f'decoding_data_erp_cluster_{cluster_name}_real_fake.xlsx')
        csv_tf = os.path.join(doc_dir, f'decoding_data_erp_cluster_{cluster_name}_real_fake.csv')
        df_export_tf.to_excel(xlsx_tf, index=False)
        df_export_tf.to_csv(csv_tf, index=False)
        
        # ----------------- 2.3 绘图 (Memory Color & True vs. Fake) -----------------
        # 绘图1: Memory Color
        fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
        t_idx = np.where((time_ms >= -200) & (time_ms <= 800))[0]
        time_plot = time_ms[t_idx]
        
        subj_colors = {'test001': '#ff7f0e', 'test002': '#2ca02c', 'test003': '#1f77b4', 'test005': '#9467bd', 'test006': '#8c564b'}
        for subj in active_subjs:
            ax.plot(time_plot, acc_data_mem[subj][t_idx], color=subj_colors[subj], lw=1.2, linestyle='--', alpha=0.5, label=f"Subj: {subj}")
        ax.plot(time_plot, group_mean_acc_mem[t_idx], color='#9b59b6', lw=3.2, label='Group Average')
        ax.axhline(0.5, color='#95a5a6', linestyle=':', lw=1.5, label='Chance (50%)')
        ax.axvline(0, color='#7f8c8d', linestyle='-', lw=1.2)
        
        has_shaded = False
        y_line = 0.73
        sig_label = 'GLMM Significant (p<0.05, >20ms)' if len(active_subjs) >= 2 else 'Binomial Significant (p<0.05, >20ms)'
        for start, end in sig_windows_mem:
            if end < -200 or start > 800:
                continue
            s_plot = max(start, -200)
            e_plot = min(end, 800)
            ax.axvspan(s_plot, e_plot, color='#d62728', alpha=0.1, zorder=1)
            lbl = sig_label if not has_shaded else ""
            ax.plot([s_plot, e_plot], [y_line, y_line], color='#d62728', lw=4.0, label=lbl, zorder=4)
            has_shaded = True
            
        ax.set_title(f"ERP Memory Color Decoding: Cluster [{cluster_name.upper()}]\n(Active Subjects N = {len(active_subjs)})", fontsize=12.5, fontweight='bold', pad=10)
        ax.set_xlabel("Time relative to stimulus onset (ms)", fontsize=10.5)
        ax.set_ylabel("Decoding Accuracy", fontsize=10.5)
        ax.set_xlim([-200, 800])
        ax.set_ylim([0.35, 0.76])
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_facecolor('#fafafa')
        ax.legend(loc='lower left', framealpha=0.9, fontsize=9.0)
        plt.tight_layout()
        out_fig_mem = os.path.join(out_fig_dir, f"erp_cluster_{cluster_name}_memory_color_decoding.png")
        plt.savefig(out_fig_mem, dpi=300)
        plt.close()
        print(f"    [SAVED FIGURE] Memory Color Decoding: {out_fig_mem}")
        
        # 绘图2: True vs. Fake Color
        fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
        for subj in active_subjs:
            ax.plot(time_plot, acc_data_tf[subj][t_idx], color=subj_colors[subj], lw=1.2, linestyle='--', alpha=0.5, label=f"Subj: {subj}")
        ax.plot(time_plot, group_mean_acc_tf[t_idx], color='#34495e', lw=3.2, label='Group Average')
        ax.axhline(0.5, color='#95a5a6', linestyle=':', lw=1.5, label='Chance (50%)')
        ax.axvline(0, color='#7f8c8d', linestyle='-', lw=1.2)
        
        has_shaded = False
        for start, end in sig_windows_tf:
            if end < -200 or start > 800:
                continue
            s_plot = max(start, -200)
            e_plot = min(end, 800)
            ax.axvspan(s_plot, e_plot, color='#d62728', alpha=0.1, zorder=1)
            lbl = sig_label if not has_shaded else ""
            ax.plot([s_plot, e_plot], [y_line, y_line], color='#d62728', lw=4.0, label=lbl, zorder=4)
            has_shaded = True
            
        ax.set_title(f"ERP True vs. Fake Fruit Color Decoding: Cluster [{cluster_name.upper()}]\n(Active Subjects N = {len(active_subjs)})", fontsize=12.5, fontweight='bold', pad=10)
        ax.set_xlabel("Time relative to stimulus onset (ms)", fontsize=10.5)
        ax.set_ylabel("Decoding Accuracy", fontsize=10.5)
        ax.set_xlim([-200, 800])
        ax.set_ylim([0.35, 0.76])
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_facecolor('#fafafa')
        ax.legend(loc='lower left', framealpha=0.9, fontsize=9.0)
        plt.tight_layout()
        out_fig_tf = os.path.join(out_fig_dir, f"erp_cluster_{cluster_name}_real_fake_decoding.png")
        plt.savefig(out_fig_tf, dpi=300)
        plt.close()
        print(f"    [SAVED FIGURE] True vs. Fake Decoding: {out_fig_tf}")
        
    print("\n" + "="*90)
    print("Step 5 successfully completed!")
    print("="*90)

if __name__ == '__main__':
    main()
