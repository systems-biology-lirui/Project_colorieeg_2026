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
out_fig_dir = os.path.join(result_dir, 'select_channel', 'decoding', 'real_fake')
os.makedirs(out_fig_dir, exist_ok=True)

subjects = ['test001', 'test002', 'test003']

# 8 个真假水果 Trigger 定义
strawberry_trigs = ['Trigger-In:121', 'Trigger-In:122'] # 真红, 假绿
watermelon_trigs = ['Trigger-In:131', 'Trigger-In:132'] # 真红, 假绿
cabbage_trigs = ['Trigger-In:102', 'Trigger-In:101']    # 假红, 真绿
kiwi_trigs = ['Trigger-In:112', 'Trigger-In:111']       # 假红, 真绿

# 区分红绿标签 (红=0, 绿=1)
red_labels = ['Trigger-In:121', 'Trigger-In:131', 'Trigger-In:102', 'Trigger-In:112']
green_labels = ['Trigger-In:122', 'Trigger-In:132', 'Trigger-In:101', 'Trigger-In:111']

# ROI 脑区匹配规则
roi_definitions = {
    'temporal_pole': ['temporal_pole', 'pole_temp'],
    'temporal_mid': ['temporal_mid', 'temp_mid'],
    'temporal_inf': ['temporal_inf', 'temp_inf'],
    'amygdala': ['amygdala']
}

# ----------------- 1. 兼容性二项检验 -----------------
def binomial_test_p(k, n, p_chance=0.5):
    try:
        from scipy.stats import binomtest
        res = binomtest(k, n, p=p_chance, alternative='greater')
        return res.pvalue
    except ImportError:
        from scipy.stats import binom_test
        return binom_test(k, n, p=p_chance, alternative='greater')

# ----------------- 2. 数据读取与基线减法 -----------------
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
            
        data_dict = {}
        for t in trigs_to_extract:
            if t in all_trigs:
                idx = all_trigs.index(t)
                trial_data = epoch['data'][idx, :, :, :]
                trial_data = trial_data[:, ch_indices, :]
                
                # Trial-wise 基线减法
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
        print(f"  [ERROR] Loading {mat_path} failed: {e}")
        return None, None

def clean_data(x):
    if x is None:
        return None
    return x[~np.isnan(x).any(axis=(1,2))]

# ----------------- 3. 单时间点 Leave-One-Group-Out SVM -----------------
def fit_eval_logo_single_t(t, group_trials):
    """
    进行 Leave-One-Group-Out 4 折交叉验证解码
    group_trials: {group_name: {trigger: cleaned_trial_data}}
    """
    group_names = ['strawberry', 'watermelon', 'cabbage', 'kiwi']
    # 提取第 t 时间步上各组的数据并分配标签 (红=0, 绿=1)
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
    
    # 4 折 Leave-One-Group-Out
    for fold_idx, test_g in enumerate(group_names):
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

# ----------------- 4. 连续显著时间窗定位 -----------------
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
    print("Step 4: Real vs. Fake Fruit Color Multi-ROI SVM Decoding Analysis on ERP")
    print("="*85)
    
    # 1. 整理被试在各个 ROI 内的电极映射关系
    # A. 提取记忆显著 ERP 通道
    sig_path = os.path.join(doc_dir, 'select_channel_memory_significance_erp.csv')
    df_sig = pd.read_csv(sig_path) if os.path.exists(sig_path) else None
    
    subj_rois = {s: {} for s in subjects}
    
    for subj in subjects:
        loc_path = os.path.join(base_dir, 'processed_data', subj, f'{subj}_ieegloc.xlsx')
        if not os.path.exists(loc_path):
            continue
            
        df_loc = pd.read_excel(loc_path)
        aal_col = 'AAL3 (MNI-linear)' if 'AAL3 (MNI-linear)' in df_loc.columns else 'AAL3 (MNI-segment)'
        ch_col = 'Channel' if 'Channel' in df_loc.columns else df_loc.columns[0]
        
        # 匹配 4 个解剖 ROI
        for r_name, keywords in roi_definitions.items():
            matched = []
            for _, row in df_loc.iterrows():
                ch = str(row[ch_col]).strip()
                aal = str(row[aal_col]).lower() if pd.notna(row[aal_col]) else ''
                if any(kw in aal for kw in keywords):
                    matched.append(ch)
            subj_rois[subj][r_name] = matched
            
        # 加上 memory_color ROI
        if df_sig is not None:
            df_subj = df_sig[(df_sig['Subject'] == subj) & (df_sig['Sig_Category'] != 'Non_Sig')]
            subj_rois[subj]['memory_color'] = df_subj['Electrode'].astype(str).tolist()
        else:
            subj_rois[subj]['memory_color'] = []
            
    # 打印 ROI 包含的电极分布
    print("ROI Electrode Coverage Details:")
    all_rois = ['temporal_pole', 'temporal_mid', 'temporal_inf', 'memory_color', 'amygdala']
    for r_name in all_rois:
        print(f"  ROI: {r_name}")
        for subj in subjects:
            elec_list = subj_rois[subj].get(r_name, [])
            print(f"    - {subj}: count = {len(elec_list)}, elecs = {elec_list}")
            
    # 2. 针对 5 个 ROI 逐个做解码分析
    for r_name in all_rois:
        print(f"\n>>> Running Decoding on ROI: {r_name}...")
        
        # 识别在此 ROI 内有有效电极的被试
        active_subjs = [s for s in subjects if s in subj_rois and len(subj_rois[s].get(r_name, [])) > 0]
        if not active_subjs:
            print(f"  [WARNING] ROI {r_name} 无任何被试包含电极！直接跳过该脑区")
            continue
            
        print(f"  Active subjects: {active_subjs}")
        
        correct_data = {}
        subj_accs = {}
        time_ms = None
        
        # 读取数据并解码
        for subj in active_subjs:
            mat_path = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
            elecs = subj_rois[subj][r_name]
            
            all_t_trigs = strawberry_trigs + watermelon_trigs + cabbage_trigs + kiwi_trigs
            data_dict, t_arr = get_data(mat_path, all_t_trigs, elecs)
            
            if data_dict is None or any(v is None for v in data_dict.values()):
                print(f"  [ERROR] {subj} 缺少 Task 2 红/绿刺激数据，跳过")
                continue
                
            if time_ms is None:
                time_ms = t_arr
                
            # 清理
            for k in data_dict.keys():
                data_dict[k] = clean_data(data_dict[k])
                
            # 整理为按组划分的数据结构
            group_trials = {
                'strawberry': {t: data_dict[t] for t in strawberry_trigs},
                'watermelon': {t: data_dict[t] for t in watermelon_trigs},
                'cabbage': {t: data_dict[t] for t in cabbage_trigs},
                'kiwi': {t: data_dict[t] for t in kiwi_trigs}
            }
            
            n_time = time_ms.shape[0]
            print(f"  Decoding {subj} (Leave-One-Group-Out SVM with {len(elecs)} elecs)...")
            
            results = Parallel(n_jobs=-1)(
                delayed(fit_eval_logo_single_t)(t, group_trials)
                for t in range(n_time)
            )
            
            accs_t = np.array([r[0] for r in results])
            corrects_t = [r[1] for r in results]
            
            subj_accs[subj] = accs_t
            correct_data[subj] = corrects_t
            
        if not subj_accs:
            continue
            
        n_time = time_ms.shape[0]
        p_vals = np.zeros(n_time)
        est_accs = np.zeros(n_time)
        z_stats = np.zeros(n_time)
        
        # 3. 统计检验 (GLMM 或降级二项检验)
        if len(active_subjs) >= 2:
            # 拟合多被试 GLMM
            print(f"  Fitting GLMM (Binomial, random intercept by Subject) across {n_time} timepoints...")
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
                    
                    p_vals[t] = p_val
                    est_accs[t] = est_acc
                    z_stats[t] = z_stat
                except Exception as e:
                    p_vals[t] = 1.0
                    est_accs[t] = df_t['Y'].mean()
                    z_stats[t] = 0.0
        else:
            # 降级为单被试二项检验
            single_subj = active_subjs[0]
            print(f"  [DEGRADED] Fitting Binomial Test for single subject {single_subj} across {n_time} timepoints...")
            for t in range(n_time):
                y_vector = correct_data[single_subj][t]
                k = np.sum(y_vector)
                n = len(y_vector)
                
                p_val = binomial_test_p(k, n, p_chance=0.5)
                p_vals[t] = p_val
                est_accs[t] = k / n if n > 0 else 0.5
                z_stats[t] = 0.0 # 单被试没有 z_stat
                
        sig_windows = find_significant_windows(p_vals, time_ms, p_thresh=0.05, min_duration=20)
        print(f"  Significant Windows (>20ms) for ROI {r_name}: {sig_windows}")
        
        # 4. 保存明细数据
        group_acc_list = [subj_accs[s] for s in active_subjs]
        group_mean_acc = np.mean(group_acc_list, axis=0)
        
        export_dict = {
            'Time_ms': time_ms,
            'Group_Mean_Acc': group_mean_acc,
            'GLMM_Est_Acc': est_accs,
            'Stats_P': p_vals,
            'Stats_Z': z_stats
        }
        for s in subjects:
            if s in active_subjs:
                export_dict[f'{s}_Acc'] = subj_accs[s]
            else:
                export_dict[f'{s}_Acc'] = np.nan
                
        df_export = pd.DataFrame(export_dict)
        xlsx_path = os.path.join(doc_dir, f'real_fake_decoding_results_{r_name}.xlsx')
        csv_path = os.path.join(doc_dir, f'real_fake_decoding_results_{r_name}.csv')
        df_export.to_excel(xlsx_path, index=False)
        df_export.to_csv(csv_path, index=False)
        print(f"  [DATA SAVED] Saved ROI decoding statistics to:\n    - {xlsx_path}\n    - {csv_path}")
        
        # 5. 绘图
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
        ax.plot(time_plot, mean_plot, color='#2c3e50', lw=3.5, label='Group Average')
        
        ax.axhline(0.5, color='#9e9e9e', linestyle=':', lw=1.5, label='Chance Level (50%)')
        ax.axvline(0, color='#757575', linestyle='-', lw=1.2)
        
        # 绘制显著区间
        y_line_val = 0.73
        has_shaded = False
        sig_label = 'GLMM Significant (p < 0.05, >20ms)' if len(active_subjs) >= 2 else 'Binomial Significant (p < 0.05, >20ms)'
        
        for start, end in sig_windows:
            if end < -200 or start > 800:
                continue
            s_plot = max(start, -200)
            e_plot = min(end, 800)
            
            ax.axvspan(s_plot, e_plot, color='#d62728', alpha=0.12, zorder=1)
            
            label_line = sig_label if not has_shaded else ""
            ax.plot([s_plot, e_plot], [y_line_val, y_line_val], color='#d62728', lw=4.5, solid_capstyle='butt', label=label_line, zorder=4)
            has_shaded = True
            
        title_str = "ERP Real vs. Fake Fruit Color Decoding: ROI [{}]\n(Active Subjects N = {})".format(r_name.upper(), len(active_subjs))
        if len(active_subjs) == 1:
            title_str += " [Degraded to Binomial Test]"
            
        ax.set_title(title_str, fontsize=13.5, fontweight='bold', pad=12)
        ax.set_xlabel("Time relative to stimulus onset (ms)", fontsize=11)
        ax.set_ylabel("Decoding Accuracy", fontsize=11)
        ax.set_xlim([-200, 800])
        ax.set_ylim([0.35, 0.76])
        ax.grid(True, linestyle=':', alpha=0.45)
        ax.set_facecolor('#fafafa')
        ax.legend(loc='lower left', framealpha=0.9, fontsize=9.5)
        
        plt.tight_layout()
        out_fig = os.path.join(out_fig_dir, f"real_fake_decoding_{r_name}.png")
        plt.savefig(out_fig, dpi=300)
        plt.close()
        print(f"  [FIGURE SAVED] Saved ROI decoding plot to: {out_fig}")
        
    print("\n" + "="*85)
    print("Step 4 Real vs. Fake Fruit Color Decoding successfully completed!")
    print("="*85)

if __name__ == '__main__':
    main()
