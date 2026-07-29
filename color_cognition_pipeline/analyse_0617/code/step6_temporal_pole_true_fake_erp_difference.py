import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
from pymatreader import read_mat
import matplotlib.pyplot as plt
import os
import warnings

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
out_fig_dir = os.path.join(result_dir, 'select_channel', 'decoding', 'real_fake', 'single_electrode')
os.makedirs(out_fig_dir, exist_ok=True)

subjects = ['test001', 'test002', 'test003']

# 电极范围定义
temporal_pole_elecs = {
    'test001': ['E6', 'E7'],
    'test002': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8'],
    'test003': [],
    'test005': ['A1', 'A2', 'A3'],
    'test006': []
}

# 真假颜色对应的 Triggers (Task 2)
# 真颜色: 草莓真红(121), 西瓜真红(131), 卷心菜真绿(101), 猕猴桃真绿(111)
true_trigs = ['Trigger-In:121', 'Trigger-In:131', 'Trigger-In:101', 'Trigger-In:111']
# 假颜色: 草莓假绿(122), 西瓜假绿(132), 卷心菜假红(102), 猕猴桃假红(112)
fake_trigs = ['Trigger-In:122', 'Trigger-In:132', 'Trigger-In:102', 'Trigger-In:112']

# ----------------- 数据提取函数 -----------------
def get_data_for_elec(mat_path, trigs_to_extract, elec_name):
    if not os.path.exists(mat_path):
        return None, None
    try:
        mat = read_mat(mat_path)
        epoch = mat['epoch']
        ch_names = list(epoch['ch']['labels'])
        time_ms = epoch['time_ms'] if 'time_ms' in epoch else np.linspace(-500, 998, epoch['data'].shape[-1])
        all_trigs = list(epoch['trigger'])
        
        if elec_name not in ch_names:
            return None, None
            
        ch_idx = ch_names.index(elec_name)
        idx_list = [all_trigs.index(t) for t in trigs_to_extract if t in all_trigs]
        if not idx_list:
            return None, None
            
        data_list = []
        for idx in idx_list:
            trial_data = epoch['data'][idx, :, ch_idx, :] # (Rep, Time)
            baseline_mask = time_ms < 0
            baseline_indices = np.where(baseline_mask)[0]
            if len(baseline_indices) > 0:
                mean_bl = np.mean(trial_data[:, baseline_indices], axis=1, keepdims=True)
                trial_data = trial_data - mean_bl
            data_list.append(trial_data)
            
        merged_data = np.concatenate(data_list, axis=0)
        merged_data = merged_data[~np.isnan(merged_data).any(axis=1)]
        return merged_data, time_ms
    except Exception as e:
        print(f"  [ERROR] Loading {elec_name} data failed: {e}")
        return None, None

# ----------------- 连续显著段查找 -----------------
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

# ----------------- 主程序 -----------------
def main():
    print("="*90)
    print("Step 6: Temporal Pole Single-Electrode True vs. Fake Color ERP Analysis")
    print("="*90)
    
    results = []
    
    for subj in subjects:
        elecs = temporal_pole_elecs[subj]
        if not elecs:
            print(f"Subject {subj} has no temporal_pole electrodes.")
            continue
            
        mat_path = os.path.join(feature_dir, subj, 'task2_ERP_epoched.mat')
        print(f"\nProcessing Subject: {subj} (electrocs: {elecs})")
        
        for elec in elecs:
            print(f"  Analysing electrode {elec}...")
            
            # 提取数据
            true_data, time_ms = get_data_for_elec(mat_path, true_trigs, elec)
            fake_data, _ = get_data_for_elec(mat_path, fake_trigs, elec)
            
            if true_data is None or fake_data is None:
                print(f"    [WARNING] Missing data for {subj}-{elec}. Skip.")
                continue
                
            n_time = time_ms.shape[0]
            
            # 1. 逐时间点 Wilcoxon 检验
            p_vals_t = np.zeros(n_time)
            z_stats_t = np.zeros(n_time)
            for t in range(n_time):
                stat, p_val = stats.ranksums(true_data[:, t], fake_data[:, t])
                p_vals_t[t] = p_val
                z_stats_t[t] = stat
                
            # 查找连续 50ms 以上的显著段
            sig_windows = find_cont_sig_windows(p_vals_t, time_ms, p_thresh=0.05, min_duration=50)
            has_cont_sig = len(sig_windows) > 0
            
            # 2. 200-500ms 均值 Wilcoxon 检验
            t_idx_200_500 = np.where((time_ms >= 200) & (time_ms <= 500))[0]
            mean_true_200_500 = np.mean(true_data[:, t_idx_200_500], axis=1)
            mean_fake_200_500 = np.mean(fake_data[:, t_idx_200_500], axis=1)
            
            mean_stat, mean_p = stats.ranksums(mean_true_200_500, mean_fake_200_500)
            is_mean_sig = mean_p < 0.05
            
            m_true = np.mean(mean_true_200_500)
            m_fake = np.mean(mean_fake_200_500)
            diff_val = m_true - m_fake
            
            results.append({
                'Subject': subj,
                'Electrode': elec,
                'Mean_True_200_500': m_true,
                'Mean_Fake_200_500': m_fake,
                'Mean_Diff_200_500': diff_val,
                'Wilcoxon_Z': mean_stat,
                'Wilcoxon_P': mean_p,
                'Is_Mean_Sig_200_500': is_mean_sig,
                'Has_Cont_50ms_Sig': has_cont_sig,
                'Cont_Sig_Windows': str(sig_windows)
            })
            
            # 3. 绘图 (1行2列)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [2.5, 1]}, dpi=300)
            
            # A. 左图: 时程曲线
            t_idx_plot = np.where((time_ms >= -200) & (time_ms <= 800))[0]
            t_plot = time_ms[t_idx_plot]
            
            # 平均值及标准误
            m_true_plot = np.mean(true_data[:, t_idx_plot], axis=0)
            m_fake_plot = np.mean(fake_data[:, t_idx_plot], axis=0)
            sem_true = stats.sem(true_data[:, t_idx_plot], axis=0)
            sem_fake = stats.sem(fake_data[:, t_idx_plot], axis=0)
            
            ax1.plot(t_plot, m_true_plot, color='#2980b9', lw=2.5, label=f'True Color (N={len(true_data)})')
            ax1.fill_between(t_plot, m_true_plot - sem_true, m_true_plot + sem_true, color='#2980b9', alpha=0.18)
            
            ax1.plot(t_plot, m_fake_plot, color='#e74c3c', lw=2.5, label=f'Fake Color (N={len(fake_data)})')
            ax1.fill_between(t_plot, m_fake_plot - sem_fake, m_fake_plot + sem_fake, color='#e74c3c', alpha=0.18)
            
            ax1.axhline(0, color='#7f8c8d', linestyle=':', lw=1.2)
            ax1.axvline(0, color='#7f8c8d', linestyle='-', lw=1.2)
            
            # 显著性阴影
            has_shading = False
            for start, end in sig_windows:
                if end < -200 or start > 800:
                    continue
                s_plot = max(start, -200)
                e_plot = min(end, 800)
                ax1.axvspan(s_plot, e_plot, color='#d35400', alpha=0.12, zorder=1)
                lbl = 'Cont. Sig. (p<0.05, >50ms)' if not has_shading else ''
                # 在底部画一条加粗的条形
                ax1.plot([s_plot, e_plot], [ax1.get_ylim()[0] + 0.05 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])] * 2, 
                         color='#d35400', lw=4.0, label=lbl, zorder=4)
                has_shading = True
                
            ax1.set_title(f"ERP Evoked Signal: {subj} - {elec}", fontsize=11.5, fontweight='bold')
            ax1.set_xlabel("Time relative to stimulus onset (ms)", fontsize=10)
            ax1.set_ylabel("Amplitude (μV)", fontsize=10)
            ax1.set_xlim([-200, 800])
            ax1.grid(True, linestyle=':', alpha=0.45)
            ax1.set_facecolor('#fafafa')
            ax1.legend(loc='upper right', framealpha=0.9, fontsize=9.0)
            
            # B. 右图: 200-500ms 平均值箱线图
            box_data = [mean_true_200_500, mean_fake_200_500]
            bp = ax2.boxplot(box_data, widths=0.45, patch_artist=True,
                             boxprops=dict(facecolor='#fafafa', color='#2c3e50', linewidth=1.2),
                             medianprops=dict(color='#e74c3c', linewidth=1.8),
                             whiskerprops=dict(color='#2c3e50', linewidth=1.2),
                             capprops=dict(color='#2c3e50', linewidth=1.2),
                             flierprops=dict(marker='o', markerfacecolor='#95a5a6', alpha=0.6, markersize=5))
            
            colors = ['#2980b9', '#e74c3c']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
                
            ax2.set_xticklabels(['True', 'Fake'], fontsize=9.5)
            ax2.set_ylabel("Mean Amplitude (200-500ms, μV)", fontsize=10)
            ax2.grid(True, linestyle=':', alpha=0.35)
            ax2.set_facecolor('#fafafa')
            
            # 标注显著性
            y_max = max(np.max(mean_true_200_500), np.max(mean_fake_200_500))
            y_min = min(np.min(mean_true_200_500), np.min(mean_fake_200_500))
            h = (y_max - y_min) * 0.05
            
            if mean_p < 0.001:
                sig_str = '***'
            elif mean_p < 0.01:
                sig_str = '**'
            elif mean_p < 0.05:
                sig_str = '*'
            else:
                sig_str = 'n.s.'
                
            ax2.plot([1, 1, 2, 2], [y_max + h, y_max + 2*h, y_max + 2*h, y_max + h], color='#2c3e50', lw=1.2)
            ax2.text(1.5, y_max + 2.2*h, sig_str, ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50')
            ax2.set_title("200-500ms Average", fontsize=10.5, fontweight='bold')
            
            plt.suptitle(f"True vs. Fake Fruit Color Comparison | {subj} - {elec} (Temporal Pole)", fontsize=13, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            out_fig = os.path.join(out_fig_dir, f"{subj}_{elec}_true_fake_difference.png")
            plt.savefig(out_fig, dpi=300)
            plt.close()
            print(f"    [SAVED PLOT] -> {out_fig}")
            
    # 4. 保存统计结果为 Excel/CSV
    df_res = pd.DataFrame(results)
    xlsx_path = os.path.join(doc_dir, 'temporal_pole_true_fake_erp_stats.xlsx')
    csv_path = os.path.join(doc_dir, 'temporal_pole_true_fake_erp_stats.csv')
    df_res.to_excel(xlsx_path, index=False)
    df_res.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Saved Temporal Pole ERP Stats to:\n  - {xlsx_path}\n  - {csv_path}")
    print("="*90)

if __name__ == '__main__':
    main()
