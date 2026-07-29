"""
Step 1_2 (论文公式修正版):
  1) 使用 (R_C - R_A) / (R_C + R_A) 公式计算各电极的颜色选择性指数 (CSI)。
     - 无色（灰色）响应 R_A: 使用黑（55）和白（56）刺激的平均响应值。
     - 每种颜色 (红、绿、黄、蓝) 分别与无色基线做单独的对比计算。
     - 同时保留彩色整体 (Overall) 与无色基线的对比。
  2) 分析包含:
     - 总体: (R_Color_mean - R_BW_mean) / (R_Color_mean + R_BW_mean)
     - 红色: (R_Red - R_BW_mean) / (R_Red + R_BW_mean)
     - 绿色: (R_Green - R_BW_mean) / (R_Green + R_BW_mean)
     - 黄色: (R_Yellow - R_BW_mean) / (R_Yellow + R_BW_mean)
     - 蓝色: (R_Blue - R_BW_mean) / (R_Blue + R_BW_mean)
  3) 绘制 1行5列 的 CSI 渐变 rank 排序图 (不覆盖原图):
     - `color_selectivity_index_paper_erp.png`
     - `color_selectivity_index_paper_hg.png`
  4) 绘制 1行5列 的与 MNI_Y 相关性回归图 (不覆盖原图):
     - `color_selectivity_mni_y_correlation_paper.png`
  5) 输出对应的表格文件到 doc 目录下。
"""
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import ranksums, spearmanr, pearsonr, linregress
from pymatreader import read_mat
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colorbar import ColorbarBase
import os, warnings

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
analyse_dir = os.path.join(pipeline_dir, 'analyse_0617')
feature_dir = os.path.join(analyse_dir, 'feature')
doc_dir = os.path.join(analyse_dir, 'doc')
result_dir = os.path.join(analyse_dir, 'result')
out_dir = os.path.join(result_dir, 'select_channel', 'color_selectivity')
os.makedirs(out_dir, exist_ok=True)

subjects = ['test001', 'test002', 'test003']

# Task3 条件触发器代码
cond_trigs = {
    'Red': 'Trigger-In:51',
    'Yellow': 'Trigger-In:52',
    'Blue': 'Trigger-In:53',
    'Green': 'Trigger-In:54',
    'Black': 'Trigger-In:55',
    'White': 'Trigger-In:56'
}

def analyze_color_selectivity_paper(is_hg=False):
    summary_path = os.path.join(doc_dir, 'select_channel_summary.xlsx')
    if not os.path.exists(summary_path):
        print("[ERROR] select_channel_summary.xlsx 不存在，请先运行 step1_1！")
        return []
    df_summary = pd.read_excel(summary_path)
    
    results = []
    
    for subj in subjects:
        subj_df = df_summary[df_summary['Subject'] == subj]
        if subj_df.empty:
            continue
            
        mat_path = os.path.join(feature_dir, subj, 'task3_hg_subband.mat' if is_hg else 'task3_ERP_epoched.mat')
        if not os.path.exists(mat_path):
            print(f"  [WARNING] {subj} Task 3 数据不存在: {mat_path}")
            continue
            
        mat = read_mat(mat_path)
        epoch = mat['epoch']
        trigs = list(epoch['trigger'])
        ch_labels = [str(x).strip() for x in epoch['ch']['labels']]
        time_ms = epoch['time_ms'] if 'time_ms' in epoch else np.linspace(-500, 998, epoch['data'].shape[-1])
        t_idx_stat = np.where((time_ms >= 50) & (time_ms <= 400))[0]
        
        # 建立映射
        trig_indices = {}
        for name, code in cond_trigs.items():
            if code in trigs:
                trig_indices[name] = trigs.index(code)
                
        if len(trig_indices) < 6:
            print(f"  [WARNING] {subj} Task 3 中缺少部分颜色/黑白触发器，跳过该被试！")
            continue
            
        for _, row in subj_df.iterrows():
            elec = str(row['Electrode']).strip()
            if elec not in ch_labels:
                continue
            ch_idx = ch_labels.index(elec)
            mni_y = float(row['MNI_Y']) if not pd.isna(row['MNI_Y']) else 0.0
            
            # 策略4判断
            if is_hg:
                is_stra4 = '4' in str(row['HG_Strategies_Matched'])
            else:
                is_stra4 = '4' in str(row['ERP_Strategies_Matched'])
                
            # 计算各条件响应值
            cond_resps = {}
            for name, idx in trig_indices.items():
                if is_hg:
                    cond_data = epoch['data_cell'][idx][:, ch_idx, t_idx_stat]
                else:
                    cond_data = epoch['data'][idx, :, ch_idx, t_idx_stat]
                
                cond_data = cond_data[~np.isnan(cond_data).any(axis=1)]
                if cond_data.shape[0] == 0:
                    cond_resps[name] = {'abs': 0.0, 'raw': 0.0, 'abs_trials': np.array([]), 'raw_trials': np.array([])}
                    continue
                
                if not is_hg:
                    abs_means = np.mean(np.abs(cond_data), axis=1)
                    raw_means = np.mean(cond_data, axis=1)
                    cond_resps[name] = {
                        'abs': np.mean(abs_means),
                        'raw': np.mean(raw_means),
                        'abs_trials': abs_means,
                        'raw_trials': raw_means
                    }
                else:
                    means = np.mean(cond_data, axis=1)
                    cond_resps[name] = {
                        'abs': np.mean(means),
                        'raw': np.mean(means),
                        'abs_trials': means,
                        'raw_trials': means
                    }
            
            # 提取无色（黑白）合并样本和均值
            achromatic_trials_abs = np.concatenate([cond_resps['Black']['abs_trials'], cond_resps['White']['abs_trials']], axis=0)
            achromatic_trials_raw = np.concatenate([cond_resps['Black']['raw_trials'], cond_resps['White']['raw_trials']], axis=0)
            R_A_abs = np.mean(achromatic_trials_abs) if len(achromatic_trials_abs) > 0 else 0.0
            R_A_raw = np.mean(achromatic_trials_raw) if len(achromatic_trials_raw) > 0 else 0.0
            
            # 彩色各条件均值
            R_Red_abs = cond_resps['Red']['abs']
            R_Green_abs = cond_resps['Green']['abs']
            R_Yellow_abs = cond_resps['Yellow']['abs']
            R_Blue_abs = cond_resps['Blue']['abs']
            R_C_abs = (R_Red_abs + R_Green_abs + R_Yellow_abs + R_Blue_abs) / 4.0
            
            R_Red_raw = cond_resps['Red']['raw']
            R_Green_raw = cond_resps['Green']['raw']
            R_Yellow_raw = cond_resps['Yellow']['raw']
            R_Blue_raw = cond_resps['Blue']['raw']
            R_C_raw = (R_Red_raw + R_Green_raw + R_Yellow_raw + R_Blue_raw) / 4.0
            
            # 1. 绝对强度指标计算 (用于图表绘制)
            overall_csi_abs = (R_C_abs - R_A_abs) / (R_C_abs + R_A_abs) if (R_C_abs + R_A_abs) != 0 else 0.0
            red_csi_abs     = (R_Red_abs - R_A_abs) / (R_Red_abs + R_A_abs) if (R_Red_abs + R_A_abs) != 0 else 0.0
            green_csi_abs   = (R_Green_abs - R_A_abs) / (R_Green_abs + R_A_abs) if (R_Green_abs + R_A_abs) != 0 else 0.0
            yellow_csi_abs  = (R_Yellow_abs - R_A_abs) / (R_Yellow_abs + R_A_abs) if (R_Yellow_abs + R_A_abs) != 0 else 0.0
            blue_csi_abs    = (R_Blue_abs - R_A_abs) / (R_Blue_abs + R_A_abs) if (R_Blue_abs + R_A_abs) != 0 else 0.0
            
            # 2. 原始代数指标计算 (Raw)
            overall_csi_raw = (R_C_raw - R_A_raw) / (R_C_raw + R_A_raw) if (R_C_raw + R_A_raw) != 0 else 0.0
            red_csi_raw     = (R_Red_raw - R_A_raw) / (R_Red_raw + R_A_raw) if (R_Red_raw + R_A_raw) != 0 else 0.0
            green_csi_raw   = (R_Green_raw - R_A_raw) / (R_Green_raw + R_A_raw) if (R_Green_raw + R_A_raw) != 0 else 0.0
            yellow_csi_raw  = (R_Yellow_raw - R_A_raw) / (R_Yellow_raw + R_A_raw) if (R_Yellow_raw + R_A_raw) != 0 else 0.0
            blue_csi_raw    = (R_Blue_raw - R_A_raw) / (R_Blue_raw + R_A_raw) if (R_Blue_raw + R_A_raw) != 0 else 0.0
            
            # 3. 统计显著性检验 (各单色 trials 与黑白合并 trials 之间的 Wilcoxon 检验)
            chromatic_trials_abs = np.concatenate([cond_resps[c]['abs_trials'] for c in ['Red', 'Yellow', 'Blue', 'Green']], axis=0)
            try:
                p_overall = ranksums(chromatic_trials_abs, achromatic_trials_abs)[1]
                p_red     = ranksums(cond_resps['Red']['abs_trials'], achromatic_trials_abs)[1]
                p_green   = ranksums(cond_resps['Green']['abs_trials'], achromatic_trials_abs)[1]
                p_yellow  = ranksums(cond_resps['Yellow']['abs_trials'], achromatic_trials_abs)[1]
                p_blue    = ranksums(cond_resps['Blue']['abs_trials'], achromatic_trials_abs)[1]
            except:
                p_overall, p_red, p_green, p_yellow, p_blue = 1.0, 1.0, 1.0, 1.0, 1.0
                
            results.append({
                'Subject': subj,
                'Electrode': elec,
                'MNI_Y': mni_y,
                'Is_Strategy_4': is_stra4,
                # 绝对指标
                'Overall_CSI': overall_csi_abs, 'Overall_P': p_overall,
                'Red_CSI': red_csi_abs, 'Red_P': p_red,
                'Green_CSI': green_csi_abs, 'Green_P': p_green,
                'Yellow_CSI': yellow_csi_abs, 'Yellow_P': p_yellow,
                'Blue_CSI': blue_csi_abs, 'Blue_P': p_blue,
                # 原始代数指标
                'Overall_CSI_raw': overall_csi_raw,
                'Red_CSI_raw': red_csi_raw,
                'Green_CSI_raw': green_csi_raw,
                'Yellow_CSI_raw': yellow_csi_raw,
                'Blue_CSI_raw': blue_csi_raw,
                # 各组分绝对平均响应强度
                'R_A_abs': R_A_abs, 'R_C_abs': R_C_abs,
                'R_Red_abs': R_Red_abs, 'R_Green_abs': R_Green_abs,
                'R_Yellow_abs': R_Yellow_abs, 'R_Blue_abs': R_Blue_abs
            })
            
    return results

def plot_csi_gradient_distribution_paper(df, is_hg=False):
    """
    绘制颜色选择性指数（CSI）的 1行5列渐变分布图
    """
    fig, axes = plt.subplots(1, 5, figsize=(35, 7.5), dpi=300)
    fig.suptitle(f"{'High Gamma' if is_hg else 'ERP'} - Color Selectivity Index (CSI) Distribution\n(Formula: (R_Color - R_BW) / (R_Color + R_BW))", fontsize=15, fontweight='bold', y=0.97)
    
    comparisons = [
        ('Overall_CSI', 'Overall_P', 'Overall Color vs. Black/White CSI\n(R_Color - R_BW) / (R_Color + R_BW)'),
        ('Red_CSI', 'Red_P', 'Red vs. Black/White CSI\n(R_Red - R_BW) / (R_Red + R_BW)'),
        ('Green_CSI', 'Green_P', 'Green vs. Black/White CSI\n(R_Green - R_BW) / (R_Green + R_BW)'),
        ('Yellow_CSI', 'Yellow_P', 'Yellow vs. Black/White CSI\n(R_Yellow - R_BW) / (R_Yellow + R_BW)'),
        ('Blue_CSI', 'Blue_P', 'Blue vs. Black/White CSI\n(R_Blue - R_BW) / (R_Blue + R_BW)')
    ]
    
    y_vals = df['MNI_Y'].values
    y_min, y_max = y_vals.min(), y_vals.max()
    y_range = (y_max - y_min) if (y_max - y_min) > 0 else 1.0
    
    for idx, (csi_col, p_col, title) in enumerate(comparisons):
        ax = axes[idx]
        df_sorted = df.sort_values(by=csi_col, ascending=True).reset_index(drop=True)
        
        # 寻找显著性分界阈值
        sig_indices = df_sorted.index[df_sorted[p_col] < 0.05].tolist()
        threshold_idx = sig_indices[0] - 0.5 if sig_indices else len(df_sorted) - 0.5
        
        for i, row in df_sorted.iterrows():
            norm_y = (row['MNI_Y'] - y_min) / y_range
            color = plt.cm.coolwarm(norm_y)
            
            if row['Is_Strategy_4']:
                ax.scatter(i, row[csi_col], color=color, s=120, edgecolors='black', linewidths=2.2, zorder=3)
            else:
                ax.scatter(i, row[csi_col], color=color, s=120, zorder=2)
                
            if row[p_col] < 0.05:
                ax.annotate(row['Electrode'], (i, row[csi_col]), xytext=(0, 8), textcoords='offset points', ha='center', 
                            fontsize=9, fontweight='bold', color='black')
                
        ax.axvline(x=threshold_idx, color='#212121', linestyle='--', lw=1.5, alpha=0.8, label='Sig Threshold (p = 0.05)')
        ax.set_title(title, fontsize=11.5, fontweight='bold')
        ax.set_xlabel("Electrode Rank (Sorted by CSI)", fontsize=10.5)
        ax.set_ylabel("Selectivity Index (CSI Value)", fontsize=10.5)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_facecolor('#fdfdfd')
        
    cax = fig.add_axes([0.65, 0.04, 0.22, 0.025])
    norm = plt.Normalize(vmin=y_min, vmax=y_max)
    cb = ColorbarBase(cax, cmap=plt.cm.coolwarm, norm=norm, orientation='horizontal')
    cb.set_label('Electrode Position: Posterior (Blue) $\\rightarrow$ Anterior (Red) [MNI Y]', fontsize=10)
    cb.ax.tick_params(labelsize=8.5)
    
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2b82c9', markersize=10, label='Standard Channel'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e65c00', markeredgecolor='black', markeredgewidth=2.2, markersize=11, label='Strategy 4 Channel'),
        mlines.Line2D([0], [0], color='#212121', linestyle='--', lw=1.5, label='Sig Limit (p < 0.05 right)')
    ]
    axes[0].legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=9.5)
    
    plt.tight_layout(rect=[0, 0.09, 1, 0.94])
    
    out_path = os.path.join(out_dir, f"color_selectivity_index_paper_{'hg' if is_hg else 'erp'}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [FIGURE] Saved paper CSI plot to: {out_path}")

def plot_correlation_paper(df_erp, df_hg):
    """
    绘制新版 CSI 与 MNI_Y 的相关性回归图 (1行5列)
    """
    fig, axes = plt.subplots(1, 5, figsize=(35, 7.5), dpi=300)
    fig.suptitle("Correlation: CSI (Individual Color vs. Achromatic Average) vs. MNI Y coordinate", fontsize=15, fontweight='bold', y=0.97)
    
    comparisons = [
        ('Overall_CSI', 'Overall_P', 'Overall Color vs. Black/White CSI', r"$CSI = \frac{R_C - R_A}{R_C + R_A}$"),
        ('Red_CSI', 'Red_P', 'Red vs. Black/White CSI', r"$CSI = \frac{R_R - R_A}{R_R + R_A}$"),
        ('Green_CSI', 'Green_P', 'Green vs. Black/White CSI', r"$CSI = \frac{R_G - R_A}{R_G + R_A}$"),
        ('Yellow_CSI', 'Yellow_P', 'Yellow vs. Black/White CSI', r"$CSI = \frac{R_Y - R_A}{R_Y + R_A}$"),
        ('Blue_CSI', 'Blue_P', 'Blue vs. Black/White CSI', r"$CSI = \frac{R_B - R_A}{R_B + R_A}$")
    ]
    
    corr_results = []
    
    for idx, (csi_col, p_col, title, formula) in enumerate(comparisons):
        ax = axes[idx]
        
        mni_y_erp = df_erp['MNI_Y'].values
        csi_erp = df_erp[csi_col].values
        mni_y_hg = df_hg['MNI_Y'].values
        csi_hg = df_hg[csi_col].values
        
        mask_erp = ~np.isnan(mni_y_erp) & ~np.isnan(csi_erp)
        mni_y_erp, csi_erp = mni_y_erp[mask_erp], csi_erp[mask_erp]
        
        mask_hg = ~np.isnan(mni_y_hg) & ~np.isnan(csi_hg)
        mni_y_hg, csi_hg = mni_y_hg[mask_hg], csi_hg[mask_hg]
        
        s_r_erp, s_p_erp = spearmanr(mni_y_erp, csi_erp)
        p_r_erp, p_p_erp = pearsonr(mni_y_erp, csi_erp)
        
        s_r_hg, s_p_hg = spearmanr(mni_y_hg, csi_hg)
        p_r_hg, p_p_hg = pearsonr(mni_y_hg, csi_hg)
        
        corr_results.append({
            'Condition': title,
            'ERP_Spearman_R': s_r_erp, 'ERP_Spearman_P': s_p_erp,
            'ERP_Pearson_R': p_r_erp, 'ERP_Pearson_P': p_p_erp,
            'HG_Spearman_R': s_r_hg, 'HG_Spearman_P': s_p_hg,
            'HG_Pearson_R': p_r_hg, 'HG_Pearson_P': p_p_hg
        })
        
        x_min = min(mni_y_erp.min(), mni_y_hg.min())
        x_max = max(mni_y_erp.max(), mni_y_hg.max())
        x_fit = np.linspace(x_min, x_max, 100)
        
        # ERP
        ax.scatter(mni_y_erp, csi_erp, color='#2b82c9', alpha=0.5, s=65, label='ERP Channel', zorder=2)
        slope_erp, intercept_erp, _, _, _ = linregress(mni_y_erp, csi_erp)
        y_fit_erp = slope_erp * x_fit + intercept_erp
        ax.plot(x_fit, y_fit_erp, color='#0f4c81', lw=2.5, linestyle='-', label=f'ERP Trend Line', zorder=3)
        
        # HG
        ax.scatter(mni_y_hg, csi_hg, color='#e65c00', alpha=0.5, s=65, marker='s', label='HG Channel', zorder=2)
        slope_hg, intercept_hg, _, _, _ = linregress(mni_y_hg, csi_hg)
        y_fit_hg = slope_hg * x_fit + intercept_hg
        ax.plot(x_fit, y_fit_hg, color='#b34700', lw=2.5, linestyle='--', label=f'HG Trend Line', zorder=3)
        
        def star(p):
            if p < 0.001: return "***"
            elif p < 0.01: return "**"
            elif p < 0.05: return "*"
            else: return "(n.s.)"
            
        text_str = (
            r"$\bf{ERP\ Correlation:}$" + "\n"
            f"Spearman $r_s$ = {s_r_erp:.3f}{star(s_p_erp)} (p = {s_p_erp:.2e})\n"
            f"Pearson $r_p$ = {p_r_erp:.3f}{star(p_p_erp)} (p = {p_p_erp:.2e})\n\n"
            r"$\bf{High\ Gamma\ Correlation:}$" + "\n"
            f"Spearman $r_s$ = {s_r_hg:.3f}{star(s_p_hg)} (p = {s_p_hg:.2e})\n"
            f"Pearson $r_p$ = {p_r_hg:.3f}{star(p_p_hg)} (p = {p_p_hg:.2e})"
        )
        ax.text(0.04, 0.96, text_str, transform=ax.transAxes, fontsize=9.5,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.45', facecolor='#fbfbfb', alpha=0.9, edgecolor='#cccccc'))
        
        formula_str = r"$\bf{Formula:}$" + f"\n{formula}"
        ax.text(0.96, 0.04, formula_str, transform=ax.transAxes, fontsize=10.5,
                horizontalalignment='right', verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', alpha=0.92, edgecolor='#bbbbbb'))
        
        ax.set_title(title, fontsize=12.5, fontweight='bold', pad=12)
        ax.set_xlabel("Electrode MNI Y Coordinate\n(Posterior <-- 0 --> Anterior)", fontsize=11)
        ax.set_ylabel("Color Selectivity Index (CSI)", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.45)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9.5)
        ax.set_facecolor('#fefefe')
        
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    out_fig_path = os.path.join(out_dir, "color_selectivity_mni_y_correlation_paper.png")
    plt.savefig(out_fig_path, dpi=300)
    plt.close()
    print(f"[FIGURE] Saved paper correlation plot to: {out_fig_path}")
    
    # 保存相关性总结表格
    df_corr = pd.DataFrame(corr_results)
    df_corr.to_excel(os.path.join(doc_dir, 'color_selectivity_correlation_summary_paper.xlsx'), index=False)
    df_corr.to_csv(os.path.join(doc_dir, 'color_selectivity_correlation_summary_paper.csv'), index=False)
    print(f"[SAVED] Correlation summary saved to: doc/color_selectivity_correlation_summary_paper.xlsx")

def main():
    print("="*65)
    print("Step 1_2 (论文公式修正版): 计算各颜色对黑白均值的 CSI")
    print("="*65)
    
    # 1. 计算 ERP 数据
    print("\n[ERP] Analyzing task3 pure color selectivity using paper ratio formula...")
    erp_res = analyze_color_selectivity_paper(is_hg=False)
    df_erp = pd.DataFrame(erp_res)
    df_erp.to_excel(os.path.join(doc_dir, 'select_channel_color_selectivity_paper_erp.xlsx'), index=False)
    df_erp.to_csv(os.path.join(doc_dir, 'select_channel_color_selectivity_paper_erp.csv'), index=False)
    plot_csi_gradient_distribution_paper(df_erp, is_hg=False)
    
    # 2. 计算 HG 数据
    print("\n[HG] Analyzing task3 pure color selectivity using paper ratio formula...")
    hg_res = analyze_color_selectivity_paper(is_hg=True)
    df_hg = pd.DataFrame(hg_res)
    df_hg.to_excel(os.path.join(doc_dir, 'select_channel_color_selectivity_paper_hg.xlsx'), index=False)
    df_hg.to_csv(os.path.join(doc_dir, 'select_channel_color_selectivity_paper_hg.csv'), index=False)
    plot_csi_gradient_distribution_paper(df_hg, is_hg=True)
    
    # 3. 相关性回归
    print("\n[CORRELATION] Analyzing MNI Y correlation with paper formula CSI...")
    plot_correlation_paper(df_erp, df_hg)
    
    print("\n" + "="*65)
    print("Step 1_2 (论文公式修正版) 分析圆满完成！")
    print("="*65)

if __name__ == '__main__':
    main()
