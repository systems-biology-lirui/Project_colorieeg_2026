import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import kruskal, ranksums
from pymatreader import read_mat
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colorbar import ColorbarBase
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

# 路径配置
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
analyse_dir = os.path.join(pipeline_dir, 'analyse_0617')
feature_dir = os.path.join(analyse_dir, 'feature')
doc_dir = os.path.join(analyse_dir, 'doc')
result_dir = os.path.join(analyse_dir, 'result')

subjects = ['test001', 'test002', 'test003']
# task3 颜色触发器代码 (51: Red, 52: Yellow, 53: Blue, 54: Green)
color_trigs = {
    'Red': 'Trigger-In:51',
    'Yellow': 'Trigger-In:52',
    'Blue': 'Trigger-In:53',
    'Green': 'Trigger-In:54'
}

# ----------------- 数据加载与 CSI 计算 -----------------
def analyze_signal_color_selectivity(is_hg=False):
    """
    计算主要电极在 ERP 或 HG 下的三大 CSI 指数
    """
    # 1. 载入主要筛选电极表
    summary_path = os.path.join(doc_dir, 'select_channel_summary.xlsx')
    if not os.path.exists(summary_path):
        print("[ERROR] select_channel_summary.xlsx 不存在，请先运行 step1_1！")
        return []
        
    df_summary = pd.read_excel(summary_path)
    
    results = []
    
    for subj in subjects:
        # 筛选出属于当前被试的电极记录
        subj_df = df_summary[df_summary['Subject'] == subj]
        if subj_df.empty:
            continue
            
        # 2. 读取被试的 task3 数据
        if is_hg:
            mat_path = os.path.join(feature_dir, subj, 'task3_hg_subband.mat')
        else:
            mat_path = os.path.join(feature_dir, subj, 'task3_ERP_epoched.mat')
            
        if not os.path.exists(mat_path):
            print(f"  [WARNING] 被试 {subj} 的 task3 mat 文件不存在，跳过：{mat_path}")
            continue
            
        mat = read_mat(mat_path)
        epoch = mat['epoch']
        trigs = list(epoch['trigger'])
        ch_labels = list(epoch['ch']['labels'])
        
        # 兼容时程轴
        if 'time_ms' in epoch:
            time_ms = epoch['time_ms']
        else:
            # 兼容 ERP
            time_ms = np.linspace(-500, 998, epoch['data'].shape[-1])
            
        t_idx_stat = np.where((time_ms >= 50) & (time_ms <= 400))[0]
        
        # 建立触发器名称到索引的映射
        trig_indices = {}
        for c_name, c_code in color_trigs.items():
            if c_code in trigs:
                trig_indices[c_name] = trigs.index(c_code)
                
        if len(trig_indices) < 4:
            print(f"  [WARNING] 被试 {subj} 的 task3 中缺少部分颜色触发器，跳过！")
            continue
            
        # 3. 对该被试的每一个主要筛选电极计算 CSI
        for _, row in subj_df.iterrows():
            elec = str(row['Electrode']).strip()
            
            # 是否在 task3 中包含此通道
            if elec not in ch_labels:
                continue
                
            ch_idx = ch_labels.index(elec)
            
            # 提取 MNI_Y 与策略 4 标记
            mni_y = float(row['MNI_Y']) if not pd.isna(row['MNI_Y']) else 0.0
            
            if is_hg:
                is_stra4 = '4' in str(row['HG_Strategies_Matched'])
            else:
                is_stra4 = '4' in str(row['ERP_Strategies_Matched'])
                
            # 提取 50-400ms 平均响应
            cond_means = {}
            for c_name, idx in trig_indices.items():
                if is_hg:
                    # data_cell list of (Rep, Ch, Time)
                    cond_data = epoch['data_cell'][idx][:, ch_idx, t_idx_stat]
                else:
                    # data 4D (Cond, Rep, Ch, Time)
                    cond_data = epoch['data'][idx, :, ch_idx, t_idx_stat]
                    
                # 过滤 NaN 的 trials
                cond_data = cond_data[~np.isnan(cond_data).any(axis=1)]
                # 计算每个 trial 在 50-400ms 均值
                mean_resp = np.nanmean(cond_data, axis=1)
                cond_means[c_name] = mean_resp
                
            # 计算 KW 和 ranksums 检验
            try:
                # 1. 4-Color Overall Kruskal-Wallis
                stat_all, p_all = kruskal(cond_means['Red'], cond_means['Yellow'], cond_means['Blue'], cond_means['Green'])
                
                # 2. Red vs Green Ranksum
                stat_rg, p_rg = ranksums(cond_means['Red'], cond_means['Green'])
                
                # 3. Yellow vs Blue Ranksum
                stat_yb, p_yb = ranksums(cond_means['Yellow'], cond_means['Blue'])
                
                results.append({
                    'Subject': subj,
                    'Electrode': elec,
                    'MNI_Y': mni_y,
                    'Is_Strategy_4': is_stra4,
                    'Overall_CSI': stat_all,
                    'Overall_P': p_all,
                    'RG_CSI': abs(stat_rg),
                    'RG_P': p_rg,
                    'YB_CSI': abs(stat_yb),
                    'YB_P': p_yb
                })
            except Exception as e:
                print(f"  [ERROR] 计算电极 {subj} - {elec} 的 CSI 失败: {e}")
                
    return results

# ----------------- 渐变 CSI 分布图绘制 -----------------
def plot_csi_gradient_distribution(df, is_hg=False):
    """
    绘制颜色选择性指数（CSI）的 1行3列渐变分布图
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
    fig.suptitle(f"{'High Gamma' if is_hg else 'ERP'} Signals - Color Selectivity Index (CSI) Distribution", fontsize=16, fontweight='bold', y=0.97)
    
    comparisons = [
        ('Overall_CSI', 'Overall_P', '4-Color Overall CSI (K-W H-stat)'),
        ('RG_CSI', 'RG_P', 'Red vs Green CSI (abs Z-stat)'),
        ('YB_CSI', 'YB_P', 'Yellow vs Blue CSI (abs Z-stat)')
    ]
    
    # 归一化 MNI_Y 用于蓝红渐变上色
    y_vals = df['MNI_Y'].values
    y_min, y_max = y_vals.min(), y_vals.max()
    # 避免除以 0
    y_range = (y_max - y_min) if (y_max - y_min) > 0 else 1.0
    
    for idx, (csi_col, p_col, title) in enumerate(comparisons):
        ax = axes[idx]
        
        # 按照 CSI 升序排序
        df_sorted = df.sort_values(by=csi_col, ascending=True).reset_index(drop=True)
        
        # 寻找显著性分界阈值索引
        sig_indices = df_sorted.index[df_sorted[p_col] < 0.05].tolist()
        if sig_indices:
            threshold_idx = sig_indices[0] - 0.5
        else:
            threshold_idx = len(df_sorted) - 0.5
            
        x = np.arange(len(df_sorted))
        
        # 逐个画点
        for i, row in df_sorted.iterrows():
            # Y 轴映射为 coolwarm 渐变色
            norm_y = (row['MNI_Y'] - y_min) / y_range
            color = plt.cm.coolwarm(norm_y)
            
            if row['Is_Strategy_4']:
                # 策略4的电极带粗黑边圈
                ax.scatter(i, row[csi_col], color=color, s=120, edgecolors='black', linewidths=2.2, zorder=3)
            else:
                ax.scatter(i, row[csi_col], color=color, s=120, zorder=2)
                
            # 标注显著的电极名称
            if row[p_col] < 0.05:
                # 显著电极名称加粗标注，稍微向上偏移
                ax.annotate(row['Electrode'], (i, row[csi_col]), xytext=(0, 8), textcoords='offset points', ha='center', 
                            fontsize=9, fontweight='bold', color='black')
                
        # 绘制显著分界虚线
        ax.axvline(x=threshold_idx, color='#212121', linestyle='--', lw=1.5, alpha=0.8, label='Significance Threshold (p = 0.05)')
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("Electrode Rank (Sorted by CSI)", fontsize=10.5)
        ax.set_ylabel("Color Selectivity Index (CSI)", fontsize=10.5)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_facecolor('#fdfdfd')
        
    # 添加整体图例与渐变色标
    # 1. 颜色映射色标 (Colorbar) 放在图底部偏右
    cax = fig.add_axes([0.65, 0.05, 0.25, 0.025])
    norm = plt.Normalize(vmin=y_min, vmax=y_max)
    cb = ColorbarBase(cax, cmap=plt.cm.coolwarm, norm=norm, orientation='horizontal')
    cb.set_label('Electrode Position: Posterior (Blue) $\\rightarrow$ Anterior (Red) [MNI Y]', fontsize=10)
    cb.ax.tick_params(labelsize=8.5)
    
    # 2. 通用图例放在底部偏左
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2b82c9', markersize=10, label='Standard Channel'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e65c00', markeredgecolor='black', markeredgewidth=2.2, markersize=11, label='Strategy 4 Channel (Black Border)'),
        mlines.Line2D([0], [0], color='#212121', linestyle='--', lw=1.5, label='Significance Limit (p < 0.05 right)')
    ]
    axes[0].legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=9.5)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.94])
    
    # 保存结果
    out_dir = os.path.join(result_dir, 'select_channel', 'color_selectivity')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"color_selectivity_index_{'hg' if is_hg else 'erp'}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved CSI Distribution figure to: {out_path}")

# ----------------- 主流程 -----------------
def run_color_selectivity_analysis():
    print("="*60)
    print("Step 1_2: Running Pure Color Selectivity Index Analysis (Task 3)")
    print("="*60)
    
    # A. 分析 ERP 数据并保存
    print("\n[ERP] Analyzing task3 pure color selectivity...")
    erp_results = analyze_signal_color_selectivity(is_hg=False)
    if erp_results:
        df_erp = pd.DataFrame(erp_results)
        erp_xlsx_path = os.path.join(doc_dir, 'select_channel_color_selectivity_erp.xlsx')
        erp_csv_path = os.path.join(doc_dir, 'select_channel_color_selectivity_erp.csv')
        df_erp.to_excel(erp_xlsx_path, index=False)
        df_erp.to_csv(erp_csv_path, index=False)
        print(f"  Saved ERP CSI data to: {erp_xlsx_path} and .csv")
        
        # 绘图
        plot_csi_gradient_distribution(df_erp, is_hg=False)
    else:
        print("  [WARNING] 无符合条件的 ERP 电极计算结果！")
        
    # B. 分析 HG 数据并保存
    print("\n[HG] Analyzing task3 pure color selectivity...")
    hg_results = analyze_signal_color_selectivity(is_hg=True)
    if hg_results:
        df_hg = pd.DataFrame(hg_results)
        hg_xlsx_path = os.path.join(doc_dir, 'select_channel_color_selectivity_hg.xlsx')
        hg_csv_path = os.path.join(doc_dir, 'select_channel_color_selectivity_hg.csv')
        df_hg.to_excel(hg_xlsx_path, index=False)
        df_hg.to_csv(hg_csv_path, index=False)
        print(f"  Saved HG CSI data to: {hg_xlsx_path} and .csv")
        
        # 绘图
        plot_csi_gradient_distribution(df_hg, is_hg=True)
    else:
        print("  [WARNING] 无符合条件的 HG 电极计算结果！")
        
    print("\n" + "="*60)
    print("Step 1_2 Color Selectivity Analysis Process Completed Successfully!")
    print("="*60)

if __name__ == '__main__':
    run_color_selectivity_analysis()
