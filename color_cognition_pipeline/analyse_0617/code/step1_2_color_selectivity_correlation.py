import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import warnings

# 忽略可能出现的一些绘图警告
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
analyse_dir = os.path.join(pipeline_dir, 'analyse_0617')
doc_dir = os.path.join(analyse_dir, 'doc')
result_dir = os.path.join(analyse_dir, 'result')
out_dir = os.path.join(result_dir, 'select_channel', 'color_selectivity')
os.makedirs(out_dir, exist_ok=True)

# 1. 加载数据
erp_path = os.path.join(doc_dir, 'select_channel_color_selectivity_erp.csv')
hg_path = os.path.join(doc_dir, 'select_channel_color_selectivity_hg.csv')

if not os.path.exists(erp_path) or not os.path.exists(hg_path):
    print("[ERROR] CSI 数据文件不存在，请确保已运行 step1_2_color_selectivity.py！")
    exit(1)

df_erp = pd.read_csv(erp_path)
df_hg = pd.read_csv(hg_path)

# 2. 相关性星号辅助函数
def get_p_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "(n.s.)"

# 3. 初始化画布 (1行3列)
fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
fig.suptitle("Correlation: Color Selectivity Index (CSI) vs. MNI Y coordinate", fontsize=16, fontweight='bold', y=0.97)

comparisons = [
    ('Overall_CSI', 'Overall_P', '4-Color Overall (Kruskal-Wallis)', 
     r"$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$"),
    ('RG_CSI', 'RG_P', 'Red vs Green (Wilcoxon Rank-Sum)', 
     r"$CSI = |Z| = \frac{|U - \mu_U|}{\sigma_U}$"),
    ('YB_CSI', 'YB_P', 'Yellow vs Blue (Wilcoxon Rank-Sum)', 
     r"$CSI = |Z| = \frac{|U - \mu_U|}{\sigma_U}$")
]

corr_results = []

for idx, (csi_col, p_col, title, formula) in enumerate(comparisons):
    ax = axes[idx]
    
    # 获取 MNI_Y 与 CSI 数据
    mni_y_erp = df_erp['MNI_Y'].values
    csi_erp = df_erp[csi_col].values
    
    mni_y_hg = df_hg['MNI_Y'].values
    csi_hg = df_hg[csi_col].values
    
    # 过滤 NaN 值以防计算报错
    mask_erp = ~np.isnan(mni_y_erp) & ~np.isnan(csi_erp)
    mni_y_erp, csi_erp = mni_y_erp[mask_erp], csi_erp[mask_erp]
    
    mask_hg = ~np.isnan(mni_y_hg) & ~np.isnan(csi_hg)
    mni_y_hg, csi_hg = mni_y_hg[mask_hg], csi_hg[mask_hg]
    
    # A. 计算相关性
    s_r_erp, s_p_erp = stats.spearmanr(mni_y_erp, csi_erp)
    p_r_erp, p_p_erp = stats.pearsonr(mni_y_erp, csi_erp)
    
    s_r_hg, s_p_hg = stats.spearmanr(mni_y_hg, csi_hg)
    p_r_hg, p_p_hg = stats.pearsonr(mni_y_hg, csi_hg)
    
    corr_results.append({
        'Condition': title,
        'ERP_Spearman_R': s_r_erp, 'ERP_Spearman_P': s_p_erp,
        'ERP_Pearson_R': p_r_erp, 'ERP_Pearson_P': p_p_erp,
        'HG_Spearman_R': s_r_hg, 'HG_Spearman_P': s_p_hg,
        'HG_Pearson_R': p_r_hg, 'HG_Pearson_P': p_p_hg
    })
    
    # 确定 X 轴拟合范围
    x_min = min(mni_y_erp.min(), mni_y_hg.min())
    x_max = max(mni_y_erp.max(), mni_y_hg.max())
    x_fit = np.linspace(x_min, x_max, 100)
    
    # B. 绘制 ERP 数据与拟合线
    ax.scatter(mni_y_erp, csi_erp, color='#2b82c9', alpha=0.5, s=65, label='ERP Channel', zorder=2)
    slope_erp, intercept_erp, _, _, _ = stats.linregress(mni_y_erp, csi_erp)
    y_fit_erp = slope_erp * x_fit + intercept_erp
    ax.plot(x_fit, y_fit_erp, color='#0f4c81', lw=2.5, linestyle='-', label=f'ERP Trend Line', zorder=3)
    
    # C. 绘制 HG 数据与拟合线
    ax.scatter(mni_y_hg, csi_hg, color='#e65c00', alpha=0.5, s=65, marker='s', label='HG Channel', zorder=2)
    slope_hg, intercept_hg, _, _, _ = stats.linregress(mni_y_hg, csi_hg)
    y_fit_hg = slope_hg * x_fit + intercept_hg
    ax.plot(x_fit, y_fit_hg, color='#b34700', lw=2.5, linestyle='--', label=f'HG Trend Line', zorder=3)
    
    # D. 标注相关系数结果文本 (含星号)
    text_str = (
        r"$\bf{ERP\ Correlation:}$" + "\n"
        f"Spearman $r_s$ = {s_r_erp:.3f}{get_p_stars(s_p_erp)} (p = {s_p_erp:.2e})\n"
        f"Pearson $r_p$ = {p_r_erp:.3f}{get_p_stars(p_p_erp)} (p = {p_p_erp:.2e})\n\n"
        r"$\bf{High\ Gamma\ Correlation:}$" + "\n"
        f"Spearman $r_s$ = {s_r_hg:.3f}{get_p_stars(s_p_hg)} (p = {s_p_hg:.2e})\n"
        f"Pearson $r_p$ = {p_r_hg:.3f}{get_p_stars(p_p_hg)} (p = {p_p_hg:.2e})"
    )
    ax.text(0.04, 0.96, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.45', facecolor='#fbfbfb', alpha=0.9, edgecolor='#cccccc'))
            
    # E. 标注数学计算公式
    formula_str = r"$\bf{CSI\ Definition\ Formula:}$" + f"\n{formula}"
    ax.text(0.96, 0.04, formula_str, transform=ax.transAxes, fontsize=11,
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', alpha=0.92, edgecolor='#bbbbbb'))
            
    # 图表细节修饰
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Electrode MNI Y Coordinate\n(Posterior <-- 0 --> Anterior)", fontsize=11)
    ax.set_ylabel("Color Selectivity Index (CSI)", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.set_facecolor('#fefefe')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9.5)

plt.tight_layout(rect=[0, 0.02, 1, 0.94])

# 保存图片
out_fig_path = os.path.join(out_dir, "color_selectivity_mni_y_correlation.png")
plt.savefig(out_fig_path, dpi=300)
plt.close()
print(f"[SUCCESS] Saved correlation distribution figure to: {out_fig_path}")

# 4. 保存计算数据为 Excel 与 CSV
corr_df = pd.DataFrame(corr_results)
corr_xlsx_path = os.path.join(doc_dir, 'color_selectivity_correlation_summary.xlsx')
corr_csv_path = os.path.join(doc_dir, 'color_selectivity_correlation_summary.csv')
corr_df.to_excel(corr_xlsx_path, index=False)
corr_df.to_csv(corr_csv_path, index=False)
print(f"[SUCCESS] Saved correlation summary tables to:\n  - {corr_xlsx_path}\n  - {corr_csv_path}")
