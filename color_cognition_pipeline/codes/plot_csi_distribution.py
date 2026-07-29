import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal, ranksums
import matplotlib.pyplot as plt
from pymatreader import read_mat

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
import sys
subject = sys.argv[1] if len(sys.argv) > 1 else 'test001'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')

# 1. Load electrode groups
type1_path = os.path.join(pipeline_dir, 'data', f'{subject}_Table_ERP_SingleCategory_Significant.csv')
type1_elecs = set()
if os.path.exists(type1_path):
    df_type1 = pd.read_csv(type1_path)
    type1_elecs = set(df_type1[df_type1['In_Target_Area'] == True]['Electrode'].astype(str))

colorsti_elecs = set()
colorsti_path = os.path.join(pipeline_dir, 'data', f'{subject}_color_selective_channels.csv')
if os.path.exists(colorsti_path):
    df_c = pd.read_csv(colorsti_path)
    colorsti_elecs = set(df_c['Electrode'].astype(str))
elif subject == 'test001':
    colorsti_elecs = {'D3', 'D4', 'D5', 'D6', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10'}

# 2. Compute Task 3 Color Selectivity
mat3_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', subject, 'task3_hg_subband.mat')
if not os.path.exists(mat3_path) and subject == 'test001':
    mat3_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', 'task3_hg_subband.mat')
mat3 = read_mat(mat3_path)
epoch3 = mat3['epoch']
time_ms3 = epoch3['time_ms']
triggers3 = epoch3['trigger']
if isinstance(triggers3, str): triggers3 = [triggers3]
data3 = epoch3['data_cell']
ch_labels3 = epoch3['ch']['labels']
if isinstance(ch_labels3, str): ch_labels3 = [ch_labels3]

idx_map3 = {trig: i for i, trig in enumerate(triggers3)}
# 51: Red, 52: Yellow, 53: Blue, 54: Green

t_idx_stat = np.where((time_ms3 >= 50) & (time_ms3 <= 400))[0]

all_results = {'Overall': [], 'Red_Green': [], 'Yellow_Blue': []}

for ch_idx, elec in enumerate(ch_labels3):
    cond_means = {}
    for trig_name, trig_code in [('Red', 'Trigger-In:51'), ('Yellow', 'Trigger-In:52'), ('Blue', 'Trigger-In:53'), ('Green', 'Trigger-In:54')]:
        if trig_code in idx_map3:
            cond_data = data3[idx_map3[trig_code]][:, ch_idx, t_idx_stat]
            mean_resp = np.nanmean(cond_data, axis=1)
            mean_resp = mean_resp[~np.isnan(mean_resp)]
            if len(mean_resp) > 0:
                cond_means[trig_name] = mean_resp
                
    if len(cond_means) == 4:
        # Overall Kruskal-Wallis
        stat_all, p_all = kruskal(cond_means['Red'], cond_means['Yellow'], cond_means['Blue'], cond_means['Green'])
        all_results['Overall'].append({
            'Electrode': elec,
            'CSI': stat_all,
            'P_value': p_all,
            'Is_Type1': elec in type1_elecs,
            'Is_ColorSti': elec in colorsti_elecs,
            'Is_Selective': p_all < 0.05
        })
        
        # Red vs Green Ranksum
        stat_rg, p_rg = ranksums(cond_means['Red'], cond_means['Green'])
        all_results['Red_Green'].append({
            'Electrode': elec,
            'CSI': abs(stat_rg),
            'P_value': p_rg,
            'Is_Type1': elec in type1_elecs,
            'Is_ColorSti': elec in colorsti_elecs,
            'Is_Selective': p_rg < 0.05
        })
        
        # Yellow vs Blue Ranksum
        stat_yb, p_yb = ranksums(cond_means['Yellow'], cond_means['Blue'])
        all_results['Yellow_Blue'].append({
            'Electrode': elec,
            'CSI': abs(stat_yb),
            'P_value': p_yb,
            'Is_Type1': elec in type1_elecs,
            'Is_ColorSti': elec in colorsti_elecs,
            'Is_Selective': p_yb < 0.05
        })

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
titles = {'Overall': '4-Color Overall CSI (K-W H-stat)', 
          'Red_Green': 'Red vs Green CSI (abs Z-stat)', 
          'Yellow_Blue': 'Yellow vs Blue CSI (abs Z-stat)'}

for idx, comp in enumerate(['Overall', 'Red_Green', 'Yellow_Blue']):
    df_res = pd.DataFrame(all_results[comp])
    df_res = df_res.sort_values(by='CSI', ascending=True).reset_index(drop=True)
    
    sig_indices = df_res.index[df_res['Is_Selective']].tolist()
    if sig_indices:
        threshold_idx = sig_indices[0] - 0.5
    else:
        threshold_idx = len(df_res)
        
    ax = axes[idx]
    x = np.arange(len(df_res))
    
    for i, row in df_res.iterrows():
        # Colors: Type1 -> Red, ColorSti -> Blue, Others -> Gray
        if row['Is_Type1']:
            color = '#D62728' # Red
        elif row['Is_ColorSti']:
            color = '#1F77B4' # Blue
        else:
            color = '#7F7F7F' # Gray
            
        if row['Is_Selective']:
            # Open circle
            ax.scatter(i, row['CSI'], facecolors='none', edgecolors=color, s=80, linewidths=2)
        else:
            # Filled circle
            ax.scatter(i, row['CSI'], color=color, s=80)

    ax.axvline(x=threshold_idx, color='k', linestyle='--', alpha=0.7)
    ax.set_title(titles[comp], fontsize=14)
    ax.set_xlabel("Electrode Rank")
    if idx == 0:
        ax.set_ylabel("Color Selectivity Index (CSI)")

    # Annotate significant Type 1 and ColorSti electrodes
    for i, row in df_res.iterrows():
        if row['Is_Selective'] and (row['Is_Type1'] or row['Is_ColorSti']):
            ax.annotate(row['Electrode'], (i, row['CSI']), xytext=(0, 10), textcoords='offset points', ha='center', 
                        color='#D62728' if row['Is_Type1'] else '#1F77B4', fontweight='bold', fontsize=9)

import matplotlib.lines as mlines
legend_elements = [
    mlines.Line2D([0], [0], marker='o', color='w', label='Type 1 (Object Color ERP)', markerfacecolor='#D62728', markersize=10),
    mlines.Line2D([0], [0], marker='o', color='w', label='colorsti (D3-D6, G5-G10)', markerfacecolor='#1F77B4', markersize=10),
    mlines.Line2D([0], [0], marker='o', color='w', label='Other Electrodes', markerfacecolor='#7F7F7F', markersize=10),
    mlines.Line2D([0], [0], marker='o', color='w', label='Non-Selective', markerfacecolor='k', markersize=10),
    mlines.Line2D([0], [0], marker='o', color='w', label='Selective (p<0.05)', markeredgecolor='k', markerfacecolor='none', markersize=10, markeredgewidth=2),
    mlines.Line2D([0], [0], color='k', linestyle='--', label='Significance Threshold')
]
axes[0].legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
out_dir = os.path.join(pipeline_dir, 'images', subject)
os.makedirs(out_dir, exist_ok=True)
out_img = os.path.join(out_dir, 'color_selectivity_index_distribution_all.png')
plt.savefig(out_img, dpi=300)
print(f"Saved Multi-CSI plot to {out_img}")
