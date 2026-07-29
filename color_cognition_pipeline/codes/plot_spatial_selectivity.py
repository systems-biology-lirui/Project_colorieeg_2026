import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal
import matplotlib.pyplot as plt
from pymatreader import read_mat

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')

# 1. Load Anatomy and coordinates
ieegloc_path = os.path.join(base_dir, 'processed_data', 'test001', 'test001_ieegloc.xlsx')
df_loc = pd.read_excel(ieegloc_path)

coords = {}
for idx, row in df_loc.iterrows():
    elec = str(row.get('Channel', row.get('label', row.iloc[0]))).strip()
    scs_str = str(row.get('SCS', ''))
    if scs_str.startswith('[') and scs_str.endswith(']'):
        try:
            # Parse '[-22.428,-28.366,43.578]'
            vals = [float(x) for x in scs_str.strip('[]').split(',')]
            coords[elec] = vals
        except:
            pass

# 2. Load Type 1 electrodes
type1_path = os.path.join(pipeline_dir, 'data', 'Table_ERP_SingleCategory_Significant.csv')
type1_elecs = set()
if os.path.exists(type1_path):
    df_type1 = pd.read_csv(type1_path)
    # Type 1 are those in target area
    type1_elecs = set(df_type1[df_type1['In_Target_Area'] == True]['Electrode'].astype(str))
    
# 3. Compute Task 3 Color Selectivity (Kruskal-Wallis across 4 colors)
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
color_trigs = ['Trigger-In:51', 'Trigger-In:52', 'Trigger-In:53', 'Trigger-In:54'] # R, Y, B, G

t_idx_stat = np.where((time_ms3 >= 50) & (time_ms3 <= 400))[0]

color_selective_elecs = set()
p_vals = {}

for ch_idx, elec in enumerate(ch_labels3):
    color_responses = []
    for t in color_trigs:
        if t in idx_map3:
            cond_data = data3[idx_map3[t]][:, ch_idx, t_idx_stat]
            # Mean over the time window for each trial
            mean_resp = np.nanmean(cond_data, axis=1)
            mean_resp = mean_resp[~np.isnan(mean_resp)]
            if len(mean_resp) > 0:
                color_responses.append(mean_resp)
                
    if len(color_responses) == 4:
        # Kruskal-Wallis requires at least 4 groups of data
        try:
            stat, p = kruskal(*color_responses)
            p_vals[elec] = p
            if p < 0.05:
                color_selective_elecs.add(elec)
        except ValueError:
            pass

print(f"Found {len(color_selective_elecs)} Color Selective electrodes (Task 3).")

# Save Color Selective Electrodes
pd.DataFrame({
    'Electrode': list(color_selective_elecs),
    'P_value': [p_vals[e] for e in color_selective_elecs]
}).to_csv(os.path.join(pipeline_dir, 'data', 'Task3_ColorSelective.csv'), index=False)

# 4. Plot 2D Spatial Distribution (Axial view: X vs Y)
plt.figure(figsize=(10, 8))

for elec in coords:
    x, y, z = coords[elec]
    is_type1 = elec in type1_elecs
    is_color_sel = elec in color_selective_elecs
    
    if is_type1 and is_color_sel:
        color = 'purple'
        label = 'Both (Type1 & ColorSel)'
        size = 150
        marker = '*'
        zorder = 4
    elif is_type1:
        color = 'red'
        label = 'Type1 (ERP Object Color)'
        size = 100
        marker = 'o'
        zorder = 3
    elif is_color_sel:
        color = 'blue'
        label = 'Color Selective (Task3 HG)'
        size = 100
        marker = 's'
        zorder = 2
    else:
        color = 'lightgray'
        label = 'Other'
        size = 30
        marker = '.'
        zorder = 1
        
    plt.scatter(x, y, c=color, s=size, marker=marker, alpha=0.8, edgecolors='k' if color != 'lightgray' else 'none', zorder=zorder)

# Add single legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.title("Spatial Distribution of Target Electrodes (Axial View: X vs Y)")
plt.xlabel("MNI X (Left -> Right)")
plt.ylabel("MNI Y (Posterior -> Anterior)")
plt.grid(True, linestyle='--', alpha=0.5)

out_img = os.path.join(pipeline_dir, 'images', 'spatial_selectivity_distribution.png')
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Saved plot to {out_img}")
