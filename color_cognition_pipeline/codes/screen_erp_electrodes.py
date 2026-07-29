import numpy as np
import pandas as pd
from scipy.stats import ranksums
import scipy.io as sio
import os
import warnings

warnings.filterwarnings('ignore')

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
out_dir = os.path.join(pipeline_dir, 'data')

# Anatomy targets
anatomy_targets = ['Fusiform', 'Temporal_Inf', 'Temporal_Mid']

# 1. Load Anatomy
ieegloc_path = os.path.join(base_dir, 'processed_data', 'test001', 'test001_ieegloc.xlsx')
df_loc = pd.read_excel(ieegloc_path)
elec_anatomy_map = {}
elec_target_anatomy = set()
for idx, row in df_loc.iterrows():
    elec = str(row.get('label', row.iloc[0])).strip()
    aal3 = str(row.get('AAL3 (MNI-linear)', ''))
    elec_anatomy_map[elec] = aal3
    if any(t.lower() in aal3.lower() for t in anatomy_targets):
        elec_target_anatomy.add(elec)

def check_continuous_sig(c_data, g_data, time_ms, t_start=50, t_end=400, consecutive_pts=25):
    """
    c_data, g_data: [Rep, Time]
    """
    t_idx = np.where((time_ms >= t_start) & (time_ms <= t_end))[0]
    if len(t_idx) < consecutive_pts:
        return False
    
    sig_array = np.zeros(len(t_idx), dtype=bool)
    for i, t in enumerate(t_idx):
        c_vals = c_data[:, t]
        g_vals = g_data[:, t]
        c_vals = c_vals[~np.isnan(c_vals)]
        g_vals = g_vals[~np.isnan(g_vals)]
        if len(c_vals) > 0 and len(g_vals) > 0:
            stat, p = ranksums(c_vals, g_vals)
            if p < 0.05 and stat > 0: # stat > 0 means Color > Gray
                sig_array[i] = True
                
    max_consecutive = 0
    current_consecutive = 0
    for val in sig_array:
        if val:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
            
    return max_consecutive >= consecutive_pts

# Load Task 1 ERP Features
print("Loading Task 1 ERP...")
mat_path = os.path.join(base_dir, 'processed_data', 'test001', 'task1_ERP_epoched.mat')
mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
epoch = mat['epoch']
data = epoch.data # [Cond, Rep, Ch, Time]
ch_arr = epoch.ch
ch_labels = [ch.labels for ch in ch_arr]
if isinstance(ch_labels, np.ndarray) and ch_labels.dtype.kind in {'U', 'S', 'O'}:
    ch_labels = [str(c) for c in ch_labels]
time_ms = epoch.time_ms

cond_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)] # (Color_idx, Gray_idx)

table_A_elecs = []
table_B_elecs = []

print("Analyzing Task 1 ERP Significance...")
for ch_idx, elec in enumerate(ch_labels):
    all_color_data = []
    all_gray_data = []
    single_cat_sig = False
    
    for c_idx, g_idx in cond_pairs:
        c_data = data[c_idx, :, ch_idx, :]
        g_data = data[g_idx, :, ch_idx, :]
        all_color_data.append(c_data)
        all_gray_data.append(g_data)
        
        # Check single category
        if check_continuous_sig(c_data, g_data, time_ms):
            single_cat_sig = True
            
    if single_cat_sig:
        table_A_elecs.append(elec)
        
    pool_color = np.vstack(all_color_data)
    pool_gray = np.vstack(all_gray_data)
    
    if check_continuous_sig(pool_color, pool_gray, time_ms):
        table_B_elecs.append(elec)

# Save result
print("Saving results...")
table_A_df = pd.DataFrame({'Electrode': table_A_elecs})
table_A_df['AAL3'] = table_A_df['Electrode'].map(elec_anatomy_map)
table_A_df['In_Target_Area'] = table_A_df['Electrode'].isin(elec_target_anatomy)

csv_A_path = os.path.join(out_dir, 'Table_ERP_SingleCategory_Significant.csv')
table_A_df.to_csv(csv_A_path, index=False)
table_B_df = pd.DataFrame({'Electrode': table_B_elecs})
table_B_df['AAL3'] = table_B_df['Electrode'].map(elec_anatomy_map)
table_B_df['In_Target_Area'] = table_B_df['Electrode'].isin(elec_target_anatomy)

csv_B_path = os.path.join(out_dir, 'Table_ERP_MergedCategory_Significant.csv')
table_B_df.to_csv(csv_B_path, index=False)

print(f"Found {len(table_A_elecs)} electrodes with single-category continuous significance in ERP.")
print("Results saved to:", csv_A_path)
print(f"Found {len(table_B_elecs)} electrodes with merged-category continuous significance in ERP.")
print("Results saved to:", csv_B_path)
