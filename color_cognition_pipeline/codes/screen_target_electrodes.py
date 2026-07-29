import numpy as np
import pandas as pd
from scipy.stats import ranksums
import os
import warnings
from pymatreader import read_mat

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

print(f"Found {len(elec_target_anatomy)} electrodes in target anatomy (Fusiform/ITG/MTG).")

def check_continuous_sig(c_data, g_data, time_ms, t_start=50, t_end=400, consecutive_pts=25):
    """
    Check if there are >= `consecutive_pts` consecutive time points in [t_start, t_end] 
    where Color > Gray and p < 0.05.
    c_data, g_data: [Rep, Time]
    """
    t_idx = np.where((time_ms >= t_start) & (time_ms <= t_end))[0]
    if len(t_idx) < consecutive_pts:
        return False
    
    # Point-by-point stats
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
    
    # Find longest continuous True sequence
    max_consecutive = 0
    current_consecutive = 0
    for val in sig_array:
        if val:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
            
    return max_consecutive >= consecutive_pts


# Load Task 1 Features
print("Loading Task 1 Subband 60-150Hz...")
mat1_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', 'task1_hg_subband.mat')
mat1 = read_mat(mat1_path)
epoch1 = mat1['epoch']
time_ms1 = epoch1['time_ms']
triggers1 = epoch1['trigger']
if isinstance(triggers1, str): triggers1 = [triggers1]
data1 = epoch1['data_cell'] # list of [Rep, Ch, Time]
ch_labels1 = epoch1['ch']['labels']
if isinstance(ch_labels1, str): ch_labels1 = [ch_labels1]

# Map Task 1 Triggers
cat_conds = {
    'Face': ('Trigger-In:11', 'Trigger-In:12'),
    'Object': ('Trigger-In:21', 'Trigger-In:22'),
    'Body': ('Trigger-In:31', 'Trigger-In:32'),
    'Place': ('Trigger-In:41', 'Trigger-In:42')
}

idx_map1 = {trig: i for i, trig in enumerate(triggers1)}

table_A_elecs = []
table_B_elecs = []

print("Analyzing Task 1 Significance...")
for ch_idx, elec in enumerate(ch_labels1):
    single_cat_sig = False
    all_color_data = []
    all_gray_data = []
    
    for cat, (c_trig, g_trig) in cat_conds.items():
        if c_trig in idx_map1 and g_trig in idx_map1:
            c_data = data1[idx_map1[c_trig]][:, ch_idx, :]
            g_data = data1[idx_map1[g_trig]][:, ch_idx, :]
            
            all_color_data.append(c_data)
            all_gray_data.append(g_data)
            
            if check_continuous_sig(c_data, g_data, time_ms1):
                single_cat_sig = True
    
    if single_cat_sig:
        table_A_elecs.append(elec)
        
    if len(all_color_data) > 0:
        pool_color = np.vstack(all_color_data)
        pool_gray = np.vstack(all_gray_data)
        if check_continuous_sig(pool_color, pool_gray, time_ms1):
            table_B_elecs.append(elec)

# Load Task 3 Features
print("Loading Task 3 Subband 60-150Hz...")
mat3_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', 'task3_hg_subband.mat')
try:
    mat3 = read_mat(mat3_path)
    epoch3 = mat3['epoch']
    time_ms3 = epoch3['time_ms']
    triggers3 = epoch3['trigger']
    if isinstance(triggers3, str): triggers3 = [triggers3]
    data3 = epoch3['data_cell']
    ch_labels3 = epoch3['ch']['labels']
    if isinstance(ch_labels3, str): ch_labels3 = [ch_labels3]
    
    # Task 3 analysis: Color Patches vs Achromatic
    color_trigs = ['Trigger-In:51', 'Trigger-In:52', 'Trigger-In:53', 'Trigger-In:54']
    achro_trigs = ['Trigger-In:55', 'Trigger-In:56']
    
    idx_map3 = {trig: i for i, trig in enumerate(triggers3)}
    
    t_idx_stat = np.where((time_ms3 >= 50) & (time_ms3 <= 400))[0]
    
    task3_selective = set()
    for ch_idx, elec in enumerate(ch_labels3):
        color_pool = []
        for t in color_trigs:
            if t in idx_map3:
                color_pool.append(data3[idx_map3[t]][:, ch_idx, :])
                
        achro_pool = []
        for t in achro_trigs:
            if t in idx_map3:
                achro_pool.append(data3[idx_map3[t]][:, ch_idx, :])
                
        if len(color_pool) > 0 and len(achro_pool) > 0:
            color_data = np.vstack(color_pool)
            achro_data = np.vstack(achro_pool)
            
            color_mean_window = np.nanmean(color_data[:, t_idx_stat], axis=1)
            achro_mean_window = np.nanmean(achro_data[:, t_idx_stat], axis=1)
            
            if len(color_mean_window) > 0 and len(achro_mean_window) > 0:
                stat, p = ranksums(color_mean_window, achro_mean_window)
                if p < 0.05 and stat > 0:
                    task3_selective.add(elec)
except Exception as e:
    print("Error loading or processing Task 3:", e)
    task3_selective = set()

# Combine & Export Results
print("Combining results...")
table_A_df = pd.DataFrame({'Electrode': table_A_elecs})
table_A_df['AAL3'] = table_A_df['Electrode'].map(elec_anatomy_map)
table_A_df.to_csv(os.path.join(out_dir, 'TableA_SingleCategory_Significant.csv'), index=False)

table_B_df = pd.DataFrame({'Electrode': table_B_elecs})
table_B_df['AAL3'] = table_B_df['Electrode'].map(elec_anatomy_map)
table_B_df.to_csv(os.path.join(out_dir, 'TableB_MergedCategory_Significant.csv'), index=False)

# Intersection: Table A or B AND Task3 Selective AND Target Anatomy
union_ab = set(table_A_elecs).union(set(table_B_elecs))
intersection_elecs = list(union_ab.intersection(task3_selective).intersection(elec_target_anatomy))

inter_df = pd.DataFrame({'Electrode': intersection_elecs})
inter_df['AAL3'] = inter_df['Electrode'].map(elec_anatomy_map)
inter_df.to_csv(os.path.join(out_dir, 'Final_Candidate_Electrodes.csv'), index=False)

print(f"Found {len(table_A_elecs)} electrodes with single-category continuous significance.")
print(f"Found {len(table_B_elecs)} electrodes with merged-category continuous significance.")
print(f"Found {len(task3_selective)} electrodes with Task 3 color patch selectivity.")
print(f"Found {len(intersection_elecs)} final candidate electrodes meeting all criteria.")

