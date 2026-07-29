import os
import sys
import numpy as np
import scipy.io as sio
from pymatreader import read_mat
import pandas as pd
from scipy.stats import kruskal, ranksums
import warnings

warnings.filterwarnings('ignore')

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')

def get_roi_electrodes(subject):
    loc_path = os.path.join(base_dir, 'processed_data', subject, f'{subject}_ieegloc.xlsx')
    if not os.path.exists(loc_path):
        print(f"Missing ieegloc.xlsx for {subject}")
        return []
    df = pd.read_excel(loc_path)
    cols = df.columns[4:]
    roi_mask = df[cols].astype(str).apply(lambda x: x.str.contains('Fusiform|Temporal_Inf|Temporal_Mid', case=False, na=False)).any(axis=1)
    return df[roi_mask].iloc[:, 0].astype(str).tolist()

def select_type1(subject):
    roi_elecs = get_roi_electrodes(subject)
    if not roi_elecs: return []
    
    erp_path = os.path.join(base_dir, 'processed_data', subject, 'task1_ERP_epoched.mat')
    if not os.path.exists(erp_path): return []
    
    mat = sio.loadmat(erp_path, squeeze_me=True, struct_as_record=False)
    epoch = mat['epoch']
    data = epoch.data # [Cond, Rep, Ch, Time]
    ch_names = [str(ch.labels) for ch in epoch.ch]
    time_ms = epoch.time_ms if hasattr(epoch, 'time_ms') else np.linspace(-500, 998, data[0].shape[-1])
    
    t_idx = np.where((time_ms >= 50) & (time_ms <= 400))[0]
    if len(t_idx) == 0: return []
    
    categories = ['Face', 'Object', 'Body', 'Place']
    cond_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)] # (Color, Gray)
    
    type1_elecs = []
    win_len = 25 # 50ms at 500Hz
    
    for elec in roi_elecs:
        if elec not in ch_names: continue
        ch_idx = ch_names.index(elec)
        
        is_type1 = False
        for c_idx, g_idx in cond_pairs:
            c_data = data[c_idx][:, ch_idx, :]
            g_data = data[g_idx][:, ch_idx, :]
            
            c_data = c_data[~np.isnan(c_data).any(axis=1)]
            g_data = g_data[~np.isnan(g_data).any(axis=1)]
            if len(c_data) == 0 or len(g_data) == 0: continue
            
            sig_array = np.zeros(len(t_idx))
            for i, t in enumerate(t_idx):
                stat, p = ranksums(c_data[:, t], g_data[:, t])
                if p < 0.05 and stat > 0: # Color > Gray
                    sig_array[i] = 1
                    
            # Find continuous 1s
            if np.any(sig_array):
                count = 0
                for val in sig_array:
                    if val == 1:
                        count += 1
                        if count >= win_len:
                            is_type1 = True
                            break
                    else:
                        count = 0
            if is_type1: break
            
        if is_type1:
            type1_elecs.append(elec)
            
    df_out = pd.DataFrame({'Electrode': type1_elecs, 'In_Target_Area': True})
    out_path = os.path.join(pipeline_dir, 'data', f'{subject}_Table_ERP_SingleCategory_Significant.csv')
    df_out.to_csv(out_path, index=False)
    print(f"[{subject}] Found {len(type1_elecs)} Type 1 electrodes.")
    return type1_elecs

def select_colorwithsti(subject):
    hg_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', subject, 'task3_hg_subband.mat')
    if not os.path.exists(hg_path): return []
    
    mat = read_mat(hg_path)
    epoch = mat['epoch']
    time_ms = epoch['time_ms']
    triggers = epoch['trigger']
    if isinstance(triggers, str): triggers = [triggers]
    data_cell = epoch['data_cell']
    ch_labels = epoch['ch']['labels']
    if isinstance(ch_labels, str): ch_labels = [ch_labels]
    
    idx_map = {trig: i for i, trig in enumerate(triggers)}
    t_idx = np.where((time_ms >= 50) & (time_ms <= 400))[0]
    
    color_elecs = []
    for ch_idx, elec in enumerate(ch_labels):
        cond_means = {}
        for trig_name, trig_code in [('Red', 'Trigger-In:51'), ('Yellow', 'Trigger-In:52'), ('Blue', 'Trigger-In:53'), ('Green', 'Trigger-In:54')]:
            if trig_code in idx_map:
                cond_data = data_cell[idx_map[trig_code]][:, ch_idx, t_idx]
                mean_resp = np.nanmean(cond_data, axis=1)
                mean_resp = mean_resp[~np.isnan(mean_resp)]
                if len(mean_resp) > 0:
                    cond_means[trig_name] = mean_resp
                    
        if len(cond_means) == 4:
            stat, p = kruskal(cond_means['Red'], cond_means['Yellow'], cond_means['Blue'], cond_means['Green'])
            if p < 0.05:
                color_elecs.append(elec)
                
    df_out = pd.DataFrame({'Electrode': color_elecs})
    out_path = os.path.join(pipeline_dir, 'data', f'{subject}_color_selective_channels.csv')
    df_out.to_csv(out_path, index=False)
    print(f"[{subject}] Found {len(color_elecs)} colorwithsti electrodes.")
    return color_elecs

if __name__ == '__main__':
    for subj in ['test002', 'test003']:
        select_type1(subj)
        select_colorwithsti(subj)
