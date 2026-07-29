import os
import numpy as np
import scipy.io as sio
from pymatreader import read_mat
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import warnings

warnings.filterwarnings('ignore')

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')

target_elec = 'F9'
red_memory_trigs = ['Trigger-In:121', 'Trigger-In:122', 'Trigger-In:123', 'Trigger-In:131', 'Trigger-In:132', 'Trigger-In:133']
green_memory_trigs = ['Trigger-In:101', 'Trigger-In:102', 'Trigger-In:103', 'Trigger-In:111', 'Trigger-In:112', 'Trigger-In:113']

def extract_condition_data(mat_data, is_erp, trigs, target_elec):
    if is_erp:
        epoch = mat_data['epoch']
        data = epoch.data # [Cond, Rep, Ch, Time]
        ch_arr = epoch.ch
        ch_names = [str(ch.labels) for ch in ch_arr]
        time_ms = epoch.time_ms
        all_trigs = [str(t) for t in epoch.eventtype] if hasattr(epoch, 'eventtype') else [str(t) for t in epoch.name]
        # In MATLAB, ERP triggers are usually the 'name' field of the epoch cell array if built manually.
        # Wait, for task1 ERP, it was in data. shape: n_cond x n_rep x n_ch x n_time.
        # If it's sio.loadmat, epoch.name might be the trigger list.
        if isinstance(epoch.name, np.ndarray):
            trigger_list = [str(x) for x in epoch.name]
        else:
            trigger_list = []
            
        try:
            ch_idx = ch_names.index(target_elec)
        except ValueError:
            return None, None
            
        idx_list = [trigger_list.index(t) for t in trigs if t in trigger_list]
        if not idx_list: return None, None
        
        c_data = np.concatenate([data[idx, :, ch_idx, :] for idx in idx_list], axis=0)
        return c_data, time_ms
    else:
        epoch = mat_data['epoch']
        time_ms = epoch['time_ms']
        all_trigs = epoch['trigger']
        if isinstance(all_trigs, str): all_trigs = [all_trigs]
        data_cell = epoch['data_cell']
        ch_names = epoch['ch']['labels']
        if isinstance(ch_names, str): ch_names = [ch_names]
        
        try:
            ch_idx = ch_names.index(target_elec)
        except ValueError:
            return None, None
            
        idx_list = [all_trigs.index(t) for t in trigs if t in all_trigs]
        if not idx_list: return None, None
        
        c_data = np.concatenate([data_cell[idx][:, ch_idx, :] for idx in idx_list], axis=0)
        return c_data, time_ms

def plot_f9_memory():
    out_dir = os.path.join(pipeline_dir, 'images', 'task2_memory_color')
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Electrode F9 - Task 2 Memory Color (Red vs Green)", fontsize=16)
    
    # ERP
    erp_path = os.path.join(base_dir, 'processed_data', 'test001', 'task2_ERP_epoched.mat')
    if os.path.exists(erp_path):
        mat_erp = sio.loadmat(erp_path, squeeze_me=True, struct_as_record=False)
        r_erp, t_erp = extract_condition_data(mat_erp, True, red_memory_trigs, target_elec)
        g_erp, _ = extract_condition_data(mat_erp, True, green_memory_trigs, target_elec)
        
        if r_erp is not None and g_erp is not None:
            plot_signal(axes[0], r_erp, g_erp, t_erp, "ERP")
    
    # Subband
    hg_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', 'task2_hg_subband.mat')
    if os.path.exists(hg_path):
        mat_hg = read_mat(hg_path)
        r_hg, t_hg = extract_condition_data(mat_hg, False, red_memory_trigs, target_elec)
        g_hg, _ = extract_condition_data(mat_hg, False, green_memory_trigs, target_elec)
        
        if r_hg is not None and g_hg is not None:
            plot_signal(axes[1], r_hg, g_hg, t_hg, "Subband 60-150Hz")
            
    out_fig = os.path.join(out_dir, "F9_Task2_Memory_Color.png")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print(f"Saved {out_fig}")

def plot_signal(ax, data_r, data_g, time_ms, title):
    # Remove NaN trials
    data_r = data_r[~np.isnan(data_r).any(axis=1)]
    data_g = data_g[~np.isnan(data_g).any(axis=1)]
    
    m_r, sem_r = np.mean(data_r, axis=0), np.std(data_r, axis=0)/np.sqrt(data_r.shape[0])
    m_g, sem_g = np.mean(data_g, axis=0), np.std(data_g, axis=0)/np.sqrt(data_g.shape[0])
    
    ax.plot(time_ms, m_r, color='red', label='Red Memory (Watermelon/Strawberry)')
    ax.fill_between(time_ms, m_r - sem_r, m_r + sem_r, color='red', alpha=0.2)
    
    ax.plot(time_ms, m_g, color='green', label='Green Memory (Kiwi/Cabbage)')
    ax.fill_between(time_ms, m_g - sem_g, m_g + sem_g, color='green', alpha=0.2)
    
    # Stats
    n_time = len(time_ms)
    ymin, ymax = ax.get_ylim()
    sig_y = ymin + (ymax - ymin) * 0.05
    for t in range(n_time):
        stat, p = ranksums(data_r[:, t], data_g[:, t])
        if p < 0.05:
            color = 'red' if stat > 0 else 'green'
            ax.plot(time_ms[t], sig_y, marker='s', color=color, markersize=3, alpha=0.7)
            
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude / Z-score")
    ax.legend(loc='upper right')
    ax.set_xlim([-200, 800])

if __name__ == '__main__':
    plot_f9_memory()
