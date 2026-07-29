import numpy as np
from pymatreader import read_mat
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import os
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
import sys
subject = sys.argv[1] if len(sys.argv) > 1 else 'test001'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')

if subject == 'test001':
    target_path = os.path.join(pipeline_dir, 'data', 'Table_ERP_SingleCategory_Significant.csv')
else:
    target_path = os.path.join(pipeline_dir, 'data', f'{subject}_Table_ERP_SingleCategory_Significant.csv')

if os.path.exists(target_path):
    df_cand = pd.read_csv(target_path)
    target_elecs = df_cand[df_cand['In_Target_Area'] == True]['Electrode'].astype(str).tolist()
else:
    target_elecs = []


categories = ['Face', 'Object', 'Body', 'Place', 'Merged_All']
cond_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), ([0, 2, 4, 6], [1, 3, 5, 7])]

features_to_plot = {
    'subband_60_150': os.path.join(pipeline_dir, 'feature', 'subband_60_150', subject, 'task1_hg_subband.mat'),
    'erp': os.path.join(base_dir, 'processed_data', subject, 'task1_ERP_epoched.mat')
}

def moving_average(a, n=5):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_for_feature(feature_type, mat_path):
    print(f"Processing feature type: {feature_type}")
    
    if not os.path.exists(mat_path):
        if feature_type == 'subband_60_150' and subject == 'test001':
            mat_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', 'task1_hg_subband.mat')
            
        if not os.path.exists(mat_path):
            print(f"File not found: {mat_path}")
            return

    out_dir = os.path.join(pipeline_dir, 'images', subject, 'channel_type1', feature_type)
    os.makedirs(out_dir, exist_ok=True)
    
    if feature_type == 'subband_60_150':
        mat = read_mat(mat_path)
        epoch = mat['epoch']
        data_cell = epoch['data_cell'] # list of [Rep, Ch, Time]
        ch_names = epoch['ch']['labels']
        if isinstance(ch_names, str): ch_names = [ch_names]
        time_ms = epoch['time_ms']
        
        n_cond = len(data_cell)
        n_rep, n_ch, n_time = data_cell[0].shape
    else:
        import scipy.io as sio
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        epoch = mat['epoch']
        data = epoch.data # [Cond, Rep, Ch, Time]
        ch_arr = epoch.ch
        ch_names = [str(ch.labels) for ch in ch_arr]
        time_ms = epoch.time_ms if hasattr(epoch, 'time_ms') else np.linspace(-500, 998, data[0].shape[-1])
        
        if isinstance(ch_names, np.ndarray) and ch_names.dtype.kind in {'U', 'S', 'O'}:
            ch_names = [str(c) for c in ch_names]
        
        data_cell = None
        n_cond, n_rep, n_ch, n_time = data.shape
        
    t_idx_stat = np.where((time_ms >= 100) & (time_ms <= 500))[0]

    for elec in target_elecs:
        try:
            ch_idx = ch_names.index(elec)
        except ValueError:
            print(f"Electrode {elec} not found in data.")
            continue
            
        fig, axes = plt.subplots(5, 2, figsize=(12, 20), gridspec_kw={'width_ratios': [3, 1]})
        fig.suptitle(f"Electrode {elec} - {feature_type.upper()}", fontsize=18)
        
        for i, (cat_name, (c_idx, g_idx)) in enumerate(zip(categories, cond_pairs)):
            ax_time = axes[i, 0]
            ax_bar = axes[i, 1]
            
            if isinstance(c_idx, list):
                if data_cell is not None:
                    c_data = np.concatenate([data_cell[idx][:, ch_idx, :] for idx in c_idx], axis=0)
                    g_data = np.concatenate([data_cell[idx][:, ch_idx, :] for idx in g_idx], axis=0)
                else:
                    c_data = np.concatenate([data[idx, :, ch_idx, :] for idx in c_idx], axis=0)
                    g_data = np.concatenate([data[idx, :, ch_idx, :] for idx in g_idx], axis=0)
            else:
                if data_cell is not None:
                    c_data = data_cell[c_idx][:, ch_idx, :]
                    g_data = data_cell[g_idx][:, ch_idx, :]
                else:
                    c_data = data[c_idx, :, ch_idx, :] # [Rep, Time]
                    g_data = data[g_idx, :, ch_idx, :] # [Rep, Time]
            
            # Remove bad epochs (NaNs)
            c_data = c_data[~np.isnan(c_data).any(axis=1)]
            g_data = g_data[~np.isnan(g_data).any(axis=1)]
            
            c_mean = np.mean(c_data, axis=0)
            c_sem = np.std(c_data, axis=0) / np.sqrt(c_data.shape[0])
            g_mean = np.mean(g_data, axis=0)
            g_sem = np.std(g_data, axis=0) / np.sqrt(g_data.shape[0])
            
            # 1. Time Series Plot
            ax_time.plot(time_ms, c_mean, color='red', label='Color')
            ax_time.fill_between(time_ms, c_mean - c_sem, c_mean + c_sem, color='red', alpha=0.2)
            
            ax_time.plot(time_ms, g_mean, color='black', label='Gray')
            ax_time.fill_between(time_ms, g_mean - g_sem, g_mean + g_sem, color='black', alpha=0.2)
            
            # Point-by-point statistics (e.g. sliding window of 25ms = 12 points at 500Hz)
            # Actually, compute point-by-point stats directly (or over a small smoothing window)
            p_vals = np.ones(n_time)
            stats = np.zeros(n_time)
            for t in range(n_time):
                stat, p = ranksums(c_data[:, t], g_data[:, t])
                p_vals[t] = p
                stats[t] = stat
            
            # Smooth the p-values slightly to avoid noise (e.g., must be sig for 5 points)
            sig_mask = p_vals < 0.05
            
            # Mark significance on the bottom of the plot
            ymin, ymax = ax_time.get_ylim()
            sig_y = ymin + (ymax - ymin) * 0.05
            for t in range(n_time):
                if sig_mask[t]:
                    color = 'yellow' if stats[t] > 0 else 'cyan' # Yellow if Color > Gray, Cyan if Gray > Color
                    ax_time.plot(time_ms[t], sig_y, marker='s', color=color, markersize=3, alpha=0.7)
            
            ax_time.set_title(f"{cat_name} (Time Course)")
            ax_time.set_xlabel("Time (ms)")
            ax_time.set_ylabel("Amplitude/z-score")
            ax_time.axvline(0, color='gray', linestyle='--')
            ax_time.set_xlim([-200, 800])
            if i == 0:
                ax_time.legend(loc='upper right')
            
            # 2. Bar Plot (100-500ms avg)
            c_vals = np.mean(c_data[:, t_idx_stat], axis=1)
            g_vals = np.mean(g_data[:, t_idx_stat], axis=1)
            
            bar_c_mean, bar_c_sem = np.mean(c_vals), np.std(c_vals)/np.sqrt(len(c_vals))
            bar_g_mean, bar_g_sem = np.mean(g_vals), np.std(g_vals)/np.sqrt(len(g_vals))
            
            ax_bar.bar([1], [bar_c_mean], yerr=[bar_c_sem], color='red', alpha=0.7, capsize=5, width=0.4, label='Color')
            ax_bar.bar([2], [bar_g_mean], yerr=[bar_g_sem], color='black', alpha=0.7, capsize=5, width=0.4, label='Gray')
            
            # Individual scatter points
            ax_bar.scatter(np.random.normal(1, 0.05, len(c_vals)), c_vals, color='darkred', alpha=0.2, s=10)
            ax_bar.scatter(np.random.normal(2, 0.05, len(g_vals)), g_vals, color='gray', alpha=0.2, s=10)
            
            stat_bar, p_val_bar = ranksums(c_vals, g_vals)
            
            ax_bar.set_xticks([1, 2])
            ax_bar.set_xticklabels(['Color', 'Gray'])
            ax_bar.set_title(f"100-500ms\np={p_val_bar:.3f}")
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        out_fig = os.path.join(out_dir, f"{elec}_{feature_type}.png")
        plt.savefig(out_fig)
        plt.close(fig)
        print(f"Saved {out_fig}")

if __name__ == '__main__':
    for feat, pth in features_to_plot.items():
        plot_for_feature(feat, pth)
    print("All plotting complete!")
