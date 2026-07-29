import mne
import numpy as np
import scipy.io as sio
from scipy.stats import ranksums
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

# 配置基础路径
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
analyse_dir = os.path.join(pipeline_dir, 'analyse_0617')
result_dir = os.path.join(analyse_dir, 'result', 'subband_70_200hz')
set_path = os.path.join(base_dir, 'seegdata', 'test1', 'erp1.set')

# 确保输出目录存在
os.makedirs(result_dir, exist_ok=True)

# 窄频带定义（每 10Hz 一个子带，覆盖 70-200Hz）
sub_bands = [
    [70, 80], [80, 90], [90, 100], [100, 110], [110, 120],
    [120, 130], [130, 140], [140, 150], [150, 160], [160, 170],
    [170, 180], [180, 190], [190, 200]
]

# 四种大类的 trigger 映射，用于 MNE Epochs 的提取
sel_event_id = {
    'Trigger-In:11': 1, # Face Color
    'Trigger-In:12': 2, # Face Gray
    'Trigger-In:21': 3, # Object Color
    'Trigger-In:22': 4, # Object Gray
    'Trigger-In:31': 5, # Body Color
    'Trigger-In:32': 6, # Body Gray
    'Trigger-In:41': 7, # Place Color
    'Trigger-In:42': 8  # Place Gray
}

categories = ['Face', 'Object', 'Body', 'Place', 'Merged_All']
cond_pairs = {
    'Face': ('Trigger-In:11', 'Trigger-In:12'),
    'Object': ('Trigger-In:21', 'Trigger-In:22'),
    'Body': ('Trigger-In:31', 'Trigger-In:32'),
    'Place': ('Trigger-In:41', 'Trigger-In:42')
}

groups_config = {
    'D': {
        'target_chans': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6'],
        'plot_chans': ['D1', 'D2', 'D3', 'D4', 'D5']
    },
    'G': {
        'target_chans': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
        'plot_chans': ['G1', 'G2', 'G3', 'G4', 'G5']
    }
}

def plot_single_electrode_subband(subject, elec, time_ms, c_data_dict, g_data_dict):
    """
    绘制指定电极的 5x2 时程与条形散点图
    """
    fig, axes = plt.subplots(5, 2, figsize=(12, 20), gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(f"Subject: {subject} | Electrode: {elec} - Subband 70-200Hz", fontsize=16, fontweight='bold', y=0.98)
    
    idx_100_400 = np.where((time_ms >= 100) & (time_ms <= 400))[0]
    
    for i, cat_name in enumerate(categories):
        ax_time = axes[i, 0]
        ax_bar = axes[i, 1]
        
        # 数据提取
        if cat_name == 'Merged_All':
            c_data = np.concatenate([c_data_dict[c] for c in ['Face', 'Object', 'Body', 'Place']], axis=0)
            g_data = np.concatenate([g_data_dict[c] for c in ['Face', 'Object', 'Body', 'Place']], axis=0)
        else:
            c_data = c_data_dict[cat_name]
            g_data = g_data_dict[cat_name]
            
        # 过滤 NaN 所在的 trials
        c_data = c_data[~np.isnan(c_data).any(axis=1)]
        g_data = g_data[~np.isnan(g_data).any(axis=1)]
        
        # 计算 Mean & SEM
        c_mean = np.mean(c_data, axis=0)
        c_sem = np.std(c_data, axis=0) / np.sqrt(c_data.shape[0]) if c_data.shape[0] > 0 else np.zeros_like(c_mean)
        g_mean = np.mean(g_data, axis=0)
        g_sem = np.std(g_data, axis=0) / np.sqrt(g_data.shape[0]) if g_data.shape[0] > 0 else np.zeros_like(g_mean)
        
        # 1. 左侧时程图
        ax_time.plot(time_ms, c_mean, color='#d32f2f', lw=2.2, label='Color')
        ax_time.fill_between(time_ms, c_mean - c_sem, c_mean + c_sem, color='#d32f2f', alpha=0.15)
        
        ax_time.plot(time_ms, g_mean, color='#212121', lw=2.2, label='Gray')
        ax_time.fill_between(time_ms, g_mean - g_sem, g_mean + g_sem, color='#212121', alpha=0.15)
        
        ax_time.axvline(0, color='#9E9E9E', linestyle='--', alpha=0.6)
        
        # 点对点显著性标记
        ymin, ymax = ax_time.get_ylim()
        if ymin == ymax:
            ymin, ymax = ymin - 1.0, ymax + 1.0
        sig_y = ymin + (ymax - ymin) * 0.05
        
        for t_idx in range(len(time_ms)):
            stat, p = ranksums(c_data[:, t_idx], g_data[:, t_idx])
            if p < 0.05:
                color = 'yellow' if stat > 0 else 'cyan'
                ax_time.plot(time_ms[t_idx], sig_y, marker='s', color=color, markersize=3, alpha=0.7)
                
        ax_time.set_title(f"{cat_name} (Time Course)", fontsize=11, fontweight='bold')
        ax_time.set_xlabel("Time (ms)", fontsize=9.5)
        ax_time.set_ylabel("Amplitude (z-score)", fontsize=9.5)
        ax_time.set_xlim([-200, 800])
        ax_time.grid(False)
        for spine in ax_time.spines.values():
            spine.set_visible(True)
            spine.set_color('#757575')
            
        if i == 0:
            ax_time.legend(loc='upper right', frameon=True, fontsize=8)
            
        # 2. 右侧 100-400ms 条形散点图
        c_vals = np.mean(c_data[:, idx_100_400], axis=1)
        g_vals = np.mean(g_data[:, idx_100_400], axis=1)
        
        bar_c_mean = np.mean(c_vals)
        bar_c_sem = np.std(c_vals) / np.sqrt(len(c_vals)) if len(c_vals) > 0 else 0.0
        bar_g_mean = np.mean(g_vals)
        bar_g_sem = np.std(g_vals) / np.sqrt(len(g_vals)) if len(g_vals) > 0 else 0.0
        
        ax_bar.bar([1], [bar_c_mean], yerr=[bar_c_sem], color='#d32f2f', alpha=0.7, capsize=5, width=0.4, error_kw={'elinewidth':1.5, 'capthick':1.5})
        ax_bar.bar([2], [bar_g_mean], yerr=[bar_g_sem], color='#212121', alpha=0.7, capsize=5, width=0.4, error_kw={'elinewidth':1.5, 'capthick':1.5})
        
        # 散点
        ax_bar.scatter(np.random.normal(1, 0.05, len(c_vals)), c_vals, color='darkred', alpha=0.2, s=8)
        ax_bar.scatter(np.random.normal(2, 0.05, len(g_vals)), g_vals, color='gray', alpha=0.2, s=8)
        
        stat_bar, p_val_bar = ranksums(c_vals, g_vals)
        
        ax_bar.set_xticks([1, 2])
        ax_bar.set_xticklabels(['Color', 'Gray'], fontsize=8.5)
        ax_bar.set_title(f"100-400ms\np={p_val_bar:.3f}", fontsize=9.5)
        ax_bar.grid(False)
        for spine in ax_bar.spines.values():
            spine.set_visible(True)
            spine.set_color('#757575')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = os.path.join(result_dir, f"{elec}_subband_70-200Hz.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图表成功保存至: {out_path}")


# 1. 统一加载整个 EEG 并进行重采样
print("1. 加载 raw eeglab 数据...")
raw_full = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

print("2. 重采样数据到 500Hz...")
raw_full.resample(500, verbose=False)

print("3. 提取事件信息...")
events, event_id = mne.events_from_annotations(raw_full, verbose=False)
events_filtered = np.array([ev for ev in events if ev[2] in sel_event_id.values()])

# 2. 分别处理 D 电极组和 G 电极组
for group_name, cfg in groups_config.items():
    print(f"\n=================== 开始处理 {group_name} 电极组 (70-200Hz Subband) ===================")
    target_chans = cfg['target_chans']
    plot_chans = cfg['plot_chans']
    
    # 筛选当前组电极
    raw_group = raw_full.copy().pick_channels(target_chans)
    raw_data = raw_group.get_data() # (n_channels, n_times)
    ch_names_raw = raw_group.info['ch_names']
    
    # 按正确顺序排列
    data_ordered = np.zeros((len(target_chans), raw_data.shape[1]))
    for idx, ch in enumerate(target_chans):
        ch_idx = ch_names_raw.index(ch)
        data_ordered[idx, :] = raw_data[ch_idx, :]
        
    # 重参考计算：
    ref_data = np.zeros((len(plot_chans), raw_data.shape[1]))
    # 第一触点双极
    ref_data[0, :] = data_ordered[0, :] - data_ordered[1, :]
    # 其余 Laplacian
    for i in range(1, len(plot_chans)):
        ref_data[i, :] = data_ordered[i, :] - 0.5 * (data_ordered[i-1, :] + data_ordered[i+1, :])
        
    info_ref = mne.create_info(ch_names=plot_chans, sfreq=raw_group.info['sfreq'], ch_types='seeg')
    raw_ref = mne.io.RawArray(ref_data, info_ref, verbose=False)
    
    # 3. 窄频带 Hilbert 变换包络的融合
    fusion_data = np.zeros_like(ref_data)
    
    for b in sub_bands:
        print(f"  计算窄频带包络: {b[0]} - {b[1]} Hz...")
        raw_band = raw_ref.copy()
        raw_band.filter(l_freq=b[0], h_freq=b[1], fir_design='firwin', verbose=False)
        raw_band.apply_hilbert(envelope=True, verbose=False)
        raw_band.filter(l_freq=None, h_freq=15.0, fir_design='firwin', verbose=False) # 15Hz 低通平滑
        fusion_data += raw_band.get_data()
        
    fusion_data /= len(sub_bands)
    
    # 构建 MNE Raw 格式以分 Epoch
    info_fusion = mne.create_info(ch_names=plot_chans, sfreq=raw_ref.info['sfreq'], ch_types='seeg')
    raw_fusion = mne.io.RawArray(fusion_data, info_fusion, verbose=False)
    
    # 4. Epoch 提取（无基线减除）
    print("  提取 Epoch...")
    epochs = mne.Epochs(raw_fusion, events_filtered, event_id=sel_event_id, 
                        tmin=-0.5, tmax=1.0, baseline=None, 
                        preload=True, verbose=False)
    
    epochs_data = epochs.get_data() * 1e6 # 转换为微伏
    times = epochs.times
    idx_baseline = np.where((times >= -0.2) & (times <= 0.0))[0]
    
    # 5. 执行 trial-by-trial Z-score 归一化
    print("  执行 trial-by-trial Z-score 基线归一化...")
    for tr in range(epochs_data.shape[0]):
        for ch in range(epochs_data.shape[1]):
            base_vals = epochs_data[tr, ch, idx_baseline]
            mean_base = np.mean(base_vals)
            std_base = np.std(base_vals)
            if std_base == 0.0:
                std_base = 1.0
            epochs_data[tr, ch, :] = (epochs_data[tr, ch, :] - mean_base) / std_base

    # 6. 对当前组的所有 plot 电极绘制并保存图片
    event_codes = epochs.events[:, 2]
    trig_to_code = sel_event_id
    
    for elec in plot_chans:
        print(f"  绘制电极 {elec}...")
        ch_idx = plot_chans.index(elec)
        
        c_data_dict = {}
        g_data_dict = {}
        
        for cat_name, (trig_c, trig_g) in cond_pairs.items():
            code_c = trig_to_code[trig_c]
            code_g = trig_to_code[trig_g]
            
            idx_c = np.where(event_codes == code_c)[0]
            idx_g = np.where(event_codes == code_g)[0]
            
            c_data_dict[cat_name] = epochs_data[idx_c, ch_idx, :]
            g_data_dict[cat_name] = epochs_data[idx_g, ch_idx, :]
            
        plot_single_electrode_subband(
            subject='test001',
            elec=elec,
            time_ms=times * 1000.0,
            c_data_dict=c_data_dict,
            g_data_dict=g_data_dict
        )

print("\n所有 70-200Hz Subband 特征提取和绘图完成！")
