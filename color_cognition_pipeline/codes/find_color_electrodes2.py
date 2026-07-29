import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'

print("Loading data...")
mat = sio.loadmat(f"{base_dir}/processed_data/test001/task1_TFA_epoched.mat", squeeze_me=True)
data = mat['epoch']['data'].item() # shape (8, 70, 98, 750)
ch_arr = mat['epoch']['ch'].item()
ch_names = ch_arr['labels']
time_ms = mat['epoch']['time_ms'].item()

# Filter design
fs = 500
b, a = butter(4, [110, 140], btype='bandpass', fs=fs)

n_cond, n_rep, n_ch, n_time = data.shape
base_idx = np.where((time_ms >= -200) & (time_ms <= 0))[0]
t_idx = np.where((time_ms >= 100) & (time_ms <= 500))[0]

print("Extracting high gamma features and square-root transforming...")
sig_sqrt = np.zeros_like(data)
for c in range(n_cond):
    for r in range(n_rep):
        for ch in range(n_ch):
            sig = data[c, r, ch, :]
            if np.any(np.isnan(sig)):
                sig_sqrt[c, r, ch, :] = np.nan
                continue
            sig_filt = filtfilt(b, a, sig)
            sig_amp = np.abs(hilbert(sig_filt))
            sig_sqrt[c, r, ch, :] = np.sqrt(sig_amp)

print("Applying global baseline normalization per channel...")
hg_zscored = np.zeros_like(sig_sqrt)
for ch in range(n_ch):
    # Collect all baseline points across all conditions and reps for this channel
    ch_baseline_data = sig_sqrt[:, :, ch, base_idx]
    base_mean = np.nanmean(ch_baseline_data)
    base_std = np.nanstd(ch_baseline_data)
    if base_std == 0 or np.isnan(base_std):
        base_std = 1.0
    hg_zscored[:, :, ch, :] = (sig_sqrt[:, :, ch, :] - base_mean) / base_std

print("Performing statistics...")
hg_mean = np.nanmean(hg_zscored[:, :, :, t_idx], axis=-1) # average 100-500ms

color_data = hg_mean[[0, 2, 4, 6], :, :].reshape(-1, n_ch)
gray_data = hg_mean[[1, 3, 5, 7], :, :].reshape(-1, n_ch)

p_values = np.zeros(n_ch)
t_stats = np.zeros(n_ch)
for ch in range(n_ch):
    c_vals = color_data[:, ch]
    c_vals = c_vals[~np.isnan(c_vals)]
    g_vals = gray_data[:, ch]
    g_vals = g_vals[~np.isnan(g_vals)]
    
    if len(c_vals) < 5 or len(g_vals) < 5:
        t_stats[ch] = 0
        p_values[ch] = 1.0
        continue
        
    stat, p = ranksums(c_vals, g_vals)
    t_stats[ch] = stat
    p_values[ch] = p

alpha = 0.05
bonf_alpha = alpha / n_ch

top_idx = np.argsort(p_values)
print(f"\nTop 10 color selective channels (uncorrected p-values):")
count = 0
for idx in top_idx:
    if t_stats[idx] > 0: # color > gray
        print(f"Channel {ch_names[idx]:<5} (idx {idx:02d}): statistic={t_stats[idx]:.3f}, p_value={p_values[idx]:.3e}")
        count += 1
    if count >= 10:
        break

print(f"\nSignificant color selective channels (Bonferroni p < {bonf_alpha:.5e}):")
bonf_sig_ch_names = []
for idx in top_idx:
    if p_values[idx] < bonf_alpha and t_stats[idx] > 0:
        bonf_sig_ch_names.append(ch_names[idx])
        print(f"Channel {ch_names[idx]} (idx {idx}): statistic={t_stats[idx]:.3f}, p_value={p_values[idx]:.3e}")

# Save visualization
plt.figure(figsize=(12, 6))
colors = ['red' if (p < bonf_alpha and t > 0) else ('orange' if p < bonf_alpha else 'grey') for p, t in zip(p_values, t_stats)]
plt.scatter(range(n_ch), -np.log10(p_values), c=colors, alpha=0.7)
plt.axhline(-np.log10(bonf_alpha), color='k', linestyle='--', label=f'Bonferroni alpha={bonf_alpha:.2e}')

for idx in np.where(p_values < bonf_alpha)[0]:
    if t_stats[idx] > 0:
        plt.text(idx, -np.log10(p_values[idx]) + 0.1, ch_names[idx], fontsize=9)

plt.xlabel('Channel Index')
plt.ylabel('-log10(p-value)')
plt.title('Broadband Gamma (110-140Hz): Color vs Grayscale (100-500ms)')
plt.legend()
plt.tight_layout()
out_png = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/color_selective_channels_test001_v2.png'
plt.savefig(out_png)

out_txt = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/color_selective_channels_v2.txt'
with open(out_txt, 'w') as f:
    f.write(f"Significant Color Selective Channels (Bonferroni p < {bonf_alpha:.5e}):\n")
    for ch in bonf_sig_ch_names:
        f.write(f"{ch}\n")

print(f"\nProcess completed and figure saved to {out_png}")
