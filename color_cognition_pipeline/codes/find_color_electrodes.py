import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'

print("Loading data...")
mat = sio.loadmat(f"{base_dir}/processed_data/test001/task1_TFA_epoched.mat", squeeze_me=True, struct_as_record=False)
data = mat['epoch'].data # shape (8, 70, 98, 750)
ch_names = mat['epoch'].ch # should be array of strings, length 98
time_ms = mat['epoch'].time_ms # should be array of length 750

# Ensure ch_names is a list of strings
if isinstance(ch_names, np.ndarray) and ch_names.dtype.kind in {'U', 'S', 'O'}:
    ch_names = [str(c) for c in ch_names]

# Filter design
fs = 500
b, a = butter(4, [110, 140], btype='bandpass', fs=fs)

n_cond, n_rep, n_ch, n_time = data.shape

print("Extracting high gamma features...")
hg_data = np.zeros_like(data)
for c in range(n_cond):
    for r in range(n_rep):
        for ch in range(n_ch):
            sig = data[c, r, ch, :]
            if np.any(np.isnan(sig)):
                hg_data[c, r, ch, :] = np.nan
                continue
            # 1. Bandpass filter 110-140Hz
            sig_filt = filtfilt(b, a, sig)
            # 2. Hilbert transform -> amplitude
            sig_amp = np.abs(hilbert(sig_filt))
            # 3. Square-root transform
            sig_sqrt = np.sqrt(sig_amp)
            # 4. Baseline normalization (z-score against -200 to 0 ms)
            base_idx = np.where((time_ms >= -200) & (time_ms <= 0))[0]
            base_mean = np.nanmean(sig_sqrt[base_idx])
            base_std = np.nanstd(sig_sqrt[base_idx])
            if base_std == 0 or np.isnan(base_std):
                base_std = 1
            sig_z = (sig_sqrt - base_mean) / base_std
            hg_data[c, r, ch, :] = sig_z

print("Feature extraction done. Performing statistics...")
# Time average 100 to 500 ms
t_idx = np.where((time_ms >= 100) & (time_ms <= 500))[0]
hg_mean = np.nanmean(hg_data[:, :, :, t_idx], axis=-1) # shape (8, 70, 98)

# Color conditions: 0, 2, 4, 6
# Gray conditions: 1, 3, 5, 7
color_data = hg_mean[[0, 2, 4, 6], :, :].reshape(-1, n_ch) # shape (280, 98)
gray_data = hg_mean[[1, 3, 5, 7], :, :].reshape(-1, n_ch) # shape (280, 98)

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

# Bonferroni correction
alpha = 0.05
bonf_alpha = alpha / n_ch
sig_ch_idx = np.where(p_values < bonf_alpha)[0]

# Uncorrected significance and ranking
p_values[np.isnan(p_values)] = 1.0 # Handle NaN p-values just in case
sig_ch_idx_uncorr = np.where(p_values < 0.05)[0]

print(f"\nTop 10 color selective channels (uncorrected p < 0.05):")
top_idx = np.argsort(p_values)
sig_ch_names = []
for idx in top_idx:
    if p_values[idx] < 0.05 and t_stats[idx] > 0:
        ch_name = ch_names[idx]
        sig_ch_names.append(ch_name)
        print(f"Channel {ch_name} (idx {idx}): statistic={t_stats[idx]:.3f}, p_value={p_values[idx]:.3e}")

print(f"\nSignificant color selective channels (Bonferroni p < {bonf_alpha:.5e}):")
bonf_sig_ch_names = []
for idx in top_idx:
    if p_values[idx] < bonf_alpha and t_stats[idx] > 0:
        ch_name = ch_names[idx]
        bonf_sig_ch_names.append(ch_name)
        print(f"Channel {ch_name} (idx {idx}): statistic={t_stats[idx]:.3f}, p_value={p_values[idx]:.3e}")

# Save visualization
plt.figure(figsize=(12, 6))
# Colors: red if significant AND color > gray, else blue/grey
colors = ['red' if (p < bonf_alpha and t > 0) else ('orange' if p < bonf_alpha else 'grey') for p, t in zip(p_values, t_stats)]
plt.scatter(range(n_ch), -np.log10(p_values), c=colors, alpha=0.7)
plt.axhline(-np.log10(bonf_alpha), color='k', linestyle='--', label=f'Bonferroni alpha={bonf_alpha:.2e}')

for idx in sig_ch_idx:
    if t_stats[idx] > 0:
        plt.text(idx, -np.log10(p_values[idx]) + 0.1, ch_names[idx], fontsize=9)

plt.xlabel('Channel Index')
plt.ylabel('-log10(p-value)')
plt.title('Broadband Gamma Selection: Color vs Grayscale (100-500ms)')
plt.legend()
plt.tight_layout()
out_png = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/color_selective_channels_test001.png'
plt.savefig(out_png)

# Also save a text file with the findings
out_txt = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/color_selective_channels.txt'
with open(out_txt, 'w') as f:
    f.write(f"Significant Color Selective Channels (Bonferroni p < {bonf_alpha:.5e}):\n")
    for ch in bonf_sig_ch_names:
        f.write(f"{ch}\n")
    f.write(f"\nTop Color Selective Channels (uncorrected p < 0.05):\n")
    for ch in sig_ch_names:
        f.write(f"{ch}\n")

print(f"\nProcess completed and figure saved to {out_png}")
