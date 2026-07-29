import mne
import numpy as np
import scipy.io as sio
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from pymatreader import read_mat
from scipy.stats import ranksums

# Paths
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
analyse_dir = os.path.join(pipeline_dir, 'analyse_0617')
feature_dir = os.path.join(analyse_dir, 'feature')

set_path = os.path.join(base_dir, 'seegdata', 'test1', 'erp1.set')
final_erp_path = os.path.join(feature_dir, 'test001', 'task1_ERP_epoched.mat')
final_hg_path = os.path.join(feature_dir, 'test001', 'task1_hg_subband.mat')
out_fig_path = os.path.join(analyse_dir, 'result', 'step0_1_G11_preprocessing_comparison.png')

print("1. Loading raw eeglab data for picks G10, G11, G12...")
raw = mne.io.read_raw_eeglab(set_path, preload=True)
raw.pick_channels(['G10', 'G11', 'G12'])

print("2. Resampling raw data to 500Hz...")
raw.resample(500)

print("3. Extracting event info...")
events, event_id = mne.events_from_annotations(raw)
trig_31_name = [k for k in event_id.keys() if '31' in k][0]
trig_32_name = [k for k in event_id.keys() if '32' in k][0]
id_31 = event_id[trig_31_name]
id_32 = event_id[trig_32_name]
sel_event_id = {trig_31_name: id_31, trig_32_name: id_32}
print(f"Body Color Event: {trig_31_name} -> ID {id_31}")
print(f"Body Gray Event: {trig_32_name} -> ID {id_32}")

tmin, tmax = -0.5, 1.0

# ----------------- DATA PREPARATION -----------------
# Helper function for mean and SEM calculation
def get_mean_sem(epochs_data):
    # epochs_data: [trials, time]
    mean_val = np.mean(epochs_data, axis=0)
    sem_val = np.std(epochs_data, axis=0) / np.sqrt(epochs_data.shape[0])
    return mean_val, sem_val

# --- Row 1: Raw (No filtering, baseline corrected for DC offset)
print("\n[Row 1] Processing Raw signals...")
epochs_raw = mne.Epochs(raw, events, event_id=sel_event_id, tmin=tmin, tmax=tmax, 
                        baseline=(-0.2, 0.0), preload=True, verbose=False)
times = epochs_raw.times * 1000.0 # to ms

raw_color_trials = epochs_raw[trig_31_name].get_data(picks=['G11']).squeeze(axis=1) * 1e6
raw_gray_trials = epochs_raw[trig_32_name].get_data(picks=['G11']).squeeze(axis=1) * 1e6
raw_color_mean, raw_color_sem = get_mean_sem(raw_color_trials)
raw_gray_mean, raw_gray_sem = get_mean_sem(raw_gray_trials)

# --- Row 2: Filtered (1Hz HP + Notch filtering, baseline corrected)
print("\n[Row 2] Processing Filtered signals...")
raw_filt = raw.copy().filter(l_freq=1.0, h_freq=None, fir_design='firwin', verbose=False)
raw_filt.notch_filter(freqs=[50, 100, 150], verbose=False)
epochs_filt = mne.Epochs(raw_filt, events, event_id=sel_event_id, tmin=tmin, tmax=tmax, 
                         baseline=(-0.2, 0.0), preload=True, verbose=False)

filt_color_trials = epochs_filt[trig_31_name].get_data(picks=['G11']).squeeze(axis=1) * 1e6
filt_gray_trials = epochs_filt[trig_32_name].get_data(picks=['G11']).squeeze(axis=1) * 1e6
filt_color_mean, filt_color_sem = get_mean_sem(filt_color_trials)
filt_gray_mean, filt_gray_sem = get_mean_sem(filt_gray_trials)

# --- Row 3: Re-referenced (Local reference + HP + Notch, baseline corrected)
print("\n[Row 3] Processing Re-referenced signals...")
ch_names = raw.info['ch_names']
idx10 = ch_names.index('G10')
idx11 = ch_names.index('G11')
idx12 = ch_names.index('G12')
data_raw = raw.get_data()
ref_data = data_raw.copy()
ref_data[idx11, :] = data_raw[idx11, :] - 0.5 * (data_raw[idx10, :] + data_raw[idx12, :])

raw_ref = mne.io.RawArray(ref_data, raw.info, verbose=False)
raw_ref_filt = raw_ref.copy().filter(l_freq=1.0, h_freq=None, fir_design='firwin', verbose=False)
raw_ref_filt.notch_filter(freqs=[50, 100, 150], verbose=False)
epochs_ref = mne.Epochs(raw_ref_filt, events, event_id=sel_event_id, tmin=tmin, tmax=tmax, 
                        baseline=(-0.2, 0.0), preload=True, verbose=False)

ref_color_trials = epochs_ref[trig_31_name].get_data(picks=['G11']).squeeze(axis=1) * 1e6
ref_gray_trials = epochs_ref[trig_32_name].get_data(picks=['G11']).squeeze(axis=1) * 1e6
ref_color_mean, ref_color_sem = get_mean_sem(ref_color_trials)
ref_gray_mean, ref_gray_sem = get_mean_sem(ref_gray_trials)

# --- Row 4 Left: HG Extraction Mechanism (9 narrow-band Subband envelopes)
print("\n[Row 4 Left] Processing Subband HG extraction mechanism...")
sub_bands = [[60, 70], [70, 80], [80, 90], [90, 100], [100, 110], [110, 120], [120, 130], [130, 140], [140, 150]]
subband_color_means = []
subband_gray_means = []

# Accumulate trials across subbands for fusion
fusion_color_trials = np.zeros_like(ref_color_trials)
fusion_gray_trials = np.zeros_like(ref_gray_trials)

for b in sub_bands:
    raw_band = raw_ref.copy().filter(l_freq=b[0], h_freq=b[1], fir_design='firwin', verbose=False)
    raw_band.apply_hilbert(envelope=True, verbose=False)
    raw_band.filter(l_freq=None, h_freq=15.0, fir_design='firwin', verbose=False)
    epochs_band = mne.Epochs(raw_band, events, event_id=sel_event_id, tmin=tmin, tmax=tmax, 
                             baseline=None, preload=True, verbose=False)
    
    band_color_t = epochs_band[trig_31_name].get_data(picks=['G11']).squeeze(axis=1) * 1e6
    band_gray_t = epochs_band[trig_32_name].get_data(picks=['G11']).squeeze(axis=1) * 1e6
    
    fusion_color_trials += band_color_t
    fusion_gray_trials += band_gray_t
    
    # store subband means for mechanisms plot
    subband_color_means.append(np.mean(band_color_t, axis=0))
    subband_gray_means.append(np.mean(band_gray_t, axis=0))

# Mean across subbands
fusion_color_trials /= len(sub_bands)
fusion_gray_trials /= len(sub_bands)

fusion_color_mean, fusion_color_sem = get_mean_sem(fusion_color_trials)
fusion_gray_mean, fusion_gray_sem = get_mean_sem(fusion_gray_trials)

# --- Row 5 Left: HG Feature Extraction (same as fusion_color/gray_trials, no baseline Z-score)
print("\n[Row 5 Left] Processing HG Extraction (No baseline)...")
# Already calculated in Row 4 as fusion_color/gray_trials (which represents HG envelope from 9 bands)
hg_raw_color_mean = fusion_color_mean
hg_raw_color_sem = fusion_color_sem
hg_raw_gray_mean = fusion_gray_mean
hg_raw_gray_sem = fusion_gray_sem

# --- Row 5 Right: ERP Feature Extraction (1-30Hz bandpass, no baseline subtraction)
print("\n[Row 5 Right] Processing ERP Extraction (No baseline)...")
raw_ref_erp_filt = raw_ref.copy().filter(l_freq=1.0, h_freq=30.0, fir_design='firwin', verbose=False)
epochs_erp_raw = mne.Epochs(raw_ref_erp_filt, events, event_id=sel_event_id, tmin=tmin, tmax=tmax, 
                            baseline=None, preload=True, verbose=False)

erp_raw_color_trials = epochs_erp_raw[trig_31_name].get_data(picks=['G11']).squeeze(axis=1) * 1e6
erp_raw_gray_trials = epochs_erp_raw[trig_32_name].get_data(picks=['G11']).squeeze(axis=1) * 1e6
erp_raw_color_mean, erp_raw_color_sem = get_mean_sem(erp_raw_color_trials)
erp_raw_gray_mean, erp_raw_gray_sem = get_mean_sem(erp_raw_gray_trials)

# --- Row 6 Left: Final Z-scored High-Gamma (From task1_hg_subband.mat)
print("\n[Row 6 Left] Loading Final HG...")
hg_mat = read_mat(final_hg_path)
hg_epoch = hg_mat['epoch']
hg_data_cell = hg_epoch['data_cell']  # List of [Rep, Ch, Time]
hg_trigs = list(hg_epoch['trigger'])
idx_color_hg = hg_trigs.index('Trigger-In:31')
idx_gray_hg = hg_trigs.index('Trigger-In:32')
hg_ch_labels = list(hg_epoch['ch']['labels'])
g11_idx_final_hg = hg_ch_labels.index('G11')

final_hg_color_trials = hg_data_cell[idx_color_hg][:, g11_idx_final_hg, :]
final_hg_gray_trials = hg_data_cell[idx_gray_hg][:, g11_idx_final_hg, :]
final_hg_color_mean, final_hg_color_sem = get_mean_sem(final_hg_color_trials)
final_hg_gray_mean, final_hg_gray_sem = get_mean_sem(final_hg_gray_trials)
time_ms_final = hg_epoch['time_ms']

# --- Row 6 Right: Final Baseline-corrected ERP (From task1_ERP_epoched.mat)
print("\n[Row 6 Right] Loading Final ERP...")
erp_mat = read_mat(final_erp_path)
erp_epoch = erp_mat['epoch']
final_erp_data = erp_epoch['data']  # (8, 70, 98, 750)
erp_trigs = list(erp_epoch['trigger'])
idx_color_erp = erp_trigs.index('Trigger-In:31')
idx_gray_erp = erp_trigs.index('Trigger-In:32')
erp_ch_labels = list(erp_epoch['ch']['labels'])
g11_idx_final = erp_ch_labels.index('G11')

final_erp_color_trials = final_erp_data[idx_color_erp, :, g11_idx_final, :]
final_erp_gray_trials = final_erp_data[idx_gray_erp, :, g11_idx_final, :]
final_erp_color_mean, final_erp_color_sem = get_mean_sem(final_erp_color_trials)
final_erp_gray_mean, final_erp_gray_sem = get_mean_sem(final_erp_gray_trials)

# ----------------- HIERARCHICAL PLOTTING -----------------
print("\nPlotting hierarchical preprocessing flow (6 rows, 3 columns)...")

# Create a large figure
fig = plt.figure(figsize=(18, 26), dpi=300)

# Define 6 rows and 3 columns gridspec. Each subplot is equal-sized.
gs = fig.add_gridspec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.55, wspace=0.3)

# Unified colors
c_color = '#d32f2f'  # Body Color - Unified red
c_gray = '#212121'   # Body Gray - Unified black

# Add axes (Middle column holds joint preprocessing, left holds HG, right holds ERP)
ax_raw = fig.add_subplot(gs[0, 1])
ax_filt = fig.add_subplot(gs[1, 1])
ax_ref = fig.add_subplot(gs[2, 1])

# Row 4: Left is HG mechanism, middle is empty, right is empty (ERP void)
ax_hg_mech = fig.add_subplot(gs[3, 0])

# Row 5: Left is HG feature extraction, middle is empty, right is ERP feature extraction
ax_hg_feat = fig.add_subplot(gs[4, 0])
ax_erp_feat = fig.add_subplot(gs[4, 2])

# Row 6: Left is HG Final, middle is empty, right is ERP Final
ax_hg_final = fig.add_subplot(gs[5, 0])
ax_erp_final = fig.add_subplot(gs[5, 2])

# Unified spines & grid removal helper
def customize_axes(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#757575')
        spine.set_linewidth(1.0)

# Generalized plotting function that computes point-wise Wilcoxon ranksums
def plot_data_with_stats(ax, x, trials_color, trials_gray, color_line, gray_line, title, ylabel, xlabel='Time (ms)'):
    customize_axes(ax)
    
    # Calculate statistics
    mean_color, sem_color = get_mean_sem(trials_color)
    mean_gray, sem_gray = get_mean_sem(trials_gray)
    
    # Plot Mean & SEM
    ax.plot(x, mean_color, color=color_line, lw=2, label='Body Color')
    ax.fill_between(x, mean_color - sem_color, mean_color + sem_color, color=color_line, alpha=0.15)
    
    ax.plot(x, mean_gray, color=gray_line, lw=2, label='Body Gray')
    ax.fill_between(x, mean_gray - sem_gray, mean_gray + sem_gray, color=gray_line, alpha=0.15)
    
    ax.axvline(0, color='#9E9E9E', linestyle='--', alpha=0.6)
    
    # Point-by-point Wilcoxon ranksums
    p_vals = np.zeros(len(x))
    for t_idx in range(len(x)):
        stat, p = ranksums(trials_color[:, t_idx], trials_gray[:, t_idx])
        p_vals[t_idx] = p
    
    sig_pts = p_vals < 0.05
    
    # Dynamically draw significance bar at the bottom 4% of y-range
    y_limits = ax.get_ylim()
    y_range = y_limits[1] - y_limits[0]
    y_sig = y_limits[0] + 0.04 * y_range
    
    # Map nan for non-significant points to break the line drawing
    sig_times = np.where(sig_pts, x, np.nan)
    sig_y = np.full_like(sig_times, y_sig)
    
    ax.plot(x, sig_y, color='#d32f2f', linewidth=3.5, solid_capstyle='round', label='Sig. Diff (p < 0.05)')
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_xlim([-500, 1000])
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)

# --- Row 1: Raw
plot_data_with_stats(ax_raw, times, raw_color_trials, raw_gray_trials, 
                     c_color, c_gray, "Step 1: Raw EEG (Baseline Corrected)", "Amplitude (μV)")

# --- Row 2: Filtered
plot_data_with_stats(ax_filt, times, filt_color_trials, filt_gray_trials, 
                     c_color, c_gray, "Step 2: Filtered EEG (1Hz HP + Notch)", "Amplitude (μV)")

# --- Row 3: Re-referenced
plot_data_with_stats(ax_ref, times, ref_color_trials, ref_gray_trials, 
                     c_color, c_gray, "Step 3: Re-referenced EEG (Laplace/Bipolar)", "Amplitude (μV)")

# --- Row 4 Left: HG Extraction Mechanism
customize_axes(ax_hg_mech)
# Plot 9 narrow-band envelopes with Oranges colormap gradients
grad_colors = [plt.cm.Oranges(i) for i in np.linspace(0.3, 0.7, 9)]
for idx, (sub_color, sub_gray) in enumerate(zip(subband_color_means, subband_gray_means)):
    ax_hg_mech.plot(times, sub_color, color=grad_colors[idx], linestyle='dotted', alpha=0.7, lw=1.0)

# Overlay final fusion mean lines (equal to fusion_color/gray_mean)
ax_hg_mech.plot(times, fusion_color_mean, color=c_color, lw=2.2, label='Fused Body Color')
ax_hg_mech.plot(times, fusion_gray_mean, color=c_gray, lw=2.2, label='Fused Body Gray')
ax_hg_mech.axvline(0, color='#9E9E9E', linestyle='--', alpha=0.6)
ax_hg_mech.set_title("Step 4a: Subband HG Mechanism (9 Filter Banks)", fontsize=11, fontweight='bold', pad=10)
ax_hg_mech.set_ylabel("Amplitude (μV)", fontsize=9.5)
ax_hg_mech.set_xlabel("Time (ms)", fontsize=9.5)
ax_hg_mech.set_xlim([-500, 1000])
ax_hg_mech.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)

# --- Row 5 Left: HG Feature Extraction (No Z-score)
plot_data_with_stats(ax_hg_feat, times, fusion_color_trials, fusion_gray_trials, 
                     c_color, c_gray, "Step 5a: HG Power Envelope (No Z-score)", "Amplitude (μV)")

# --- Row 5 Right: ERP Feature Extraction (1-30Hz Filtered, no baseline)
plot_data_with_stats(ax_erp_feat, times, erp_raw_color_trials, erp_raw_gray_trials, 
                     c_color, c_gray, "Step 5b: ERP Signal (1-30Hz, No Baseline)", "Amplitude (μV)")

# --- Row 6 Left: Final Z-scored HG
plot_data_with_stats(ax_hg_final, time_ms_final, final_hg_color_trials, final_hg_gray_trials, 
                     c_color, c_gray, "Step 6a: Final Z-scored HG", "HG Power (Z-score)")

# --- Row 6 Right: Final ERP
plot_data_with_stats(ax_erp_final, time_ms_final, final_erp_color_trials, final_erp_gray_trials, 
                     c_color, c_gray, "Step 6b: Final ERP (Baseline Corrected)", "Amplitude (μV)")

# Add Super Title
plt.suptitle("Hierarchical Preprocessing Pipeline Comparison - Electrode G11", fontsize=16, fontweight='bold', y=0.99)

# ----------------- LOGICAL CONNECTIONS WITH ARROWS -----------------
arrow_style = dict(arrowstyle="-|>", color="#424242", lw=2.5, mutation_scale=20)

# Row 1 -> Row 2
con1 = ConnectionPatch(xyA=(0.5, 0.0), xyB=(0.5, 1.0), coordsA="axes fraction", coordsB="axes fraction",
                      axesA=ax_raw, axesB=ax_filt, **arrow_style)
fig.add_artist(con1)

# Row 2 -> Row 3
con2 = ConnectionPatch(xyA=(0.5, 0.0), xyB=(0.5, 1.0), coordsA="axes fraction", coordsB="axes fraction",
                      axesA=ax_filt, axesB=ax_ref, **arrow_style)
fig.add_artist(con2)

# Row 3 -> Row 4 Left (HG Mechanism)
# Left bottom of Row 3 -> top center of HG Mech (Row 4 Col 0)
con3_l = ConnectionPatch(xyA=(0.0, 0.0), xyB=(0.5, 1.0), coordsA="axes fraction", coordsB="axes fraction",
                        axesA=ax_ref, axesB=ax_hg_mech, **arrow_style)
fig.add_artist(con3_l)

# Row 3 -> Row 5 Right (ERP Feature Extraction)
# Right bottom of Row 3 -> top center of ERP Feat (Row 5 Col 2)
con3_r = ConnectionPatch(xyA=(1.0, 0.0), xyB=(0.5, 1.0), coordsA="axes fraction", coordsB="axes fraction",
                        axesA=ax_ref, axesB=ax_erp_feat, **arrow_style)
fig.add_artist(con3_r)

# HG Mechanism -> HG Extraction (Row 4 Left -> Row 5 Left)
con4_l = ConnectionPatch(xyA=(0.5, 0.0), xyB=(0.5, 1.0), coordsA="axes fraction", coordsB="axes fraction",
                        axesA=ax_hg_mech, axesB=ax_hg_feat, **arrow_style)
fig.add_artist(con4_l)

# HG Extraction -> HG Final (Row 5 Left -> Row 6 Left)
con5_l = ConnectionPatch(xyA=(0.5, 0.0), xyB=(0.5, 1.0), coordsA="axes fraction", coordsB="axes fraction",
                        axesA=ax_hg_feat, axesB=ax_hg_final, **arrow_style)
fig.add_artist(con5_l)

# ERP Extraction -> ERP Final (Row 5 Right -> Row 6 Right)
con5_r = ConnectionPatch(xyA=(0.5, 0.0), xyB=(0.5, 1.0), coordsA="axes fraction", coordsB="axes fraction",
                        axesA=ax_erp_feat, axesB=ax_erp_final, **arrow_style)
fig.add_artist(con5_r)

# Save figure
os.makedirs(os.path.dirname(out_fig_path), exist_ok=True)
plt.savefig(out_fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"New hierarchical preprocess visualization figure successfully saved to: {out_fig_path}")
