import os
import sys
import time
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

# ---------------------------------------------------------
# Config & Paths
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBJECT = 'test001'
FEATURE_TYPE = 'lowgamma' # 'lowgamma' or 'erp'
ROI_NAME = 'Color_patch' # or 'Fusiform_R', 'Temporal_Inf_R'

FEATURE_MAT = os.path.join(PROJECT_ROOT, 'feature', FEATURE_TYPE, SUBJECT, f'{ROI_NAME}.mat')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'result', 'tgm_benchmark', SUBJECT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Time window parameters
FS = 500
T_START = -100
T_END = 1000
N_POINTS = 550
TIMES = np.linspace(T_START, T_END, N_POINTS)

# Performance Benchmark settings
DECODING_STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 5 # Default 5 (10ms step). Pass 1 for max resolution (2ms step, 550x550 grid)
N_REPEATS = 5     # 5 bootstrap iterations for AUC estimation
N_JOBS = -1       # Use all available CPU cores

def baseline_zscore(X, baseline_end_idx):
    """Z-score normalize across time based on baseline mean and std."""
    base_mean = np.mean(X[:, :, :baseline_end_idx], axis=2, keepdims=True)
    base_std = np.std(X[:, :, :baseline_end_idx], axis=2, keepdims=True) + 1e-8
    return (X - base_mean) / base_std

def extract_cross_data(mat_path):
    """Extract Task 3 (Red vs Green) & Task 2 (Red memory vs Green memory)"""
    mat = sio.loadmat(mat_path)
    
    # Resolve feature key
    train_key = [k for k in mat.keys() if 'task3' in k.lower()][0]
    test_key = [k for k in mat.keys() if 'task2' in k.lower()][0]
    
    train_data = mat[train_key] # [Cond, Rep, Ch, Time]
    test_data = mat[test_key]
    
    # Task 3: Cond 0 (Red, 51) vs Cond 3 (Green, 54)
    X_train_0 = train_data[0, :, :, :] # Red
    X_train_1 = train_data[3, :, :, :] # Green
    X_train = np.concatenate([X_train_0, X_train_1], axis=0)
    y_train = np.concatenate([np.zeros(X_train_0.shape[0]), np.ones(X_train_1.shape[0])])
    
    # Task 2: Gray Red Memory (Cond 2, 5) vs Gray Green Memory (Cond 8, 11) if available or standard conditions
    # Check shape: Cond dimension
    n_cond_t2 = test_data.shape[0]
    if n_cond_t2 >= 12:
        class0_conds = [2, 5]
        class1_conds = [8, 11]
    else:
        # standard 8 condition layout: 0,1,2,3 color; 4,5,6,7 gray
        class0_conds = [4] # strawberry gray (red)
        class1_conds = [5] # kiwi gray (green)
        
    X_test_0 = np.concatenate([test_data[idx, :, :, :] for idx in class0_conds if idx < n_cond_t2], axis=0)
    X_test_1 = np.concatenate([test_data[idx, :, :, :] for idx in class1_conds if idx < n_cond_t2], axis=0)
    
    X_test = np.concatenate([X_test_0, X_test_1], axis=0)
    y_test = np.concatenate([np.zeros(X_test_0.shape[0]), np.ones(X_test_1.shape[0])])
    
    return X_train, y_train, X_test, y_test

def train_t_row(t_idx, X_train, y_train, X_test, y_test, time_indices):
    """Worker function to train at t_train=t_idx and test across all t_test in time_indices."""
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    clf.fit(X_train[:, :, t_idx], y_train)
    
    n_test_time = len(time_indices)
    row_auc = np.zeros(n_test_time)
    
    for j, t_test in enumerate(time_indices):
        y_prob = clf.predict_proba(X_test[:, :, t_test])[:, 1]
        if len(np.unique(y_test)) > 1:
            row_auc[j] = roc_auc_score(y_test, y_prob)
        else:
            row_auc[j] = 0.5
            
    return row_auc

def main():
    print("=" * 70)
    print(f"[START] Starting High-Performance Temporal Generalization Decoding Benchmark")
    print(f"Subject: {SUBJECT} | ROI: {ROI_NAME} | Feature: {FEATURE_TYPE}")
    print(f"Data Path: {FEATURE_MAT}")
    print("=" * 70)
    
    if not os.path.exists(FEATURE_MAT):
        print(f"[ERROR] Feature mat file not found at {FEATURE_MAT}")
        sys.exit(1)
        
    # 1. Load Data
    t0_load = time.perf_counter()
    X_train, y_train, X_test, y_test = extract_cross_data(FEATURE_MAT)
    t_load = time.perf_counter() - t0_load
    
    # Baseline Z-score
    baseline_end = np.searchsorted(TIMES, 0) # t=0ms index
    X_train = baseline_zscore(X_train, baseline_end)
    X_test = baseline_zscore(X_test, baseline_end)
    
    n_ch = X_train.shape[1]
    n_time = X_train.shape[2]
    time_indices = np.arange(0, n_time, DECODING_STEP)
    n_steps = len(time_indices)
    plot_times = TIMES[time_indices]
    
    print(f"[DATASET] Loaded in {t_load:.3f}s:")
    print(f"   - Task 3 (Train): Shape {X_train.shape} (N_samples={len(y_train)}, Channels={n_ch})")
    print(f"   - Task 2 (Test) : Shape {X_test.shape} (N_samples={len(y_test)}, Channels={n_ch})")
    print(f"   - TGM Resolution: {n_steps} x {n_steps} grid ({DECODING_STEP * 2} ms / step)")
    print(f"   - Total Classifiers Fitted & Evaluated: {n_steps} x {n_steps} = {n_steps * n_steps:,}")
    print("-" * 70)
    
    # 2. Run Parallel Benchmark
    print("[COMPUTE] Executing Parallel 2D TGM Computation...")
    t0_calc = time.perf_counter()
    
    results = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(train_t_row)(t_idx, X_train, y_train, X_test, y_test, time_indices)
        for t_idx in time_indices
    )
    
    t_calc = time.perf_counter() - t0_calc
    tgm_matrix = np.array(results) # Shape: [n_steps, n_steps]
    
    fits_per_sec = (n_steps * n_steps) / t_calc
    
    print("-" * 70)
    print(f"[DONE] TGM Computation Completed!")
    print(f"Total Compute Time : {t_calc:.3f} seconds ({t_calc/60:.2f} minutes)")
    print(f"Performance Throughput: {fits_per_sec:.1f} evaluations/sec")
    print(f"Peak AUC Value      : {np.max(tgm_matrix):.4f}")
    print(f"Mean AUC Value      : {np.mean(tgm_matrix):.4f}")
    print("=" * 70)
    
    # 3. Save Matrix & Plot Heatmap
    npz_path = os.path.join(OUTPUT_DIR, f'tgm_{SUBJECT}_{ROI_NAME}_{FEATURE_TYPE}.npz')
    np.savez_compressed(
        npz_path,
        tgm=tgm_matrix,
        times=plot_times,
        subject=SUBJECT,
        roi=ROI_NAME,
        feature=FEATURE_TYPE,
        compute_time_sec=t_calc,
    )
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    im = ax.imshow(
        tgm_matrix,
        origin='lower',
        extent=[plot_times[0], plot_times[-1], plot_times[0], plot_times[-1]],
        cmap='RdBu_r',
        vmin=0.35,
        vmax=0.65,
        aspect='equal'
    )
    plt.colorbar(im, ax=ax, label='ROC-AUC')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.plot([plot_times[0], plot_times[-1]], [plot_times[0], plot_times[-1]], 'k--', alpha=0.5, linewidth=1, label='Diagonal (t_train = t_test)')
    
    ax.set_xlabel('Task 2 Test Time (ms)')
    ax.set_ylabel('Task 3 Train Time (ms)')
    ax.set_title(f'Time Generalization Matrix (TGM)\n{SUBJECT} | {ROI_NAME} ({FEATURE_TYPE.upper()}) | Task3 -> Task2\nCompute Time: {t_calc:.2f}s ({n_steps}x{n_steps})')
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, f'tgm_{SUBJECT}_{ROI_NAME}_{FEATURE_TYPE}.png')
    plt.savefig(png_path)
    plt.close()
    
    print(f"[SAVE] Result matrix saved to: {npz_path}")
    print(f"[SAVE] Heatmap image saved to : {png_path}")

if __name__ == '__main__':
    main()
