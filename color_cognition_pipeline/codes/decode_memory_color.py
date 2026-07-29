import os
import numpy as np
import scipy.io as sio
from pymatreader import read_mat
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')

colorsti_elecs = ['D3', 'D4', 'D5', 'D6', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10']

# Triggers
task3_red = ['Trigger-In:51']
task3_green = ['Trigger-In:54']

task2_red = ['Trigger-In:121', 'Trigger-In:122', 'Trigger-In:123', 'Trigger-In:131', 'Trigger-In:132', 'Trigger-In:133']
task2_green = ['Trigger-In:101', 'Trigger-In:102', 'Trigger-In:103', 'Trigger-In:111', 'Trigger-In:112', 'Trigger-In:113']

def get_data(mat_path, is_erp, trigs, elecs):
    if is_erp:
        mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        epoch = mat_data['epoch']
        data = epoch.data # [Cond, Rep, Ch, Time]
        ch_arr = epoch.ch
        ch_names = [str(ch.labels) for ch in ch_arr]
        time_ms = epoch.time_ms
        if hasattr(epoch, 'eventtype'):
            all_trigs = [str(t) for t in epoch.eventtype]
        else:
            all_trigs = [str(t) for t in epoch.name] if isinstance(epoch.name, np.ndarray) else []
            
        ch_indices = [ch_names.index(e) for e in elecs if e in ch_names]
        if not ch_indices: return None, None
        
        idx_list = [all_trigs.index(t) for t in trigs if t in all_trigs]
        if not idx_list: return None, None
        
        c_data = np.concatenate([data[idx][:, ch_indices, :] for idx in idx_list], axis=0)
        return c_data, time_ms
    else:
        mat_data = read_mat(mat_path)
        epoch = mat_data['epoch']
        time_ms = epoch['time_ms']
        all_trigs = epoch['trigger']
        if isinstance(all_trigs, str): all_trigs = [all_trigs]
        data_cell = epoch['data_cell']
        ch_names = epoch['ch']['labels']
        if isinstance(ch_names, str): ch_names = [ch_names]
        
        ch_indices = [ch_names.index(e) for e in elecs if e in ch_names]
        if not ch_indices: return None, None
            
        idx_list = [all_trigs.index(t) for t in trigs if t in all_trigs]
        if not idx_list: return None, None
        
        c_data = np.concatenate([data_cell[idx][:, ch_indices, :] for idx in idx_list], axis=0)
        return c_data, time_ms

def run_decoding(feature_type):
    print(f"Running decoding for {feature_type}...")
    if feature_type == 'erp':
        task3_path = os.path.join(base_dir, 'processed_data', 'test001', 'task3_ERP_epoched.mat')
        task2_path = os.path.join(base_dir, 'processed_data', 'test001', 'task2_ERP_epoched.mat')
        is_erp = True
    else:
        task3_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', 'task3_hg_subband.mat')
        task2_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', 'task2_hg_subband.mat')
        is_erp = False
        
    if not os.path.exists(task3_path) or not os.path.exists(task2_path):
        print(f"Missing files for {feature_type}")
        return None, None
        
    # Load Training Data (Task 3)
    train_r, t_train = get_data(task3_path, is_erp, task3_red, colorsti_elecs)
    train_g, _ = get_data(task3_path, is_erp, task3_green, colorsti_elecs)
    if train_r is None or train_g is None:
        return None, None
        
    # Load Testing Data (Task 2)
    test_r, t_test = get_data(task2_path, is_erp, task2_red, colorsti_elecs)
    test_g, _ = get_data(task2_path, is_erp, task2_green, colorsti_elecs)
    if test_r is None or test_g is None:
        return None, None
        
    # Remove NaNs
    train_r = train_r[~np.isnan(train_r).any(axis=(1,2))]
    train_g = train_g[~np.isnan(train_g).any(axis=(1,2))]
    test_r = test_r[~np.isnan(test_r).any(axis=(1,2))]
    test_g = test_g[~np.isnan(test_g).any(axis=(1,2))]
    
    n_time = train_r.shape[2]
    accuracies = []
    
    for t in range(n_time):
        X_train = np.vstack([train_r[:, :, t], train_g[:, :, t]])
        y_train = np.hstack([np.zeros(train_r.shape[0]), np.ones(train_g.shape[0])])
        
        X_test = np.vstack([test_r[:, :, t], test_g[:, :, t]])
        y_test = np.hstack([np.zeros(test_r.shape[0]), np.ones(test_g.shape[0])])
        
        clf = SVC(kernel='linear', C=1.0)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        
    return t_train, accuracies

def plot_decoding():
    out_dir = os.path.join(pipeline_dir, 'images', 'decoding')
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t_erp, acc_erp = run_decoding('erp')
    if acc_erp:
        ax.plot(t_erp, acc_erp, color='blue', label='ERP', linewidth=2)
        
    t_hg, acc_hg = run_decoding('subband_60_150')
    if acc_hg:
        ax.plot(t_hg, acc_hg, color='red', label='Subband 60-150Hz', linewidth=2)
        
    ax.axhline(0.5, color='gray', linestyle='--', label='Chance Level (50%)')
    ax.axvline(0, color='k', linestyle='-')
    
    # Optional smoothing
    def smooth(y, box_pts=5):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    if acc_erp:
        ax.plot(t_erp, smooth(acc_erp, 10), color='blue', linestyle=':', alpha=0.5)
    if acc_hg:
        ax.plot(t_hg, smooth(acc_hg, 10), color='red', linestyle=':', alpha=0.5)
        
    ax.set_title("Cross-Condition Decoding (Train: Task 3 True Color -> Test: Task 2 Memory Color)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Decoding Accuracy")
    ax.set_xlim([-200, 800])
    ax.set_ylim([0.3, 0.8])
    ax.legend(loc='upper right')
    
    out_fig = os.path.join(out_dir, 'Decoding_Task3_to_Task2.png')
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print(f"Saved {out_fig}")

if __name__ == '__main__':
    plot_decoding()
