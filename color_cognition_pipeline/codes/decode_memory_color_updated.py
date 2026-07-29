import os
import sys
import numpy as np
import scipy.io as sio
from pymatreader import read_mat
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')

def get_data(mat_path, is_erp, trigs, elecs):
    if not os.path.exists(mat_path): return None, None
    if is_erp:
        mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        epoch = mat_data['epoch']
        data = epoch.data # [Cond, Rep, Ch, Time]
        ch_arr = epoch.ch
        ch_names = [str(ch.labels) for ch in ch_arr]
        time_ms = epoch.time_ms if hasattr(epoch, 'time_ms') else np.linspace(-500, 998, data[0].shape[-1])
        if hasattr(epoch, 'trigger'):
            all_trigs = [str(t) for t in epoch.trigger]
        elif hasattr(epoch, 'eventtype'):
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
        mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        epoch = mat_data['epoch']
        time_ms = epoch.time_ms if hasattr(epoch, 'time_ms') else np.linspace(-500, 998, epoch.data_cell[0].shape[-1])
        if hasattr(epoch, 'trigger'):
            all_trigs = [str(t) for t in epoch.trigger]
        elif hasattr(epoch, 'eventtype'):
            all_trigs = [str(t) for t in epoch.eventtype]
        else:
            all_trigs = [str(t) for t in epoch.name] if isinstance(epoch.name, np.ndarray) else []
            
        ch_arr = epoch.ch
        ch_names = [str(ch.labels) for ch in ch_arr]
        
        ch_indices = [ch_names.index(e) for e in elecs if e in ch_names]
        if not ch_indices: return None, None
            
        idx_list = [all_trigs.index(t) for t in trigs if t in all_trigs]
        if not idx_list: return None, None
        
        c_data = np.concatenate([epoch.data_cell[idx][:, ch_indices, :] for idx in idx_list], axis=0)
        return c_data, time_ms

def load_electrode_groups(subject):
    groups = {'type1': [], 'colorwithsti': [], 'temporal_pole': []}
    
    # type1
    if subject == 'test001':
        type1_path = os.path.join(pipeline_dir, 'data', 'Table_ERP_SingleCategory_Significant.csv')
    else:
        type1_path = os.path.join(pipeline_dir, 'data', f'{subject}_Table_ERP_SingleCategory_Significant.csv')
    if os.path.exists(type1_path):
        df_type1 = pd.read_csv(type1_path)
        groups['type1'] = df_type1[df_type1['In_Target_Area'] == True]['Electrode'].astype(str).tolist()
        
    # colorwithsti (directly from ieegloc.xlsx AAL3 column)
    loc_path = os.path.join(base_dir, 'processed_data', subject, f'{subject}_ieegloc.xlsx')
    if os.path.exists(loc_path):
        df_loc = pd.read_excel(loc_path)
        cols = df_loc.columns.tolist()
        aal_col = 'AAL3 (MNI-linear)' if 'AAL3 (MNI-linear)' in cols else ('AAL3 (MNI-segment)' if 'AAL3 (MNI-segment)' in cols else '')
        if aal_col:
            is_color_sti = df_loc[aal_col].astype(str).str.lower().str.replace('-', '_').str.replace(' ', '_') == 'color_with_sti'
            groups['colorwithsti'] = df_loc[is_color_sti]['Channel'].astype(str).unique().tolist()
            
    # temporal_pole
    if os.path.exists(loc_path):
        df_loc = pd.read_excel(loc_path)
        cols = df_loc.columns.tolist()
        # Find which columns are anatomical labels (Desikan, DKT, AAL3 etc)
        anat_cols = [c for c in cols if any(term in c.lower() for term in ['desikan', 'dkt', 'aal3'])]
        if anat_cols:
            mask = df_loc[anat_cols].astype(str).apply(lambda x: x.str.contains('Temporal_Pole', case=False, na=False)).any(axis=1)
            groups['temporal_pole'] = df_loc[mask]['Channel'].astype(str).unique().tolist()
        
    return groups

def decode_memory_pairs(subject, feature_type, elecs_group_name, elecs, is_erp, task2_path):
    print(f"[{subject}] Memory Pairs Decoding on {elecs_group_name} ({feature_type})...")
    
    # Gray Memory Fruits
    r1_trigs = ['Trigger-In:123'] # Strawberry Gray
    r2_trigs = ['Trigger-In:133'] # Watermelon Gray
    g1_trigs = ['Trigger-In:103'] # Cabbage Gray
    g2_trigs = ['Trigger-In:113'] # Kiwi Gray
    
    d_r1, t_ms = get_data(task2_path, is_erp, r1_trigs, elecs)
    d_r2, _ = get_data(task2_path, is_erp, r2_trigs, elecs)
    d_g1, _ = get_data(task2_path, is_erp, g1_trigs, elecs)
    d_g2, _ = get_data(task2_path, is_erp, g2_trigs, elecs)
    
    if any(x is None for x in [d_r1, d_r2, d_g1, d_g2]):
        print("  Missing data for one of the fruits.")
        return
        
    # Remove NaNs
    def clean(x): return x[~np.isnan(x).any(axis=(1,2))]
    d_r1, d_r2, d_g1, d_g2 = map(clean, [d_r1, d_r2, d_g1, d_g2])
    
    pairs = [
        (d_r1, d_g1, d_r2, d_g2),
        (d_r1, d_g2, d_r2, d_g1),
        (d_r2, d_g1, d_r1, d_g2),
        (d_r2, d_g2, d_r1, d_g1)
    ]
    
    n_time = t_ms.shape[0]
    all_acc = np.zeros((4, n_time))
    
    for i, (train_r, train_g, test_r, test_g) in enumerate(pairs):
        def _fit_eval_mem(t):
            X_tr = np.vstack([train_r[:, :, t], train_g[:, :, t]])
            y_tr = np.hstack([np.zeros(train_r.shape[0]), np.ones(train_g.shape[0])])
            X_te = np.vstack([test_r[:, :, t], test_g[:, :, t]])
            y_te = np.hstack([np.zeros(test_r.shape[0]), np.ones(test_g.shape[0])])
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            clf = SVC(kernel='linear', C=1.0)
            clf.fit(X_tr, y_tr)
            return accuracy_score(y_te, clf.predict(X_te))
            
        all_acc[i, :] = Parallel(n_jobs=-1)(delayed(_fit_eval_mem)(t) for t in range(n_time))
            
    mean_acc = np.mean(all_acc, axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_ms, mean_acc, color='purple', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.axvline(0, color='k')
    ax.set_title(f"Memory Color Decoding ({elecs_group_name})")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0.3, 0.8])
    
    out_dir = os.path.join(pipeline_dir, 'images', subject, 'decoding', 'memory_pairs')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{feature_type}_{elecs_group_name}_acc.npy"), mean_acc)
    np.save(os.path.join(out_dir, f"{feature_type}_{elecs_group_name}_time.npy"), t_ms)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{feature_type}_{elecs_group_name}.png"), dpi=300)
    plt.close()

def decode_true_false(subject, feature_type, elecs_group_name, elecs, is_erp, task2_path):
    print(f"[{subject}] True/False Decoding on {elecs_group_name} ({feature_type})...")
    
    fruits = [
        (['Trigger-In:101'], ['Trigger-In:102']), # Cabbage
        (['Trigger-In:111'], ['Trigger-In:112']), # Kiwi
        (['Trigger-In:121'], ['Trigger-In:122']), # Strawberry
        (['Trigger-In:131'], ['Trigger-In:132'])  # Watermelon
    ]
    
    all_data = []
    t_ms = None
    for t_trigs, f_trigs in fruits:
        d_t, t = get_data(task2_path, is_erp, t_trigs, elecs)
        d_f, _ = get_data(task2_path, is_erp, f_trigs, elecs)
        if d_t is not None and d_f is not None:
            # Clean
            d_t = d_t[~np.isnan(d_t).any(axis=(1,2))]
            d_f = d_f[~np.isnan(d_f).any(axis=(1,2))]
            all_data.append((d_t, d_f))
            t_ms = t
            
    if len(all_data) != 4:
        print("  Missing True/False data for fruits.")
        return
        
    n_time = t_ms.shape[0]
    all_acc = np.zeros((4, n_time))
    
    for test_idx in range(4):
        test_t, test_f = all_data[test_idx]
        train_t_list = [all_data[i][0] for i in range(4) if i != test_idx]
        train_f_list = [all_data[i][1] for i in range(4) if i != test_idx]
        
        train_t = np.concatenate(train_t_list, axis=0)
        train_f = np.concatenate(train_f_list, axis=0)
        
        def _fit_eval_tf(t):
            X_tr = np.vstack([train_t[:, :, t], train_f[:, :, t]])
            y_tr = np.hstack([np.zeros(train_t.shape[0]), np.ones(train_f.shape[0])])
            X_te = np.vstack([test_t[:, :, t], test_f[:, :, t]])
            y_te = np.hstack([np.zeros(test_t.shape[0]), np.ones(test_f.shape[0])])
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            clf = SVC(kernel='linear', C=1.0)
            clf.fit(X_tr, y_tr)
            return accuracy_score(y_te, clf.predict(X_te))
            
        all_acc[test_idx, :] = Parallel(n_jobs=-1)(delayed(_fit_eval_tf)(t) for t in range(n_time))
            
    mean_acc = np.mean(all_acc, axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_ms, mean_acc, color='orange', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.axvline(0, color='k')
    ax.set_title(f"True vs False Decoding ({elecs_group_name})")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0.3, 0.8])
    
    out_dir = os.path.join(pipeline_dir, 'images', subject, 'decoding', 'true_false')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{feature_type}_{elecs_group_name}_acc.npy"), mean_acc)
    np.save(os.path.join(out_dir, f"{feature_type}_{elecs_group_name}_time.npy"), t_ms)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{feature_type}_{elecs_group_name}.png"), dpi=300)
    plt.close()

def decode_time_generalization(subject, feature_type, elecs_group_name, elecs, is_erp, task3_path, task2_path):
    print(f"[{subject}] Time Generalization Decoding on {elecs_group_name} ({feature_type})...")
    
    t3_r = ['Trigger-In:51']
    t3_g = ['Trigger-In:54']
    t2_r = ['Trigger-In:123', 'Trigger-In:133'] # Gray memory red
    t2_g = ['Trigger-In:103', 'Trigger-In:113'] # Gray memory green
    
    train_r, t_train = get_data(task3_path, is_erp, t3_r, elecs)
    train_g, _ = get_data(task3_path, is_erp, t3_g, elecs)
    test_r, t_test = get_data(task2_path, is_erp, t2_r, elecs)
    test_g, _ = get_data(task2_path, is_erp, t2_g, elecs)
    
    if any(x is None for x in [train_r, train_g, test_r, test_g]):
        print(f"  Missing data for TGM ({feature_type}).")
        return
        
    def clean(x): return x[~np.isnan(x).any(axis=(1,2))]
    train_r, train_g, test_r, test_g = map(clean, [train_r, train_g, test_r, test_g])
    
    n_train_t = t_train.shape[0]
    n_test_t = t_test.shape[0]
    
    # Build TGM with downsampling (every 10th point -> 50Hz / 20ms resolution)
    step = 10
    tr_indices = list(range(0, n_train_t, step))
    te_indices = list(range(0, n_test_t, step))
    tgm = np.zeros((len(tr_indices), len(te_indices)))
    
    def _fit_eval_tgm(tr_t):
        X_tr = np.vstack([train_r[:, :, tr_t], train_g[:, :, tr_t]])
        y_tr = np.hstack([np.zeros(train_r.shape[0]), np.ones(train_g.shape[0])])
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        
        clf = SVC(kernel='linear', C=1.0)
        clf.fit(X_tr, y_tr)
        
        row_acc = np.zeros(len(te_indices))
        for j, te_t in enumerate(te_indices):
            X_te = np.vstack([test_r[:, :, te_t], test_g[:, :, te_t]])
            X_te = scaler.transform(X_te)
            y_te = np.hstack([np.zeros(test_r.shape[0]), np.ones(test_g.shape[0])])
            row_acc[j] = accuracy_score(y_te, clf.predict(X_te))
        return row_acc
        
    tgm_list = Parallel(n_jobs=-1)(delayed(_fit_eval_tgm)(tr_t) for tr_t in tr_indices)
    tgm = np.vstack(tgm_list)
            
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.imshow(tgm, origin='lower', extent=[t_test[0], t_test[-1], t_train[0], t_train[-1]],
                    cmap='RdBu_r', vmin=0.3, vmax=0.7)
    fig.colorbar(cax, label='Accuracy')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Test Time (ms) [Task 2]')
    ax.set_ylabel('Train Time (ms) [Task 3]')
    ax.set_title(f"Time Generalization ({elecs_group_name}) - {feature_type.upper()}")
    
    out_dir = os.path.join(pipeline_dir, 'images', subject, 'decoding', 'time_generalization')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{feature_type}_{elecs_group_name}_tgm.npy"), tgm)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{feature_type}_{elecs_group_name}.png"), dpi=300)
    plt.close()

if __name__ == '__main__':
    subject = sys.argv[1] if len(sys.argv) > 1 else 'test001'
    groups = load_electrode_groups(subject)
    
    for ftype in ['erp', 'subband_60_150']:
        if ftype == 'erp':
            t2_path = os.path.join(base_dir, 'processed_data', subject, 'task2_ERP_epoched.mat')
            t3_path = os.path.join(base_dir, 'processed_data', subject, 'task3_ERP_epoched.mat')
            is_erp = True
        else:
            t2_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', subject, 'task2_hg_subband.mat')
            if not os.path.exists(t2_path):
                t2_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', 'task2_hg_subband.mat') # fallback for test001
            t3_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', subject, 'task3_hg_subband.mat')
            if not os.path.exists(t3_path):
                t3_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', 'task3_hg_subband.mat') # fallback for test001
            is_erp = False
            
        for group_name in ['colorwithsti', 'type1']:
            elecs = groups.get(group_name, [])
            if elecs:
                decode_memory_pairs(subject, ftype, group_name, elecs, is_erp, t2_path)
                
        for group_name in ['colorwithsti', 'type1', 'temporal_pole']:
            elecs = groups.get(group_name, [])
            if elecs:
                decode_true_false(subject, ftype, group_name, elecs, is_erp, t2_path)
                
        # Time Generalization
        for group_name in ['colorwithsti', 'type1']:
            elecs = groups.get(group_name, [])
            if elecs:
                decode_time_generalization(subject, ftype, group_name, elecs, is_erp, t3_path, t2_path)

