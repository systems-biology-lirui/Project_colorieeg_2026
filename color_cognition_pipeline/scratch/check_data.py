import os
import scipy.io as sio
import numpy as np
from pymatreader import read_mat

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
subjects = ['test001', 'test002', 'test003']

print("="*60)
print("Data Integrity Check for ERP and Subband 60-150Hz")
print("="*60)

def safe_load_mat(path):
    # Try pymatreader first, then scipy.io
    try:
        data = read_mat(path)
        return data, 'pymatreader'
    except Exception as e1:
        try:
            data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
            return data, 'scipy'
        except Exception as e2:
            raise Exception(f"Failed to load with both reader. Error1: {e1}. Error2: {e2}")

for subj in subjects:
    print(f"\nSubject: {subj}")
    
    # Check ERP files
    print("  --- ERP Files ---")
    for task in [1, 2, 3]:
        erp_path = os.path.join(base_dir, 'processed_data', subj, f'task{task}_ERP_epoched.mat')
        if not os.path.exists(erp_path):
            print(f"    Task {task} ERP: Missing! Path: {erp_path}")
            continue
        try:
            mat_data, reader = safe_load_mat(erp_path)
            epoch = mat_data['epoch']
            
            if reader == 'pymatreader':
                data = epoch['data']
                ch_arr = epoch['ch']
                ch_names = list(ch_arr['labels']) if 'labels' in ch_arr else []
                time_ms = epoch.get('time_ms', None)
                if time_ms is None:
                    time_ms = epoch.get('times', None)
                if time_ms is None:
                    # fallback
                    time_ms = np.linspace(-500, 998, data[0].shape[-1])
                triggers = epoch.get('trigger', [])
                if isinstance(triggers, str): triggers = [triggers]
                
                n_conds = len(data)
                total_trials = 0
                has_nan = False
                for cond_idx in range(n_conds):
                    cond_data = data[cond_idx]
                    total_trials += cond_data.shape[0]
                    if np.isnan(cond_data).any():
                        has_nan = True
            else:
                data = epoch.data
                ch_arr = epoch.ch
                ch_names = [str(ch.labels) for ch in ch_arr] if hasattr(ch_arr[0], 'labels') else [str(ch) for ch in ch_arr]
                time_ms = epoch.time_ms if hasattr(epoch, 'time_ms') else (epoch.times if hasattr(epoch, 'times') else None)
                if time_ms is None:
                    time_ms = np.linspace(-500, 998, data[0].shape[-1])
                
                n_conds = len(data)
                total_trials = 0
                has_nan = False
                for cond_idx in range(n_conds):
                    cond_data = data[cond_idx]
                    total_trials += cond_data.shape[0]
                    if np.isnan(cond_data).any():
                        has_nan = True
            
            print(f"    Task {task} ERP ({reader}): Loaded successfully.")
            print(f"      Conditions: {n_conds}, Total Trials: {total_trials}, Channels: {len(ch_names)}, Time points: {len(time_ms)}")
            print(f"      Time range: {time_ms[0]} to {time_ms[-1]} ms")
            print(f"      Contains NaN: {has_nan}")
        except Exception as e:
            print(f"    Task {task} ERP error: {e}")
            
    # Check Subband High Gamma files
    print("  --- Subband 60-150Hz HG Files ---")
    for task in [1, 2, 3]:
        if subj == 'test001':
            hg_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', f'task{task}_hg_subband.mat')
        else:
            hg_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', subj, f'task{task}_hg_subband.mat')
            
        if not os.path.exists(hg_path):
            print(f"    Task {task} HG: Missing! Path: {hg_path}")
            continue
        try:
            mat_data, reader = safe_load_mat(hg_path)
            epoch = mat_data['epoch']
            
            if reader == 'pymatreader':
                data_cell = epoch['data_cell']
                ch_arr = epoch['ch']
                ch_names = list(ch_arr['labels']) if 'labels' in ch_arr else []
                time_ms = epoch.get('time_ms', None)
                if time_ms is None:
                    time_ms = epoch.get('times', None)
                if time_ms is None:
                    time_ms = np.linspace(-500, 998, data_cell[0].shape[-1])
                triggers = epoch.get('trigger', [])
                if isinstance(triggers, str): triggers = [triggers]
                
                n_conds = len(data_cell)
                total_trials = 0
                has_nan = False
                for cond_idx in range(n_conds):
                    cond_data = data_cell[cond_idx]
                    total_trials += cond_data.shape[0]
                    if np.isnan(cond_data).any():
                        has_nan = True
            else:
                data_cell = epoch.data_cell
                ch_arr = epoch.ch
                ch_names = [str(ch.labels) for ch in ch_arr] if hasattr(ch_arr[0], 'labels') else [str(ch) for ch in ch_arr]
                time_ms = epoch.time_ms if hasattr(epoch, 'time_ms') else (epoch.times if hasattr(epoch, 'times') else None)
                if time_ms is None:
                    time_ms = np.linspace(-500, 998, data_cell[0].shape[-1])
                
                n_conds = len(data_cell)
                total_trials = 0
                has_nan = False
                for cond_idx in range(n_conds):
                    cond_data = data_cell[cond_idx]
                    total_trials += cond_data.shape[0]
                    if np.isnan(cond_data).any():
                        has_nan = True
                    
            print(f"    Task {task} HG ({reader}): Loaded successfully.")
            print(f"      Conditions: {n_conds}, Total Trials: {total_trials}, Channels: {len(ch_names)}, Time points: {len(time_ms)}")
            print(f"      Time range: {time_ms[0]} to {time_ms[-1]} ms")
            print(f"      Contains NaN: {has_nan}")
        except Exception as e:
            print(f"    Task {task} HG error: {e}")
