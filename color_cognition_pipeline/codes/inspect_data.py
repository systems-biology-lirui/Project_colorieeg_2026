import scipy.io as sio
import pandas as pd
import numpy as np

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'

print("--- Inspecting task1_TFA_epoched.mat ---")
try:
    mat = sio.loadmat(f"{base_dir}/processed_data/test001/task1_TFA_epoched.mat")
    for k, v in mat.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: {v.shape}")
except Exception as e:
    print(f"Error loading TFA: {e}")

print("\n--- Inspecting test001_ieegloc.xlsx ---")
try:
    df = pd.read_excel(f"{base_dir}/processed_data/test001/test001_ieegloc.xlsx")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Number of channels: {len(df)}")
    print(df.head())
except Exception as e:
    print(f"Error loading ieegloc: {e}")
