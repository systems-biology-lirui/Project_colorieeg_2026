import scipy.io as sio
import numpy as np

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'

print("--- Inspecting task1_TFA_epoched.mat ---")
try:
    mat = sio.loadmat(f"{base_dir}/processed_data/test001/task1_TFA_epoched.mat", squeeze_me=True, struct_as_record=False)
    for k in mat.keys():
        if not k.startswith('__'):
            val = mat[k]
            if isinstance(val, np.ndarray):
                print(f"{k}: {val.shape}, type: {type(val)}")
            else:
                print(f"{k}: type: {type(val)}, properties: {dir(val)}")
                if hasattr(val, 'data'):
                    print(f"data shape: {val.data.shape}")
except Exception as e:
    print(f"Error: {e}")

try:
    print("\n--- Inspecting groupedData.mat ---")
    gd = sio.loadmat(f"{base_dir}/processed_data/test001/groupedData.mat")
    print("Keys in groupedData.mat:", gd.keys())
    if 'groupedData' in gd:
        print("groupedData shape:", gd['groupedData'].shape)
except Exception as e:
    pass
