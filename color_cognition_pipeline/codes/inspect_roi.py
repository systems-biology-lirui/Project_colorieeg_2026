import scipy.io as sio

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'

print("--- Inspecting feature/highgamma/test001/Color_patch.mat ---")
try:
    mat = sio.loadmat(f"{base_dir}/feature/highgamma/test001/Color_patch.mat")
    for k in mat.keys():
        if not k.startswith('__'):
            val = mat[k]
            print(f"{k}: shape {getattr(val, 'shape', 'No shape')}")
            
    # Print channel names if present
    if 'channel_names' in mat:
        print("Channels:", mat['channel_names'])
    if 'channels' in mat:
        print("Channels:", mat['channels'])
except Exception as e:
    print(f"Error: {e}")
