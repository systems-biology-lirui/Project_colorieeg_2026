import mne
import os

set_path = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026/seegdata/test1/erp1.set'
print("Exists:", os.path.exists(set_path))

try:
    raw = mne.io.read_raw_eeglab(set_path, preload=False)
    print("Channels:", raw.info['ch_names'])
    print("Srate:", raw.info['sfreq'])
except Exception as e:
    print("Error:", e)
