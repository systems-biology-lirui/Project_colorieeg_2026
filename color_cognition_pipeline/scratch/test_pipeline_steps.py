import mne
import scipy.io as sio
import numpy as np
import os
from pymatreader import read_mat

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')

set_path = os.path.join(base_dir, 'seegdata', 'test1', 'erp1.set')
final_erp_path = os.path.join(base_dir, 'processed_data', 'test001', 'task1_ERP_epoched.mat')
final_hg_path = os.path.join(pipeline_dir, 'feature', 'subband_60_150', 'task1_hg_subband.mat')

print("Loading raw eeglab data...")
raw = mne.io.read_raw_eeglab(set_path, preload=False)
events, event_id = mne.events_from_annotations(raw)
print("Event ID Map:", event_id)

print("\nLoading final ERP mat...")
erp_mat = read_mat(final_erp_path)
erp_epoch = erp_mat['epoch']
print("ERP keys:", erp_epoch.keys())
print("ERP data shape:", erp_epoch['data'].shape) # Expected [Cond, Rep, Ch, Time]
print("ERP triggers:", erp_epoch['trigger'])
print("ERP channels (first 15):", erp_epoch['ch']['labels'][:15])

print("\nLoading final HG mat...")
hg_mat = read_mat(final_hg_path)
hg_epoch = hg_mat['epoch']
print("HG keys:", hg_epoch.keys())
print("HG data_cell type:", type(hg_epoch['data_cell']))
print("HG data_cell length:", len(hg_epoch['data_cell']))
print("HG data_cell[0] shape:", hg_epoch['data_cell'][0].shape) # Expected [Rep, Ch, Time]
print("HG triggers:", hg_epoch['trigger'])
print("HG channels (first 15):", hg_epoch['ch']['labels'][:15])
