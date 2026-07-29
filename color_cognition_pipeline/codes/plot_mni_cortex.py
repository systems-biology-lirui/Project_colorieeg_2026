import os
import ast
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
subjects = ['test001', 'test002', 'test003']

def load_electrode_groups(subject, df_loc):
    groups = {'type1': [], 'colorwithsti': []}
    
    # type1
    if subject == 'test001':
        type1_path = os.path.join(pipeline_dir, 'data', 'Table_ERP_SingleCategory_Significant.csv')
    else:
        type1_path = os.path.join(pipeline_dir, 'data', f'{subject}_Table_ERP_SingleCategory_Significant.csv')
    if os.path.exists(type1_path):
        df_type1 = pd.read_csv(type1_path)
        groups['type1'] = df_type1[df_type1['In_Target_Area'] == True]['Electrode'].astype(str).tolist()
        
    # colorwithsti (AAL3 column is labeled color_with_sti)
    cols = df_loc.columns.tolist()
    aal_col = 'AAL3 (MNI-linear)' if 'AAL3 (MNI-linear)' in cols else ('AAL3 (MNI-segment)' if 'AAL3 (MNI-segment)' in cols else '')
    if aal_col:
        is_color_sti = df_loc[aal_col].astype(str).str.lower().str.replace('-', '_').str.replace(' ', '_') == 'color_with_sti'
        groups['colorwithsti'] = df_loc[is_color_sti]['Channel'].astype(str).unique().tolist()
        
    return groups

# Setup PyVista off_screen to save image headlessly
mne.viz.set_3d_backend('pyvistaqt')
subjects_dir = os.path.dirname(mne.datasets.fetch_fsaverage())
brain = mne.viz.Brain('fsaverage', hemi='rh', subjects_dir=subjects_dir, surf='pial', 
                      cortex='bone', alpha=0.3, background='white', size=(800, 800))

color_sel_lh, color_sel_rh = [], []
type1_only_lh, type1_only_rh = [], []

for subj in subjects:
    loc_path = os.path.join(base_dir, 'processed_data', subj, f'{subj}_ieegloc.xlsx')
    if not os.path.exists(loc_path): continue
    
    df_loc = pd.read_excel(loc_path)
    groups = load_electrode_groups(subj, df_loc)
    
    type1 = set(groups['type1'])
    colorsti = set(groups['colorwithsti'])
    
    processed_channels = set()
    for idx, row in df_loc.iterrows():
        ch = str(row['Channel'])
        if ch in processed_channels: continue
        
        if ch in type1 or ch in colorsti:
            mni_str = str(row['MNI'])
            try:
                # the coordinates are e.g. "[29.104,-44.106,4.362]" in mm
                coords = np.array(ast.literal_eval(mni_str))
                hemi = 'lh' if coords[0] < 0 else 'rh'
                
                # add to lists
                if ch in colorsti:
                    if hemi == 'lh': color_sel_lh.append(coords)
                    else: color_sel_rh.append(coords)
                else:
                    if hemi == 'lh': type1_only_lh.append(coords)
                    else: type1_only_rh.append(coords)
                
                processed_channels.add(ch)
            except Exception as e:
                print(f"Error parsing MNI for {subj} {ch}: {mni_str}")

# Add foci
if type1_only_lh:
    brain.add_foci(np.array(type1_only_lh), coords_as_verts=False, hemi='lh', scale_factor=0.3, color='blue', alpha=0.9, name='type1_lh')
if type1_only_rh:
    brain.add_foci(np.array(type1_only_rh), coords_as_verts=False, hemi='rh', scale_factor=0.3, color='blue', alpha=0.9, name='type1_rh')
    
if color_sel_lh:
    brain.add_foci(np.array(color_sel_lh), coords_as_verts=False, hemi='lh', scale_factor=0.6, color='red', alpha=1.0, name='color_lh')
if color_sel_rh:
    brain.add_foci(np.array(color_sel_rh), coords_as_verts=False, hemi='rh', scale_factor=0.6, color='red', alpha=1.0, name='color_rh')

# Configure view
out_dir = os.path.join(pipeline_dir, 'images')
os.makedirs(out_dir, exist_ok=True)

# Right hemisphere lateral
brain.show_view('lateral')
brain.save_image(os.path.join(out_dir, 'mne_cortex_rh_lateral.png'))

# Right hemisphere medial
brain.show_view('medial')
brain.save_image(os.path.join(out_dir, 'mne_cortex_rh_medial.png'))

# Posterior looking up 20 deg
# For rh (Right Hemisphere):
# lateral is azimuth=0, medial is azimuth=180, rostral is azimuth=90, caudal (posterior) is azimuth=270.
# elevation=90 is horizontal, elevation=110 is looking from below by 20 degrees (仰视20度).
brain.show_view(azimuth=270, elevation=110)
brain.save_image(os.path.join(out_dir, 'mne_cortex_posterior_up20.png'))

# Copy generated images to brain directory so they show up correctly in artifacts
brain_dir = '/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895'
if os.path.exists(brain_dir):
    import shutil
    for img_name in ['mne_cortex_rh_lateral.png', 'mne_cortex_rh_medial.png', 'mne_cortex_posterior_up20.png']:
        src = os.path.join(out_dir, img_name)
        dst = os.path.join(brain_dir, img_name)
        try:
            shutil.copy(src, dst)
            print(f"Copied {img_name} to brain dir.")
        except Exception as e:
            print(f"Failed to copy {img_name}: {e}")

print("Saved multiple brain views!")
brain.close()
