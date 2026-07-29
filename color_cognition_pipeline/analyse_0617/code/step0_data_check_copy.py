import os
import shutil

base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
pipeline_dir = os.path.join(base_dir, 'color_cognition_pipeline')
analyse_dir = os.path.join(pipeline_dir, 'analyse_0617')
feature_dir = os.path.join(analyse_dir, 'feature')

subjects = ['test001', 'test002', 'test003']
tasks = [1, 2, 3]

print("="*60)
print("Step 0: Creating Feature Directories and Copying Preprocessed Data")
print("="*60)

# 1. Create directory structure in analyse_0617/feature/
dirs_to_create = [
    os.path.join(analyse_dir, 'code'),
    os.path.join(analyse_dir, 'result'),
    feature_dir,
]
for subj in subjects:
    dirs_to_create.append(os.path.join(feature_dir, subj))

for d in dirs_to_create:
    if not os.path.exists(d):
        os.makedirs(d)
        print(f"Created directory: {d}")
    else:
        print(f"Directory already exists: {d}")

# 2. Copy ERP and HG files to feature/subject/
for subj in subjects:
    print(f"\nProcessing Subject: {subj}")
    
    # ERP Files
    for task in tasks:
        src_erp = os.path.join(base_dir, 'processed_data', subj, f'task{task}_ERP_epoched.mat')
        dst_erp = os.path.join(feature_dir, subj, f'task{task}_ERP_epoched.mat')
        
        if os.path.exists(src_erp):
            print(f"  Copying ERP: {src_erp} -> {dst_erp}")
            shutil.copy2(src_erp, dst_erp)
        else:
            print(f"  [ERROR] Source ERP missing: {src_erp}")
            
    # Subband HG Files
    for task in tasks:
        if subj == 'test001':
            src_hg = os.path.join(pipeline_dir, 'feature', 'subband_60_150', f'task{task}_hg_subband.mat')
        else:
            src_hg = os.path.join(pipeline_dir, 'feature', 'subband_60_150', subj, f'task{task}_hg_subband.mat')
            
        dst_hg = os.path.join(feature_dir, subj, f'task{task}_hg_subband.mat')
        
        if os.path.exists(src_hg):
            print(f"  Copying HG: {src_hg} -> {dst_hg}")
            shutil.copy2(src_hg, dst_hg)
        else:
            print(f"  [ERROR] Source HG missing: {src_hg}")

# 3. Clean up the old subject directories directly under analyse_0617/
print("\nCleaning up old subject directories directly under analyse_0617/...")
for subj in subjects:
    old_dir = os.path.join(analyse_dir, subj)
    if os.path.exists(old_dir):
        print(f"  Removing old directory: {old_dir}")
        shutil.rmtree(old_dir)

print("\nStep 0 Data copying and cleanup completed!")
