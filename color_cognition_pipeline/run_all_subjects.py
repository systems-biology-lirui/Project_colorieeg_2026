import os
import sys
import subprocess

subjects = ['test002', 'test003']

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

python_exe = '/home/lirui/anaconda3/envs/lr2026/bin/python'

for subj in subjects:
    print(f"\n================ Processing {subj} ================")
    
    # 1. Select Electrodes (type1 and colorwithsti)
    # The select_electrodes.py internally loops over test002 and test003, but we can run it once or refactor it.
    # Actually select_electrodes.py already loops over test002 and test003. We'll run it outside the loop.
    pass

print("\n--- 1. Selecting Electrodes ---")
run_cmd(f"{python_exe} codes/select_electrodes.py")

for subj in subjects:
    print(f"\n--- 2. Plotting Type 1 Electrodes for {subj} ---")
    run_cmd(f"{python_exe} codes/plot_type1_electrodes.py {subj}")
    
    print(f"\n--- 3. Plotting Target (colorwithsti) Electrodes for {subj} ---")
    run_cmd(f"{python_exe} codes/plot_target_electrodes.py {subj}")
    
    print(f"\n--- 4. Plotting CSI Distribution for {subj} ---")
    run_cmd(f"{python_exe} codes/plot_csi_distribution.py {subj}")
    
    print(f"\n--- 5. Running Decoding for {subj} ---")
    run_cmd(f"{python_exe} codes/decode_memory_color_updated.py {subj}")

print("\nAll Python analyses completed!")
