import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

pipeline_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline'
images_dir = os.path.join(pipeline_dir, 'images')
out_avg_dir = os.path.join(images_dir, 'average_decoding')
os.makedirs(out_avg_dir, exist_ok=True)

subjects = ['test001', 'test002', 'test003']
tasks = ['memory_pairs', 'true_false']
features = ['erp', 'subband_60_150']
elecs_groups = ['colorwithsti', 'type1', 'temporal_pole']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colors for the three subjects

for task in tasks:
    for feature in features:
        for group in elecs_groups:
            all_acc = []
            time_arr = None
            subj_labels = []
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for i, subj in enumerate(subjects):
                acc_file = os.path.join(images_dir, subj, 'decoding', task, f"{feature}_{group}_acc.npy")
                time_file = os.path.join(images_dir, subj, 'decoding', task, f"{feature}_{group}_time.npy")
                
                if os.path.exists(acc_file) and os.path.exists(time_file):
                    acc = np.load(acc_file)
                    time_ms = np.load(time_file)
                    
                    if time_arr is None:
                        time_arr = time_ms
                        
                    # Smooth the accuracy curve
                    acc_smooth = gaussian_filter1d(acc, sigma=2)
                    all_acc.append(acc_smooth)
                    subj_labels.append(subj)
                    
                    # Plot thin line for single subject
                    ax.plot(time_ms, acc_smooth, color=colors[i], linewidth=1.5, alpha=0.6, label=subj)
                    
            if not all_acc:
                plt.close(fig)
                continue
                
            # Calculate and plot average
            all_acc_mat = np.array(all_acc)
            mean_acc = np.mean(all_acc_mat, axis=0)
            
            ax.plot(time_arr, mean_acc, color='black', linewidth=3.5, label='Average')
            
            # Chance level
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
            ax.axvline(0, color='gray', linestyle='--', linewidth=1)
            
            ax.set_title(f"Average Decoding: {task.replace('_', ' ').title()}\n{feature} | {group}")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Accuracy")
            ax.set_ylim([0.3, 0.8])
            ax.legend()
            
            plt.tight_layout()
            out_file = os.path.join(out_avg_dir, f"{task}_{feature}_{group}.png")
            plt.savefig(out_file, dpi=300)
            plt.close(fig)
            print(f"Saved {out_file}")

print("Average plotting completed!")
