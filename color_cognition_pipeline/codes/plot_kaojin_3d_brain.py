#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a 3D HTML brain view using Nilearn.
Plots ONLY the selected target electrodes closest to the PC, CC, and AC patches (12 electrodes)
and the PC, CC, and AC color patches (as large, semi-transparent markers on both hemispheres).
Saves to result/kaojin/color_patches_and_electrodes_3d.html
"""

import os
import shutil
import numpy as np
import pandas as pd
import ast
from nilearn import plotting

# 基础路径
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
kaojin_base = os.path.join(base_dir, 'color_cognition_pipeline', 'analyse_0617', 'result', 'kaojin')
artifacts_base = '/home/lirui/.gemini/antigravity-ide/brain/1870c00b-e214-4064-b47f-91fb6f22337f'

subjects = ['test001', 'test002', 'test003', 'test006']

# 颜色配置 (RGBA 格式，浮点数 [0, 1])
subj_colors = {
    'test001': (0.12, 0.53, 0.90, 0.85), # 🔵 鲜蓝色
    'test002': (1.00, 0.50, 0.05, 0.85), # 🟠 亮橙色
    'test003': (0.58, 0.40, 0.74, 0.85), # 🟣 柔紫色
    'test006': (0.84, 0.15, 0.16, 0.85)  # 🔴 艳红色
}

# 3 个 Patch 气泡的 3D 坐标
patches_config = {
    'PC': {
        'coords': [[-33.34, -80.21, 2.82], [33.34, -80.21, 2.82]],
        'color': (0.17, 0.63, 0.17, 0.30) # 🟢 半透明绿色，代表 PC
    },
    'CC': {
        'coords': [[-25.89, -56.93, -2.28], [25.89, -56.93, -2.28]],
        'color': (0.74, 0.74, 0.13, 0.30) # 🟡 半透明金黄色，代表 CC
    },
    'AC': {
        'coords': [[-33.57, -6.44, -3.50], [33.57, -6.44, -3.50]],
        'color': (0.09, 0.75, 0.81, 0.30) # 🔵 半透明青色，代表 AC
    }
}

# 12 个被选中的目标电极配置
TARGET_ELECTRODES = {
    'PC': {
        'test001': 'D13',
        'test002': 'C6',
        'test003': 'H11',
        'test006': 'A10'
    },
    'CC': {
        'test001': 'H1',
        'test002': 'C1',
        'test003': 'H4',
        'test006': 'G5'
    },
    'AC': {
        'test001': 'F5',
        'test002': 'F1',
        'test003': 'B5',
        'test006': 'G2'
    }
}

def main():
    print("="*60)
    print("Starting 3D HTML Brain View Generation (Target Electrodes Only)")
    print("="*60)
    
    os.makedirs(kaojin_base, exist_ok=True)
    
    all_coords = []
    all_colors = []
    all_sizes = []
    all_labels = []
    
    # 建立一个 (subject, channel) -> patch_name 的映射，用于辅助判定和标签
    target_lookup = {}
    for patch_name, sub_dict in TARGET_ELECTRODES.items():
        for subj, ch in sub_dict.items():
            target_lookup[(subj, ch)] = patch_name
            
    # 1. 提取目标电极通道坐标 (只保留 12 个)
    for subj in subjects:
        loc_path = os.path.join(base_dir, 'processed_data', subj, f'{subj}_ieegloc.xlsx')
        if not os.path.exists(loc_path):
            print(f"  [Warning] Location file not found for {subj}: {loc_path}")
            continue
            
        df = pd.read_excel(loc_path)
        ch_col = 'Channel' if 'Channel' in df.columns else df.columns[0]
        
        count = 0
        for idx, row in df.iterrows():
            ch = str(row[ch_col]).strip()
            # 只有当该通道在 12 个目标通道内时才绘制！
            if (subj, ch) in target_lookup:
                mni_str = row.get('MNI', None)
                if pd.notna(mni_str):
                    try:
                        coords = ast.literal_eval(str(mni_str))
                        if isinstance(coords, (list, tuple)) and len(coords) == 3:
                            all_coords.append(coords)
                            all_colors.append(subj_colors[subj])
                            all_sizes.append(12) # 电极球大小设为 12 像素
                            p_name = target_lookup[(subj, ch)]
                            all_labels.append(f"{subj}: {ch} (Matched {p_name}, MNI: {coords})")
                            count += 1
                    except:
                        pass
        print(f"  Loaded {count} target electrodes for {subj}")
        
    # 2. 提取 Color Patch 气泡 (左右双侧对称大球)
    print("  Adding Color Patch (PC, CC, AC) transparent bubble markers...")
    for patch_name, cfg in patches_config.items():
        for coord in cfg['coords']:
            all_coords.append(coord)
            all_colors.append(cfg['color'])
            all_sizes.append(45) # 气泡大球设为 45 像素
            all_labels.append(f"Color Patch: {patch_name} (MNI: {coord})")
            
    # 转换为 numpy 格式
    all_coords = np.array(all_coords)
    
    # 3. 使用 Nilearn 绘制 3D 脑电极分布图
    print("  Rendering 3D glass brain with Nilearn view_markers...")
    try:
        # view_markers 会生成三维的交互视图 (WebGL)
        view = plotting.view_markers(
            marker_coords=all_coords,
            marker_color=all_colors,
            marker_size=all_sizes,
            title="3D Brain View: Color Patches & 12 Target Electrodes"
        )
        
        # 导出为 HTML 文件
        out_html = os.path.join(kaojin_base, 'color_patches_and_electrodes_3d.html')
        view.save_as_html(out_html)
        print(f"  Saved 3D HTML view to: {out_html}")
        
        # 同步拷贝到 artifacts
        shutil.copy(out_html, os.path.join(artifacts_base, 'color_patches_and_electrodes_3d.html'))
        print(f"  Copied 3D HTML view to artifacts: {os.path.join(artifacts_base, 'color_patches_and_electrodes_3d.html')}")
    except Exception as e:
        print(f"  [Error] Failed to render 3D brain view: {e}")
        import traceback; traceback.print_exc()
        
    print("\n" + "="*60)
    print("3D HTML Brain View Pipeline (Target Only) Complete!")
    print("="*60)

if __name__ == '__main__':
    main()
