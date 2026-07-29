#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilateral fMRI p=0.005 Clustered Mask Generation (Bilateral Color_vs_Grey),
Multimodal (ERP / High Gamma 70-150Hz) single-channel OLS analysis,
Bilateral Electrode sequence numbering, AAL3 anatomical legend,
and Stacked Temporal Signal & 100-400ms Average Difference Profile Visualization (G1 style).
Saves all outputs in 'fmri_seeg_comparison' directory.
GLM Model includes ALL 4 Categories (Face, Object, Body, Place).
"""

import os
import re
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.ndimage as ndimage
import nibabel as nib
from scipy.signal import butter, filtfilt, hilbert
from pymatreader import read_mat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nilearn.plotting as nlp

# 基础路径定义
PROJECT_ROOT = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026'
FMRI_RESULT_DIR = '/home/lirui/liulab_project/ColorLocalizer_Exp/result'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'color_cognition_pipeline', 'analyse_0617', 'result', 'fmri_seeg_comparison')

# 被试列表
SUBJECTS = ['test001', 'test002', 'test003', 'test005', 'test006']
FMRI_MAPPING = {
    'test001': 'P001_20260108',
    'test003': 'P002_20260331',
    'test006': 'P003_20260627'
}

# 刺激条件：包含条件 0, 1, 2, 3, 4, 5, 6, 7 (全部 4 刺激类别参与 GLM)
GLM_CONDS = [0, 1, 2, 3, 4, 5, 6, 7]
# 条件所属的刺激类别
COND_TO_CAT = {
    0: 'face', 1: 'face',
    2: 'object', 3: 'object',
    4: 'body', 5: 'body',
    6: 'place', 7: 'place'
}

# 全局保存 test001 提取出的 fMRI 峰值，供 test002 和 test005 借用
TEST001_GLOBAL_PEAKS = {
    'peak_l': None, 't_l': None,
    'peak_r': None, 't_r': None
}

def parse_mni_coords(mni_str):
    """
    解析电极定位表中的MNI坐标。
    """
    if not isinstance(mni_str, str):
        return None
    nums = re.findall(r'[-+]?\d*\.\d+|\d+', mni_str)
    if len(nums) == 3:
        return [float(x) for x in nums]
    return None

def get_tstat_subbrick_index_by_label(nii_path, label):
    """
    通过 AFNI 3dinfo 工具匹配特定的 Tstat 子卷。
    """
    try:
        cmd = f"3dinfo -verb {nii_path}"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        stdout = result.stdout
        
        match = re.search(rf"sub-brick\s+#(\d+)\s+'{label}'", stdout, re.IGNORECASE)
        if match:
            idx = int(match.group(1))
            print(f"  [AFNI 3dinfo] 成功找到 {label} 子卷索引为: {idx}")
            return idx
    except Exception as e:
        print(f"  [AFNI 3dinfo] 查询子卷失败 ({e})，使用默认 Fallback")
    return 29

def generate_and_save_global_mask(nii_path, subject_out_dir, subject):
    """
    生成全局 Color_vs_Grey p=0.005 显著聚类 Mask (体素数 >= 20)。
    """
    label = 'Color_vs_Grey_GLT#0_Tstat'
    print(f"--- 重新生成全局 Color_vs_Grey 显著 Mask ---")
    t_idx = get_tstat_subbrick_index_by_label(nii_path, label)
    img = nib.load(nii_path)
    data = img.get_fdata()
    t_map = data[..., 0, t_idx]
    
    t_thresh = stats.t.ppf(1 - 0.005 / 2, df=623)
    binary_map = (t_map >= t_thresh).astype(int)
    
    structure = ndimage.generate_binary_structure(3, 1)  # NN=1
    labeled_array, num_features = ndimage.label(binary_map, structure=structure)
    
    unique, counts = np.unique(labeled_array, return_counts=True)
    sig_clusters = unique[(counts >= 20) & (unique > 0)]
    print(f"  [Color_vs_Grey] 找到 {num_features} 个连通域，体素数 >= 20 的有 {len(sig_clusters)} 个")
    
    clustered_mask = np.isin(labeled_array, sig_clusters).astype(np.int16)
    
    mask_name = f"{subject}_Color_vs_Grey_p005_mask.nii"
    mask_out_path = os.path.join(subject_out_dir, mask_name)
    nib.save(nib.Nifti1Image(clustered_mask, img.affine, img.header), mask_out_path)
    print(f"  显著性 Mask 成功保存至: {mask_out_path}")
    
    return clustered_mask, t_map, img.affine

def extract_peaks_inside_mask(clustered_mask, t_map, affine):
    """
    在显著 mask 内寻找左、右半球腹侧通路的激活峰值。
    """
    nx, ny, nz = t_map.shape
    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    voxel_coords = np.stack([I.flatten(), J.flatten(), K.flatten(), np.ones_like(I.flatten())], axis=1)
    mni_coords = (affine @ voxel_coords.T).T[:, :3]
    t_flat = t_map.flatten()
    mask_flat = clustered_mask.flatten()
    
    # 左脑腹侧通路范围
    mask_left = (
        (mask_flat == 1) &
        (mni_coords[:, 0] >= -50) & (mni_coords[:, 0] <= -15) &
        (mni_coords[:, 1] >= -95) & (mni_coords[:, 1] <= -35) &
        (mni_coords[:, 2] >= -30) & (mni_coords[:, 2] <= 0)
    )
    t_left = t_flat[mask_left]
    mni_left = mni_coords[mask_left]
    
    peak_left, t_val_left = None, None
    if len(t_left) > 0:
        max_l = np.argmax(t_left)
        peak_left = mni_left[max_l]
        t_val_left = t_left[max_l]
        print(f"  在 Mask 内找到左脑 Peak: MNI {np.round(peak_left, 2)}, T-stat: {t_val_left:.4f}")
    else:
        mask_fallback_l = (mask_flat == 1) & (mni_coords[:, 0] < 0)
        t_fallback_l = t_flat[mask_fallback_l]
        if len(t_fallback_l) > 0:
            max_fl = np.argmax(t_fallback_l)
            peak_left = mni_coords[mask_fallback_l][max_fl]
            t_val_left = t_fallback_l[max_fl]
            print(f"  [Fallback] 全左脑显著区 Peak: MNI {np.round(peak_left, 2)}, T: {t_val_left:.4f}")
        else:
            print("  左侧半球显著 Mask 区域完全无激活")
            
    # 右脑腹侧通路范围
    mask_right = (
        (mask_flat == 1) &
        (mni_coords[:, 0] >= 15) & (mni_coords[:, 0] <= 50) &
        (mni_coords[:, 1] >= -95) & (mni_coords[:, 1] <= -35) &
        (mni_coords[:, 2] >= -30) & (mni_coords[:, 2] <= 0)
    )
    t_right = t_flat[mask_right]
    mni_right = mni_coords[mask_right]
    
    peak_right, t_val_right = None, None
    if len(t_right) > 0:
        max_r = np.argmax(t_right)
        peak_right = mni_right[max_r]
        t_val_right = t_right[max_r]
        print(f"  在 Mask 内找到右脑 Peak: MNI {np.round(peak_right, 2)}, T-stat: {t_val_right:.4f}")
    else:
        mask_fallback_r = (mask_flat == 1) & (mni_coords[:, 0] > 0)
        t_fallback_r = t_flat[mask_fallback_r]
        if len(t_fallback_r) > 0:
            max_fr = np.argmax(t_fallback_r)
            peak_right = mni_coords[mask_fallback_r][max_fr]
            t_val_right = t_fallback_r[max_fr]
            print(f"  [Fallback] 全右脑显著区 Peak: MNI {np.round(peak_right, 2)}, T: {t_val_right:.4f}")
        else:
            print("  右侧半球显著 Mask 区域完全无激活")
            
    return (peak_left, t_val_left), (peak_right, t_val_right)

def extract_high_gamma_envelope(data_array, fs=500.0):
    """
    向量化提取 70-150 Hz High Gamma 功率包络。
    data_array: 形状为 (Cond, Rep, Ch, Time) 的 numpy 数组
    """
    nyq = 0.5 * fs
    low = 70.0 / nyq
    high = 150.0 / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, data_array, axis=-1)
    env = np.abs(hilbert(filtered, axis=-1))
    return env

def baseline_zscore(data, times, baseline_start=-200.0, baseline_end=0.0):
    """
    对 High Gamma 包络沿时间轴进行 Z-score 基线校正。
    data: [Cond, Rep, Ch, Time]
    """
    base_mask = (times >= baseline_start) & (times <= baseline_end)
    base_mean = np.mean(data[..., base_mask], axis=-1, keepdims=True)
    base_std = np.std(data[..., base_mask], axis=-1, keepdims=True) + 1e-8
    return (data - base_mean) / base_std

def load_and_preprocess_seeg_signals(mat_path, signal_type):
    """
    加载 SEEG 信号：
      - erp: 加载 task1_ERP_epoched.mat 的 'data'，直接使用。
      - hg: 加载 task1_TFA_epoched.mat 的 'data' (1-150Hz 宽带)，实时滤波和 Hilbert 变换提取 70-150Hz 包络，并做 Z-score 基线校正。
    """
    print(f"  [信号加载] 正在预处理 {signal_type.upper()} 信号数据...")
    data_dict = read_mat(mat_path)
    epoch = data_dict['epoch']
    data = epoch['data']  # [Cond, Rep, Ch, Time]
    labels = [lbl.strip().upper() for lbl in epoch['ch']['labels']]
    
    if 'time_ms' in epoch:
        time_ms = epoch['time_ms']
    else:
        n_pts = data.shape[-1]
        time_ms = np.arange(-500, -500 + n_pts * 2, 2)
        
    if signal_type == 'erp':
        # ERP (1-30 Hz) 直接在 100-400 ms 上求均值，用于 OLS
        time_mask = (time_ms >= 100) & (time_ms <= 400)
        data_window_avg = np.mean(data[..., time_mask], axis=-1)  # [Cond, Rep, Ch]
        data_full_time = data
    else:
        # High Gamma (70-150 Hz)
        hg_env = extract_high_gamma_envelope(data, fs=500.0)
        hg_z = baseline_zscore(hg_env, time_ms, baseline_start=-200.0, baseline_end=0.0)
        time_mask = (time_ms >= 100) & (time_ms <= 400)
        data_window_avg = np.mean(hg_z[..., time_mask], axis=-1)  # [Cond, Rep, Ch]
        data_full_time = hg_z
        
    return data_window_avg, data_full_time, time_ms, labels

def analyze_seeg_lfp_glm(data_window_avg, ch_idx):
    """
    一元多变量 OLS 回归模型: Value ~ Is_Color + C(Category)
    控制 face, object, body, place 四类刺激的混杂。
    """
    import statsmodels.formula.api as smf
    
    rows = []
    for c in GLM_CONDS:
        is_color = 1 if c in [0, 2, 4, 6] else 0
        cat = COND_TO_CAT[c]
        for r in range(data_window_avg.shape[1]):
            val = data_window_avg[c, r, ch_idx]
            rows.append({
                'Value': val,
                'Is_Color': is_color,
                'Category': cat
            })
            
    df_glm = pd.DataFrame(rows)
    
    try:
        model = smf.ols('Value ~ Is_Color + C(Category)', data=df_glm).fit()
        mean_diff = model.params['Is_Color']
        t_stat = model.tvalues['Is_Color']
        p_val = model.pvalues['Is_Color']
    except Exception as e:
        mean_diff, t_stat, p_val = np.nan, np.nan, np.nan
        
    return t_stat, p_val, mean_diff

def run_glm_stats_and_save(df_elec, data_window_avg, labels, subject_out_dir, subject, modality):
    """
    对指定模态的电极数据进行普通 GLM 统计，并保存到 CSV 中。
    """
    rows_matched = []
    
    for _, row in df_elec.iterrows():
        ch_name = row['Channel']
        mx, my, mz = row['MNI_X'], row['MNI_Y'], row['MNI_Z']
        dist = row['Distance_to_peak']
        peak_side = row['Aligned_Peak_Side']
        aal_loc = row.get('AAL3', 'Unknown')
        
        p_val = np.nan
        mean_diff = np.nan
        t_stat = np.nan
        
        if ch_name in labels:
            ch_idx = labels.index(ch_name)
            t_stat, p_val, mean_diff = analyze_seeg_lfp_glm(data_window_avg, ch_idx)
            
        rows_matched.append({
            'Channel': ch_name,
            'MNI_X': mx,
            'MNI_Y': my,
            'MNI_Z': mz,
            'Distance_to_peak_mm': dist,
            'Aligned_Peak_Side': peak_side,
            'AAL3': aal_loc,
            't_stat': t_stat,
            'p_value': p_val,
            'Color_minus_Gray': mean_diff,
            'Is_Significant': (p_val < 0.05) if not np.isnan(p_val) else False
        })
        
    df_results = pd.DataFrame(rows_matched)
    csv_path = os.path.join(subject_out_dir, f'{subject}_fmri_seeg_stat_results_{modality}.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"  {modality.upper()} GLM 结果保存至: {csv_path}")
    return df_results

def draw_3d_brain_with_sequence(df_results, subject_out_dir, subject, peak_l, peak_r, modality, scope_name):
    """
    绘制近域/全脑显著 3D 玻璃脑脑图。
    在显著通道（红蓝色球）右上方标注序号 1, 2, 3... (不挡点，不带圈)。
    同时在脑图正底端中央位置横向排放说明 Legend，防止遮挡。
    """
    # 显著通道过滤并排序
    df_sig = df_results[df_results['Is_Significant'] == True].copy()
    
    # 过滤范围
    if scope_name == 'near':
        df_display_all = df_results[df_results['Distance_to_peak_mm'] <= 35.0].copy()
        df_sig = df_sig[df_sig['Distance_to_peak_mm'] <= 35.0].copy()
    else:
        # all_sig 只画显著通道
        df_display_all = df_sig.copy()
        
    df_sig_sorted = df_sig.sort_values(by='Distance_to_peak_mm').reset_index(drop=True)
    
    coords_static = []
    size_static = []
    color_static = []
    node_border_colors = []
    node_border_widths = []
    node_styles = [] # 'sig' or 'unsig' or 'peak'
    node_indices = [] # 显著电极的序号，非显著电极或 Peak 为 -1
    
    # 1. 绘入金色的 fMRI Peak 点
    if peak_l is not None:
        coords_static.append(peak_l)
        size_static.append(250.0)
        color_static.append('#ffd700')
        node_border_colors.append('none')
        node_border_widths.append(0.0)
        node_styles.append('peak')
        node_indices.append(-1)
    if peak_r is not None:
        coords_static.append(peak_r)
        size_static.append(250.0)
        color_static.append('#ffd700')
        node_border_colors.append('none')
        node_border_widths.append(0.0)
        node_styles.append('peak')
        node_indices.append(-1)
        
    # 2. 绘入去重后的电极点
    for _, row in df_display_all.iterrows():
        mx, my, mz = row['MNI_X'], row['MNI_Y'], row['MNI_Z']
        p_val = row['p_value']
        diff_val = row['Color_minus_Gray']
        ch_name = row['Channel']
        
        coords_static.append([mx, my, mz])
        
        if np.isnan(p_val) or p_val >= 0.05:
            # 不显著点
            size_static.append(10.0)
            color_static.append('#dcdcdc')
            node_border_colors.append('none')
            node_border_widths.append(0.0)
            node_styles.append('unsig')
            node_indices.append(-1)
        else:
            # 显著点
            sig_factor = (0.05 - p_val) / 0.05
            sz = 30.0 + 120.0 * sig_factor
            size_static.append(sz)
            color_static.append('#ff3333' if diff_val > 0 else '#3333ff')
            
            # 查找在排好序的显著列表中的序号
            matched = df_sig_sorted[df_sig_sorted['Channel'] == ch_name]
            if len(matched) > 0:
                sig_idx = matched.index[0] # 0-indexed
                node_indices.append(sig_idx)
            else:
                node_indices.append(-1)
                
            if scope_name == 'all_sig':
                node_border_colors.append('black')
                node_border_widths.append(1.5)
            else:
                node_border_colors.append('none')
                node_border_widths.append(0.0)
            node_styles.append('sig')
            
    coords_static = np.array(coords_static)
    
    if len(coords_static) > 0:
        # 画布增加高度，为下方的横向说明留出足够空间
        plt.figure(figsize=(11, 9.2))
        display = nlp.plot_markers(
            node_values=[0.0] * len(coords_static),
            node_coords=coords_static,
            node_size=[0.0] * len(coords_static),
            display_mode='ortho',
            title=f"{subject} {modality.upper()} ({scope_name.upper()})",
            colorbar=False
        )
        
        # 遍历 scatter 子图 ax，分别在投影坐标上重叠绘制点和数字标注
        for ax_name, ax in display.axes.items():
            for i, coord in enumerate(coords_static):
                if ax_name == 'x':   # 矢状位 (Y, Z)
                    x_p, y_p = coord[1], coord[2]
                elif ax_name == 'y': # 冠状位 (X, Z)
                    x_p, y_p = coord[0], coord[2]
                else:                # 横断位 (X, Y)
                    x_p, y_p = coord[0], coord[1]
                    
                style = node_styles[i]
                
                # 绘制点
                if style == 'peak':
                    ax.ax.scatter(
                        x_p, y_p,
                        s=size_static[i],
                        c=color_static[i],
                        edgecolors='none',
                        alpha=0.85,
                        zorder=12
                    )
                elif style == 'sig':
                    # 显著点，如果是 all_sig 则包裹黑色虚线边框
                    edge_c = node_border_colors[i]
                    lw = node_border_widths[i]
                    ls = '--' if edge_c == 'black' else '-'
                    ax.ax.scatter(
                        x_p, y_p,
                        s=size_static[i],
                        c=color_static[i],
                        edgecolors=edge_c,
                        linewidths=lw,
                        linestyle=ls,
                        alpha=0.8,
                        zorder=10
                    )
                    
                    # 显著点标注序号：向右上方偏置 1.8mm，且不带任何 bbox 背景圈，靠右上对齐
                    sig_idx = node_indices[i]
                    if sig_idx != -1:
                        num_str = str(sig_idx + 1)
                        ax.ax.text(
                            x_p + 1.8, y_p + 1.8, num_str,
                            color='black', fontsize=7.5, fontweight='bold',
                            ha='left', va='bottom', zorder=15
                        )
                else:
                    # 不显著点 (只在 near 图中绘制)
                    if scope_name == 'near':
                        ax.ax.scatter(
                            x_p, y_p,
                            s=size_static[i],
                            c=color_static[i],
                            edgecolors='none',
                            alpha=0.6,
                            zorder=8
                        )
                        
        # 组装正底端中央位置横向排放的 Legend 说明 (每行并排 4 个通道，不遮挡侧边脑图)
        if len(df_sig_sorted) > 0:
            legend_chunks = []
            for idx, row in df_sig_sorted.iterrows():
                ch = row['Channel']
                aal = row.get('AAL3', 'Unknown')
                d = row['Distance_to_peak_mm']
                legend_chunks.append(f"{idx+1}:{ch}({aal}) d={d:.1f}mm")
                
            lines = []
            chunk_size = 4
            for i in range(0, len(legend_chunks), chunk_size):
                line = "   |   ".join(legend_chunks[i:i+chunk_size])
                lines.append(line)
            legend_text = "\n".join(lines)
            
            bbox_props = dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#dddddd", alpha=0.9, lw=0.5)
            plt.gcf().text(
                0.5, 0.01,
                f"Significant Channels ({modality.upper()}):\n{legend_text}",
                fontsize=6.0, color='black', ha='center', va='bottom', bbox=bbox_props
            )
            
        png_out = os.path.join(subject_out_dir, f"{subject}_fmri_seeg_3d_brain_{modality}_{scope_name}.png")
        plt.savefig(png_out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  脑图已成功生成并保存静态图至: {png_out}")
        return png_out
    else:
        plt.close()
        return None

def draw_sig_channels_profiles(df_results, data_full_time, time_ms, subject_out_dir, subject, modality):
    """
    垂直堆叠大图绘制：从上到下按距离升序排列，绘制每个显著通道的 Color vs Gray 时域波形 (折线+SEM)
    以及 100-400ms 均值对比散点条形图 (G1 style)。
    """
    df_sig = df_results[df_results['Is_Significant'] == True].copy()
    if len(df_sig) == 0:
        print(f"  [提示] 被试 {subject} 在 {modality.upper()} 模态下无显著差异位点，跳过堆叠剖面图绘制。")
        return None
        
    df_sig_sorted = df_sig.sort_values(by='Distance_to_peak_mm').reset_index(drop=True)
    K = len(df_sig_sorted)
    
    fig, axes = plt.subplots(K, 2, figsize=(13, 3.5 * K), squeeze=False)
    
    # 裁剪时域波形的时间范围 (-200 到 800 ms，与参考图一致)
    time_mask = (time_ms >= -200) & (time_ms <= 800)
    plot_times = time_ms[time_mask]
    
    # 100-400ms 的均值区间
    avg_mask = (time_ms >= 100) & (time_ms <= 400)
    
    # 统计用的条件划分：Color 为条件 0,2,4,6，Gray 为条件 1,3,5,7
    color_conds = [0, 2, 4, 6]
    gray_conds = [1, 3, 5, 7]
    
    y_label_str = "Amplitude (μV)" if modality == 'erp' else "Envelope Power (Z)"
    
    for idx, row in df_sig_sorted.iterrows():
        ch_name = row['Channel']
        aal = row.get('AAL3', 'Unknown')
        dist = row['Distance_to_peak_mm']
        p_val = row['p_value']
        
        # 寻找电极数据索引
        ch_labels_upper = [lbl.upper() for lbl in df_results['Channel']]
        ch_idx = ch_labels_upper.index(ch_name)
        
        # 提取 Color 条件试次：[4 * Rep, Time]
        color_trials_time = []
        for c in color_conds:
            color_trials_time.append(data_full_time[c, :, ch_idx, :])
        color_trials_time = np.concatenate(color_trials_time, axis=0)  # [N_trials, Time]
        
        # 提取 Gray 条件试次：[4 * Rep, Time]
        gray_trials_time = []
        for c in gray_conds:
            gray_trials_time.append(data_full_time[c, :, ch_idx, :])
        gray_trials_time = np.concatenate(gray_trials_time, axis=0)    # [N_trials, Time]
        
        # 计算时域平均值及标准误 SEM
        color_mean_time = np.mean(color_trials_time[..., time_mask], axis=0)
        color_sem_time = stats.sem(color_trials_time[..., time_mask], axis=0)
        
        gray_mean_time = np.mean(gray_trials_time[..., time_mask], axis=0)
        gray_sem_time = stats.sem(gray_trials_time[..., time_mask], axis=0)
        
        # -------------------------------------------------------------
        # 左侧子图：时域波形图
        # -------------------------------------------------------------
        ax_time = axes[idx, 0]
        # 绘制 Color (红线)
        ax_time.plot(plot_times, color_mean_time, color='#d62728', linewidth=2.0, label='Color')
        ax_time.fill_between(plot_times, color_mean_time - color_sem_time, color_mean_time + color_sem_time, color='#d62728', alpha=0.18)
        
        # 绘制 Gray (蓝线)
        ax_time.plot(plot_times, gray_mean_time, color='#1f77b4', linewidth=2.0, label='Gray')
        ax_time.fill_between(plot_times, gray_mean_time - gray_sem_time, gray_mean_time + gray_sem_time, color='#1f77b4', alpha=0.18)
        
        ax_time.axvline(0.0, color='gray', linestyle='--', linewidth=1.0)
        ax_time.grid(True, linestyle='--', alpha=0.35)
        ax_time.set_xlim(-200, 800)
        ax_time.set_ylabel(y_label_str, fontsize=9)
        ax_time.set_xlabel("Time (ms)", fontsize=8)
        ax_time.set_title(f"#{idx+1}: {ch_name} ({aal}) | d={dist:.1f}mm", fontsize=10, fontweight='bold')
        
        if idx == 0:
            ax_time.legend(loc='best', fontsize=8)
            
        # -------------------------------------------------------------
        # 右侧子图：100-400ms 均值对比散点条形图 (G1 style)
        # -------------------------------------------------------------
        ax_bar = axes[idx, 1]
        
        # 提取 100-400 ms 窗内各试次的均值
        color_trials_avg = np.mean(color_trials_time[..., avg_mask], axis=-1)
        gray_trials_avg = np.mean(gray_trials_time[..., avg_mask], axis=-1)
        
        # 添加轻微横向抖动以进行散点图可视化
        x_color = np.random.normal(0, 0.08, size=len(color_trials_avg))
        x_gray = np.random.normal(1, 0.08, size=len(gray_trials_avg))
        
        # 绘制散点
        ax_bar.scatter(x_color, color_trials_avg, color='#ff9999', alpha=0.4, s=15, edgecolors='none', zorder=2)
        ax_bar.scatter(x_gray, gray_trials_avg, color='#9ecae1', alpha=0.4, s=15, edgecolors='none', zorder=2)
        
        # 绘制条形图加误差线
        mean_c = np.mean(color_trials_avg)
        mean_g = np.mean(gray_trials_avg)
        sem_c = stats.sem(color_trials_avg)
        sem_g = stats.sem(gray_trials_avg)
        
        ax_bar.bar(
            [0, 1], [mean_c, mean_g],
            color=['#d62728', '#1f77b4'], alpha=0.62, width=0.4,
            yerr=[sem_c, sem_g],
            error_kw=dict(lw=1.5, capsize=4, capthick=1.5),
            zorder=3
        )
        
        ax_bar.set_xticks([0, 1])
        ax_bar.set_xticklabels(['Color', 'Gray'], fontsize=8)
        ax_bar.grid(axis='y', linestyle='--', alpha=0.35)
        ax_bar.set_title(f"100-400ms\np={p_val:.4f}", fontsize=9)
        
    plt.suptitle(f"Subject: {subject} | {modality.upper()} Significant Channels Profile", fontsize=13, y=0.99, fontweight='bold')
    plt.tight_layout()
    
    png_out = os.path.join(subject_out_dir, f"{subject}_{modality}_sig_channels_profiles.png")
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  显著电极信号剖面堆叠大图已成功生成并保存至: {png_out}")
    return png_out

def cleanup_old_files(subject_out_dir):
    """
    运行前清理被试目录下的旧文件：包括所有带 ttest、glm_nobody 后缀的 png 图件与数据，保持主目录干净利落。
    """
    if not os.path.exists(subject_out_dir):
        return
    for item in os.listdir(subject_out_dir):
        if 'ttest' in item.lower() or 'glm_nobody' in item.lower():
            p = os.path.join(subject_out_dir, item)
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
    print(f"  [清理] 已彻底清除 {subject_out_dir} 目录下的旧版 ttest/glm_nobody 冗余文件。")

def process_subject_multimodal(subject):
    print(f"\n==================== 开始处理被试: {subject} ====================")
    subj_out_dir = os.path.join(OUTPUT_DIR, subject)
    os.makedirs(subj_out_dir, exist_ok=True)
    
    # 1. 自动清除冗余旧版文件
    cleanup_old_files(subj_out_dir)
    
    use_test001_fmri = subject in ['test002', 'test005']
    
    # 2. 加载电极定位表，并进行 Channel 唯一性去重（防止冗余重复行绘制）
    loc_xlsx = os.path.join(PROJECT_ROOT, 'processed_data', subject, f'{subject}_ieegloc.xlsx')
    loc_tsv = os.path.join(PROJECT_ROOT, 'processed_data', subject, f'{subject}.tsv')
    if os.path.exists(loc_xlsx):
        df_loc = pd.read_excel(loc_xlsx)
    elif os.path.exists(loc_tsv):
        df_loc = pd.read_csv(loc_tsv, sep='\t')
    else:
        raise FileNotFoundError(f"未找到被试 {subject} 的定位表")
        
    # 对读取到的表格在 Channel 字段进行唯一性去重
    df_loc = df_loc.drop_duplicates(subset=['Channel']).reset_index(drop=True)
    print(f"  [定位表去重] 去重后共有 {len(df_loc)} 个唯一电极通道。")
        
    # 模糊匹配 AAL3 列
    aal_cols = [c for c in df_loc.columns if 'aal3' in c.lower()]
    aal_col = aal_cols[0] if len(aal_cols) > 0 else None
    
    # 3. 提取全局 fMRI 峰值
    if use_test001_fmri:
        print(f"  [借用模式] 借用 test001 对应的全局 fMRI 激活峰值坐标")
        peak_l = TEST001_GLOBAL_PEAKS['peak_l']
        t_l = TEST001_GLOBAL_PEAKS['t_l']
        peak_r = TEST001_GLOBAL_PEAKS['peak_r']
        t_r = TEST001_GLOBAL_PEAKS['t_r']
    else:
        fmri_id = FMRI_MAPPING[subject]
        nii_path = os.path.join(FMRI_RESULT_DIR, fmri_id, '1.data_preprocess', f'inside_final_stats.{fmri_id}.nii')
        clustered_mask, t_map, affine = generate_and_save_global_mask(nii_path, subj_out_dir, subject)
        (peak_l, t_l), (peak_r, t_r) = extract_peaks_inside_mask(clustered_mask, t_map, affine)
        
        if subject == 'test001':
            TEST001_GLOBAL_PEAKS['peak_l'] = peak_l
            TEST001_GLOBAL_PEAKS['t_l'] = t_l
            TEST001_GLOBAL_PEAKS['peak_r'] = peak_r
            TEST001_GLOBAL_PEAKS['t_r'] = t_r
            
    # 4. 构建电极位置与其同侧 Peak 距离
    electrodes = []
    for _, row in df_loc.iterrows():
        ch_name = str(row['Channel']).strip().upper()
        mni_str = row['MNI']
        coords = parse_mni_coords(mni_str)
        aal_loc = str(row[aal_col]).strip() if aal_col and not pd.isna(row[aal_col]) else 'Unknown'
        
        if coords is not None:
            mx = coords[0]
            if mx < 0:
                target_peak = peak_l if peak_l is not None else peak_r
                peak_label = 'Left_Peak'
            else:
                target_peak = peak_r if peak_r is not None else peak_l
                peak_label = 'Right_Peak'
                
            dist = np.linalg.norm(np.array(coords) - target_peak) if target_peak is not None else 999.0
            electrodes.append({
                'Channel': ch_name,
                'MNI_X': coords[0],
                'MNI_Y': coords[1],
                'MNI_Z': coords[2],
                'Distance_to_peak': dist,
                'Aligned_Peak_Side': peak_label,
                'AAL3': aal_loc
            })
    df_elec = pd.DataFrame(electrodes)
    
    # 5. 加载 ERP 与 TFA 信号
    mat_erp = os.path.join(PROJECT_ROOT, 'processed_data', subject, 'task1_ERP_epoched.mat')
    mat_tfa = os.path.join(PROJECT_ROOT, 'processed_data', subject, 'task1_TFA_epoched.mat')
    
    erp_window_avg, erp_full_time, erp_times, erp_labels = load_and_preprocess_seeg_signals(mat_erp, 'erp')
    hg_window_avg, hg_full_time, hg_times, hg_labels = load_and_preprocess_seeg_signals(mat_tfa, 'hg')
    
    subj_stats = {}
    
    # 6. 分 ERP 和 High Gamma 模态进行 GLM 统计差异提取与绘图
    for modality in ['erp', 'hg']:
        print(f"\n--- 进行模态统计与图件渲染: {modality.upper()} ---")
        window_avg = erp_window_avg if modality == 'erp' else hg_window_avg
        full_time = erp_full_time if modality == 'erp' else hg_full_time
        times_vec = erp_times if modality == 'erp' else hg_times
        labels_list = erp_labels if modality == 'erp' else hg_labels
        
        # 回归分析并保存 CSV
        df_results = run_glm_stats_and_save(df_elec, window_avg, labels_list, subj_out_dir, subject, modality)
        
        # 绘制近域脑图 (3D玻璃脑，标注电极序号，底端横排 Legend)
        draw_3d_brain_with_sequence(df_results, subj_out_dir, subject, peak_l, peak_r, modality, 'near')
        
        # 绘制全脑显著图 (3D玻璃脑，标注电极序号，底端横排 Legend，显著通道虚线，不显著点隐藏)
        draw_3d_brain_with_sequence(df_results, subj_out_dir, subject, peak_l, peak_r, modality, 'all_sig')
        
        # 绘制新增的显著通道时域折线波形+散点图剖面大图 (G1 style)
        draw_sig_channels_profiles(df_results, full_time, times_vec, subj_out_dir, subject, modality)
        
        # 收集数值以便生成最终报告
        near_tested = len(df_results[df_results['Distance_to_peak_mm'] <= 35.0])
        sig_count = len(df_results[(df_results['Distance_to_peak_mm'] <= 35.0) & (df_results['Is_Significant'])])
        all_sig_count = len(df_results[df_results['Is_Significant'] == True])
        
        subj_stats[modality] = {
            'near_tested': near_tested,
            'sig_count': sig_count,
            'all_sig_count': all_sig_count
        }
        
    subj_stats['peak_left'] = list(peak_l) if peak_l is not None else None
    subj_stats['peak_left_t'] = t_l
    subj_stats['peak_right'] = list(peak_r) if peak_r is not None else None
    subj_stats['peak_right_t'] = t_r
    
    return {
        'subject': subject,
        'stats': subj_stats,
        'borrowed': use_test001_fmri
    }

def main():
    print("======================================================================")
    print("fMRI 全局 Color Mask 与多模态 (ERP/High Gamma) GLM (包含Body) 联合分析管线")
    print("======================================================================")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_results = []
    
    for subject in SUBJECTS:
        try:
            res = process_subject_multimodal(subject)
            summary_results.append(res)
        except Exception as e:
            print(f"[错误] 被试 {subject} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            
    # 输出汇总 Markdown 报告 (包含 Body 回归模型)
    print("\n--- 正在生成多模态联合分析的 Markdown 汇总报告 ---")
    summary_md_path = os.path.join(OUTPUT_DIR, 'fmri_seeg_joint_analysis_report.md')
    
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write("# fMRI 全局显著聚类与 SEEG 联合分析报告 (GLM_With_Body & Multimodal)\n\n")
        f.write("此报告总结了 5 位被试在包含 **Body 类别** (即 Face, Object, Body, Place 共 8 条件) 的普通一般线性模型 (GLM) 下，**ERP** 与 **High Gamma (70-150Hz)** 两电信号模态下的激活对齐与显著电极响应统计差异。\n")
        f.write("注：去重排除了定位表中的冗余多行。脑图电极序号已做偏置优化，无圈且绝不挡点，Legend 均移至底部长条，避免遮挡 3D 脑图投影。\n\n")
        
        f.write("## 1. fMRI 双侧激活峰值坐标 (全局 Color_vs_Grey p=0.005 聚类)\n\n")
        f.write("| 被试 ID | 左脑显著 Peak MNI & T | 右脑显著 Peak MNI & T | 激活定位模式 |\n")
        f.write("| --- | --- | --- | --- |\n")
        for r in summary_results:
            st = r['stats']
            p_l_str = f"`[{st['peak_left'][0]:.1f}, {st['peak_left'][1]:.1f}, {st['peak_left'][2]:.1f}]` (T={st['peak_left_t']:.2f})" if st['peak_left'] else "N/A"
            p_r_str = f"`[{st['peak_right'][0]:.1f}, {st['peak_right'][1]:.1f}, {st['peak_right'][2]:.1f}]` (T={st['peak_right_t']:.2f})" if st['peak_right'] else "N/A"
            mode_str = "借用 test001 峰值" if r['borrowed'] else "个体化 fMRI 峰值"
            f.write(f"| {r['subject']} | {p_l_str} | {p_r_str} | {mode_str} |\n")
            
        f.write("\n## 2. SEEG 差异显著性通道统计 (包含 Body 回归)\n\n")
        f.write("| 被试 ID | 模态 | 临近唯一电极数 (d < 35mm) | 临近显著通道数 | 全脑显著通道数 (p < 0.05) |\n")
        f.write("| --- | --- | ---: | ---: | ---: |\n")
        for r in summary_results:
            st = r['stats']
            # ERP
            f.write(f"| {r['subject']} | ERP | {st['erp']['near_tested']} | {st['erp']['sig_count']} | {st['erp']['all_sig_count']} |\n")
            # High Gamma
            f.write(f"| {r['subject']} | High Gamma | {st['hg']['near_tested']} | {st['hg']['sig_count']} | {st['hg']['all_sig_count']} |\n")
            
        f.write("\n## 3. 结论与科学发现\n\n")
        for r in summary_results:
            st = r['stats']
            f.write(f"### 👤 被试 {r['subject']}\n")
            if r['borrowed']:
                f.write("> **说明**：该被试借用了 test001 的全局激活峰值坐标进行距离算子对齐。\n\n")
            f.write(f"*   **ERP 模态**：近域 $d < 35\\text{{mm}}$ 内有 **{st['erp']['sig_count']}** 个通道具有显著的 Color-Gray 差异；全脑共检测到 **{st['erp']['all_sig_count']}** 个显著通道。\n")
            f.write(f"*   **High Gamma 模态**：近域 $d < 35\\text{{mm}}$ 内有 **{st['hg']['sig_count']}** 个通道具有显著的 Color-Gray 差异；全脑共检测到 **{st['hg']['all_sig_count']}** 个显著通道。\n\n")
            
    print(f"多模态联合报告已成功保存至: {summary_md_path}")
    print("======================================================================")
    print("多模态全局包含 Body 分析管线运行完毕！")
    print("======================================================================")

if __name__ == '__main__':
    main()
