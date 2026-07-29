#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilateral fMRI p=0.005 Clustered Mask Generation (4 Categories: Face, Object, Body, Place),
Multimodal (ERP / High Gamma 70-150Hz) single-channel OLS analysis,
and Mapped 3D Static Glass Brain Visualization.
Saves all outputs in 'fmri_seeg_comparison_by_category_multimodal' directory.
"""

import os
import re
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'color_cognition_pipeline', 'analyse_0617', 'result', 'fmri_seeg_comparison_by_category_multimodal')

# 被试列表
SUBJECTS = ['test001', 'test002', 'test003', 'test005', 'test006']
FMRI_MAPPING = {
    'test001': 'P001_20260108',
    'test003': 'P002_20260331',
    'test006': 'P003_20260627'
}

# 4大类别定义与参数映射 (Body 使用全局 Color 对比作为其 fMRI 基准)
CATEGORIES = ['face', 'object', 'body', 'place']
CAT_LABELS = {
    'face': 'fC_vs_fG_GLT#0_Tstat',
    'object': 'oC_vs_oG_GLT#0_Tstat',
    'body': 'Color_vs_Grey_GLT#0_Tstat',  # Body 类别使用全局 Color-Gray Tstat
    'place': 'pC_vs_pG_GLT#0_Tstat'
}
CAT_CONDS = {
    'face': [0, 1],
    'object': [2, 3],
    'body': [4, 5],
    'place': [6, 7]
}

# 全局保存 test001 提取出的分刺激/多类别激活峰值，供 test002 和 test005 借用
TEST001_CAT_PEAKS = {
    'face': {'peak_l': None, 't_l': None, 'peak_r': None, 't_r': None},
    'object': {'peak_l': None, 't_l': None, 'peak_r': None, 't_r': None},
    'body': {'peak_l': None, 't_l': None, 'peak_r': None, 't_r': None},
    'place': {'peak_l': None, 't_l': None, 'peak_r': None, 't_r': None}
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
        
    if 'fC_vs_fG' in label:
        return 20
    elif 'oC_vs_oG' in label:
        return 23
    elif 'Color_vs_Grey' in label:
        return 29
    else:
        return 26

def generate_and_save_category_mask(nii_path, subject_out_dir, subject, category):
    """
    针对特定被试和刺激类别生成 p=0.005 (T>=2.8124) 且体素数 >= 20 的 3D 聚类 Mask。
    """
    label = CAT_LABELS[category]
    print(f"--- 重新生成 {category.upper()} 类别显著 Mask ---")
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
    print(f"  [{category.upper()}] 找到 {num_features} 个连通域，体素数 >= 20 的有 {len(sig_clusters)} 个")
    
    clustered_mask = np.isin(labeled_array, sig_clusters).astype(np.int16)
    
    mask_nii = nib.Nifti1Image(clustered_mask, img.affine, img.header)
    mask_nii.set_data_dtype(np.int16)
    
    mask_name = f"{subject}_{category}_p005_mask.nii"
    mask_out_path = os.path.join(subject_out_dir, mask_name)
    nib.save(mask_nii, mask_out_path)
    print(f"  显著性 Mask 成功保存至: {mask_out_path}")
    
    return clustered_mask, t_map, img.affine

def extract_peaks_inside_mask(clustered_mask, t_map, affine, category):
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
        print(f"  在 {category.upper()} Mask 内找到左脑 Peak: MNI {np.round(peak_left, 2)}, T-stat: {t_val_left:.4f}")
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
        print(f"  在 {category.upper()} Mask 内找到右脑 Peak: MNI {np.round(peak_right, 2)}, T-stat: {t_val_right:.4f}")
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
    
    # 沿最后一个维度 (Time) 执行带通滤波
    filtered = filtfilt(b, a, data_array, axis=-1)
    
    # 沿最后一个维度执行 Hilbert 变换
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
    加载并预处理 SEEG 信号：
      - erp: 加载 task1_ERP_epoched.mat 的 'data'，直接取平均值。
      - hg: 加载 task1_TFA_epoched.mat 的 'data' (1-150Hz 宽带)，滤波 + Hilbert 变换提取 70-150Hz 能量，并做 Z-score 基线校正。
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
        # ERP (1-30 Hz) 直接在 100-400 ms 上求均值
        time_mask = (time_ms >= 100) & (time_ms <= 400)
        data_window_avg = np.mean(data[..., time_mask], axis=-1)  # [Cond, Rep, Ch]
    else:
        # High Gamma (70-150 Hz)
        # 1. 提取包络
        hg_env = extract_high_gamma_envelope(data, fs=500.0)
        # 2. 基线校正 (以 -200 到 0ms 为 baseline)
        hg_z = baseline_zscore(hg_env, time_ms, baseline_start=-200.0, baseline_end=0.0)
        # 3. 提取 100-400 ms
        time_mask = (time_ms >= 100) & (time_ms <= 400)
        data_window_avg = np.mean(hg_z[..., time_mask], axis=-1)  # [Cond, Rep, Ch]
        
    return data_window_avg, labels

def analyze_seeg_lfp_category_glm(data_window_avg, ch_idx, category):
    """
    OLS 回归模型: Value ~ Is_Color
    针对特定类别下（Face, Object, Body, Place）Color vs Gray 进行 OLS 回归。
    """
    import statsmodels.formula.api as smf
    conds = CAT_CONDS[category]
    
    rows = []
    for c in conds:
        is_color = 1 if c in [0, 2, 4, 6] else 0
        for r in range(data_window_avg.shape[1]):
            val = data_window_avg[c, r, ch_idx]
            rows.append({
                'Value': val,
                'Is_Color': is_color
            })
            
    df_glm = pd.DataFrame(rows)
    
    try:
        model = smf.ols('Value ~ Is_Color', data=df_glm).fit()
        mean_diff = model.params['Is_Color']
        t_stat = model.tvalues['Is_Color']
        p_val = model.pvalues['Is_Color']
    except Exception as e:
        mean_diff, t_stat, p_val = np.nan, np.nan, np.nan
        
    return t_stat, p_val, mean_diff

def run_stats_and_save_category(df_elec, data_window_avg, labels, subject_out_dir, subject, category, modality):
    """
    对指定模态和类别的电极数据进行统计，并保存到 CSV 中。
    """
    rows_matched = []
    
    for _, row in df_elec.iterrows():
        ch_name = row['Channel']
        mx, my, mz = row['MNI_X'], row['MNI_Y'], row['MNI_Z']
        dist = row['Distance_to_peak']
        peak_side = row['Aligned_Peak_Side']
        
        p_val = np.nan
        mean_diff = np.nan
        t_stat = np.nan
        
        if ch_name in labels:
            ch_idx = labels.index(ch_name)
            t_stat, p_val, mean_diff = analyze_seeg_lfp_category_glm(data_window_avg, ch_idx, category)
            
        rows_matched.append({
            'Channel': ch_name,
            'MNI_X': mx,
            'MNI_Y': my,
            'MNI_Z': mz,
            'Distance_to_peak_mm': dist,
            'Aligned_Peak_Side': peak_side,
            't_stat': t_stat,
            'p_value': p_val,
            'Color_minus_Gray': mean_diff,
            'Is_Significant': (p_val < 0.05) if not np.isnan(p_val) else False
        })
        
    df_results = pd.DataFrame(rows_matched)
    csv_path = os.path.join(subject_out_dir, f'{subject}_{category}_{modality}_seeg_stat_results.csv')
    df_results.to_csv(csv_path, index=False)
    return df_results

def draw_category_visualizations(df_results, subject_out_dir, subject, peak_l, peak_r, category, modality):
    """
    绘制近域分类电极图。要求与之前的 near 图一致：
      - 仅保留距离 peak < 35mm 的电极。
      - p >= 0.05 绘制为小灰色点。
      - p < 0.05 根据差异正负着红蓝色，且大小与显著度正相关。
    """
    coords_static = []
    size_static = []
    color_static = []
    
    # 1. 绘入当前类别的金色 fMRI Peak 点
    if peak_l is not None:
        coords_static.append(peak_l)
        size_static.append(250.0)
        color_static.append('#ffd700')
    if peak_r is not None:
        coords_static.append(peak_r)
        size_static.append(250.0)
        color_static.append('#ffd700')
        
    # 2. 绘入电极点
    for _, row in df_results.iterrows():
        dist = row['Distance_to_peak_mm']
        if dist > 35.0:
            continue
            
        mx, my, mz = row['MNI_X'], row['MNI_Y'], row['MNI_Z']
        p_val = row['p_value']
        diff_val = row['Color_minus_Gray']
        
        if np.isnan(p_val):
            coords_static.append([mx, my, mz])
            size_static.append(10.0)
            color_static.append('#dcdcdc')
            continue
            
        coords_static.append([mx, my, mz])
        
        if p_val >= 0.05:
            size_static.append(10.0)
            color_static.append('#dcdcdc')
        else:
            sig_factor = (0.05 - p_val) / 0.05
            sz = 30.0 + 120.0 * sig_factor
            size_static.append(sz)
            color_static.append('#ff3333' if diff_val > 0 else '#3333ff')
            
    coords_static = np.array(coords_static)
    
    if len(coords_static) > 0:
        plt.figure(figsize=(10, 8))
        display = nlp.plot_markers(
            node_values=[0.0] * len(coords_static),
            node_coords=coords_static,
            node_size=[0.0] * len(coords_static),
            display_mode='ortho',
            title=f"{subject} {category.upper()} ({modality.upper()} NEAR)",
            colorbar=False
        )
        
        # 自定义 scatter 覆盖
        for ax_name, ax in display.axes.items():
            for i, coord in enumerate(coords_static):
                if ax_name == 'x':   # 矢状位 (Y, Z)
                    x_p, y_p = coord[1], coord[2]
                elif ax_name == 'y': # 冠状位 (X, Z)
                    x_p, y_p = coord[0], coord[2]
                else:                # 横断位 (X, Y)
                    x_p, y_p = coord[0], coord[1]
                    
                ax.ax.scatter(
                    x_p, y_p,
                    s=size_static[i],
                    c=color_static[i],
                    edgecolors='none',
                    alpha=0.8,
                    zorder=10
                )
                
        png_out = os.path.join(subject_out_dir, f"{subject}_{category}_{modality}_near.png")
        plt.savefig(png_out, dpi=300, bbox_inches='tight')
        plt.close()
        return png_out
    else:
        plt.close()
        return None

def process_subject_multimodal(subject):
    print(f"\n==================== 开始处理被试: {subject} ====================")
    subj_out_dir = os.path.join(OUTPUT_DIR, subject)
    os.makedirs(subj_out_dir, exist_ok=True)
    
    use_test001_fmri = subject in ['test002', 'test005']
    
    # 1. 加载电极定位表
    loc_xlsx = os.path.join(PROJECT_ROOT, 'processed_data', subject, f'{subject}_ieegloc.xlsx')
    loc_tsv = os.path.join(PROJECT_ROOT, 'processed_data', subject, f'{subject}.tsv')
    if os.path.exists(loc_xlsx):
        df_loc = pd.read_excel(loc_xlsx)
    elif os.path.exists(loc_tsv):
        df_loc = pd.read_csv(loc_tsv, sep='\t')
    else:
        raise FileNotFoundError(f"未找到被试 {subject} 的定位表")
        
    # 2. 预先加载并提取双模态 SEEG 信号能量特征
    mat_erp = os.path.join(PROJECT_ROOT, 'processed_data', subject, 'task1_ERP_epoched.mat')
    mat_tfa = os.path.join(PROJECT_ROOT, 'processed_data', subject, 'task1_TFA_epoched.mat')
    
    erp_data, erp_ch_labels = load_and_preprocess_seeg_signals(mat_erp, 'erp')
    hg_data, hg_ch_labels = load_and_preprocess_seeg_signals(mat_tfa, 'hg')
    
    subj_stats = {}
    
    # 3. 循环 4 大类别
    for category in CATEGORIES:
        print(f"\n  >> 正在处理刺激类别: {category.upper()}")
        
        # 3a. 定位 fMRI 显著 peak 坐标
        if use_test001_fmri:
            print(f"    [借用模式] 借用 test001 在 {category.upper()} 类别下的激活峰值坐标")
            peak_l = TEST001_CAT_PEAKS[category]['peak_l']
            t_l = TEST001_CAT_PEAKS[category]['t_l']
            peak_r = TEST001_CAT_PEAKS[category]['peak_r']
            t_r = TEST001_CAT_PEAKS[category]['t_r']
        else:
            fmri_id = FMRI_MAPPING[subject]
            nii_path = os.path.join(FMRI_RESULT_DIR, fmri_id, '1.data_preprocess', f'inside_final_stats.{fmri_id}.nii')
            clustered_mask, t_map, affine = generate_and_save_category_mask(nii_path, subj_out_dir, subject, category)
            (peak_l, t_l), (peak_r, t_r) = extract_peaks_inside_mask(clustered_mask, t_map, affine, category)
            
            if subject == 'test001':
                TEST001_CAT_PEAKS[category]['peak_l'] = peak_l
                TEST001_CAT_PEAKS[category]['t_l'] = t_l
                TEST001_CAT_PEAKS[category]['peak_r'] = peak_r
                TEST001_CAT_PEAKS[category]['t_r'] = t_r
                
        # 3b. 计算电极与该类别 Peak 的距离
        electrodes = []
        for _, row in df_loc.iterrows():
            ch_name = str(row['Channel']).strip().upper()
            mni_str = row['MNI']
            coords = parse_mni_coords(mni_str)
            if coords is not None:
                mx = coords[0]
                # 对齐到同侧
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
                    'Aligned_Peak_Side': peak_label
                })
        df_elec = pd.DataFrame(electrodes)
        
        subj_stats[category] = {
            'peak_left': list(peak_l) if peak_l is not None else None,
            'peak_left_t': t_l,
            'peak_right': list(peak_r) if peak_r is not None else None,
            'peak_right_t': t_r
        }
        
        # 3c. 分模态进行回归与画图
        for modality in ['erp', 'hg']:
            data_window_avg = erp_data if modality == 'erp' else hg_data
            ch_labels = erp_ch_labels if modality == 'erp' else hg_ch_labels
            
            # 回归统计
            df_results = run_stats_and_save_category(
                df_elec, data_window_avg, ch_labels, subj_out_dir, subject, category, modality
            )
            
            # 画图
            draw_category_visualizations(df_results, subj_out_dir, subject, peak_l, peak_r, category, modality)
            
            # 记录数据
            near_tested = len(df_results[df_results['Distance_to_peak_mm'] < 25.0])
            sig_count = len(df_results[(df_results['Distance_to_peak_mm'] < 25.0) & (df_results['Is_Significant'])])
            all_sig_count = len(df_results[df_results['Is_Significant'] == True])
            
            subj_stats[category][f'{modality}_near_tested'] = near_tested
            subj_stats[category][f'{modality}_sig_count'] = sig_count
            subj_stats[category][f'{modality}_all_sig_count'] = all_sig_count
            
    return {
        'subject': subject,
        'stats': subj_stats,
        'borrowed': use_test001_fmri
    }

def main():
    print("======================================================================")
    print("fMRI 多模态 (ERP/High Gamma) & 4类别对齐差异分析")
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
            
    # 输出汇总 Markdown 报告
    print("\n--- 正在生成多模态联合分析的 Markdown 汇总报告 ---")
    summary_md_path = os.path.join(OUTPUT_DIR, 'fmri_seeg_joint_analysis_report_multimodal.md')
    
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write("# fMRI 多模态 (ERP & High Gamma) 与四类别联合分析报告\n\n")
        f.write("此报告总结了 5 位被试在四个类别（Face、Object、Body、Place）下，**ERP** 与 **High Gamma (70-150Hz)** 两脑电信号模态下的激活对齐与显著电极响应统计差异。\n")
        f.write("注：`test002` 与 `test005` 分类别直接借用了 `test001` 的激活峰值坐标。由于 fMRI 实验中无单独 Body 条件，其对应峰值基准均借用全局 `Color_vs_Grey` 峰值坐标。项目结果保存在独立的 `category_multimodal` 文件夹下。\n\n")
        
        f.write("## 1. fMRI 双侧激活峰值坐标 (p=0.005 聚类)\n\n")
        f.write("| 被试 ID | 刺激类别 | 左脑显著 Peak MNI & T | 右脑显著 Peak MNI & T | 激活定位模式 |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for r in summary_results:
            for cat in CATEGORIES:
                st = r['stats'][cat]
                p_l_str = f"`[{st['peak_left'][0]:.1f}, {st['peak_left'][1]:.1f}, {st['peak_left'][2]:.1f}]` (T={st['peak_left_t']:.2f})" if st['peak_left'] else "N/A"
                p_r_str = f"`[{st['peak_right'][0]:.1f}, {st['peak_right'][1]:.1f}, {st['peak_right'][2]:.1f}]` (T={st['peak_right_t']:.2f})" if st['peak_right'] else "N/A"
                mode_str = "借用 test001 峰值" if r['borrowed'] else "个体化 fMRI 峰值"
                f.write(f"| {r['subject']} | {cat.upper()} | {p_l_str} | {p_r_str} | {mode_str} |\n")
                
        f.write("\n## 2. SEEG 差异显著性通道统计 (ERP vs High Gamma)\n\n")
        f.write("| 被试 ID | 刺激类别 | 模态 | 临近电极数 (d < 25mm) | 临近显著通道数 | 全脑显著通道数 (p < 0.05) |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: |\n")
        for r in summary_results:
            for cat in CATEGORIES:
                st = r['stats'][cat]
                # ERP
                f.write(f"| {r['subject']} | {cat.upper()} | ERP | {st['erp_near_tested']} | {st['erp_sig_count']} | {st['erp_all_sig_count']} |\n")
                # High Gamma
                f.write(f"| {r['subject']} | {cat.upper()} | High Gamma | {st['hg_near_tested']} | {st['hg_sig_count']} | {st['hg_all_sig_count']} |\n")
                
        f.write("\n## 3. 结论与科学发现\n\n")
        for r in summary_results:
            f.write(f"### 👤 被试 {r['subject']}\n")
            if r['borrowed']:
                f.write("> **说明**：该被试分类别借用了 test001 的 Color Patch Peak 坐标进行距离算子对齐。\n\n")
            for cat in CATEGORIES:
                st = r['stats'][cat]
                f.write(f"#### 刺激类别: {cat.upper()}\n")
                # ERP
                f.write(f"*   **ERP 模态**：临近有 **{st['erp_sig_count']}** 个显著通道；全脑共有 **{st['erp_all_sig_count']}** 个显著通道。\n")
                # High Gamma
                f.write(f"*   **High Gamma 模态**：临近有 **{st['hg_sig_count']}** 个显著通道；全脑共有 **{st['hg_all_sig_count']}** 个显著通道。\n")
            f.write("\n")
            
    print(f"多模态联合报告已成功保存至: {summary_md_path}")
    print("======================================================================")
    print("多模态四类别分析管线运行完毕！")
    print("======================================================================")

if __name__ == '__main__':
    main()
