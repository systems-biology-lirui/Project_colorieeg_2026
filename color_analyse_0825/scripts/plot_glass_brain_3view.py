# -*- coding: utf-8 -*-
"""
===============================================================================
脚本名称: plot_glass_brain_3view.py
功能说明:
  1. 读取 8 位被试 (sub001~sub008) 的三维 MNI 真实空间坐标。
  2. 读取 C04 全频段色彩效应汇总表，筛选总体显著通道 (is_significant == 1)。
  3. 加载标准 MNI 玻璃脑皮层表面 (fsaverage5)。
  4. 为每个生理频段绘制学术级 3D 玻璃脑三视图 (矢状侧视、冠状后视、轴向俯视):
     - 每个频段使用专属标志色 (高伽马:陶土红, 低伽马:琥珀金, Beta:翡翠绿, Alpha:皇家紫, Theta:湖水蓝, Delta:深海蓝)
     - 彩色 > 灰色 (Color > Gray): 实心圆点
     - 灰色 > 彩色 (Gray > Color): 空心虚线圆
     - 点的大小严格与 |Color - Gray| 绝对差值成比例 (效应越强点越大)
  5. 自动输出高清科研图片至: color_analyse_0825/result/figures/glass_brain/
===============================================================================
"""

import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import nilearn.datasets as ndata
import nilearn.surface as nsurf

# -----------------------------------------------------------------------------
# 1. 核心参数与路径配置 (平铺直观，可随时修改调节)
# -----------------------------------------------------------------------------
cfg = {
    # 频段名称与专属科研配色 (高对比度、色盲友好)
    'bands': ['High_Gamma', 'Low_Gamma', 'Beta', 'Alpha', 'Theta', 'Delta'],
    'band_colors': {
        'High_Gamma': '#D9381E',  # 暖陶土红 (70-140 Hz)
        'Low_Gamma':  '#E67E22',  # 暖琥珀金 (30-60 Hz)
        'Beta':       '#27AE60',  # 森林翡翠绿 (13-30 Hz)
        'Alpha':      '#8E44AD',  # 皇家紫 (8-13 Hz)
        'Theta':      '#16A085',  # 湖水青蓝 (4-8 Hz)
        'Delta':      '#2980B9',  # 深海靛蓝 (1-4 Hz)
    },
    'band_freq_labels': {
        'High_Gamma': '70–140 Hz',
        'Low_Gamma':  '30–60 Hz',
        'Beta':       '13–30 Hz',
        'Alpha':      '8–13 Hz',
        'Theta':      '4–8 Hz',
        'Delta':      '1–4 Hz',
    },
    # 点的大小映射参数 (像素面积)
    'min_marker_size': 45,
    'max_marker_size': 260,
    'scale_ref_diff':  1.5,    # 效应量参考值 (dB)
    
    # 路径配置
    'loc_dir':    'color_analyse_0825/metadata/ieeg_location',
    'table_file': 'color_analyse_0825/metadata/C04_全频段色彩统一效应汇总.csv',
    'out_dir':    'color_analyse_0825/result/figures/glass_brain',
    
    # 图像分辨率
    'dpi': 150
}

# 字体支持 (确保中文正常显示)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs(cfg['out_dir'], exist_ok=True)

print("=" * 70)
print("  【0825 玻璃脑电极三视图绘制流水线】")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2. 读取各被试三维 MNI 空间坐标 (构建 lookup 词典)
# -----------------------------------------------------------------------------
print("\n>>> 正在加载被试三维 MNI 坐标表 ...")
loc_map = {}
subjects = [f'sub00{i}' for i in range(1, 9)]

for sub in subjects:
    loc_file = os.path.join(cfg['loc_dir'], f'{sub}_ieegloc.xlsx')
    if not os.path.exists(loc_file):
        print(f"  [-] 未找到 {sub} 定位文件: {loc_file}")
        continue
    
    df_loc = pd.read_excel(loc_file)
    valid_count = 0
    for _, row in df_loc.iterrows():
        ch_name = str(row['Channel']).strip()
        mni_val = row.get('MNI', None)
        if pd.notnull(mni_val):
            try:
                coords = ast.literal_eval(str(mni_val).strip())
                loc_map[(sub, ch_name)] = coords
                valid_count += 1
            except Exception:
                pass
    print(f"  [+] {sub}: 成功解析 {valid_count} 个电极接触点的 3D MNI 坐标")

# -----------------------------------------------------------------------------
# 3. 读取 C04 汇总表并关联坐标
# -----------------------------------------------------------------------------
print(f"\n>>> 正在读取 C04 色彩效应汇总表: {cfg['table_file']} ...")
df_summary = pd.read_csv(cfg['table_file'])

# 仅筛选总体显著通道 (is_significant == 1)
sig_df = df_summary[df_summary['is_significant'] == 1].copy()
print(f"  [+] 总体显著通道-频段记录数: {len(sig_df)} 条")

# 匹配坐标
sig_df['coords'] = sig_df.apply(lambda r: loc_map.get((r['subject'], str(r['channel']).strip())), axis=1)
has_coords_mask = sig_df['coords'].notnull()
valid_sig_df = sig_df[has_coords_mask].copy()

n_missing = len(sig_df) - len(valid_sig_df)
print(f"  [+] 成功匹配 MNI 坐标并可上脑绘制的位点数: {len(valid_sig_df)} 个")
print(f"  [*] 未绘制位点数: {n_missing} 个 (主要为 sub004 缺失定位轴及极浅层骨外触点)")

# -----------------------------------------------------------------------------
# 4. 加载真实标准玻璃脑表面网格 (fsaverage5)
# -----------------------------------------------------------------------------
print("\n>>> 正在加载标准 MNI 玻璃脑表面网格 (fsaverage5) ...")
fsaverage5 = ndata.fetch_surf_fsaverage('fsaverage5')
mesh_l_coords, mesh_l_faces = nsurf.load_surf_mesh(fsaverage5['pial_left'])
mesh_r_coords, mesh_r_faces = nsurf.load_surf_mesh(fsaverage5['pial_right'])
print(f"  [+] 左半球顶点: {mesh_l_coords.shape[0]} | 右半球顶点: {mesh_r_coords.shape[0]}")

# -----------------------------------------------------------------------------
# 5. 逐频段绘制标准三视图并保存
# -----------------------------------------------------------------------------
print("\n>>> 开始逐频段生成玻璃脑三视图 ...")

# 三视图的视角配置 (医学与工程标准规范)
views_cfg = [
    {
        'title': '矢状面 (Sagittal / Lateral View)',
        'elev': 0, 'azim': 0,
        'label': '侧视 (P ← 枕叶 | 额叶 → A)'
    },
    {
        'title': '冠状面 (Coronal / Posterior View)',
        'elev': 0, 'azim': -90,
        'label': '后视 (L ← 左半球 | 右半球 → R)'
    },
    {
        'title': '轴状面 (Axial / Dorsal View)',
        'elev': 90, 'azim': -90,
        'label': '俯视 (A ↑ 额叶 | 枕叶 ↓ P)'
    }
]

saved_figures = []

for band in cfg['bands']:
    band_color = cfg['band_colors'][band]
    freq_label = cfg['band_freq_labels'][band]
    
    # 筛选本频段有效点
    b_df = valid_sig_df[valid_sig_df['freq_band'] == band].copy()
    n_pts = len(b_df)
    
    if n_pts == 0:
        print(f"  [-] 频段 [{band}] 无有效坐标显著位点，跳过。")
        continue
    
    pts_xyz = np.array(b_df['coords'].tolist())
    effects = b_df['general_effect_100_400ms'].values
    
    # 判定彩色与灰色强弱
    is_col_greater = effects > 0    # 彩色 > 灰色 (实心)
    is_gry_greater = effects < 0    # 灰色 > 彩色 (空心虚线)
    
    n_col = int(np.sum(is_col_greater))
    n_gry = int(np.sum(is_gry_greater))
    
    # 差异绝对值映射到散点尺寸
    abs_eff = np.abs(effects)
    pt_sizes = cfg['min_marker_size'] + (abs_eff / cfg['scale_ref_diff']) * (cfg['max_marker_size'] - cfg['min_marker_size'])
    pt_sizes = np.clip(pt_sizes, cfg['min_marker_size'], cfg['max_marker_size'] * 1.2)
    
    # 创建三联图画布
    fig = plt.figure(figsize=(20, 7.5), dpi=cfg['dpi'])
    fig.patch.set_facecolor('white')
    
    for v_idx, v in enumerate(views_cfg):
        ax = fig.add_subplot(1, 3, v_idx + 1, projection='3d')
        ax.set_facecolor('white')
        
        # 1) 绘制半透明玻璃脑网格 (柔和灰白色，展现真实皮层沟回层次)
        ax.plot_trisurf(mesh_l_coords[:, 0], mesh_l_coords[:, 1], mesh_l_coords[:, 2],
                        triangles=mesh_l_faces, color='#E2E6EA', alpha=0.08, edgecolor='none', shade=True)
        ax.plot_trisurf(mesh_r_coords[:, 0], mesh_r_coords[:, 1], mesh_r_coords[:, 2],
                        triangles=mesh_r_faces, color='#E2E6EA', alpha=0.08, edgecolor='none', shade=True)
        
        # 2) 绘制彩色 > 灰色 (实心点)
        if n_col > 0:
            ax.scatter(pts_xyz[is_col_greater, 0],
                       pts_xyz[is_col_greater, 1],
                       pts_xyz[is_col_greater, 2],
                       s=pt_sizes[is_col_greater],
                       c=band_color,
                       edgecolors='#333333',
                       linewidths=0.8,
                       alpha=0.92,
                       depthshade=False,
                       zorder=10)
            
        # 3) 绘制灰色 > 彩色 (空心虚线圆)
        if n_gry > 0:
            ax.scatter(pts_xyz[is_gry_greater, 0],
                       pts_xyz[is_gry_greater, 1],
                       pts_xyz[is_gry_greater, 2],
                       s=pt_sizes[is_gry_greater],
                       facecolors='none',
                       edgecolors=band_color,
                       linestyles='--',
                       linewidths=2.0,
                       alpha=0.95,
                       depthshade=False,
                       zorder=10)
            
        # 视角与轴设置
        ax.view_init(elev=v['elev'], azim=v['azim'])
        ax.set_axis_off()
        ax.set_box_aspect([1, 1.25, 0.95])
        ax.set_title(f"{v['title']}\n{v['label']}", fontsize=12, fontweight='bold', pad=-6, color='#2C3E50')
    
    # 顶部总标题
    fig.suptitle(f"{band} ({freq_label}) 色彩显著电极玻璃脑空间分布 (三视图)\n"
                 f"显著通道总数: {n_pts} | 彩色 > 灰色 (实心): {n_col} | 灰色 > 彩色 (虚线空心): {n_gry}",
                 fontsize=15, fontweight='bold', y=0.97, color='#1A252F')
    
    # 底部说明图例
    legend_items = [
        Line2D([0], [0], marker='o', color='w', label=f'彩色 > 灰色 (实心, N={n_col})',
               markerfacecolor=band_color, markeredgecolor='#333333', markeredgewidth=0.8, markersize=10),
        Line2D([0], [0], marker='o', color='w', label=f'灰色 > 彩色 (空心虚线, N={n_gry})',
               markerfacecolor='none', markeredgecolor=band_color, markeredgewidth=2.0, markersize=11, linestyle='None'),
        Line2D([0], [0], marker='o', color='w', label='|Color - Gray| 差异量 (点越大效应越强)',
               markerfacecolor='#888888', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='',
               markerfacecolor='#888888', markersize=12)
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=4, frameon=True,
               fontsize=11, facecolor='#F8F9F9', edgecolor='#BDC3C7', bbox_to_anchor=(0.5, 0.02))
    
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.86, wspace=0.04)
    
    fig_filename = f"glass_brain_{band}_3view.png"
    out_file = os.path.join(cfg['out_dir'], fig_filename)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    saved_figures.append(out_file)
    print(f"  [✓] 已生成: {fig_filename} (显著点: {n_pts}, 实心: {n_col}, 虚心: {n_gry})")

# -----------------------------------------------------------------------------
# 6. 生成全频段合集画廊 (6 频段一览总表)
# -----------------------------------------------------------------------------
print("\n>>> 正在生成 6 频段全景汇总总览图 (Gallery View) ...")
fig_gallery, axes_gallery = plt.subplots(2, 3, figsize=(22, 14), dpi=cfg['dpi'],
                                         subplot_kw={'projection': '3d'})
fig_gallery.patch.set_facecolor('white')

for idx, band in enumerate(cfg['bands']):
    ax = axes_gallery.flat[idx]
    ax.set_facecolor('white')
    band_color = cfg['band_colors'][band]
    freq_label = cfg['band_freq_labels'][band]
    
    b_df = valid_sig_df[valid_sig_df['freq_band'] == band].copy()
    if len(b_df) > 0:
        pts_xyz = np.array(b_df['coords'].tolist())
        effects = b_df['general_effect_100_400ms'].values
        is_col = effects > 0
        is_gry = effects < 0
        abs_eff = np.abs(effects)
        pt_sizes = cfg['min_marker_size'] + (abs_eff / cfg['scale_ref_diff']) * (cfg['max_marker_size'] - cfg['min_marker_size'])
        pt_sizes = np.clip(pt_sizes, cfg['min_marker_size'], cfg['max_marker_size'])
        
        # 玻璃脑网格
        ax.plot_trisurf(mesh_l_coords[:, 0], mesh_l_coords[:, 1], mesh_l_coords[:, 2],
                        triangles=mesh_l_faces, color='#E2E6EA', alpha=0.07, edgecolor='none', shade=True)
        ax.plot_trisurf(mesh_r_coords[:, 0], mesh_r_coords[:, 1], mesh_r_coords[:, 2],
                        triangles=mesh_r_faces, color='#E2E6EA', alpha=0.07, edgecolor='none', shade=True)
        
        # 散点
        if np.any(is_col):
            ax.scatter(pts_xyz[is_col, 0], pts_xyz[is_col, 1], pts_xyz[is_col, 2],
                       s=pt_sizes[is_col]*0.8, c=band_color, edgecolors='#333333', linewidths=0.6,
                       alpha=0.92, depthshade=False, zorder=10)
        if np.any(is_gry):
            ax.scatter(pts_xyz[is_gry, 0], pts_xyz[is_gry, 1], pts_xyz[is_gry, 2],
                       s=pt_sizes[is_gry]*0.8, facecolors='none', edgecolors=band_color,
                       linestyles='--', linewidths=1.8, alpha=0.95, depthshade=False, zorder=10)
            
    # 采用冠状后视 (最具解剖代表性，展示枕颞叶双侧对比)
    ax.view_init(elev=0, azim=-90)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1.25, 0.95])
    ax.set_title(f"{band} ({freq_label})\n显著位点: {len(b_df)} (实心: {np.sum(b_df['general_effect_100_400ms']>0)}, 虚心: {np.sum(b_df['general_effect_100_400ms']<0)})",
                 fontsize=13, fontweight='bold', color=band_color, pad=-4)

fig_gallery.suptitle("全频段色彩显著电极玻璃脑分布全景对比 (冠状后视 Coronal Posterior View)\n"
                     "实心圆: 彩色 > 灰色 (ERS 增强) | 虚线空心圆: 灰色 > 彩色 (ERD 抑制) | 点尺寸: 效应绝对差值大小",
                     fontsize=16, fontweight='bold', y=0.97, color='#1A252F')

plt.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.90, wspace=0.02, hspace=0.08)
gallery_file = os.path.join(cfg['out_dir'], 'glass_brain_all_bands_summary.png')
plt.savefig(gallery_file, bbox_inches='tight')
plt.close(fig_gallery)
print(f"  [✓] 全频段对比总览图已生成: {gallery_file}")

print("\n" + "=" * 70)
print(f"  【全部 7 张学术级玻璃脑图谱生成完毕！】\n  保存目录: {os.path.abspath(cfg['out_dir'])}")
print("=" * 70)
