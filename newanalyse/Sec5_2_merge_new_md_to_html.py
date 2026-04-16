import os
import re
import json
import time
import pandas as pd
import ast
import numpy as np
import argparse
import base64
import io

# ──────────────────────────────────────────────────────────────
# 绘图相关导入
# ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')   # 无显示器环境必须放在 pyplot import 之前
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches

from nilearn import plotting as nlplot
from nilearn import datasets as nldatasets



# ──────────────────────────────────────────────────────────────
# ★ AAL1 atlas → ROI名 → MNI坐标 查找表
#
# 使用 nilearn 内置 AAL1（116 ROIs），无需下载额外文件。
# 对 AAL1 中不存在的特殊 ROI 名（如 V1/V2/MT 等视觉皮层缩写）
# 做手动映射，指向 AAL1 中解剖位置最接近的 ROI。
# ──────────────────────────────────────────────────────────────

import re as _re

# ── 特殊 ROI 名 → AAL1 最近似 ROI 的映射 ──────────────────────
# 规则：优先保留半球侧（_L/_R），找不到侧化版本时取左侧作为默认。
# 如果你的数据里出现其他找不到的 ROI 名，在此处补充即可。
_SPECIAL_NAME_MAP = {
    # ── 视觉皮层 ──────────────────────────────────────────────
    'V1':            'Calcarine',        # 初级视觉皮层 ≈ 距状沟
    'V2':            'Occipital_Sup',    # 次级视觉皮层 ≈ 枕上回
    'V3':            'Occipital_Mid',    # V3 ≈ 枕中回
    'V4':            'Occipital_Inf',    # V4 ≈ 枕下回
    'V5':            'Occipital_Mid',    # V5/MT ≈ 枕中回
    'MT':            'Occipital_Mid',    # 运动视觉区
    'MST':           'Occipital_Mid',
    'IT':            'Temporal_Inf',     # 下颞叶视觉区
    'LOC':           'Occipital_Inf',    # 侧枕皮层
    'LO':            'Occipital_Mid',
    'FFA':           'Fusiform',         # 梭状回面孔区
    'PPA':           'ParaHippocampal',  # 海马旁回位置区
    'RSC':           'Cingulum_Post',    # 压后皮层 ≈ 后扣带
    'EBA':           'Temporal_Mid',     # 身体区 ≈ 颞中回
    'ERC':           'ParaHippocampal',  # 内嗅皮层 ≈ 海马旁
    'OPA':           'Occipital_Sup',    # 枕旁区
    # ── 额叶 ──────────────────────────────────────────────────
    'DLPFC':         'Frontal_Mid',      # 背外侧前额叶
    'VLPFC':         'Frontal_Inf_Tri',  # 腹外侧前额叶
    'MPFC':          'Frontal_Med_Orb',  # 内侧前额叶
    'OFC':           'Frontal_Med_Orb',  # 眶额皮层
    'ACC':           'Cingulum_Ant',     # 前扣带
    'PCC':           'Cingulum_Post',    # 后扣带
    'MCC':           'Cingulum_Mid',     # 中扣带
    'SMA':           'Supp_Motor_Area',  # 辅助运动区
    'preSMA':        'Supp_Motor_Area',
    'M1':            'Precentral',       # 初级运动皮层
    'PMC':           'Precentral',       # 运动前皮层
    'PMd':           'Frontal_Sup',      # 背侧运动前
    'PMv':           'Frontal_Inf_Oper', # 腹侧运动前
    'FEF':           'Frontal_Sup',      # 额叶眼动区
    'IFG':           'Frontal_Inf_Tri',  # 额下回
    'MFG':           'Frontal_Mid',      # 额中回
    'SFG':           'Frontal_Sup',      # 额上回
    'IFG_op':        'Frontal_Inf_Oper',
    'IFG_tri':       'Frontal_Inf_Tri',
    'IFG_orb':       'Frontal_Inf_Orb',
    # ── 顶叶 ──────────────────────────────────────────────────
    'S1':            'Postcentral',      # 初级躯体感觉皮层
    'S2':            'Parietal_Inf',     # 次级躯体感觉
    'IPS':           'Parietal_Sup',     # 顶内沟
    'SPL':           'Parietal_Sup',     # 顶上小叶
    'IPL':           'Parietal_Inf',     # 顶下小叶
    'SMG':           'SupraMarginal',    # 缘上回
    'AG':            'Angular',          # 角回
    'Precuneus':     'Precuneus',        # 楔前叶（AAL1 有此名）
    # ── 颞叶 ──────────────────────────────────────────────────
    'STG':           'Temporal_Sup',     # 颞上回
    'MTG':           'Temporal_Mid',     # 颞中回
    'ITG':           'Temporal_Inf',     # 颞下回
    'STS':           'Temporal_Sup',     # 颞上沟
    'TP':            'Temporal_Pole_Sup',# 颞极
    'HPC':           'Hippocampus',      # 海马
    'PHC':           'ParaHippocampal',  # 海马旁皮层
    # ── 枕叶 ──────────────────────────────────────────────────
    'LingG':         'Lingual',          # 舌回
    'CunG':          'Cuneus',           # 楔叶
    'CalcG':         'Calcarine',        # 距状沟
    'SOG':           'Occipital_Sup',    # 枕上回
    'MOG':           'Occipital_Mid',    # 枕中回
    'IOG':           'Occipital_Inf',    # 枕下回
    # ── 皮层下 ────────────────────────────────────────────────
    'AMY':           'Amygdala',         # 杏仁核
    'Amygdala':      'Amygdala',
    'Hippocampus':   'Hippocampus',
    'Thalamus':      'Thalamus',
    'Thal':          'Thalamus',
    'BG':            'Putamen',          # 基底节（近似用壳核）
    'Caudate':       'Caudate',
    'Putamen':       'Putamen',
    'Pallidum':      'Pallidum',
    # ── 岛叶 ──────────────────────────────────────────────────
    'Insula':        'Insula',
    'Ins':           'Insula',
    'INS':           'Insula',
    'AI':            'Insula',           # 前岛叶
    'PI':            'Insula',           # 后岛叶
    # ── 小脑 ──────────────────────────────────────────────────
    'CB':            'Cerebelum_6',      # 小脑（近似）
    'CbL':           'Cerebelum_6',
    'CbR':           'Cerebelum_6',
}


def build_aal_coord_lookup():
    """
    加载 AAL1 atlas（116 ROIs）并计算每个 ROI 的 MNI 重心坐标。

    关键：nilearn >= 0.10 的 fetch_atlas_aal() 默认下载 AAL3v2，
    必须显式传 version='SPM5' 才能使用本地缓存的 AAL1，不联网。
    若仍失败则尝试 SSL 补丁再重试一次。
    """
    from nilearn import datasets as nlds
    from nilearn.image import load_img

    # ── 1. 加载 AAL1（强制 version='SPM5'，本地缓存，不联网）──
    print("  [AAL1] Loading atlas (version=SPM5, no network needed) ...")
    aal = None
    try:
        aal = nlds.fetch_atlas_aal(version='SPM5')
    except Exception as e1:
        print(f"  [AAL1] First attempt failed: {e1}")
        # ── SSL 补丁：临时禁用证书验证后重试 ──────────────────
        print("  [AAL1] Retrying with SSL verification disabled ...")
        import ssl, urllib.request
        _orig_create = ssl.create_default_context
        ssl.create_default_context = lambda *a, **kw: (
            ctx := _orig_create(*a, **kw),
            setattr(ctx, 'check_hostname', False),
            setattr(ctx, 'verify_mode', ssl.CERT_NONE),
            ctx
        )[-1]
        try:
            import requests, urllib3
            urllib3.disable_warnings()
            _orig_send = requests.Session.send
            def _unverified_send(self, req, **kw):
                kw['verify'] = False
                return _orig_send(self, req, **kw)
            requests.Session.send = _unverified_send
            aal = nlds.fetch_atlas_aal(version='SPM5')
        except Exception as e2:
            raise RuntimeError(
                f"Cannot load AAL1 atlas.\n"
                f"  Error 1: {e1}\n"
                f"  Error 2: {e2}\n"
                "  → 请确认已安装 nilearn 并能访问网络（或已缓存本地文件）"
            ) from e2
        finally:
            ssl.create_default_context = _orig_create   # 恢复 SSL 设置

    # ── 2. 计算每个 ROI 的 MNI 重心坐标 ───────────────────────
    atlas_img  = load_img(aal.maps)
    atlas_data = atlas_img.get_fdata()
    affine     = atlas_img.affine
    indices    = [int(x) for x in aal.indices]
    labels     = list(aal.labels)

    coord_map = {}
    for idx, label in zip(indices, labels):
        voxels = np.argwhere(atlas_data == idx)
        if len(voxels) == 0:
            continue
        centroid_vox = voxels.mean(axis=0)
        centroid_mni = affine[:3, :3] @ centroid_vox + affine[:3, 3]
        key = label.strip().replace(' ', '_')
        coord_map[key] = [float(c) for c in centroid_mni]

    print(f"  [AAL1] {len(coord_map)} ROIs loaded.")

    # ── 3. 注册特殊别名（V2、MT 等 → AAL1 最近似 ROI）─────────
    for short, full_base in _SPECIAL_NAME_MAP.items():
        full_base = full_base.replace(' ', '_')
        # 带侧化后缀版本（_L 和 _R 都注册，不覆盖已有条目）
        for side in ('_L', '_R'):
            short_sided = short + side
            if short_sided not in coord_map:
                for candidate in (full_base + side, full_base):
                    if candidate in coord_map:
                        coord_map[short_sided] = coord_map[candidate]
                        break
        # 不带后缀版本（取 _L 坐标作为默认）
        if short not in coord_map:
            for candidate in (full_base + '_L', full_base + '_R', full_base):
                if candidate in coord_map:
                    coord_map[short] = coord_map[candidate]
                    break

    print(f"  [AAL1] {len(coord_map)} entries after alias registration.")
    return coord_map


# ── 全局单例，整个脚本只加载一次 ──────────────────────────────
_AAL_COORD_LOOKUP = None


def get_aal_mni(roi_name):
    """
    查询 ROI 的 MNI 坐标，返回 [x, y, z] 或 None。

    匹配优先级：
      1. 精确匹配
      2. 大小写不敏感精确匹配
      3. 末尾数字剥离（Frontal_Sup_L2 → Frontal_Sup_L）
      4. 前缀 / 子串包含（兜底，最宽松）
    """
    global _AAL_COORD_LOOKUP
    if _AAL_COORD_LOOKUP is None:
        _AAL_COORD_LOOKUP = build_aal_coord_lookup()

    key = roi_name.strip().replace(' ', '_')

    # 1. 精确匹配
    if key in _AAL_COORD_LOOKUP:
        return _AAL_COORD_LOOKUP[key]

    # 2. 大小写不敏感
    key_lower = key.lower()
    for k, v in _AAL_COORD_LOOKUP.items():
        if k.lower() == key_lower:
            return v

    # 3. 末尾数字剥离
    key_stripped = _re.sub(r'\d+$', '', key)
    if key_stripped != key:
        if key_stripped in _AAL_COORD_LOOKUP:
            return _AAL_COORD_LOOKUP[key_stripped]
        for k, v in _AAL_COORD_LOOKUP.items():
            if k.lower() == key_stripped.lower():
                return v

    # 4. 前缀 / 子串（兜底）
    for k, v in _AAL_COORD_LOOKUP.items():
        kl = k.lower()
        if kl.startswith(key_lower) or key_lower.startswith(kl):
            return v

    return None



def process_subject(subject_id, base_project_dir):
    """
    Process a single subject and return its structured data.
    坐标来源：nilearn AAL atlas（不再依赖 Excel 文件）
    """
    subject_dir = os.path.join(base_project_dir, 'result', 'reports', 'replot_within_decoding', subject_id)

    md_files = {
        'lowgamma':  os.path.join(subject_dir, 'lowgamma/perm1000/NEW/significant_rois_summary.md'),
        'erp':       os.path.join(subject_dir, 'erp/perm1000/NEW/significant_rois_summary.md'),
        'highgamma': os.path.join(subject_dir, 'highgamma/perm1000/NEW/significant_rois_summary.md'),
    }

    task_name_mapping = {
        'Task 3 Pure Color Self-Decoding: Condition 1 vs Condition 4': 'task3_1vs4_self',
        'Task 3 Pure Color Self-Decoding: Condition 2 vs Condition 3': 'task3_2vs3_self',
        'Task 1 Color vs Gray Pair Holdout Decoding':                  'task1_color_vs_gray_pair_cv',
        'Task 2 Gray Fruit Memory-Color Decoding':                     'task2_gray_memory_color_cross',
        'Task 2 True vs False Fruit Color Decoding':                   'task2_true_vs_false',
    }

    subject_data = []

    for band, filepath in md_files.items():
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_task = None
        for line in lines:
            line = line.strip()
            if line.startswith('## '):
                raw_task = line.replace('## ', '').strip()
                current_task = task_name_mapping.get(raw_task, raw_task)
            elif line.startswith('|') and current_task:
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 2 and parts[0] != 'ROI Name' and not parts[0].startswith('---'):
                    roi = parts[0]
                    earliest_ms = parts[1]
                    try:
                        earliest_val = float(earliest_ms)
                        coords = get_aal_mni(roi)   # ★ 从 AAL atlas 查坐标
                        if coords is None:
                            print(f"    [WARN] ROI not found in AAL atlas: {roi}")
                            coords = [0, 0, 0]
                        subject_data.append({
                            'task':    current_task,
                            'band':    band,
                            'roi':     roi,
                            'earliest': earliest_val,
                            'mni_x':   coords[0],
                            'mni_y':   coords[1],
                            'mni_z':   coords[2],
                            'remarks': 'Significant',
                        })
                    except ValueError:
                        pass

    return subject_data




# ──────────────────────────────────────────────────────────────
# ★ 核心函数：nilearn 玻璃脑背景 + SEEG ROI 散点
#    双视角（sagittal + axial）拼成一张图，输出 SVG base64
# ──────────────────────────────────────────────────────────────

# 颜色方案（每个频段一套）
BAND_CMAPS = {
    'erp':       'cool',
    'lowgamma':  'YlOrRd',
    'highgamma': 'YlGn',
}


def _fig_to_svg_b64(fig):
    """将 matplotlib Figure 转为 base64 SVG 字符串，嵌入 <img> 标签"""
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"


def generate_seeg_brain_map(band_data, title="SEEG ROI Brain Map"):
    """
    使用 nilearn plot_glass_brain 作为背景，
    让 nilearn 自己完成 MNI 坐标→2D 投影（坐标完全准确），
    在 sagittal + axial 两个视角上叠加 SEEG ROI 散点与标注。

    Parameters
    ----------
    band_data : list of dict
        每个 dict 含 'roi', 'earliest', 'mni_x', 'mni_y', 'mni_z'
    title : str
        图标题（显示在图上方）

    Returns
    -------
    str  SVG base64 字符串，可直接用作 <img src="...">
    """
    band_key  = band_data[0]['band'] if band_data else 'erp'
    cmap_name = BAND_CMAPS.get(band_key, 'YlOrRd')

    # ── 过滤有效数据（排除坐标全为 0 的占位点）──────────────
    valid = [d for d in band_data
             if d['earliest'] is not None
             and not (d['mni_x'] == 0 and d['mni_y'] == 0 and d['mni_z'] == 0)]

    # ── 空数据占位图 ────────────────────────────────────────
    if not valid:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
        ax.axis('off')
        ax.text(0.5, 0.5, f'No valid SEEG ROI data\n({band_key})',
                ha='center', va='center', fontsize=13, color='#aaa',
                transform=ax.transAxes)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return _fig_to_svg_b64(fig)

    coords_mni = np.array([[d['mni_x'], d['mni_y'], d['mni_z']] for d in valid])
    values     = np.array([d['earliest'] for d in valid])
    roi_names  = [d['roi'] for d in valid]

    vmin, vmax = values.min(), float(values.max())
    if vmax == vmin:
        vmax = vmin + 1
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap   = plt.get_cmap(cmap_name)
    # 点大小：越早激活越大（100~300），让早期 ROI 更突出
    marker_sizes = 100 + 200 * (1 - norm(values))

    # ── 逐视角生成玻璃脑图，各自保存为 PNG bytes，再拼合 ───
    # 避免使用已废弃的 tostring_rgb()，改用 savefig → BytesIO → PIL
    from PIL import Image

    view_specs = [
        ('l', 'Sagittal View'),   # 左侧矢状面
        ('z', 'Axial View'),      # 轴状面（俯视）
    ]

    sub_imgs = []   # 存 PIL Image 对象
    for display_mode, view_label in view_specs:
        fig_sub = plt.figure(figsize=(8, 5.5), facecolor='white')

        # ---- 玻璃脑背景 ----
        display = nlplot.plot_glass_brain(
            None,
            display_mode=display_mode,
            figure=fig_sub,
            colorbar=False,
            alpha=0.25,
            title=None,
            annotate=True,
        )

        # ---- add_markers：MNI 坐标直接传入，nilearn 自动投影 ----
        display.add_markers(
            coords_mni,
            marker_color=[cmap(norm(v)) for v in values],
            marker_size=marker_sizes,
        )

        # ── ROI 文字标注 ──────────────────────────────────────
        # nilearn 的 OrthoSlicer axes 数据坐标系就是 MNI mm，
        # 根据视角选择对应的 ax 和投影轴
        #
        # display.axes 是 dict，key 为方向字符串：
        #   'l' → 左矢状面  axes 数据坐标：x=Y(MNI), y=Z(MNI)
        #   'z' → 轴状面    axes 数据坐标：x=X(MNI), y=Y(MNI)
        #
        # 直接用 MNI 坐标的对应分量在 ax 上 annotate，坐标完全准确。

        ax_key = display_mode   # 'l' 或 'z'
        if ax_key in display.axes:
            brain_ax = display.axes[ax_key].ax

            if display_mode == 'l':
                # sagittal：ax 数据坐标 x=Y(MNI后前), y=Z(MNI下上)
                proj = lambda c: (c[1], c[2])
            else:
                # axial：ax 数据坐标 x=X(MNI右左), y=Y(MNI后前)
                proj = lambda c: (c[0], c[1])

            for name, coord, val in zip(roi_names, coords_mni, values):
                px, py = proj(coord)
                brain_ax.annotate(
                    f"{name}\n{val:.0f} ms",
                    xy=(px, py),
                    xytext=(12, 8),
                    textcoords='offset points',
                    fontsize=9,
                    color='#111111',
                    fontweight='bold',
                    ha='left',
                    arrowprops=dict(
                        arrowstyle='-',
                        color='#888888',
                        lw=0.9,
                    ),
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        fc='white',
                        alpha=0.88,
                        ec='#bbbbbb',
                        lw=0.6,
                    ),
                    zorder=30,
                )

        # 视角标题
        fig_sub.text(0.5, 0.97, view_label,
                     ha='center', va='top',
                     fontsize=14,
                     color='#222222',
                     fontweight='bold',
                     transform=fig_sub.transFigure)

        # ---- 保存为 PNG bytes → PIL Image（不依赖 tostring_rgb）----
        buf_sub = io.BytesIO()
        fig_sub.savefig(buf_sub, format='png', dpi=150,
                        bbox_inches='tight', facecolor='white')
        plt.close(fig_sub)
        buf_sub.seek(0)
        sub_imgs.append(Image.open(buf_sub).copy())   # copy() 解除文件锁

    # ── 横向拼合两张 PIL Image ───────────────────────────────
    total_w = sum(img.width for img in sub_imgs)
    max_h   = max(img.height for img in sub_imgs)
    combined = Image.new('RGB', (total_w, max_h), (255, 255, 255))
    x_offset = 0
    for img in sub_imgs:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width

    # ── 用 matplotlib 加 colorbar + 标题后输出 SVG ──────────
    fig_dpi   = 150
    fig_w_in  = total_w / fig_dpi
    fig_h_in  = max_h   / fig_dpi + 1.0    # 留出标题和图例空间

    fig_final = plt.figure(figsize=(fig_w_in + 1.8, fig_h_in),
                           facecolor='white')
    fig_final.suptitle(title,
                       fontsize=16,              # ★ 总标题字号
                       fontweight='bold',
                       y=0.98,
                       color='#222')

    # 主图区（贴拼合后的 PIL 图）
    ax_main = fig_final.add_axes([0.0, 0.06, 0.89, 0.88])
    ax_main.imshow(np.array(combined))
    ax_main.axis('off')

    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig_final.add_axes([0.91, 0.15, 0.018, 0.65])
    cbar = fig_final.colorbar(sm, cax=cbar_ax)
    cbar.set_label('First Latency (ms)',
                   fontsize=13,              # ★ colorbar 标签字号
                   color='#333',
                   labelpad=10)
    cbar.ax.tick_params(labelsize=11)        # ★ colorbar 刻度字号

    # 图例文字
    fig_final.text(
        0.45, 0.01,
        f'SEEG · iEEG  |  band: {band_key.upper()}  |  '
        f'n = {len(valid)} ROIs  |  Glass Brain (MNI152)',
        ha='center',
        fontsize=11,                         # ★ 底部说明字号
        color='#555',
        transform=fig_final.transFigure,
    )

    return _fig_to_svg_b64(fig_final)


# ──────────────────────────────────────────────────────────────
# 主脚本
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Generate multi-subject HTML report')
_script_start_time = time.time()
parser.add_argument('--subjects', nargs='+', default=['test001', 'test002'],
                    help='List of subject IDs to include')
parser.add_argument('--base-dir',
                    default='/home/lirui/liulab_project/ieeg/Project_colorieeg_2026',
                    help='Base project directory')

args = parser.parse_args()

output_html = os.path.join(
    args.base_dir, 'result', 'reports', 'replot_within_decoding', 'multi_subject_report.html'
)

all_subjects_data = {}
# ★ 新增：预先生成所有 MNE 脑图，存为 base64
#    结构：mne_images[subject][task][band] = "data:image/png;base64,..."
mne_images = {}

for subject in args.subjects:
    print(f"Processing subject: {subject}")
    sub_data = process_subject(subject, args.base_dir)
    if sub_data:
        tasks_data = {}
        for item in sub_data:
            task = item['task']
            if task not in tasks_data:
                tasks_data[task] = []
            tasks_data[task].append(item)
        all_subjects_data[subject] = tasks_data

        # ── 为该 subject 的每个 task × band 生成 MNE 图 ──
        mne_images[subject] = {}
        for task, items in tasks_data.items():
            mne_images[subject][task] = {}
            bands = list({d['band'] for d in items})
            for band in bands:
                band_items = [d for d in items if d['band'] == band]
                print(f"  Generating MNE map: {subject} / {task} / {band} ...")
                try:
                    b64 = generate_seeg_brain_map(
                        band_items,
                        title=f"{subject}  |  {task}  |  {band.upper()}",
                    )
                except Exception as e:
                    print(f"    !! MNE map generation failed: {e}")
                    b64 = ""
                mne_images[subject][task][band] = b64
    else:
        print(f"No valid data found for {subject}")


# ── 本地库读取（CSS/JS 嵌入）────────────────────────────────
def read_local_file(filepath):
    full_path = os.path.join(
        '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026', filepath)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


# ── HTML 模板 ────────────────────────────────────────────────
html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EEG/SEEG 实验数据综合分析报告 (基于 NEW 汇总表)</title>

    <style>
        {read_local_file('bootstrap.min.css')}
        {read_local_file('datatables.min.css')}

        body {{
            background-color: #f4f7f6;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
        }}
        .sidebar {{
            position: fixed; top: 0; bottom: 0; left: 0; z-index: 100;
            padding: 48px 0 0;
            box-shadow: inset -1px 0 0 rgba(0,0,0,.1);
            background-color: #343a40; color: white;
        }}
        .sidebar-sticky {{
            position: relative; top: 0;
            height: calc(100vh - 48px);
            padding-top: .5rem;
            overflow-x: hidden; overflow-y: auto;
        }}
        main {{ padding-top: 2rem; padding-bottom: 3rem; }}
        .header-title {{
            color: #2c3e50; font-weight: bold; margin-bottom: 1.5rem;
            padding-bottom: 1rem; border-bottom: 2px solid #e9ecef;
        }}
        .nav-link {{ color: #adb5bd; font-weight: 500; padding: 10px 20px; }}
        .nav-link:hover, .nav-link.active {{ color: #fff; background-color: #495057; }}
        .task-section {{
            background: white; border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            padding: 30px; margin-bottom: 40px; display: none;
        }}
        .task-section.active {{ display: block; }}
        .task-title {{
            color: #0d6efd; border-left: 5px solid #0d6efd;
            padding-left: 15px; margin-bottom: 25px;
        }}
        .card-header {{ background-color: #f8f9fa; font-weight: bold; border-bottom: 1px solid #e9ecef; }}
        .chart-container {{ height: 400px; width: 100%; }}

        /* ★ 脑图容器（SVG 矢量图，清晰无损） */
        .mne-img-container {{
            width: 100%; background: white;
            border: 1px solid #eee; border-radius: 8px;
            overflow: hidden; text-align: center; padding: 8px;
        }}
        .mne-img-container img {{
            max-width: 100%; height: auto;
            border-radius: 4px;
            image-rendering: auto;
        }}
        .mne-placeholder {{
            height: 260px; display: flex; align-items: center;
            justify-content: center; color: #999; font-size: 14px;
        }}

        .badge-erp {{ background-color: #1f77b4; }}
        .badge-lowgamma {{ background-color: #ff7f0e; }}
        .badge-highgamma {{ background-color: #2ca02c; }}
        .table-responsive {{ margin-top: 20px; }}
        .dataTables_wrapper .dataTables_paginate .paginate_button {{ padding: 0.2em 0.5em; }}
        .band-section {{
            background-color: #fcfcfc; border: 1px solid #e2e2e2;
            border-radius: 10px; padding: 20px; margin-bottom: 30px;
        }}
        .band-title {{
            font-size: 1.25rem; font-weight: 600; margin-bottom: 15px;
            color: #444; border-bottom: 2px solid #ddd; padding-bottom: 10px;
        }}
    </style>
</head>
<body>
<div class="container-fluid">
  <div class="row">
    <!-- Sidebar -->
    <nav id="sidebarMenu" class="col-md-3 col-lg-2 d-md-block sidebar collapse">
      <h4 class="text-center mb-4 px-3">📊 NEW脑区报告</h4>
      <div class="sidebar-sticky pt-3">
        <ul class="nav flex-column" id="navList"></ul>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4">
      <h2 class="header-title">综合实验数据可视化分析 (基于NEW汇总文档)</h2>
      <div class="alert alert-info" role="alert">
        <strong>提示：</strong> 左侧菜单切换 Task；每个 Task 展示 nilearn 生成的 SEEG 脑区投影图（Sagittal + Axial 双视角，MNI152 标准脑背景）、时序图及详细数据表。
      </div>
      <div id="contentArea"></div>
    </main>
  </div>
</div>

<!-- Embedded JS Libraries -->
<script>{read_local_file('jquery.min.js')}</script>
<script>{read_local_file('bootstrap.bundle.min.js')}</script>
<script>{read_local_file('datatables.min.js')}</script>
<script>{read_local_file('datatables-bs5.min.js')}</script>
<script>{read_local_file('echarts.min.js')}</script>

<!-- Application Logic -->
<script>
// ── 数据注入 ──────────────────────────────────────────────────
const allSubjectsData = JSON.parse('__DATA_JSON__');

// ★ MNE 图片 base64 字典：mneImages[subject][task][band]
const mneImages = JSON.parse('__MNE_IMAGES_JSON__');

let chartInstances = [];

$(document).ready(function() {{
    initNavigation();
    const subjects = Object.keys(allSubjectsData);
    if (subjects.length > 0) {{
        const firstTask = Object.keys(allSubjectsData[subjects[0]] || {{}})[0];
        if (firstTask) activateTask(firstTask);
    }}
    window.addEventListener('resize', () => chartInstances.forEach(c => c.resize()));
}});

function initNavigation() {{
    const navList = $('#navList');
    let allTasks = new Set();
    Object.values(allSubjectsData).forEach(sd => Object.keys(sd).forEach(t => allTasks.add(t)));
    Array.from(allTasks).sort().forEach(task => {{
        const li = $(`<li class="nav-item">
            <a class="nav-link" href="#" data-task="${{task}}">📝 ${{task}}</a>
        </li>`);
        li.find('a').on('click', e => {{ e.preventDefault(); activateTask(task); }});
        navList.append(li);
    }});
}}

function activateTask(taskName) {{
    $('.nav-link').removeClass('active');
    $(`.nav-link[data-task="${{taskName}}"]`).addClass('active');
    $('#contentArea').empty();
    chartInstances.forEach(c => c.dispose()); chartInstances = [];

    const availableSubjects = Object.keys(allSubjectsData).filter(s => allSubjectsData[s][taskName]);
    if (!availableSubjects.length) return;

    let subjectSelectorHtml = `
        <div class="mb-4 d-flex align-items-center bg-white p-3 rounded shadow-sm border">
            <label class="fw-bold me-3 text-secondary mb-0">切换被试 (Subject): </label>
            <div class="btn-group" role="group" id="subjectToggleGroup">`;
    availableSubjects.forEach((sub, idx) => {{
        subjectSelectorHtml += `
            <input type="radio" class="btn-check subject-radio" name="subjectRadio"
                   id="radio-${{sub}}" value="${{sub}}" autocomplete="off" ${{idx===0?'checked':''}}>
            <label class="btn btn-outline-primary" for="radio-${{sub}}">${{sub}}</label>`;
    }});
    subjectSelectorHtml += `</div></div>`;

    $('#contentArea').html(`
        <div class="task-section active" id="section-${{taskName}}">
            <h3 class="task-title">${{taskName}}</h3>
            ${{subjectSelectorHtml}}
            <div id="subjectContentArea"></div>
        </div>`);

    $('.subject-radio').on('change', function() {{
        renderSubjectContent(taskName, this.value);
    }});
    renderSubjectContent(taskName, availableSubjects[0]);
}}

function renderSubjectContent(taskName, subject) {{
    const subjectContainer = $('#subjectContentArea');
    subjectContainer.empty();
    chartInstances.forEach(c => c.dispose()); chartInstances = [];

    const taskData = allSubjectsData[subject][taskName];
    if (!taskData) return;
    taskData.sort((a, b) => (a.earliest||0) - (b.earliest||0));

    const bands = [...new Set(taskData.map(d => d.band))];
    let html = '';

    bands.forEach(band => {{
        // ★ 取对应 MNE 图片
        const imgSrc = (mneImages[subject] && mneImages[subject][taskName] && mneImages[subject][taskName][band])
            ? mneImages[subject][taskName][band] : '';

        const mapHtml = imgSrc
            ? `<div class="mne-img-container">
                   <img src="${{imgSrc}}" alt="MNE Brain Map - ${{band}}" />
               </div>`
            : `<div class="mne-img-container mne-placeholder">暂无脑图数据</div>`;

        html += `
        <div class="band-section">
            <div class="band-title">
                频段: <span class="text-uppercase text-primary">${{band}}</span>
                <span class="badge bg-secondary ms-2 fs-6">阈值: p&lt;0.05</span>
            </div>
            <div class="row">
                <!-- 左：MNE 2D 脑图 -->
                <div class="col-lg-6 mb-4 mb-lg-0">
                    <div class="card h-100">
                        <div class="card-header bg-light">SEEG 脑区投影图 (Sagittal + Axial · MNI152)</div>
                        <div class="card-body p-2">${{mapHtml}}</div>
                    </div>
                </div>
                <!-- 右：时序图 -->
                <div class="col-lg-6">
                    <div class="card h-100">
                        <div class="card-header bg-light">时序动态分析 (First Latency)</div>
                        <div class="card-body">
                            <div id="timeline-${{taskName}}-${{band}}" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>`;
    }});

    // 数据表
    html += `
    <div class="card mt-4">
        <div class="card-header bg-dark text-white">详细数据表 (${{subject}})</div>
        <div class="card-body">
            <div class="table-responsive">
                <table id="table-${{taskName}}" class="table table-striped table-bordered w-100">
                    <thead class="table-light">
                        <tr>
                            <th>频段</th><th>显著 ROI</th>
                            <th>Earliest Latency (ms)</th>
                            <th>MNI (X, Y, Z)</th><th>备注</th>
                        </tr>
                    </thead>
                    <tbody>`;
    taskData.forEach(row => {{
        const bc = row.band==='erp' ? 'badge-erp' : (row.band==='lowgamma' ? 'badge-lowgamma' : 'badge-highgamma');
        html += `<tr>
            <td><span class="badge ${{bc}}">${{row.band}}</span></td>
            <td><strong>${{row.roi}}</strong></td>
            <td>${{row.earliest !== null ? row.earliest : '-'}}</td>
            <td>(${{row.mni_x.toFixed(1)}}, ${{row.mni_y.toFixed(1)}}, ${{row.mni_z.toFixed(1)}})</td>
            <td>${{row.remarks}}</td>
        </tr>`;
    }});
    html += `</tbody></table></div></div></div>`;

    subjectContainer.html(html);

    // 初始化时序图
    bands.forEach(band => {{
        const bandData = taskData.filter(d => d.band === band);
        initTimelineChart(`timeline-${{taskName}}-${{band}}`, bandData, band);
    }});

    $(`#table-${{taskName}}`).DataTable({{
        pageLength: 10, responsive: true, destroy: true,
        language: {{ url: "https://cdn.datatables.net/plug-ins/1.13.6/i18n/zh-HANT.json" }}
    }});
}}

function initTimelineChart(containerId, bandData, band) {{
    const chartDom = document.getElementById(containerId);
    if (!chartDom) return;
    const myChart = echarts.init(chartDom);
    chartInstances.push(myChart);

    const validData = bandData.filter(d => d.earliest !== null);
    const plotData  = [...validData].sort((a, b) => b.earliest - a.earliest);
    const yAxisData  = plotData.map(d => d.roi);
    const colorMap   = {{ erp: '#1f77b4', lowgamma: '#ff7f0e', highgamma: '#2ca02c' }};
    const seriesData = plotData.map(d => ({{
        value: d.earliest,
        itemStyle: {{ color: colorMap[band] || '#333' }}
    }}));

    myChart.setOption({{
        tooltip: {{
            trigger: 'axis', axisPointer: {{ type: 'shadow' }},
            formatter: p => `ROI: ${{p[0].name}}<br/>First Latency: <strong>${{p[0].value}} ms</strong>`
        }},
        grid: {{ left: '3%', right: '12%', bottom: '5%', top: '5%', containLabel: true }},
        xAxis: {{
            type: 'value', name: 'First Latency (ms)',
            splitLine: {{ show: true, lineStyle: {{ type: 'dashed' }} }}
        }},
        yAxis: {{
            type: 'category', data: yAxisData,
            axisLabel: {{ fontSize: 11, fontWeight: 'bold' }}
        }},
        series: [{{
            type: 'bar', barWidth: '40%', data: seriesData,
            itemStyle: {{ borderRadius: [0,4,4,0] }},
            label: {{ show: true, position: 'right', formatter: '{{c}} ms', fontSize: 10 }}
        }}]
    }});
}}
</script>
</body>
</html>
"""

# ── 注入数据 JSON ─────────────────────────────────────────────
final_html = html_template.replace(
    '__DATA_JSON__',
    json.dumps(all_subjects_data, ensure_ascii=False).replace("'", "\\'")
).replace(
    '__MNE_IMAGES_JSON__',
    json.dumps(mne_images, ensure_ascii=False).replace("'", "\\'")
)

with open(output_html, 'w', encoding='utf-8') as f:
    f.write(final_html)

print(f"✅ HTML report generated: {output_html}")
print(f"Total runtime: {time.time() - _script_start_time:.2f} s")