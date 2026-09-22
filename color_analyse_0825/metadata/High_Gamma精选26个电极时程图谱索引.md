# 26 个精选 High-Gamma 色彩一致性电极时程图谱与数据索引

本目录遵循**【结果图片分文件夹管理】**原则：
- **图片根目录**：`color_analyse_0825/qc/high_gamma_timecourses/`
- **被试子目录**：`color_analyse_0825/qc/high_gamma_timecourses/<sub_id>/`
- **数据持久化目录**：`color_analyse_0825/process_data/selected_timecourses/high_gamma/`

---

## 1. 26 个目标通道明细表

| 序号 | 被试编号 | 电极通道 | 解剖定位 (DKT/AAL) | MNI 坐标 [X, Y, Z] | 一致性类型 | 100-400ms 效应量 (dB) | 置换 p 值 |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| 1 | `test001` | **D14** | Lateral occipital R | [35.3, -83.9, 4.7] | ▼ 一致负向抑制 | -0.51 dB | 0.0120 |
| 2 | `test001` | **F14** | N/A | [NaN, NaN, NaN] | ▼ 一致负向抑制 | -0.34 dB | 0.0420 |
| 3 | `test002` | **D3** | Cuneus R | [8.4, -95.6, 17.8] | ★ 一致正向增强 | +0.51 dB | 0.0185 |
| 4 | `test002` | **D4** | Lateral occipital R | [11.2, -98.4, 18.0] | ★ 一致正向增强 | +0.49 dB | 0.0110 |
| 5 | `test002` | **D6** | White matter | [16.8, -103.9, 18.3] | ★ 一致正向增强 | +0.33 dB | 0.0420 |
| 6 | `test003` | **D11** | White matter R | [39.1, -66.5, 4.7] | ▼ 一致负向抑制 | -0.37 dB | 0.0205 |
| 7 | `test003` | **E14** | Lateral occipital R | [46.7, -78.4, 1.8] | ▼ 一致负向抑制 | -0.48 dB | 0.0040 |
| 8 | `test003` | **G5** | Middle temporal R | [49.3, -37.5, -4.9] | ★ 一致正向增强 | +0.42 dB | 0.0145 |
| 9 | `test003` | **H11** | White matter R | [42.1, -68.8, 13.2] | ★ 一致正向增强 | +0.50 dB | 0.0200 |
| 10 | `test004` | **D5** | White matter R | [39.7, -24.6, -19.0] | ▼ 一致负向抑制 | -0.53 dB | 0.0020 |
| 11 | `test004` | **D10** | Middle temporal R | [55.8, -32.8, -10.3] | ▼ 一致负向抑制 | -0.38 dB | 0.0215 |
| 12 | `test004` | **E2** | N/A | [NaN, NaN, NaN] | ★ 一致正向增强 | +0.50 dB | 0.0345 |
| 13 | `test004` | **J3** | Amygdala R | [18.7, -6.9, -15.8] | ★ 一致正向增强 | +0.34 dB | 0.0375 |
| 14 | `test004` | **J6** | Hippocampus R | [26.9, -13.0, -17.7] | ▼ 一致负向抑制 | -0.38 dB | 0.0215 |
| 15 | `test004` | **L5** | Hippocampus R | [29.6, -19.7, -12.4] | ★ 一致正向增强 | +0.32 dB | 0.0370 |
| 16 | `test005` | **B6** | Fusiform R | [34.7, -49.6, -15.8] | ▼ 一致负向抑制 | -0.40 dB | 0.0155 |
| 17 | `test005` | **H8** | Lateral occipital R | [47.5, -73.6, -3.9] | ★ 一致正向增强 | +0.34 dB | 0.0460 |
| 18 | `test006` | **B5** | Inferior temporal L | [-49.4, -49.3, -11.6] | ▼ 一致负向抑制 | -0.46 dB | 0.0095 |
| 19 | `test006` | **C5** | Middle temporal L | [-56.5, -45.5, -2.1] | ★ 一致正向增强 | +0.34 dB | 0.0385 |
| 20 | `test007` | **C3** | Hippocampus L | [-22.1, -11.8, -21.5] | ★ 一致正向增强 | +0.48 dB | 0.0190 |
| 21 | `test007` | **F8** | White matter L | [-30.2, -44.5, 36.8] | ★ 一致正向增强 | +0.54 dB | 0.0155 |
| 22 | `test007` | **H5** | Cuneus L | [-16.9, -70.0, 31.1] | ★ 一致正向增强 | +0.50 dB | 0.0175 |
| 23 | `test008` | **A10** | N/A | [NaN, NaN, NaN] | ★ 一致正向增强 | +0.46 dB | 0.0125 |
| 24 | `test008` | **A11** | N/A | [NaN, NaN, NaN] | ★ 一致正向增强 | +0.37 dB | 0.0460 |
| 25 | `test008` | **D3** | N/A | [NaN, NaN, NaN] | ★ 一致正向增强 | +0.46 dB | 0.0415 |
| 26 | `test008` | **H7** | N/A | [NaN, NaN, NaN] | ▼ 一致负向抑制 | -0.53 dB | 0.0010 |

---

## 2. 文件夹管理结构说明

```
color_analyse_0825/
├── qc/
│   ├── screening_multiband/       # C04 多频段初筛柱状总览图 (8位被试)
│   └── high_gamma_timecourses/    # C05 High-Gamma 26个精选电极时程图
│       ├── test001/               # D14, F14
│       ├── test002/               # D3, D4, D6
│       ├── test003/               # D11, E14, G5, H11
│       ├── test004/               # D5, D10, E2, J3, J6, L5
│       ├── test005/               # B6, H8
│       ├── test006/               # B5, C5
│       ├── test007/               # C3, F8, H5
│       └── test008/               # A10, A11, D3, H7
└── process_data/
    └── selected_timecourses/
        └── high_gamma/            # 对应的 26 个 .mat 原始时程与全试次数据
```
