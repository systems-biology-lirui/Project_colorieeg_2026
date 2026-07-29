codexject_colorieeg_2026

Intracranial EEG (SEEG) 颜色认知分析工程，包含完整的预处理、特征提取、ROI decoding、CCEP 分析和实验控制代码。

项目核心问题：**颜色信息在人脑颞叶皮层的神经编码机制**。通过 SEEG 记录视觉刺激（真实/灰度图像、颜色色块、记忆颜色物体）下的神经活动，使用多模态 decoding 和时频分析来定位颜色表征的时空特征。

---

## 目录总览

| 目录 | 用途 | 主要语言 |
|------|------|---------| 
| `newanalyse/` | 主分析管线：预处理(Sec1) → 特征提取(Sec2) → decoding/统计(Sec3) → 批处理(Sec4) → 重绘/汇总(Sec5) | MATLAB + Python |
| `ccepcode/` | CCEP 单脉冲电刺激分析管线，结构与 newanalyse 平行 | MATLAB + Python |
| `ccepcode_standalone/` | CCEP 自包含版本（含数据/workspace/文档），可直接迁移独立运行 | MATLAB + Python |
| `feature/` | ROI 级特征文件，newanalyse Sec2 脚本的输出 / Sec3 脚本的输入 | |
| `processed_data/` | 被试级预处理数据：task epoched .mat + 电极定位表 + groupedData | |
| `seegdata/` | 原始连续 SEEG（EEGLAB .set + .fdt），按被试/run 分 | |
| `rawdata/` | 原始 BDF 格式数据（按采集 session 分文件夹） | |
| `result/` | 管线分析结果 | |
| `testcode/` | 配对 decoding 验证脚本 + 统计方法对比 + 审计工具 | Python |
| `visual_experiment/` | 实验范式代码（Psychtoolbox）+ 任务配置 + 刺激材料 | MATLAB |
| `mne_data/` | MNE-Python 模板数据（fsaverage） | |
| `python_libs/` | Python site-packages 副本（conda 环境，已 gitignore） | |
| `trigger.xlsx` | 任务触发码表 | |
| `ccep.zip` | CCEP 数据/代码归档 | |

---

## 被试信息

| ID | 姓名 | 采集日期 | processed_data | seegdata (EEGLAB) | rawdata (BDF) |
|----|------|---------|---------------|-------------------|---------------|
| test001 | 孙冠东 | 2026-01-13 | ✅ task mat + ieegloc + processed .set + ccep 子目录 | test1（erp1/2/3 + ccep） | ✅ 6 个 session |
| test002 | 金宇佳 | 2026-02-03 | ✅ task mat + ieegloc（无 processed .set，无 ccep） | test2（erp1/2/3） | ✅ 7 个 session |
| test003 | 冉龙建 | 2026-04-08 | ✅ task mat + ieegloc（无 processed .set，无 ccep） | test3（erp1/2/3） | ✅ 4 个 session |

> **说明**：`processed_data/subject.txt` 仅记录了 test001 和 test002 的姓名映射。

### processed_data 结构

**所有被试共有**：

```
task{1,2,3}_ERP_epoched.mat     # ERP epoched: [Cond, Rep, Ch, Time]
task{1,2,3}_TFA_epoched.mat     # TFA epoched: [Cond, Rep, Ch, Time]
{subject}_ieegloc.xlsx           # 电极定位表 (AAL3/ROI + SCS/MNI/World 坐标)
groupedData.mat                  # task1 color-gray 配对表
task3groupedData.mat             # task3 配对表
```

**仅 test001 额外包含**：

```
processed_ERP.set/.fdt           # 预处理连续数据 (ERP: 1-30 Hz)
processed_TFA.set/.fdt           # 预处理连续数据 (TFA: 1-150 Hz)
selchannel.jpg                   # 选通道参考图
ccep/                            # CCEP 预处理输出
  ├── ccep_ERP_epoched.mat
  ├── ccep_TFA_epoched.mat
  ├── ccep_block_summary.tsv
  └── processed_ERP/TFA.set/.fdt
```

### feature 目录结构

```
feature/
├── erp/               # ERP ROI 特征 (Sec2_1)
├── highgamma/         # 70-150 Hz 高 gamma 包络 (Sec2_2)
├── lowgamma/          # 30-70 Hz 低 gamma 包络 (Sec2_3)
├── tfa/               # 1-150 Hz 宽带时域 (Sec2_4)
├── gamma/             # 30-100 Hz 宽频 gamma 包络 (Sec2_5)
├── gamma_multiband/   # 多频段 gamma，含 band 元数据 (Sec2_6)
├── ccep_erp/          # CCEP ERP 特征
├── ccep_tfa/          # CCEP TFA 特征
└── erp_bad_epoch/     # ERP 坏段标记
```

> **注意**：当前仅 test001 在各 feature 子目录下有输出文件。

---

## 采集参数

| 参数 | 值 |
|------|------|
| 原始采样率 | 1000 Hz |
| 预处理降采样 | 500 Hz |
| 参考方式 | SEEG 局部混合重参考（1D Laplace + Bipolar） |
| 陷波滤波 | 50 / 100 / 150 Hz |
| 高通滤波 | 1 Hz |
| ERP 分支带通 | 1–30 Hz |
| TFA 分支带通 | 1–150 Hz |
| 分段窗 | −500 ~ 1000 ms（Sec2 再裁剪至 −100 ~ 1000 ms，550 点） |
| 基线校正 | −250 ~ −50 ms |

---

## 核心数据契约

### 任务条件顺序

> ⚠️ **重要**：实验范式代码（`visual_experiment/Configs/`）中的 task 编号与数据分析管线（`newanalyse/`）中的 task 编号**存在颠倒关系**。具体来说：`config_task2.m` 的输出对应分析中的 task3（颜色色块），`config_task3.m` 的输出对应分析中的 task2（水果记忆颜色）。以下以**分析管线（newanalyse）的命名**为准。

**task1** — Passive Real/Gray（4 类别 × 颜色/灰度 = 8 条件）：

| 条件索引 | 名称 | 触发码 |
|----------|------|--------|
| 0 | face_color | 11 |
| 1 | face_gray | 12 |
| 2 | object_color | 21 |
| 3 | object_gray | 22 |
| 4 | body_color | 31 |
| 5 | body_gray | 32 |
| 6 | place_color | 41 |
| 7 | place_gray | 42 |

**task2** — Fruit Memory Color（4 水果 × 颜色/灰度 = 8 条件）：

| 条件索引 | 名称 | 说明 |
|----------|------|------|
| 0 | strawberry_color | 草莓真实颜色 |
| 1 | kiwi_color | 猕猴桃真实颜色 |
| 2 | cabbage_color | 卷心菜真实颜色 |
| 3 | watermelon_color | 西瓜真实颜色 |
| 4 | strawberry_gray | 草莓灰度 |
| 5 | kiwi_gray | 猕猴桃灰度 |
| 6 | cabbage_gray | 卷心菜灰度 |
| 7 | watermelon_gray | 西瓜灰度 |

> 实验范式源码（`config_task3.m`）中每种水果实际有 3 个条件（True/False/Gray，各自触发码如 101/102/103），在预处理阶段被重组为上述 8 条件格式。

**task3** — Color Patches（6 种纯色块）：

| 条件索引 | 名称 | 触发码 |
|----------|------|--------|
| 0 | red | 51 |
| 1 | yellow | 52 |
| 2 | blue | 53 |
| 3 | green | 54 |
| 4 | black | 55 |
| 5 | white | 56 |

### Epoched 数据格式

所有 `*_ERP_epoched.mat` 和 `*_TFA_epoched.mat` 均为 `[Cond, Rep, Ch, Time]`。`TFA` 分支是 1-150 Hz 宽带时域信号（**不是**时频图）。

### task 触发码

触发码表详见 `trigger.xlsx`。newanalyse 各脚本通过 `TASKS` 配置字典中的 `trigger_codes` 指定使用的任务，当前默认以 task1 为主。

### ROI 映射

- 电极定位表 `{subject}_ieegloc.xlsx` 提供解剖 ROI（AAL3）+ 功能 ROI
- 通道可同时属于多个 ROI **（非互斥）** — 影响结果独立性的解释
- 标准解剖 ROI 分组：fusiform, temporal_inf, temporal_mid, temporal_sup, temporal_pole_mid, temporal_pole_sup, parahippocampal, amygdala, hippocampus
- 功能 ROI：Color_with_sti, Color_patch

### groupedData 配对

`groupedData.mat` 内变量名 `groupedData`，4×2 cell array（行=类别，列1=color trial id，列2=gray trial id），用于配对 decoding 以消除 trial 间差异。

---

## 管线流程

### newanalyse（主分析）

> 详细的代码说明与核查结论请参阅 [`newanalyse/readme.md`](newanalyse/readme.md)。

#### 主管线脚本

```
Sec1_preanalyse.m                     # 原始连续数据 → task epoched .mat
Sec2_1~6_preprocess_*.m               # epoched → ROI 特征（erp/hg/lg/tfa/gamma/gmb）
Sec3_1~6_all_roi_result_*.py          # ROI decoding + 显著性
Sec3_7_all_roi_result_cross.py        # Cross decoding（task1→task2 等）
Sec3_8_all_roi_result_cross_category_average.py  # 跨类别平均
Sec3_s1~s4_*.py                       # ROI 条件比较, TFA, 电极重要性
Sec4_1~4_*.py                         # 批处理入口
Sec5_1~4_*.py                         # 重绘、HTML 报告、显著 ROI 追踪
```

#### 辅助文件

| 文件 | 用途 |
|------|------|
| `newanalyse_paths.py` | 项目级路径生成器（feature/result/processed_data 等） |
| `newanalyse_paths.m` | MATLAB 版路径配置 |
| `newanalyse_load_run_config.m` | MATLAB 端运行时配置加载器 |
| `runtime_config.py` | Python 端运行时配置（subject/超参数覆写） |
| `batch_runner_utils.py` | 批处理辅助工具函数 |
| `get_roi_map.m` | ROI 映射函数（从 ieegloc.xlsx 提取） |
| `groupeddata_pairing.py` | groupedData 配对逻辑实现 |
| `IEEGDataAnalyzer.m` | iEEG 数据分析工具类 |
| `preanalyse.m` | Sec1 的旧版本（保留参考） |
| `sec1_config.json` | Sec1 的 JSON 配置文件 |
| `sec4_batch_config_example.json` | Sec4 批处理配置示例 |
| `Sec2_s1_check_highgamma_envelope.m` | 高 gamma 包络质量检查 |
| `Sec2_s2_plot_erp_task1_channels.py` | ERP 单通道波形绘制 |

#### 模态定义

| 模态 | 频段 | 来源 | 说明 |
|------|------|------|------|
| ERP | 1-30 Hz | ERP 分支 | 时域 |
| highgamma (hg) | 70-150 Hz 多频段 | TFA 分支提取 | 8 个子带 (10 Hz step)，Hilbert 包络 + 平方 + 平均 |
| lowgamma (lg) | 30-70 Hz | TFA 分支提取 | 同 hg 流程 |
| tfa | 1-150 Hz | TFA 分支 | 宽带时域（**不是**时频图） |
| gamma (g) | 30-100 Hz | TFA 分支提取 | 宽带包络 |
| gamma_multiband (gmb) | 多频段 | TFA 分支提取 | 含 band 元数据，最自描述的格式 |

### ccepcode（CCEP 子项目）

```
Sec1_ccep_preanalyse.m               # CCEP 连续数据预处理
Sec2_ccep_preprocess_roi_features.m   # ROI 特征提取
Sec3_ccep_electrode_response_stats.py # 逐电极统计 + 图表
Sec4_ccep_batch_run.py                # 批处理
ccep_batch_config_example.json        # 批处理配置示例
```

### CCEP 自包含版本（`ccepcode_standalone/`）

复制了 `ccepcode/` 的代码 + 数据 + workspace，可直接迁移。包含独立的 [README.md](ccepcode_standalone/README.md) 和 `CCEP预处理.pdf` 文档。

目录结构：

```
ccepcode_standalone/
├── code/              # 独立化的分析脚本（含 runtime_config.py、get_roi_map.m）
├── data/
│   ├── raw/           # 原始 CCEP .set/.fdt
│   └── metadata/      # 电极定位表
├── workspace/
│   ├── processed/     # CCEP 预处理输出
│   ├── feature/       # ROI 特征文件
│   └── result/        # 统计结果
├── README.md          # 运行说明
└── CCEP预处理.pdf     # 预处理流程文档
```

运行示例：

```bash
cd ccepcode_standalone
python code/Sec3_ccep_electrode_response_stats.py
python code/Sec4_ccep_batch_run.py --config code/ccep_batch_config_example.json
```

---

## 数据流

```
rawdata/{subject}/ (原始 BDF，按采集 session 分文件夹)
  │
  ├──→ seegdata/testN/erp{1,2,3}.set (EEGLAB 连续数据，1 kHz)
  │    └── testN/ccep.set (仅 test1 有)
  │
  └──→ newanalyse/Sec1_preanalyse.m
        ├── 降采样 500 Hz
        ├── 陷波 + HP + 局部重参考
        ├── ERP 分支 (1-30 Hz BP) → task*_ERP_epoched.mat
        │                              │
        │                         Sec2_1 → feature/erp/
        │                              │
        │                         Sec3_1 → result/ (ROI decoding)
        │
        └── TFA 分支 (1-150 Hz BP) → task*_TFA_epoched.mat
                                       │
                                  Sec2_2~6 → feature/{hg,lg,tfa,g,gmb}/
                                       │
                                  Sec3_2~6 → result/ (ROI decoding)
```

### result 输出路径约定

```
result/
├── decoding/{task_id}/{modality}/{subject}/{batch_name}/{perm_tag}/with_sti/
├── cross_decoding/{task_id}/{modality}/{subject}/{batch_name}/{perm_tag}/
├── roi_condition_tfa/{task}/{modality}/{subject}/{comparison_id}/
├── roi_electrode_condition/{task}/{modality}/{subject}/{comparison_id}/
├── all_electrode_decoding/{task_id}/{modality}/{subject}/{batch_name}/{perm_tag}/
├── smoothing_compare/{task_id}/{modality}/{subject}/{smooth_tag}/{perm_tag}/
├── reports/
└── ccep/{subject}/{modality}/
```

---

## visual_experiment（实验范式）

使用 MATLAB + Psychtoolbox 实现的视觉实验范式。

### 目录结构

```
visual_experiment/
├── main_experiment.m          # 主实验入口
├── run_heeg_experiment.m      # HEEG 实验运行器
├── HEEG_StimControl.m         # 刺激控制类
├── Configs/
│   ├── config_common.m        # 公共配置（屏幕、按键、时序、串口、注视点等）
│   ├── config_task1.m         # Task1: Passive Real/Gray（触发码 11-42）
│   ├── config_task2.m         # Task2: Color Patches（触发码 51-56）⚠️ 对应分析 task3
│   ├── config_task3.m         # Task3: Fruit Full（触发码 101-133）⚠️ 对应分析 task2
│   ├── config_task4~8.m       # 扩展任务配置
│   └── test.m / test.py       # 测试脚本
├── Utils/
│   ├── run_passive_phase.m    # 被动观看范式运行器
│   ├── run_estim_phase.m      # 电刺激阶段运行器
│   ├── calibration_task.m     # 亮度校准任务
│   ├── fixation_grid_task.m   # 注视网格任务
│   ├── io_utils.m             # I/O 工具（串口通信等）
│   ├── preprocess_luminance_alignment.m  # 亮度归一化预处理
│   └── trigger_test_tool.m    # 触发码测试工具
└── stimuli_pic/
    ├── Stimuli_Task1/         # 真实/灰度图像（face, object, body, place）
    ├── Stimuli_Task2/         # 颜色色块
    ├── Stimuli_Task2_origin/  # Task2 原始未处理刺激
    ├── Stimuli_Task3/         # 水果图像（true/false/gray）
    └── Stimuli_Task4~8/       # 扩展任务刺激
```

### 实验时序参数

| 参数 | 值 |
|------|------|
| 刺激呈现 | 300 ms |
| 空白期 | 0.9–1.2 s（50 ms 步长抖动） |
| 背景色 | RGB(100, 100, 100) |
| Catch trial 概率 | 10%（触发码 99） |
| 每类别图片数 | Task1: 35, Task2: 30, Task3: 20 |

---

## testcode（验证脚本）

| 脚本 | 用途 |
|------|------|
| `task1_paired_decoding/paired_group_cv_decoding.py` | 配对 grouped CV decoding |
| `task1_paired_decoding/within_pair_centered_decoding.py` | 减去 pair 均值后 decoding |
| `task1_paired_decoding/pair_difference_decoding.py` | (±) 差向量 decoding |
| `task1_paired_decoding/cross_category_average_group_cv_decoding.py` | 跨类别平均 grouped CV decoding |
| `task1_paired_decoding/plot_condition_cosine_similarity.py` | 条件间余弦相似度可视化 |
| `task1_paired_decoding/plot_condition_pca_trajectories.py` | 条件 PCA 轨迹可视化 |
| `task1_paired_decoding/plot_pairwise_pca.py` | 配对 PCA 分析 |
| `task1_paired_decoding/plot_single_trial_category_dissimilarity.py` | 单 trial 类别区分度 |
| `task1_paired_decoding/plot_single_trial_cross_category_average_dissimilarity.py` | 跨类别平均区分度 |
| `task1_paired_decoding/common.py` | 配对 decoding 公共模块 |
| `compare_acc_auc_significance_erp.py` | ERP 的 accuracy vs AUC 显著性比较 |
| `compare_perm_regimes_erp.py` | 不同置换检验方案对比 |
| `compare_task1_color_gray_fourclass.py` | Task1 color/gray 四分类比较 |
| `recompute_pointwise_significance.py` | 重计算逐点显著性 |
| `audit_sec3_s4_electrode_selection.py` | 审计电极选择标准 |

---

## 关键 Python 依赖

环境 `conda activate lr2026`，核心包：

- numpy, pandas, scipy, matplotlib, scikit-learn
- mne, joblib, h5py, openpyxl
- 无独立 requirements.txt（`python_libs/` = site-packages 副本，已 gitignore）

---

## 常用操作

```bash
conda activate lr2026

# MATLAB 预处理（在 MATLAB 中，项目目录下）
# newanalyse/Sec1_preanalyse.m          # 原始数据 → epoched
# newanalyse/Sec2_1_preprocess_erp.m    # ERP ROI 特征

# Python decoding
python newanalyse/Sec3_1_all_roi_result_erp.py          # 按需修改被试/任务
python newanalyse/Sec4_2_batch_run_modalities.py        # 批处理

# CCEP
python ccepcode/Sec3_ccep_electrode_response_stats.py

# CCEP 独立版本
cd ccepcode_standalone
python code/Sec3_ccep_electrode_response_stats.py

# 配对 decoding 验证
python testcode/task1_paired_decoding/paired_group_cv_decoding.py \
  --subject test001 --feature-kind erp --roi-name Color_with_sti \
  --grouped-data-mat processed_data/test001/groupedData.mat
```

---

## 迁移注意事项

1. **ROI 非互斥**：一个通道可跨多个 ROI（如 `G3/G4` 同时属于 `Color_with_sti` 和 `ParaHippocampal_R`），统计解释时需注意结果非独立。
2. **task2/3 编号颠倒**：实验范式 `config_task2.m` 对应分析管线的 task3（颜色色块），`config_task3.m` 对应分析管线的 task2（水果记忆颜色）。触发码表和 `Sec1_preanalyse.m` 内部有映射处理。
3. **TFA ≠ 时频图**：`tfa` 命名易混淆，它是 1-150 Hz 宽带时域信号。真正的 Morlet 时频分解在 `Sec3_s2_roi_condition_tfa.py`。
4. **`processed_TFA.set` 是已分段数据**（如 396 trials × 750 pts），不是连续数据。
5. **连续原始数据在 `seegdata/testN/erpN.set/.fdt`**，可用 h5py + np.fromfile 读取。
6. **`ccepcode_standalone/`** 是最易迁移的 CCEP 包（含数据），见其内部 README。
7. **`python_libs/`** 是 conda site-packages 副本，新机器应直接 `conda install` 而不是手动复制。
8. **`ccep.zip`** 可展开后替代 `ccepcode_standalone/` 的部分数据。
9. **min-trial 截断**：`Sec1_preanalyse.m` 会对每个 task 内所有条件截断到最小 trial 数，且保留最前面的 trial（非随机抽样），可能引入时间漂移偏差。
10. **显著性窗口不一致**：ERP/TFA 主 decoding 会屏蔽 20 ms 以前时间点，高/低 gamma 等模态默认不会，导致跨模态 earliest latency 不可直接比较。
11. **被试数据完整度差异**：test001 数据最完整（含 processed .set、ccep、selchannel.jpg），test002/003 仅有 epoched .mat 和定位表。
12. **Sec1 路径**：`Sec1_preanalyse.m` 顶部路径仍为 Windows 本地路径，Linux 环境需手动修改。
13. **gamma 宽频主线未纳入批处理**：`Sec2_5` 和 `Sec3_5` 仍为独立旧结构，不在 `Sec4_2_batch_run_modalities.py` 调度范围内。

---

## .gitignore 说明

当前 `.gitignore` 排除了大多数数据文件（.mat, .set, .fdt, .npy, .csv, .json, .pdf, 图片等），以及 `rawdata/`, `mne_data/`, `python_libs/` 整个目录。

例外白名单：

- `processed_data/test00{1,2,3}/groupedData.mat` — 配对数据已显式保留

---

## 版本与更新

- `newanalyse/readme.md` — 最后核查日期：2026-04-13（含代码 review 结论和已知问题清单）
- `ccepcode_standalone/README.md` — 独立 CCEP 包使用说明
