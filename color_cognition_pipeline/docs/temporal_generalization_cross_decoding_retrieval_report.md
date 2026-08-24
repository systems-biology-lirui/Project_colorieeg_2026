# 🧠 跨范式时间泛化 (Cross-Decoding / Temporal Generalization) 既有成果全库检索报告

本报告全盘检索了 `Project_colorieeg_2026` 根目录下过去所有关于**跨范式时间泛化 (Temporal Generalization Matrix, TGM)** 的结果图像、数据表格、原始 `.npy` 计算矩阵以及对应的分析脚本。

时间泛化分析采用 **Task 3 物理纯色块 (Train) $\rightarrow$ Task 2 灰色水果 (Test)** 的跨范式训练测试框架，探究感知颜色与隐性常识颜色在不同时间步上的共享表征迁移。

---

## 🖼️ 一、 组水平跨范式时间泛化热图 (Group-level TGM Heatmaps)

组水平热图涵盖了不同筛选规则和先验通道集合在跨 5 名被试下的 2D 表征迁移矩阵：

| 图形类型与方案 | 图形文件路径 (Clickable Link) | 包含策略/集合说明 |
| :--- | :--- | :--- |
| **Strategy 1 独立集合** | [strategy1_group_temporal_generalization.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/cross_decoding/strategy1_group_temporal_generalization.png) | 基于 Strategy 1 选中的通道导出的组水平 2D TGM |
| **Strategy 1 并集集合** | [strategy1_group_temporal_generalization_union.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/cross_decoding/strategy1_group_temporal_generalization_union.png) | 取包含 Strategy 1 的联合集合导出的组水平 2D TGM |
| **Strategy 2 独立集合** | [strategy2_group_temporal_generalization.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/cross_decoding/strategy2_group_temporal_generalization.png) | 基于 Strategy 2 选中的通道导出的组水平 2D TGM |
| **Strategy 2 并集集合** | [strategy2_group_temporal_generalization_union.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/cross_decoding/strategy2_group_temporal_generalization_union.png) | 取包含 Strategy 2 的联合集合导出的组水平 2D TGM |
| **`color_with_sti` 先验通道** | [erp_color_with_sti_cross_decoding_tg_heatmap_group.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/color_with_sti/erp_color_with_sti_cross_decoding_tg_heatmap_group.png) | 仅针对具备电刺激光幻视/颜色觉效应的 18 个先验通道 |
| **跨范式 1D 解码与 GLMM** | [group_glmm_task2_to_task3_erp_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_glmm_task2_to_task3_erp_decoding.png) | 跨范式 1D 概率曲线与 5 被试二项 GLMM 显著时间窗 |

---

## 🎯 二、 单电极跨范式时间泛化矩阵与原始数据 (.npy)

除组水平结果外，项目还针对 38 个色块敏感通道逐个输出了单电极 2D 时间泛化矩阵：

### 1. 单通道时间泛化图像 (部分代表性示例)
- **`test001 - B5`**：[test001_B5_cross_decoding_generalization.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/single_electrode/cross_decoding/test001_B5_cross_decoding_generalization.png)
- **`test002 - C1`**：[test002_C1_cross_decoding_generalization.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/single_electrode/cross_decoding/test002_C1_cross_decoding_generalization.png)
- **`test003 - H13`**：[test003_H13_cross_decoding_generalization.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/single_electrode/cross_decoding/test003_H13_cross_decoding_generalization.png)
- **`test005 - E14`**：[test005_E14_cross_decoding_generalization.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/single_electrode/cross_decoding/test005_E14_cross_decoding_generalization.png)
*(注：该目录下包含了全部 5 被试 38 个敏感通道的单独 2D TGM 图)*

### 2. 既有原始数值矩阵 (.npy)
在中期分析目录中，还保存了单被试 ERP 与 High Gamma (60-150Hz) 信号的 2D TGM 原始数值矩阵：
- `images/test001/decoding/time_generalization/erp_type1_tgm.npy`
- `images/test001/decoding/time_generalization/erp_colorwithsti_tgm.npy`
- `images/test001/decoding/time_generalization/subband_60_150_type1_tgm.npy`
- `images/test002/decoding/time_generalization/erp_type1_tgm.npy`
- `images/test003/decoding/time_generalization/erp_type1_tgm.npy`

---

## 📊 三、 时间泛化数据表格与分析脚本

### 1. 数据表格文件
- [cross_decoding_tg_strategy1.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/doc/cross_decoding_tg_strategy1.xlsx) (及 `.csv`)
- [cross_decoding_tg_strategy1_union.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/doc/cross_decoding_tg_strategy1_union.xlsx)
- [cross_decoding_tg_strategy2.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/doc/cross_decoding_tg_strategy2.xlsx)
- [cross_decoding_tg_color_with_sti.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/doc/cross_decoding_tg_color_with_sti.xlsx)

### 2. 核心分析与计算脚本
- `analyse_0617/code/step3_2_cross_decoding_generalization.py` (组水平热图导出)
- `analyse_0617/code/step3_2_cross_decoding_generalization_union.py` (并集组水平热图导出)
- `analyse_0617/code/step3_3_single_electrode_generalization.py` (单通道 TGM 热图批量导出)
- `analyse_0720/notebooks/05_cross_task_and_true_false.ipynb` (0720 新测试 Notebook)
