# 颜色分析管线 v2 报告

**分析日期**：2026-08-06

## 1. Executive summary

本报告基于 7 位被试 21 个 HDF5，当前运行时间窗为 100-400 ms，信号变体为 低频 1-30 Hz / HDF5 1-200 Hz。

| variant | common_centers | strategy1 | strategy2 | N2_union | CSC | CSC_merged | CSC_fdr |
|---|---|---|---|---|---|---|---|
| 100-400_lf30 | 548 | 19 | 46 | 110 | 12 | 12 | 0 |
| 100-400_raw200 | 548 | 16 | 44 | 110 | 10 | 10 | 0 |

功能筛选标准（2026-08-05 起）：`strategy1` = 二因素 ANOVA（颜色 × 类别）类型 II 颜色主效应 p<0.05；`strategy2` = 任一类别 Welch t 检验 p<0.05（两组 ANOVA 等价于 t 检验）。MWU pooled（`strategy1_merged`）与四类各自显著性（`*_p_raw`）保留为信息列；FDR 仅作信息列，不作筛选门槛。

当前主时间窗：100-400 ms；正式电极筛选使用 Task 1 的 ANOVA 颜色主效应。

注意：跨通道不校正的 raw p 筛选与完全随机假设下的期望数量级一致（详见 stage01_selection 表），CSC 结论应视为探索性。

## 2. 变体 100-400_lf30（100-400 ms / low-frequency 1-30 Hz）

### 电极集合计数

| subject | window | signal | common_all_task_centers | strategy1 | strategy2 | N2_PC | N2_CC | N2_AC | N2_union | CSC | CSC_merged | CSC_strategy1 | CSC_strategy2 | color_with_sti_in_common |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test001 | 100-400 | lf30 | 74 | 7 | 15 | 8 | 13 | 18 | 29 | 4 | 4 | 3 | 4 | 9 |
| test002 | 100-400 | lf30 | 58 | 0 | 5 | 17 | 3 | 0 | 17 | 2 | 2 | 0 | 2 | 5 |
| test003 | 100-400 | lf30 | 109 | 2 | 4 | 0 | 3 | 21 | 21 | 0 | 0 | 0 | 0 | 8 |
| test004 | 100-400 | lf30 | 46 | 6 | 10 | 0 | 0 | 8 | 8 | 2 | 2 | 0 | 2 | 0 |
| test005 | 100-400 | lf30 | 75 | 1 | 1 | 0 | 7 | 15 | 15 | 0 | 0 | 0 | 0 | 4 |
| test006 | 100-400 | lf30 | 88 | 0 | 2 | 0 | 0 | 6 | 6 | 0 | 0 | 0 | 0 | 2 |
| test007 | 100-400 | lf30 | 98 | 3 | 9 | 0 | 0 | 14 | 14 | 4 | 4 | 1 | 3 | 0 |

### CSC 电极（test001:D6, test001:D8, test001:G5, test001:G6, test002:A4, test002:A5, test004:D6, test004:D7, test007:C10, test007:D7, test007:D8, test007:D9）

### 幅度统计：四水果 ANOVA 12 通道，red-vs-green MWU 24 条

### 频谱级解码（置换 p）

| analysis | n_electrodes | mean_accuracy | min_p | n_p_lt_0_05 |
|---|---|---|---|---|
| task2_gray_memory_color | 12 | 0.4853 | 0.034 | 1 |
| task3_red_green | 12 | 0.5722 | 0.002 | 7 |

## 2. 变体 100-400_raw200（100-400 ms / raw 1-200 Hz (HDF5 epochs)）

### 电极集合计数

| subject | window | signal | common_all_task_centers | strategy1 | strategy2 | N2_PC | N2_CC | N2_AC | N2_union | CSC | CSC_merged | CSC_strategy1 | CSC_strategy2 | color_with_sti_in_common |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test001 | 100-400 | raw200 | 74 | 5 | 16 | 8 | 13 | 18 | 29 | 4 | 4 | 2 | 4 | 9 |
| test002 | 100-400 | raw200 | 58 | 0 | 3 | 17 | 3 | 0 | 17 | 2 | 2 | 0 | 2 | 5 |
| test003 | 100-400 | raw200 | 109 | 2 | 3 | 0 | 3 | 21 | 21 | 0 | 0 | 0 | 0 | 8 |
| test004 | 100-400 | raw200 | 46 | 5 | 10 | 0 | 0 | 8 | 8 | 1 | 1 | 0 | 1 | 0 |
| test005 | 100-400 | raw200 | 75 | 1 | 2 | 0 | 7 | 15 | 15 | 0 | 0 | 0 | 0 | 4 |
| test006 | 100-400 | raw200 | 88 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 | 0 | 0 | 2 |
| test007 | 100-400 | raw200 | 98 | 3 | 10 | 0 | 0 | 14 | 14 | 3 | 3 | 0 | 3 | 0 |

### CSC 电极（test001:D4, test001:D6, test001:G5, test001:G6, test002:A4, test002:A5, test004:D6, test007:C10, test007:D8, test007:D9）

### 幅度统计：四水果 ANOVA 10 通道，red-vs-green MWU 20 条

### 频谱级解码（置换 p）

| analysis | n_electrodes | mean_accuracy | min_p | n_p_lt_0_05 |
|---|---|---|---|---|
| task2_gray_memory_color | 10 | 0.5048 | 0.001 | 1 |
| task3_red_green | 10 | 0.5963 | 0.001 | 6 |

## 3. S1/ANOVA 与当前假设分析范围

当前主电极选择固定为 Task 1 二因素 ANOVA 的 color 主效应 `p<0.05`。`100-400_lf30` 共 19 个 S1 电极，整体二项富集 p=0.0096；`100-400_raw200` 共 16 个 S1 电极，整体二项 p=0.0633，仅作敏感性分析。

本次没有把旧版 1–300 ms 的 H2/H3/TGM 结果复制到新目录。100–400 ms 的 S1/S2 时间分辨曲线见 `stage06_exploration/`；单电极 permutation p 尚未运行，当前曲线为描述性结果。

## 4. 刺激亮度审计

| category | n_pairs | mean_abs_dLum | max_abs_dLum | mean_abs_dContrast | wilcoxon_p_lum_color_vs_gray | flag_luminance_mismatch | mean_colorfulness_color | category_x_condition_interaction_p | flag_category_interaction | max_category_mean_abs_dLum |
|---|---|---|---|---|---|---|---|---|---|---|
| face | 70 | 1.999 | 3.959 | 1.136 | 3.558e-13 | False | 35.88 | 1.159e-09 | False | 1.999 |
| object | 70 | 0.6928 | 3.088 | 0.8356 | 0.003337 | False | 41.15 | 1.159e-09 | False | 1.999 |
| body | 70 | 1.353 | 8.627 | 0.7075 | 3.345e-09 | False | 30.04 | 1.159e-09 | False | 1.999 |
| place | 70 | 0.6596 | 2.758 | 0.2905 | 0.07865 | False | 37.98 | 1.159e-09 | False | 1.999 |

详见 stage04_luminance/。

## 5. 关键图

- [electrode_set_counts_100-400_lf30.png](stage01_selection/figures/electrode_set_counts_100-400_lf30.png)
- [electrode_set_counts_100-400_raw200.png](stage01_selection/figures/electrode_set_counts_100-400_raw200.png)
- [patch_and_csc_mni_projections_100-400_lf30.png](stage01_selection/figures/patch_and_csc_mni_projections_100-400_lf30.png)
- [patch_and_csc_mni_projections_100-400_raw200.png](stage01_selection/figures/patch_and_csc_mni_projections_100-400_raw200.png)
- [test001_cross_overlap_pies_100-400_lf30.png](stage01_selection/figures/test001_cross_overlap_pies_100-400_lf30.png)
- [test001_cross_overlap_pies_100-400_raw200.png](stage01_selection/figures/test001_cross_overlap_pies_100-400_raw200.png)
- [test002_cross_overlap_pies_100-400_lf30.png](stage01_selection/figures/test002_cross_overlap_pies_100-400_lf30.png)
- [test002_cross_overlap_pies_100-400_raw200.png](stage01_selection/figures/test002_cross_overlap_pies_100-400_raw200.png)
- [test003_cross_overlap_pies_100-400_lf30.png](stage01_selection/figures/test003_cross_overlap_pies_100-400_lf30.png)
- [test003_cross_overlap_pies_100-400_raw200.png](stage01_selection/figures/test003_cross_overlap_pies_100-400_raw200.png)
- [test004_cross_overlap_pies_100-400_lf30.png](stage01_selection/figures/test004_cross_overlap_pies_100-400_lf30.png)
- [test004_cross_overlap_pies_100-400_raw200.png](stage01_selection/figures/test004_cross_overlap_pies_100-400_raw200.png)
- [test005_cross_overlap_pies_100-400_lf30.png](stage01_selection/figures/test005_cross_overlap_pies_100-400_lf30.png)
- [test005_cross_overlap_pies_100-400_raw200.png](stage01_selection/figures/test005_cross_overlap_pies_100-400_raw200.png)
- [test006_cross_overlap_pies_100-400_lf30.png](stage01_selection/figures/test006_cross_overlap_pies_100-400_lf30.png)
- [test006_cross_overlap_pies_100-400_raw200.png](stage01_selection/figures/test006_cross_overlap_pies_100-400_raw200.png)
- [test007_cross_overlap_pies_100-400_lf30.png](stage01_selection/figures/test007_cross_overlap_pies_100-400_lf30.png)
- [test007_cross_overlap_pies_100-400_raw200.png](stage01_selection/figures/test007_cross_overlap_pies_100-400_raw200.png)

- [csc_task2_gray_fruit_amplitude_100-400_lf30.png](stage02_amplitude_spectral/figures/csc_task2_gray_fruit_amplitude_100-400_lf30.png)
- [csc_task2_gray_fruit_amplitude_100-400_raw200.png](stage02_amplitude_spectral/figures/csc_task2_gray_fruit_amplitude_100-400_raw200.png)
- [csc_task2_spectral_100-400_lf30.png](stage02_amplitude_spectral/figures/csc_task2_spectral_100-400_lf30.png)
- [csc_task2_spectral_100-400_raw200.png](stage02_amplitude_spectral/figures/csc_task2_spectral_100-400_raw200.png)

- [decoding_task2_gray_memory_color_100-400_lf30.png](stage03_decoding/figures/decoding_task2_gray_memory_color_100-400_lf30.png)
- [decoding_task2_gray_memory_color_100-400_raw200.png](stage03_decoding/figures/decoding_task2_gray_memory_color_100-400_raw200.png)
- [decoding_task3_red_green_100-400_lf30.png](stage03_decoding/figures/decoding_task3_red_green_100-400_lf30.png)
- [decoding_task3_red_green_100-400_raw200.png](stage03_decoding/figures/decoding_task3_red_green_100-400_raw200.png)

## 6. Limitations

1. 功能筛选使用 raw 双尾 p<0.05（跨通道/通道内 FDR 均不加门槛）；在该阈值下显著通道数量与全零假设期望一致，CSC 为探索性集合。
2. 解码为频谱级（单窗 16 频带特征）探索，置换 1000 次；逐时间点解码与 cluster 校正在下一阶段实现，本报告不报告未校正的逐时间点显著窗。
3. 规范二使用论文群体 Talairach→MNI 坐标和 20 mm 阈值，不是被试个体化 fMRI。
4. 记忆颜色类内方向一致性诊断见 stage03_decoding/；类内方向翻转会削弱线性 cross-fruit 解码。
5. 样本重复检查：Task 1 无重复（70 图/条件）；Task 3 每色仅 3 张唯一图片，但 HDF5 未保留 trial-level 图片身份，因此不能把平均 epoch/图片数写成精确重复次数；物理颜色解码可能部分受样本级特征影响。Task 2 每水果 15 张（约 4 次重复），但记忆颜色解码采用 leave-one-fruit-pair-out，对样本泄漏免疫。详见 stage06_exploration/exemplar_identity_audit_actual_trials.csv。

## 7. Reproducibility

运行入口：`analysis/run_final_analysis.py --perms 1000 --workers 8`；参数见 analysis_parameters.json。
