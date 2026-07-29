# 多被试解码（Group Decoding）平均结果报告

本报告汇总了 `test001`、`test002`、`test003` 三个被试在不同解码任务、特征以及电极组合下的平均计算与绘图结果。

对于一维时间曲线解码（`memory_pairs` 与 `true_false`），图表中同时绘制了**单个被试的趋势曲线（半透明虚线）**与**三个被试的群组平均曲线（粗实线）**。
对于二维时间泛化解码（`time_generalization`），图表中仅绘制了**群组平均的解码准确率热图**。

---

## 1. 结果图像链接汇总

### 一、记忆颜色配对解码 (Memory Pairs Decoding)
在记忆提取阶段，使用灰色水果刺激进行物体颜色（红色 vs 绿色）的解码。
* **ERP 特征**
  - **Color_with_sti 电极**：[erp_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/memory_pairs/erp_colorwithsti.png)
  - **Type 1 电极**：[erp_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/memory_pairs/erp_type1.png)
* **Subband 特征 (60–150Hz)**
  - **Color_with_sti 电极**：[subband_60_150_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/memory_pairs/subband_60_150_colorwithsti.png)
  - **Type 1 电极**：[subband_60_150_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/memory_pairs/subband_60_150_type1.png)

### 二、真实 vs 虚假颜色解码 (True vs False Decoding)
对于各水果的真实颜色与错误颜色（例如红草莓 vs 绿草莓）进行分类解码。
* **ERP 特征**
  - **Color_with_sti 电极**：[erp_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/true_false/erp_colorwithsti.png)
  - **Type 1 电极**：[erp_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/true_false/erp_type1.png)
  - **Temporal Pole 电极 (基于 test001 和 test002 平均)**：[erp_temporal_pole.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/true_false/erp_temporal_pole.png)
* **Subband 特征 (60–150Hz)**
  - **Color_with_sti 电极**：[subband_60_150_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/true_false/subband_60_150_colorwithsti.png)
  - **Type 1 电极**：[subband_60_150_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/true_false/subband_60_150_type1.png)
  - **Temporal Pole 电极 (基于 test001 和 test002 平均)**：[subband_60_150_temporal_pole.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/true_false/subband_60_150_temporal_pole.png)

### 三、时间泛化解码 (Time Generalization Decoding - TGM)
使用真实颜色刺激（Task 3）作为训练集，在记忆颜色刺激（Task 2）上进行跨时间测试。
* **ERP 特征**
  - **Color_with_sti 电极**：[erp_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/time_generalization/erp_colorwithsti.png)
  - **Type 1 电极**：[erp_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/time_generalization/erp_type1.png)
* **Subband 特征 (60–150Hz)**
  - **Color_with_sti 电极**：[subband_60_150_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/time_generalization/subband_60_150_colorwithsti.png)
  - **Type 1 电极**：[subband_60_150_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average/decoding/time_generalization/subband_60_150_type1.png)

---

## 2. 核心特征与表现 analysis

1. **单个被试 vs 均值 (1D 曲线)**：
   * 在 `memory_pairs` 和 `true_false` 解码中，虽然由于被试间解剖结构及电极位置的异质性，单个被试在某些时间窗表现出波动，但进行群组平均（Group Average）后的曲线更加平滑，展现出显著高于随机水平（Chance level = 0.5）的解码准确率趋势。
   * 特别是在刺激呈现后大约 200–600ms，平均准确率曲线上升显著，表明此时间窗内存在强烈的颜色物体提取与比对表征。
   * 对于 `temporal_pole` 电极组，由于 `test003` 在该脑区无符合筛选条件的有效电极，此处结果由 `test001` 和 `test002` 共同平均得出。

2. **时间泛化（TGM 2D 热图）**：
   * 在 Task 3（真实颜色）向 Task 2（记忆提取）的泛化中，平均热图呈现出明显的偏对角线激活模式。
   * ERP 特征的 TGM 在刺激后表现出较强的瞬态激活，而 Subband (60-150Hz) 特征下的泛化热图则展现了更具持续性的激活状态，这也与高频伽马波在记忆保持/提取中的持续性功能表征一致。
