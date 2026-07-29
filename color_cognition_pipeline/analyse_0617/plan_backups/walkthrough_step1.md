# 电极筛选与可视化结果总结 (Step1)

我们已成功实现了核心电极与物理邻近扩展电极的筛选逻辑，并通过 `step1_select_channel.py` 完成了全部 3 个被试（test001、test002、test003）的分析及绘图。

## 筛选结果统计

### 1. 主要电极筛选 (`select_channel`)
必须满足：属于指定的 **3类核心 ROI**（枕叶、颞叶后/下部、颞叶前/上部），且满足 **4种统计检验策略** 之一。
- **test001**：筛选出 **18** 个电极（如 `B5`, `G11` 等）。
- **test002**：筛选出 **15** 个电极（如 `A3`, `H7` 等）。
- **test003**：筛选出 **7** 个电极（如 `G11`, `A11` 等）。

### 2. 邻近扩展电极筛选 (`more_select_channel`)
物理排布上与已筛选的主电极处于同一轴（shaft）且序号相差为 $\pm 1$ 的邻居，其 AAL3 标注必须属于 `'unknown'`, `'N/A'`（空值）或 `'parahippocampus'`（旁海马区），且必须满足 **4种统计策略** 之一。
- **test001**：筛选出 **1** 个扩展电极：
  - `H2`（主电极 `H1` 的邻居，符合策略 `3,4`）。
- **test002**：筛选出 **2** 个扩展电极：
  - `C3`（主电极 `C4` 的邻居，符合策略 `4`）；
  - `F9`（主电极 `F8` 的邻居，符合策略 `1,2,4`）。
- **test003**：无符合条件的物理邻近扩展电极。

---

## 典型筛选电极示例

以 **test001** 的典型电极 **B5** 为例，其同时满足了所有四种筛选策略（`stra1_2_3_4_B5.png`）。

![test001 B5 ERP 时程与幅值差异图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/stra1_2_3_4_B5.png)

*注：图中左侧在显著差异区域（p < 0.05）的底部以黄色（Color > Gray）或青色（Color < Gray）的方块进行标记；右侧为 100-400ms 内的平均波幅条形散点图对比，并标注了对应的 Wilcoxon 秩和检验 p 值。*

---

## 输出表格与文件链接

1. **筛选代码**：
   - [step1_select_channel.py](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_select_channel.py)
2. **汇总表格 (XLSX)**：
   - 主要电极汇总：[select_channel_summary.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/select_channel_summary.xlsx)
   - 扩展电极汇总：[more_select_channel_summary.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/more_select_channel_summary.xlsx)
3. **输出图表目录**：
   - 主要电极结果图：[select_channel/](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/select_channel/)
   - 扩展邻居电极结果图：[more_select_channel/](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/more_select_channel/)
