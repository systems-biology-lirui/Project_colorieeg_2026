# 纯色选择性指数计算与电极分布排布可视化计划 (Step1_2)

本计划旨在实现 `step1_2_color_selectivity.py` 脚本，提取在前一步筛选中选出的核心电极（`select_channel`）在 `task3`（纯色刺激）中的数据，进行颜色选择性指数（CSI）的统计学计算与绘图分析。

## User Review Required

> [!IMPORTANT]
> **分析逻辑与图表设计**：
> 1. **数据与信号类型**：分别针对 ERP 和 HG 两类特征数据，只针对上一步选出的靶区主要电极进行计算。
> 2. **颜色选择性指数 (CSI)**：
>    - **四色整体 (4-Color Overall)**：Kruskal-Wallis 检验，CSI 为 H-statistic，以 $p < 0.05$ 判定显著。
>    - **红绿对比 (Red vs Green)**：Wilcoxon 秩和检验，CSI 为 `abs(Z-statistic)`，以 $p < 0.05$ 判定显著。
>    - **黄蓝对比 (Yellow vs Blue)**：Wilcoxon 秩和检验，CSI 为 `abs(Z-statistic)`，以 $p < 0.05$ 判定显著。
> 3. **绘图样式与排布**：
>    - **横轴**：按对应子图 CSI 大小升序排列的电极 rank 顺序。
>    - **纵轴**：CSI 数值。
>    - **点的颜色**：根据电极 MNI Y 坐标从后脑到前脑逐渐从蓝色变成红色（使用 `coolwarm` 映射）。
>    - **黑圈标记**：对于满足“策略 4”的电极，圆点外面增加黑圈。
>    - **显著性虚线**：在横轴上以一条竖直虚线分隔不显著和显著电极。
> 4. **数据保存**：在画图前，将计算出的完整 CSI 统计表保存为 `analyse_0617/doc/` 目录下的 Excel 和 CSV 备份文件。

## Open Questions

> [!NOTE]
> 当前无开放性疑问。我们将严格按照设定的三种 CSI 指数计算公式和美学排布方案来执行。

## Proposed Changes

---

### 电极纯色选择性分析脚本

#### [NEW] [step1_2_color_selectivity.py](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_2_color_selectivity.py)
1. **数据加载**：
   - 载入 `analyse_0617/doc/select_channel_summary.xlsx`，提取出所有的核心主要电极。
   - 读取各被试 `task3_ERP_epoched.mat` 和 `task3_hg_subband.mat` 中的通道数据。
2. **CSI 计算与过滤**：
   - 提取各电极在 `[50, 400]` ms 内 4 种纯色条件（Red: 51, Yellow: 52, Blue: 53, Green: 54）的 trial 平均值。
   - 计算 4-Color KW H-stat, Red-Green ranksum abs(Z-stat), Yellow-Blue ranksum abs(Z-stat) 及其 p 值。
3. **绘图数据导出**：
   - 将计算结果整理为 `select_channel_color_selectivity_erp.xlsx` 和 `select_channel_color_selectivity_hg.xlsx` 并保存于 `analyse_0617/doc/` 目录下。
4. **CSI 渐变图绘制**：
   - 分别为 ERP 和 HG 绘制包含 3 个子图的 CSI 渐变排布分布图，保存于 `analyse_0617/result/select_channel/color_selectivity/`。
5. **备份计划**：
   - 拷贝本实施计划至 `analyse_0617/plan_backups/implementation_plan_step1_2.md`。

## Verification Plan

### Automated Tests
- 运行 `/home/lirui/anaconda3/envs/lr2026/bin/python color_cognition_pipeline/analyse_0617/code/step1_2_color_selectivity.py`。
- 确认产生两个包含 3 子图的渐变图与两份导出的 CSI 数据表格。

### Manual Verification
- 检查 `analyse_0617/result/select_channel/color_selectivity/` 下生成的 ERP 和 HG 图，确保横轴排序正确、包含竖直虚线、颜色从蓝渐变到红、策略 4 选中的点带黑圈。
- 确认 doc 中的 Excel 数据与图表上的点和分界线完全对应。
