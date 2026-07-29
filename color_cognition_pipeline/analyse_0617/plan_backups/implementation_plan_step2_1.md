# 记忆颜色显著性分析与多维空间分布可视化计划 (Step2_1)

本计划旨在实现 `step2_1_memory_color_significance.py` 脚本，对筛选出的 75 个主要电极（`select_channel`）在 `task2`（记忆颜色任务：红记忆 vs 绿记忆）中的特征响应进行显著性计算与三维空间分布分析。

## User Review Required

> [!IMPORTANT]
> **分析逻辑与核心设计**：
> 1. **数据源与提取条件**：
>    - 使用 `task2` 的 epoch 特征数据（包括 ERP 与 HG 分别处理）。
>    - **红记忆（Red Memory）** Triggers：`['Trigger-In:121', 'Trigger-In:122', 'Trigger-In:123', 'Trigger-In:131', 'Trigger-In:132', 'Trigger-In:133']`
>    - **绿记忆（Green Memory）** Triggers：`['Trigger-In:101', 'Trigger-In:102', 'Trigger-In:103', 'Trigger-In:111', 'Trigger-In:112', 'Trigger-In:113']`
> 2. **显著性定义**：
>    - **窗口平均显著**：对 100-400ms 内每个 trial 的平均值进行非配对 Wilcoxon 秩和检验，若 $p < 0.05$ 则认为显著。
>    - **连续 50ms 以上显著**：对 100-400ms 时间范围内逐点进行非配对 Wilcoxon 秩和检验，若存在连续显著（$p < 0.05$）的区间且累计时长 $\ge 50$ms，则认为显著。
> 3. **分类与上色规则 (2D 脑图)**：
>    为了直观地展示显著电极的重叠情况，我们将主要电极点分成四类，并在 Nilearn 2D 玻璃脑图上进行上色：
>    - **仅窗口平均显著 (Mean Sig Only)**：橙黄色点（例如 `#ffaa00`）
>    - **仅连续 50ms 以上显著 (Cont 50ms Sig Only)**：深蓝色点（例如 `#1f77b4`）
>    - **两者皆显著 (Both Sig)**：鲜红色点（例如 `#d62728`）
>    - **其他非显著的主要电极 (Non-Significant)**：淡灰色点（例如 `#c0c0c0`）
> 4. **多饼图设计**：
>    - 针对 ERP 与 HG 各自绘制一张 1行4列 的饼图大图。
>    - 四个子饼图分别对应筛选策略 1, 2, 3, 4。
>    - 饼图展示在各策略包含的电极中，在记忆颜色任务里“表现出显著差异（满足上述两类显著条件之一）”的比例，直观评价策略的筛选效能。

---

## Open Questions

> [!NOTE]
> 1. **显著性电极的分类重叠处理**：
>    上述分类中，我们将“窗口平均显著”和“连续 50ms 以上显著”存在重合的电极独立归类为“两者皆显著 (Both Sig)”，以红色点突出显示。这相比硬性划分能更客观地反映神经元的选择性特征。请确认该设计是否符合您的期望？
> 2. **滑动窗口时序检验的校正**：
>    逐点检验通常会产生多重比较问题。但按项目先前的做法，仅要求非校正的 Wilcoxon 秩和检验在 $\ge 50$ ms 连续段上显著作为经验阈值。我们将遵循此原则以与前文保持一致。如果需要使用 FDR 或 Bonferroni 校正，请告知。

---

## Proposed Changes

### 记忆颜色统计与绘图脚本

#### [NEW] [step2_1_memory_color_significance.py](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step2_1_memory_color_significance.py)
1. **数据载入与对齐**：
   - 读取 `analyse_0617/doc/select_channel_summary.xlsx` 获取 75 个主要电极坐标、最高匹配策略及脑区信息。
   - 载入各被试的 `task2_ERP_epoched.mat` 和 `task2_hg_subband.mat`。
2. **统计学检验**：
   - 提取红色和绿色记忆条件的 trial 响应矩阵（过滤 NaN）。
   - 计算 100-400ms 内 trial 平均值的 Wilcoxon 秩和检验 $p$ 值。
   - 对 100-400ms 逐时间点进行秩和检验，检测是否存在连续显著（$p < 0.05$）且时长 $\ge 50$ms 的时间窗口。
3. **数据表格导出**：
   - 导出为 ERP 与 HG 两份独立的 Excel / CSV 结果明细表，保存在 `analyse_0617/doc/` 目录下，并包含以下列：`Subject`, `Electrode`, `Strategies_Matched`, `Mean_P`, `Is_Mean_Sig`, `Max_Cont_Duration_ms`, `Is_Cont_Sig`, `Overall_Sig_Type` (四分类)。
4. **单通道大图绘制**：
   - 对 75 个主要电极依次绘制“双子图”主图。
   - 左子图：画出 -200ms 到 800ms 时间范围内的 ERP 波形（或 HG 功率曲线）平均响应及 SEM 阴影。在底端使用小红点标注 `Red > Green且显著` 的时间点，小绿点标注 `Green > Red且显著` 的时间点。
   - 右子图：画出 100-400ms trial 均值的 Barplot（Mean + SEM），上方以文本形式标注 Wilcoxon 检验的具体 $p$ 值。
   - 存储路径：`analyse_0617/result/select_channel/memory_color/erp_single/` 和 `hg_single/`。
5. **多饼图绘制**：
   - 统计属于策略 1-4 的通道子集中，记忆显著（Mean 或 Cont 显著）的比例。
   - 绘制 ERP 和 HG 的多饼图，保存在 `result/select_channel/memory_color/`。
6. **Nilearn 2D 脑图绘制**：
   - 分别为 ERP 与 HG 绘制 2D 玻璃脑正交投影。
   - 按“仅均值显著（橙色）”、“仅连续显著（蓝色）”、“两者皆显著（红色）”、“不显著主要电极（灰色）”分类并染色，保存于结果目录。
7. **历史备份**：
   - 拷贝本实施计划至 `analyse_0617/plan_backups/implementation_plan_step2_1.md`。

---

## Verification Plan

### Automated Tests
- 运行 `/home/lirui/anaconda3/envs/lr2026/bin/python color_cognition_pipeline/analyse_0617/code/step2_1_memory_color_significance.py`。
- 确认产生两份导出的记忆颜色显著性 Excel / CSV 数据表格。
- 验证 `result/` 下正确生成 150 张单通道图（75张 ERP，75张 HG）、多饼图大图和 2D 玻璃脑电极图。

### Manual Verification
- 检查单通道图的左右子图格式是否对齐，显著性时间段和均值条形图的 $p$ 值是否对应。
- 确认玻璃脑投影中的灰色、橙色、蓝色和红色点数量，与 doc 表格中的分类电极数完全一致。
- 确认多饼图中的电极总数在策略交叠时依然准确（按策略子集计算）。
