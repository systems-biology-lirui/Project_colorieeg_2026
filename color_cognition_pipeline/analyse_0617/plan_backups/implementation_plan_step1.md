# 基于脑区定位与四种统计策略的电极筛选 (Step1)

本计划旨在实现 `step1_select_channel.py` 脚本，用于对被试 `test001`、`test002` 和 `test003` 进行电极筛选，定位核心电极并寻找邻近的候选未知电极。

## User Review Required

> [!IMPORTANT]
> **关于时间窗口和点数计算**：
> 1. **策略1和策略3（均值显著）**：时间窗口指定为 **100-400ms** 平均值进行 Wilcoxon 秩和检验（双尾，p < 0.05）。
> 2. **策略2和策略4（连续显著）**：在 **50-400ms** 范围内进行点对点 Wilcoxon 秩和检验，寻找连续显著且时间跨度 $\ge 50$ ms 的窗口。由于采样率为 500 Hz（采样间隔 2 ms），这对应连续 $\ge 25$ 个采样点满足 $p < 0.05$。
> 3. **绘图样式**：波形图的 $x$ 轴范围为 `[-200, 800]` ms。时间系列下方绘制点对点显著性标记点（当 $p < 0.05$ 且 Color > Gray 时画黄色方块，Color < Gray 时画青色方块），条形图展示 `100-400ms` 的平均波幅散点及 $p$ 值。这一设计完全兼容并复现了原有的优秀美学方案。

## Open Questions

> [!NOTE]
> 当前无开放性疑问。如果有任何参数变动要求，可以在执行前提出。

## Proposed Changes

---

### 电极筛选与可视化脚本

#### [NEW] [step1_select_channel.py](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_select_channel.py)
1. **脑区筛选**：
   - 枕叶：`Calcarine`, `Occipital_Inf`, `Occipital_Mid`, `Lingual`
   - 颞叶后/下部：`Fusiform`, `Temporal_Inf`
   - 颞叶前/上部：`Temporal_Mid`, `Temporal_Pole`
2. **四种统计策略筛选**：
   - **策略1**：混合类别的 color/gray 100-400ms 平均值显著。
   - **策略2**：混合类别在 50-400ms 窗口下点对点连续显著 $\ge 50$ ms (即 $\ge 25$ 个采样点)。
   - **策略3**：至少一个单一类别的 100-400ms 平均值显著。
   - **策略4**：至少一个单一类别在 50-400ms 窗口下点对点连续显著 $\ge 50$ ms。
3. **邻近电极扩展筛选**：
   - 找出与主要筛选出的电极处于同一根 shaft 上（物理命名字母前缀相同，触点序号相差 $\pm 1$）且 AAL3 标注为 `'unknown'`, `'N/A'` (空值) 或包含 `'parahippocampal'` / `'parahippocampus'` 的邻近电极。
   - 如果这些邻近电极也符合上述 4 种统计学策略中的任意一种，则将其保存为 `more_select_channel`。
4. **生成图像和汇总表格**：
   - 在 `analyse_0617/result/select_channel/{subject}/` 下生成所有被选电极的 ERP 图。图片命名如 `stra1_3_B5.png`。
   - 在 `analyse_0617/result/more_select_channel/{subject}/` 下生成邻近扩展电极的 ERP 图。
   - 在 `analyse_0617/doc/` 下生成 `select_channel_summary.xlsx` 和 `more_select_channel_summary.xlsx` 汇总表格。
5. **备份计划**：
   - 拷贝本实施计划至 `analyse_0617/plan_backups/implementation_plan_step1.md`。

## Verification Plan

### Automated Tests
- 运行 `/home/lirui/anaconda3/envs/lr2026/bin/python color_cognition_pipeline/analyse_0617/code/step1_select_channel.py`。
- 验证生成的图片和表格。

### Manual Verification
- 检查 `analyse_0617/result/select_channel/` 下图片的命名和内容。
- 检查 `analyse_0617/doc/select_channel_summary.xlsx` 和 `more_select_channel_summary.xlsx` 内容，确保筛选出符合条件的电极且数据准确。
