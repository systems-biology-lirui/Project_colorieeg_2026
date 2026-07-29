# 双信号特征筛选、全脑可视化与电极颜色选择性及记忆颜色显著性分析总结 (Step1_1, Step1_2 & Step2_1)

我们已成功实现了电极筛选的扩展方案（`step1_1_select_channel_extended.py`）、纯色刺激的颜色选择性（CSI）统计分析（`step1_2_color_selectivity.py` & `step1_2_color_selectivity_correlation.py`），以及记忆颜色显著性计算与多维空间分布可视化（`step2_1_memory_color_significance.py`）。该文档总结了这三个阶段的核心工作、统计结果和生物学意义。

---

## 📊 电极统计与全脑对比图 (Step1_1)

### 1. 电极筛选数量结果

| 电极组别 | 被试 | ERP 筛选通道数 | HG 筛选通道数 | 并集去重通道数 | 扩展邻近通道数 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **主要电极 (`select_channel`)** | test001 | 18 | 18 | **27** | - |
| | test002 | 15 | 24 | **33** | - |
| | test003 | 7 | 10 | **15** | - |
| **扩展电极 (`more_select_channel`)** | test001 | 1 | 4 | **5** | 5 (邻居) |
| | test002 | 2 | 2 | **3** | 3 (邻居) |
| | test003 | 0 | 1 | **1** | 1 (邻居) |

---

### 2. 总体电极筛选策略数量对比柱状图

下图展示了 ERP 与 HG 条件下，全脑（Whole Brain）与限定在核心脑区（Target Area）内的 4 种不同策略筛选出的电极数对比柱状图：

![主要电极筛选策略对比柱状图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/electrode_selection_comparison.png)

*注：图中绿色（In Target Area）即为我们最终筛选的主要电极。*

---

### 3. Nilearn 2D 玻璃脑投影电极图 (Group Level)

下图展示了 3 个被试所筛选的主要电极在全脑中的 MNI 空间正交投影。
- 实心圆代表 ERP
- 空心圆代表 HG
- 颜色代表最高匹配策略（🟢 绿-策略1，🔵 蓝-策略2，🟣 紫-策略3，🟡 橙黄-策略4）

![全脑电极 2D 玻璃脑投影分布图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/nilearn_glass_brain_electrodes.png)

---

## 📂 升级后的汇总表与数据裁剪目录

1. **升级版汇总表格**：
   - 增加了 `AAL3_ROI` (枕叶/颞叶等 ROI 靶区分类)、`MNI_X`, `MNI_Y`, `MNI_Z` 三维坐标列。
   - 用 `ERP_Selected` / `ERP_Strategies_Matched` 和 `HG_Selected` / `HG_Strategies_Matched` 详细标明了两个特征通道的筛选细节。
   - [主要电极汇总表](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/select_channel_summary.xlsx)（备份：[select_channel_summary.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/select_channel/select_channel_summary.xlsx)）
   - [扩展电极汇总表](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/more_select_channel_summary.xlsx)（备份：[more_select_channel_summary.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/more_select_channel/more_select_channel_summary.xlsx)）

2. **裁剪后的特征 Mat 数据**：
   - 提取并保存了每个被试只包含筛选出通道的 ERP 与 HG 数据（格式为包含 `'labels'` 等裁剪列表）：
     - 主要通道数据存放路径：[select_channel/](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/feature/select_channel/) 下的各被试目录。
     - 扩展物理邻近电极数据路径：[more_select_channel/](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/feature/more_select_channel/) 下的各被试目录。

---

## 🎨 纯色刺激颜色选择性分析 (Step1_2)

我们针对筛选出的 75 个主要电极，利用 `task3`（纯色色块刺激）中的四种颜色（红：51，黄：52，蓝：53，绿：54）在 `50-400ms` 的平均响应数据，计算了三种颜色选择性指数（Color Selectivity Index, CSI）：
1. **四色整体差异 (Overall CSI)**：采用 Kruskal-Wallis 检验，CSI 定义为 $H$-statistic，双尾检验 $p < 0.05$ 判定显著。
2. **红绿选择性 (RG CSI)**：采用 Wilcoxon 秩和检验，CSI 定义为 $\text{abs}(Z\text{-stat})$，双尾检验 $p < 0.05$ 判定显著。
3. **黄蓝选择性 (YB CSI)**：采用 Wilcoxon 秩和检验，CSI 定义为 $\text{abs}(Z\text{-stat})$，双尾检验 $p < 0.05$ 判定显著。

### 1. 颜色选择性显著电极统计对比

在全脑 75 个初筛主要电极中，通过严格统计检验显著的电极数量如下：

| 信号特征类型 | 主要电极总数 | 四色整体显著数 (KW) | 红绿对比显著数 (RG) | 黄蓝对比显著数 (YB) |
| :--- | :---: | :---: | :---: | :---: |
| **ERP** | 75 | **74** (98.7%) | **61** (81.3%) | **62** (82.7%) |
| **High Gamma (HG)** | 75 | **4** (5.3%) | **2** (2.7%) | **7** (9.3%) |

---

### 2. 颜色选择性指数渐变排布分布图

按要求，横轴为电极排布（按 CSI 升序排序），纵轴为 CSI 值。
- **电极点的颜色**：按电极 MNI Y 坐标从后脑到前脑逐渐从**蓝色**变成**红色**（基于 `coolwarm` 映射）。
- **策略 4 电极**：外圈增加**黑圈边框**。
- **阈值虚线**：一条竖直灰色虚线划分出 $p \ge 0.05$（左侧，不标注电极）与 $p < 0.05$ 显著（右侧，标注电极标签）。

#### (1) ERP 信号 CSI 排布分布图

![ERP 颜色选择性指数渐变分布图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/color_selectivity_index_erp.png)

#### (2) High Gamma 信号 CSI 排布分布图

![HG 颜色选择性指数渐变分布图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/color_selectivity_index_hg.png)

---

## 📈 颜色选择性与前后脑位置相关性及公式展示 (Step1_2 扩展)

为了进一步探究颜色选择性指数在脑内的解剖空间分布规律，我们计算了三类 CSI 指数与颅内电极前后方向坐标（MNI Y）的统计学相关性，并在散点图上标注了对应的计算数学公式。

### 1. 颜色选择性与 MNI Y 的空间相关性统计表

电极的 MNI Y 坐标由后脑（负值，例如 -90 代表枕叶枕极）延伸至前脑（正值，例如 15 代表前颞叶/颞极）。相关系数计算如下：

| 条件 (Condition) | 信号特征 | Spearman 相关系数 ($r_s$) | Spearman 显著性 ($p_s$) | Pearson 相关系数 ($r_p$) | Pearson 显著性 ($p_p$) | 显著标记 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **4-Color Overall** | **ERP** | **0.470** | $2.15 \times 10^{-5}$ | **0.390** | $5.35 \times 10^{-4}$ | **极显著正相关 (\*\*\*)** |
| | **HG** | -0.095 | $0.419$ | -0.048 | $0.684$ | 无相关 (n.s.) |
| **Red vs Green** | **ERP** | **0.410** | $2.63 \times 10^{-4}$ | **0.433** | $1.04 \times 10^{-4}$ | **极显著正相关 (\*\*\*)** |
| | **HG** | 0.061 | $0.603$ | 0.152 | $0.193$ | 无相关 (n.s.) |
| **Yellow vs Blue** | **ERP** | -0.047 | $0.688$ | 0.010 | $0.929$ | 无相关 (n.s.) |
| | **HG** | **-0.228** | $0.049$ | -0.198 | $0.088$ | **边缘显著负相关 (\*)** |

---

### 2. 相关性散点趋势与数学公式图

下图包含了 3 个子图。横轴代表颅内前后位置（MNI Y），纵轴代表颜色选择性指数（CSI）。
- **蓝色点和实线**：代表 **ERP** 电极散点与线性拟合趋势。
- **橙色点和虚线**：代表 **High Gamma (HG)** 电极散点与线性拟合趋势。

![颜色选择性与前后脑位置空间相关性图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/color_selectivity_mni_y_correlation.png)

---

## 🧠 记忆颜色显著性差异分析 (Step2_1)

我们针对 75 个主要电极，利用 `task2`（物体记忆颜色任务）中红色记忆（Trigger: 121-123, 131-133）与绿色记忆（Trigger: 101-103, 111-113）在 `100-400ms` 的响应，开展了显著性分析：
- **窗口平均显著**：100-400ms 的 trial 平均值通过 Wilcoxon 秩和检验（双尾 $p < 0.05$）。
- **连续显著段时长**：100-400ms 内存在连续显著的逐时间点检验区且累计时长 $\ge 50$ms。

### 1. 记忆颜色显著性分类电极统计

| 信号特征 | 总通道数 | 两者皆显著 (Both_Sig) | 仅均值显著 (Mean_Sig_Only) | 仅连续显著 (Cont_Sig_Only) | 记忆显著通道并集 (Total Sig) | 不显著电极数 (Non_Sig) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ERP** | 75 | **17** (22.7%) | **0** (0%) | **14** (18.7%) | **31** (41.3%) | 44 (58.7%) |
| **High Gamma (HG)** | 75 | **6** (8.0%) | **0** (0%) | **6** (8.0%) | **12** (16.0%) | 63 (84.0%) |

> [!IMPORTANT]
> **连续 50ms 显著检验的科学价值**
> - 在 ERP 和 HG 中，各有 **14** 个和 **6** 个通道属于“仅连续显著 (Cont_Sig_Only)”类别。
> - 这些通道在 100-400ms 窗口内的平均响应并不显著（因为在 300ms 范围的积分窗口中，时程波形的极性翻转或震荡会导致正负值相消）。然而，它们的动态时程波形差异显著且累计时间达到了 50ms 以上。如果仅用传统的“窗口均值”来做显著性测试，这些极具时序颜色编码信息的特征电极将会被全部漏掉。这强烈证明了连续 50ms 判定标准的科学性和必要性！

---

### 2. 代表性电极分析示例 (test001 - G11)

下面展示了位于颞下回（`Temporal_Inf` ROI）的核心通道 `G11` 在红、绿记忆颜色下的响应波形及 100-400ms 平均响应对比：
- 左图：显示 -200ms 到 800ms 时程，底端标注了 $p < 0.05$ 时的显著点（🔴 红色代表红记忆 > 绿记忆，🟢 绿色代表绿记忆 > 红记忆）。
- 右图：展示 100-400ms 窗口均值对比（柱状图 + SEM），顶部标出了 Wilcoxon 秩和检验 $p$ 值。

#### (1) G11 ERP 记忆选择性分析

![G11 ERP 记忆选择性分析图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/test001_G11_ERP_Memory_Significance.png)

#### (2) G11 High Gamma 记忆选择性分析

![G11 HG 记忆选择性分析图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/test001_G11_HG_Memory_Significance.png)

*注：G11 无论在 ERP 还是 HG 信号中都表现出极其强烈的红绿记忆选择性差异（其 ERP $p = 9.62 \times 10^{-38}$；HG $p = 0.042$）。*

---

### 3. 多饼图：不同筛选策略电极在记忆颜色中的显著占比

对于初始筛选的 4 种策略电极，展现其中有多少通道在记忆颜色中仍呈现显著响应（Mean或Cont之一显著）。
- **Sig（莫兰迪红）**：记忆颜色显著电极。
- **Non-Sig（淡灰蓝）**：不显著电极。

#### (1) ERP 各筛选策略多饼图 (N=75)
![ERP 各筛选策略下记忆显著占比饼图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/memory_color_strategy_pie_erp.png)

#### (2) High Gamma 各筛选策略多饼图 (N=75)
![HG 各筛选策略下记忆显著占比饼图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/memory_color_strategy_pie_hg.png)

*说明：以“单一类别下连续 50ms 显著”为特征的策略 4 通道，在记忆颜色中被证实具有非常高比例的显著电极。*

---

### 4. Nilearn 2D 玻璃脑投影分类定位图

我们在 MNI 空间正交投影上把 75 个主要电极分为四类上色：
- 🔴 **红色 (Both_Sig)**：两者皆显著
- 🟡 **橙色 (Mean_Sig_Only)**：仅均值显著
- 🔵 **蓝色 (Cont_Sig_Only)**：仅连续 50ms 显著
- ⚪ **灰色 (Non_Sig)**：主要电极在记忆颜色中不显著
*注：ERP 脑图使用实心圆表示；HG 脑图使用空心圆圈表示。*

#### (1) ERP 脑投影分类定位图
![ERP 脑图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/memory_color_glass_brain_erp.png)

#### (2) High Gamma 脑投影分类定位图
![HG 脑图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/memory_color_glass_brain_hg.png)

---

### 5. 数据保存路径
- **ERP 统计明细表**：[select_channel_memory_significance_erp.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/select_channel_memory_significance_erp.xlsx) (备份: [csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/select_channel_memory_significance_erp.csv))
- **HG 统计明细表**：[select_channel_memory_significance_hg.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/select_channel_memory_significance_hg.xlsx) (备份: [csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/select_channel_memory_significance_hg.csv))

---

## 🔗 相关代码链接
- [step1_1_select_channel_extended.py (Step1_1 电极筛选与裁剪)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_1_select_channel_extended.py)
- [step1_2_color_selectivity.py (Step1_2 纯色选择性分析与绘图)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_2_color_selectivity.py)
- [step1_2_color_selectivity_correlation.py (Step1_2 空间相关性与公式绘制)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_2_color_selectivity_correlation.py)
- [step2_1_memory_color_significance.py (Step2_1 记忆颜色显著性检验与脑图绘制)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step2_1_memory_color_significance.py)
