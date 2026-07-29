# 双信号特征筛选、全脑可视化与电极颜色选择性分析总结 (Step1_1 & Step1_2)

我们已成功实现了电极筛选的扩展方案（`step1_1_select_channel_extended.py`）以及纯色刺激的颜色选择性（CSI）统计分析、空间相关性计算与渐变分布可视化（`step1_2_color_selectivity.py` & `step1_2_color_selectivity_correlation.py`）。该文档总结了这两个阶段的核心工作、统计结果和生物学意义。

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
   - 提取并保存了每个被试只包含筛选出通道 of ERP 与 HG 数据（格式为包含 `'labels'` 等裁剪列表）：
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

> [!TIP]
> **神经科学讨论：ERP 与 HG 的选择性差异**
> - **ERP** 表现出极其普遍和强烈的颜色选择性差异（75 个电极中有 74 个整体显著）。这是因为 ERP 反映的是较广脑区内树突电位和突触后电位的低频电位同步活动，颜色作为强视觉特征引发了广泛脑区（包括枕叶早期视觉、后颞叶、前颞叶等）的神经元群体共振差异。
> - **High Gamma (HG)** 信号颜色选择性则呈现高度局域化的特点（仅有少数电极显著）。这是因为 HG 紧密偶联于电极下极局域皮层（~1-2 mm 空间范围）的局部多单位发放（MUA），空间分辨率极高。大多数初筛出来的通道可能是由于 color-gray 整体激活的差异入选，而在具体的纯色间细分差异上，只有分布在非常核心“颜色斑块（Color patches）”上的少数 HG 通道才能表现出统计学显著的纯色选择性。

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

### 1. 颜色选择性指数的数学计算公式

- **Kruskal-Wallis 检验 ($H$-statistic)**（四色整体 CSI）：
  $$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$
  *其中，$N$ 为所有组的总 trial 数，$k=4$ 为颜色刺激类别数，$n_i$ 为第 $i$ 种颜色条件下的 trial 数，$R_i$ 为第 $i$ 种颜色在合并排序后的秩和。*

- **Wilcoxon 秩和检验 ($|Z|$-statistic)**（两色对比 CSI）：
  $$CSI = |Z| = \frac{|U - \mu_U|}{\sigma_U}$$
  *其中，$U = \min(U_1, U_2)$ 为两独立样本秩和统计量，均值 $\mu_U = \frac{n_1 n_2}{2}$，标准差 $\sigma_U = \sqrt{\frac{n_1 n_2(n_1+n_2+1)}{12}}$，$|Z|$ 即为标准分数的绝对值。*

---

### 2. 颜色选择性与 MNI Y 的空间相关性统计表

电极的 MNI Y 坐标由后脑（负值，例如 -90 代表枕叶枕极）延伸至前脑（正值，例如 15 代表前颞叶/颞极）。相关系数计算如下：

| 条件 (Condition) | 信号特征 | Spearman 相关系数 ($r_s$) | Spearman 显著性 ($p_s$) | Pearson 相关系数 ($r_p$) | Pearson 显著性 ($p_p$) | 显著标记 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **4-Color Overall** | **ERP** | **0.470** | $2.15 \times 10^{-5}$ | **0.390** | $5.35 \times 10^{-4}$ | **极显著正相关 (\*\*\*)** |
| | **HG** | -0.095 | $0.419$ | -0.048 | $0.684$ | 无相关 (n.s.) |
| **Red vs Green** | **ERP** | **0.410** | $2.63 \times 10^{-4}$ | **0.433** | $1.04 \times 10^{-4}$ | **极显著正相关 (\*\*\*)** |
| | **HG** | 0.061 | $0.603$ | 0.152 | $0.193$ | 无相关 (n.s.) |
| **Yellow vs Blue** | **ERP** | -0.047 | $0.688$ | 0.010 | $0.929$ | 无相关 (n.s.) |
| | **HG** | **-0.228** | $0.049$ | -0.198 | $0.088$ | **边缘显著负相关 (\*)** |

*注：\*\*\* 代表 $p < 0.001$，\*\* 代表 $p < 0.01$，\* 代表 $p < 0.05$。*

---

### 3. 相关性散点趋势与数学公式图

下图包含了 3 个子图。横轴代表颅内前后位置（MNI Y），纵轴代表颜色选择性指数（CSI）。
- **蓝色点和实线**：代表 **ERP** 电极散点与线性拟合趋势。
- **橙色点和虚线**：代表 **High Gamma (HG)** 电极散点与线性拟合趋势。
- 图中左上角标注了相关性统计的 $r$ 与 $p$ 值，右下角标注了对应的选择性数学计算公式。

![颜色选择性与前后脑位置空间相关性图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/color_selectivity_mni_y_correlation.png)

---

### 4. 神经科学讨论

1. **ERP 信号在脑前部表现出更强的选择性**：
   - ERP 的 4-Color Overall 和 Red vs Green CSI 指数与电极的前后位置（MNI Y）存在极显著的正相关（$p < 0.001$）。
   - **物理解释**：虽然初级视觉皮层（后脑，Y坐标极小）能对颜色做出快速的感觉编码，但在颜色认知（如纯色色块刺激任务）中，前脑（如颞中回、颞极等语义和记忆脑区）在较晚的时间窗内（50-400ms）可能发生了涉及概念匹配、任务决策的更强烈的神经元群体同步放电活动。这种认知同步性在 ERP 这种反映大范围突触后电位的低频分量中表现得尤为突出，从而在偏前脑部位显示出更大的低频波形差异（即更大的 CSI 指数）。
2. **HG 信号的空间局限性与非线性**：
   - 与 ERP 相比，HG 信号的 CSI 在整体和红绿对比中并没有随脑的前后空间表现出任何线性梯度，在黄蓝对比中仅表现出极弱的边缘负相关。
   - **物理解释**：HG 信号反映局部皮层超精细的放电（多单位活性，空间局限在 1-2mm）。它在纯色刺激下的选择性完全取决于电极是否精准放置于极局限的“颜色斑块（Color patch）”中。脑区大体的前后解剖梯度对如此精细的反应无法进行宏观的线性控制，这进一步印证了 HG 信号对于研究高空间分辨率的局域功能分工具有不可替代的价值。

---

### 5. 数据保存备份路径
- **相关性计算结果统计表**：[color_selectivity_correlation_summary.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/color_selectivity_correlation_summary.xlsx) (备份: [csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/color_selectivity_correlation_summary.csv))

---

## 🔗 相关代码链接
- [step1_1_select_channel_extended.py (Step1_1 电极筛选与裁剪)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_1_select_channel_extended.py)
- [step1_2_color_selectivity.py (Step1_2 纯色选择性分析与绘图)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_2_color_selectivity.py)
- [step1_2_color_selectivity_correlation.py (Step1_2 空间相关性与公式绘制)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_2_color_selectivity_correlation.py)
