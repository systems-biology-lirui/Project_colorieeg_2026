# 双信号特征筛选、全脑可视化与电极选择性及记忆颜色解码分析总结 (Step1_1, Step1_2, Step2_1 & Step2_2)

我们已成功实现了电极筛选的扩展方案（`step1_1_select_channel_extended.py`）、纯色刺激的颜色选择性（CSI）统计分析（`step1_2_color_selectivity.py` & `step1_2_color_selectivity_correlation.py`）、记忆颜色显著性差异分析（`step2_1_memory_color_significance.py`），以及多电极记忆颜色 SVM 解码与 Bayes 混合效应模型（GLMM）组水平显著性统计分析（`step2_2_memory_color_decoding_glmm.py`）。该文档总结了这一系列核心工作、统计结果和生物学意义。

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
   - [主要电极汇总表](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/select_channel_summary.xlsx)（备份：[select_channel_summary.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/select_channel/select_channel_summary.xlsx)）
   - [扩展电极汇总表](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/more_select_channel_summary.xlsx)（备份：[more_select_channel_summary.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/more_select_channel/more_select_channel_summary.xlsx)）

2. **裁剪后的特征 Mat 数据**：
   - 主要通道数据存放路径：[select_channel/](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/feature/select_channel/) 各被试目录。
   - 扩展物理邻近电极数据路径：[more_select_channel/](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/feature/more_select_channel/) 各被试目录。

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

## 📈 颜色选择性与前后脑位置相关性及公式展示 (Step1_2 扩展)

### 1. 相关性散点趋势与数学公式图

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

---

## 🧠 记忆颜色多电极解码与组水平 GLMM 显著性分析 (Step2_2)

我们对三套不同的电极方案进行了多电极记忆颜色解码分析：
- **方案 1 (Strategy 4)**：仅使用匹配了策略 4 的电极。
- **方案 2 (Union)**：使用主要筛选电极表中的并集电极（即 `ERP_Selected == True` / `HG_Selected == True` 的通道）。
- **方案 3 (Memory Color Significant)**：仅使用 Step 2_1 中表现显著的记忆颜色通道。

### 1. 解码管线深度优化 (防止过拟合)

我们在计算中实施了以下两项重要的解码管线优化（依据 `optimize_multi_electrode.py` 脚本）：
1. **向量化 Trial-wise 基线减除 (Baseline Subtraction)**：
   在读取数据后，提取基线期（$t < 0$ ms）特征均值，采用 NumPy 向量化广播机制，扣除每个 trial 在各个电极通道上的背景基线水平。这极大地抑制了脑电信号的基线直流漂移 (DC offsets) 与慢波波动。
2. **引入强正则化阻尼分类器**：
   将支持向量机参数设置为 `SVC(kernel='linear', C=0.1)`。相比强惩罚的 `C=1.0`，`C=0.1` 能提高线性分类边界的软间隔（Soft-margin），提供极佳的正则化阻尼，彻底压低了基线期内由于多电极噪声虚高导致的假阳性激活，突出了刺激后的真实正确率。

---

### 2. 组水平 GLMM 显著时间窗定位结果

我们在每个时间步合并 3 名被试的测试集 trial 对错向量，以被试为随机截距拟合二项分布的 Bayes 混合效应模型（GLMM），定位固定效应截距显著大于 0 （单尾 $p < 0.05$ 且持续窗口 $>20$ ms）的显著解码时间窗：

| 解码电极选择方案 | 信号特征 | 组水平 GLMM 显著时间窗列表 ($p < 0.05$ 单尾且 $\ge 20$ms) | 生物学阶段分析 |
| :--- | :---: | :--- | :--- |
| **Scheme 1 (Strategy 4)** | **ERP** | **136 ms ~ 194 ms**<br>**240 ms ~ 300 ms**<br>**370 ms ~ 392 ms** | 视觉早期信息检索与概念维持 (120-400ms) |
| | **HG** | **616 ms ~ 648 ms** | 晚期认知表征与维持 (600ms后) |
| **Scheme 2 (Union)** | **ERP** | **128 ms ~ 196 ms**<br>**246 ms ~ 284 ms**<br>**366 ms ~ 388 ms** | 表现稳定，基本对应视觉腹侧通路激活 |
| | **HG** | **602 ms ~ 666 ms**<br>**686 ms ~ 720 ms** | 晚期表征显著，在并集通道上更宽 |
| **Scheme 3 (Memory Sig)** | **ERP** | **124 ms ~ 202 ms**<br>**232 ms ~ 282 ms**<br>**320 ms ~ 392 ms** | 获得了时间跨度最大且连续最完整的显著性时窗 |
| | **HG** | **无 (\[ \])** | 未通过严格的 20ms 连续显著阈值要求 |

---

### 3. 解码正确率曲线与 GLMM 显著阴影图

下图中，细虚线代表各单个被试的解码时程，黑色/深紫色粗实线代表群组平均曲线，淡橙红色阴影和顶部粗红线标注了 GLMM 检验显著的连续时区（阈值 $p < 0.05$, 持续时间 $\ge 20$ ms）。

#### (1) Scheme 1: Strategy 4 电极方案
````carousel
![ERP Strategy 4 Decoding](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_strategy4_decoding.png)
<!-- slide -->
![HG Strategy 4 Decoding](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/hg_strategy4_decoding.png)
````

#### (2) Scheme 2: Union of Selected 并集电极方案
````carousel
![ERP Union Decoding](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_union_decoding.png)
<!-- slide -->
![HG Union Decoding](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/hg_union_decoding.png)
````

#### (3) Scheme 3: Significant Memory Color 记忆显著电极方案
````carousel
![ERP Memory Sig Decoding](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_memorysig_decoding.png)
<!-- slide -->
![HG Memory Sig Decoding](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/hg_memorysig_decoding.png)
````

---

### 4. 统计与画图数据备份路径

在绘图前，每套方案的每个时间点上的各被试正确率及 GLMM 统计参数（Z值、P值等）已提前妥善保存：
- **ERP 解码数据**：
  - [Scheme 1 (Strategy 4) 数据表](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_strategy4.xlsx) (及 [csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_strategy4.csv))
  - [Scheme 2 (Union) 数据表](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_union.xlsx) (及 [csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_union.csv))
  - [Scheme 3 (Memory Sig) 数据表](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_memorysig.xlsx) (及 [csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_memorysig.csv))
- **HG 解码数据**：
  - [Scheme 1 (Strategy 4) 数据表](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_hg_strategy4.xlsx) (及 [csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_hg_strategy4.csv))
  - [Scheme 2 (Union) 数据表](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_hg_union.xlsx) (及 [csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_hg_union.csv))
  - [Scheme 3 (Memory Sig) 数据表](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_hg_memorysig.xlsx) (及 [csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_hg_memorysig.csv))

---

## 🔗 相关代码链接
- [step1_1_select_channel_extended.py (Step1_1 电极筛选与裁剪)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_1_select_channel_extended.py)
- [step1_2_color_selectivity.py (Step1_2 纯色选择性分析与绘图)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_2_color_selectivity.py)
- [step1_2_color_selectivity_correlation.py (Step1_2 空间相关性与公式绘制)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_2_color_selectivity_correlation.py)
- [step2_1_memory_color_significance.py (Step2_1 记忆颜色显著性检验与脑图绘制)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step2_1_memory_color_significance.py)
- [step2_2_memory_color_decoding_glmm.py (Step2_2 多电极解码与 GLMM 统计分析)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step2_2_memory_color_decoding_glmm.py)
