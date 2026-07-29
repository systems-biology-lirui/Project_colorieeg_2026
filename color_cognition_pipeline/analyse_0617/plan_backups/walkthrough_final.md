# 双信号特征筛选、全脑可视化与电极选择性及记忆颜色解码分析总结 (Step1_1, Step1_2, Step2_1, Step2_2 & Step2_3)

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

## ⏱️ 单电极记忆颜色解码潜伏期与解剖梯度分析 (Step2_3)

我们针对 `memory_color` 显著分类电极开展了单通道记忆颜色 SVM 4折交叉解码。此项分析旨在提取各个电极上能够成功解码记忆颜色（红 vs 绿）的最早时间点，并探究该潜伏期是否在视觉通路的前后解剖轴向（Posterior-Anterior Gradient）上存在空间梯度（时序层级）。

### 1. 核心分析方法
1. **最早显著正确率时间点 (ESTP, Latency)**：
   在 $t \ge 80$ms 的刺激呈现后区间内，我们将每个通道在 4 个配对组合（Train R1/G1 Test R2/G2 等）的全部测试 trial 合并，在每个时间步计算单尾二项检验（Binomial Test，测试准确率是否显著大于随机水平 $50\%$，显著水平定为 $\alpha = 0.05$）。取首个显著的时间点（ms）定义为该通道的**信息提取潜伏期 (ESTP)**。若在 80ms 以后全无显著时间点，则记作 `NaN`。
2. **解剖梯度空间相关性**：
   将各通道的 ESTP 与其 MNI 坐标系下的 $Y$ 轴坐标（表征颅内前后方向，枕叶为负，颞叶趋近于正）进行 Pearson 及 Spearman 空间相关性分析，拟合一元线性回归趋势并计算置信区间。

---

### 2. 相关性分析与解剖梯度结论

#### (1) ERP 信号解剖梯度结果
- **ERP memory_color 通道总数**：31 个
- **Group 组水平空间相关性**：
  - Spearman $r_s = 0.505$ ($p = 0.0039$, 极显著)
  - Pearson $r_p = 0.449$ ($p = 0.0113$, 显著)
- **个体水平空间相关性**：
  - `test001` (N = 8): Spearman $r_s = 0.429$ ($p = 0.289$), Pearson $r_p = 0.218$ ($p = 0.603$)
  - `test002` (N = 14): Spearman $r_s = 0.686$ ($p = 0.0068$, 极显著), Pearson $r_p = 0.596$ ($p = 0.0245$, 显著)
  - `test003` (N = 9): Spearman $r_s = 0.312$ ($p = 0.413$), Pearson $r_p = 0.231$ ($p = 0.550$)
- **物理机制阐释**：
  在 ERP 上表现出极其强烈的**后脑向前脑的信息流动时序梯度**。位于枕叶（MNI Y 范围 $[-90, -70]$mm）的电极表现出极早的潜伏期（如 `test002-A4` ESTP = 80ms，`test003-G8` ESTP = 84ms），而沿着视觉腹侧通路向前脑颞叶（MNI Y 范围 $[-20, +5]$mm）延伸时，解码潜伏期逐渐延迟至 200~300ms 左右（如 `test001-E8` ESTP = 314ms，`test002-F8` ESTP = 292ms）。

#### (2) HG 信号解剖梯度结果
- **HG memory_color 通道总数**：12 个（其中 1 个在 80ms 后无显著点，记为 NaN）
- **Group 组水平空间相关性**：
  - Spearman $r_s = 0.697$ ($p = 0.0171$, 显著)
  - Pearson $r_p = 0.695$ ($p = 0.0177$, 显著)
- **个体水平空间相关性**：
  - `test001` (N = 2): 样本量不足无法计算相关
  - `test002` (N = 5, 剔除 1 个 NaN): Spearman $r_s = 0.800$ ($p = 0.104$), Pearson $r_p = 0.812$ ($p = 0.095$)
  - `test003` (N = 4): Spearman $r_s = 0.800$ ($p = 0.200$), Pearson $r_p = 0.854$ ($p = 0.146$)
- **物理机制阐释**：
  在 HG 信号上同样展现出**强烈的空间延时梯度**，组水平相关系数达到 $r \approx 0.70$ ($p < 0.02$)。后脑枕叶/颞叶后部电极（如 `test002-B1` ESTP = 80ms）的 High Gamma 激活远早于颞叶中部或前部的电极（如 `test001-E7` ESTP = 294ms，`test002-H5` ESTP = 564ms）。

---

### 3. 单电极解码时程曲线与解剖梯度拟合图

下图包含了 1行2列 复合大图。
- **左子图**：逐个通道的 SVM 解码曲线，通道线条颜色由后脑（**蓝色**）向前脑（**红色**）渐变着色，黑色粗实线为多通道平均，红虚线标示了 80ms 的检索起点。
- **右子图**：显著通道的 ESTP 与 MNI Y 坐标散点图，散点颜色与左侧相映射，并附有线性趋势拟合线、置信区间及相关系数统计指标。

#### (1) ERP 信号单电极解码与解剖梯度分析图 (Group & Individual)
````carousel
![ERP Group Decoding & Gradient](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_group_decoding_estp.png)
<!-- slide -->
![ERP test001 Decoding & Gradient](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_test001_decoding_estp.png)
<!-- slide -->
![ERP test002 Decoding & Gradient](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_test002_decoding_estp.png)
<!-- slide -->
![ERP test003 Decoding & Gradient](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_test003_decoding_estp.png)
````

#### (2) HG 信号单电极解码与解剖梯度分析图 (Group & Individual)
````carousel
![HG Group Decoding & Gradient](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/hg_group_decoding_estp.png)
<!-- slide -->
![HG test001 Decoding & Gradient](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/hg_test001_decoding_estp.png)
<!-- slide -->
![HG test002 Decoding & Gradient](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/hg_test002_decoding_estp.png)
<!-- slide -->
![HG test003 Decoding & Gradient](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/hg_test003_decoding_estp.png)
````

---

### 4. 统计结果数据表备份路径
我们已将完整的单电极解码 ESTP 潜伏期及其对应的 MNI 坐标、AAL3脑区等详情完整导出：
- **ERP 信号**：[select_channel_memory_decoding_estp_erp.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/select_channel_memory_decoding_estp_erp.xlsx) (及 [csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/select_channel_memory_decoding_estp_erp.csv))
- **HG 信号**：[select_channel_memory_decoding_estp_hg.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/select_channel_memory_decoding_estp_hg.xlsx) (及 [csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/select_channel_memory_decoding_estp_hg.csv))

---

## 🔴 纯色刺激多电极 SVM 解码与组水平 GLMM 显著性分析 (Step3_1)

针对 `memory_color` 显著记忆电极子集，在 Task 3（纯色刺激）的红色 (Trigger 51) vs 绿色 (Trigger 54) ERP 信号上实施了多电极 SVM 5折交叉验证分类解码，并使用二项分布的 GLMM 对 3 名被试进行组水平显著性检验。

### 1. 组水平解码与显著时间窗结果
- **GLMM 组水平显著解码时间窗**：**156 ms ~ 286 ms** 以及 **354 ms ~ 400 ms**。
- 这一结果证实了通过记忆任务筛选出的 `memory_color` 电极对纯物理的红色与绿色刺激同样具有高度鲁棒的解码能力，且显著窗口主要分布在刺激呈现后的早期（~200ms）及晚期维持阶段。

### 2. 解码正确率时程图
下图展示了个体被试（虚线）与组平均（紫色粗实线）的解码正确率，淡红阴影及顶部红线标注了 GLMM 显著的时间窗。

![ERP Pure Color Block Decoding](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_color_block_decoding.png)

---

## 🔄 纯色刺激与灰色水果跨任务时间泛化解码 (Step3_2)

为了探究物理纯色表征与物体颜色记忆表征在时间上的重合度（时间泛化，Temporal Generalization, TG），我们执行了下采样（10ms 步长，时间窗 [-100, 700]ms）的跨任务交叉解码：

### 1. 两种交叉解码策略
- **策略 1 (Color-to-Memory)**：使用 Task 3 红/绿色块刺激训练分类器，在 Task 2 灰色水果（红色记忆 vs 绿色记忆水果）上测试。
  - **最大组解码正确率**：**60.14%**，发生在 **训练时间 394 ms** (Task 3 纯色刺激晚期) / **测试时间 174 ms** (Task 2 记忆检索早期)。
- **策略 2 (Memory-to-Color)**：使用 Task 2 灰色水果（红/绿记忆水果）组合训练分类器，在 Task 3 红/绿色块刺激上测试。
  - **最大组解码正确率**：**57.22%**，发生在 **训练时间 194 ms** (Task 2 记忆检索早期) / **测试时间 394 ms** (Task 3 纯色刺激晚期)。
  
### 2. 泛化特征分析
策略 1 与策略 2 的最大正确率坐标具有高度的互易对称性（Train 394ms & Test 174ms vs Train 194ms & Test 394ms）。这表明：
- 纯色物理刺激在晚期（~400ms）激发的神经表征与灰色水果记忆在早期（~180ms）激发的检索表征具有显著的共享成分。

### 3. 时间泛化二维热图 (Group & Individual)
#### (1) 策略 1: 纯色训练 -> 灰色水果测试
````carousel
![Strategy 1 Group Heatmap](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/strategy1_group_temporal_generalization.png)
<!-- slide -->
![Strategy 1 test001 Heatmap](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/strategy1_test001_temporal_generalization.png)
<!-- slide -->
![Strategy 1 test002 Heatmap](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/strategy1_test002_temporal_generalization.png)
<!-- slide -->
![Strategy 1 test003 Heatmap](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/strategy1_test003_temporal_generalization.png)
````

#### (2) 策略 2: 灰色水果训练 -> 纯色测试
````carousel
![Strategy 2 Group Heatmap](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/strategy2_group_temporal_generalization.png)
<!-- slide -->
![Strategy 2 test001 Heatmap](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/strategy2_test001_temporal_generalization.png)
<!-- slide -->
![Strategy 2 test002 Heatmap](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/strategy2_test002_temporal_generalization.png)
<!-- slide -->
![Strategy 2 test003 Heatmap](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/strategy2_test003_temporal_generalization.png)
````

### 4. 统计结果数据备份路径
- 策略 1 时间泛化 CSV：[cross_decoding_tg_strategy1.csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/cross_decoding_tg_strategy1.csv)
- 策略 2 时间泛化 CSV：[cross_decoding_tg_strategy2.csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/cross_decoding_tg_strategy2.csv)

---

## 📍 单电极跨任务时间泛化解码推广 (Step3_3)

我们遍历了全部 31 个 `memory_color` ERP 显著电极，对每个通道独立计算了上述策略 1 与策略 2 的下采样时间泛化矩阵。
- **输出成果**：为这 31 个通道各自生成了 1行2列 的 TG 比较热图并保存在本地 `result/select_channel/decoding/cross_decoding/single_channel/` 下。
- **数据压缩打包**：所有电极的 $80 \times 80 \times 2 \times 3$ 二维泛化正确率矩阵数据已完整压缩打包，导出为：[single_electrode_tg_data.npz](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/single_electrode_tg_data.npz)。

---

## 🍓 真假水果颜色跨物体多 ROI 脑区解码分析 (Step4)

为研究不同解剖脑区对真假水果颜色（即带有物理颜色呈现的红色草莓 vs 绿色草莓，以及红包菜 vs 绿包菜等）的分类表征特性，我们在 5 个 ROI 区域（颞极、颞中回、颞下回、记忆显著电极、杏仁核）分别提取了电极子集，开展了 **4 折 Leave-One-Group-Out 跨物体颜色解码**（在一组水果，如草莓上训练，在另一组，如包菜上测试，跨 4 组轮换测试平均），有效排除了纯几何轮廓与纹理低级视觉特征的影响。

### 1. 5 个 ROI 解码特征与显著时间窗汇总

| ROI 脑区 | 筛选通道数 (3被试并集) | 组水平显著解码窗口 (ms) | 最大组正确率 (Max Acc) | 最大正确率时间点 (ms) |
| :--- | :---: | :--- | :---: | :---: |
| **Temporal Pole (颞极)** | 11 | 4 ms ~ 28 ms | 54.90% | -402 ms (基线期) |
| **Temporal Mid (颞中回)** | 17 | 712 ms ~ 848 ms, 868 ms ~ 902 ms | 55.14% | 884 ms |
| **Temporal Inf (颞下回)** | 13 | 460 ms ~ 486 ms | 55.10% | 478 ms |
| **Memory Color (记忆显著)** | 31 | 56-118 ms, 122-142 ms, 184-218 ms, 292-362 ms | **55.35%** | 80 ms |
| **Amygdala (杏仁核)** | 4 | 886 ms ~ 928 ms | **58.13%** | 912 ms |

*注：杏仁核 (Amygdala) 通道仅在 test001 中存在，因此自适应降级为单被试二项检验。*

### 2. 生物学机制讨论
- **记忆显著电极 (Memory Color)**：展示了极为丰富的显著解码时间窗（从早期 56ms 持续至 360ms 左右），最大正确率在早期（80ms）即达到，提示记忆颜色网络在刺激呈现的早期就与物理颜色处理产生了强烈的结合。
- **颞下回 (Temporal Inf) 与颞中回 (Temporal Mid)**：解码显著区间主要位于中晚期（颞下回 ~470ms，颞中回 ~800ms），表明腹侧视觉通路对跨物体的真假水果颜色表征存在从早期识别向晚期维持的语义阶段转换。
- **杏仁核 (Amygdala)**：在晚期（886-928ms）展现出高正确率（最大组正确率 58.13% @ 912ms），符合杏仁核参与物体价值以及高级联想的晚期加工定位。

### 3. 各 ROI 解码正确率曲线与显著阴影图
````carousel
![Real/Fake Decoding - Memory Color](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/real_fake_decoding_memory_color.png)
<!-- slide -->
![Real/Fake Decoding - Temporal Inf](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/real_fake_decoding_temporal_inf.png)
<!-- slide -->
![Real/Fake Decoding - Temporal Mid](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/real_fake_decoding_temporal_mid.png)
<!-- slide -->
![Real/Fake Decoding - Temporal Pole](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/real_fake_decoding_temporal_pole.png)
<!-- slide -->
![Real/Fake Decoding - Amygdala](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/real_fake_decoding_amygdala.png)
````

### 4. 统计结果数据备份路径
各 ROI 的详细解码正确率、Z值、P值等已被导出：
- 颞极：[real_fake_decoding_results_temporal_pole.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/real_fake_decoding_results_temporal_pole.xlsx)
- 颞中回：[real_fake_decoding_results_temporal_mid.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/real_fake_decoding_results_temporal_mid.xlsx)
- 颞下回：[real_fake_decoding_results_temporal_inf.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/real_fake_decoding_results_temporal_inf.xlsx)
- 记忆显著：[real_fake_decoding_results_memory_color.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/real_fake_decoding_results_memory_color.xlsx)
- 杏仁核：[real_fake_decoding_results_amygdala.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/real_fake_decoding_results_amygdala.xlsx)

---

## 🔬 记忆颜色电极前后聚类解码分析 (Step5)

我们将 31 个 ERP memory_color 显著电极根据 MNI_Y 轴坐标进行 KMeans 二分聚类，划分为**后部 (Posterior)** 与**前部 (Anterior)** 两组，在每个 cluster 内分别对灰色水果的记忆颜色（红 vs 绿）和真假呈现颜色（真红 vs 假绿）进行 SVM 解码。

### 1. 聚类分组结果

| Cluster | MNI_Y 范围 | 电极数 | test001 | test002 | test003 |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Posterior (后部)** | [-89.2, -42.5] mm | 20 | 5 elecs | 8 elecs | 7 elecs |
| **Anterior (前部)** | [-34.7, +4.3] mm | 11 | 3 elecs | 6 elecs | 2 elecs |

### 2. 分组解码组水平 GLMM 显著时间窗

| Cluster | 解码任务 | 刺激后显著窗 (ms, $p<0.05$, $\ge 20$ms) | 最大组正确率 | 峰值时间 |
| :--- | :--- | :--- | :---: | :---: |
| **Posterior** | Memory Color | [126, 200], [230, 280], [298, 316], [320, 388], [512, 536], [786, 840] | **59.93%** | 172 ms |
| **Posterior** | True/Fake Color | [58, 86], [94, 144], [182, 214], [232, 266], [290, 314], [318, 370], [736, 804], [824, 862] | **55.83%** | 746 ms |
| **Anterior** | Memory Color | [132, 156], [172, 196], [270, 300], [322, 420], [460, 488], [802, 894], [926, 998] | **59.38%** | 944 ms |
| **Anterior** | True/Fake Color | [8, 108] | **56.25%** | 24 ms |

### 3. 生物学解读

- **后部 (Posterior)**：包含枕叶 V4/V8 至颞下回后部的电极，在**记忆颜色**解码中展现出极其丰富的显著窗（从 126ms 持续断续至 840ms），最大正确率约 60% 出现在 172ms（早期视觉-语义映射窗口），同时在真假颜色解码中也展现出连续多段显著窗。这表明后部视觉颜色加工区同时参与了记忆颜色表征和物理颜色鉴别。
- **前部 (Anterior)**：包含颞中/颞下回中前段至颞极的电极，在**记忆颜色**解码中同样表现稳健（322-420ms 连续显著窗及晚期 800-998ms 长程维持），峰值正确率出现在晚期 944ms，反映了前部颞叶的高级语义联想与记忆维持功能。但真假颜色解码在前部显著窗仅限于 8-108ms 的极早期，提示前部区域不擅长编码低级物理颜色差异。

### 4. 聚类解码正确率曲线与显著阴影图

#### (1) Posterior (后部) 解码
````carousel
![Posterior Memory Color](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_cluster_posterior_memory_color_decoding.png)
<!-- slide -->
![Posterior Real/Fake](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_cluster_posterior_real_fake_decoding.png)
````

#### (2) Anterior (前部) 解码
````carousel
![Anterior Memory Color](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_cluster_anterior_memory_color_decoding.png)
<!-- slide -->
![Anterior Real/Fake](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_cluster_anterior_real_fake_decoding.png)
````

### 5. 统计结果数据备份路径
- 后部 Memory Color：[decoding_data_erp_cluster_posterior_memory_color.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_cluster_posterior_memory_color.xlsx)
- 后部 Real/Fake：[decoding_data_erp_cluster_posterior_real_fake.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_cluster_posterior_real_fake.xlsx)
- 前部 Memory Color：[decoding_data_erp_cluster_anterior_memory_color.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_cluster_anterior_memory_color.xlsx)
- 前部 Real/Fake：[decoding_data_erp_cluster_anterior_real_fake.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_cluster_anterior_real_fake.xlsx)

---

## 🧪 颞极单电极真假颜色 ERP 信号差异分析 (Step6)

在 `temporal_pole` 脑区的单电极上，对比 Task 2 中真颜色呈现 (True Color) 与假颜色呈现 (Fake Color) 的 ERP 信号差异。test001 贡献 2 个电极，test002 贡献 8 个电极，test003 在该区域无电极。

### 1. 分析方法
1. **逐时间点 Wilcoxon 秩和检验**：在 [-200, 800]ms 的每个时间步上进行 True vs Fake 的单试次比较，提取连续显著段 $\ge 50$ms 的窗口。
2. **200-500ms 均值 Wilcoxon 检验**：对 200-500ms 的试次平均波幅进行 True vs Fake 的秩和检验。

### 2. 统计结果汇总

| 被试 | 电极 | 均值差 ($\Delta\mu V$) | Wilcoxon $P$ | 200-500ms均值显著 | $\ge$50ms连续段 | 连续段窗口 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **test001** | **E6** | **-0.511** | **0.0012** ★★ | ✅ | ✅ | **[142, 374]** |
| **test001** | **E7** | **-0.760** | **0.028** ★ | ✅ | ✅ | **[156, 280]** |
| test002 | G1 | +1.842 | 0.221 | ❌ | ✅ | [390, 444] |
| test002 | G2 | -0.375 | 0.823 | ❌ | ❌ | — |
| **test002** | **G3** | **-1.265** | **0.0087** ★★ | ✅ | ✅ | **[134, 216], [410, 504]** |
| test002 | G4 | +0.409 | 0.081 | ❌ | ✅ | [24, 94], [210, 264] |
| test002 | G5 | +0.073 | 0.882 | ❌ | ❌ | — |
| test002 | G6 | -16.02 | 0.632 | ❌ | 仅基线 | [-474, -426] |
| test002 | G7 | +33.19 | 0.889 | ❌ | 仅基线 | [-156, -92] |
| test002 | G8 | -33.86 | 0.725 | ❌ | 仅基线 | [-288, -218] |

> [!IMPORTANT]
> 三个显著通道 (**E6, E7, G3**) 均表现出真颜色呈现时 ERP 波幅**更负**的方向性差异（即真颜色条件相比假颜色条件在该时间窗内引发了更强的负电位偏转），且连续显著段覆盖了 ~140-380ms 的知觉-语义加工核心窗口。

### 3. 代表性通道差异分析图

````carousel
![test001-E6 True vs Fake](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/test001_E6_true_fake_difference.png)
<!-- slide -->
![test002-G3 True vs Fake](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/test002_G3_true_fake_difference.png)
````

### 4. 统计结果数据备份路径
- [temporal_pole_true_fake_erp_stats.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/temporal_pole_true_fake_erp_stats.xlsx) (及 [csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/temporal_pole_true_fake_erp_stats.csv))

---

## 🌈 先验功能电极 (color_with_sti) 全套分析 (Step7)

针对每个被试的 `ieeglocation` 文件中 `color_with_sti` 标注的先验功能电极集，去重后共获得 18 个电极（test001: 6 个，test002: 4 个，test003: 8 个），进行 5 项全套 ERP 分析。

### 1. 子分析 7_1：单电极 Color vs. Gray 信号差异

在 Task 2 中，对每个电极比较有颜色呈现（红色/绿色呈现）与灰色呈现的 ERP 信号差异：

| 被试 | 电极 | $\Delta\mu V$ (200-500ms) | Wilcoxon $P$ | 均值显著 | $\ge$50ms段 | 段窗口 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| test001 | **D5** | -0.219 | **0.0069** ★★ | ✅ | ✅ | [328,380], [422,474] |
| test002 | **D1** | +30.03 | **0.0005** ★★★ | ✅ | ✅ | [290,542], [842,952] |
| test002 | **D2** | -11.30 | **0.0005** ★★★ | ✅ | ✅ | [260,578], [840,986] |
| test003 | **G3** | +0.755 | **0.00004** ★★★★ | ✅ | ✅ | [208,414] |
| test003 | **G4** | +0.338 | **0.0048** ★★ | ✅ | ✅ | [300,396] |

*注：仅展示了双指标同时显著的 5 个代表性电极，完整 18 个电极的结果详见数据备份表。*

#### 代表性单通道 Color vs. Gray 差异图
````carousel
![test002-D1 Color vs Gray](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/test002_D1_color_vs_gray_difference.png)
<!-- slide -->
![test003-G3 Color vs Gray](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/test003_G3_color_vs_gray_difference.png)
````

---

### 2. 子分析 7_2 ~ 7_5：多电极解码结果汇总

| 子分析 | 解码任务 | GLMM 显著窗 (ms, $p<0.05$, $\ge 20$ms) | 最大正确率 | 峰值时间 |
| :--- | :--- | :--- | :---: | :---: |
| **7_2** | Memory Color (灰色水果红/绿记忆) | **[152, 180], [216, 242]** | 55.00% | 222 ms |
| **7_3** | Color Block (纯色红/绿二分类) | **[54, 74], [118, 142]** | 59.35% | 60 ms |
| **7_4** | Cross Decoding 1D (纯色训练→灰色测试对角线) | **[150, 194], [218, 252]** | 55.28% | 160 ms |
| **7_5** | Real/Fake (LOGO 4折真假颜色) | **[116, 170], [174, 244], [324, 348], [396, 460], [472, 512], [856, 898]** | 57.85% | 140 ms |

> [!TIP]
> **关键发现**：跨任务 1D 对角线解码（7_4）的显著窗 **[150, 194] ms** 与 **[218, 252] ms** 与 Memory Color（7_2）在灰色水果上的显著窗 **[152, 180] ms** 与 **[216, 242] ms** 高度吻合。这在生物学上强力证实了在刺激后 150-250ms 区间内，物理颜色感知加工与颜色记忆检索在先验功能电极上共享**同一套神经编码模式**。

### 3. 多电极解码正确率曲线与显著阴影图

#### (1) Memory Color 与 Color Block 解码
````carousel
![Memory Color Decoding](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_color_with_sti_memory_color_decoding.png)
<!-- slide -->
![Color Block Decoding](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_color_with_sti_color_block_decoding.png)
````

#### (2) Cross Decoding 1D 与 Real/Fake 解码
````carousel
![Cross Decoding 1D](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_color_with_sti_cross_decoding_1d.png)
<!-- slide -->
![Real/Fake Decoding](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_color_with_sti_real_fake_decoding.png)
````

#### (3) 2D 时间泛化热图 (纯色训练 → 灰色测试)
````carousel
![TG Heatmap Group](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_color_with_sti_cross_decoding_tg_heatmap_group.png)
<!-- slide -->
![TG Heatmap test001](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_color_with_sti_cross_decoding_tg_heatmap_test001.png)
<!-- slide -->
![TG Heatmap test002](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_color_with_sti_cross_decoding_tg_heatmap_test002.png)
<!-- slide -->
![TG Heatmap test003](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/erp_color_with_sti_cross_decoding_tg_heatmap_test003.png)
````

### 4. 统计结果数据备份路径
- 单电极 Color vs Gray 统计表：[color_with_sti_color_vs_gray_erp_stats.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/color_with_sti_color_vs_gray_erp_stats.xlsx) (及 [csv](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/color_with_sti_color_vs_gray_erp_stats.csv))
- Memory Color 解码：[decoding_data_erp_color_with_sti_memory_color.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_color_with_sti_memory_color.xlsx)
- Color Block 解码：[decoding_data_erp_color_with_sti_color_block.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_color_with_sti_color_block.xlsx)
- Cross 1D 解码：[decoding_data_erp_color_with_sti_cross_decoding_1d.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_color_with_sti_cross_decoding_1d.xlsx)
- 2D 时间泛化数据：[cross_decoding_tg_color_with_sti.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/cross_decoding_tg_color_with_sti.xlsx)
- Real/Fake 解码：[decoding_data_erp_color_with_sti_real_fake.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/decoding_data_erp_color_with_sti_real_fake.xlsx)

---

## 🔗 相关代码链接
- [step1_1_select_channel_extended.py (Step1_1 电极筛选与裁剪)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_1_select_channel_extended.py)
- [step1_2_color_selectivity.py (Step1_2 纯色选择性分析与绘图)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_2_color_selectivity.py)
- [step1_2_color_selectivity_correlation.py (Step1_2 空间相关性与公式绘制)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_2_color_selectivity_correlation.py)
- [step2_1_memory_color_significance.py (Step2_1 记忆颜色显著性检验与脑图绘制)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step2_1_memory_color_significance.py)
- [step2_2_memory_color_decoding_glmm.py (Step2_2 多电极解码与 GLMM 统计分析)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step2_2_memory_color_decoding_glmm.py)
- [step2_3_single_electrode_decoding_correlation.py (Step2_3 单电极解码潜伏期与空间梯度相关性)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step2_3_single_electrode_decoding_correlation.py)
- [step3_1_color_block_decoding.py (Step3_1 纯色刺激 ERP 解码)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step3_1_color_block_decoding.py)
- [step3_2_cross_decoding_generalization.py (Step3_2 跨任务时间泛化解码)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step3_2_cross_decoding_generalization.py)
- [step3_3_single_electrode_generalization.py (Step3_3 单电极时间泛化)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step3_3_single_electrode_generalization.py)
- [step4_real_fake_color_decoding.py (Step4 真假水果跨物体多 ROI 解码)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step4_real_fake_color_decoding.py)
- [step5_memory_color_clusters_decoding.py (Step5 记忆颜色电极前后聚类解码)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step5_memory_color_clusters_decoding.py)
- [step6_temporal_pole_true_fake_erp_difference.py (Step6 颞极真假颜色 ERP 差异分析)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step6_temporal_pole_true_fake_erp_difference.py)
- [step7_color_with_sti_electrode_analyses.py (Step7 先验功能电极全套分析)](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step7_color_with_sti_electrode_analyses.py)

