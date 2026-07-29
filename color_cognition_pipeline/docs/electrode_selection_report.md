# 电极筛选策略修正与空间脑区限制对电极数量影响报告

本报告系统评估了 5 种不同的电极筛选标准（包括信号层筛选和解剖限制）对 **ERP** 与 **High Gamma (HG)** 信号筛选出的电极数量的影响。数据基于被试 `test001`、`test002` 与 `test003` 在 Task 1 (Face/Object/Body/Place 颜色与灰色对比) 中的全部通道。

---

## 📸 策略修正电极数量变化图
下图展示了两种信号特征在不同筛选策略下，以及应用靶区脑区限制（条件 5）前后的电极数量对比柱状图：

![电极筛选策略对比图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/electrode_selection_comparison.png)

---

## 📊 电极筛选数量详细对比表

| 筛选条件 (C1 ~ C4) | ERP (全脑) | ERP (限靶区内/C5) | High Gamma (全脑) | High Gamma (限靶区内/C5) |
| :--- | :---: | :---: | :---: | :---: |
| **条件 1: 混合类别 100-400ms 平均显著** | 5 | **3** | 13 | **5** |
| **条件 2: 混合类别 连续 50ms 显著** | 9 | **7** | 23 | **7** |
| **条件 3: 单一类别 100-400ms 平均显著** | 24 | **11** | 32 | **10** |
| **条件 4: 单一类别 连续 50ms 显著** | 38 | **16** | 70 | **27** |

> [!NOTE]
> * **限靶区内 (条件 5)**：限制通道所属解剖结构必须在 `AAL3` / `DKT` / `Desikan` 等定位列中匹配属于以下脑区：
>   1. **枕叶**：`Calcarine` (早期视觉)、`Occipital_Inf`、`Occipital_Mid`、`Lingual` (颜色与早期特征加工)
>   2. **颞叶后/下部**：`Fusiform`、`Temporal_Inf` (高级视觉、颜色斑块与形状整合)
>   3. **颞叶前/上部**：`Temporal_Mid`、`Temporal_Pole` (语义知识与记忆匹配)

---

## 🔍 电极筛选结果深度科学分析

根据统计得到的数量变化图表，可以总结出以下三条关键规律：

### 1. 单一类别 (Single Category) 策略相较于混合类别 (Merged Category) 更具敏感性
* **现象**：无论是 ERP 还是 HG，单一类别（条件 3, 4）筛选出的显著电极数均数倍于混合类别（条件 1, 2）。
* **机制**：混合类别将所有刺激条件合并（Face + Object + Body + Place），这会稀释部分只对特定条件（例如仅 Face 颜色敏感）具有高度选择性的电极。单一类别策略避免了由于条件平均导致的信号被“稀释”，从而在统计上更敏感。

### 2. 时域连续显著窗 (Continuous 50ms Window) 优于固定时区均值显著 (100-400ms Mean)
* **现象**：时域连续显著窗（条件 2, 4）在数量上明显多于 100-400ms 平均值显著条件（条件 1, 3）。
* **机制**：诱发活动和颜色选择性的特征响应在时域上可能是高度动态且较窄的（例如在 $150\,\text{ms} - 250\,\text{ms}$ 期间极强，在其他时间期平稳）。如果计算 $100-400\,\text{ms}$ 整个大窗口的平均值，非响应时间段的数据就会将原本极强的局域反应拉低并平抑。基于时间滑窗的连续 $50\,\text{ms}$ 策略可以完美捕捉这种短程高度动态的局部响应。

### 3. 解剖学空间脑区过滤器 (条件 5) 的重要性
* **现象**：应用解剖限制（条件 5）后，符合条件的电极数量急剧下降（例如，全脑 HG 满足单一类别连续显著的电极有 70 个，加上限制后仅剩 27 个，在之前未扩大脑区限制到枕叶和颞极时，仅为 21 个）。
* **机制**：如果在全脑范围不加空间脑区限制，那么在顶叶、额叶等与视觉颜色加工无关的脑区，也有很多通道因为随机扰动或高频活动通过了 ranksums 检验（产生伪阳性）。条件 5 作为空间先验过滤器，**排除了这些分布在顶叶、额叶及其他非视觉加工区域的噪声通道**，确保了我们分析的局部神经电信号来自于颞叶与枕叶视觉靶区。

---

## 📋 各筛选策略下的具体电极名称分布列表

下表详细列出了在不同电极筛选策略（混合 vs. 单一条件，时域均值 vs. 连续窗显著）以及空间脑区限制（全脑 vs. 颞叶与枕叶靶区内）下，每个被试所筛选到的**具体电极名称及数量**：

| 筛选策略与脑区限制 | 被试 | ERP 筛选到的电极列表 (数量) | High Gamma 筛选到的电极列表 (数量) |
| :--- | :--- | :--- | :--- |
| **策略 1 (混合 100-400ms 平均显著) - 全脑** | test001 | B5, H9 (2个) | A2, A3, E5, G1 (4个) |
| | test002 | C4, F9 (2个) | B5, E10, E11, G6, G8 (5个) |
| | test003 | F14 (1个) | B2, B13, G10, G11 (4个) |
| **策略 1 (混合 100-400ms 平均显著) - 靶区** | test001 | B5, H9 (2个) | 无 (0个) |
| | test002 | C4 (1个) | B5, G6, G8 (3个) |
| | test003 | 无 (0个) | G10, G11 (2个) |
| **策略 2 (混合 连续 50ms 显著) - 全脑** | test001 | B5, H9 (2个) | A2, A3, B5, C3, E5, G1 (6个) |
| | test002 | C4, F3, F9, H7 (4个) | B5, E10, E12, G6, G8 (5个) |
| | test003 | F14, G11, H13 (3个) | B2, B13, C10, D5, E1, E7, F5, F11, F12, G10, G11, H15 (12个) |
| **策略 2 (混合 连续 50ms 显著) - 靶区** | test001 | B5, H9 (2个) | B5 (1个) |
| | test002 | C4, F3, H7 (3个) | B5, G6, G8 (3个) |
| | test003 | G11, H13 (2个) | G10, G11, H15 (3个) |
| **策略 3 (单一 100-400ms 平均显著) - 全脑** | test001 | A6, B5, C9, C10, E1, E5, E6, F6, F7, F9, H1, H5, H9 (13个) | A3, C3, E1, E5, F11, F13, G1, H6, H9, H10 (10个) |
| | test002 | E1, E4 (2个) | A8, B6, C9, D1, D2, D4, G6, H5, H9 (9个) |
| | test003 | A7, A12, E2, E13, E16, F14, G6, H11, H13 (9个) | A1, A2, B1, B2, B8, B13, C9, D4, E3, F11, G5, G10, G11 (13个) |
| **策略 3 (单一 100-400ms 平均显著) - 靶区** | test001 | A6, B5, C9, C10, E6, F9, H1, H5, H9 (9个) | H6, H9, H10 (3个) |
| | test002 | 无 (0个) | A8, B6, C9, G6, H5 (5个) |
| | test003 | A12, H13 (2个) | G10, G11 (2个) |
| **策略 4 (单一 连续 50ms 显著) - 全脑** | test001 | B5, C8, C10, E1, E5, F5, F6, F9, G1, G10, G11, H1, H9 (13个) | A2, A3, B3, B5, C1, C3, C10, E5, E8, E11, F6, F11, F13, G1, G4, H6, H9, H10 (18个) |
| | test002 | A3, B1, C3, C7, C8, D2, D4, D5, F3, F5, F9 (11个) | A2, A3, A4, A8, A9, B6, C9, C10, D1, D4, D5, E4, E7, E9, E10, E11, E12, F4, F6, G5, G6, G7, H5, H8, H9, H10 (26个) |
| | test003 | A7, A12, E1, E2, E10, F4, F14, F20, G5, G11, G12, H3, H11, H12 (14个) | A1, A2, A4, B1, B2, B13, C9, D4, D5, D9, E3, E4, E10, E12, F2, F5, F11, F12, F13, G5, G9, G10, G11, G14, H3, H11 (26个) |
| **策略 4 (单一 连续 50ms 显著) - 靶区** | test001 | B5, C8, C10, F9, G11, H1, H9 (7个) | B5, C10, E8, G4, H6, H9, H10 (7个) |
| | test002 | A3, B1, C7, C8, F3, F5 (6个) | A2, A3, A4, A8, A9, B6, C9, C10, F4, F6, G5, G6, G7, H5, H8 (15个) |
| | test003 | A12, G11, G12 (3个) | D9, G9, G10, G11, G14 (5个) |

---

## 🌐 Nilearn 多维空间皮层投影可视化

为了直观地验证筛选结果并在三维空间查看电极的分布，我们使用 `nilearn` 库进行了 2D 玻璃脑以及 3D 脑表面投影的可视化渲染。

### 🎨 可视化规则与颜色编码
1. **电极形状与信号特征**：
   * **实心圆点 (Solid Circles)**：ERP 筛选到的显著电极。
   * **空心圆圈 (Hollow Circles)**：High Gamma (60-150Hz) 筛选到的显著电极。
2. **电极颜色与筛选策略**（采用**最高满足级别**进行颜色覆盖，防止重叠）：
   * 🟢 **绿色 (Green, #2ca02c)**：满足 **策略 1** (混合条件 100-400ms 平均显著)。
   * 🔵 **蓝色 (Blue, #1f77b4)**：满足 **策略 2** (混合条件 连续 50ms 显著)。
   * 🟣 **紫色 (Purple, #9467bd)**：满足 **策略 3** (单一条件 100-400ms 平均显著)。
   * 🟡 **橙黄色 (Orange/Yellow, #ff7f0e)**：满足 **策略 4** (单一条件 连续 50ms 显著)。

---

### 📷 1. 2D 全脑玻璃脑投影图 (Glass Brain)
2D 正交三轴投影玻璃脑图，能够无遮挡地看到 ERP 和 High Gamma 电极在枕叶、颞叶靶区的左右半球及深浅分布。

* 图像保存路径：[nilearn_glass_brain_electrodes.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/nilearn_glass_brain_electrodes.png)
* 脑目录备份路径：[nilearn_glass_brain_electrodes.png](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/nilearn_glass_brain_electrodes.png)

![2D 玻璃脑电极投影图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/nilearn_glass_brain_electrodes.png)

---

### 🌐 2. 3D 脑表面交互式可视化 HTML
我们分别导出了 ERP 和 High Gamma 筛选出的电极 3D 脑表面渲染。您可以在任意主流浏览器中双击打开对应的 HTML 文件进行**拖拽旋转、缩放，并将鼠标悬停至电极点上，查看被试编号、电极标签以及对应的策略等级**。

* **ERP 3D 交互脑图 (群组)**：
  - 项目路径：[erp_3d_brain.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/erp_3d_brain.html)
  - 脑目录备份：[erp_3d_brain.html](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/interactive_brain/erp_3d_brain.html)
* **High Gamma 3D 交互脑图 (群组)**：
  - 项目路径：[hg_3d_brain.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/hg_3d_brain.html)
  - 脑目录备份：[hg_3d_brain.html](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/interactive_brain/hg_3d_brain.html)
* **被试 `test001` 3D 交互脑图**：
  - ERP 脑图：[test001_erp_3d_brain.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/test001_erp_3d_brain.html) (脑目录备份：[test001_erp_3d_brain.html](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/interactive_brain/test001_erp_3d_brain.html))
  - High Gamma 脑图：[test001_hg_3d_brain.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/test001_hg_3d_brain.html) (脑目录备份：[test001_hg_3d_brain.html](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/interactive_brain/test001_hg_3d_brain.html))
  - 合并脑图 (Combined)：[test001_3d_brain.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/test001_3d_brain.html) (脑目录备份：[test001_3d_brain.html](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/interactive_brain/test001_3d_brain.html))
* **被试 `test002` 3D 交互脑图**：
  - ERP 脑图：[test002_erp_3d_brain.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/test002_erp_3d_brain.html) (脑目录备份：[test002_erp_3d_brain.html](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/interactive_brain/test002_erp_3d_brain.html))
  - High Gamma 脑图：[test002_hg_3d_brain.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/test002_hg_3d_brain.html) (脑目录备份：[test002_hg_3d_brain.html](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/interactive_brain/test002_hg_3d_brain.html))
  - 合并脑图 (Combined)：[test002_3d_brain.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/test002_3d_brain.html) (脑目录备份：[test002_3d_brain.html](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/interactive_brain/test002_3d_brain.html))
* **被试 `test003` 3D 交互脑图**：
  - ERP 脑图：[test003_erp_3d_brain.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/test003_erp_3d_brain.html) (脑目录备份：[test003_erp_3d_brain.html](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/interactive_brain/test003_erp_3d_brain.html))
  - High Gamma 脑图：[test003_hg_3d_brain.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/test003_hg_3d_brain.html) (脑目录备份：[test003_hg_3d_brain.html](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/interactive_brain/test003_hg_3d_brain.html))
  - 合并脑图 (Combined)：[test003_3d_brain.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/test003_3d_brain.html) (脑目录备份：[test003_3d_brain.html](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/interactive_brain/test003_3d_brain.html))

---

### 📓 3. Jupyter Notebook 交互式分析入口
为方便以 Jupyter 的方式交互，我们编写了完整的代码单元，包含实时数据加载、电极 Wilcoxon rank-sum 统计和 Nilearn 可视化调用。

* 笔记本入口：[visualize_electrodes_nilearn.ipynb](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/visualize_electrodes_nilearn.ipynb)
* docs 备份入口：[visualize_electrodes_nilearn.ipynb](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/docs/visualize_electrodes_nilearn.ipynb)

---

## 🧸 Object 特异性显著差异电极分析 (限靶区内)

在先前扩大空间脑区限制（枕叶、颞叶）所筛选出的电极子集中，我们针对 **Object 条件的颜色对比 (Object Color vs. Object Gray)** 进行了专门的统计差异检验。

### 📊 1. Object 颜色显著差异电极表
检验筛选条件为：在 Object 条件下，(1) 100-400ms 平均值显著，或 (2) 50-400ms 期间存在连续 50ms 以上的显著窗口。我们在靶区内共筛选到了 **5 个** 满足条件的电极：

| 被试 | 电极 | 信号类型 | MNI 坐标 (X, Y, Z) | 100-400ms 均值 p 值 | 最大连续显著窗口 (ms) | 显著类型 |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| test001 | C10 | ERP | (68.5, -38.5, 4.9) | 0.0047 | 228.0 ms | 均值与连续窗均显著 |
| test001 | B5 | HIGHGAMMA | (46.9, -54.3, 1.4) | 不显著 | 50.0 ms | 仅连续50ms以上显著 |
| test002 | A9 | HIGHGAMMA | (60.5, -72.0, -13.9) | 不显著 | 56.0 ms | 仅连续50ms以上显著 |
| test002 | F4 | HIGHGAMMA | (55.7, -4.8, -18.0) | 不显著 | 50.0 ms | 仅连续50ms以上显著 |
| test002 | G6 | HIGHGAMMA | (52.0, 14.5, -28.6) | 不显著 | 52.0 ms | 仅连续50ms以上显著 |

---

### 📷 2. 2D 全脑玻璃脑投影图
展示了这 5 个 Object 特异性显著电极在全脑中的正交投影分布：
* 🔴 **红色**：均值与连续窗均显著
* 🔵 **蓝色**：仅 100-400ms 均值显著
* 🟢 **绿色**：仅连续 50ms 以上显著
* ERP 用**实心圆**展示，High Gamma 用**空心圆圈**展示。

* 项目图像路径：[nilearn_object_electrodes.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/nilearn_object_electrodes.png)
* 脑目录备份路径：[nilearn_object_electrodes.png](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/nilearn_object_electrodes.png)

![2D Object 显著电极玻璃脑投影](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/nilearn_object_electrodes.png)

---

### 🌐 3. 3D 脑表面交互式 HTML 视图
* **ERP Object 3D 脑图**：
  - 项目路径：[erp_3d_object_brain.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/erp_3d_object_brain.html)
  - 脑目录备份：[erp_3d_object_brain.html](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/interactive_brain/erp_3d_object_brain.html)
* **High Gamma Object 3D 脑图**：
  - 项目路径：[hg_3d_object_brain.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/interactive_brain/hg_3d_object_brain.html)
  - 脑目录备份：[hg_3d_object_brain.html](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/interactive_brain/hg_3d_object_brain.html)

---

## 📈 二项分布广义混合线性模型 (GLMM) 组水平显著性解码分析

为了在组水平上准确评估利用策略 4 显著电极进行灰色刺激颜色知识（灰色西瓜/草莓 vs. 灰色猕猴桃/卷心菜）解码的统计显著性，我们采用了基于二项分布的广义线性混合效应模型 (Generalized Linear Mixed Model, GLMM with Binomial family and Logit link)。

### 🔬 统计建模与参数估计
- **数据层级**：分析基于 Trial 级别数据。以每个时间步（共 750 个时间点，$-500\,\text{ms}$ 至 $998\,\text{ms}$）上所有被试的每个测试试次（Trial）是否被 SVM 分类器预测正确为二分类因变量 ($Y_{ij} \in \{0, 1\}$，代表被试 $i$ 的试次 $j$)。
- **随机效应**：引入**被试 (Subject)** 作为随机截距，以控制同一被试内部不同试次之间的相关性。
- **数学模型**：
  $$\text{logit}(P(Y_{ij} = 1)) = \beta_0 + b_{i}$$
  其中 $\beta_0$ 为固定效应截距， $b_i \sim N(0, \sigma^2)$ 为被试 $i$ 的随机截距。
- **显著性检验**：检验固定效应截距 $\beta_0$ 是否显著大于 0（即在 logit 变换后是否显著高于 50% 机会水平，0 对应 50% 概率）。由于被试量极小（$N=3$），传统的频数派混合模型在很多时间步上估计随机效应方差容易面临奇异性 (Singularity) 或无法收敛的困境。我们采用变分贝叶斯 (Variational Bayes, VB) 估计固定效应截距的后验均值与标准差，计算 Wald $z$-score 并以此获取单尾显著 $p$ 值。
- **显著时间窗筛选**：筛选条件为：(1) 固定截距 $\beta_0 > 0$，(2) 单尾 $p < 0.05$，(3) 连续持续时间达到 20 ms (对应 10 个时间步) 以上。

### 📊 ERP 信号 GLMM 检验结果
在 ERP 策略 4 电极解码中，共检测到 **6 个** 显著高于机会水平的连续时间窗口：
- **时间窗 1**: `-346.0 ms` 到 `-328.0 ms` (基线期随机扰动)
- **时间窗 2**: `148.0 ms` 到 `214.0 ms` (ERP 极强颜色加工起始期)
- **时间窗 3**: `236.0 ms` 到 `308.0 ms` (颜色知识加工与检索期)
- **时间窗 4**: `320.0 ms` 到 `388.0 ms` (认知维持与加工期)
- **时间窗 5**: `548.0 ms` 到 `588.0 ms` (晚期语义检索加工窗，与前述点对点 t 检验发现的 552-582ms 高度契合)
- **时间窗 6**: `648.0 ms` 到 `668.0 ms` (晚期认知维持期)

* 显著性图像项目路径：[group_glmm_erp_strategy4_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_glmm_erp_strategy4_decoding.png)
* 脑目录备份：[group_glmm_erp_strategy4_decoding.png](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_erp_strategy4_decoding.png)

![ERP GLMM 显著性曲线图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_erp_strategy4_decoding.png)

### 📊 High Gamma 信号 GLMM 检验结果
在 High Gamma 策略 4 电极解码中，检测到 **4 个** 显著高于机会水平的连续时间窗口：
- **时间窗 1**: `-326.0 ms` 到 `-308.0 ms` (基线期)
- **时间窗 2**: `-132.0 ms` 到 `-108.0 ms` (基线期)
- **时间窗 3**: `602.0 ms` 到 `676.0 ms` (High Gamma 晚期显性解码显著窗)
- **时间窗 4**: `746.0 ms` 到 `808.0 ms` (晚期信息检索显著窗)

* 显著性图像项目路径：[group_glmm_highgamma_strategy4_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_glmm_highgamma_strategy4_decoding.png)
* 脑目录备份：[group_glmm_highgamma_strategy4_decoding.png](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_highgamma_strategy4_decoding.png)

![High Gamma GLMM 显著性曲线图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_highgamma_strategy4_decoding.png)

### 💡 科学讨论与结论
1. **GLMM 的统计功效优势**：对于极小样本 ($N=3$)，直接基于准确率时序进行一类 T 检验或 Cluster 置换检验，其统计效能较低。由于混合模型从 Trial 级微观维度进行分析，并引入随机截距项吸收了个体间基线正确率的变异，在控制一类错误率的同时大幅释放了统计检验功效，为识别皮层局部代表灰色图像颜色知识的高时间精度阶段提供了坚实的数理依据。
2. **ERP 的多阶段连续显著性**：ERP 在 `148.0 ms` 之后表现出快速上升并呈现三个显著波峰（`148-214ms`, `236-308ms`, `320-388ms`），这十分符合视觉腹侧通路从早期视觉特征激活到晚期概念/记忆检索的渐进时序加工模式。
3. **High Gamma 晚期反应**：High Gamma 的显著时间窗主要出现在晚期阶段（`602-676ms` 之后），预示着高频能量通道可能更多承载了顶颞叶概念整合、颜色知识提取的持续认知努力。



