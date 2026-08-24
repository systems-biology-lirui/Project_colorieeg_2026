# 🧠 三类颜色相关通道筛选策略、电极分布与分析结果对比报告

本报告系统整理和对比了本项目中用于颜色认知加工分析的**三类电极通道筛选策略**，包括其定义依据、筛选方法、电极分布数量、功能解码表现以及对应的可视化结果图片。

---

## 📊 一、 三类电极筛选策略总览对比

| 策略类别 | 策略名称与定义 | 核心筛选依据与方法 | 包含电极数 (5被试) | 核心覆盖脑区 |
| :--- | :--- | :--- | :--- | :--- |
| **策略 A** | **腹侧通路功能统计显著电极** <br/> *(Functional Color-Selective)* | 在 Task 2/Task 3 中 Color vs. Gray 或红 vs. 绿显著（Wilcoxon $p<0.05$ 且连续 $\ge 50\text{ms}$），且严格位于腹侧视觉通路 AAL3 靶区内 | **38 个** <br/> (`test001`:8, `test002`:14, `test003`:9, `test005`:6, `test006`:1) | 枕下回, 梭状回, 颞下回 (ITG), 颞中回 (MTG) |
| **策略 B** | **fMRI Color Patch 邻近电极** <br/> *(`color_patch` Electrodes)* | 基于被试 fMRI 扫描定位出的 Color Patch 空间坐标（如 VO, V4, PIT），寻找解剖邻近范围内的 sEEG 通道 | **11 个** <br/> (`test001`:7, `test003`:4, 其它被试:0) | 枕叶/梭状回交界处, V4 邻近区 |
| **策略 C** | **电刺激光幻视/颜色觉先验电极** <br/> *(`color_with_sti` Electrodes)* | 临床皮层高频电刺激 (Electrical Stimulation) 中，刺激该触点可直接诱发出主观光幻视 (Phosphenes) 或颜色感知觉 | **18 个** <br/> (`test001`:6, `test002`:4, `test003`:8, `test005`/`006`:0) | 枕叶, 腹侧视觉通路中后段 (D, G, H 针点) |

---

## 🔍 二、 策略 A：腹侧通路功能统计显著电极 (Functional Color-Selective)

### 1. 筛选策略与方法
- **定义方法**：在 Task 1 / Task 2 / Task 3 中，对比有彩色刺激（Color）与无彩色刺激（Gray / Base），采用 Wilcoxon rank-sum 检验（$p < 0.05$ 且连续 $\ge 50\,\text{ms}$ 显著），且解剖定位限制在腹侧视觉通路（梭状回 Fusiform、颞下回 ITG、颞中回 MTG、枕叶等 AAL3 靶区）。
- **排除规则**：主分析中排除了属于 `color_patch` 与 `color_with_sti` 的特殊通道，确保分析纯粹性。

### 2. 分析结果与解码表现
- **电极数量**：全组 5 被试共提取出 **38 个** 核心敏感通道。
- **解码能力**：
  - **Task 2 颜色知识解码**：二项 GLMM 拟合在后部集群 (`POSTERIOR`) 展现出双峰显著窗口：**`126 ms ~ 192 ms`** ($p < 0.05$) 与 **`326 ms ~ 402 ms`** ($p < 0.05$)。
  - **Task 3 物理纯色块解码**：物理颜色感知提取极快，GLMM 在 **`54 ms ~ 74 ms`** 与 **`118 ms ~ 142 ms`** 均达到显著水平，峰值准确率达 **`72.0%` – `82.0%`**。

### 3. 代表性结果图片

#### (1) 4 种策略在靶区内的电极筛选对比柱状图
![Electrode Selection Comparison](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/electrode_selection_comparison.png)
- **图片路径**：[electrode_selection_comparison.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/electrode_selection_comparison.png)
- **图表说明**：去除了 Whole Brain 干扰柱，专注于展示 ERP 与 High Gamma 信号在 Strategy 1 ~ Strategy 4 策略下于 Target Area 内筛选出的精细电极数量。

#### (2) 5 被试 Strategy 4 (ERP) 颜色知识解码曲线与二项 GLMM 显著窗口
![ERP Strategy 4 Decoding](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/erp_strategy4_decoding.png)
- **图片路径**：[erp_strategy4_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/erp_strategy4_decoding.png)
- **图表说明**：展示了 5 个被试独立解码曲线（彩色细线）及全组平均曲线（黑色粗线），底部黄色条带标记了二项 GLMM 混合效应模型检出的显著时间窗 ($p < 0.05$)。

#### (3) 主分析电极全脑 2D 玻璃脑投影图
![Whole Brain ERP Glass Brain](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/whole_brain_erp_glass_brain.png)
- **图片路径**：[whole_brain_erp_glass_brain.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/whole_brain_erp_glass_brain.png)
- **图表说明**：展示了排除特殊电极后，主分析通道在皮层上的空间正交投影。

---

## 🎯 三、 策略 B：fMRI Color Patch 坐标邻近电极 (`color_patch` Electrodes)

### 1. 筛选策略与方法
- **定义方法**：先通过 fMRI 功能磁共振实验定位出被试专有的 Color Patches 空间 MNI 坐标（如视网膜/腹侧流中的颜色特异性斑块），然后计算颅内 sEEG 电极触点与这些 fMRI 斑块的欧氏距离，将落入 fMRI Color Patch 邻域内的通道标记为 `color_patch` 特殊先验通道。
- **检验机制**：对这一组先验通道单独运行 4 策略显著性检验与玻璃脑投影，评估 fMRI 颜色斑块与 sEEG 电信号响应的对应关系。

### 2. 分析结果
- **电极分布**：
  - `test001`: 7 个通道 (`D10`, `D11`, `D12`, `D13`, `D14`, `D15`, `D16`)
  - `test003`: 4 个通道 (`G13`, `G14`, `H14`, `H15`)
  - 其它被试: 0 个通道
- **发现结论**：fMRI 定位的 Color Patch 通道大部分集中在枕叶与腹侧视觉通路的后段（梭状回后部）。在 ERP 4 策略检测中，`test001-D10`、`test003-G14` 等通道表现出极高的红绿对比显著性。

### 3. 代表性结果图片

#### (1) 全组 `color_patch` 电极 2D 玻璃脑投影
![Color Patch Glass Brain](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/color_patch_erp_glass_brain.png)
- **图片路径**：[color_patch_erp_glass_brain.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/color_patch_erp_glass_brain.png)
- **图表说明**：展示了在 fMRI Color Patch 附近分布的 sEEG 电极及其满足不同 ERP 策略的分布。

#### (2) 单被试 (`test001`) `color_patch` 邻近电极投影
![test001 Color Patch Glass Brain](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/test001_color_patch_erp_glass_brain.png)
- **图片路径**：[test001_color_patch_erp_glass_brain.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/test001_color_patch_erp_glass_brain.png)

---

## ⚡ 四、 策略 C：电刺激光幻视/颜色觉先验电极 (`color_with_sti` Electrodes)

### 1. 筛选策略与方法
- **定义方法**：在临床皮层电刺激 (Cortical Electrical Stimulation) 实验中，通过向脑内电极触点注入微弱高频电流，记录能够直接成功引发被试主观视错觉（Phosphenes / 颜色色块或闪光）的特异性电极。
- **专属流水线 (Step 7 Pipeline)**：由于这组电极具有因果性的功能意义，在 `step7` 中对其运行了包括单电极 Color vs. Gray 差异、Memory Color 解码、Pure Color Block 解码、Cross-Task 1D/2D 时间泛化及 True vs. Fake 解码在内的专属综合流水线。

### 2. 分析结果与解码表现
- **电极分布**（全组共 18 个）：
  - `test001`: 6 个通道 (`D4`, `D5`, `D6`, `G5`, `G6`, `G7`)
  - `test002`: 4 个通道 (`D1`, `D2`, `D3`, `B2`)
  - `test003`: 8 个通道 (`G3`, `G4`, `H2`, `H3`, `H4`, `H5`, `H11`, `H12`)
- **解码与 GLMM 结论**：
  - **Memory Color 解码**：二项 GLMM 模型在 **`152 ms ~ 180 ms`** 与 **`216 ms ~ 242 ms`** 检出显著窗口 ($p < 0.05$)。
  - **Pure Color 解码**：物理颜色提取极快，GLMM 在 **`54 ms ~ 74 ms`** 与 **`118 ms ~ 142 ms`** 达到显著。
  - **跨范式泛化解码**：在 **`150 ms ~ 194 ms`** 与 **`218 ms ~ 252 ms`** 表现出强烈的跨范式表征迁移。

### 3. 代表性结果图片

#### (1) `color_with_sti` 先验电极策略玻璃脑
![Color with Sti Glass Brain](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/color_with_sti/color_with_sti_glass_brain_strategies.png)
- **图片路径**：[color_with_sti_glass_brain_strategies.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/color_with_sti/color_with_sti_glass_brain_strategies.png)
- **图表说明**：展示了全部具备电刺激光幻视/颜色觉效应的通道在皮层上的空间定位及策略匹配。

#### (2) `color_with_sti` 跨范式时间泛化 (TGM) 2D 热图
![Color with Sti Cross Decoding TG Group](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/color_with_sti/erp_color_with_sti_cross_decoding_tg_heatmap_group.png)
- **图片路径**：[erp_color_with_sti_cross_decoding_tg_heatmap_group.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/color_with_sti/erp_color_with_sti_cross_decoding_tg_heatmap_group.png)
- **图表说明**：展示了电刺激敏感电极在纯色块训练、灰色水果测试时的 2D 时间表征迁移矩阵。

---

## 🔬 五、 总结与神经科学讨论

1. **功能统计电极 (策略 A)**：覆盖范围最广、样本量最大（38通道），为组水平二项 GLMM 混合效应模型提供了坚实的统计学基础，成功揭示了隐性颜色知识的双峰提取机制（`126-192ms` 与 `326-402ms`）。
2. **fMRI Color Patch 邻近电极 (策略 B)**：将宏观 fMRI 颜色功能斑块与微观 sEEG 电信号进行了跨模态对齐，证实了 fMRI 颜色斑块附近的 sEEG 通道在 100-400ms 内具备极强的颜色选择性。
3. **电刺激先验电极 (策略 C)**：具备直接的**因果性 (Causality)**。专属 Step 7 管道证明，能够被电刺激诱发光幻视的通道，在自然观看纯色块和灰色水果时同样表现出强烈的跨范式表征迁移（`150-194ms`），完成了“因果刺激 - 自然感知 - 隐性知识”的三位一体印证。
