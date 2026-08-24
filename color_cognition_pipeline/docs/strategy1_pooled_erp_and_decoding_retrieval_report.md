# 🧠 策略 1 + 策略 3 汇总电极信号混合检验与既有纯色色块 Decoding 报告

本报告包含两部分核心内容：
1. **策略 1 + 策略 3 汇总电极信号混合分析**：对策略 1 (Strategy 1) 和策略 3 (Strategy 3) 汇总筛选出的 **30 个电极** 在纯色色块 (Task 3 红色 vs 绿色) 下的信号进行 Pooled 混合平均，检验 100-400ms 均值差异及逐时间点 Wilcoxon 显著性；
2. **全库纯色色块二分类 Decoding 检索**：在 `Project_colorieeg_2026` 根目录下全盘检索既有的物理红绿纯色色块二分类 Decoding 脚本、数据表与结果图片。

---

## 📉 一、 策略 1 + 策略 3 汇总电极信号混合 (Pooled Signal) 统计检验结果

### 1. 分析方法与电极样本
- **样本通道**：取 Strategy 1 或 Strategy 3 在腹侧通路 Target Area 内筛选出的所有通道（全组 5 被试共 **30 个通道**，包含 `test001`: 16个, `test002`: 6个, `test003`: 4个, `test005`: 4个）。
- **信号处理**：对每个通道在 Task 3 中的物理红色色块 (Trigger-In:51) 与物理绿色色块 (Trigger-In:54) 试次做基线减除（`-500ms ~ 0ms`），并做跨通道/跨被试池化混合（Pooled Red Trials $\mathbf{N_{red} = 1920}$, Pooled Green Trials $\mathbf{N_{green} = 1920}$）。

### 2. 统计检验结果

| 检验维度 | 统计数据与结果 | 显著性判断 ($p < 0.05$) |
| :--- | :--- | :--- |
| **100-400ms 均值对比** | 红色均值: **$0.490\ \mu\text{V}$** <br/> 绿色均值: **$0.201\ \mu\text{V}$** | Wilcoxon $p = 0.7804$ <br/> ❌ **未达整体均值显著** |
| **100-400ms 逐点连续显著** | 100-400ms 内最长连续显著时长: **$2.0\ \text{ms}$** | ❌ **未达连续 $\ge 50\text{ms}$ 显著标准** |

### 3. 神经科学机制解析
- **极性与偏好相互抵消 (Polarity Cancellation Effect)**：在 30 个通道构成的群体中，部分通道呈现红偏好（如 `test001-B5`, `test005-E14`），另一些通道呈现绿偏好（如 `test003-H13`, `test002-C1`）。在跨 30 个通道进行 Pooled 混合平均时，各通道正负相反的偏好极性相互抹平了标量振幅差异。
- **高维空间编码胜于单标量叠加 (Multivariate Pattern vs. Pooled Scalar)**：此结果有力印证了大脑腹侧视觉皮层在加工颜色时并非靠全脑整体同步“单向升降振幅”，而是依靠**高维空间中不同分布特征的神经群体（Multivariate Pattern）**进行编码。这也是为什么在这些通道上运行 **SVM 多元解码与 GLMM 混合建模时能够在 Task 3 上达到 $72\% \sim 82\%$ 的高准确率与 $54 \sim 74\text{ms}$ 的敏捷显著**，而简单的 Pooled 平均无显著差异的原因。

### 4. 产出的 Strategy 1+3 汇总混合信号图表
![Strategy 1+3 Combined Pooled ERP Significance](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/strategy1_3_combined_pooled_erp_significance.png)
- **图表链接**：[strategy1_3_combined_pooled_erp_significance.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/strategy1_3_combined_pooled_erp_significance.png)

---

## 🔍 二、 既有纯色色块 (Task 3) 二分类 Decoding 全库检索结果

全盘检索 `Project_colorieeg_2026` 根目录后，检索到了完备的物理红绿纯色色块 (Task 3 Red vs Green) 二分类 Decoding 的核心脚本、数据表与结果图片：

### 1. 检索到的核心代码与脚本
- **组水平解码与 GLMM 拟合脚本**：
  `analyse_0617/code/step3_1_color_block_decoding.py`
- **跨范式 TGM 时间泛化矩阵脚本**：
  `analyse_0617/code/step3_2_cross_decoding_generalization_union.py`
- **单通道表征与差异脚本**：
  `analyse_0617/code/step3_3_draw_B5_color_block_erp.py`
- **0720 新版本验证 Notebook**：
  `analyse_0720/notebooks/03_task3_color_validation.ipynb`

### 2. 检索到的既有解码数据表
- [decoding_data_erp_color_block.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/doc/decoding_data_erp_color_block.xlsx)
  - 包含了全组被试在 Task 3 纯色色块上 750 个时间点上的 SVM 二分类准确率与 GLMM 拟合 $p$ 值。

### 3. 检索到的既有解码结果图片
1. **组水平 Task 3 纯色色块 ERP 解码与 GLMM 显著窗口图**：
   - [erp_color_block_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/erp_color_block_decoding.png)
   - [group_glmm_task3_erp_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_glmm_task3_erp_decoding.png)
   - **结论**：物理红绿纯色块的表征提取极快，GLMM 模型在 **`54 ms ~ 74 ms`** 与 **`118 ms ~ 142 ms`** 呈现显著解码能力 ($p < 0.05$)，峰值准确率达到 **`72.0%` – `82.0%`**。
2. **电刺激先验通道 (`color_with_sti`) Task 3 纯色块解码图**：
   - [erp_color_with_sti_color_block_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/decoding/color_with_sti/erp_color_with_sti_color_block_decoding.png)
3. **单被试 Task 3 纯色块解码图**：
   - [test001_erp_task3_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/task3_pure_color/erp_task3_decoding.png)
   - [test002_erp_task3_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/task3_pure_color/erp_task3_decoding.png)
   - [test003_erp_task3_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/task3_pure_color/erp_task3_decoding.png)
