# 二项分布广义混合线性模型 (GLMM) 解码显著性分析报告

本分析在组水平 (Group-level) 上采用 **基于二项分布的广义线性混合模型 (Generalized Linear Mixed Model, GLMM with Binomial family and Logit link)** 对每个时间点的灰色刺激颜色知识解码进行了显著性检验。

### 🔬 统计建模方法与公式
- **数据层次**：本检验在 Trial 级别上进行。将所有被试的每个测试试次（Trial）是否分类正确作为二分类因变量 ($Y \in \{0, 1\}$)。
- **随机效应**：将 **被试 (Subject)** 作为随机截距效应，以控制被试内试次之间的相关性。
- **数学模型**：
  $$\text{logit}(P(Y=1)) = \beta_0 + b_{0,\text{Subject}}$$
  其中 $\beta_0$ 为固定效应截距， $b_{0,\text{Subject}} \sim N(0, \sigma^2)$ 为被试随机截距。
- **显著性定义**：检验固定效应截距 $\beta_0$ 是否显著大于 0（即分类概率是否显著高于 50% 机会水平，在 logit 空间对应 0）。使用变分贝叶斯 (Variational Bayes) 估计截距的后验均值和标准差，计算 Wald $z$-score 及单尾 $p$ 值。

### 📈 ERP 信号 GLMM 检验结果
共检测到 **5** 个显著高于机会水平的连续时间段 (logit $\beta_0 > 0$, 单尾 $p < 0.05$, 连续 >20ms)：
- **显著时间窗 1**: `122.0 ms` 到 `206.0 ms` (GLMM 显著)
- **显著时间窗 2**: `210.0 ms` 到 `300.0 ms` (GLMM 显著)
- **显著时间窗 3**: `324.0 ms` 到 `408.0 ms` (GLMM 显著)
- **显著时间窗 4**: `680.0 ms` 到 `698.0 ms` (GLMM 显著)
- **显著时间窗 5**: `764.0 ms` 到 `788.0 ms` (GLMM 显著)

### 📈 HIGHGAMMA 信号 GLMM 检验结果
共检测到 **2** 个显著高于机会水平的连续时间段 (logit $\beta_0 > 0$, 单尾 $p < 0.05$, 连续 >20ms)：
- **显著时间窗 1**: `602.0 ms` 到 `676.0 ms` (GLMM 显著)
- **显著时间窗 2**: `748.0 ms` 到 `812.0 ms` (GLMM 显著)

---
### 🖼️ 显著性标记曲线图链接
1. **ERP 解码 GLMM 显著性标注图**：
   - 项目路径：[group_glmm_erp_strategy4_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_glmm_erp_strategy4_decoding.png)
   - 脑目录路径：[group_glmm_erp_strategy4_decoding.png](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_erp_strategy4_decoding.png)

   ![ERP GLMM显著性图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_erp_strategy4_decoding.png)

2. **High Gamma 解码 GLMM 显著性标注图**：
   - 项目路径：[group_glmm_highgamma_strategy4_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_glmm_highgamma_strategy4_decoding.png)
   - 脑目录路径：[group_glmm_highgamma_strategy4_decoding.png](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_highgamma_strategy4_decoding.png)

   ![High Gamma GLMM显著性图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_highgamma_strategy4_decoding.png)
