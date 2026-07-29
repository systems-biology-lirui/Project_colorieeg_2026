# Task 3 纯色块红绿解码 GLMM 组水平显著性分析报告

本分析在组水平 (Group-level) 上采用 **基于二项分布的广义线性混合模型 (GLMM with Binomial family and Logit link)** 对 Task 3 纯色块红（Trigger-In:51） vs. 绿（Trigger-In:54）解码进行了时间点级别的显著性检验。

### 🔬 统计建模与模型参数
- **数据来源**：在每个时间点合并 3 名被试各自在 4 折交叉验证测试集上的所有试次（约共计 480 - 600 个试次）。
- **混合效应设计**：试次分类对错（1/0）为二分类目标变量。将 **被试 (Subject)** 作为随机截距，拟合固定效应截距 $\beta_0$。
- **显著判定标准**：Wald 单尾后验 $p < 0.05$ 且连续持续窗口 $>20\,\text{ms}$，表征在该窗口内组水平解码显著高于 50% 机会水平。

### 📈 Task 3 ERP 信号 GLMM 显著窗口
共发现 **6** 个显著高于机会水平的连续时间段：
- **显著时间窗 1**: `182.0 ms` 到 `238.0 ms` (GLMM 显著)
- **显著时间窗 2**: `302.0 ms` 到 `334.0 ms` (GLMM 显著)
- **显著时间窗 3**: `338.0 ms` 到 `356.0 ms` (GLMM 显著)
- **显著时间窗 4**: `368.0 ms` 到 `408.0 ms` (GLMM 显著)
- **显著时间窗 5**: `418.0 ms` 到 `464.0 ms` (GLMM 显著)
- **显著时间窗 6**: `512.0 ms` 到 `554.0 ms` (GLMM 显著)

### 📈 Task 3 HIGHGAMMA 信号 GLMM 显著窗口
共发现 **4** 个显著高于机会水平的连续时间段：
- **显著时间窗 1**: `-214.0 ms` 到 `-196.0 ms` (GLMM 显著)
- **显著时间窗 2**: `-72.0 ms` 到 `-14.0 ms` (GLMM 显著)
- **显著时间窗 3**: `32.0 ms` 到 `50.0 ms` (GLMM 显著)
- **显著时间窗 4**: `858.0 ms` 到 `888.0 ms` (GLMM 显著)

---
### 🖼️ 显著性标记曲线图链接
1. **Task 3 ERP 解码 GLMM 显著性标注图**：
   - 项目路径：[group_glmm_task3_erp_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_glmm_task3_erp_decoding.png)
   - 脑目录路径：[group_glmm_task3_erp_decoding.png](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_task3_erp_decoding.png)

   ![Task 3 ERP GLMM显著性图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_task3_erp_decoding.png)

2. **Task 3 High Gamma 解码 GLMM 显著性标注图**：
   - 项目路径：[group_glmm_task3_highgamma_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_glmm_task3_highgamma_decoding.png)
   - 脑目录路径：[group_glmm_task3_highgamma_decoding.png](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_task3_highgamma_decoding.png)

   ![Task 3 HG GLMM显著性图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_task3_highgamma_decoding.png)
