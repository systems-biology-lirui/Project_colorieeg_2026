# 跨任务解码 (Task 2 联想 $\rightarrow$ Task 3 物理) GLMM 组水平显著性分析报告

本分析在组水平 (Group-level) 上采用 **基于二项分布的广义线性混合模型 (GLMM with Binomial family and Logit link)** 对**使用 Task 2 隐含颜色知识（灰色刺激）进行训练、在 Task 3 真实纯色块刺激上测试**的跨任务泛化解码进行了显著性检验。

### 🔬 统计建模与模型参数
- **科学假说**：若使用无彩色隐含颜色常识检索（Task 2）训练出的分类超平面，能够显著预测真实的物理颜色刺激（Task 3），表明大脑在自上而下联想颜色时，与自下而上物理感知共享了相同或重叠的神经编码代码。
- **混合效应设计**：合并 3 被试在 Task 3 测试集上的全部试次对错数据（每个时间步约 480 - 600 个样本）。以 **被试 (Subject)** 作为随机截距，检验固定截距效应 $\beta_0$ 是否显著大于 0。
- **显著判定标准**：Wald 单尾后验 $p < 0.05$ 且连续持续窗口 $>20\,\text{ms}$，标定组水平解码显著高于 50% 机会水平。

### 📈 Task 2 $\rightarrow$ Task 3 ERP 信号 GLMM 显著窗口
- *未发现符合条件的连续显著时间窗口。*

### 📈 Task 2 $\rightarrow$ Task 3 HIGHGAMMA 信号 GLMM 显著窗口
共发现 **2** 个显著高于机会水平的连续时间段：
- **显著时间窗 1**: `-306.0 ms` 到 `-284.0 ms` (GLMM 显著)
- **显著时间窗 2**: `320.0 ms` 到 `342.0 ms` (GLMM 显著)

---
### 🖼️ 显著性标记曲线图链接
1. **Task 2 $\rightarrow$ Task 3 ERP 解码 GLMM 显著性标注图**：
   - 项目路径：[group_glmm_task2_to_task3_erp_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_glmm_task2_to_task3_erp_decoding.png)
   - 脑目录路径：[group_glmm_task2_to_task3_erp_decoding.png](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_task2_to_task3_erp_decoding.png)

   ![Task 2 -> Task 3 ERP GLMM显著性图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_task2_to_task3_erp_decoding.png)

2. **Task 2 $\rightarrow$ Task 3 High Gamma 解码 GLMM 显著性标注图**：
   - 项目路径：[group_glmm_task2_to_task3_highgamma_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_glmm_task2_to_task3_highgamma_decoding.png)
   - 脑目录路径：[group_glmm_task2_to_task3_highgamma_decoding.png](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_task2_to_task3_highgamma_decoding.png)

   ![Task 2 -> Task 3 HG GLMM显著性图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_glmm_task2_to_task3_highgamma_decoding.png)
