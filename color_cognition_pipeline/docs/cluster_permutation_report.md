# Cluster-based 置换检验与 T 检验显著性分析报告

本报告汇总了使用 **基于聚类的一样本置换检验 (Cluster-based Permutation 1-sample t-test, 1000次符号置换)** 与 **点对点单样本 t 检验 (Point-wise 1-sample t-test, 连续 >20ms)** 对基于策略 4 显著电极的颜色知识解码准确率时序进行统计显著性分析，以检验群组平均准确率是否显著高于 50% 的机会水平。

### 📈 ERP 显著性分析结果
#### (A) 严格 Cluster-based 置换检验 (纠正后 p < 0.05)
- *未发现经纠正后显著的连续聚类。原因解释：由于被试量较小 (N = 3)，所有置换组合空间仅为 $2^3=8$ 种。精确置换检验的最小 p 值边界为 $1/8 = 0.125$，因此数学上不可能在群组水平得出低于 0.05 的经纠正显著聚类。*
#### (B) 点对点样本 t 检验 (未纠正，连续时间窗 > 20ms, p < 0.05)
- **显著时间窗 1**: `376.0 ms` 到 `394.0 ms` (点对点单尾 p < 0.05)

### 📈 HIGHGAMMA 显著性分析结果
#### (A) 严格 Cluster-based 置换检验 (纠正后 p < 0.05)
- *未发现经纠正后显著的连续聚类。原因解释：由于被试量较小 (N = 3)，所有置换组合空间仅为 $2^3=8$ 种。精确置换检验的最小 p 值边界为 $1/8 = 0.125$，因此数学上不可能在群组水平得出低于 0.05 的经纠正显著聚类。*
#### (B) 点对点样本 t 检验 (未纠正，连续时间窗 > 20ms, p < 0.05)
- **显著时间窗 1**: `-126.0 ms` 到 `-108.0 ms` (点对点单尾 p < 0.05)

---
### 🖼️ 显著性标记曲线图链接
1. **ERP 解码带显著性阴影图**：
   - 项目路径：[group_average_erp_strategy4_decoding_significant.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average_erp_strategy4_decoding_significant.png)
   - 脑目录路径：[group_average_erp_strategy4_decoding_significant.png](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_average_erp_strategy4_decoding_significant.png)

   ![ERP显著性图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_average_erp_strategy4_decoding_significant.png)

2. **High Gamma 解码带显著性阴影图**：
   - 项目路径：[group_average_hg_strategy4_decoding_significant.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/group_average_hg_strategy4_decoding_significant.png)
   - 脑目录路径：[group_average_hg_strategy4_decoding_significant.png](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_average_hg_strategy4_decoding_significant.png)

   ![High Gamma显著性图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/group_average_hg_strategy4_decoding_significant.png)
