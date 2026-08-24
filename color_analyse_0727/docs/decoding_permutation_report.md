# 🧪 单通道 Decoding 置换检验 (Permutation Testing) 显著性报告

本报告利用并发计算在 24 核 CPU 上完成了 `200` 次置换检验，统计评估 14 个交集电极在抽象记忆颜色（红 vs 绿）上的解码显著性。

## 一、 统计规则与并行设置

- **置换次数 (N_perm)**：`200` 次
- **并行核心数 (n_jobs)**：`16` 核
- **显著性阈值**：$p < 0.05$（即真实解码正确率高于 95% 置换零假设分布）
- **评估窗口**：刺激出现后 $t > 0\text{ ms}$

## 二、 各电极最早显著起效潜伏期 (Onset Latency)

| 被试 | 通道名 | MNI Y 坐标 | 最早显著潜伏期 (Onset Latency) |
| :--- | :--- | :--- | :--- |
| test002 | `A1` | `-62.4` | `112 ms` |
| test001 | `G10` | `-62.0` | `336 ms` |
| test005 | `E12` | `-60.5` | `16 ms` |
| test005 | `E11` | `-57.8` | `116 ms` |
| test001 | `G8` | `-55.6` | `72 ms` |
| test005 | `E10` | `-55.2` | `372 ms` |
| test001 | `B5` | `-54.3` | `168 ms` |
| test001 | `H6` | `-53.8` | `236 ms` |
| test001 | `D6` | `-53.0` | `92 ms` |
| test003 | `G8` | `-52.0` | `148 ms` |
| test003 | `G6` | `-47.6` | `136 ms` |
| test001 | `G5` | `-45.9` | `268 ms` |
| test005 | `E6` | `-44.8` | `24 ms` |
| test001 | `C2` | `-39.0` | `140 ms` |

## 三、 空间位置 (MNI Y 轴) 与起效潜伏期的相关性分析

- **评估样本数**：`14` 个具有显著起效潜伏期的交集电极
- **Pearson 相关系数**：$r = -0.099$ ($p = 0.7360$)
- **Spearman 秩相关系数**：$\rho = +0.055$ ($p = 0.8520$)
- **相关性散点回归图**：[latency_vs_y_correlation.png](file:///e:/liulab_project/Project_colorieeg_2026/color_analyse_0727/result/figures/latency_vs_y_correlation.png)
- **三维脑空间 Latency 分布图**：[latency_3d_brain_space.png](file:///e:/liulab_project/Project_colorieeg_2026/color_analyse_0727/result/figures/latency_3d_brain_space.png)

### 3.1 MNI Z 轴 (腹侧-背侧) 与起效潜伏期的相关性

- **Pearson 相关系数**：$r = +0.066$ ($p = 0.8215$)
- **Spearman 秩相关系数**：$\rho = +0.513$ ($p = 0.5126$)
- **Z 轴相关性回归图**：[latency_vs_z_correlation.png](file:///e:/liulab_project/Project_colorieeg_2026/color_analyse_0727/result/figures/latency_vs_z_correlation.png)
- **Y轴与Z轴双视角对比图**：[latency_spatial_axes_correlation.png](file:///e:/liulab_project/Project_colorieeg_2026/color_analyse_0727/result/figures/latency_spatial_axes_correlation.png)
