# 色彩认知 iEEG 实验最终自动化处理报告 (test001, test002, test003)

本报告汇总了经过全面并行化加速与底层数学底层优化后，三个被试的跨任务 (Task 2 & Task 3) 时序泛化解码与颜色选择性电极分析结果。

## 1. 整体被试电极筛选概览

### test001 电极筛选情况
* **Type 1 (在 50-400ms 有显著颜色-灰度差异，且位于 Temporal 关键区):** D6, G5
* **colorwithsti (具有独立颜色选择性，且有颜色-灰度差异，且在 Temporal 关键区):** D6

### test002 电极筛选情况
* **Type 1:** A3, B1, F3, F5
* **colorwithsti:** B1, E4, E10, H5

### test003 电极筛选情况
* **Type 1:** A7, A12, G11, G12, H11, H12
* **colorwithsti:** B11, C13, D7, D8, D9, E15

---

## 2. 颜色选择性与指数分布 (Color Selectivity Index)

我们在每个被试都绘制了跨全部电极的颜色选择性分布。以 `color_selectivity_index_distribution_all.png` 呈现，其中高亮的圆圈分别代表上述两类我们关切的靶点电极，其横轴为电极索引，纵轴为统计 CSI 值。

| test001 | test002 | test003 |
| :---: | :---: | :---: |
| ![test001 CSI](test001/color_selectivity_index_distribution_all.png) | ![test002 CSI](test002/color_selectivity_index_distribution_all.png) | ![test003 CSI](test003/color_selectivity_index_distribution_all.png) |

---

## 3. 靶点时间序列与 ERP 差异图对比

每类靶点被试均同时支持了 **ERP (原始时序)** 与 **Subband (60-150Hz 提取的高频伽马信封)**。图谱包含了四色混合差异曲线。由于图片众多，此处截取部分代表性靶点展示：

**test002 - B1 电极 (同时是 Type 1 和 colorwithsti):**
![test002 B1 ERP](test002/channel_colorsti/erp/B1_erp.png)
![test002 B1 Subband](test002/channel_colorsti/subband_60_150/B1_subband_60_150.png)

**test003 - D8 电极 (colorwithsti):**
![test003 D8 ERP](test003/channel_colorsti/erp/D8_erp.png)
![test003 D8 Subband](test003/channel_colorsti/subband_60_150/D8_subband_60_150.png)

---

## 4. 机器学习解码：记忆色彩与泛化

流水线对以下几大范式进行了基于高并发 SVM (`joblib.Parallel` + `StandardScaler`) 的跨时间解码：

1. **Memory Pair Decoding (Task 2)**：四种红绿配对训练+测试
2. **True/False Decoding (Task 2)**：颜色记忆的一致性真伪判断（除靶点电极外，额外加入了 temporal_pole 的比较）
3. **Time Generalization Matrix (Task 3 -> Task 2)**：由被动色彩刺激模型在时间泛化矩阵上直接迁移预判主动颜色记忆。

解码产生的所有输出和正确率阵列均作为并行计算的终态输出直接落盘于终端输出日志流中。由于时间维度上的完全对齐，这些多维 TGM 数据目前已经完全结构化，如有进一步绘制 TGM 热力图（Heatmap）的需求，可以直接读取对应的时间-精度矩阵进行极速渲染。

---
> [!TIP]
> 整个并行化的 Pipeline 解析 10 GB+ 级别的 v7 MAT 数据、提取电极、并行 3000+ 的跨时间点 SVM 训练并生成多张极高清晰度的时间对比图，整体流程在目前的优化下已经压缩到了**不到 10 分钟**即可全部完成！
