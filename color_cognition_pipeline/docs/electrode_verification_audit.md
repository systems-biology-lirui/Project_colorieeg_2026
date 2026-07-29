# 电极一致性审核与对齐报告 (Electrode Consistency Audit Report)

为了验证并确保在整个分析工作流中，所使用的具有“颜色选择性”电极（即 `colorwithsti` / `channel_colorsti`）**完全且严格**基于各被试 `ieegloc.xlsx` 表格文件中的 `AAL3` 列中标记为 `color_with_sti`（或小写/变体）的电极，我们对全链路的代码逻辑与实际运行时的电极集进行了系统性的审核。

---

## 1. 审核的分析环节

我们在以下 4 个独立的分析组件中，对电极提取逻辑与实际生成的电极列表进行了追踪与比对：
1. **原始数据定义 (`xlsx_elecs`)**：直接从被试 `processed_data/{subject}/{subject}_ieegloc.xlsx` 的 `AAL3` 列中匹配为 `color_with_sti` 标签的原始电极。
2. **解码分析模块 (`dec_elecs`)**：运行在 [decode_memory_color_updated.py](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/codes/decode_memory_color_updated.py) 中，用于加载特征并计算 SVM 的实际电极。
3. **皮层 3D 渲染图 (`cortex_elecs`)**：运行在 [plot_mni_cortex.py](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/codes/plot_mni_cortex.py) 中，在 MNI 平均脑上投影为**红色标记点**的电极。
4. **单通道时序电极图 (`target_elecs`)**：运行在 [plot_target_electrodes.py](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/codes/plot_target_electrodes.py) 中，用于分析 Color vs. Gray 波动曲线并保存为通道结果图像的电极。

---

## 2. 审核比对结果明细

### 被试: test001
* **XLSX 中 AAL3 标注的原始电极**：`['D4', 'D5', 'D6', 'G5', 'G6', 'G7']` (共 6 个)
* **解码分析 (decoding) 实际电极**：`['D4', 'D5', 'D6', 'G5', 'G6', 'G7']`
* **皮层投影 (MNE 3D) 实际电极**：`['D4', 'D5', 'D6', 'G5', 'G6', 'G7']`
* **单电极曲线 (Time course) 绘图电极**：`['D4', 'D5', 'D6', 'G5', 'G6', 'G7']`
* **对齐结论**：**100% 吻合**。

### 被试: test002
* **XLSX 中 AAL3 标注的原始电极**：`['B2', 'D1', 'D2', 'D3']` (共 4 个)
* **解码分析 (decoding) 实际电极**：`['B2', 'D1', 'D2', 'D3']`
* **皮层投影 (MNE 3D) 实际电极**：`['B2', 'D1', 'D2', 'D3']`
* **单电极曲线 (Time course) 绘图电极**：`['B2', 'D1', 'D2', 'D3']`
* **对齐结论**：**100% 吻合**。

### 被试: test003
* **XLSX 中 AAL3 标注的原始电极**（经去重）：`['G3', 'G4', 'H11', 'H12', 'H2', 'H3', 'H4', 'H5']` (共 8 个)
* **解码分析 (decoding) 实际电极**：`['G3', 'G4', 'H11', 'H12', 'H2', 'H3', 'H4', 'H5']`
* **皮层投影 (MNE 3D) 实际电极**：`['G3', 'G4', 'H11', 'H12', 'H2', 'H3', 'H4', 'H5']`
* **单电极曲线 (Time course) 绘图电极**：`['G3', 'G4', 'H11', 'H12', 'H2', 'H3', 'H4', 'H5']`
* **对齐结论**：**100% 吻合**。

---

## 3. 总体结论

经全链路代码与数据审计：
无论在**解码计算 (decoding)**、**皮层 3D 绘图 (plot_mni_cortex)** 还是**电极时序绘图 (plot_target_electrodes)** 中，所有的分析**全部且严格**建立在各被试 `ieegloc.xlsx` 表格文件中的 `AAL3` 列标注为 `color_with_sti`（或小写拼写 `color_with_sti`）的电极集之上，没有任何其他电极的掺杂或遗漏，逻辑与数据完全闭环、对齐无误。
