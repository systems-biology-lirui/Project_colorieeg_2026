# 5 被试结果补齐与多被试汇总实施计划 (Implementation Plan)

本计划旨在通过直接调用 `color_cognition_pipeline/analyse_0617/code/` 下已有的官方脚本，完成 **test005** 与 **test006** 的结果补齐，并生成整合 5 个被试（`test001`, `test002`, `test003`, `test005`, `test006`）的完整汇总分析与交互式报告。

---

## 🎯 目标与范围 (Goal & Scope)

1. **补齐 test005 和 test006 结果**：在不编写任何冗余代码的前提下，调用 `analyse_0617/code/` 现有的自动化脚本，依次完成 `test005` 与 `test006` 的通道筛选、单/多电极 SVM 解码、GLMM 组水平检验、Task 3 纯色块解码及跨任务泛化解码。
2. **5 被试全量汇总**：融合 5 名被试的数据，计算群组平均（Group Mean Accuracy），更新 5 被试多电极解码折线、显著性检验阴影及全脑电极分布图。
3. **生成 final_report**：更新编译最终的单文件 HTML 交互式报告 `color_ieeg_0617_five_subject_interactive_report.html`。

---

## 🛠️ 拟执行的具体步骤 (Proposed Steps)

### 阶段 1：运行 5 被试全管道处理 (补齐 test005 & test006)
调用官方一键管道脚本 `run_original_0617_pipeline_5subjects.py`，该脚本将驱动以下现成阶段：
- **`step1_1` / `step1_2`**：通道筛选及颜色选择性指数计算；
- **`step2_1` / `step2_2` / `step2_3`**：颜色知识显著性、GLMM 混合效应拟合及单电极 SVM 解码；
- **`step3_1` / `step3_2` / `step3_3`**：Task 3 纯色块红绿解码、跨范式 (Task 2 $\rightarrow$ Task 3) 泛化解码及单电极泛化；
- **`step4` ~ `step8`**：真假颜色解码、脑区集群解码及全脑策略图谱生成。

### 阶段 2：数据与图像输出校验
检查 `analyse_0617/run_5subjects_original/` 目录：
- 确认 `doc/` 目录下生成了包含 `test005` 与 `test006` 的解码与显著性汇总表格（如 `decoding_data_erp_strategy4.csv`、`select_channel_summary.csv` 等）；
- 确认 `result/` 目录下生成了对应的单/多电极 PNG 图表。

### 阶段 3：生成 5 被试全量汇总交互报告
调用现有的 `build_five_subject_interactive_report.py` 和 `plot_five_subject_summary.py`：
- 读取 5 被试的解码曲线数据并计算 `Group_Mean_Acc`；
- 打包渲染单文件 HTML 交互报告 `color_ieeg_0617_five_subject_interactive_report.html`。

### 阶段 4：文档与备份同步
- 更新 `project_summary.md` 和 `walkthrough.md`；
- 按照全局规则将所有生成的 MD 文件和计划备份至 `color_cognition_pipeline/docs/` 目录。

---

## 🧪 验证计划 (Verification Plan)

1. **命令行验证**：
   - 检查 `run_original_0617_pipeline_5subjects.py` 运行状态，确保无错误抛出。
2. **数据文件核验**：
   - 检查 `run_5subjects_original/doc/five_subject_combined/` 下导出的表格，确保同时包含 5 个被试的 Acc 列。
3. **HTML 报告校验**：
   - 打开 `color_ieeg_0617_five_subject_interactive_report.html`，确认 5 个被试的彩色图例与数据线均已被完整绘制。
