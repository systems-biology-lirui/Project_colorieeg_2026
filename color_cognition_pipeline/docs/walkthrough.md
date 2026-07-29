# 🚀 5 被试 iEEG 全流程管道与汇总报告完成度 Walkthrough

## 一、 完成的核心工作

1. **管道全线连贯自动化运行 (`run_original_0617_pipeline_5subjects.py`)**：
   - 无缝覆盖了 `test001`, `test002`, `test003`, `test005`, `test006` 全部 5 个被试。
   - 完成了全部 16 个阶段的计算，包括双信号统计通道筛选、38 个色块敏感电极的单电极 4 折 CV 解码与 ESTP 早期显著起始点计算、5 被试在 750 个时间点上的 Binomial GLMM 混合效应拟合、Task 3 物理纯色块解码、多电极与单电极 2D 跨范式时间泛化热图导出、多 ROI（颞极、颞中回、颞下回、颜色记忆区、杏仁核）真假水果解码、后部/前部集群解码、Temporal Pole ERP 差异波以及全脑 4 策略和 `color_with_sti` 特殊电极可视化图谱。
2. **纯色色块 (Task 3) 重点通道信号响应差异精细绘制**：
   - 按照标准双子图风格（左：-200~800ms 时程响应 + SEM 阴影 + 100-400ms 高亮 + 逐点 Wilcoxon 显著标记；右：100-400ms 均值柱状图 + p 值显著性连线及统计框），完成了以下 4 个代表性通道在物理红色 (Trigger-51) 与物理绿色 (Trigger-54) 色块刺激下的响应图绘制：
     - `test001 - B5`
     - `test002 - C1`
     - `test003 - H13`
     - `test005 - E14`
3. **代码兼容性修复与补全**：
   - 修复了 `step4`、`step5` 和 `step7` 中的 `subj_colors` 局部字典缺少 `test005` 与 `test006` 的 `KeyError`。
   - 修复了 `step6` 中 `temporal_pole_elecs` 字典缺少 `test005` (`['A1', 'A2', 'A3']`) 和 `test006` (`[]`) 的 `KeyError`。
   - 增强了 `run_original_0617_pipeline_5subjects.py` 中的 `runtime_root` 注入正则匹配，完美兼容 `step8_2` 和 `step8_cws` 的路径定义。
4. **单文件 5 被试 HTML 汇总报告编译**：
   - 运行 `build_five_subject_interactive_report.py` 成功编译输出了全量 5 被试交互式 HTML 报告：[final_report/color_ieeg_0617_five_subject_interactive_report.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/final_report/color_ieeg_0617_five_subject_interactive_report.html)。
5. **项目文档与计划备份**：
   - 实施计划 `implementation_plan.md`、任务表 `task.md`、项目总结 `project_summary.md` 以及本演练文档 `walkthrough.md` 均已更新并同步备份至 [color_cognition_pipeline/docs/](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/docs/) 目录。

---

## 二、 关键计算结果与生理学发现

1. **重点通道纯色色块 (Task 3) 红绿 ERP 响应图谱**：
   - ![Combined 4 Electrodes Pure Color Block ERP](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/Combined_4_Electrodes_Pure_Color_Block_ERP.png)
2. **5 被试电极分布**：全组筛选出 **38 个** 颜色选择性敏感电极（`test001`: 8, `test002`: 14, `test003`: 9, `test005`: 6, `test006`: 1）。
3. **Task 2 颜色知识二项 GLMM 显著时间窗**：
   - 后部集群 (`POSTERIOR`, Y 轴解剖区间 `[-90.3, -42.3] mm`)：**`126 ms ~ 192 ms`** ($p < 0.05$) 与 **`326 ms ~ 402 ms`** ($p < 0.05$) 双峰显性表征。
4. **Task 3 物理纯色块 GLMM 显著时间窗**：
   - 物理感知提取极快：**`54 ms ~ 74 ms`** 与 **`118 ms ~ 142 ms`** ($p < 0.05$)。
5. **跨范式 1D 表征迁移 GLMM 显著窗口**：
   - **`150 ms ~ 194 ms`** 与 **`218 ms ~ 252 ms`** ($p < 0.05$) 表现出强的跨任务认知迁移。
