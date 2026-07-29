# Task Progress Ledger (5-Subject Full Pipeline Integration)

- `[x]` 方案制定与被试列表扩展 (`test001`, `test002`, `test003`, `test005`, `test006`)
- `[x]` 管道底层逻辑修整与防范性补全 (解算全局字典 `subj_colors` 与 `temporal_pole_elecs`)
- `[x]` 执行全被试一键管道计算 (`run_original_0617_pipeline_5subjects.py`)
  - `[x]` `step1_1` / `step1_2`: 5 被试电极通道双信号 (ERP & HG) 统计双重筛选、波形渲染、特征裁剪与 MNI Y 坐标分布拟合
  - `[x]` `step2_1` / `step2_2` / `step2_3`: 5 被试 Task 2 颜色知识显著性检验、38 个色块敏感通道 4 折 CV 解码与 ESTP 延迟计算，以及 750 时间点二项 GLMM 混合效应模型拟合
  - `[x]` `step3_1` / `step3_2` / `step3_3`: Task 3 纯色块红绿 SVM 解码、5 被试 2D 跨范式时间泛化 (Temporal Generalization) 热图计算
  - `[x]` `step4` / `step5`: 颞极、颞中回、颞下回、颜色记忆区及后部/前部脑区集群真假水果 5 被试二项 GLMM 拟合
  - `[x]` `step6` / `step7`: 颞极单电极真假颜色 ERP 差异波形绘制与 `color_with_sti` 先验知识电极全套分析
  - `[x]` `step8_2` / `step8_cws`: 全脑 4 策略无盲区映射、全脑玻璃脑投影与单电极 MNI Y 轴 Latency 显性回归拟合
- `[x]` 编译生成 5 被试单文件交互式 HTML 汇总报告 (`build_five_subject_interactive_report.py`)
- `[x]` 更新 `project_summary.md` 与 `walkthrough.md` 并双重同步备份至项目 `color_cognition_pipeline/docs/` 目录
