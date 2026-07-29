# Step 3 & Step 4 ERP 多维度解码分析进度表

- [x] 备份 Step 3 & 4 实施计划到 `analyse_0617/plan_backups/` 目录
- [x] 方案设计与用户沟通（等待用户审查 `implementation_plan.md`）
- [x] 开发与执行 Step 3_1：纯色刺激（task3）红绿多电极解码及 GLMM 显著窗检验
  - [x] 提取 task3 51(红) vs 54(绿) 的 ERP 试次，在 memory_color 电极子集上做 SVM
  - [x] 拟合多被试 GLMM 定位显著时间窗口并绘制 5 折交叉验证解码曲线图
  - [x] 导出正确率与统计明细为 Excel/CSV 并备份
- [x] 开发与执行 Step 3_2：纯色刺激与灰色水果跨任务时间泛化（Temporal Generalization）
  - [x] 信号时域下采样（10ms 步长）与 [-100, 700]ms 裁剪优化
  - [x] 策略 1 (Color-to-Gray-Memory) 计算与 3 被试个体 + Group 平均时间泛化热图绘制
  - [x] 策略 2 (Gray-Memory-to-Color) 计算与 3 被试个体 + Group 平均时间泛化热图绘制
  - [x] 保存时间泛化矩阵明细数据为 CSV/Excel 并备份
- [x] 开发与执行 Step 3_3：单通道时间泛化推广
  - [x] 遍历 31 个 memory_color ERP 电极，分别计算策略 1 和 2 的下采样时间泛化矩阵
  - [x] 对每个电极独立绘制 1行2列 时间泛化热图并输出
  - [x] 将所有电极的二维矩阵压缩打包保存为 `single_electrode_tg_data.npz`
- [x] 开发与执行 Step 4：真假水果颜色跨物体多 ROI 脑区解码
  - [x] 匹配 5 个 ROI 脑区（颞极、颞中回、颞下回、记忆显著、杏仁核）的电极分布
  - [x] 提取 task2 的真/假颜色试次，构造 4 折 Leave-One-Group-Out 跨物体颜色解码
  - [x] 执行多电极 SVM 并在 5 个 ROI 分别计算解码正确率
  - [x] 组水平显著性检验：对多被试 ROI 拟合 GLMM，对单被试 ROI（杏仁核）运行二项检验
  - [x] 绘制 5 张 ROI 组+个体解码时程大图并保存
  - [x] 导出全部 ROI 解码数据为 CSV/Excel
- [x] 完成整体 Walkthrough 汇总报告的编写与备份

