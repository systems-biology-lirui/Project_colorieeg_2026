# Step 2_2 记忆颜色解码与 GLMM 显著性进度表

- [x] 备份实施计划到 `analyse_0617/plan_backups/` 目录
- [x] 编写核心解码与 GLMM 拟合代码 `step2_2_memory_color_decoding_glmm.py`
  - [x] 实现多核并行有效性检测 benchmark
  - [x] 实现三套方案电极的自动过滤与提取（Strategy 4, Union, Memory Color Sig）
  - [x] 实现 SVM 记忆颜色 4 种交叉配对解码与对错 0/1 试次收集
  - [x] 实现 750 个时间点上的 Trial-level 二项分布混合效应模型 (GLMM) 拟合（包含被试随机截距）
  - [x] 提前保存解码与 GLMM 统计数据至 Excel 与 CSV 文件，存放于 `analyse_0617/doc/`
  - [x] 绘制包含个体虚线、均值实线、GLMM 显著阴影及顶部红线的解码图（共 6 张图）
- [x] 运行程序，验证生成的数据和图表
- [x] 完成 walkthrough 总结修改并备份
