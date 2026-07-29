# Step 2_3 单电极记忆颜色解码与解剖梯度相关性进度表

- [x] 备份实施计划到 `analyse_0617/plan_backups/` 目录
- [x] 编写核心解码与空间相关分析代码 `step2_3_single_electrode_decoding_correlation.py`
  - [x] 自动读取 memory_color 显著电极表作为输入通道
  - [x] 实现单通道 SVM 交叉配对解码及时序对错 0/1 试次收集
  - [x] 实现 80ms 以后最早显著正确率时间点 (ESTP) 二项检验提取算法
  - [x] 对 Group 水平和各被试独立计算 MNI Y 与 ESTP 间 Pearson/Spearman 相关性
  - [x] 提前保存 ESTP 统计与相关性详情为 Excel/CSV 格式，存放至 `analyse_0617/doc/`
  - [x] 绘制 1行2列 渐变折线+相关拟合大图（ERP 4张，HG 4张，小计 8 张大图）
- [x] 运行程序，验证生成的数据和图表
- [x] 完成 walkthrough 总结修改并备份
