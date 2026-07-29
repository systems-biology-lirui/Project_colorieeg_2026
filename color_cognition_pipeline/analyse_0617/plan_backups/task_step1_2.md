# Step 1_2 纯色选择性分析进度表

- [x] 备份实施计划到 `analyse_0617/plan_backups/` 目录
- [x] 编写核心颜色选择性分析代码 `step1_2_color_selectivity.py`
  - [x] 导入前一步筛选出的核心主要电极名单
  - [x] 编写 task3（纯色刺激）的条件映射及 50-400ms 平均响应计算
  - [x] 实现 Kruskal-Wallis 整体显著性、红绿 Wilcoxon、黄蓝 Wilcoxon 计算
  - [x] 提前保存绘图数据（包含 MNI_Y、CSI、P值及策略4标记）为 Excel/CSV 格式
  - [x] 编写 CSI 排序与蓝红渐变绘图 logic (ERP 和 HG 各一套 3 子图的大图)
  - [x] 实现策略 4 电极黑圈边框标记与显著性竖直分界虚线
- [x] 运行程序，检查输出文件与图片
  - [x] 检查 `analyse_0617/doc/` 下保存的绘图数据表是否齐全
  - [x] 检查 `analyse_0617/result/select_channel/color_selectivity/` 下生成的 ERP 和 HG 分布图
- [x] 完成 walkthrough 总结修改并备份
