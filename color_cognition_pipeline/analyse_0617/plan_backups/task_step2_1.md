# Step 2_1 记忆颜色显著性分析进度表

- [x] 备份实施计划到 `analyse_0617/plan_backups/` 目录
- [x] 编写核心分析与绘图代码 `step2_1_memory_color_significance.py`
  - [x] 加载 `select_channel_summary.xlsx` 获取主要电极名单及元数据
  - [x] 加载被试 task2 的 ERP 与 HG 脑电数据
  - [x] 实现 100-400ms 窗口平均 Wilcoxon 秩和检验
  - [x] 实现 100-400ms 时间范围内逐点时序 Wilcoxon 检验及连续 50ms 显著判断
  - [x] 保存完整的数据明细表为 ERP 与 HG 两份独立 Excel/CSV 并存放至 `analyse_0617/doc/`
  - [x] 批量绘制 75 个主要电极的单通道图（左：时间曲线+显著点；右：100-400ms 均值 Bar 图）
  - [x] 统计并绘制多策略饼图（ERP 与 HG 各一张大图）
  - [x] 绘制 Nilearn 2D 玻璃脑电极分类投影图（ERP 与 HG 各一张，区分 Mean, Cont 50ms, Both, Non-Sig）
- [x] 运行程序，验证生成的数据和图表
- [x] 完成 walkthrough 总结修改并备份
