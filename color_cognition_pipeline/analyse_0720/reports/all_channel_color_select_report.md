# 全通道 color-select 分析报告

- 本轮不使用 fMRI 坐标或距离进行预筛选。
- test004 因 FreeSurfer/电极定位失败排除。
- test001、test002、test003、test005、test006 完成全通道 ERP/HG 预处理。
- 筛选使用 Task1 100–400 ms color–gray 对比，快速模式 100 次 permutation。
- 本轮按用户要求不进行被试内 FDR；color-select 定义为固定窗内 color−gray effect>0 且单侧 permutation p<0.05。结果表仍保留 q 值供敏感性分析。
- 共获得 36 个 color-select 电极：test001=8、test002=9、test003=11、test005=5、test006=4；test004 仍排除。
- 已完成单被试 decoding、被试平均准确率汇总、50 次伪回合的虚拟被试 decoding，以及 Nilearn 组电极分布图。
- 空间组按 atlas×半球汇总；当前满足至少 3 个电极且至少 2 个被试的组为 left | Supramarginal L、left | White L、right | White R。

这是探索性统计筛选结果，不应表述为“脑内不存在颜色信息”；由于未进行 FDR，多重比较风险必须在正式报告中明确。被试平均结果以被试为统计单位，虚拟被试结果仅表示跨被试电极模式。
