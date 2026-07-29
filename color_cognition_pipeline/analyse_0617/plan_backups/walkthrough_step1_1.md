# 双信号特征筛选、全脑可视化与数据裁剪总结 (Step1_1)

我们已成功实现了电极筛选的扩展方案（`step1_1_select_channel_extended.py`），对 ERP 与 High Gamma (HG) 信号特征同时进行了 4 种策略的统计学筛选，并完成了通道数据的裁剪另存、全脑柱状数量对比图以及 Nilearn 2D 玻璃脑投影图的批量绘制。

## 📊 电极统计与全脑对比图

### 1. 电极筛选数量结果

| 电极组别 | 被试 | ERP 筛选通道数 | HG 筛选通道数 | 并集去重通道数 | 扩展邻近通道数 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **主要电极 (`select_channel`)** | test001 | 18 | 18 | **27** | - |
| | test002 | 15 | 24 | **33** | - |
| | test003 | 7 | 10 | **15** | - |
| **扩展电极 (`more_select_channel`)** | test001 | 1 | 4 | **5** | 5 (邻居) |
| | test002 | 2 | 2 | **3** | 3 (邻居) |
| | test003 | 0 | 1 | **1** | 1 (邻居) |

---

### 2. 总体电极筛选策略数量对比柱状图

下图展示了 ERP 与 HG 条件下，全脑（Whole Brain）与限定在核心脑区（Target Area）内的 4 种不同策略筛选出的电极数对比柱状图：

![主要电极筛选策略对比柱状图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/electrode_selection_comparison.png)

*注：图中绿色（In Target Area）即为我们最终筛选的主要电极。*

---

### 3. Nilearn 2D 玻璃脑投影电极图 (Group Level)

下图展示了 3 个被试所筛选的主要电极在全脑中的 MNI 空间正交投影。
- 实心圆代表 ERP
- 空心圆代表 HG
- 颜色代表最高匹配策略（🟢 绿-策略1，🔵 蓝-策略2，🟣 紫-策略3，🟡 橙黄-策略4）

![全脑电极 2D 玻璃脑投影分布图](/home/lirui/.gemini/antigravity-ide/brain/5f014c5f-12bd-44b1-9e95-6fc5fef4e12d/nilearn_glass_brain_electrodes.png)

---

## 📂 升级后的汇总表与数据裁剪目录

1. **升级版汇总表格**：
   - 增加了 `AAL3_ROI` (枕叶/颞叶等 ROI 靶区分类)、`MNI_X`, `MNI_Y`, `MNI_Z` 三维坐标列。
   - 用 `ERP_Selected` / `ERP_Strategies_Matched` 和 `HG_Selected` / `HG_Strategies_Matched` 详细标明了两个特征通道的筛选细节。
   - [主要电极汇总表](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/select_channel_summary.xlsx)（备份：[select_channel_summary.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/select_channel/select_channel_summary.xlsx)）
   - [扩展电极汇总表](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/doc/more_select_channel_summary.xlsx)（备份：[more_select_channel_summary.xlsx](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/more_select_channel/more_select_channel_summary.xlsx)）

2. **裁剪后的特征 Mat 数据**：
   - 提取并保存了每个被试只包含筛选出通道的 ERP 与 HG 数据（格式为包含 `'labels'` 等裁剪列表，利用 `long_field_names=True` 兼容性写入）：
     - 主要通道数据存放路径：[select_channel/](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/feature/select_channel/) 下的各被试目录。
     - 扩展物理邻近电极数据路径：[more_select_channel/](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/feature/more_select_channel/) 下的各被试目录。

3. **信号图更新目录**：
   - 分门别类保存在以下文件夹下，以避免文件混乱：
     - 主要电极时程图：[ERP](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/select_channel/erp/) / [High Gamma](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/select_channel/hg/)
     - 扩展邻近电极时程图：[ERP](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/more_select_channel/erp/) / [High Gamma](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/more_select_channel/hg/)
   - 总体和各个被试的对比柱状图与玻璃脑图已分类存放在结果目录 `result/select_channel/` 和 `result/more_select_channel/` 之中。

---

## 🔗 相关代码链接
- [step1_1_select_channel_extended.py](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_1_select_channel_extended.py)
