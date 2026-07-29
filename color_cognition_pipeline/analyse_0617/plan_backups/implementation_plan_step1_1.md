# 电极筛选与可视化扩展计划 (Step1_1)

根据用户最新反馈，我们需要对电极筛选进行系统性的升级，并在 ERP 与 HG（高频）特征上同时进行统计学策略检验、文件截取分组，以及绘制多张全脑图（柱状对比图与玻璃脑图）。

## User Review Required

> [!IMPORTANT]
> **关于图片和特征分组**：
> 1. **数据截取存放**：
>    - `select_channel`（主要电极并集）：裁剪出 ERP 和 HG 对应的通道，存入 `analyse_0617/feature/select_channel/{subject}/`。
>    - `more_select_channel`（扩展电极并集）：裁剪对应通道，存入 `analyse_0617/feature/more_select_channel/{subject}/`。
> 2. **ERP/HG 信号图更换路径**：
>    - 绘制的 ERP 信号差异图将保存于 `.../result/select_channel/erp/{subject}/` 和 `.../result/more_select_channel/erp/{subject}/`。
>    - 绘制的 HG 信号差异图将保存于 `.../result/select_channel/hg/{subject}/` 和 `.../result/more_select_channel/hg/{subject}/`。
> 3. **全脑对比图与玻璃脑图**：
>    - **总体+各被试的对比柱状图**（2x4 = 8张）：分别绘制主要和扩展电极的柱状图，保存于 `select_channel/` 和 `more_select_channel/`。
>    - **总体+各被试的玻璃脑图**（2x4 = 8张）：采用 `nilearn` 的 `plot_glass_brain`。实心代表 ERP，空心代表 HG，颜色代表最高匹配策略（绿 > 蓝 > 紫 > 橙黄）。

## Open Questions

> [!NOTE]
> 当前无开放性疑问。我们将完美复现原图的专业配色、标签及布局。

## Proposed Changes

---

### 电极统计学策略与全脑可视化脚本

#### [NEW] [step1_1_select_channel_extended.py](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step1_1_select_channel_extended.py)
1. **双信号（ERP & HG）双重筛选**：
   - 提取各被试 ERP (`task1_ERP_epoched.mat`) 和 HG (`task1_hg_subband.mat`) 数据的通道统计学属性。
   - 分别判定 ERP 与 HG 下各通道是否符合四种策略。
   - 获取 ERP & HG 的主要通道筛选（在枕叶/颞叶靶区内，且 ERP 或 HG 符合至少一种策略）。
   - 获取 ERP & HG 物理邻近（同一轴 $\pm 1$ 且解剖为 unknown / N/A / parahippocampus，且符合至少一种策略）的扩展通道筛选。
2. **汇总表格（XLSX/CSV）升级**：
   - 解析 xlsx 中 MNI 列的坐标串，扩展为 `MNI_X`, `MNI_Y`, `MNI_Z` 三列。
   - 表明 AAL3 ROI 所属大类。
   - 显式表示该电极在 ERP 下符合的策略、在 HG 下符合的策略。
3. **图像生成与路径结构重组**：
   - 绘制 ERP 信号差异图并保存至 `erp/` 子文件夹。
   - 绘制 HG 信号差异图并保存至 `hg/` 子文件夹。
4. **对比图与玻璃脑图生成**：
   - 编写 `plot_electrode_selection_comparison`：为主要和扩展电极各生成 1 张总体和 3 张被试独立的对比柱状图，完全对齐 `electrode_selection_comparison.png` 的美学样式。
   - 编写 `plot_nilearn_glass_brain_electrodes`：为主要和扩展电极各生成 1 张总体和 3 张被试独立的玻璃脑图，完全对齐 `nilearn_glass_brain_electrodes.png`。
5. **特征文件裁剪**：
   - 对 ERP 和 HG 数据做通道截取，并使用 `long_field_names=True` 另存为新的 mat 文件放入对应的 `feature/select_channel/{subject}/` 和 `feature/more_select_channel/{subject}/` 文件夹中。
6. **备份计划**：
   - 拷贝本实施计划至 `analyse_0617/plan_backups/implementation_plan_step1_1.md`。

## Verification Plan

### Automated Tests
- 运行 `/home/lirui/anaconda3/envs/lr2026/bin/python color_cognition_pipeline/analyse_0617/code/step1_1_select_channel_extended.py`。
- 验证所有生成的 16 张全脑大图（8张对比图，8张玻璃脑图）、大量电极时程差异图、特征 mat 文件和汇总表格。

### Manual Verification
- 验证 `analyse_0617/result/select_channel/` 下生成的柱状图、玻璃脑图。
- 验证 `analyse_0617/feature/` 下的裁剪数据是否可被重新加载读取。
- 确认 doc 的 Excel 表中成功增加了 MNI 坐标三列与 ROI 属性。
