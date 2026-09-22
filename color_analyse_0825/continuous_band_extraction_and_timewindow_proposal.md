
# 0825 现有代码编号与 QC 文件整理方案

日期：2026-09-07。状态：整理提案，未执行移动、改名或删除，未修改 MATLAB，未重跑分析。

本文直接回答：当前哪些代码保留、改名、合并、修正、退出主目录；现有 QC 图片保留在哪里、哪些可以删除。此前的《新主线全面修正方案》作为科学逻辑修正的补充，不作为文件整理操作清单。

所有相对路径均相对于 `E:/liulab_project/Project_colorieeg_2026/color_analyse_0825/`。逐张图片的来源、目标路径、建议动作与 SHA-256 见同目录的 `qc文件整理清单_20260907.json`，共268条记录。

## 1. 整理后的主入口：C00–C05

**不建议按现有 C00 到 C07 全部顺序运行。建议压缩为 C00–C05 六个清楚的入口，QC 使用 Q 编号，通用绘图函数不编号。**

```text
C00_raw_to_seegdata_0825.m
    原始导入/连续段登记；原始输入准备好后不必每次重跑
C01_preprocess_continuous_0825.m
    唯一连续预处理和严格 Laplacian
C02_match_events_and_behavior_0825.m
    事件/行为对齐和 trial 表
C03_extract_multiband_epochs_0825.m
    连续信号频段功率 → epoch → 特征缓存
C04_screen_color_effects_0825.m
    图片配对、类别等权统计、结果表和候选名单
C05_plot_color_effects_0825.m
    总览图、批量时程图和指定通道导出
```

其中新C03来自旧C06，新C04来自旧C07。**编号改变不代表分析算法改变；需要修正的算法另行列出，不能把改名当作已修复。**

### 1.1 现有每个 MATLAB 文件的处理表

| 当前文件（matlab下）                           | 处理                     | 整理后的文件                                             | 具体调整                                                                         |
| ---------------------------------------------- | ------------------------ | -------------------------------------------------------- | -------------------------------------------------------------------------------- |
| C00_raw_to_seegdata_0825.m                     | 保留编号、修正           | C00_raw_to_seegdata_0825.m                               | 保留原始Session/连续段边界及事件来源；不必为重新编号重导已有原始数据             |
| C01_match_events_0825.m                        | 提前为第一步             | C01_match_events_0825.m                                  | 独立前置读取脑电与行为Trigger，执行LCS对齐并提取pic_id，解耦输出标准trial_info表 |
| C02_preprocess_erp_0825.m                      | 顺延并专注于ERP          | C02_preprocess_erp_0825.m                                | 读取C01 trial_info，去偏置、500Hz重采样、1–30Hz带通滤波提取纯ERP，同杆Laplacian并切分Epoch |
| C03_color_selective_channels_0825.m            | 从主目录移出，归档       | archive/pre_reorganization_20260907/code/matlab/原文件名 | 旧epoch-first HFA流程，不再执行；不把它改成新C03                                 |
| C04_continuous_multiband_screening_0825.m      | 从主目录移出，归档       | 同上                                                     | 旧“提取+统计”一体脚本退出；有用的概览图布局迁入新C05，不保留第二套提取/统计    |
| C05_export_selected_timecourses_0825.m         | 合并职责后移出           | 同上                                                     | 指定通道清单和可选MAT导出并入新C05；删除活跃主线中的原始加载/滤波/Hilbert副本    |
| C05_plot_high_gamma_concordant_channels_0825.m | 合并职责后移出           | 同上                                                     | 26通道列表作为历史展示清单保存；新C05通过列表选择通道，不单独写HG提取流程        |
| C06_extract_and_save_multiband_epochs_0825.m   | 改号、修正               | C03_extract_multiband_epochs_0825.m                      | 唯一频段提取入口，改读新C01连续产物；移除重复去偏置、陷波和重参考                |
| C07_screen_category_unified_multiband_0825.m   | 改号、修正               | C04_screen_color_effects_0825.m                          | 修复同图配对；统一统计；移出批量绘图循环；不再创建空screening目录                |
| plot_channel_timecourse.m                      | 保留名字、修正接口       | plot_channel_timecourse.m                                | 保留最新双栏样式，输入已计算的统计结果；移除内部ttest/ttest2和重新配对           |
| qc/trigger_sync_analysis_0825.m                | 改名、保留为QC函数       | qc/Q01_check_trigger_alignment_0825.m                    | 使用C02同一匹配表，输出同步QC；不另算一套与主表分离的匹配                        |
| qc/check_baseline_validity.m                   | 归档旧实现，新建同职责QC | qc/Q02_check_band_baseline_0825.m                        | 使用新连续特征的归一化前基线；不再从旧epoch重提70–90 Hz                         |

另新增 `C05_plot_color_effects_0825.m`，它是批量绘图入口，不是新增一套分析。它调用保留的 `plot_channel_timecourse.m`，并在入口内直接展示候选列表、输出路径及总览图构造。

### 1.2 需要删去的是哪些代码块

| 删除位置            | 删除内容                                                                   | 替代来源                        |
| ------------------- | -------------------------------------------------------------------------- | ------------------------------- |
| 原两个C05的活跃逻辑 | 原始load、全通道陷波、重参考、重采样、bandpass、Hilbert、epoch、基线       | 新C03保存的特征                 |
| 新C03（旧C06）      | 重复原始预处理                                                             | 新C01连续结果                   |
| 新C04（旧C07）      | 逐通道调用绘图函数的循环、空qc_screen_dir配置、未用n_trials/cat_names      | 新C05集中绘图                   |
| 通用绘图函数        | 自行intersect(pic_id)、ttest及不足配对后自动退回ttest2、自己生成显著性结论 | 新C04输出的效应、p值和配对集    |
| 新C02               | 完整epoch_data读取/筛选/重写、cfg=orig_cfg覆盖                             | 独立trial元数据和分开的阶段配置 |
| 旧一体化C04         | 第二套频段定义、第二套提取和置换                                           | 从活跃主目录整体退出            |

历史代码保留一份原文快照，之后主目录不再出现这些失效入口。不要把旧代码复制到 `matlab/legacy/` 再用 `addpath(genpath(matlab))` 一起加入路径；归档应在matlab目录之外。

### 1.3 保留用户最新绘图修改

本轮重新读取到 `plot_channel_timecourse.m` 已是“左侧总体时程、右侧四类别柱图”的双栏版本，橙棕与灰色配色，300 dpi导出。现有C07批量图片仍是之前的三栏版，二者不能简单视为同一次生成。

布局修改保留，不回退成旧三栏。需要改的是图内统计：当前总体按跨类别裸pic_id求交集，会混淆不同类别同尾号图片；配对不足还改用独立样本检验。应让绘图只消费新C04统计。类别星号若没有对应的明确检验结果，则不展示，不能从总体显著性推导。

## 2. 新目录结构：QC 与结果图分开

**`qc/` 只放信号/事件质量检查，电极筛选和颜色效应图放 `result/`。** 图在哪里由内容决定，不因历史脚本把它叫“QC”就继续混在QC根目录。

```text
color_analyse_0825/
├── matlab/
│   ├── C00_raw_to_seegdata_0825.m
│   ├── C01_preprocess_continuous_0825.m
│   ├── C02_match_events_and_behavior_0825.m
│   ├── C03_extract_multiband_epochs_0825.m
│   ├── C04_screen_color_effects_0825.m
│   ├── C05_plot_color_effects_0825.m
│   ├── plot_channel_timecourse.m
│   └── qc/
│       ├── Q01_check_trigger_alignment_0825.m
│       └── Q02_check_band_baseline_0825.m
├── metadata/                       人工输入、任务映射、通道决定
├── process_data/<feature_run_id>/   新连续数据/特征；按subject/task分层
├── qc/<run_id>/
│   ├── trigger_alignment/<subject>/
│   ├── preprocessing/<subject>/
│   └── band_baseline/<subject>/
├── result/<analysis_run_id>/
│   ├── tables/
│   ├── figures/screening_overview/
│   ├── figures/timecourses/<subject>/<band>/
│   └── exports/                    仅按需要导出的单通道MAT
├── runs/<run_id>/                   配置、命令、日志、环境、输入输出索引
├── docs/
│   └── figure_style_reference/      当前2张双栏样式参考
└── archive/pre_reorganization_20260907/
    ├── code/matlab/                 归档时保持旧相对层级
    ├── tables/                      旧结果表及说明
    └── figures/                     下节指定的旧图
```

这是便于日常查看的布局：runs保存追溯信息，数据/图表按功能保存并带run_id；不要求在runs中再复制一份大数据。它取代前一文档将全部产物放在runs下的目录建议。所有阶段显式传入run路径，禁止自动找“最新文件”。

当前 `process_data/testNNN/` 和 `process_data_new/` 暂不改名、移动、删除；先保留兼容读取，待代码切换成功后再写历史索引。不能仅为了统一命名搬动大MAT或强制重跑。

## 3. 当前 QC 文件盘点

本次实际枚举到268个文件，均为PNG；没有发现 `trigger_sync_*.png`。代码能生成某种图不代表当前目录里已有该图，不能把不存在的同步图列进搬迁清单。

| 当前位置                                          | 数量 | 性质                       |
| ------------------------------------------------- | ---: | -------------------------- |
| qc/C03_color_selectivity_test*.png                |    8 | 旧epoch-first HFA筛选结果  |
| qc/C04_multiband_selectivity_test*.png            |    8 | C04多频段概览根目录副本    |
| qc/c04_screening_multiband/*.png                  |    8 | 与上一行逐一SHA-256相同    |
| qc/C05_timecourse_*.png                           |    7 | 精选多频段旧时程图         |
| qc/high_gamma_timecourses/*.png                   |   26 | HG根目录展示副本/变体      |
| qc/high_gamma_timecourses/testNNN/*.png           |   26 | 按被试组织的HG图           |
| qc/c07_category_unified_timecourses/testNNN/*.png |  183 | 当前缓存主线的旧批量三栏图 |
| qc/test_plot/*.png                                |    2 | 最新双栏样式示例           |
| qc/c07_screening_multiband/                       |    0 | 空目录                     |

C07的183张图分布为：test001 27张、test002 16张、test003 19张、test004 37张、test005 9张、test006 25张、test007 22张、test008 28张。

哈希盘点确认24组重复文件，每组2张。High-Gamma同名根目录图与被试子目录图只有16对完全相同，另外10对不同。不同可能包含内容或编码差异；本方案未对10对进行逐像素/完整视觉验证，因此不把它们列为可删除重复。

## 4. 每类图片的明确处理决定

### 4.1 保留并移到历史结果目录：242张

| 来源                                              | 数量 | 目标目录                                                                              | 原因                                         |
| ------------------------------------------------- | ---: | ------------------------------------------------------------------------------------- | -------------------------------------------- |
| qc/C03_color_selectivity_test*.png                |    8 | archive/pre_reorganization_20260907/figures/c03_epoch_first/                          | 旧处理顺序，保留对照，不作为新主线图         |
| qc/c04_screening_multiband/*.png                  |    8 | archive/pre_reorganization_20260907/figures/c04_screening_multiband/                  | 每被试概览保留一份                           |
| qc/C05_timecourse_*.png                           |    7 | archive/pre_reorganization_20260907/figures/c05_selected/                             | 原精选导出，与后续图不是字节相同副本         |
| qc/high_gamma_timecourses/testNNN/*.png           |   26 | archive/pre_reorganization_20260907/figures/c05_high_gamma/by_subject/testNNN/        | 26电极旧图保留一套按被试版本                 |
| HG根目录中10张不同版本                            |   10 | archive/pre_reorganization_20260907/figures/c05_high_gamma/root_variants/testNNN/     | 同名不同哈希，保留变体待核对                 |
| qc/c07_category_unified_timecourses/testNNN/*.png |  183 | archive/pre_reorganization_20260907/figures/c07_category_unified_timecourses/testNNN/ | 旧C07结果，保留当前基准；修复后应生成新run图 |

历史PNG内部保留原来的C03/C04等编号，不为了新编号改图内标题或重生成。归档README写明“旧编号”，避免新C04与旧C04混淆。

**183张C07图建议保留，但不直接重命名成新C04正式结果。** 其筛选来自现有C07配对/未校正p逻辑；文件整理不会修复科学含义。新统计通过验收后，由新C05另存新结果。

### 4.2 可以删除的重复文件：24张

删除只针对以下确定的副本，保留对应规范副本并记录原路径映射。执行当天再次比对哈希，若文件变化则从删除清单移出。

**A. QC根目录8张C04副本：**

`qc/C04_multiband_selectivity_test001.png` 至 `test008.png`。

它们分别与 `qc/c04_screening_multiband/` 下同名图SHA-256一致。保留下层目录版本并归档，删除根目录副本。

**B. `qc/high_gamma_timecourses/` 根目录以下16张：**

| 被试    | 可以删除根目录副本的通道 |
| ------- | ------------------------ |
| test001 | D14                      |
| test002 | D4、D6                   |
| test003 | D11、E14、G5             |
| test004 | D10、E2、J3              |
| test005 | H8                       |
| test006 | C5                       |
| test007 | F8、H5                   |
| test008 | A10、A11、H7             |

对应文件名统一为 `<subject>_<channel>_High_Gamma.png`。只删除根目录版本，保留 `<subject>/` 子目录内版本。

24张重复副本合计4,391,781字节，约4.19 MiB。删除主要改善组织结构，不会明显减少整个项目的数据体积。

### 4.3 不能按重复文件删除的10张HG根目录变体

| 被试    | 通道       |
| ------- | ---------- |
| test001 | F14        |
| test002 | D3         |
| test003 | H11        |
| test004 | D5、J6、L5 |
| test005 | B6         |
| test006 | B5         |
| test007 | C3         |
| test008 | D3         |

全部放入上表 `root_variants/<subject>/`。以后若验证仅是编码/排版差异且不含独有信息，可另列删除清单；本轮不建议直接删。

### 4.4 test_plot两张：作为样式参考保留

```text
qc/test_plot/test001_G13_Alpha_timecourse.png
qc/test_plot/test002_D4_High_Gamma_timecourse.png
```

移动至 `docs/figure_style_reference/`，保留原文件名。配套说明写明“布局参考，不作为统计结果”。这两张并非上述旧三栏图的重复文件。

本轮已抽查test002 D4的新双栏图与C07旧三栏图：版式明显不同。双栏图中的星号/n.s.由当前绘图函数另行检验产生，不能当作C07置换结果。未来修复统计后可以保留这个布局，但重新生成正式图。

### 4.5 删除空目录和整理后的空壳

- `qc/c07_screening_multiband/` 当前为空，建议删除，同时移除创建它的无用代码。
- 其他旧图移走后，`qc/test_plot/`、`qc/high_gamma_timecourses/` 等若为空再删除。
- 不为凑齐目录树创建空图目录；真正生成某类QC时再创建。

按本方案实施后：268张现有图 → 删除24张完全重复副本 + 归档242张历史图/变体 + 保留2张样式参考。当前没有可直接认定为修正后新主线正式结果的图片。

## 5. 表格、MAT和说明文档配套整理

图片移动必须同步处理索引和表格，避免图仍指向旧统计或链接失效。

| 现有文件/目录                                  | 建议                                                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------------------------ |
| metadata/人工坏道判定表.csv                    | 保留为人工输入，不删除、不自动改判定                                                 |
| metadata/raw_to_seeg_task_map.csv              | 保留为原始任务映射                                                                   |
| metadata/C00_原始数据转SEEG记录.csv            | 保留历史导入记录，写入索引；不为了改号重导                                           |
| metadata/C01_预处理执行汇总.csv                | 历史执行记录保留；新run另生成，不覆盖                                                |
| metadata/C02_Trigger对齐与标签审计.csv         | 保留历史QC审计；新Q01另存run级表，不修改旧表含义                                     |
| metadata/C03_颜色选择电极判定汇总.csv          | 移至archive/pre_reorganization_20260907/tables/，保留原名                            |
| metadata/C04_全频段色彩统一效应汇总.csv        | 同上，与旧C04图片关联                                                                |
| metadata/C07_全频段色彩统一效应汇总.csv        | 同上，与183张旧图关联；不能改名当新C04表                                             |
| metadata/C04_汇总表字段与指标阅读指南.md       | 随旧C04表归档，标记旧字段版本                                                        |
| metadata/High_Gamma精选26个电极时程图谱索引.md | 随HG旧图归档并更新链接；保留26电极历史清单，不当新候选门槛                           |
| process_data/selected_timecourses/（33个MAT）  | 暂保留原位、标记旧导出，归档索引引用；不能按同通道就判重复                           |
| process_data/test001–test008/                 | 暂保留；仍有旧元数据/ERP依赖，待新流程验证后再决定冷存储                             |
| process_data_new/                              | 暂保留当前特征缓存；不因C06改号就删除；新科学参数需新缓存                            |
| README.md                                      | 改成新C00–C05入口和实际输出路径，移除不存在的run_preprocess/validate_preprocess入口 |
| docs/新主线全面修正方案_20260907.md            | 保留作为逻辑修正补充，文件组织以本方案为准                                           |

旧表格归档要在旧C05退出主线、相关读取路径已改之后执行，否则旧HG绘图脚本会找不到C04表。MAT没有在本轮逐变量比较，因此没有MAT删除建议。

## 6. 新输出命名规则

1. 代码用阶段编号；数据和图片名使用内容，不再使用C03/C04等易变编号。
2. 路径带run_id，文件名带subject/task，电极图再带channel/band。
3. 主频段和探索频段地位写进表/索引，不通过拷贝同一图到多个目录表示。
4. 一个图只保存一个规范PNG；需要高质量矢量版可另存PDF，此为不同格式，不视为无用副本。

示例（拟定的新文件，不代表当前已存在）：

```text
qc/<run_id>/trigger_alignment/test001/test001_task1_trigger_alignment.png
qc/<run_id>/band_baseline/test001/test001_task1_High_Gamma_baseline.png
result/<run_id>/tables/task1_color_effects_all_channels.csv
result/<run_id>/tables/task1_color_effects_selected_channels.csv
result/<run_id>/figures/screening_overview/test001_task1_multiband_effects.png
result/<run_id>/figures/timecourses/test001/High_Gamma/test001_task1_D14_High_Gamma.png
```

若需要“26张精选图一眼看完”，生成一个Markdown/HTML索引链接到规范文件，不再把26张PNG在根目录又存一遍。

## 7. 改号时必须同步修改的位置

- C00/Q01等函数声明需与新文件名一致。
- C02对trigger QC的调用、新C05对绘图函数的调用。
- 绘图函数中“请先执行C06”等报错提示与注释。
- C07原有标题、日志、结果名，改为新C04或改用不带编号的内容名称。
- 所有读取旧 `metadata/C04...csv` 的脚本改为显式读取新run结果。
- 所有 `proc_new` 等硬编码旧缓存路径改为传入具体feature_run路径。
- README、图谱索引和科学修正文档中的新旧编号映射。
- MATLAB路径只加主代码及其qc目录，归档目录不加路径。

尤其不能只把旧C06改名C03、旧C07改名C04，却让旧C03/旧C04继续留在主目录；这会形成同号不同算法。归档保留旧完整文件名，主目录只留下新的六个入口。

## 8. 执行顺序和完成检查

本节是后续实施顺序，本次未执行。

1. 保存当前代码快照、图表哈希与源→目标清单；快照包括最新双栏绘图修改。
2. 在新文件名下完成活跃代码调整，先保留旧数据输入；不立即移动仍被引用的表/MAT。
3. 核对入口调用与输出路径，完成必要小范围验证；纯改名不启动全被试分析。
4. 将退出主线的旧脚本移到归档目录，确认MATLAB路径不包含归档。
5. 按JSON清单先归档规范图片，核验目标哈希；再处理24张重复副本，不能先删掉唯一仍存在的规范文件。
6. 保留10张HG差异变体和2张样式参考，更新历史索引。
7. 迁移旧统计表/说明，检查所有引用；空目录确认仍为空后删除。
8. 输出实际执行日志：原路径、目标路径、前后哈希、动作、时间、跳过原因；确认没有未分类文件。

搬移/删除当天应重新读取文件清单和哈希：若文件比本次盘点新增或变化，只处理仍匹配清单的项目，不用通配符一把删整个目录。递归操作前核验绝对路径始终位于0825目标范围。

验收：主目录只有C00–C05及必要工具；主线不调用旧频段提取脚本；QC根目录无颜色效应散图；24份已确认重复文件各有保留副本；242份历史图和2份样式图都有目标记录；旧数据未因整理被删除；未把旧图/表假装成新结果。
