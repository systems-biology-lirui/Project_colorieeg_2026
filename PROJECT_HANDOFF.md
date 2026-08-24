# Project Color iEEG 2026：代码交接与目录说明

最后核查：2026-08-07

这份文档是给普通程序员或新的 Coding Agent 的入口说明。它回答三个问题：

1. 这个项目要研究什么；
2. 哪些目录是当前主线，哪些只是旧代码或独立子项目；
3. 数据从哪里来，经过什么处理，下一步如何运行和扩展。

## 1. 科学目标

项目使用颅内脑电（SEEG/iEEG）研究颜色信息在腹侧视觉通路中的神经编码，重点比较两类信息：

- 物理颜色：真实颜色图片或纯颜色色块直接带来的颜色感知信号；
- 记忆颜色：看到灰色物体时，根据常识自动检索的颜色知识，例如看到灰色草莓时仍然激活“红色”相关表征。

核心问题是：物理颜色和记忆颜色是否在腹侧视觉通路的相近电极、相近时间窗或相似多频段模式中被编码。

主要科学路线：

1. 用 Task 1 的 color-vs-gray 差异做功能筛选；
2. 用论文中的 PC、CC、AC Color Patch 坐标做空间先验筛选；
3. 取两类筛选的交集 CSC（Color-Selective Channel）；
4. 在 CSC 上比较幅度、频谱和单电极 decoding；
5. 后续再单独分析 Task 2 的 true/false 条件，以及全部共同电极。

## 2. 当前主线：`color_analyse_0727`

`color_analyse_0727/` 是当前唯一应该继续扩展的颜色分析工作区。它不读取旧版结果文件作为输入，当前最终分析直接读取：

```text
根目录/seegdata/                 # EEGLAB 连续数据：.set + .fdt
根目录/processed_data/           # 当前电极定位表和坐标/ROI 元数据
color_analyse_0727/Data/         # 原始 MATLAB 实验记录、刺激图片身份、marker、行为和刺激日志
color_analyse_0727/qc/           # 坏道候选表和人工决定
color_analyse_0727/process_data/ # 重建后的 HDF5
```

当前工作区目录：

```text
color_analyse_0727/
├── Data/            原始实验端 MATLAB 记录（按被试保存，不能与 process_data 混淆）
├── pipeline/       可复用的读取、条件、QC、滤波、重参考、epoch、HDF5函数
├── scripts/        命令行入口：审计、manifest、HDF5构建、验证
├── notebooks/      人类可读的流程入口（目前仍是00和01两个notebook）
├── matlab/         rawdata→seegdata桥接和示例绘图
├── metadata/       manifest、坏道决定、定位备份、刺激行为记录、验证报告
├── qc/             候选坏道表、滤波后诊断图、人工审核说明
├── process_data/   当前21个最终HDF5，后续分析的标准输入
├── analysis/       当前颜色选择、CSC、频谱和decoding分析
├── result/         生成的图、表、Markdown、PPT和缓存
├── prompt/         用户需求和分析规范的历史记录
└── docs/           历史报告和方法说明
```

### 当前有效的数据合同

每个被试有3个HDF5：

```text
process_data/testXXX/task1_epoched_1_200Hz.h5
process_data/testXXX/task2_epoched_1_200Hz.h5
process_data/testXXX/task3_epoched_1_200Hz.h5
```

每个HDF5的主要内容：

- `epochs/<condition>`：`[trial, bipolar_center, time]`；
- `time_ms`：−500到998 ms，共750个点，采样率500 Hz；
- `labels`：严格左右相邻重参考后保留的中心电极名；
- `condition_names`、`condition_triggers`：条件注册表；
- `trial_counts`：每个条件的试次数；
- `excluded_channels`：该被试明确排除的单端坏道。

当前HDF5在预处理阶段不做事件级baseline correction。分析脚本可以按分析目的使用−200到0 ms baseline，避免把预处理和统计策略绑定在一起。

### 原始实验记录：`Data/`

`color_analyse_0727/Data/testXXX/` 是新增的原始实验记录层，不是解码结果，也不是连续 EEG。当前共 7 个被试、96 个文件（93 个 `.mat` 和 3 个辅助 `.m`），约 1.22 MB。主要内容为：

- `testXXX_Task1PassiveRealGray_*.mat`：Task 1 真实/灰色物体图片实验；
- `testXXX_Task2PassiveFruitFull_*.mat`：Task 2 水果 × true/false/gray 实验；
- `testXXX_Task3PassiveColorPatches_*.mat`：Task 3 纯颜色色块实验；
- `session_history_Task*.mat`：实验 session history；
- `groupedData.mat`、`task3groupedData.mat`：按类别/形状组织的 trial 索引；
- `testXXX_Task6/7/8*.mat`：电刺激相关实验记录，当前被动颜色 decoding 主线尚未纳入；
- `color_calibration.mat`、`event.mat`：颜色校准和实验事件辅助记录。

被动任务 session 文件通常包含 `cfg`、`results` 和 `stimData` 三类变量。`results` 保留 `imgName`、`marker`、`onsetTime`、`offsetTime`、`duration`、`response`、`isCatch` 等 trial-level 信息；`stimData` 保留呈现时使用的图片文件名、类别和 marker。该层因此比当前 HDF5 的 condition-level epoch 更适合恢复“每一个 EEG trial 对应哪一张图片”。

当前对齐脚本为 `color_analyse_0727/scripts/normalize_task_labels_and_index_passivecolorpatch.py`，其流程是：

```text
Data/testXXX/Task3 results + stimData
  → 统一前四位被试的 Task2/Task3 文件名颠倒问题
  → 读取 results.marker
  → 与 seegdata/testX/erp3.set 的 EEG event marker 做序列对齐
  → 排除 catch / 删除的异常事件
  → 写出 trial-level passivecolorpatch_shape_trial_index.csv
  → 写出 passivecolorpatch_alignment_audit.csv
```

目前对齐审计显示：test001、002、003、005、006 的 marker 序列完全匹配；test004 和 test007 存在已记录的删除/缺失事件，不能假设 log trial index 与 EEG epoch index 永远相同。后续 Task3 exemplar-wise decoding 必须使用该 trial-level index，而不能仅根据 HDF5 中的条件顺序推断图片身份。

## 3. 任务和触发码：必须以当前注册表为准

唯一主来源是：

```text
color_analyse_0727/pipeline/condition_registry.py
```

不要只根据实验文件名猜任务编号。实验范式的 `config_task2.m` 与 `config_task3.m` 和旧分析命名存在颠倒关系。

### Task 1：真实/灰度物体图片

```text
11 face_color       12 face_gray
21 object_color     22 object_gray
31 body_color       32 body_gray
41 place_color      42 place_gray
```

### Task 2：水果 × 记忆判断

当前HDF5完整保留12个条件，而不是旧README中的8个条件：

```text
cabbage:    101 true, 102 false, 103 gray
kiwi:       111 true, 112 false, 113 gray
strawberry: 121 true, 122 false, 123 gray
watermelon: 131 true, 132 false, 133 gray
```

主分析当前只使用四种gray水果进行记忆颜色分析；true和false必须在后续作为独立条件分析，不能直接与gray混合。

### Task 3：纯颜色色块

```text
51 red, 52 yellow, 53 blue, 54 green, 55 black, 56 white
```

## 4. 当前预处理流程

所有处理都在连续信号上完成，然后再epoch：

```text
.set/.fdt
  → DC偏置中心化
  → 重采样 1000 → 500 Hz
  → 1–200 Hz 零相位带通
  → 50/100/150 Hz 陷波，Q=30
  → 应用人工确认的坏道排除
  → 严格左右相邻 Laplacian：center − (left + right)/2
  → −500…1000 ms epoch
  → 按条件写入HDF5
```

严格相邻的意思是：中心触点必须同时存在左右相邻触点，且三者都没有被排除；端点或缺少邻居的触点不生成分析中心。最终分析使用的是“信号通道与电极定位表交集”中的中心电极。

### 当前人工确认坏道

```text
test003: B15, I1, I2, I3
test004: K3
test005: G1
test007: 18
```

`test007:18` 必须保持数字18，不能改写为I8。`test003:I1` 是用户确认的trigger-like异常通道，不应该用来改变整体坏道策略。

坏道处理原则不是“标准差大就删除”。机器只产生候选表，人工决定写在：

```text
color_analyse_0727/metadata/manual_channel_decisions.csv
color_analyse_0727/qc/bad_channel_candidates.csv
```

只有明确写成`exclude`的通道才会进入HDF5导出器的排除集合；空白默认保留。

## 5. 当前分析实现和结果

当前入口：

```text
color_analyse_0727/analysis/run_final_analysis.py
```

它完成：

1. Norm 1：Task 1 的四类color-vs-gray差异；
2. Norm 2：PC/CC/AC的Talairach坐标用Brett分段仿射逆变换估计到MNI，并做双侧镜像；
3. N1与N2交集的CSC；
4. CSC上的Task 2四种gray水果幅度和多频段频谱比较；
5. CSC上的Task 3 red-vs-green单电极decoding；
6. CSC上的Task 2灰色水果记忆颜色decoding；
7. 结果表、空间图、单电极曲线、报告和PPT。

2026-08-04正式结果：

- 7个被试、21个HDF5全部完成；
- 三任务共同中心共548个；
- Strategy 1为0个；
- Strategy 2为40个；
- CSC为8个，分布在test001、test002、test004、test007；
- 单电极decoding使用100次标签置换，是探索性结果，不是最终发表级统计。

### 2026-08-05 更新：筛选标准与结果

- 功能筛选标准改为 **ANOVA（二因素：颜色 × 类别，类型 II 颜色主效应 p<0.05）**；
  策略二为任一类别 Welch t 检验 p<0.05（两组 ANOVA 等价于 t 检验）。MWU 结果保留为信息列。
- 时间窗标准定为 **1–300 ms**（完整刺激呈现期；与 0–300 ms 等价，Jaccard 0.95）。
- 信号变体改为 `lf30`（1–30 Hz padded 滤波）与 `raw200`（HDF5 1–200 Hz 原样）。
- 新结果目录：`color_analyse_0727/result/final_analysis_anova_20260805/`：
  - 1-300_lf30：S1=19、S2=48、CSC=9；
  - 1-300_raw200：S1=18、S2=50、CSC=8。
- 展示电极：`test004-B10`（temporal_mid，MNI 53.8,-9.7,-22.9），ANOVA 颜色主效应
  p=0.00029（lf30）/0.00020（raw200），交互不显著（效应跨类别一致）；
  结果图见该目录 `stage02_amplitude_spectral/figures/best_electrode_test004_B10.png`。
- nilearn 脑图（S1 电极 + PC/CC/AC 色斑 + N2 邻域电极）：
  `stage01_selection/figures/nilearn_s1_patches_n2_1-300_lf30.png`。
- 运行命令（lr0727 环境）：
  `C:\Users\saber_soul\.conda\envs\lr0727\python.exe -m analysis.run_final_analysis
  --out result/final_analysis_anova_20260805 --windows 1-300 --signals lf30 raw200
  --perms 1000 --workers 21`

### 2026-08-05（晚场）更新：SEEG 假设驱动分析（H1-H4）

- 新增时间分辨解码与簇置换（组水平 5000 次、逐电极 1000 次仅 lf30）、预注册假设 H1-H4、
  探索模块（多电极 MVPA、ERP 幅度、样本重复检查）。历史结果目录：
  `color_analyse_0727/result/final_analysis_seeg_20260805/`。
- H1 ✅（S1=19 vs 期望 10.5）、H2 ✅（物理颜色解码 40-500 ms p=0.008）、
  H3 ❌（记忆颜色单电极组水平 null，cross-fruit 免疫样本泄漏）、H4 ❌（TGM/重叠不显著）。
- 关键 caveat：Task3 每色目前只有 3 个刺激文件，而多数被试每色约有 60 个 epoch；实验代码允许循环重用图片，因此平均约 20 个 epoch/图片，但当前 HDF5 未保留 trial-level 图片身份，精确重复次数和顺序仍待恢复。H2 有样本级混淆风险；
  Task2 每水果 15 图（cross-fruit 已规避）。详见
  `color_analyse_0727/docs/2026-08-05_seeg_hypothesis_handoff.md`（新会话必读）。

### 2026-08-06（晚场）更新：主时间窗改为 100–400 ms

- 经重新考虑，当前主时间窗改为 **100–400 ms**；旧的 1–300 ms 结果只保留在历史记录中，不再作为主分析窗口。
- 正式选电极仍固定为 Task 1 二因素 ANOVA（color × category）的 color 主效应 `p<0.05`；此前的电极成员记录以 `lf30` 为准，但当前 decoding 输入已切换为 `raw200`。
- 修正后的唯一主结果目录：`color_analyse_0727/result/final_analysis_seeg_20260806_corrected/`。
- 新的 `100-400_lf30` 结果：S1=19、S2=46、CSC=12；`100-400_raw200` 为敏感性结果：S1=16、S2=44、CSC=10。
- `100-400_lf30` 的 ANOVA S1 电极为：test001-A6/A7/A8/D6/D8/G5/H9；test003-C13/G10；
  test004-A10/B7/B10/C6/C7/C9；test005-A4；test007-C17/D7/E7。
- 旧 `result` 版本已在本次整理中删除；后续新分析统一写入上述主目录。
- 当前主目录中的时间分辨 S1/S2 单电极曲线仍是描述性结果，`electrode_p_value` 尚未运行；不得把旧版组水平 H2/H3/TGM 数字直接沿用到 100–400 ms。

### 2026-08-06（CSC批量解码，旧 lf30 版本）更新

- 已对 `100-400_lf30` 的全部 12 个 CSC 电极完成优化版 8 类解码，100 次置换；时频曲线和单电极 cluster 检验限制在 0–800 ms，频谱固定窗仍为 1–1000 ms。
- 单电极时频 cluster 使用 max-cluster-mass、形成阈值 p=0.05、最小簇 20 ms、+1 p 修正；合并图不做置换检验，也没有跨电极/跨分析分支的全局校正。
- 结果、NPZ 置换数据、单电极图、合并曲线图和频带主导效应图均在 `color_analyse_0727/result/final_analysis_seeg_20260806_corrected/stage07_csc_decoding/`。
- 特征主导效应图基于标准化 LinearSVC 系数绝对值，只作描述性 feature importance；当前多个分支中 75–85 Hz、85–95 Hz 权重较高，不能直接解释为频带的因果效应。

### 2026-08-06（Norm1 S1/S2 全套单电极，旧 lf30 版本）更新

- 已去掉 Norm2/CSC 空间过滤，直接对当前 Norm1 S1=19、S2=46（重叠13，唯一52）个电极完成8类频谱/时频 decoding、100次置换和单电极 cluster 检验。
- Task2 memory-color cross-fruit 时频在 S1 中有16/19个电极、36个显著簇，在S2中有30/46个电极、70个显著簇；S1/S2平均曲线仍约为chance（0.502/0.499），这些是单电极探索性结果，不是组水平结论。
- 最早显著 cluster latency 与 MNI x/y/z 没有明显描述性相关：S1 ρ=0.05/-0.10/0.11，S2 ρ=-0.03/-0.05/0.14；未做跨电极全局校正。
- 结果在 `color_analyse_0727/result/final_analysis_seeg_20260806_corrected/stage08_s1_s2_single_electrode_decoding/`；主图是 `figures/S1_memory_color_latency_coordinate.png` 和 `figures/S2_memory_color_latency_coordinate.png`。

### 2026-08-06（raw200 解码数据源修正）更新

- 审计发现旧 `lf30` decoding 路径先将输入滤为 1–30 Hz，但频谱/时频特征仍提取完整 5–195 Hz 频带；因此旧 feature-dominance 图中 30 Hz 以上，尤其 75–95 Hz 的高权重，不能解释为真实高频神经活动。
- `stage03_decoding` 已有 raw200 频谱级输出；`stage06_exploration`、`stage07_csc_decoding` 和 `stage08_s1_s2_single_electrode_decoding` 已重新用 raw200 输入完成。stage07/08 为了隔离信号来源变化，保留此前 lf30 选出的电极成员：CSC=12，S1=19，S2=46，重叠=13，唯一=52。
- raw200 stage07 参数记录为 `variant=100-400_raw200`、`selection_variant=100-400_lf30`；stage08 同样记录。时频曲线与单电极 cluster 仍限制在 0–800 ms，频谱固定窗口仍为 1–1000 ms，置换次数为 100。
- raw200 后 CSC Task3 自身频谱平均准确率约 0.792，Task2 cross-fruit 频谱约 0.493；S1/S2 Task2 memory-color 时频平均约 0.502/0.502。raw200 特征权重不再稳定集中于旧图的 75–95 Hz，但部分分支仍出现 65–75 Hz 较高权重，故仍需频带消融和独立验证。
- 旧 lf30 输出文件暂不删除，作为历史对照；它们不再作为当前高频特征解释或主 decoding 结论。

### 2026-08-11（Task1 条件均值时频 RSA）更新

- 新增独立 `stage09_task1_condition_rsa_raw200`，不覆盖 stage01–08；输入为 Task1 HDF5 中已经完成 1–200 Hz 预处理的 raw200 信号。
- 对每个 trial 先在完整 −500–1000 ms epoch 上做 STFT，再对 log-power 相对于 −200–0 ms 做被试内、全部条件/全部 trial 的 baseline z-score；随后在每个条件内平均，并在 0–800 ms 按 50 ms 汇总 16 个既有频带。
- 8 个条件和 RDM 轴顺序固定为四个彩色条件 `face/object/body/place`，再接四个灰色条件；距离为 `1 − Pearson correlation`。本版本只做条件均值层面的描述，不做 bootstrap、显著性检验或跨电极统计。
- 共生成 52 个唯一 subject-channel 电极：S1=19、S2=46、CSC=12；重叠电极只计算一次并在 `electrode_sets_used.csv` 中记录集合归属。共生成 52 张单电极主图和 S1/S2/CSC 三张直接平均距离矩阵图。
- 结果目录：`color_analyse_0727/result/final_analysis_seeg_20260806_corrected/stage09_task1_condition_rsa_raw200/`；完整参数、输入 SHA256、Git commit、环境、时间和 warnings 在根目录 `runs/20260811_174212_stage09_task1_condition_rsa_raw200/`。
- 解释边界：这是“条件平均层面的神经表征几何结构”，不能据此声称已经排除了图片 exemplar、trial 构成或图片身份方差；`Data/` 只用于 trial 身份/trigger 审计，神经特征仍直接来自 HDF5。

### 2026-08-17（Task2 灰色水果与 Task3 纯色红绿 RSA）更新

- 新增 `stage09_1_task2_grayfruit_rsa_raw200`：使用 `strawberry_gray`、`watermelon_gray`、`cabbage_gray`、`kiwi_gray` 四种灰色水果，固定矩阵顺序为 R1、R2、G1、G2。
- 每个通道、每个 50 ms 时间窗计算 4×4 correlation-distance RDM，并输出两条距离曲线：同记忆颜色距离 `(R1R2+G1G2)/2`；异记忆颜色距离 `(R1G1+R1G2+R2G1+R2G2)/4`。
- 新增 `stage09_2_task3_purecolor_rsa_raw200`：使用 Task3 的 red（trigger 51）与 green（trigger 54）纯色色块，输出 red–green correlation distance 的时间变化。
- 两个 stage 均新增逐频段 trial-level decoding：16 个频带 × 16 个 50 ms 时间窗；Task2 使用四个无水果重叠的跨水果方向，Task3 使用 red-vs-green 五折交叉验证。使用 20 个 joblib threads，仅作描述性 balanced accuracy，不做置换显著性。
- 共处理 52 个唯一电极：S1=19、S2=46、CSC=12；每个 stage 生成 52 张通道图和 S1/S2/CSC 集合图。结果目录分别为 `color_analyse_0727/result/final_analysis_seeg_20260806_corrected/stage09_1_task2_grayfruit_rsa_raw200/` 和 `.../stage09_2_task3_purecolor_rsa_raw200/`。
- 最终运行 provenance：`runs/20260817_101517_stage09_1_09_2_rsa_singleband/`。这两个 stage 不覆盖原有 Task1 stage09。

### 2026-08-17（Task2 灰色水果 × Task3 纯色色块跨任务 RSA）更新

- 新增独立 `stage09_3_task2_task3_cross_rsa_raw200`。这里将四种 Task2 灰色水果与 Task3 red/green 纯色色块联合为六个条件 `R1、R2、G1、G2、red、green`，不覆盖 stage09、stage09_1 或 stage09_2。
- 六类 trial 在同一 subject-electrode 内联合后进行 raw200 STFT 与 −200–0 ms baseline z-score；每 50 ms 输出一个 6×6 correlation-distance RDM。
- 主要距离曲线为：同记忆颜色 `(R1-red + R2-red + G1-green + G2-green)/4`，异记忆颜色 `(R1-green + R2-green + G1-red + G2-red)/4`。集合图是单电极 RDM/曲线的直接平均，不含显著性检验。
- 共完成 52 个唯一电极（S1=19、S2=46、CSC=12）。结果目录为 `color_analyse_0727/result/final_analysis_seeg_20260806_corrected/stage09_3_task2_task3_cross_rsa_raw200/`；运行记录为 `runs/20260817_110936_stage09_3_task2_task3_cross_rsa_raw200/`。

### 2026-08-17（stage09 单电极结果显式整理）更新

- stage09、stage09_1、stage09_2 和 stage09_3 均保留 52 张单电极主图，并在原有 `condition_rdm_long.csv`、`condition_mean_tf_features.csv` 等长表中保留 `subject/channel` 级数值。
- 另新增 `single_electrode_index.csv` 和 `single_electrode_rdm_summary.csv`：前者给出每个单电极的集合归属、坐标、图路径；后者按 subject-channel × 50 ms 时间窗保存宽格式完整 RDM，并合并已有的距离曲线指标。
- 这些文件只是对已完成结果的单电极索引/宽格式整理，没有重新计算或改变 RSA 特征、距离定义和集合平均方法。整理运行记录为 `runs/20260817_111728_stage09_single_electrode_outputs/`。

### 2026-08-17（stage09 peak latency 与 MNI 坐标相关）更新

- stage09_1：每个电极取 `abs(between_memory_color_distance − within_memory_color_distance)` 最大的 50 ms 时间窗；stage09_3：取 `abs(different_memory_color_distance − same_memory_color_distance)` 最大的时间窗。CSV 同时保留该时点的有符号差异，便于判断方向。
- stage09_2：每个电极取 Task3 red–green correlation distance 最大的 50 ms 时间窗。
- 三个 stage 分别输出 peak latency 与 MNI x/y/z 的散点回归图、逐电极 peak 表和 Pearson/Spearman 统计 JSON。没有把 peak latency 当作显著性证据，也未做跨坐标多重比较校正。
- stage09_1 和 stage09_3 进一步按 peak 时点的有符号方向拆分电极，且 peak 选择限制在 0–400 ms：stage09_1 为 `between > within` 与 `within > between`；stage09_3 为 `different > same` 与 `same > different`，两组分别保存图、CSV 和统计 JSON。当前分组数量分别为 stage09_1：20/32，stage09_3：26/26。
- 原始 peak 相关图运行记录为 `runs/20260817_114158_stage09_peak_latency_coordinate_correlations/`；本次点大小更新记录为 `runs/20260817_115619_stage09_peak_latency_coordinate_correlations/`。
- 原始 peak latency–MNI x/y/z 图已增加点大小编码：stage09_1/3 的点大小对应 peak 两线绝对差值，stage09_2 的点大小对应 peak red–green distance；颜色仍区分 CSC 与非 CSC 电极。

### 2026-08-17（peak value × peak latency composite 与 MNI Y）更新

- 新增 composite 指标：`early_strength_index = z(peak value) − z(peak latency)`；数值越高表示 peak 更大且出现更早。三个 RSA stage 的 peak 统一限制在 0–400 ms，并在 MNI Y 方向绘制相关图。
- stage09_1 和 stage09_3 仍按两条曲线的方向分面；stage09_2 使用全部 52 个电极作为一个分面。Pearson/Spearman 结果保存在各 stage 的 statistics JSON。
- 当前 composite 与 MNI Y 的相关均不显著：stage09_1 两组 Pearson r=-0.06/-0.03；stage09_2 r=0.13；stage09_3 两组 r=-0.13/0.14。该 composite 是探索性指标，不替代分别检验 peak value 和 peak latency 的多元模型。
- 运行记录为 `runs/20260817_115249_stage09_composite_peak_index_mni_y/`。

### 2026-08-17（stage01 三集合电极重叠汇总图）更新

- 新增 stage01 的整体三集合交叠图：`Norm1 S1 × Norm1 S2 × Norm2`。其中 Norm1 S1 对应 `strategy1`（Task1 二因素 ANOVA 的 color 主效应），Norm1 S2 对应 `strategy2`（至少一个类别 Welch t 检验），Norm2 对应 `N2_union`（PC/CC/AC 空间筛选）；CSC 不作为第三个独立集合，因为 CSC 定义为 `Norm2 ∩ (S1 ∪ S2)`。
- 图中八个互斥区域的数字是全部被试的 subject-channel records 汇总，不是去重后的通道名称。新增文件为 `stage01_selection/figures/norm1s1_norm1s2_norm2_overall_venn_100-400_lf30.png` 和对应的 `..._raw200.png`。
- 该图只改变 stage01 的汇总可视化；原有按被试和 patch 的饼图、筛选表及 ANOVA/N2/CSC 判定均保留不变。

### 2026-08-07（当前分析方法合同）更新

当前所有需要作为主线解释的 decoding 结果使用以下方法合同：

1. **数据层级**：原始实验行为/刺激记录来自 `Data/`；连续 EEG 来自 `seegdata/`；定位和 ROI 来自 `processed_data/`；标准分析输入为 `process_data/` 中的 21 个 HDF5；派生结果写入 `result/final_analysis_seeg_20260806_corrected/`。
2. **预处理**：连续信号 DC 中心化、1000→500 Hz 重采样、1–200 Hz 零相位带通、50/100/150 Hz 陷波、人工坏道排除、严格左右相邻 Laplacian，随后生成 −500…1000 ms epoch。
3. **主时间窗**：Task 1 功能选电极使用 100–400 ms；频谱解码固定提取 1–1000 ms；时频曲线使用 10 ms 步进并正式报告 0–800 ms。
4. **电极选择**：S1 为 Task 1 二因素 ANOVA 的 color 主效应 `p<0.05`；S2 为至少一个类别 Welch t 检验 `p<0.05`。CSC 定义为 `N2 ∩ (S1 ∪ S2)`，不是严格的 `N2 ∩ S1`。
5. **信号与特征**：当前 decoding 使用 `raw200`，即直接使用 HDF5 的 1–200 Hz 信号；16 个 Welch/STFT 频带为 5–15、15–25、25–35、35–45、55–65、65–75、75–85、85–95、105–115、115–125、125–135、135–145、155–165、165–175、175–185、185–195 Hz，去除 50/100/150 Hz 附近线噪声频带。
6. **分类与置换**：使用 StandardScaler + LinearSVC（`dual=False`）；stage07/08 每个电极使用 100 次标签置换。Task2 记忆颜色采用 leave-one-fruit-pair-out：strawberry/watermelon 为 memory-red，cabbage/kiwi 为 memory-green；Task3 为 red-vs-green；另包括 Task2↔Task3 双向 cross-decoding。
7. **时间簇检验**：stage07/08 对单电极时频曲线使用一维 max-cluster-mass，形成阈值 `p<0.05`、最小簇 20 ms、`+1` 置换校正；未做跨电极、跨分析分支的全局 FWER 校正，因此单电极结果只能作为探索性证据。
8. **stage06 与 stage08 的区别**：stage06 只计算 Task3/Task2 两条时间分辨曲线，并对被试/电极集合平均后做 5000 次 sign-flip 组簇；电极曲线本身是描述性结果。stage08 对每个 S1/S2 电极完成完整 8 类频谱/时频 decoding、100 次置换和单电极簇检验。
9. **频带主导效应**：feature dominance 是标准化 LinearSVC 系数绝对值的描述性汇总，不是频带显著性检验，也不能直接解释为因果频带。旧 `lf30` 结果中 30 Hz 以上权重不再作为有效高频证据。
10. **stage09 Task1 RSA**：每 trial 先做完整 epoch STFT 和 −200–0 ms baseline z-score，再在条件内平均；每 50 ms 输出 8×8 correlation-distance RDM。电极集合图是距离矩阵的直接算术平均，不是单电极推断，也不含显著性检验。

结果目录：

```text
color_analyse_0727/result/final_analysis_seeg_20260806_corrected/
├── report/final_analysis_report.md
├── stage01_selection/
├── stage02_amplitude_spectral/
├── stage03_decoding/
├── stage04_luminance/
├── stage05_hypotheses/
├── stage06_exploration/
├── stage07_csc_decoding/
├── stage08_s1_s2_single_electrode_decoding/
├── stage09_task1_condition_rsa_raw200/
├── stage09_1_task2_grayfruit_rsa_raw200/
├── stage09_2_task3_purecolor_rsa_raw200/
└── stage09_3_task2_task3_cross_rsa_raw200/
    
（stage07–09 的参数与运行 provenance 同时保存在根目录 `runs/`。）
```

## 6. 推荐运行顺序

在PowerShell中：

```powershell
cd E:\liulab_project\Project_colorieeg_2026\color_analyse_0727
$PY = 'E:\software\Anaconda\python.exe'

& $PY scripts\validate_conditions.py
& $PY scripts\audit_bad_channels.py
# 人工检查 qc/bad_channel_candidates.csv 和 qc/channel_diagnostics/
& $PY scripts\build_electrode_manifest.py
& $PY scripts\build_hdf5.py
& $PY scripts\validate_hdf5.py

# 当前全管线配置（100–400 ms；selection 保留 lf30/ANOVA，decoding 主输入为 raw200）
& $PY analysis\run_final_analysis.py --windows 100-400 --signals lf30 raw200 --perms 1000 --workers 8
```

如果只是做后续统计，通常不需要重新构建HDF5；直接读取`process_data/`即可。

## 7. 当前notebook架构的注意事项

现在有：

```text
notebooks/00_rawdata_to_seegdata.ipynb
notebooks/01_preprocess_and_export.ipynb
```

这与用户希望“整个预处理只保留一个notebook”还有一个小冲突。后续建议整合为一个：

```text
notebooks/00_preprocessing_pipeline.ipynb
```

其中按章节放置：

1. rawdata→seegdata；
2. 条件和数据完整性检查；
3. 坏道候选表和人工审核入口；
4. 电极定位表与刺激行为记录更新；
5. manifest构建；
6. HDF5导出和验证；
7. 示例波形与均值/SEM图。

具体实现仍应放在`pipeline/`中，notebook只负责展示、调用和小测试，不要把完整算法复制到notebook里。

## 8. 根目录单独文件：当前状态

用户已经清理了根目录中原先的旧README、研究日志、一次性MATLAB脚本和无关表格。本次审计不会恢复这些文件，也没有再执行删除。

当前根目录直接放置、需要特别说明的文件主要是：

| 文件 | 实际用途 | 是否是0727当前运行依赖 | 建议 |
|---|---|---:|---|
| `PROJECT_HANDOFF.md` | 当前项目交接入口 | 否 | 保留；新Coding Agent先读此文件 |
| `ccep.zip` | CCEP代码/数据归档 | 否，属于CCEP | 不要放入0727；确认`ccepcode_standalone`完整后再移到外部归档 |

触发码表已经位于当前工作区：

```text
color_analyse_0727/metadata/trigger.xlsx
```

它是采集记录参考；程序运行时的唯一条件来源仍然是`color_analyse_0727/pipeline/condition_registry.py`。

根目录中已经被清理的旧文件包括旧`README.md`、`main_plan.md`、`2026-07-22.md`、旧版raw导入/epoch脚本、旧high-gamma启动脚本和无关的行政统计表。它们不再是新Agent的阅读入口。

### 根目录文件的整合原则

不要把所有文件物理移动到`color_analyse_0727/`。正确的整合方式是：

- 算法整合：把可复用代码放入`0727/pipeline/`；
- 命令整合：把批处理入口放入`0727/scripts/`；
- 条件/路径/坏道记录整合：放入`0727/metadata/`；当前触发码表已经完成这一步；
- 历史代码：放入单独的`legacy/`或保留原目录，不参与当前import；
- 原始数据：继续放在根目录，或未来用配置/环境变量指定外部数据根目录，不复制出多份35 GB原始数据。

## 9. 根目录主要文件夹：用途和处理建议

| 文件夹 | 用途 | 与0727关系 | 建议 |
|---|---|---|---|
| `rawdata/` | 原始BDF和采集session | 原始来源 | 必须保留，禁止修改；可迁移到外部数据盘 |
| `seegdata/` | 从BDF导入的EEGLAB `.set/.fdt` | 0727当前直接输入 | 必须保留，除非修改`pipeline/config.py`并完成迁移验证 |
| `processed_data/` | 电极定位表、旧MAT和旧处理产物 | 0727当前读取定位表 | 保留定位表；旧MAT可在确认无旧分析需求后归档 |
| `color_analyse_0727/` | 新的预处理和当前分析主线 | 当前主工作区 | 保留并继续扩展 |
| `newanalyse/` | 旧MAT/ROI feature/decoding主线 | 不被0727读取 | 保留为legacy，不能和0727的HDF5结果混合 |
| `color_cognition_pipeline/` | 旧颜色分析代码和大量历史结果 | 方法参考 | 保留只读或整体归档；不要直接调用旧result作为当前输入 |
| `visual_experiment/` | Psychtoolbox实验、刺激、触发和刺激行为记录 | 条件定义和刺激记录参考 | 必须保留；不要并入预处理代码 |
| `testcode/` | 旧解码、统计和审计实验脚本 | 方法参考 | 保留；每个脚本运行前先核对输入路径和任务编号 |
| `ccepcode_standalone/` | CCEP独立项目，含数据和workspace | 与被动颜色分析分开 | 保留独立；不要合并到0727 |
| `liulab_cceptoolbox/` | CCEP工具箱/依赖 | CCEP依赖 | 只有放弃CCEP时才考虑删除 |
| `mne_data/`、`.mne/` | MNE模板/缓存 | 主要服务旧玻璃脑图 | 当前0727不依赖；旧空间图不再需要时可归档 |
| `MathWorks/` | MATLAB本地缓存，当前只有`graphicsState.bin` | 非项目源码 | 可删除，通常会自动重建 |
| `赵明慧毕业论文笔记_images/` | 论文笔记和图片 | 科研参考 | 保留或移到个人文献资料目录 |
| `.git/`、`.github/`、`.agents/` | 版本控制、协作配置、Agent技能 | 工程基础设施 | 保留 |
| `.vscode/` | 本机编辑器设置 | 非运行依赖 | 可保留或删除，已被gitignore |

## 10. 不应该混淆的三套代码

### A. 当前颜色分析主线

```text
color_analyse_0727/pipeline + scripts + notebooks + analysis
```

输入当前`.set/.fdt`和定位表，输出HDF5、QC、CSC分析和报告。

### B. 旧颜色分析主线

```text
newanalyse/
color_cognition_pipeline/
testcode/
```

这些代码使用旧的MAT/feature/result数据合同，包含许多有价值的历史方法，但不能因为文件名相似就和当前HDF5直接混用。

### C. CCEP分析主线

```text
ccepcode_standalone/
liulab_cceptoolbox/
ccep.zip
```

这是电刺激诱发电位项目，不是当前被动颜色分析。它可以共享电极定位思想，但不应成为0727的输入或预处理步骤。

## 11. 当前已知问题和后续工程任务

1. `color_analyse_0727/notebooks/`仍有两个notebook，需要按用户偏好整合为一个预处理notebook。
2. `color_analyse_0727/pipeline/config.py`仍默认从根目录读取`seegdata`和`processed_data`；若要让0727完全自包含，需要迁移路径并做一次全量验证。
3. `processed_data`中的定位表是当前0727不可缺少的元数据，应先复制/登记，再考虑移动。
4. 当前Norm 2使用论文坐标的群体级Talairach→MNI估计和20 mm距离阈值，不是被试个体化fMRI Color Patch。
5. 当前Norm 1使用raw p阈值，FDR只是敏感性分析；Strategy 1为0不代表没有颜色相关信号，而是四类同时显著的条件太严格。
6. 当前频谱 decoding 使用1000次置换；CSC 已完成100次置换的单电极时间分辨 cluster 检验，但 S1/S2 仍未完成同规格单电极 permutation；所有100次结果仍应视为筛查/探索性证据。
7. Task 2 true/false尚未加入当前主CSC分析；后续必须单独建分析分支。
8. 当前主线在100–400 ms使用Task 1 ANOVA选出的S1电极；全部548个共同中心的单电极decoding不作为主结果。
9. 当前工作区和旧代码都存在未提交/未跟踪文件，任何删除或重命名前应先做备份并确认Git状态。

## 12. 给新Coding Agent的最短指令

如果一个新Agent只需要快速开始，应按以下顺序阅读：

1. `PROJECT_HANDOFF.md`；
2. `color_analyse_0727/README.md`；
3. `color_analyse_0727/pipeline/condition_registry.py`；
4. `color_analyse_0727/pipeline/config.py`；
5. `color_analyse_0727/pipeline/preprocess.py`；
6. `color_analyse_0727/analysis/run_final_analysis.py`；
7. `color_analyse_0727/prompt/user_prompt_0727.md` 和 `user_prompt_0803.md`；
8. `color_analyse_0727/result/final_analysis_seeg_20260806_corrected/README.md`。

开始任何新分析前，必须确认：

- 使用的是`color_analyse_0727/process_data/`中的HDF5；
- 条件名称来自`condition_registry.py`；
- 没有把Task 2的gray、true、false混为一个条件；
- 没有把旧`newanalyse`或`color_cognition_pipeline`的结果数字当作当前结果；
- 没有修改`rawdata/`；
- 如果要改变坏道，先修改人工审核记录，再重建HDF5并验证。
