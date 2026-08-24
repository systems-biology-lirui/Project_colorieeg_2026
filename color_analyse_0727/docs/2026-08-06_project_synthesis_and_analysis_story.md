# 项目综合结论与完整分析路线

**日期：2026-08-06；记录更新：2026-08-07（新增 Data 原始记录层与当前方法合同）**  
**主工作区：`color_analyse_0727/`**

本文件把当前项目的证据、已经完成的分析、尚未完成的分析和最终应当形成的科学故事放在同一条可复现的链条中。这里的“思考过程”采用可审计的决策记录：每个判断都对应现有结果文件、代码或数据合同；不把尚未运行的计划写成结果。

## 一句话结论

当前数据最稳妥地支持以下结论：

> 腹侧视觉通路的部分电极确实携带物理颜色信息；Task 1 的颜色选择性电极在空间上高于随机预期。100–400 ms 修正后的 S1/S2 范围分析在 Task 3 红/绿上出现探索性时间簇，但单电极显著性和 exemplar-safe 检验尚未完成。对于灰色水果所诱发的记忆颜色，目前没有足够强的证据证明稳定的红/绿抽象表征；Task 3 物理颜色到 Task 2 记忆颜色的跨阶段泛化仍未建立。因此，现阶段的科学故事应写成“物理颜色编码有初步证据，记忆颜色和跨阶段共享代码仍待验证”，而不是“记忆颜色已经被证明”或“记忆颜色不存在”。

> **当前分析口径**：主时间窗为 100–400 ms；正式电极选择固定为 Task 1 二因素 ANOVA 的 color 主效应 `p<0.05`。当前 decoding 主信号改为 `raw200`；stage07/08 为了直接比较信号来源，保留此前 lf30 选择出的 CSC、S1、S2 电极成员。旧 1–300 ms 和旧 lf30 decoding 数字不再作为当前频段主结论。

这不是项目失败，而是把问题拆成了三个层次：

```text
物理颜色是否能被编码？          —— 当前数据支持
灰色物体的记忆颜色是否能被编码？—— 当前数据尚未支持稳定结论
两种颜色表征是否共享神经代码？  —— 当前 TGM/RSA 证据尚未建立
```

## 1. 数据、预处理和当前分析合同

当前主线只使用 `color_analyse_0727/process_data/` 中的 21 个 HDF5，不把旧的 `newanalyse/`、`color_cognition_pipeline/` 结果数字直接混入当前结论。

- 7 位被试，Task 1/2/3 共 21 个 HDF5；
- 三任务共同的严格相邻 Laplacian 中心电极共 548 个；
- 腹侧视觉 ROI 主要为 fusiform、temporal_inf、temporal_mid，共 210 个中心；
- 当前正式时间窗为 100–400 ms；
- 当前信号变体为 `lf30`（1–30 Hz）和 `raw200`（HDF5 中的 1–200 Hz）；
- 频谱特征为 16 个 Welch log-power 频带，按 −200–0 ms baseline 做 trial 内标准化；
- Task 2 的 `gray`、`true`、`false` 条件在 HDF5 中分开保存，不能混合解释。

### 新增原始数据层：`color_analyse_0727/Data/`

`Data/` 保存每个被试实验端的 MATLAB 记录，与连续 EEG 和 HDF5 的职责不同：

| 数据层 | 内容 | 当前用途 |
|---|---|---|
| `Data/testXXX/` | `cfg`、`results`、`stimData`、session history、图片文件名、marker、呈现时间、catch/response | 恢复 trial-level 刺激身份和行为信息 |
| 根目录 `seegdata/` | EEGLAB `.set/.fdt` 连续 EEG 和 event stream | 连续信号预处理与事件对齐 |
| `processed_data/` | 电极定位、MNI/Talairach 坐标、ROI | 电极定位和空间筛选 |
| `process_data/` | 21 个标准 HDF5 epoch | 当前 decoding 的标准输入 |
| `metadata/` | manifest、坏道决定、对齐审计、trial index | 可审计的中间数据合同 |

新增 `Data/` 共包含 7 个被试、96 个文件。被动任务 `.mat` 的 `results` 保存 `imgName`、`marker`、`onsetTime`、`offsetTime`、`duration`、`response` 和 `isCatch`；`stimData` 保存实际使用的 `filename`、类别和 marker。因此它可以解决之前“Task3 每种颜色只有 3 张图片、但 HDF5 只保留 condition-level epoch”的关键限制。

`scripts/normalize_task_labels_and_index_passivecolorpatch.py` 已将 `Data` 中 Task3 日志与 `seegdata/testX/erp3.set` 的 event marker 对齐，生成：

- `metadata/passivecolorpatch_shape_trial_index.csv`：每个 log trial 对应的 EEG event/epoch、颜色、shape、图片文件名和呈现时间；
- `metadata/passivecolorpatch_alignment_audit.csv`：每个被试的序列匹配、删除事件和缺失事件审计；
- `metadata/task_label_rename_manifest.csv`：前四位被试 Task2/Task3 文件名颠倒的修正记录。

目前 test001、002、003、005、006 的 marker 序列完全匹配；test004、test007 有已记录的删除/缺失事件。后续 Task3 主解码必须基于该 trial-level index 做 exemplar-wise cross-validation；在此之前，Task3 解码只能表述为“颜色类别可解码”，不能表述为已经排除了具体图片身份。

## 当前分析方法合同

1. 预处理：连续信号 DC 中心化 → 1000→500 Hz 重采样 → 1–200 Hz 零相位带通 → 50/100/150 Hz 陷波 → 人工坏道排除 → 严格左右相邻 Laplacian → −500…1000 ms epoch。
2. 选电极：100–400 ms Task 1 二因素 ANOVA color 主效应 `p<0.05` 定义 S1；至少一个类别 Welch t 检验 `p<0.05` 定义 S2；CSC=`N2∩(S1∪S2)`。
3. 解码信号：当前主 decoding 使用 raw200，即 HDF5 中的 1–200 Hz 信号；旧 lf30 结果保留为历史对照，不能把其中 30 Hz 以上的权重解释为真实高频信息。
4. 特征：16 个 Welch/STFT log-power 频带，去除 45–55、95–105、145–155 Hz 线噪声频带；频谱固定窗口 1–1000 ms，时频曲线 10 ms 步进并报告 0–800 ms。
5. 解码任务：Task3 red-vs-green；Task2 灰色水果 leave-one-fruit-pair-out memory-red/memory-green；以及 Task3→Task2、Task2→Task3 双向 cross-decoding。分类器为 StandardScaler + LinearSVC(`dual=False`)。
6. 统计：stage06 为集合/被试平均后的 5000 次 sign-flip 组簇；stage07/08 为每电极 100 次标签置换的 max-cluster-mass，形成阈值 `p<0.05`、最小簇 20 ms、`+1` 校正；未做跨电极全局校正。
7. 频带主导图只展示标准化 SVM 绝对权重，不能视为 feature-level 显著性或因果频带证据。

### 2026-08-06 信号来源修正

旧版 `lf30` 路径先把输入滤为 1–30 Hz，但频谱/时频特征函数仍提取完整的 5–195 Hz 频带。因此旧 feature-dominance 图中 30 Hz 以上的权重不能解释为真实高频活动；75–95 Hz 的集中模式可能是截止频率以上残留/数值噪声经过标准化后的分类权重。现在 stage06、stage07、stage08 的 decoding 输入统一改为 `raw200`，即直接使用 HDF5 中已保留的 1–200 Hz 信号。旧 lf30 文件暂时保留作历史对照，不再用于当前频率解释。

预处理顺序是：连续信号 DC 中心化 → 500 Hz 重采样 → 1–200 Hz 零相位带通 → 50/100/150 Hz 陷波 → 人工确认坏道排除 → 严格左右相邻 Laplacian → −500…1000 ms epoch → 按条件写入 HDF5。

主要依据：

- [PROJECT_HANDOFF.md](../../PROJECT_HANDOFF.md)
- [condition_registry.py](../pipeline/condition_registry.py)
- [preprocess.py](../pipeline/preprocess.py)
- [analysis_parameters.json](../result/final_analysis_seeg_20260806_corrected/analysis_parameters.json)
- [hdf5_validation_report.csv](../metadata/hdf5_validation_report.csv)

## 2. 当前电极筛选的证据

### 2.1 Norm 1 的 S1/S2

当前功能筛选使用 Task 1 的 color-vs-gray 差异：

- **S1**：二因素 ANOVA（颜色 × 类别）中的颜色主效应 p<0.05；
- **S2**：至少一个类别的 Welch t 检验 p<0.05；
- 当前为 raw p 探索性筛选，未把跨电极 FDR 作为硬门槛。

在 100–400 ms、`lf30` 变体中：

| 被试 | 共同中心 | S1 | S2 | CSC |
|---|---:|---:|---:|---:|
| test001 | 74 | 7 | 15 | 4 |
| test002 | 58 | 0 | 5 | 2 |
| test003 | 109 | 2 | 4 | 0 |
| test004 | 46 | 6 | 10 | 2 |
| test005 | 75 | 1 | 1 | 0 |
| test006 | 88 | 0 | 2 | 0 |
| test007 | 98 | 3 | 9 | 4 |
| **合计** | **548** | **19** | **46** | **12** |

在 `raw200` 中为 S1=16、S2=44、CSC=10。S1 的总数在 `lf30` 中仍高于完全随机期望 10.5：

- `lf30`：19，二项 p=0.0096；
- `raw200`：16，二项 p=0.0633，作为敏感性结果未达到 0.05。

这支持“颜色选择性电极在腹侧视觉 ROI 中有一定富集”，但不等于每一个 S1/S2 电极都是真正的颜色专一电极；因为筛选阈值是 raw p，CSC 应标记为探索性空间集合。

### 2.2 Norm 2 和 CSC

Norm 2 使用论文中 PC/CC/AC 色斑的群体 Talairach→MNI 估计，并以双侧色斑 20 mm 距离作为空间先验。它不是被试个体化 fMRI 色斑。

因此 CSC 的含义是：

```text
CSC = (S1 ∪ S2) ∩ N2
```

它适合做腹侧视觉颜色位点的探索性重点分析，但不应被描述成严格独立验证的颜色 patch 电极。

## 3. 已经得到的结果及其强度

### H1：S1 是否富集？—— 支持

S1=19（`lf30`）或 16（`raw200`）。`lf30` 高于随机期望，二项 p=0.0096；`raw200` 的二项 p=0.0633，作为敏感性结果未达 0.05。主结论使用 `lf30`。

### H2：物理颜色是否能解码？—— 100–400 ms 修正结果作为探索性支持，Task3 刺激身份仍需补强

100–400 ms 修正后的 S1/S2 范围分析显示：

- S1 Task3：150–230 ms、350–470 ms 的被试层面探索性簇 p=0.034；
- S2 Task3：80–130 ms、160–250 ms 的探索性簇 p=0.006，430–460 ms p=0.043；
- Task2 memory-color 在 S1、S2、S1∪S2 和 CSC 中均未形成探索性组簇；
- 这些不是当前主结论，也不能替代单电极 permutation p；单电极显著性尚未运行。

需要区分两个版本：2026-08-05 的 `p=0.0078` 来自旧版时间分辨运行；2026-08-06 审计发现部分解码路径没有真正执行 `lf30` 滤波，现已在 `decoding.py`、`hypotheses.py` 和 `exploration.py` 中修正 `prepare_signal` 调用。旧目录应视为历史结果，修正后的主结论以新目录和 S1/S2 范围分析为准。

因此主文可以说“存在物理颜色类别信息”，但在恢复图片身份前，不能把它写成完全排除了具体图片/低层视觉样本差异的抽象颜色代码。

修正后的 `raw200` S1/S2 单电极时间分辨范围分析（100–400 ms）提供了一个中间结果：

- S1 被试平均曲线在 70–200 ms 形成探索性组簇，p=0.0284；
- S2 被试平均曲线在 50–320 ms、350–440 ms 和 460–500 ms 形成探索性组簇；
- S1∪S2 在 50–320 ms、340–440 ms 和 460–500 ms 等区间出现探索性组簇；
- Task2 跨水果记忆颜色在 S1、S2、S1∪S2 和 CSC 组曲线中均未形成显著簇。

这些范围分析是按集合分别做的探索性 sign-flip 检验，尚未对四个电极集合和多个物理颜色簇做全局 family-wise 校正；单电极曲线只保存峰值和时间，尚未赋予单电极显著性。因此它们强化了“物理颜色在 S1/S2 可见、记忆颜色目前无组簇”的方向，但不能替代预注册的全 ROI 高置换重跑。

### H3：记忆颜色是否能解码？—— 当前未支持

Task 2 使用四种灰色水果，把草莓/西瓜归为记忆红色，把卷心菜/猕猴桃归为记忆绿色，并采用 leave-one-fruit-pair-out，尽量避免训练和测试出现相同水果。

当前结果：

- 组水平时间簇为空；
- 当前 raw200 CSC 频谱级跨水果平均准确率约 0.493，时频约 0.507；
- 当前 raw200 S1/S2 的 Task2 memory-color 时频平均准确率约为 S1=0.502、S2=0.502；
- 当前时间分辨单电极 p 值尚未运行，不能把峰值曲线标为显著电极；
- 旧 1–300 ms 的 test007-D8 和 ERP 170–210 ms 数字不再作为当前主结果。

所以当前正确表述是：

> 记忆颜色的跨水果线性解码在现有 CSC、现有频谱特征和现有样本量下没有获得组水平显著证据；ERP 170–210 ms 的接近显著结果值得作为后续特征分支检验，但不能倒推为已证实的记忆颜色编码。

### H4：物理颜色和记忆颜色是否共享代码？—— 当前未支持

Task 3→Task 2 的跨阶段 TGM 没有显著泛化簇；位点重叠为 `lf30` 的 1/9，超几何 p=1.0，`raw200` 为 0/8。

这个负结果只说明在当前 CSC 空间集合、16 个频谱特征、当前时间分辨率、当前 Task3/Task2 条件设计、当前样本量和分类器下，没有发现可重复的跨阶段共享代码。它不能推出“物理颜色和记忆颜色一定使用完全不同的神经机制”。

## 4. Task3 图片重复问题：应如何准确写入结论

这里需要把“已证实”和“推断”分开。

### 已证实

`visual_experiment/stimuli_pic/Stimuli_Task3/` 中每种颜色目前有 3 个 bmp 文件，例如 `Red_Color_01.bmp`、`02.bmp`、`03.bmp`。这是直接的文件清单证据。

当前 HDF5 的 Task 3 条件通常约有 60 个 epoch；test003 为 90 个，test004/test007 个别条件为 59 个。因此多数被试的数量关系是：

```text
60 个条件 epoch / 3 个图片文件 = 平均 20 个 epoch / 图片
```

### 支持“会重复使用图片”的代码证据

在 [run_passive_phase.m](../../visual_experiment/Utils/run_passive_phase.m:46) 中，图片按类别读入；第 76–79 行通过 `mod(currIdx,nTotal)+1` 循环取图。当所需试次数超过图片数量时，会循环使用已有图片。`config_task3.m` 将每类的 `countPerCategory` 设为 30，因此单次运行中 3 张图会被循环使用。

### 尚未证实

当前 HDF5 只保存 condition-level epoch，没有保存每个 epoch 对应的 `imgName`。因此还不能直接证明每一张图片的精确重复次数、60 个 epoch 是否来自两轮相同实验运行、以及图片在 EEG epoch 中的实际呈现顺序。

因此项目文档应使用下面这句话：

> Task3 每种颜色只有 3 个刺激文件，而多数被试每种颜色约有 60 个 EEG epoch，实验代码也允许循环重用图片；这提示平均每张图片约对应 20 个 epoch，但由于当前 HDF5 未保留 trial-level 图片身份，精确重复次数和顺序仍待从实验保存的 `results.imgName` 或原始行为日志恢复。样本级混淆目前是风险提示，不是已经完全验证的结论。

这会影响 Task3 within-task decoding 的解释，但不应把当前 H2 直接判为无效。最稳妥的处理是：恢复图片身份后做 exemplar-wise cross-validation；若无法恢复，则把 H2 限定为“颜色类别可解码”，不称为已排除图片样本差异的抽象颜色代码。

## 5. 当前工作完成度审计

| 分析模块 | 当前状态 | 可否作为主结论 |
|---|---|---|
| 统一条件注册表、预处理、HDF5 QC | 已完成 | 可以 |
| Task1 S1/S2 功能筛选 | 已完成 | 可以，但注明 raw-p 探索性 |
| Norm2 色斑邻近和 CSC | 已完成 | 可以作为空间先验，不是个体 fMRI 验证 |
| CSC Task2 幅度/频谱统计 | 已完成 | 可作描述性和辅助结果 |
| CSC 频谱级 Task3 物理颜色解码 | 已完成 | 可以，需注明图片身份风险 |
| CSC 频谱级 Task2 跨水果记忆颜色解码 | 已完成 | 负结果可报告 |
| 时间分辨 Task3 S1/S2 探索 | 已完成 100–400 ms raw200 第一轮 | 只能作中间结果 |
| 时间分辨 Task2 S1/S2 探索 | 已完成第一轮；当前未形成组簇 | 只能作中间结果 |
| Task3→Task2 TGM | 当前主窗尚未重跑 | 不能沿用旧 1–300 ms 数字 |
| 全部 S1/S2 的单电极时间分辨 decoding | 已完成 100–400 ms raw200 描述性曲线和单电极 cluster；未做跨电极全局校正 | 只能作中间结果 |
| ERP、低频、high-gamma 多特征平行比较 | 当前仅有部分探索 | 需要统一补齐 |
| Task1 条件均值时频 RSA | 已完成 stage09 raw200 描述性版本 | 可以作为条件平均几何结构；不等同于 trial/exemplar 控制或跨任务共享表征 |
| Task2 true/false 独立分析 | HDF5 已保存，主线尚未完成 | 不能与记忆颜色混合 |
| 电刺激颜色因果分析 | 资料已登记，尚未接入主结论 | 单独作为因果扩展 |
| Task3 trial→图片身份恢复 | 尚未完成 | 是下一步质量门槛 |

截至 2026-08-06，100–400 ms 的 S1/S2 范围分析已经完成第一轮 `raw200` 描述性曲线和被试层面组簇；单电极结果仍未进行跨电极全局多重比较，因此只能作为探索性中间结果。

## 6. 建议的完整科学故事

### 第一层：建立物理颜色基线

使用 Task 1 的 S1/S2 作为功能位点，Task 3 作为物理颜色基线。

1. 在 S1、S2、S1∪S2、CSC 四套电极集合中分别做红/绿解码；
2. 同时分析 ERP、1–30 Hz、70–150 Hz/high-gamma、16-band spectrum；
3. 以时间分辨 accuracy/AUC 曲线和 cluster permutation 为主统计；
4. 用恢复后的图片身份做 exemplar-wise CV；
5. 结果重点是颜色信息的出现时间、持续时间、ROI/电极分布和特征依赖性。

这一层的当前结论是：100–400 ms raw200 的 S1/S2 时间曲线出现物理颜色探索性簇，但单电极显著性、全局校正和图片身份控制仍需完成。

### 第二层：检验灰色物体中的记忆颜色

只使用 Task 2 的 gray 条件，不把 true/false 混入颜色分析。

红/绿标签为：

```text
memory-red   = strawberry_gray + watermelon_gray
memory-green = cabbage_gray + kiwi_gray
```

主分析必须继续使用 leave-one-fruit-pair-out：训练和测试水果完全分离。建议在 S1、S2 和 CSC 上分别做单电极 ERP amplitude decoding、单电极频带功率 decoding、多电极 MVPA、within-task temporal generalization 和 RSA/模型比较。

如果单电极不显著但多电极或 RSA 显著，应解释为“分布式/几何表征”，不能称为单一电极上的颜色代码。反过来，如果只在单个电极 nominal 显著而组水平不显著，应保留为个体探索性发现。

### 第三层：检验物理颜色与记忆颜色是否共享代码

这层应有两种互补检验：

1. **跨阶段 decoding/TGM**：用 Task 3 的红/绿模式训练，在 Task 2 gray 水果测试；S1、S2、CSC 和多电极 ROI 分别运行；
2. **跨阶段 RSA**：比较 Task 3 的物理颜色 RDM 与 Task 2 的记忆颜色 RDM，控制水果身份、图片身份和低层视觉差异。

只有当跨阶段泛化或 RSA 在独立校正下显著，才能说两种条件存在共享表征；仅仅“两个任务分别可解码”不足以证明共享代码。

## 7. RSA 的具体设计

当前已实现第一步：Task1 八条件的条件均值时频 RSA，结果位于 `result/final_analysis_seeg_20260806_corrected/stage09_task1_condition_rsa_raw200/`。这一步是描述性几何表征，不包含 bootstrap、显著性检验、trial-wise cross-validation 或跨任务泛化；后续 Task2/Task3 trial-wise、exemplar-controlled 和跨任务 RSA 仍属于后续工作。

### 已完成：Task1 条件均值 RSA（stage09）

- 每个 trial 在完整 −500–1000 ms epoch 上做 raw200 STFT；log-power 相对于 −200–0 ms 做被试内全部条件/全部 trial 的 baseline z-score；之后在条件内平均。
- 在 0–800 ms 以连续 50 ms 时间窗汇总 16 个既有频带；每个唯一 subject-channel 电极输出 16 个 8×8 RDM。
- RDM 条件顺序固定为四个彩色条件 + 四个灰色条件，距离为 `1 − Pearson correlation`。
- S1/S2/CSC 集合图对电极距离矩阵直接算术平均，不做 Fisher z，不进行跨电极统计；因此集合图不能替代单电极结果。
- 由于是条件均值，仍可能受到图片 exemplar、trial 构成和图片身份差异影响。`Data/` 仅用于 trial 身份与 trigger 审计，神经特征直接来自 Task1 HDF5。

### 新增：Task2 灰色水果与 Task3 纯色红绿 RSA

- `stage09_1_task2_grayfruit_rsa_raw200/` 使用四种灰色水果，RDM 顺序为 R1=strawberry、R2=watermelon、G1=cabbage、G2=kiwi。每个通道和 50 ms 时间窗输出 4×4 RDM，并计算同记忆颜色距离 `(R1R2+G1G2)/2` 与异记忆颜色距离 `(R1G1+R1G2+R2G1+R2G2)/4`。
- `stage09_2_task3_purecolor_rsa_raw200/` 使用 Task3 red 和 green 纯色色块，输出 red–green correlation distance 的 0–800 ms 变化。
- `stage09_3_task2_task3_cross_rsa_raw200/` 将四种灰色水果和 Task3 red/green 纯色色块放入同一六条件表征空间，计算灰色水果到同记忆颜色纯色块、以及到异记忆颜色纯色块的距离。
- 两个 stage 同时输出逐频段 trial-level decoding：16 个频带×16 个时间窗。Task2 使用四个无水果重叠的跨水果方向，Task3 使用 red-vs-green 五折交叉验证；使用 20 个线程，仅作描述性 balanced accuracy，不做置换检验。
- 每个通道图包含条件均值曲线、16 个 RDM 热图、距离曲线和单频段 decoding 热图；完整 RDM 数值保存在 `condition_rdm_long.csv`，没有为每个频段单独生成额外图片。
- 这两个结果可以检验“四种灰色水果是否出现共同的记忆颜色几何结构”，但不能单独证明已经排除水果语义或图片身份差异。
- 当前四个 RSA 分支均已补充 `single_electrode_index.csv` 和 `single_electrode_rdm_summary.csv`，因此可直接在 subject-channel × 时间窗层面读取单电极 RDM、距离指标和对应图形。
- stage09_1/3 进一步保存每个电极两条距离曲线绝对差异最大的 latency，并与 MNI x/y/z 做 Pearson/Spearman 描述性相关；stage09_2 保存 red–green 距离 peak latency 的同类坐标图。peak 选择不等于显著性检验，且未做跨坐标多重比较校正。
- stage09_1/3 的 peak 电极还在 0–400 ms 内按 peak 时点的方向拆分为第二条线更大和第一条线更大两组，分别绘制坐标相关图；这只是条件化描述，不能当作独立的预先定义电极选择。
- 进一步定义 `early_strength_index = z(peak value) − z(peak latency)`，统一在 0–400 ms 取 peak 后与 MNI Y 做相关；三个 stage 当前均未显示明确的 composite–Y 相关，因此不能据此支持空间传播趋势。
- 原始 peak latency–坐标图现在用点大小编码 peak 幅值，保留了 peak latency 与 MNI 坐标的原始关系，同时可观察 peak 强弱的分布。

### 神经 RDM

在每个时间点或短时间窗构造 trial-by-trial 神经模式：

- 单电极：频带向量，或 ERP + 频带拼接向量；
- 多电极：固定电极顺序的电极 × 特征向量；
- 距离优先使用 crossnobis 或严格 split-half cross-validated distance；
- 不直接在同一数据上选择最佳时间点再检验同一点。

### 模型 RDM

至少比较 physical-color、memory-color、fruit-identity、true/false、image-identity/低层视觉、luminance/颜色空间距离模型。

Task 2 gray 的关键检验是：memory-color model 是否解释神经 RDM，并在控制 fruit identity 后仍保留解释力。Task 3→Task2 的关键检验是：两任务的神经 RDM 是否在相同时间段和相同电极集合中相关，而不是只看两个任务的 decoding accuracy。

### 统计

- 先在每个被试得到时间曲线；
- 再做 subject-level group statistic；
- 时间维度用 cluster permutation；
- RSA 模型之间用 partial correlation 或 cross-validated model comparison；
- 单电极 RSA 为探索性，多电极 RSA 为主要验证；
- 对 feature branch、electrode set 和 model family 预先指定主检验，其他作为敏感性分析。

## 8. 推荐执行优先级

### P0：质量门槛

1. 从实验保存的 MATLAB `results.imgName` 或原始日志恢复 Task3 trial→图片映射；
2. 若恢复失败，明确把 H2 改写为颜色类别解码，并保留风险注释；
3. 修正自动审计脚本，让它读取真实 HDF5 trial counts，不再把所有被试都硬编码为 60。

### P1：补齐用户最关心的 S1/S2 单电极分析

在 S1、S2、S1∪S2、CSC 四套电极中，对每个电极分别输出 Task3 red/green 和 Task2 gray memory-red/memory-green 的 ERP、lf30、high-gamma、频谱特征时间曲线，以及 onset、peak、cluster mass、permutation p、FDR p、ROI、MNI 和 S1/S2/CSC 标签。

### P2：记忆颜色特征扩展

优先检验 ERP 170–210 ms 线索，并加入 70–150 Hz/high-gamma、多电极 MVPA 和 within-task TGM。所有模型保持 cross-fruit 训练/测试隔离。

### P3：RSA 和跨阶段泛化

先做 Task2 内部 memory-color RSA，再做 Task3→Task2 cross-task RSA/TGM。只有 P2 有稳定记忆颜色信号后，P3 的共享代码问题才有足够解释基础。

### P4：true/false 与刺激因果扩展

Task2 true/false 是独立的语义/知识判断问题，不应与 gray memory-color 结果合并。`color_with_sti` 是因果扩展，应在被动任务主线稳定后单独比较。

## 9. 最终可写入论文的当前结论

### 可以现在写

1. 预处理和 HDF5 数据合同已经统一，并通过 7 位被试 × 3 任务的验证；
2. 腹侧视觉 ROI 内的 S1 电极数量高于随机期望；
3. 100–400 ms raw200 的 S1/S2 范围分析出现 Task3 物理红/绿探索性时间簇；
4. 当前 CSC 上没有稳定的 Task2 跨水果记忆颜色组水平解码；
5. 当前没有显著的 Task3→Task2 时间泛化或显著位点重叠。

### 2026-08-06 CSC 单电极批量结果补充

已对此前筛选的 12 个 CSC 电极用 raw200 完成 8 类频谱/时频解码。时频曲线与单电极 cluster 检验限制在 0–800 ms；100 次置换下，Task2 cross-fruit 时频记录 18 个、Task3 within 时频记录 17 个 `p≤0.05` 簇，仍未做跨电极、跨分析分支的全局多重比较校正。raw200 后 feature dominance 不再稳定集中于旧 lf30 图中的 75–95 Hz，但部分分支仍出现 65–75 Hz 权重较高；因此需要频带消融和独立验证，不能把权重图当作频带因果证据。

### 现在不能写成确定结论

1. “Task2 记忆颜色不存在”；
2. “Task3 的解码完全排除了具体图片身份差异”；
3. “物理颜色和记忆颜色使用完全不同的脑机制”；
4. “某个 nominal p<0.05 的单电极就是稳定的记忆颜色电极”；
5. “已有 RSA 证明了共享表征”。

### Norm1 S1/S2 单电极补充结果

在不使用 Norm2/CSC 空间过滤的情况下，对此前 S1=19、S2=46 个电极完成了 raw200 全套单电极 decoding。Task2 memory-color cross-fruit 时频分析在 S1 有16/19个电极、34个显著簇，在S2有40/46个电极、79个显著簇；但 S1/S2 平均 memory-color 曲线均约为0.502，因此单电极 cluster 结果不能直接升级为组水平记忆颜色结论。raw200 的最早显著 onset 为 S1=10–730 ms、S2=0–730 ms，仍不支持稳定的空间-时间传播顺序。

此前所谓的“CSC 12个电极”实际采用的是 `N2 ∩ (S1∪S2)`，不是严格的 `N2 ∩ S1`。在该宽集合中，Task3自身时频有7个电极、10个显著簇，Task2 memory-color cross-fruit时频有9个电极、16个显著簇，cross-task时频没有显著簇；其平均 memory-color accuracy 约0.496。因此它支持“部分电极存在个体探索性记忆颜色可解码”，但不能支持“Norm2∩S1 CSC位点具有稳定记忆颜色代码”。

### stage01 三集合交叠可视化

为便于直接判断三类电极集合的关系，stage01 新增了全部 subject-channel records 的 `Norm1 S1 × Norm1 S2 × Norm2` 交叠汇总图：

- [lf30 三集合交叠图](../result/final_analysis_seeg_20260806_corrected/stage01_selection/figures/norm1s1_norm1s2_norm2_overall_venn_100-400_lf30.png)
- [raw200 三集合交叠图](../result/final_analysis_seeg_20260806_corrected/stage01_selection/figures/norm1s1_norm1s2_norm2_overall_venn_100-400_raw200.png)

该图是集合关系的整体汇总，不是三个独立饼图的并列比较；原有 subject/patch 饼图仍作为下钻结果保留。CSC 仍然是 `Norm2 ∩ (S1 ∪ S2)` 的派生集合。

## 10. 关键结果文件索引

- [正式报告](../result/final_analysis_seeg_20260806_corrected/report/final_analysis_report.md)
- [100–400 ms 参数](../result/final_analysis_seeg_20260806_corrected/analysis_parameters.json)
- [S1/S2 电极集合](../result/final_analysis_seeg_20260806_corrected/stage01_selection/electrode_set_summary_by_subject_100-400_lf30.csv)
- [ANOVA 选电极明细](../result/final_analysis_seeg_20260806_corrected/stage01_selection/functional_selection_100-400_lf30.csv)
- [单电极频谱解码](../result/final_analysis_seeg_20260806_corrected/stage03_decoding/decoding_summary_100-400_lf30.csv)
- [实际 HDF5 试次数审计](../result/final_analysis_seeg_20260806_corrected/stage06_exploration/exemplar_identity_audit_actual_trials.csv)
- [实际试次数审计脚本](../analysis/audit_stimulus_identity.py)
- [S1/S2 raw200 单电极汇总](../result/final_analysis_seeg_20260806_corrected/stage06_exploration/s1s2_timeresolved_electrode_summary_100-400_raw200.csv)
- [S1/S2 raw200 组簇](../result/final_analysis_seeg_20260806_corrected/stage06_exploration/s1s2_timeresolved_group_clusters_100-400_raw200.csv)
- [S1/S2 时间分辨分析脚本](../analysis/s1s2_timeresolved.py)
- [S1/S2 单电极时间分辨曲线图](../result/final_analysis_seeg_20260806_corrected/stage06_exploration/s1s2_timeresolved_single_electrode_curves_100-400_lf30.png)
- [CSC raw200 批量分析参数](../result/final_analysis_seeg_20260806_corrected/stage07_csc_decoding/csc_decoding_parameters_100perm.json)
- [CSC 解码汇总](../result/final_analysis_seeg_20260806_corrected/stage07_csc_decoding/csc_decoding_summary_100perm.csv)
- [CSC 单电极 cluster 结果](../result/final_analysis_seeg_20260806_corrected/stage07_csc_decoding/csc_decoding_cluster_results_100perm.csv)
- [CSC 时频曲线合并图](../result/final_analysis_seeg_20260806_corrected/stage07_csc_decoding/figures/csc_all_electrodes_acc_time_combined_no_permutation.png)
- [CSC raw200 特征主导效应图](../result/final_analysis_seeg_20260806_corrected/stage07_csc_decoding/figures/csc_feature_dominance_group_100-400_raw200.png)
- [S1 memory-color latency/坐标图](../result/final_analysis_seeg_20260806_corrected/stage08_s1_s2_single_electrode_decoding/figures/S1_memory_color_latency_coordinate.png)
- [S2 memory-color latency/坐标图](../result/final_analysis_seeg_20260806_corrected/stage08_s1_s2_single_electrode_decoding/figures/S2_memory_color_latency_coordinate.png)
- [S1/S2 memory-color显著电极 latency/坐标表](../result/final_analysis_seeg_20260806_corrected/stage08_s1_s2_single_electrode_decoding/memory_color_significant_latency_coordinates_100perm.csv)
- [Task3 图片目录](../../visual_experiment/stimuli_pic/Stimuli_Task3)
- [图片重复实现](../../visual_experiment/Utils/run_passive_phase.m)
