# 颜色分析管线 v2 结果目录

生成日期：2026-08-06

## 目录结构

- `analysis_parameters.json`：本次运行全部参数
- `output_index.csv`：全部输出文件的索引（可用 pandas 按 stage/window/signal/subject 过滤）
- `report/`：最终报告 final_analysis_report.md 与 .pptx
- `stage01_selection/`：功能筛选、N2 空间筛选、CSC 电极集合与重叠图
- `stage02_amplitude_spectral/`：幅度统计、16 频带功率统计与频谱图
- `stage03_decoding/`：频谱级解码、置换 p 值、类内方向一致性
- `stage04_luminance/`：刺激亮度/对比度/色彩度审计
- `stage05_hypotheses/`：100–400 ms 的 S1 富集检查；H2/H3/TGM 未把旧版 1–300 ms 结果搬入
- `stage06_exploration/`：Task3 图片身份审计、S1/S2 时间分辨曲线和被试层面探索性簇
- `stage07_csc_decoding/`：CSC 全部电极 raw200 decoding
- `stage08_s1_s2_single_electrode_decoding/`：S1/S2 全部电极 raw200 decoding
- `stage09_task1_condition_rsa_raw200/`：Task1 条件均值时频 RSA
- `stage09_1_task2_grayfruit_rsa_raw200/`：Task2 四种灰色水果 RDM、同/异记忆颜色距离和单频段 decoding
- `stage09_2_task3_purecolor_rsa_raw200/`：Task3 red/green 纯色色块 RDM、red-green 距离和单频段 decoding
- `stage09_3_task2_task3_cross_rsa_raw200/`：Task2 灰色水果与 Task3 纯色色块的跨任务六条件 RDM、同/异记忆颜色距离
- `cache/`：中间特征数组（不浏览）

### stage01 三集合重叠汇总图

- `stage01_selection/figures/norm1s1_norm1s2_norm2_overall_venn_100-400_lf30.png`
- `stage01_selection/figures/norm1s1_norm1s2_norm2_overall_venn_100-400_raw200.png`

这两张图将 Norm1 S1（`strategy1`）、Norm1 S2（`strategy2`）和 Norm2（`N2_union`）作为三个集合，在全部 subject-channel records 上绘制同一张交叠汇总图。数字表示八个互斥区域的记录数；CSC 仍按 `Norm2 ∩ (S1 ∪ S2)` 定义，不作为独立第三集合。原有按被试/patch 的饼图仍保留，筛选逻辑和 CSV 数据没有改变。

## 当前主分析

- 主时间窗：100–400 ms；
- 当前 decoding 主信号：`raw200`（直接使用 HDF5 中的 1–200 Hz 信号）；
- 正式电极选择：Task 1 二因素 ANOVA（color × category）的 color 主效应 `p<0.05`；
- `100-400_lf30`：S1=19、S2=46、CSC=12；
- `100-400_raw200`：S1=16、S2=44、CSC=10；
- 本次 stage06–08 的 raw200 解码为直接信号源比较：stage07/08 保留此前 lf30 选出的 CSC、S1、S2 电极成员，避免把电极集合变化与频段变化混在一起；参数文件中记录了 `selection_variant=100-400_lf30`。
- 旧的 lf30 解码文件暂不删除，但只作为历史对照；其中 30 Hz 以上的 feature dominance 不能解释为真实高频活动。

## 数据来源与分析方法

- 原始实验记录：`color_analyse_0727/Data/testXXX/`，包含 Task1–3 的 MATLAB `cfg/results/stimData`、图片文件名、marker、呈现时间、catch trial 和 response，也包含 Task6–8 刺激实验记录。
- 连续 EEG：根目录 `seegdata/`；定位/ROI：`processed_data/`；标准分析输入：`process_data/` 的 21 个 HDF5；对齐和审计：`metadata/`。
- `Data` 中的 Task3 trial log 已通过 `scripts/normalize_task_labels_and_index_passivecolorpatch.py` 与 `seegdata/erp3.set` 的 event marker 对齐，输出 `metadata/passivecolorpatch_shape_trial_index.csv` 和 `metadata/passivecolorpatch_alignment_audit.csv`。该 trial-level index 是后续 exemplar-wise decoding 的唯一推荐依据。
- 预处理为 1–200 Hz、50/100/150 Hz 陷波、严格左右相邻 Laplacian、−500–1000 ms epoch；当前主 decoding 使用 raw200。
- 频谱特征为 16 个 Welch log-power 频带，频谱窗口 1–1000 ms；时频特征为 256 ms STFT 窗、10 ms 步进，正式时间范围 0–800 ms。
- Task2 使用 leave-one-fruit-pair-out；stage06 进行 5000 次组 sign-flip，stage07/08 进行每电极 100 次标签置换和 max-cluster-mass 检验。所有 feature dominance 图均为描述性 SVM 权重。

## Task1 条件均值时频 RSA（stage09）

- 独立分析目录：`stage09_task1_condition_rsa_raw200/`，不覆盖 stage01–08；输入为 Task1 HDF5 的 raw200（1–200 Hz）epoch。
- 8 个条件输出顺序固定为 `face_color, object_color, body_color, place_color, face_gray, object_gray, body_gray, place_gray`；每个 trial 先做完整 −500–1000 ms STFT，再对 log-power 相对于 −200–0 ms 做 baseline z-score，最后在条件内平均。
- 在 0–800 ms 按 50 ms 分成 16 个连续时间窗，使用既有 16 个频带；每个唯一 subject-channel 输出 16 个 8×8 correlation-distance RDM（`1 − Pearson correlation`）。
- 共 52 个唯一电极，集合数量 S1=19、S2=46、CSC=12；单电极图 52 张，集合平均图 3 张。集合图对距离矩阵直接算术平均，不做 Fisher z 或显著性检验。
- 输出包括 `condition_mean_timecourses.csv`、`condition_mean_tf_features.csv`、`condition_rdm_long.csv`、`condition_trial_counts.csv`、`electrode_sets_used.csv`、`rsa_parameters.json` 和 `README.md`。
- 解释边界：这是条件平均层面的神经表征几何结构，不能直接解释为已经排除了图片 exemplar、trial variance 或图片身份差异；`Data/` 仅用于身份/trigger 审计。

## Task2 灰色水果与 Task3 纯色红绿 RSA（stage09_1 / stage09_2）

- `stage09_1_task2_grayfruit_rsa_raw200/` 的四水果顺序为 R1=strawberry、R2=watermelon、G1=cabbage、G2=kiwi；每个通道和 50 ms 时间窗输出 4×4 RDM，并绘制同记忆颜色距离 `(R1R2+G1G2)/2` 与异记忆颜色距离 `(R1G1+R1G2+R2G1+R2G2)/4`。
- `stage09_2_task3_purecolor_rsa_raw200/` 使用 red（51）与 green（54）纯色色块，输出 red-green correlation distance 的时间变化。
- 两个 stage 均输出 16 个频带×16 个时间窗的单频段 trial-level decoding；Task2 使用四个跨水果方向，Task3 使用五折 red-vs-green CV；使用 20 个线程，不做置换显著性检验。
- 每个 stage 均包含 52 张通道图和 S1/S2/CSC 集合图；完整数据在 `condition_mean_tf_features.csv`、`condition_rdm_long.csv`、`memory_color_distance_curves.csv` 和 `single_band_decoding.csv`。

## Task2 灰色水果 × Task3 纯色色块跨任务 RSA（stage09_3）

- `stage09_3_task2_task3_cross_rsa_raw200/` 将四种 Task2 灰色水果和 Task3 red/green 纯色色块合并为六个条件：`R1, R2, G1, G2, red, green`。
- 六类 trial 在同一 subject-electrode 内联合后进行 raw200 STFT 和 −200–0 ms baseline z-score，再在 0–800 ms 的 16 个 50 ms 时间窗计算 6×6 correlation-distance RDM。
- 同记忆颜色距离为 `(R1-red + R2-red + G1-green + G2-green)/4`；异记忆颜色距离为 `(R1-green + R2-green + G1-red + G2-red)/4`。主输出为 `cross_task_memory_color_distance_curves.csv`。
- 每个唯一电极、S1/S2/CSC 集合均有主图；集合图对单电极 RDM 和曲线做直接算术平均，不进行显著性检验。

所有当前 RSA 分支均已保存单电极层面的显式索引和宽格式 RDM 汇总：每个文件包含 52 个 subject-channel 电极，并按 50 ms 时间窗逐行保存 RDM 单元格；`single_electrode_index.csv` 还提供对应主图路径。

stage09_1、stage09_2 和 stage09_3 另保存每个电极的 peak latency–MNI 坐标相关图。stage09_1/3 按两条距离曲线的绝对差异取 peak，并保留有符号差异；stage09_2 按 red–green 距离最大值取 peak。

stage09_1/3 的两条曲线还按 peak 时点的方向拆分为“第二条线更大”和“第一条线更大”两组，分别输出相关图和统计表。

新增 composite 指标 `early_strength_index = z(peak value) − z(peak latency)`，统一在 0–400 ms 内取 peak，并与 MNI Y 做相关；结果保存在三个 stage 的 `composite_peak_index_mni_y_0_400ms.*` 文件中。

原有 peak latency–MNI x/y/z 图已更新为点大小表示 peak 幅值：stage09_1/3 为两条距离线的绝对差值，stage09_2 为 peak red–green distance。

## 单电极性能基准

- 测试电极：`test001-D6`；置换次数：100；并行进程：20/24 个逻辑 CPU；内存：约 31.4 GB；未检测到 NVIDIA GPU。
- 测试内容：1–1000 ms 频谱 Task2/Task3、Task2↔Task3 cross-decoding，以及 10 ms 步进的时频 Task2/Task3、Task2↔Task3 cross-decoding，共 101 个时间点。
- 总耗时：44.045 s/电极/100 次置换；其中时频部分约 41.63 s，占总时间约 94.5%。按近似线性外推，1000 次置换约 7.3 min/电极。
- 原始基准记录：`stage06_exploration/benchmark_one_electrode_100perm.json`；复现脚本：`analysis/benchmark_one_electrode.py`。
- 优化版同电极基准：使用 `LinearSVC(dual=False)`，并缓存每个训练 fold/时间点的标准化结果；总耗时 **9.749 s/电极/100次置换**，相对原始 44.045 s 约 **4.52 倍加速**。记录：`stage06_exploration/benchmark_one_electrode_100perm_optimized.json`；脚本：`analysis/benchmark_one_electrode_optimized.py`。
- 注意：当前 STFT 为约 256 ms 窗、10 ms 步进；现有帧覆盖到约 878 ms，1–1000 ms 的末端时间点暂按最近帧映射。正式批量分析前应决定是否补零/延长 epoch，且应先修正 Task2 时间分辨 cross-fruit 的特征索引实现。

## stage06 探索性时间分辨分析

- `stage06_exploration/` 同时保存 Task3 图片身份审计、单电极运行基准和 S1/S2/CSC 的两条时间分辨曲线：Task3 physical red/green 与 Task2 gray-fruit memory red/green。
- 当前 raw200 版本对每个电极只输出描述性曲线，再按被试和电极集合平均，使用 5000 次 sign-flip 做组水平 cluster；它不等同于 stage08 的单电极 8 分支 permutation 分析。
- 当前 raw200 组簇文件为 `s1s2_timeresolved_group_clusters_100-400_raw200.csv`，曲线图为 `s1s2_timeresolved_single_electrode_curves_100-400_raw200.png`。Task3 有探索性组簇，Task2 memory-color 未形成组水平显著簇。

## CSC 全部电极批量解码

- 当前分析集合：此前 `100-400_lf30` 选择出的 12 个 CSC 电极，解码输入改为 `100-400_raw200`；频谱特征使用 1–1000 ms，时频 acc 曲线和 cluster 检验按用户要求限制在 0–800 ms。
- 每个电极运行 8 类分析：Task3 自身频谱/时频、Task2 cross-fruit 频谱/时频、Task3→Task2 和 Task2→Task3 的频谱/时频 cross-decoding；100 次置换；使用优化版 `dual=False` SVM 与缓存标准化。
- 每个电极的时频曲线单独做 1D max-cluster-mass permutation（形成阈值 p=0.05、最小簇 20 ms、+1 p 修正）；合并曲线只用于展示，不做置换检验，也不对电极/分析分支做跨集合校正。
- raw200 后 Task3 自身频谱平均准确率约 0.792，Task2 cross-fruit 频谱约 0.493；时频分别约 0.548 和 0.507。特征权重不再稳定集中于旧 lf30 图中的 75–95 Hz，而在 15–25、65–75、85–95、105–125、165–185 Hz 等频带间分散；feature dominance 仍只是描述性结果。
- 原始数据：`stage07_csc_decoding/electrode_npz/`；汇总：`csc_decoding_summary_100perm.csv`、`csc_decoding_time_curves_100perm.csv`、`csc_decoding_cluster_results_100perm.csv`、`csc_decoding_feature_dominance_100perm.csv`。
- 当前图形：`stage07_csc_decoding/figures/individual_electrodes/`、`csc_all_electrodes_acc_time_combined_no_permutation.png`、`csc_feature_dominance_group_100-400_raw200.png` 和 `csc_feature_dominance_electrode_100-400_raw200.png`。
- 批量脚本：`analysis/run_csc_decoding.py`；参数记录：`stage07_csc_decoding/csc_decoding_parameters_100perm.json`。

## Norm1 S1/S2 全套单电极解码

- 不应用 Norm2/CSC 空间过滤，使用此前 lf30 选择的 `strategy1` S1=19、`strategy2` S2=46，重叠 13 个，唯一电极 52 个；解码输入改为 raw200。
- 每个电极完成同样的 8 类频谱/时频 decoding、100 次置换和单电极 max-cluster-mass 检验；时频图和检验范围为 0–800 ms。
- raw200 Task2 memory-color（cross-fruit 时频）在 S1 中有 16/19 个电极、34 个显著簇；S2 中有 40/46 个电极、79 个显著簇。两组平均 memory-color accuracy 约为 S1=0.502、S2=0.502，单电极显著不等于组水平显著。
- raw200 的最早显著 cluster onset 范围为 S1=10–730 ms、S2=0–730 ms；这些单电极 latency 仍不能建立方向性信息流。
- 结果目录：`stage08_s1_s2_single_electrode_decoding/`；两张主图为 `figures/S1_memory_color_latency_coordinate.png` 和 `figures/S2_memory_color_latency_coordinate.png`。
- 汇总数据：`s1s2_decoding_summary_100perm.csv`、`s1s2_decoding_cluster_results_100perm.csv`、`memory_color_significant_latency_coordinates_100perm.csv`、`s1s2_decoding_feature_dominance_100perm.csv`；脚本：`analysis/run_s1s2_decoding.py`。

## 本次 raw200 修正说明

`lf30` 的旧路径先将输入信号滤为 1–30 Hz，但随后仍对该信号提取完整的 5–195 Hz 特征。因此旧图中 35 Hz 以上的权重不能代表真实高频活动，75–95 Hz 的集中模式很可能来自截止频率以上残留/数值噪声经过标准化后的分类权重。raw200 重跑后，权重不再稳定集中于 75–95 Hz，但 65–75 Hz 在部分分析中仍较高，所以不能把所有 60–80 Hz 效应都归因于旧滤波问题；它需要进一步用频带消融、嵌套交叉验证和独立置换验证。

## 命名规则

变体专属文件带 `{时间窗}_{信号}` 后缀（`0-300_lf`、`100-400_broadband` 等）；
所有表都带 `window` 与 `signal` 列，便于过滤。

## 重跑命令

```powershell
C:\Users\saber_soul\.conda\envs\lr0727\python.exe -m analysis.run_final_analysis --perms 1000 --workers 8
```

只跑部分阶段：`--stages selection csc stats decoding luminance report` 中任选；
当前主变体：`--windows 100-400 --signals lf30 raw200`；也可显式指定其他窗口做敏感性分析。

## 浏览方式

打开 `notebooks/02_browse_results.ipynb`，设置 RESULT_DIR 后可按阶段/窗口/信号/被试过滤查看。
