# Stage09 Task1 条件均值时频 RSA（raw200）

> 这是独立的描述性分析，不覆盖 stage01–08。

## 分析定义

- 输入：Task1 HDF5 中已经完成 1–200 Hz 预处理的 epoch（raw200）。
- 条件顺序：face_color, object_color, body_color, place_color, face_gray, object_gray, body_gray, place_gray。
- 每个 trial 先做完整 -500–1000 ms epoch 的 STFT，再以每个 trial 的 log-power 相对于 -200–0 ms 做 baseline z-score；baseline 的均值和标准差在该被试全部八个条件的 trial 上估计，避免使用条件标签估计 baseline。
- 条件均值在 trial 级 TF 特征上计算，然后按连续 50 ms 时间窗汇总 16 个频带。时间信号图也按 trial 先做 -200–0 ms 基线扣除，再在条件内平均。
- 每个时间窗的 RDM 使用 16 个频带特征，距离为 `1 - Pearson correlation`；对角线为 0。
- 集合平均对电极的距离矩阵直接取算术平均，不做 Fisher z；因此集合图是电极距离的平均描述，不是单电极推断。

## 解释边界

这些结果表示条件平均层面的神经表征几何结构。条件平均不能排除图片 exemplar、trial 构成或被试内样本方差的影响，因此不能直接解释为已经消除了图片身份或 trial variance。`Data/` 目录仅用于实验 trial 身份和触发器审计，本分析的神经特征直接来自 HDF5。

## 输出

- `electrode_figures/`：每个唯一 subject-channel 一张时间信号 + 16 个 RDM 热图。
- `set_figures/`：S1、S2、CSC 的电极距离矩阵平均图。
- `condition_mean_timecourses.csv`：条件均值时间信号。
- `condition_mean_tf_features.csv`：条件均值、50 ms 分箱、16 频带 TF 特征。
- `condition_rdm_long.csv`：每个电极和时间窗的完整 8×8 RDM。
- `single_electrode_rdm_summary.csv`：每个 subject-channel × 时间窗一行的宽格式 8×8 RDM 单电极结果。
- `single_electrode_index.csv`：52 个单电极的集合归属、坐标、时间窗数量和主图路径。
- `condition_trial_counts.csv`：每个被试、电极、条件的有效 trial 数和 QC 状态。
- `electrode_sets_used.csv`：去重后的电极及 S1/S2/CSC 归属。
- `rsa_parameters.json`：参数、触发器、输入和输出记录。

## 运行摘要

- unique electrodes: 52
- RDM time bins: 16
- set counts: {"S1": 19, "S2": 46, "CSC": 12}
- output path: `E:\liulab_project\Project_colorieeg_2026\color_analyse_0727\result\final_analysis_seeg_20260806_corrected\stage09_task1_condition_rsa_raw200`

完整的命令、输入 SHA256、Git commit、环境和 warnings 保存在项目根目录 `runs/` 下对应的本次运行目录。
