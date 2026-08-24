# stage09_2_task3_purecolor_rsa_raw200

> 独立新增结果，不覆盖现有 stage09 Task1 结果。

## 方法

- 输入为 raw200 HDF5；每个 trial 先在完整 −500–1000 ms epoch 上做 STFT。
- log-power 相对于 −200–0 ms、该被试全部请求条件的 trial 做 baseline z-score，然后在条件内平均得到 RDM。
- 使用 16 个既有频带和 0–800 ms 的 16 个连续 50 ms 时间窗。
- 条件：只分析 Task3 的 red（trigger 51）与 green（trigger 54）纯色色块。
- 距离：每个时间窗计算 red 与 green 两个条件的 1−Pearson correlation distance。
- 单频段 decoding 对每个电极、每个频带、每个时间窗独立进行；只报告 balanced accuracy，不做置换显著性检验。
- Task2 使用四个无水果重叠的跨水果训练/测试方向；Task3 使用 red-vs-green 的五折 trial-level CV。
- 图中每个电极包含四/两条条件均值时间信号、16 个 RDM 热图、距离曲线和单频段 decoding 热图。完整数值保存在 CSV，不把每个频段单独输出成图片。
- `single_electrode_rdm_summary.csv`：每个 subject-channel × 50 ms 时间窗一行的宽格式 2×2 RDM，并保存 red-green 距离。
- `single_electrode_index.csv`：52 个单电极的集合归属、坐标、时间窗数量和主图路径。
- `figures/peak_red_green_distance_latency_coordinate_correlation.png`：每个电极取 red–green correlation distance 最大的时间窗后，peak latency 与 MNI x/y/z 的相关图。
- `peak_red_green_distance_latency_coordinates.csv`：逐电极 peak latency、peak red–green distance 和 MNI 坐标。
- `peak_red_green_distance_latency_coordinate_statistics.json`：Pearson/Spearman 相关统计。
- 原始 peak latency–坐标图中的点大小现在表示 peak red–green distance，点越大代表红绿色块距离 peak 越大。
- `figures/composite_peak_index_mni_y_0_400ms.png`：`z(peak red-green distance) − z(peak latency)` 与 MNI Y 的相关图。
- `composite_peak_index_mni_y_0_400ms.csv` 和对应 statistics JSON：逐电极 composite 指标及 Pearson/Spearman 结果。

## 解释边界

该结果是条件平均层面的几何和逐频段解码描述。Task2 四种水果中每种记忆颜色只有两种水果，因此跨水果泛化支持共同结构，但不能完全排除水果语义差异；Task3 red-green 是物理颜色参照，不等同于跨任务共享表征检验。

## 摘要
- unique electrodes: 52
- set counts: {"S1": 19, "S2": 46, "CSC": 12}
- workers: 20
- output directory: `E:\liulab_project\Project_colorieeg_2026\color_analyse_0727\result\final_analysis_seeg_20260806_corrected\stage09_2_task3_purecolor_rsa_raw200`

完整命令、输入 SHA256、Git commit、Python 环境、时间和 warnings 保存在项目根目录 `runs/` 的对应运行目录。
