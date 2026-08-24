# stage09_3_task2_task3_cross_rsa_raw200

> 独立新增结果，不覆盖 stage01–08、stage09、stage09_1 或 stage09_2。

## 方法

- Task2 的四种灰色水果与 Task3 的 red/green 纯色色块在同一 subject-electrode 的六条件集合中联合分析。
- 条件顺序固定为 R1=strawberry_gray、R2=watermelon_gray、G1=cabbage_gray、G2=kiwi_gray、red=Task3 red、green=Task3 green。
- 六类 trial 合并后，对完整 −500–1000 ms epoch 做 trial-level raw200 STFT；log-power 相对于联合六条件的 −200–0 ms baseline 做 z-score，然后在条件内平均。
- 在 0–800 ms 计算 16 个连续 50 ms 时间窗的 6×6 correlation-distance RDM，距离为 `1 − Pearson correlation`，特征为既有 16 个频带。
- 同记忆颜色距离为 `(R1-red + R2-red + G1-green + G2-green) / 4`；异记忆颜色距离为 `(R1-green + R2-green + G1-red + G2-red) / 4`。
- 集合图采用单电极 RDM 和距离曲线的直接算术平均；本版本不做 bootstrap、置换检验或跨电极显著性检验。
- `single_electrode_rdm_summary.csv`：每个 subject-channel × 50 ms 时间窗一行的宽格式 6×6 RDM，并合并同/异记忆颜色距离。
- `single_electrode_index.csv`：52 个单电极的集合归属、坐标、时间窗数量和主图路径。
- `figures/peak_cross_task_memory_color_distance_latency_coordinate_correlation.png`：每个电极取 `abs(different − same)` 最大的时间窗后，peak latency 与 MNI x/y/z 的相关图。
- `peak_cross_task_memory_color_distance_latency_coordinates.csv`：逐电极 peak latency、peak 两条距离、signed/absolute difference 和 MNI 坐标。
- `peak_cross_task_memory_color_distance_latency_coordinate_statistics.json`：Pearson/Spearman 相关统计。
- 原始 peak latency–坐标图中的点大小现在表示 peak absolute line difference，点越大代表 same/different 距离曲线的 peak 分离幅度越大。
- 方向拆分图（peak 限制在 0–400 ms）：`figures/peak_latency_coordinate_correlation_different_gt_same_0_400ms.png` 与 `figures/peak_latency_coordinate_correlation_same_gt_different_0_400ms.png`。
- 方向拆分表：`peak_latency_coordinates_different_gt_same_0_400ms.csv` 与 `peak_latency_coordinates_same_gt_different_0_400ms.csv`。
- `figures/composite_peak_index_mni_y_0_400ms.png`：`z(peak absolute difference) − z(peak latency)` 与 MNI Y 的相关图，按 `different > same` / `same > different` 分面。
- `composite_peak_index_mni_y_0_400ms.csv` 和对应 statistics JSON：逐电极 composite 指标及 Pearson/Spearman 结果。

## 解释边界

同记忆颜色距离低于异记忆颜色距离，才是与跨任务共同记忆颜色结构一致的描述性模式；该结果仍不能排除 Task2/Task3 任务差异、图片/刺激身份或低层视觉差异。

## 摘要
- unique electrodes: 52
- set counts: {"S1": 19, "S2": 46, "CSC": 12}
- conditions: ["strawberry_gray", "watermelon_gray", "cabbage_gray", "kiwi_gray", "red_patch", "green_patch"]
- output directory: `E:\liulab_project\Project_colorieeg_2026\color_analyse_0727\result\final_analysis_seeg_20260806_corrected\stage09_3_task2_task3_cross_rsa_raw200`

完整命令、输入 SHA256、Git commit、Python 环境、时间和 warnings 保存在项目根目录 `runs/` 的对应运行目录。
