# color_analyse_0825

`color_analyse_0825` 是独立于冻结版 `color_analyse_0727` 的新分析版本。

## 当前预处理规范

- 输入信号：`seegdata/testN/erp1-3.set/.fdt`
- 行为日志：`visual_experiment/Data/testNNN/*Passive*Session*.mat`
- 连续通道稳健 DC 中心化
- 1000 Hz 重采样至 500 Hz
- 四阶 Butterworth、零相位 `0.1–200 Hz` 带通
- 50/100/150 Hz、Q=30、零相位陷波
- 仅排除 `metadata/manual_channel_decisions.csv` 中明确确认的坏道
- 严格同电极杆左右直接相邻 Laplacian
- `[-500, 1000) ms` epoch，共 750 点；不做事件级基线校正
- 输出为 `trial × channel × time` 的 MATLAB v7.3 `.mat`

本版本相对 0727 的预处理科学参数变化是带通下限从 `1 Hz` 改为用户指定的 `0.1 Hz`。

## MATLAB 入口

```matlab
addpath('E:\liulab_project\Project_colorieeg_2026\color_analyse_0825\matlab');

% 全部 test001-test008、Task 1-3
summary = run_preprocess_0825();

% 小范围运行
summary = run_preprocess_0825('subjects', {'test001'}, 'tasks', 1);

% 独立验证
report = validate_preprocess_0825();
```

## 代码审查入口

预处理只有一个科学主流程：`matlab/run_preprocess_0825.m`。配置、事件提取、
行为对齐、人工坏道排除、严格 Laplacian 公式、滤波、切段和保存均按执行顺序
直接写在该文件中。文件末尾仅保留 Trigger 注册表、Trigger/文本格式转换和
UTF-8 文本写入这 4 个重复使用的小函数；关键科学步骤不在辅助函数中。

主脚本对原始采样率、连续记录 trial 维、FDT 字节数、各阶段数组 shape、行为
对齐率、输出 trial 数、500 Hz 采样率、750 个时间点和 NaN/Inf 均使用 `assert`
检查。独立验证脚本 `matlab/validate_preprocess_0825.m` 会在输出生成后再次检查
同一数据契约。

每次运行的完整配置、命令、输入文件身份、Git commit、环境、日志、警告和输出路径保存在 `runs/preprocess_0p1_200Hz_*`。实际 epoch 文件保存在：

```text
process_data/testNNN/taskK_epoched_0p1_200Hz.mat
```

每个 `.mat` 只保存一个顶层结构体 `preprocessed_epoch_struct`，其中核心字段为：

- `voltage_trial_x_channel_x_time`：`trial × channel × 750` 的单精度信号；
- `time_ms_1_x_time`：`-500:2:998 ms`；
- `channel_labels_nchannel_x1`：严格 Laplacian 中心触点；
- `reference_members_nchannel_x1`：每个通道的 `center|left|right`；
- `trial_info_ntrial_x_field`：EEG 事件和行为日志的对齐元数据。

`behavior_matched=false` 的 trial 仍保留信号，但后续逐图片 color-gray 配对分析
不得使用。
