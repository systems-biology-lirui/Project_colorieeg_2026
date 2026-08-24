# color_analyse_0727：统一数据转换与预处理管线

本目录是项目后续分析的主工作区。旧版 `code`、`code_old`、`feature*` 和 `result*` 已清理；当前 `process_data/` 只保存重建后的 HDF5；`docs/` 与 `prompt/` 仅作为历史分析资料保留，不参与当前预处理。

当前阶段已经完成：

- 统一任务、触发码和条件名称；
- 对 7 位被试、3 个任务的 `.set/.fdt` 输入进行只读审计；
- 生成坏道候选表和候选通道诊断图；
- 保留坏道候选表并完成用户确认的坏道决定；
- 完成全长滤波后的 QC，并生成 7 位被试 × 3 个任务的 HDF5；
- 按信号通道与定位表交集生成邻接双极中心；
- 根据电刺激行为学记录更新 7 位被试的定位表，并保留原始备份；
- 生成 epoch 均值与 SEM 阴影示例图。

## 1. 数据输入

当前管线只读取：

```text
E:/liulab_project/Project_colorieeg_2026/seegdata/test1/erp1.set + erp1.fdt
...
E:/liulab_project/Project_colorieeg_2026/seegdata/test7/erp3.set + erp3.fdt
```

`processed_data/` 中的电极定位表作为元数据参考，现已在备份后补充电刺激行为学字段；原始 `rawdata/` 不会被修改。rawdata→seegdata 的通用映射和转换函数分别位于 `metadata/raw_to_seeg_task_map.csv` 与 `matlab/raw_to_seegdata.m`。

电刺激标注规则是：只有行为记录中出现明确颜色证据的刺激对，其接触点才将 `color_with_sti=True`；白光、灰光、透明光和无反应不标记，弱/模糊颜色证据保留在 `stimulation_behavioral_annotation.csv` 的 review 字段中。行为学记录摘要中共有 `test001: 9`、`test002: 6`、`test003: 11`、`test004: 0`、`test005: 4`、`test006: 2`、`test007: 0` 个清晰颜色接触点；其中 test003 的数字通道 16/17/18 不在定位表中，所以实际写入定位表并能进入定位交集的是 `test003: 8`，其余被试分别为 `9/6/0/4/2/0`。原始表备份位于 `metadata/localization_original/`。

## 2. 统一条件注册表

### Task 1：物体类别 × 颜色状态

| 触发码 | 条件 |
|---:|---|
| 11 | face_color |
| 12 | face_gray |
| 21 | object_color |
| 22 | object_gray |
| 31 | body_color |
| 32 | body_gray |
| 41 | place_color |
| 42 | place_gray |

### Task 2：水果 × 记忆判断状态

Task 2 保留三种状态，绝不把它们混成一个条件。当前记忆颜色解码先使用 `gray`，之后可独立使用 `true` 和 `false`。

| 水果 | true | false | gray |
|---|---:|---:|---:|
| cabbage | 101 | 102 | 103 |
| kiwi | 111 | 112 | 113 |
| strawberry | 121 | 122 | 123 |
| watermelon | 131 | 132 | 133 |

### Task 3：纯色块

`51=red`、`52=yellow`、`53=blue`、`54=green`、`55=black`、`56=white`。

唯一的条件定义位于 `pipeline/condition_registry.py`，事件码采用精确匹配，不使用可能造成误配的前缀匹配。

## 3. 当前预处理规范

`pipeline/preprocess.py` 按以下顺序执行：

1. 逐通道去除连续记录中的稳健 DC 偏置；这不是事件级基线校正；
2. 1000 Hz 连续信号重采样至 500 Hz；
3. 零相位 1–200 Hz 带通滤波；
4. 50/100/150 Hz 陷波，Q=30；
5. 依据人工审核表排除明确标记为 `exclude` 的单端通道；
6. 对每个电极排使用严格左右相邻 Laplacian 重参考；端点或缺少左右邻居的触点不生成重参考信号；
7. 按 `[-500, 1000] ms` 分段，保存 750 个时间点；
8. 按条件分别写入 HDF5。Task 2 的 12 个条件全部保留，后续分析时再选择 `gray`、`true` 或 `false` 子集。

当前导出不做基线扣除；基线校正或 -200–0 ms 标准化留给具体 ERP、High-Gamma 或 decoding 分析模块，以避免把预处理和分析策略混在一起。

## 4. 坏道审核原则

本管线不会根据单一标准差阈值自动删道。候选判断综合考虑：缺失值、平线、饱和、连续跳变、相对工频异常，以及旧项目中已有的坏道记录。`baseline_rms`、`response_rms` 和 `response_to_baseline_ratio` 仅用于帮助判断，不会因为诱发响应较大而自动删道。

候选表中的 `candidate_level` 只是审计提示，不等于最终坏道结论。当前主 QC 只保留经过 DC 中心化和完整滤波链处理后的结果：

- `normal`：没有触发当前保守规则；
- `review_candidate`：建议查看诊断图和原始波形；
- `high_confidence_candidate`：通常是旧记录中的坏道或明显硬件异常，但仍由人工确认；
- `protected_review`：功能/光幻视相关保护通道，只允许人工保留，导出器拒绝将其排除。

人工审核请编辑：

```text
qc/bad_channel_candidates.csv
```

完整表包含所有通道；如果希望只看需要判断的行，可参考当前滤波后生成的 `qc/bad_channel_candidates_to_review_filtered.csv`。最终供导出器读取的仍是完整表 `bad_channel_candidates.csv`。只需填写 `manual_decision` 和 `manual_comment`：`exclude` 表示确认剔除，`keep` 表示确认保留；空白也按保留处理，但建议把所有非 `normal` 行明确填写。审核图在 `qc/channel_diagnostics/`。

已经由用户明确确认的坏道另外记录在 `metadata/manual_channel_decisions.csv`，导出时会与审核表中的明确 `exclude` 决定合并使用。

## 5. 运行命令

推荐通过两个 Notebook 阅读和运行：

```text
notebooks/00_rawdata_to_seegdata.ipynb
notebooks/01_preprocess_and_export.ipynb
```

命令行等价入口：

```powershell
<your-python> scripts\validate_conditions.py
<your-python> scripts\audit_bad_channels.py
<your-python> scripts\plot_bad_channel_candidates.py
```

人工编辑 `qc/bad_channel_candidates.csv` 后，如需重建，再执行：

```powershell
E:\software\Anaconda\python.exe scripts\build_hdf5.py
```

新数据写入：

```text
color_analyse_0727/process_data/<subject>/task<task>_epoched_1_200Hz.h5
```

当前 21 个 HDF5 均已通过 `metadata/hdf5_validation_report.csv` 验证。示例图位于 `result/epoch_examples/`。

## 6. 目录说明

```text
color_analyse_0727/
├── pipeline/
│   ├── condition_registry.py  # 唯一任务/触发码/条件注册表
│   ├── config.py              # 路径与预处理参数
│   ├── io_seeg.py             # .set/.fdt 只读读取与事件解析
│   ├── quality_audit.py       # 保守的坏道候选审计
│   ├── quality_plots.py       # 候选通道波形/PSD诊断图
│   ├── epoch_plots.py         # epoch 均值与 SEM/SD 阴影
│   ├── hdf5_io.py             # HDF5 读取与验证
│   └── preprocess.py          # 滤波、重参考、epoch、HDF5导出
├── matlab/
│   ├── raw_to_seegdata.m      # 通用 rawdata→seegdata 转换
│   └── plot_epoch_examples.m  # MATLAB 环境下的示例图桥接
├── notebooks/
│   ├── 00_rawdata_to_seegdata.ipynb
│   └── 01_preprocess_and_export.ipynb
├── scripts/
│   ├── build_electrode_manifest.py
│   ├── validate_conditions.py
│   ├── audit_bad_channels.py
│   ├── plot_bad_channel_candidates.py
│   ├── build_hdf5.py
│   └── validate_hdf5.py
├── qc/                        # 审计结果、人工审核表和诊断图
├── metadata/                  # 坏道、定位表、电刺激记录和 HDF5 审计元数据
├── docs/                      # 历史分析报告，仅供参考
└── prompt/                    # 历史需求记录，仅供参考
```

后续真正开始 decoding 前，应在 `01_preprocess_and_export.ipynb` 中按分析目的选择 Task 2 的 `gray`、`true` 或 `false` 条件集合；这些条件已经在同一个 HDF5 中分别保存。
