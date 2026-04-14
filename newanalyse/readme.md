# newanalyse 代码说明与核查结论

这份说明只覆盖 `newanalyse/` 目录中的当前流程，并以 2026-04-13 对代码和实际输出文件的核查结果为准。

本目录现在可以分成四类内容：

1. `Sec1_*`：从原始 `.set` 生成 `processed_data/{subject}` 下的 task 级 epoched `.mat`。
2. `Sec2_*`：把 task 级 epoched 数据提取为 ROI 级特征文件。
3. `Sec3_*`：做 ROI decoding、ROI 内条件比较，或全电极 decoding。
4. `Sec4_*` / `Sec5_*`：批处理、重绘和汇总脚本。

最重要的结论先写在前面：

1. 当前 ROI feature 文件的数据契约已经核实，`Sec3` 主 decoding 脚本读取 trial 轴的方式是对的。
2. ROI 分组不是互斥分组，而是允许一个通道同时属于多个 ROI；这会影响结果解释。
3. 目录里名为 `tfa` 的主线流程并不直接保存频率维；它保存的是 1-150 Hz 分支的时域 epoched 信号。真正的 Morlet 时频分解发生在 `Sec3_s2_roi_condition_tfa.py`，不是 `Sec2_4_preprocess_tfa.m` 或 `Sec3_4_all_roi_result_tfa.py`。
4. 不同模态的显著性时间窗规则目前不完全一致：ERP/TFA 主 decoding 会屏蔽 20 ms 以前时间点，高 gamma / low gamma / gamma / gamma multiband 主 decoding 默认不会。

---

## 1. 目录与执行顺序

推荐顺序：

1. `Sec1_preanalyse.m`
2. 对应模态的 `Sec2_*`
3. 对应模态的 `Sec3_*`
4. 需要批处理时使用 `Sec4_*`
5. 需要重绘或页面汇总时再使用 `Sec5_*`

默认工作目录与结果目录：

1. 输入数据：`processed_data/{subject}`
2. ROI 特征：`feature/{modality}/{subject}`
3. 主 decoding 结果：`result/decoding/{task_id}/{modality}/{subject}/{batch_name}/{perm_tag}/with_sti`
4. cross decoding 结果：`result/cross_decoding/{task_id}/{modality}/{subject}/{batch_name}/{perm_tag}`
5. ROI 条件比较与补充报告：`result/roi_condition_tfa/...`、`result/reports/...`

---

## 2. 统一数据契约

### 2.1 采样与时间窗

当前主线默认采样率是 500 Hz。

`Sec1_preanalyse.m` 生成的 epoched 数据原始时间窗是 `-500 ~ 1000 ms`。后续大多数 `Sec2_*` 脚本都会再裁剪到 `-100 ~ 1000 ms`，也就是 550 个时间点。

### 2.2 task 级 epoched `.mat` 的真实内容

`Sec1_preanalyse.m` 的两个核心输出是：

1. `task{1,2,3}_ERP_epoched.mat`
2. `task{1,2,3}_TFA_epoched.mat`

这两个文件内部的 `epoch.data` 都是 4 维数组：

`[Cond, Rep, Ch, Time]`

这里的含义已经用实际文件核实过，例如 `test003` 的 `Color_with_sti.mat`：

1. `erp_task1.shape = (8, 70, 8, 550)`
2. `tfa_task1.shape = (8, 70, 8, 550)`
3. `gmb_task1.shape = (8, 70, 64, 550)`

这说明 `Sec3` 主 decoding 代码在读取条件后，把 `Rep` 维展开成 trial 维的基本方式是正确的。

### 2.3 ERP 与 TFA 分支的真实区别

`Sec1_preanalyse.m` 中：

1. ERP 分支做的是 `1-30 Hz` 滤波、epoch、基线校正。
2. TFA 分支做的是 `1-150 Hz` 滤波、epoch、基线校正。

注意：这里的 `TFA` 只是一个更宽频段的时域分支，不是已经带频率维的时频图。

所以：

1. `Sec2_4_preprocess_tfa.m` 只是 ROI 裁剪，不做 Morlet。
2. `Sec3_4_all_roi_result_tfa.py` 只是对这个 1-150 Hz 分支做 decoding。
3. 如果你要真正的 trial-level time-frequency map，请使用 `Sec3_s2_roi_condition_tfa.py`。

这一点一定要和论文/汇报中的术语保持一致，避免把 `Sec3_4` 的结果误写成“真正的时频 decoding”。

### 2.4 trial 数处理方式

`Sec1_preanalyse.m` 会在每个 task 内把所有条件截断到同一个 `min_trials`，然后才保存到 `epoch.data`。

这意味着：

1. 条件间的试次数被强制对齐了。
2. 如果某些条件原本 trial 更多，多出来的部分会被丢弃。
3. 当前实现是保留每个条件最前面的 `min_trials` 个 trial，而不是随机抽样。

如果 trial 顺序本身带有时间漂移、疲劳或学习效应，这会引入系统性偏差。当前代码里没有随机重采样步骤，解释结果时要记住这一点。

---

## 3. ROI 分组规则

### 3.1 当前实现

ROI 分组入口统一是 `get_roi_map(loc_file, common_channels)`。

`Sec2_1/2/3/4/5/6` 的逻辑是一致的：

1. 先找 task1/task2/task3 三者公共通道。
2. 再读取 `processed_data/{subject}/{subject}_ieegloc.xlsx`。
3. 用 `AAL3` 或可替代 ROI 列把通道映射到 ROI。
4. 每个 ROI 输出一个 `.mat` 文件。

### 3.2 这个映射不是互斥的

`get_roi_map.m` 的实现允许一个通道进入多个 ROI。原因是它会读取 `ieegloc.xlsx` 中所有同名行，并把该通道加入所有命中的 ROI。

在 `test003` 上我实际检查到：

1. 共 129 个通道。
2. 其中 21 个通道被映射到多个 ROI。

典型例子：

1. `G3/G4` 同时属于 `Color_with_sti` 和 `ParaHippocampal_R`
2. `H2/H3/H4/H5/H11` 同时属于 `Color_patch`、`Color_with_sti` 和解剖 ROI
3. 多个 `B* / C* / H*` 通道同时属于 `Color_patch` 和 `Temporal_Mid_R`

因此当前 ROI 结果的解释应该是：

1. `Color_patch`、`Color_with_sti` 之类功能 ROI 与解剖 ROI 不是彼此独立的分析单元。
2. 不同 ROI 的统计显著不能直接当作“独立脑区重复验证”。
3. 如果你后面要做 ROI 间比较，必须先决定是否接受这种 many-to-many 映射。

如果你的目标是“每个通道只能进一个 ROI”，那就需要重写 ROI 映射规则，而不是继续沿用当前实现。

---

## 4. Sec2 特征提取脚本

### 4.1 `Sec2_1_preprocess_erp.m`

输入：`task*_ERP_epoched.mat`

输出：`erp_task1|2|3`

数据形状：`[Cond, Rep, Ch, Time]`

行为：只裁剪 ROI 通道和时间窗，不再做额外滤波或基线校正。

### 4.2 `Sec2_2_preprocess_highgamma.m`

输入：`task*_TFA_epoched.mat`

输出：`hg_task1|2|3`

行为：从 1-150 Hz 分支中再次做 `70-150 Hz` 带通、Hilbert 包络、平方功率、时间平滑，然后裁剪到 `-100 ~ 1000 ms`。

### 4.3 `Sec2_3_preprocess_lowgamma.m`

输入：`task*_TFA_epoched.mat`

输出：`lg_task1|2|3`

行为：和 high gamma 同样的处理，但频段是 `30-70 Hz`。

### 4.4 `Sec2_4_preprocess_tfa.m`

输入：`task*_TFA_epoched.mat`

输出：`tfa_task1|2|3`，以及 `tfa_roi_channels`、`tfa_time_ms`

关键说明：它不产生频率维，不是 Morlet 结果，只是 ROI 层面的 1-150 Hz 宽频时域输入。

### 4.5 `Sec2_5_preprocess_gamma.m`

输入：`task*_TFA_epoched.mat`

输出：`g_task1|2|3`

行为：提取 `30-100 Hz` 宽频 gamma 包络。

当前状态：

1. 仍然是独立脚本，未接入 `newanalyse_load_run_config.m`。
2. 默认 subject 是硬编码。
3. `Sec4_2_batch_run_modalities.py` 当前不会自动调用它。

如果要把 gamma 主线纳入统一批处理，需要先把这个脚本和 `Sec3_5_all_roi_result_gamma.py` 一起做成和其他模态相同的 runtime-config 接口。

### 4.6 `Sec2_6_preprocess_gamma_multiband.m`

输入：`task*_TFA_epoched.mat`

输出：`gmb_task1|2|3`，以及：

1. `gmb_roi_channels`
2. `gmb_band_names`
3. `gmb_band_ranges`
4. `gmb_feature_labels`
5. `gmb_feature_band_index`
6. `gmb_feature_channel_index`
7. `gmb_time_ms`

这是当前最完整、最自描述的 feature 文件格式。后续所有 band/channel 级统计都依赖这些元数据，不能删。

---

## 5. Sec3 主 decoding 脚本

### 5.1 已核实的 decoding 输入方式

以 `Sec3_1_all_roi_result_erp.py` 为例，主 decoding 逻辑是：

1. 先从 ROI `.mat` 中取出某个条件的 `data[cond, :, :, :]`。
2. 得到的单条件形状是 `[Rep, Ch, Time]`。
3. 再把多个条件拼到 trial 轴上，形成 `[Trial, Ch, Time]`。
4. 在每个时间点上用 `StandardScaler + LDA(shrinkage='auto')` 做分类。

这个 trial 轴处理方式与 feature 文件真实形状是一致的。

### 5.2 当前默认 active task

大部分主 decoding 脚本当前默认只打开了 task1 的 `within_category_color_gray` 任务，其他任务多数保留在注释里。

所以实际运行前要先看清楚每个 `TASKS` 里真正启用的是哪一项，而不是只看文件名。

### 5.3 模态间显著性窗口不一致

ERP 与 TFA 主 decoding 的 `cluster_permutation_significance()` 会显式屏蔽 `20 ms` 以前的时间点。

但 high gamma、low gamma、gamma、gamma multiband 对应函数目前默认不会做这个屏蔽。

这会直接导致：

1. 不同模态的 earliest latency 不可直接横向比较。
2. 某些高 gamma / low gamma 结果会在刺激前就出现显著 cluster。

我用现有结果文件扫描过，确实发现：

1. high gamma 存在 earliest latency = `-100 ms` 的结果。
2. low gamma 也存在 earliest latency = `-100 ms` 的结果。

这不是数值异常，而是代码定义不一致导致的结果解释问题。

### 5.4 `Sec3_4_all_roi_result_tfa.py` 的解释边界

虽然脚本名叫 `tfa`，但它吃进去的是 `tfa_task*` 这种没有频率维的 ROI 时域数据。

所以它更准确的解释应该是：

1. “1-150 Hz 宽频分支 decoding”
2. 或者“宽频 TFA branch decoding”

不应该直接写成“time-frequency decoding”。

### 5.5 `Sec3_5_all_roi_result_gamma.py`

这个脚本当前仍是老结构：

1. 没接入 `runtime_config.py`
2. 输出路径仍然写死为 `perm1000`
3. 不在 `Sec4_2_batch_run_modalities.py` 的调度列表里

因此它可以单独跑，但不能算已经纳入当前统一批处理框架。

### 5.6 `Sec3_7_all_roi_result_cross.py`

这是旧式 cross decoding 脚本，和当前 `Sec3_1/2/3/4/6` 的 runtime-config 体系不一致。保留可用，但更像独立实验脚本，不是当前主批处理的一部分。

---

## 6. ROI 内条件比较与补充分析

### 6.1 `Sec3_s1_roi_electrode_condition_erp_stats.py`

用途：在 ROI 内按电极做时序条件比较，支持 `erp`、`highgamma`、`lowgamma`。

当前已修正一个明确 bug：

1. `lowgamma` 现在使用正确的字段前缀 `lg_`。

### 6.2 `Sec3_s2_roi_condition_tfa.py`

这是当前目录中真正执行 trial-level Morlet 分解的脚本。

流程：

1. 从 `feature/tfa/{subject}` 读取 ROI 时域数据。
2. 根据 `GROUP_A`、`GROUP_B` 取出 trial。
3. 对每个 channel、每个 trial 做 Morlet 功率计算。
4. 做基线校正。
5. 输出 ROI 图、channel 图和 `.mat`。

如果你想做真正的时频条件差异，这个脚本才是关键入口。

### 6.3 `Sec3_s3_roi_electrode_condition_gamma_multiband_stats.py`

它会把 `gmb_task*` 的 feature 维重新还原成 `Band × Channel`，然后对每个 channel 的 band-time 图做条件比较。

这个脚本依赖 `gmb_feature_band_index` 和 `gmb_feature_channel_index`，因此 multiband gamma 文件必须保持完整元数据。

### 6.4 `Sec3_s4_all_electrode_decoding_importance.py`

这个脚本不是 ROI 平均，而是把所有 ROI 文件重新拼回全电极特征，然后做全电极 decoding 和 leave-one-electrode-out importance。

适合回答“哪个电极最重要”，不适合回答“哪个 ROI 独立显著”。

---

## 7. 批处理与运行配置

### 7.1 `Sec4_2_batch_run_modalities.py`

这是当前主批处理入口。

它目前已经接入的模态是：

1. ERP
2. High gamma
3. Low gamma
4. TFA branch
5. Gamma multiband

当前没有接入：

1. Gamma 宽频主线
2. `Sec1_preanalyse.m`

也就是说，`Sec4_2` 不是全流程总入口，而是 `Sec2 + Sec3` 的主入口。

### 7.2 runtime config 机制

MATLAB 脚本通过：

1. `NEWANALYSE_USE_CONFIG`
2. `NEWANALYSE_CONFIG_PATH`
3. `newanalyse_load_run_config.m`

Python 脚本通过：

1. `runtime_config.py`
2. `batch_runner_utils.py`

共同完成 subject 和超参数覆写。

示例 JSON 在 `sec4_batch_config_example.json`。

### 7.3 当前已修正的批处理脚本引用错误

这次 review 中修正了两个确定性路径错误：

1. `Sec4_1_batch_roi_condition_stats_report.py` 现在正确调用 `Sec3_s1_roi_electrode_condition_erp_stats.py`
2. `Sec4_3_batch_roi_condition_tfa.py` 现在正确调用 `Sec3_s2_roi_condition_tfa.py`

如果你之前直接运行过这两个脚本但失败，原因就是它们原本指向了不存在的文件名。

### 7.4 `Sec4_4_batch_compare_time_smoothing.py`

这个脚本会对多个 smoothing window 重复跑主 decoding，再比较曲线差异。

它当前支持：

1. ERP
2. High gamma
3. Low gamma
4. TFA branch
5. Gamma multiband

不支持 gamma 宽频主线。

---

## 8. Sec5 脚本状态

### 8.1 `Sec5_1_replot_all_roi_result_erp.py`

这个脚本更像旧结果目录的重绘器，默认假设结果路径仍然是：

1. `perm1000/erp/...`
2. `perm1000/highgamma/...`
3. `perm1000/lowgamma/...`

而不是当前 `Sec4_2` 常见的 `perm200/with_sti/...` 或 `real_only/with_sti/...`。

所以它是可参考的 legacy 工具，不是当前最稳的结果重绘入口。

### 8.2 `Sec5_2_merge_new_md_to_html.py`

这个脚本依赖 `Sec5_1` 生成的 markdown 汇总结果，并把 ROI 坐标映射到 AAL atlas 做 HTML 展示。

同样属于较旧的汇总路径设计，使用前先确认输入目录结构和任务名是否仍然匹配当前输出。

---

## 9. 当前代码 review 的关键结论

### 9.1 可以确认正确的部分

1. ROI feature 文件的主数据形状是统一的，主 decoding 对 trial 轴的读取方式没有写反。
2. ERP / high gamma / low gamma / gamma multiband 的 ROI 提取逻辑彼此一致，都是先找公共通道，再按 ROI 保存。
3. `Sec3_s2_roi_condition_tfa.py` 对“真正时频分析”的定位是合理的，因为它确实在 Python 端重新做了 Morlet。

### 9.2 需要你在解释结果时特别小心的部分

1. `tfa` 主线名称会误导，因为 `Sec3_4` 不是频率维 decoding。
2. ROI 映射是 many-to-many，不是独立分组。
3. `Sec1_preanalyse.m` 的 min-trial 截断会丢弃多余 trial，而且不是随机抽样。
4. 不同模态的显著性时间窗定义还不统一。

### 9.3 当前已知但不在这次直接修复范围内的工程问题

1. `Sec1_preanalyse.m` 仍是手工入口，顶部路径是 Windows 本地路径，不是当前 Linux 工作区可直接运行的配置化脚本。
2. `Sec2_5_preprocess_gamma.m` 和 `Sec3_5_all_roi_result_gamma.py` 仍是独立旧结构，没有纳入 `Sec4_2`。
3. `Sec5_*` 仍然偏向历史输出目录结构。

---

## 10. 建议的实际使用方式

如果你当前目标是稳定地产生可解释的主结果，建议优先走下面两条线：

### 10.1 主 ROI decoding 线

1. 用 `Sec2_1/2/3/4/6` 生成 ROI 特征。
2. 用 `Sec3_1/2/3/4/6` 做主 decoding。
3. 用 `Sec4_2_batch_run_modalities.py` 做批处理。

### 10.2 真正的 ROI 时频条件比较线

1. 用 `Sec2_4_preprocess_tfa.m` 生成 ROI 时域输入。
2. 用 `Sec3_s2_roi_condition_tfa.py` 做 Morlet 分解与条件比较。
3. 需要批量跑时用 `Sec4_3_batch_roi_condition_tfa.py`。

如果你的核心科学问题是“某个 ROI 是否存在真正的频率特异性差异”，不要只看 `Sec3_4_all_roi_result_tfa.py`，而应优先看 `Sec3_s2_roi_condition_tfa.py`。