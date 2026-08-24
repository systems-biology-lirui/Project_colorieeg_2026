# 坏道候选表人工审核说明

## 需要编辑的文件

`bad_channel_candidates.csv` 是按“被试 × 单端通道”汇总的审核表。它包含跨 Task 1/2/3 的质量摘要；`task_coverage` 表示该通道在哪些任务中出现。

当前主表的 `signal_stage` 为 `filtered_1_200Hz_notched`：先逐通道去 DC 偏置，再重采样至 500 Hz、1–200 Hz 带通，并执行 50/100/150 Hz 陷波。原始未滤波和未去 DC 的中间 QC 已清理，当前只保留最终滤波后结果。

用户已经确认的坏道记录在 `../metadata/manual_channel_decisions.csv`，包括 test003:B15/I1/I2/I3、test004:K3、test005:G1、test007:18。

只想快速查看候选时，可使用当前滤波后的 `bad_channel_candidates_to_review_filtered.csv`；它是完整表中 `candidate_level != normal` 的子集。确认后仍请把决定写回完整的 `bad_channel_candidates.csv`。

只编辑两列：

- `manual_decision`：填写 `keep` 或 `exclude`；
- `manual_comment`：记录判断依据，例如“保留，响应形态正常”“确认持续跳变”“旧记录确认坏道”。

导出器只会把明确写成 `exclude`（以及同义值）的通道排除。空白按保留处理。保护通道 `test001:D3`、`test002:D1/D2/D3` 不允许排除。

## 建议审核顺序

1. 先看 `candidate_level` 不是 `normal` 的 47 个通道；
2. 结合 `candidate_reasons`、`robust_std`、`line_noise_relative_db` 和 `jump_fraction` 判断；
3. 打开同名 PNG 查看前 20 秒波形和 PSD；
4. 如果只是幅度大、低频漂移明显或诱发响应强，不应单独据此剔除；
5. 只有在确认持续平线、饱和、反复跳变、明显工频污染或已有可靠坏道记录时，才填写 `exclude`。

## 当前候选级别统计

| 级别 | 数量 |
|---|---:|
| normal | 783 |
| protected_review | 4 |
| review_candidate | 30 |
| high_confidence_candidate | 13 |

候选级别是机器审计提示，不是自动删除结论。审核完成后，保留修改后的 CSV，再运行 `scripts/build_hdf5.py`。
