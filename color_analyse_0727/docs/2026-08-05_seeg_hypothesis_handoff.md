# 2026-08-05 Codex 会话交接：SEEG 假设驱动分析（历史记录）

> 目的：本文件记录 2026-08-05 会话中确立的标准、完成的分析、关键结果与未做的事项，
> 供切换模型/新会话后直接接续。任何新模型应先读 `PROJECT_HANDOFF.md` 再读本文件。

> **状态更新（2026-08-06）**：本文件中的 1–300 ms 结果和 `final_analysis_seeg_20260805` 目录均为历史记录，
> 不再是当前主分析。当前主时间窗已改为 100–400 ms，唯一主目录为
> `result/final_analysis_seeg_20260806_corrected/`；请以 `PROJECT_HANDOFF.md` 和该目录 README 为准。

## 1. 运行环境（重要）

- 主 Python（用户指定）：`C:\Users\saber_soul\.conda\envs\lr0727\python.exe`
- 该环境已装：nilearn 0.14.0、nibabel 5.4.2（安装时 sklearn 从 1.9.0 降到 1.8.0）；
  无 python-pptx（PPTX 生成会跳过）；无 tabulate（管线已改为内置 markdown 生成，不依赖）。
- 基础环境 `E:\software\Anaconda\python.exe` 也可运行管线（无 nilearn/pptx）。
- nilearn ICBM152 模板已缓存：`C:\Users\saber_soul\nilearn_data\icbm152_2009`（离线可用）。
- 模板/脑图绘制命令在 lr0727 下运行；`fetch_icbm152_2009()` 已下载，无需再联网。

## 2. 已确立的分析标准（本会话决策，勿再改回）

- **功能筛选标准 = 二因素 ANOVA**（颜色 × 类别，Type II 颜色主效应 p<0.05）→ `strategy1`；
  `strategy2` = 任一类别 Welch t 检验 p<0.05。MWU pooled / 四类各自 p 保留为信息列。
- **历史时间窗 = 1–300 ms**（完整刺激呈现期；与 0–300 ms 等价，Jaccard 0.95）。
  2026-08-06 起当前主时间窗改为 100–400 ms，详见 `PROJECT_HANDOFF.md`。
- **信号**：`lf30`（1–30 Hz padded 滤波）、`raw200`（HDF5 1–200 Hz 原样，不再二次滤波）。
- **电极集**：三任务共同中心 548；ROI 210（腹侧视觉：fusiform/temporal_inf/temporal_mid）；
  CSC = (S1 ∪ S2) ∩ N2，lf30=9、raw200=8。N2 = 距 PC/CC/AC 双侧色斑 ≤20 mm。
- **特征**：16 个 Welch 频带 log 功率（5–195 Hz，每 10 Hz，剔除 45–55/95–105/145–155），
  trial 内 baseline z（−200–0 ms）；解码前训练集 StandardScaler（两层标准化）。
- **统计**：时间分辨结论一律簇置换（组水平 5000 次符号翻转；逐电极 1000 次标签置换，仅 lf30）；
  零数据校准经验 FWER≈10%（名义 5%，已如实说明）。

## 3. 结果目录（按新旧排列）

- `result/final_analysis_seeg_20260805/`：历史正式结果（已被 2026-08-06 修正版本替代；stage01_selection、stage02_amplitude_spectral、
  stage03_decoding、stage05_hypotheses、stage06_exploration、stage04_luminance、report）。
- `result/final_analysis_anova_20260805/`：ANOVA 标准早期结果（1-300 × lf30/raw200，含 best electrode 图）。
- `result/final_analysis_20260805/`：更早版本（0-300/100-400 × lf/broadband）。
- `result/final_analysis_20260804/`：历史旧结果（勿动）。

## 4. 历史预注册假设结果（H1–H4，详见旧版 decision_summary.md）

以下数字来自 1–300 ms 的历史运行，只用于保留当时的分析轨迹；不应覆盖 100–400 ms 修正主目录中的结果。

| 假设 | lf30 | raw200 | 结论 |
|---|---|---|---|
| H1 S1 富集 | 19 vs 期望 10.5，二项 p=0.0096 | 18 vs 10.5，p=0.019 | ✅ 通过 |
| H2 物理颜色解码（Task3 红绿） | 组水平簇 40–500 ms，p=0.008 | 同 | ✅ 通过 |
| H3 记忆颜色解码（Task2 灰水果 cross-fruit） | 无显著簇 | 无显著簇 | ❌ 未通过 |
| H4 TGM（Task3→Task2）+ 位点重叠 | 无显著泛化簇；重叠 1/9 ns | 重叠 0/8 ns | ❌ 未通过 |

探索结果：
- 多电极 MVPA 复现 H2（40–500 ms，p=0.0078）。
- ERP 幅度解码对记忆颜色 170–210 ms p=0.053（接近显著，值得换特征深挖）。
- 样本重复检查（`stage06_exploration/exploration_exemplar_repeat_check.csv`）：
  Task1 每条件 70 唯一图 × 70 试次（安全）；Task3 每色仅 3 个刺激文件，而多数被试每色约 60 个 epoch；实验代码允许循环重用图片，因此平均约 20 个 epoch/图片，但当前 HDF5 未保留 trial-level 图片身份，精确重复次数和顺序尚未直接验证（H2 有样本级混淆风险；test003 每色约 90 个 epoch）；
  Task2 每水果 15 图 × 4 次（H3 用 leave-one-fruit-pair-out，已免疫）。

## 5. 代码结构（color_analyse_0727/）

- `pipeline/spectral_features.py`：padded_bandpass、welch_band_power、band_power_baseline_z、16 频带常量。
- `analysis/decoding_timeresolved.py`：STFT 滑动窗特征、单电极逐时间解码、1D/组水平簇置换。
- `analysis/hypotheses.py`：H1（ANOVA 置换+二项检验）、H2/H3（组水平+逐电极）、H4（TGM 2D 簇+超几何重叠）。
- `analysis/exploration.py`：多电极 MVPA、ERP 幅度解码、样本重复检查。
- `analysis/selection.py`（ANOVA 标准）、`csc.py`、`reporting.py`、`run_final_analysis.py`（编排，含 hypotheses/exploration 阶段）。
- 运行命令（在 color_analyse_0727 下）：
  `C:\Users\saber_soul\.conda\envs\lr0727\python.exe -m analysis.run_final_analysis
  --out result/final_analysis_seeg_20260805 --windows 1-300 --signals lf30 raw200 --perms 2000 --workers 21`

## 6. 本会话发现的历史/外部资产（后续可用）

- 历史电极记录：`color_cognition_pipeline/analyse_0617/doc/select_channel_summary.csv`、
  `whole_brain_erp_strategy_summary.csv`（历史"策略一"= 四类合并 pooled 检验，与 0727 prompt 的
  "四类各自显著"定义不同——这是早期 S1=0 争论的根源，现已用 ANOVA 口径统一）。
- 个体 fMRI（test001/003/006，待核对映射）：`E:\liulab_project\ColorLocalizer_Exp_results\P00X_*/1.data_preprocess/inside_final_stats.P00X.nii`
  （含 Color_vs_Grey GLT）；个体 PC/CC/AC 色斑中心**尚未提取**（Norm 2 升级的抓手）。
- 电刺激颜色感记录：`visual_experiment/电刺激行为学记录/seeg电刺激行为学记录.xlsx`（color_with_sti，因果臂备用）。
- 现有 7 人无行为反应记录（passive 范式）；新被试协议建议：灰水果 trial 后加红/绿二选一按键并写入 EEG 标记。

## 7. 下一步（未完成，按优先级）

1. **H2 去样本混淆**：确认/恢复 trial→图片映射（Task3 每色仅 3 图），做 exemplar-safe 交叉验证；
   拿不到映射则把 H2 表述为"颜色类别可解码（含低层差异）"。
2. **H3 换特征再攻**：ERP 幅度 170–210 ms（p=0.053）用 5000 置换确认；加 RSA（红块 vs 红记忆水果模式相似性）；
   模式级（多电极）TGM 替代单电极 TGM。
3. **新被试 + 行为臂 + 个体 fMRI 色斑**（用户已同意补充被试、合并分析）。
4. 电刺激因果分析（color_with_sti 富集/解码对比）在 SEEG 主结论稳定后接入。
5. PPTX 生成需在 lr0727 安装 python-pptx 后重跑 report 阶段。
