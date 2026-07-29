# 单电极记忆解码时序显著点与前后脑解剖相关性分析计划 (Step2_3)

本计划旨在实现 `step2_3_single_electrode_decoding_correlation.py` 脚本，对 Step 2_1 筛选出的记忆颜色显著通道（`memory_color` 通道）分别独立进行多试次 SVM 交叉解码。通过二项检验定位各电极在刺激呈现 80ms 后的最早显著时间点（ESTP），将其作为潜伏期指数与电极颅内前后位置（MNI Y 坐标）进行空间相关分析，并基于 MNI Y 坐标由后脑到前脑由蓝到红渐变绘制时Course解码折线图与相关散点图。

## User Review Required

> [!IMPORTANT]
> **分析与统计学方案设计**：
> 1. **靶区通道加载**：
>    - 载入 `select_channel_memory_significance_erp.csv`（ERP）与 `hg.csv`（HG），提取 `Is_Mean_Sig == True` 或 `Is_Cont_Sig == True` 的电极作为分析对象。
> 2. **单电极分类解码**：
>    - 针对每个电极独立作为一维特征（$D=1$），开展红绿记忆 4 种交叉配对分类解码，保存 750 个时间点的平均准确率序列。数据加载时应用 Trial-wise 基线减除对齐。
> 3. **80ms 后最早显著时间点 (ESTP)**：
>    - **检验方法**：在每个时间点上，单电极有 $N_{\text{test}}$ 个测试 trials（合并 4 个配对所有的测试 trials，每个被试约 150 - 200 个样本）。每个试次对应 1/0 的预测正确或错误，我们使用精确的**单尾二项检验 (Binomial Test)**，检验成功概率是否显著大于 0.5：
>      $$p\_val = \text{BinomialTest}(k = \sum \text{correct\_trials}, n = N_{\text{test}}, p = 0.5, \text{alternative='greater'})$$
>    - **提取指数**：定位所有 $t \ge 80$ms 且 $p\_val < 0.05$ 的时间点，返回其中最早的那个时间点 `time_ms[t]` 作为 ESTP 指数（代表该通道在颅内最早能解码出物体记忆颜色知识的潜伏期）。
>    - **剔除不显著电极**：如果某通道在 80ms 后无任何时间点通过二项检验，则其 ESTP 记为 `None`，不参与相关性计算。
> 4. **空间相关性分析 (Correlation)**：
>    - 对提取出的有效 ESTP 值与各电极 MNI Y 坐标进行 **Pearson 与 Spearman 秩相关** 计算，求出 $r$ 与 $p$ 值，拟合线性趋势线。
>    - 分析分为两个层级：
>      1. **组水平 (Group-level)**：合并 3 个被试的所有通道进行计算与绘图。
>      2. **个体水平 (Subject-level)**：对 3 个被试各自通道独立进行计算与绘图。
> 5. **绘图规格 (左折线，右相关)**：
>    - 针对 ERP 与 HG 各自绘制 4 张 1行2列 的多子图大图（合计 8 张大图：组水平 1 张，被试水平 3 张）。
>    - **左子图（时程曲线图）**：
>      - X 轴 `[-200, 800]` ms。
>      - 每一条细线代表一个电极通道。
>      - 曲线颜色基于电极的 MNI Y 坐标通过 `coolwarm` 颜色图做渐变上色（越负/后脑越偏**蓝色**，越正/前脑越偏**红色**）。
>      - 包含一条黑色粗实线代表所有通道的群体平均。
>    - **右子图（相关散点图）**：
>      - X 轴为 MNI Y 坐标，Y 轴为 ESTP 最早显著时间点。
>      - 每一个圆点代表一个通道，点颜色与左图中电极颜色对齐。
>      - 绘制线性趋势拟合直线（95% 置信区间阴影），并以文本框标注 Pearson 和 Spearman 的 $r$ 值与 $p$ 值。
> 6. **数据保存**：在绘图前，将每个电极的 MNI Y 坐标、计算得到的 ESTP 等明细数据保存为 Excel / CSV 格式存放于 `analyse_0617/doc/`。

---

## Open Questions

> [!NOTE]
> 1. **单电极二项检验的合理性**：
>    对于 Trial 级别的对/错二分类结果（1和0），单尾二项检验是检验准确率是否显著大于随机机会（50%）在数理上最精确、最无参数假设的统计方案。我们认为该方案最能保证单通道显著时间定位的科学性。
> 2. **ESTP 大于 80ms 的边界**：
>    为了过滤由于刺激呈现前基线漂移、视觉传入潜伏期（通常早期视觉反应在 50-80ms 以后才发生）产生的虚假激活噪声，我们将最早显著时间点提取限制在刺激呈现 $80$ ms 以后（$t \ge 80$ms）。这符合视觉神经生理学的一般规律。

---

## Proposed Changes

### 单电极解码与空间相关分析脚本

#### [NEW] [step2_3_single_electrode_decoding_correlation.py](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step2_3_single_electrode_decoding_correlation.py)
1. **显著通道名单载入**：
   - 读取 Step 2_1 的显著表，确认每个被试符合条件的 `memory_color` 通道列表。
2. **单电极 SVM 解码**：
   - 对每个通道独立运行 Joblib 并行加速的 4 折交叉验证 SVM 解码，收集 750 个时间点上的正确率和 Trial 对错向量。
3. **时域二项检验与 ESTP 提取**：
   - 编写兼容版二项检验算法，提取 $t \ge 80$ ms 且单尾 $p < 0.05$ 的最早显著时间点。
4. **前后脑相关性计算与保存**：
   - 在 Group-level 和被试个体水平计算 MNI Y 与 ESTP 间的 Pearson 与 Spearman 秩相关。
   - 提前保存数据表格为 `analyse_0617/doc/select_channel_memory_decoding_estp_[erp/hg].xlsx` & `.csv`。
5. **渐变折线+相关散点大图绘制**：
   - 绘制 ERP 和 HG 信号下的组大图以及各被试的大图（一共 2 x 4 = 8 张），保存于结果目录 `result/select_channel/decoding/single_electrode/`。
6. **历史备份**：
   - 拷贝本实施计划至 `analyse_0617/plan_backups/implementation_plan_step2_3.md`。

---

## Verification Plan

### Automated Tests
- 运行 `/home/lirui/anaconda3/envs/lr2026/bin/python color_cognition_pipeline/analyse_0617/code/step2_3_single_electrode_decoding_correlation.py`。
- 确认在 `analyse_0617/doc/` 下生成了 ERP 和 HG 的电极最早显著时间点明细表。
- 确认在 `analyse_0617/result/select_channel/decoding/single_electrode/` 下正确生成 8 张 1行2列 的渐变折线及相关拟合大图。

### Manual Verification
- 检查大图中电极折线颜色是否表现出后脑（蓝）到前脑（红）的连贯渐变，散点图的圆点颜色与折线图是否一一对应。
- 确认相关分析中标注的 Pearson 与 Spearman r 值及 p 值的数学方向性是否合理，拟合线是否正确绘制。
- 确认基线期噪声被有效压低，且 ESTP 数据点没有落在 $80$ ms 以前。
