# 双信号记忆电极聚类、颞极真假差异及 color_with_sti 全套解码分析实施计划 (Step 5, Step 6 & Step 7)

此计划针对用户新提出的三项核心分析任务进行设计，分析均只在 **ERP 信号** 上进行。

---

## 📌 用户审查与设计决策

> [!IMPORTANT]
> ### 1. memory_color 电极聚类边界与被试参与情况
> 经 MNI_Y 轴（前后轴）坐标分析，31 个 `memory_color` 显著记忆电极的 $Y$ 坐标分布在 $[-90.3, 2.4]$ mm 区间。我们采用 1D K-Means 聚类，可完美将其划分为两个 Cluster：
> - **Posterior Cluster (后部)**：$Y \in [-90.3, -42.3]$ mm，共 25 个电极。被试电极分布为：`test001` (4个)、`test002` (12个)、`test003` (9个)。
>   - **解码策略**：全部 3 名被试参与，采用以被试为随机截距的二项分布广义线性混合效应模型（GLMM）进行组水平显著检验。
> - **Anterior Cluster (前部)**：$Y \in [-14.1, 2.4]$ mm，共 6 个电极。被试电极分布为：`test001` (4个)、`test002` (2个)、`test003` (**0个**)。
>   - **解码策略**：`test003` 因无电极不参与此 cluster 的解码。`test001` 和 `test002` 参与，采用以被试为随机截距的 GLMM 检验（2名被试的数据足以拟合二项 GLMM，若遇奇异性则降级为固定效应检验）。
>
> ### 2. 单电极真/假颜色 (True/Fake) 差异分析的设计细节
> 针对 `temporal_pole` 的单电极（`test001` 2个；`test002` 8个；`test003` 0个），在 Task 2（有色水果刺激）中提取 **真颜色呈现 (True Color)**（Trigger 101, 111, 121, 131） vs **假颜色呈现 (Fake Color)**（Trigger 102, 112, 122, 132）的 trials。
> - 统计标准：
>   1) 逐时间点进行 Wilcoxon 秩和检验，检测是否存在 $\ge 50$ms 连续显著窗口（$p < 0.05$）；
>   2) 对 **`200-500ms`** 均值进行 Wilcoxon 秩和检验。
> - 我们将为每个电极绘制 ERP 时程差异及 $200\text{--}500$ms 平均值箱线图，并汇总成统计表。
>
> ### 3. color_with_sti 电极集定义与去重
> 经读取被试定位 xlsx 中的 AAL3 列匹配（不区分大小写及连字符），我们提取出：
> - `test001`: `['D4', 'D5', 'D6', 'G5', 'G6', 'G7']` (6个)
> - `test002`: `['D1', 'D2', 'D3', 'B2']` (4个)
> - `test003`: `['G3', 'G4', 'H2', 'H3', 'H4', 'H5', 'H11', 'H12']` (去重后共 8 个)
> 我们将严格使用这组电极子集，开展 5 项全套分析。

---

## 🙋 开放问题
1. **Color vs Gray 条件定义**：在 Step 7_1 中进行“单电极的不同类别 color，gray 的信号差别”分析，我们建议使用 Task 2 中“所有有色呈现水果的 trials（Color）” vs “所有灰色呈现水果的 trials（Gray）”，以在相同的物体概念下纯粹探究物理颜色呈现与非呈现的 ERP 差异。您是否同意此定义？
2. **红绿 Cross Decoding 的 2D 与 1D 展示**：在 Step 7_4 中，对于“红绿训练，灰色图片测试的 cross decoding”，我们建议**同时**进行 2D 时间泛化热图（下采样 10ms 步长，展示时间泛化规律）与对角线 $T_{train} = T_{test}$ 的 1D 时程曲线计算（并在 1D 上拟合 GLMM 定位组水平显著时间窗），这样可以兼顾 2D 时间泛化物理图景与 1D 显著时区定位。您是否同意此方案？

---

## 🛠️ 拟引入的修改与方案步骤

### 1. [Component: Step 5] memory_color 电极聚类解码 (`step5_memory_color_clusters_decoding.py`)
- **聚类规则**：利用 $Y$ 轴坐标中位数（$Y \approx -30$mm）将 31 个 ERP `memory_color` 电极划分为 Posterior ($Y \le -30$mm) 与 Anterior ($Y > -30$mm) 两组。
- **解码任务**：对两个 Cluster 分别进行：
  - **Memory Color Decoding**：红色记忆 vs 绿色记忆，测试于灰色水果（4种水果条件配对平均）。
  - **True vs Fake Color Decoding**：4折 Leave-One-Group-Out 跨物体真/假颜色解码。
- **统计与绘图**：
  - 绘制 4 张解码曲线大图（2个 cluster $\times$ 2个解码任务），包含个体曲线、Group 均值、GLMM 显著时间阴影（$p < 0.05$ 且 $\ge 20$ms）。
  - 导出数据 Excel 到 `analyse_0617/doc/`，图片到 `result/select_channel/decoding/clusters/`。

### 2. [Component: Step 6] 颞极单电极真假颜色差异分析 (`step6_temporal_pole_true_fake_erp_difference.py`)
- **电极范围**：`temporal_pole` 电极子集（test001: E6, E7; test002: G1~G8; test003: 无）。
- **对比条件**：Task 2 中有色水果刺激的 True Color（Trigger 101, 111, 121, 131） vs Fake Color（Trigger 102, 112, 122, 132）。
- **统计指标**：逐时间点 Wilcoxon 检验（$\ge 50$ms 显著段）以及 `200-500ms` 均值 Wilcoxon 检验。
- **结果绘图**：对 10 个电极的每一个，绘制 1行2列 复合图：
  - 左子图：True vs Fake ERP 时程曲线（均值 $\pm$ SEM，标注连续显著阴影）；
  - 右子图：200-500ms 均值比较的箱线图。
- **统计报告**：生成并保存详细的显著性统计 Excel 表。

### 3. [Component: Step 7] color_with_sti 电极集全套分析 (`step7_color_with_sti_electrode_analyses.py`)
针对去重后的 `color_with_sti` 电极集（3被试均有电极，共 18 个），顺序执行并完成 5 个子任务：
1. **7_1. Color vs Gray 信号差异**：有色水果 (Color) vs 灰色水果 (Gray) 单通道 ERP 差异（显著连续段及 200-500ms 均值检验，绘制 18 张单电极 1行2列 差异图并导出统计表）。
2. **7_2. Memory Color Decoding**：红色记忆 vs 绿色记忆（灰色测试），多电极 SVM 解码，GLMM 显著检验，绘制 1D 曲线图。
3. **7_3. Color Block Decoding**：红 (51) vs 绿 (54) 色块多电极 SVM 解码，GLMM 显著检验，绘制 1D 曲线图。
4. **7_4. Cross Decoding**：红绿色块训练，灰色图片测试。
   - **2D TG**：计算 [-100, 700]ms 10ms 步长的 2D 时间泛化热图，绘制 Group 均值及 3 被试热图；
   - **1D 对角线**：提取对角线时程，进行多被试 GLMM 显著性检验并画图。
5. **7_5. True vs Fake Decoding**：真假颜色多电极 SVM 解码（4折 LOGO 跨物体），GLMM 显著检验，绘制 1D 曲线图。
- **输出成果**：
  - 数据：保存上述所有解码与统计结果至 `analyse_0617/doc/` 的对应 Excel/CSV 文件。
  - 图片：分类保存至 `result/select_channel/decoding/color_with_sti/` 目录下。

---

## 🧪 验证与备份方案

### 1. 自动与逻辑校验
- 每次 SVM 解码均要使用 trial-wise 基线扣除，并使用强正则 `SVC(C=0.1, kernel='linear')`。
- 在拟合 GLMM 时若只有一个被试有数据，则自动降级为单被试二项检验。
- 确认 `test003` 在前部聚类和颞极分析中由于无电极而干净地跳过，并给出日志输出。

### 2. 物理备份与 Walkthrough 更新
- 计算完成后，运行自动备份命令，将所有新生成的数据表、图表以及执行代码打包并备份至 `analyse_0617/plan_backups/`。
- 更新 `walkthrough.md`，把聚类解码窗口、单通道真假差异统计、以及 `color_with_sti` 全套解码的结论与图片详细写入。
