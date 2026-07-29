# Step 3 & Step 4: 纯色与记忆颜色 Cross Decoding 时间泛化及真假水果颜色多 ROI 解码实施计划

此实施计划针对 `step3`（纯色解码、Cross Decoding 时间泛化及单电极推广）与 `step4`（真假水果跨物体颜色解码及多 ROI 脑区对比）提出具体技术方案。**所有后续计算只在 ERP 信号上进行。**

---

## 📌 用户审查与设计决策

> [!IMPORTANT]
> ### 1. ROI 脑区电极分布的客观限制与降级统计方案
> 经程序初步检测，3 名被试植入的颅内电极对以下五个 ROI 脑区的覆盖范围存在高度差异（部分被试在某些 ROI 中电极数为 0）：
> - **`temporal_pole` (颞极)**: `test001` (2 个), `test002` (8 个), `test003` (0 个)
> - **`temporal_mid` (颞中回)**: `test001` (35 个), `test002` (7 个), `test003` (40 个)
> - **`temporal_inf` (颞下回)**: `test001` (7 个), `test002` (11 个), `test003` (0 个)
> - **`amygdala` (杏仁核)**: `test001` (4 个), `test002` (0 个), `test003` (0 个)
> - **`memory_color` (记忆显著电极)**: `test001` (8 个), `test002` (14 个), `test003` (9 个)
> 
> **应对方案**：对于只有 1 个被试拥有电极的 ROI（如 `amygdala` 只有 `test001` 覆盖），在进行组水平显著性检验时，将**无法运行**以“被试”为随机截距的二项分布广义线性混合效应模型（GLMM，会导致方差不收敛）。我们将在此类 ROI 上降级为**单被试二项检验 (Binomial Test)**；对于有 2 个及以上被试覆盖的 ROI（如其他 4 个），我们将如期运行以被试为随机截距的 GLMM。
> 
> ### 2. 时间泛化 (Temporal Generalization) 的重采样时效优化
> 如果直接在原始采样率（500 Hz，约 750 个时间点）上进行完整的时间泛化二维矩阵计算，矩阵大小将达到 $750 \times 750 = 562,500$ 个时间格子。对每个电极的每种策略、每折交叉验证都进行此高维计算，将导致严重的时效拖累或内存溢出。
> **应对方案**：我们将在计算二维时间泛化矩阵前，对 $[-100, 700]$ ms 区间内的 ERP 信号进行 **10 ms 时间步长均值重采样**（即每 5 个相邻点取均值，共 80 个时间点，矩阵大小为 $80 \times 80 = 6,400$ 像素）。这样不仅可以极大地消除高频电噪对时间泛化二维模式的干扰，还能将单个电极的计算时间缩短在数秒内，保证 31 个电极的全部计算与绘图可在几分钟内并行完成。

---

## 🙋 开放问题
1. **时间泛化的平滑度与参数**：我们推荐使用 10ms 作为时间泛化（TG）重采样的基本分辨率，并使用 `clim = [0.45, 0.58]` 进行热图映射。您是否对分辨率或色彩范围有特殊偏好？
2. **真假水果 Cross Decoding 的折数**：在 Step 4 中，针对真假水果颜色的二分类解码（正类为呈现绿色，负类为呈现红色），我们设计了 **Leave-One-Group-Out**（留一水果组法，共 4 折，每次用三组水果的真假颜色训练，在剩下的一组上测试）的跨物体颜色解码方案。这能最大化泛化分类器的泛化边界并控制外形特征泄漏，被认为是认知解码中最科学的策略。如您有其他想法，请向我们反馈。

---

## 🛠️ 拟引入的修改与方案步骤

### 1. [Component: Step3_1] 纯色刺激红绿多电极解码 (`step3_1_color_block_decoding.py`)
- **数据集**：`task3` 纯色刺激数据。
- **刺激触发**：红色色块 (`Trigger-In:51`) vs 绿色色块 (`Trigger-In:54`)。
- **电极范围**：使用各自被试所属的 `memory_color` 显著 ERP 电极（共 31 个）。
- **分类设置**：`StandardScaler` 标准化 -> 强正则 `SVC(C=0.1, kernel='linear')` -> 5 折交叉验证。
- **显著检验**：多被试二项分布 GLMM 检验（单尾 $p < 0.05$ 且连续 $\ge 20$ms 定位为显著窗并绘制阴影）。
- **结果绘图**：包含 3 个被试的解码时程虚线、组平均实线、GLMM 阴影与 50% 机会水平基准线。
- **输出数据**：数据导出到 `analyse_0617/doc/decoding_data_erp_color_block.xlsx / .csv`。
- **图片路径**：[erp_color_block_decoding.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/select_channel/decoding/erp_color_block_decoding.png)

### 2. [Component: Step3_2] 纯色与灰色水果跨任务时间泛化热图 (`step3_2_cross_decoding_generalization.py`)
- **策略 1 (Color-to-Gray-Memory)**：
  - **训练集**：Task 3 的红绿纯色色块数据 (`Trigger-In:51` vs `Trigger-In:54`)。
  - **测试集**：Task 2 的灰色水果数据（对 4 种红色/绿色灰色水果配对分别测试并求正确率矩阵的平均）。
- **策略 2 (Gray-Memory-to-Color)**：
  - **训练集**：Task 2 的灰色水果数据（用 4 种配对组合分别训练 SVM 分类器）。
  - **测试集**：Task 3 的红绿纯色色块数据。对 4 次泛化正确率矩阵求平均。
- **优化设计**：提取 $[-100, 700]$ms 区间，进行 **10ms 时间均值下采样**。
- **绘图指标**：
  - 绘制 3 被试单独的时间泛化二维热图，以及多被试平均（Group Level）热图（策略1和2各 4 张，共 8 张热图）。
  - 保存热图数据到 `analyse_0617/doc/` 的对应 xlsx/csv。
- **图片路径**：`result/select_channel/decoding/cross_decoding/`

### 3. [Component: Step3_3] 单电极时间泛化推广 (`step3_3_single_electrode_generalization.py`)
- **核心逻辑**：对 31 个 `memory_color` ERP 电极中的每一个，独立运行上述 Step 3_2 中的策略 1 和策略 2 的下采样时间泛化计算。
- **结果绘图**：每个电极生成一张 1行2列 的图（左图：策略 1 热图；右图：策略 2 热图）。
- **图片保存**：`result/select_channel/decoding/single_electrode/cross_decoding/` 目录下，文件命名为 `{Subject}_{Electrode}_cross_decoding_generalization.png`。
- **数据保存**：所有单通道的二维热图矩阵数据压缩保存至 `analyse_0617/doc/single_electrode_tg_data.npz`，便于后续读取。

### 4. [Component: Step4] 真假水果颜色跨物体多 ROI 解码 (`step4_real_fake_color_decoding.py`)
- **数据集**：`task2`（物体记忆颜色任务）的真假颜色刺激。
  - 正样本（绿色呈现）：`122` (绿草莓), `132` (绿西瓜), `101` (绿卷心菜), `111` (绿猕猴桃)
  - 负样本（红色呈现）：`121` (红草莓), `131` (红西瓜), `102` (红卷心菜), `112` (红猕猴桃)
- **解码方案**：**4 折 Leave-One-Group-Out** 跨物体分类。
  - 每次留出一组水果（如草莓组 `121`/`122`）作为测试集，用另外三组的真假颜色数据作为训练集，循环 4 次取测试集正确率的平均。
- **评估 ROI 脑区**：
  1. `temporal_pole` (仅 test001, test002 进行多电极解码，组水平 GLMM 由此 2 名被试拟合)
  2. `temporal_mid` (test001, test002, test003 进行多电极解码，组水平 GLMM 由 3 名被试拟合)
  3. `temporal_inf` (仅 test001, test002 进行多电极解码，组水平 GLMM 由此 2 名被试拟合)
  4. `amygdala` (仅 `test001` 进行多电极解码，组水平显著性由**单被试二项检验**降级替代)
  5. `memory_color` (test001, test002, test003 进行多电极解码，组水平 GLMM 由 3 被试拟合)
- **结果绘图**：针对 5 个 ROI 各绘制一张大图，包含有电极被试的解码折线、被试平均曲线、GLMM/二项检验显著阴影区间。
- **保存位置**：
  - 数据表：`analyse_0617/doc/real_fake_decoding_results_{ROI}.csv / .xlsx`
  - 图表：`result/select_channel/decoding/real_fake/` 下的 `real_fake_decoding_{ROI}.png`

---

## 🧪 验证方案

### 1. 自动检验与完整性检查
- 确保每次计算的分类器正确配置了 trial-wise 基线减除（提取基线期 $t < 0$ ms 均值并扣除）及 `SVC(C=0.1, kernel='linear')`。
- 检查每个输出文件是否存在。
- 确认生成的 5 张 ROI 真假水果解码图以及全部时间泛化热图格式和坐标轴标注正确。

### 2. 统计检验对比
- 对比不同 ROI 脑区在真假水果颜色解码上的潜伏期和峰值正确率差异，检验是否符合高级视觉区及记忆匹配区的信息流动规律。
