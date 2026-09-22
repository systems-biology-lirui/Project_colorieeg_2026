# C07 纯色色块（Task 3）红绿神经解码方案设计

> **文档状态**: 方案就绪（基于 C08 实测提取数据量身定制）  
> **制定时间**: 2026-09-10  
> **适用模块**: `color_analyse_0825/matlab/C07_task3_pure_color_decoding_0825.m`  
> **核心原则**: 遵循简单可控、参数平铺、逻辑透明规范，100% MATLAB 原生实现。

---

## 一、 任务背景与核心科学问题

### 1.1 认知阶梯定位
在本项目的三大任务设计中：
- **Task 1 (Real & Gray Objects)**: 复杂自然物体的彩色与灰度呈现（包含人脸、物体、身体、场景四类，探究广义颜色知觉与高阶物体识别交互）。
- **Task 2 (Memory Fruits)**: 灰度水果刺激下的自上而下记忆颜色提取（探讨无真实物理颜色下的概念色表征）。
- **Task 3 (Passive Color Patches)**: **纯色色块被动注视任务**。去除了所有的物体轮廓、高阶语义和记忆成分，是**最纯粹的物理视知觉色彩输入**。

### 1.2 核心科学问题
1. **纯物理知觉的红绿可分性**: 大脑腹侧视通路在面对无形状、纯颜色的红与绿刺激时，能否从 SEEG 多频段振荡中解码出颜色类别？
2. **时空动力学特征**: 相比 Task 2 的记忆色解码（潜伏期较晚，通常在 200–500 ms 出现），Task 3 作为物理刺激诱发，其解码峰值与显著时间簇是否在 100–250 ms 迅速达到峰值？
3. **频段贡献消融**: 纯色块解码中，是 High-Gamma（反映局部神经元同步放电）占主导，还是低频相位/功率（Alpha/Theta 去同步化）共同协同？
4. **跨任务表征对齐基础**: Task 3 的纯物理红绿解码权重与表征空间，将作为后续跨任务泛化（Task 3 -> Task 2 记忆色迁移）的核心基石。

---

## 二、 C08 实测提取结果审计与试次基底

C08 脚本已完成对 `sub001` ~ `sub008` 全部 8 位被试的数据提取并保存在 `process_data_new/<sub_id>/task3_multiband_epoched.mat`。实测数据统计如下：

| 被试编号 | 有效 Laplacian 通道数 | 总有效非 Catch 试次 | 红色 (Red) 试次 | 绿色 (Green) 试次 | 红绿试次总计 | 数据匹配状态 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **sub001** | 82 | 360 | 60 | 60 | **120** | 100% 完美对齐 |
| **sub002** | 58 | 360 | 60 | 60 | **120** | 100% 完美对齐 |
| **sub003** | 111 | 540 | 90 | 90 | **180** | 100% 完美对齐 (3 Session) |
| **sub004** | 158 | 357 | 60 | 59 | **119** | 99.2% LCS 严格对齐 |
| **sub005** | 84 | 360 | 60 | 60 | **120** | 100% 完美对齐 |
| **sub006** | 88 | 360 | 60 | 60 | **120** | 100% 完美对齐 |
| **sub007** | 104 | 359 | 59 | 60 | **119** | 99.7% LCS 严格对齐 |
| **sub008** | 108 | 360 | 60 | 60 | **120** | 100% 完美对齐 |

**审计结论**:
- 红绿两类的试次数量在所有被试中保持严格平衡（每种颜色在标准 2 个 Session 中各约 60 次，sub003 为 90 次）。
- 每种颜色刺激均由 3 种不同的色块图片（`01`, `02`, `03`）构成，每个色块图片重复 20 次（sub003 重复 30 次）。

---

## 三、 C07 解码方案架构与数学原理

### 3.1 样本空间与标签定义
仅选取红色和绿色试次构建二分类问题：
- 红色（Red, Trigger 51）：标签 $y = 1$
- 绿色（Green, Trigger 54）：标签 $y = 0$
- 样本总数：$N \approx 120$（sub003 为 180）。

### 3.2 交叉验证设计：双模式支持

#### 模式一：跨色块 3 折交叉验证（Leave-One-Patch-Pair-Out，推荐默认）
- **科学动机**: 杜绝图片特异性边缘伪影。Red 和 Green 均包含 3 个图片版本（Patch 1, Patch 2, Patch 3）。
- **折叠划分**:
  - Fold 1: 训练集使用 Patch 1 & 2，测试集使用 Patch 3；
  - Fold 2: 训练集使用 Patch 1 & 3，测试集使用 Patch 2；
  - Fold 3: 训练集使用 Patch 2 & 3，测试集使用 Patch 1。
- **意义**: 确保分类器学习到的是普遍的“红色 vs 绿色”物理神经反应，而非特定图片的低阶轮廓特征。

#### 模式二：分层 5 折交叉验证（Stratified 5-Fold CV）
- 将红绿试次等比例随机划分为 5 等份，4 份训练，1 份测试。

### 3.3 滑动时间窗与特征张量
- **时间范围**: $[-200, 800]\text{ ms}$。
- **窗长与步长**: 窗长 $W = 20\text{ ms}$，步长 $\Delta t = 20\text{ ms}$。
- **时间中心点**:
  $$t_c \in \{-200, -180, \dots, 780, 800\}\text{ ms}, \quad N_{\text{win}} = 51$$
- **特征提取**:
  对各频段 $b \in \{\text{Delta}, \text{Theta}, \text{Alpha}, \text{Beta}, \text{Low-Gamma}, \text{High-Gamma}\}$，在第 $i$ 个试次、通道 $c$、窗口 $w$ 内：
  $$X_{i, b}(w) = \frac{1}{|T(w)|} \sum_{t \in T(w)} P_{\text{dB}, i, c, b}(t)$$
  其中 $T(w) = [t_c - W/2, t_c + W/2)$。
- **特征标准化**:
  严格在训练折内部计算均值 $\mu_{\text{tr}}$ 与标准差 $\sigma_{\text{tr}}$，对测试折执行相同变换：
  $$\tilde{X}_{\text{tr}} = \frac{X_{\text{tr}} - \mu_{\text{tr}}}{\sigma_{\text{tr}}}, \quad \tilde{X}_{\text{te}} = \frac{X_{\text{te}} - \mu_{\text{tr}}}{\sigma_{\text{tr}}}$$

### 3.4 分类模型与性能度量
- **分类器**: 线性岭正则化支持向量机（Linear SVM）：
  $$\min_{\mathbf{w}, b_0} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i \in \text{tr}} \max\left(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b_0)\right)$$
  MATLAB 实现: `fitclinear(..., 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', 0.01)`。
- **平衡准确率 (Balanced Accuracy)**:
  $$\text{BA} = \frac{1}{2}\left(\frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}}\right)$$
- **多频段联合与单频段消融**:
  - Multi-Band: 特征维度为 6（拼接 6 大生理频段）；
  - 单频段: 特征维度为 1，独立评估每个频段的解码能力。

### 3.5 统计置换检验与基于时间簇的多重比较校正 (Cluster-based Permutation Test)
1. **置换原假设**: 试次标签 $y$ 与神经特征张量相互独立。
2. **置换过程**: 打乱标签 $y$，重复执行完整的交叉验证与解码流程 $N_{\text{perm}} = 200$ 次，获得经验零假设分布。
3. **时程平滑**: 真实曲线与置换曲线均采用 5 点高斯平滑 (`smoothdata(..., 'gaussian', 5)`)。
4. **单点 p 值**:
   $$p_{\text{pt}}(t) = \frac{1 + \sum_{k=1}^{N_{\text{perm}}} \mathbb{I}\left(\text{BA}_{\text{null}, k}(t) \ge \text{BA}_{\text{real}}(t)\right)}{1 + N_{\text{perm}}}$$
5. **时间簇质量检验 (Cluster-mass Test)**:
   - 提取 $t \ge 0\text{ ms}$ 且 $p_{\text{pt}} < 0.05$ 的连续时间段构成的簇 $C_j$；
   - 计算各簇质量: $M_j = \sum_{t \in C_j} (\text{BA}(t) - 0.5)$；
   - 提取零分布下各次置换的最大簇质量 $M_{\text{null}, k}^{\max}$；
   - 簇级别校正 p 值:
     $$p_{\text{cluster}, j} = \frac{1 + \sum_{k=1}^{N_{\text{perm}}} \mathbb{I}\left(M_{\text{null}, k}^{\max} \ge M_j\right)}{1 + N_{\text{perm}}}$$
   - 若 $p_{\text{cluster}, j} < 0.05$，则该时间簇具有统计显著性。

---

## 四、 目标电极筛选与分析范围策略

为兼顾计算效力与科研针对性，C07 提供清晰的 `cfg.target_mode` 参数选择：

1. **`c04_concordant` (默认推荐)**:
   - 读取 C04 导出的 `color_effects_summary.csv`，仅对 Task 1 中**四类别同向显著**的 157 个核心颜色电极进行解码。
2. **`c04_all_sig`**:
   - 对 Task 1 中所有显著电极（共 229 个，包含同向与不同向显著）进行解码。
3. **`custom` (定制重点通道模式)**:
   - 针对重点电极（如 `sub001-G13`, `sub007-C4` 等典型电极）快速出图验证。
4. **`all_laplacian`**:
   - 全脑所有接触点全量遍历。

---

## 五、 学术可视化与输出规范 (遵循用户既定规则)

严格恪守用户此前在 C05 与 C07 中确认的视觉标准：
1. **排版**: 1:1 双子图并列 (`[1200, 480]`)。
2. **左子图**:
   - 解码准确率时程曲线（$-200\sim 800\text{ ms}$）。
   - 严禁添加背景方格 (`grid off`)。
   - 浅灰阴影区域标定置换检验零分布 95% 置信带。
   - 显著时间簇标记为透明金黄底色，显著时间点以陶土橙褐色小方块打底。
   - 6 条细彩色折线代表 6 个单频段，1 条加粗陶土橙褐色粗折线代表 Multi-Band。
   - 图例仅包含频段标签（`Multi-Band`, `Delta`, `Theta`, `Alpha`, `Beta`, `Low-Gamma`, `High-Gamma`）。
3. **右子图**:
   - 各频段峰值准确率对比柱状图 (`grid off`)。
   - 柱顶清晰标明峰值百分比（如 `78.5%`）。
4. **主标题**:
   - 严格只包含被试与通道，无冗余说明（如: `sub001 - G13`）。
5. **文件归档路径**:
   - 图表目录: `color_analyse_0825/result/figures/decoding_task3_purecolor/`
   - 单通道时程 CSV: `color_analyse_0825/result/tables/decoding_task3_purecolor_timecourses/`
   - 总汇总表: `color_analyse_0825/result/tables/task3_purecolor_decoding_summary.csv`

---

## 六、 脚本代码编写架构规范

按照用户的核心原则，C07 脚本将保持极其平铺、简洁、可控的结构：
```matlab
%% 1. 主参数配置 (置顶直观，简写平铺)
cfg = struct();
cfg.target_mode   = 'c04_concordant'; % 'c04_concordant', 'c04_all_sig', 'custom'
cfg.cv_mode       = 'leave_patch_out';% 'leave_patch_out' 或 'kfold'
cfg.win_len       = 20;               % 滑动窗长 20ms
cfg.win_step      = 20;               % 滑动步长 20ms
cfg.t_range       = [-200, 800];      % 时程范围 ms
cfg.smooth_pts    = 5;                % 高斯平滑点数
cfg.n_perm        = 200;              % 置换检验次数
cfg.n_workers     = 16;               % 并行 Worker 数量
...
```
整个脚本模块划分清晰：
- 模块 1: 参数配置与目录检查
- 模块 2: 目标通道加载与筛选
- 模块 3: 启动并行池
- 模块 4: 按被试单次载入数据并执行逐通道 SVM 解码与置换检验
- 模块 5: 汇总表导出与出图
