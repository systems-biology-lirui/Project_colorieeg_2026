# Task 3 对 Task 2 跨任务跨表征神经解码方案设计 (Cross-Task Decoding Scheme)

> **项目分支**: `color_analyse_0825`  
> **文档版本**: 1.1 (针对用户关于分折机制、标准化细节与 TGM 免置换检验要求进行了精细化扩充)  
> **制定日期**: 2026-09-20  
> **核心原则**: 遵循简单可控、参数平铺、逻辑透明规范，100% MATLAB 原生实现。

---

## 一、 项目背景与任务层级构建

在 `color_analyse_0825` 子项目中，实验范式与神经加工机制呈现出清晰的三级认知阶梯：

```
+-----------------------------------------------------------------------------------+
|  Task 1 (基础感知筛选): 自然物体彩色 vs 灰度 (人脸/物体/身体/场景，各70对，总280对)  |
|  --> 核心功能: 严格配对置换检验，筛选腹侧视通路中对“颜色”具有普遍响应的特征电极        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  Task 3 (底层物理色觉): 纯色色块被动注视 (红、绿等色块，各60试次，无轮廓/语义/记忆)      |
|  --> 核心功能: 提取最纯粹的视知觉物理红绿颜色神经表征 (Bottom-up Sensory Code)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v [Cross-Decoding 跨任务迁移验证]
+-----------------------------------------------------------------------------------+
|  Task 2 (高阶记忆颜色): 水果图片 (草莓/西瓜=红记忆; 白菜/猕猴桃=绿记忆)                |
|  --> 核心功能: 在灰度状态下(无任何物理颜色)，检验是否自上而下唤醒纯色块的神经编码       |
+-----------------------------------------------------------------------------------+
```

### 1.1 核心科学问题
1. **感觉编码复用假说 (Perceptual Replay Hypothesis)**:
   当大脑看到黑白灰度的草莓或西瓜时，是否会自上而下在腹侧视觉皮层“重放”或者“唤醒”看到真实纯红色彩色块（Task 3）时的神经放电模式？
2. **彻底规避形状与类别混淆**:
   在 Task 2 内部直接解码草莓/西瓜 vs 白菜/猕猴桃，分类器可能学习的是低阶轮廓特征（圆形 vs 长条形）、表面纹理或高阶概念类别，而非颜色本身。  
   **Cross-Decoding 是认知神经科学界公认的“黄金金标准”**：用完全不含水果形状的纯色块（Task 3）训练分类器，直接在灰度水果（Task 2）上测试。若解码成功，则**100% 证明该表征必然是纯色彩编码，彻底排除了任何形状、纹理或类别的混淆**！
3. **时空动力学与时序延迟 (Temporal Dynamics)**:
   物理颜色的感知（Task 3）在早期即迅速爆发（~100–250 ms）；而记忆颜色的提取（Task 2 Gray）需要先识别物体几何轮廓再调取记忆属性，通常发生在中晚期（~250–500 ms）。

---

## 二、 数据基底与电极集定义 (严格源自 Task 1)

为彻底避免“双重浸入 (Double Dipping)”或循环分析偏差，解码电极的挑选**完全独立于 Task 2 与 Task 3**，严格基于 Task 1（C04 筛选结果）。

### 2.1 Task 1 候选电极集合划分
根据 C04 导出的 `result/tables/color_effects_summary.mat`，可灵活配置以下 5 类电极子集：

1. **`concord` (All Concordant, 默认推荐)**:
   - 包含四类别全正（Concordant Positive, 105 个）与全负（Concordant Negative, 52 个）的同向显著电极，共 157 个位点。
2. **`all_sig` (All Significant)**:
   - Task 1 全部总体显著电极（包含 72 个类别偏向电极，共 229 个位点）。
3. **`pos` (Concordant Positive)**:
   - 仅包含 ERS 正向增强电极。
4. **`neg` (Concordant Negative)**:
   - 仅包含 ERD 抑制去同步化电极。
5. **`custom` (定制重点电极)**:
   - 典型电极快速验证（如 `sub001-G13`, `sub007-C4`）。

### 2.2 试次样本空间
1. **训练集 (Task 3 纯物理红绿)**:
   - 红色色块试次：$y = 1$（Red，每位试 60 试次，包含 3 种不同 patch：`01`, `02`, `03` 各 20 试次）。
   - 绿色色块试次：$y = 0$（Green，每位试 60 试次，包含 3 种不同 patch：`01`, `02`, `03` 各 20 试次）。
   - 训练样本总数：$N_{\text{tr}} = 120$（sub003 为 180）。
2. **测试集 (Task 2 灰度记忆水果)**:
   - 仅选取 `state == 'gray'` 的试次（图片完全无物理彩色）。
   - 红色记忆水果：$y = 1$（草莓 Strawberry 60 试次、西瓜 Watermelon 60 试次，共 120 试次）。
   - 绿色记忆水果：$y = 0$（白菜 Cabbage 60 试次、猕猴桃 Kiwi 60 试次，共 120 试次）。
   - 测试样本总数：$N_{\text{te}} = 240$。

---

## 三、 数据划分 (分折机制) 与特征标准化深入详解

这是本方案中最关键的两个统计与工程细节：

### 3.1 跨任务解码的分折机制 (Data Splitting / Cross-Validation)

#### 为什么跨任务解码通常不需要在 Task 2 和 Task 3 之间做常规 K-Fold？
- **单任务解码**（例如仅在 Task 2 内部）：训练集和测试集来自同一个 Session/Task。如果不做交叉验证分折（K-Fold 或 Leave-One-Out），模型就会死记硬背训练样本的噪声，产生虚假的高准确率（严重过拟合）。
- **跨任务解码 (Task 3 -> Task 2)**：
  - **天然完全外推（Natural Out-of-Distribution Generalization）**:
    Task 3（纯色块）和 Task 2（灰度水果）在物理采集上是完全独立的两个实验阶段（甚至不同 Block/Run），两者的刺激材料没有任何交集（一个是色块，一个是水果）。
  - **因此，最纯粹、信噪比最高的黄金标准是：全量外推模式 (Full Cross-Task Generalization)**：
    - **训练阶段**: 使用 Task 3 的全部 120 个红绿试次训练线性分类器。这样能让 SVM 充分学习到所有 3 种色块（消除单一色块的偶然特征），拟合出最稳健、方差最小的物理颜色分类超平面 $(\mathbf{w}, b_0)$。
    - **测试阶段**: 将训练好的分类器直接作用于 Task 2 的全部 240 个灰度水果试次上，直接得出预测标签并计算平衡准确率。
    - **零数据泄露保证**: 训练集与测试集完全物理隔离，数学上不存在任何信息泄露（Data Leakage）可能！

#### 备选补充模式：跨色块集成模式 (Patch-Fold Ensemble Mode)
为了检验模型是否过度依赖 Task 3 中某一个具体的色块图片，我们同时提供可选的 3 折色块集成模式（在 `cfg.split_mode = 'patch_ensemble'` 下启用）：
1. **Fold 1**: 用 Task 3 的 Patch 1 & 2 训练模型 $M_1$，测试 Task 2 全量 240 个灰度试次，输出决策值 $d_1$；
2. **Fold 2**: 用 Task 3 的 Patch 1 & 3 训练模型 $M_2$，测试 Task 2 全量 240 个灰度试次，输出决策值 $d_2$；
3. **Fold 3**: 用 Task 3 的 Patch 2 & 3 训练模型 $M_3$，测试 Task 2 全量 240 个灰度试次，输出决策值 $d_3$；
4. 最终测试决策值为 3 个模型的集成均值：$d_{\text{final}} = \frac{1}{3}(d_1 + d_2 + d_3)$，再计算准确率。
> **代码设计**: 在主脚本中平铺设置 `cfg.split_mode = 'full'`（默认推荐全量外推）与 `'patch_ensemble'`，用户可通过一行配置直接自由切换。

---

### 3.2 特征标准化详细步骤 (Strict Out-of-Sample Z-Score Standardization)

在机器学习中，**特征标准化的参数必须严格源自训练集，严禁混合测试集计算！**

在时频分析中，每个时间窗 $w \in \{1, 2, \dots, 51\}$（对应 $[-200, 800]\text{ ms}$）包含特定频段或多频段的功率特征。具体标准化算法如下：

#### 步骤 1: 在训练集 (Task 3) 上计算基准均值与标准差
对于给定的训练时间窗 $w_3$（对角线时 $w_3 = w_2 = w$）：
假设特征向量为 $\mathbf{x} \in \mathbb{R}^D$（若为单频段，则 $D=1$；若为 Multi-Band 联合特征，则 $D=6$）。
对 Task 3 的 $N_{\text{tr}} = 120$ 个试次计算逐维度的样本均值 $\boldsymbol{\mu}_{\text{tr}}$ 与样本标准差 $\boldsymbol{\sigma}_{\text{tr}}$：
$$\mu_{\text{tr}, d}(w_3) = \frac{1}{N_{\text{tr}}} \sum_{i=1}^{N_{\text{tr}}} X_{\text{Task3}, i, d}(w_3), \quad d = 1, 2, \dots, D$$
$$\sigma_{\text{tr}, d}(w_3) = \sqrt{\frac{1}{N_{\text{tr}} - 1} \sum_{i=1}^{N_{\text{tr}}} \left(X_{\text{Task3}, i, d}(w_3) - \mu_{\text{tr}, d}(w_3)\right)^2}$$
- **数值保护**:
  若某维度 $\sigma_{\text{tr}, d}(w_3) < 10^{-6}$，强制重置为 $\sigma_{\text{tr}, d}(w_3) = 1.0$，防止除以零产生奇异值。

#### 步骤 2: 标准化 Task 3 训练特征
$$\tilde{X}_{\text{Task3}, i, d}(w_3) = \frac{X_{\text{Task3}, i, d}(w_3) - \mu_{\text{tr}, d}(w_3)}{\sigma_{\text{tr}, d}(w_3)}$$
标准化后，Task 3 在训练时间窗 $w_3$ 上的特征均值为 0，方差为 1。

#### 步骤 3: 严格使用 Task 3 的参数转换 Task 2 测试特征
**关键科学原理**:
绝对不能使用 Task 2 自身的均值进行 Z-score！因为一旦用 Task 2 自身的均值减去自身，就会强行抹平 Task 2 在该频段的物理基线偏置。  
必须将 Task 2 投影到 Task 3 建立的“物理色觉空间”中：
$$\tilde{X}_{\text{Task2}, j, d}(w_2) = \frac{X_{\text{Task2}, j, d}(w_2) - \mu_{\text{tr}, d}(w_3)}{\sigma_{\text{tr}, d}(w_3)}, \quad j = 1, 2, \dots, N_{\text{te}}$$
这样保证了测试样本与训练样本处于完全同一量纲和基准空间下。

---

## 四、 分类模型与两种时间分析维度

### 4.1 线性分类模型 (Linear SVM)
采用岭正则化线性 SVM，在训练集上优化目标函数：
$$\min_{\mathbf{w}, b_0} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{N_{\text{tr}}} \max\left(0, 1 - y_{\text{Task3}, i}(\mathbf{w}^T \tilde{\mathbf{x}}_{\text{Task3}, i}(w_3) + b_0)\right)$$
- MATLAB 原生调用:
  ```matlab
  mdl = fitclinear(X_tr_norm, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', 0.01);
  ```
- 测试集预测与平衡准确率 (Balanced Accuracy, BA):
  $$\hat{y}_{\text{te}, j} = \text{predict}(mdl, \tilde{X}_{\text{Task2}, j, :}(w_2))$$
  $$\text{BA} = \frac{1}{2}\left(\frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}}\right)$$

### 4.2 维度一：对角线同步解码 (Diagonal Decoding)
- **设定**: $w_3 = w_2 = w$（时间完全对应，即 $t_{\text{train}} = t_{\text{test}}$）。
- **输出**: 覆盖 $[-200, 800]\text{ ms}$ 的 1D 解码准确率时程曲线（51 个时间点）。
- **物理意义**: 评估物理色觉与记忆色觉在同一生理潜伏期下的重叠程度。

### 4.3 维度二：时间泛化矩阵 (Temporal Generalization Matrix, TGM)
- **设定**: 遍历所有训练时间点 $t_3 \in [-200, 800]\text{ ms}$ 与测试时间点 $t_2 \in [-200, 800]\text{ ms}$。
- **输出**: $51 \times 51$ 的 2D 准确率热力图（横轴：Task 2 记忆色时间；纵轴：Task 3 物理色时间）。
- **物理意义**:
  如果知觉编码在 150 ms 激活，而记忆色在 350 ms 才被唤醒，热力图会在 $(t_2=350, t_3=150)$ 处形成显著的离轴峰值（Off-diagonal Peak），从而精准捕捉认知重现的时序延迟。

---

## 五、 统计推断策略 (严格响应用户“TGM免置换检验”指示)

> **采纳指示**: 2D TGM 矩阵包含 $51 \times 51 = 2601$ 个网格点，若进行 200 次置换检验需要训练 $2601 \times 200 = 520,200$ 次 SVM，计算极其耗时。  
> **极速轻量化统计架构**:
> 1. **TGM (2D 网格)**: **仅计算真实的经验准确率热力图，暂不进行置换检验显著性校正**。通过准确率色阶直观展示全时程动态分布，秒级出图。
> 2. **对角线 (1D 曲线)**: **执行严格的非参数标签置换检验与时间簇质量检验 (Cluster-Mass Test)**。

### 5.1 1D 对角线置换检验与时间簇校正算法
1. **标签打乱**:
   保持 Task 3 训练标签不变，对 Task 2 测试集的 240 个试次标签进行随机置换打乱：
   $$y_{\text{te}}^{\text{perm}} = y_{\text{te}}(\text{randperm}(N_{\text{te}}))$$
2. **重复迭代**: 执行 $N_{\text{perm}} = 200$ 次置换，每次计算 1D 对角线解码曲线 $\text{BA}_{\text{null}, k}(t)$。
3. **时程平滑**: 真实曲线与置换曲线均执行 5 点高斯平滑 (`smoothdata(..., 'gaussian', 5)`)。
4. **单点 p 值**:
   $$p_{\text{pt}}(t) = \frac{1 + \sum_{k=1}^{N_{\text{perm}}} \mathbb{I}(\text{BA}_{\text{null}, k}(t) \ge \text{BA}_{\text{real}}(t))}{1 + N_{\text{perm}}}$$
5. **1D 时间簇质量检验 (Cluster-Mass Test)**:
   - 提取 $t \ge 0\text{ ms}$ 且 $p_{\text{pt}}(t) < 0.05$ 的连续时间点簇 $C_m$；
   - 计算簇质量: $M_m = \sum_{t \in C_m} (\text{BA}_{\text{real}}(t) - 0.5)$；
   - 提取置换零分布下的最大簇质量 $M_{\text{null}, k}^{\max}$；
   - 簇级别多重比较校正 p 值:
     $$p_{\text{cluster}, m} = \frac{1 + \sum_{k=1}^{N_{\text{perm}}} \mathbb{I}(M_{\text{null}, k}^{\max} \ge M_m)}{1 + N_{\text{perm}}}$$
   - 若 $p_{\text{cluster}, m} < 0.05$，在图表中用浅金黄底色高亮该显著时间簇。

---

## 六、 脚本工程落地架构 (`C09_cross_decoding_task3_to_task2_0825.m`)

遵照用户规则，代码结构扁平直观，置顶平铺配置：

```matlab
%% 1. 主参数配置 (平铺直观，可控易读)
cfg = struct();
cfg.target_mode    = 'concordant';       % 'concordant' (157个), 'all_sig' (229个), 'custom'
cfg.split_mode     = 'full';             % 'full' (100%全量外推，默认推荐) 或 'patch_ensemble'
cfg.test_state     = 'gray';             % 测试条件: 'gray' (灰度水果记忆色)
cfg.win_len        = 20;                 % 滑动窗长 20 ms
cfg.win_step       = 20;                 % 滑动步长 20 ms (覆盖 -200 到 800 ms，共 51 窗)
cfg.smooth_pts     = 5;                  % 平滑点数 (5点高斯平滑)
cfg.n_perm         = 200;                % 仅对角线进行 200 次置换检验 (TGM免置换，极大提速)
cfg.n_workers      = 20;                 % 20 线程极速并行池
cfg.svm_lambda     = 0.01;               % 岭正则化参数

% 频段定义
cfg.bands          = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.bands_disp     = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low-Gamma', 'High-Gamma'};
cfg.n_bands        = numel(cfg.bands);
```

### 6.1 视觉输出规范 (1:1 左右双子图)
- **图像大小**: `[1200, 520]`。
- **左子图 (1D 对角线解码时程)**:
  - 横轴：时间 $[-200, 800]\text{ ms}$；纵轴：跨任务解码准确率（%）。
  - 粗暖陶土橙褐线：`Multi-Band` 联合特征；6 条彩色细线：6 大独立频段。
  - 灰色阴影带：置换检验 95% 经验零分布置信区间；金色色带：$p_{\text{cluster}} < 0.05$ 显著时间簇。
  - 无背景方格 (`grid off`)。
- **右子图 (2D TGM 时间泛化热力图)**:
  - 横轴：Task 2 记忆色时间 $[-200, 800]\text{ ms}$；纵轴：Task 3 物理色时间 $[-200, 800]\text{ ms}$。
  - 对角虚线（$t_3 = t_2$）与 0 ms 刺激呈现刻度线。
  - 色阶（Colormap）：科研暖色渐变，颜色栏标注百分比。
- **标题格式**: 严格仅包含被试与通道（例如：`sub001 - G13`）。
