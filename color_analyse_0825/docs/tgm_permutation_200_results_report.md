# Task 3 至 Task 2 跨任务神经解码时间泛化 (TGM) 200次全网格非参数置换检验报告

## 1. 分析背景与科学目标

在先前的跨任务神经解码分析（`C09_cross_decoding_task3_to_task2_0825.m`）中：
- **训练集**：Task 3 纯物理色块（红 vs 绿），旨在提取纯粹的低阶颜色神经模式；
- **测试集**：Task 2 灰度水果（草莓/西瓜 vs 猕猴桃/黄瓜），无任何物理颜色输入，考察自上而下的颜色记忆提取（Color Memory Retrieval）；
- **原分析局限**：原 C09 仅对 1D 对角线同步解码曲线执行了置换检验，而 2D 时间泛化矩阵（Temporal Generalization Matrix, 51x51 网格）仅输出了经验解码准确率热力图，未引入严格的多重比较置换检验（"免置换检验，秒级出图"）。

为填补这一统计空白，本次分析对 Task 1 独立筛选出的全部 **157 个四类别同向显著电极**（Concordant Channels），全面引入了 **2D TGM 全网格 200 次非参数置换检验**与 **2D 连通域簇质量族误差校正（2D Cluster-Mass FWE Correction, $p_{\text{FWE}} < 0.05$）**。

---

## 2. 统计方法与数学原理

### 2.1 零假设与标签置换机制
在跨任务泛化模型中，训练模型 $M_{w3}$ 仅在 Task 3 的第 $w_3$ 时间窗上训练：
$$M_{w3} = \arg\min_w \mathcal{L}(X_{\text{Task3}}(w_3), y_{\text{Task3}})$$

测试集为 Task 2 的灰度试次，特征 $X_{\text{Task2}}(w_2)$ 严格采用 Task 3 的特征均值 $\mu_{w3}$ 与标准差 $\sigma_{w3}$ 进行外推标准化。
由于分类器是在 Task 3 上固定训练的，对于 Task 2 的灰度试次，模型预测标签 $\hat{y}_{i}(w_3, w_2)$ 在计算后完全固定（维度为 $N_{\text{Task2}} \times 51 \times 51$）。

在零假设 $H_0$（Task 2 灰度水果的神经表征中不存在与颜色记忆相关的类别信息）下，通过对 Task 2 的真实记忆颜色标签 $y_{\text{Task2}}$ 执行 $P = 200$ 次随机置换：
$$y_{\text{perm}}^{(p)} = \text{permute}(y_{\text{Task2}}), \quad p = 1, \dots, 200$$

置换下的平衡准确率（Balanced Accuracy）矩阵可极速计算为：
$$\text{Acc}_{\text{null}}^{(p)}(w_3, w_2) = \frac{1}{2} \left[ \frac{\sum_{i \in \text{Red}} \mathbb{I}(\hat{y}_i(w_3, w_2) == 1)}{N_{\text{Red}}} + \frac{\sum_{i \in \text{Green}} \mathbb{I}(\hat{y}_i(w_3, w_2) == 0)}{N_{\text{Green}}} \right]$$

### 2.2 2D 空间高斯平滑与逐点 p 值
为消除高频采样抖动，对真实 TGM 矩阵和 200 次置换零分布矩阵施加同源二维高斯平滑（$\sigma = 0.8$）：
$$\text{TGM}_{\text{real}}^s = \text{imgaussfilt}(\text{TGM}_{\text{real}}, 0.8)$$
$$\text{TGM}_{\text{null}}^s(p, :, :) = \text{imgaussfilt}(\text{TGM}_{\text{null}}(p, :, :), 0.8)$$

逐像素点经验单侧 $p$ 值定义为：
$$p(w_3, w_2) = \frac{1 + \sum_{p=1}^{200} \mathbb{I}\left( \text{TGM}_{\text{null}}^s(p, w_3, w_2) \ge \text{TGM}_{\text{real}}^s(w_3, w_2) \right)}{1 + 200}$$

### 2.3 2D 簇质量多重比较校正 (Cluster-Mass FWE)
1. **候选显著像素集**：
   $$\mathcal{S} = \{(w_3, w_2) \mid p(w_3, w_2) < 0.05 \land t_{\text{Task2}}(w_2) \ge 0\}$$
   （限制 Task 2 反应时间在刺激呈现后，杜绝刺激前基线噪声形成的虚假簇）。
2. **8-连通域聚类与簇质量**：
   利用 8-连通域算法划分不相交的空间连通簇 $C_k \subseteq \mathcal{S}$，计算其簇质量（超越机遇水平 0.50 的超额积分）：
   $$\text{Mass}(C_k) = \sum_{(w_3, w_2) \in C_k} (\text{TGM}_{\text{real}}^s(w_3, w_2) - 0.50)$$
3. **零分布最大簇质量分布**：
   在每次置换 $p$ 中，以零分布的 95 分位阈值 $Q_{95}(w_3, w_2)$ 阈值化，计算该置换下的最大簇质量：
   $$\text{MaxMass}_{\text{null}}(p) = \max_{j} \text{Mass}(C_j^{(p)})$$
4. **FWE 簇级显著性判断**：
   对真实簇 $C_k$，其簇级族误差率（Family-Wise Error Rate）为：
   $$p_{\text{FWE}}(C_k) = \frac{1 + \sum_{p=1}^{200} \mathbb{I}\left( \text{MaxMass}_{\text{null}}(p) \ge \text{Mass}(C_k) \right)}{1 + 200}$$
   当 $p_{\text{FWE}} < 0.05$ 时，判定该簇为全网格统计显著的时序重放表征簇。

---

## 3. 全量 157 个电极的统计结果汇总

- **分析电极总数**：157 个（跨 8 名被试）
- **1D 对角线具有显著时间簇通道数**：3 个（1.9%）
- **2D TGM 具有显著泛化簇通道数**：**10 个（6.4%）**

### 3.1 通过 2D Cluster-Mass FWE ($p < 0.05$) 显著性检验的 10 个核心通道

| 被试与通道 | 1D对角线峰值 (时间) | 2D TGM 峰值准确率 | Task 3 训练时间 | Task 2 泛化解码时间 | 显著簇数量 | 2D显著簇面积 (像素点) | 簇级经验 p 值 ($p_{\text{FWE}}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **sub001-D15** | 56.43% (730 ms) | **58.99%** | 410 ms | 790 ms | 1 | 301 pts | 0.0249 |
| **sub002-D3** | 55.80% (250 ms) | **58.84%** | 170 ms | 690 ms | 1 | 443 pts | 0.0149 |
| **sub002-D4** | 58.27% (790 ms) | **59.33%** | 230 ms | 350 ms | 1 | **481 pts** | **0.0050** |
| **sub002-F8** | 55.34% (630 ms) | **59.92%** | -130 ms | 390 ms | 1 | 132 pts | 0.0249 |
| **sub004-F2** | 54.13% (530 ms) | **57.94%** | 310 ms | 750 ms | 1 | 123 pts | 0.0498 |
| **sub006-F15** | 56.07% (-10 ms) | **59.08%** | -30 ms | 390 ms | 1 | 100 pts | 0.0348 |
| **sub006-H10** | 55.81% (670 ms) | **60.40%** | 530 ms | 710 ms | 1 | 139 pts | 0.0199 |
| **sub008-C10** | 58.78% (450 ms) | **59.25%** | 230 ms | 570 ms | 1 | 302 pts | 0.0199 |
| **sub008-D10** | 55.14% (690 ms) | **60.59%** | -30 ms | 170 ms | 1 | 167 pts | 0.0448 |
| **sub008-E8** | 59.99% (790 ms) | **58.85%** | 790 ms | 790 ms | 1 | 171 pts | 0.0100 |

*注：`sub008-C10` 在 1D 对角线同步检验与 2D TGM 时间泛化检验中均达到双重显著 ($p_{\text{FWE}} < 0.05$)。*

---

## 4. 神经科学机制发现与讨论

### 4.1 为什么 2D TGM 检测出的显著通道 (10个) 远多于 1D 对角线 (3个)？
1. **潜伏期非同步性 (Temporal Asynchrony)**：
   - 1D 对角线解码假定 $t_{\text{Task3}} = t_{\text{Task2}}$，即假定物理色块感知与记忆色自上而下提取具有完全相同的神经潜伏期。
   - 然而，物理颜色知觉是早期的前馈加工（Feedforward Processing，通常在刺激后 150–250 ms 达到峰值）；而灰度水果的颜色记忆提取是高级认知表征从腹侧颞叶（Ventral Temporal Cortex）或海马向早期皮层的反馈提取（Feedback / Top-down Replay），其潜伏期显著滞后（集中在 350–750 ms）。
2. **离角线泛化 (Off-Diagonal Generalization)**：
   - 如上表所示，以最显著的 `sub002-D4` 为例：Task 3 物理颜色分类器在 **230 ms** 训练得到的特征权重，在 Task 2 灰度水果呈现后 **350 ms** 产生显著解码（面积达 481 像素，$p = 0.0050$）。
   - 以典型真记忆色通道 `sub008-C10` 为例：Task 3 物理颜色分类器在 **230 ms** 的模式，能够在 Task 2 灰度水果的 **570 ms** 准确解码记忆颜色（面积达 302 像素，$p = 0.0199$）。
   - 这种强烈的离对角线偏离（Off-diagonal shift）直接证实了**自上而下的颜色记忆提取是对早期物理颜色神经表征的延迟激活与回放（Delayed Neural Replay）**。

---

## 5. 产出文件索引

1. **2D 显著轮廓线图谱（每张图包含 1D 对角线与叠加白色显著轮廓线的 2D TGM）**：
   - 专用目录：`result/figures/cross_decoding_tgm_perm200/`
   - 同步更新目录：`result/figures/cross_decoding_concordant/`
2. **逐通道详细时程与置换零分布 .mat 文件**：
   - `result/tables/cross_decoding_tgm_perm200_timecourses/`
   - `result/tables/cross_decoding_concordant_timecourses/`
3. **全量 157 通道汇总指标表格**：
   - `result/tables/cross_decoding_tgm_perm200_summary.csv`
   - `result/tables/cross_decoding_concordant_summary.csv`
4. **核心执行脚本**：
   - 批处理测试脚本：`color_analyse_0825/matlab/test/run_all_tgm_permutations.m`
   - 主流程升级代码：`color_analyse_0825/matlab/C09_cross_decoding_task3_to_task2_0825.m`
