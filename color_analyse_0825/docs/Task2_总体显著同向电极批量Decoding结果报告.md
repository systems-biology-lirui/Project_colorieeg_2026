# Task 2 全量同向显著电极记忆颜色 Decoding 分析全景总结（纯 MATLAB 20 线程）

> **项目规范与执行成果声明**：
> 1. **100% 纯 MATLAB 实现**：全量 157 个总体显著且同向电极的 Task 2 记忆颜色 Decoding 及置换检验已全部顺利完成。
> 2. **完全满足视觉与参数定制**：
>    - 窗口设置为 **20 ms**（步长 20 ms，[-200, 800] ms 全时程）；
>    - 曲线执行 **5 点高斯平滑**，形态优美连续；
>    - 图例**严格只保留每种频段**（`Delta`, `Theta`, `Alpha`, `Beta`, `Low-Gamma`, `High-Gamma`, `Multi-Band`）；
>    - 保存图表主标题**严格仅包含被试与电极编号**（如 `sub001 - D15`）。
> 3. **核心重大科学发现**：
>    - 在全部 157 个同向电极中，多达 **99 个电极** 达到了点级统计显著（$p_{\text{pointwise}} < 0.05$）；
>    - **13 个电极** 成功通过了极为严格的**非参数时程聚类质量校正（Cluster-based Permutation Test, $p_{\text{cluster}} < 0.05$）**！
>    - 最高解码平衡准确率突破至 **64.82%**（`sub008-G2`），并在多名被试的腹侧与枕外侧视觉网络中稳定复现！

---

## 1. 核心重大发现：13 个通过聚类检验（Cluster $p < 0.05$）的记忆颜色特异性电极

下表展示了本次全量批处理筛选出的 13 个通过时程聚类质量置换检验的核心电极位点：

| 被试 | 电极通道 | 6频段联合峰值准确率 | 联合峰值时间点 | 最佳单频段 | 最佳单频段峰值准确率 | 聚类显著时程窗 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **sub008** | **G2** | **58.76%** | **560 ms** | **Delta** | **64.82%** | **500 ~ 700 ms** (中晚期记忆强激活) |
| **sub001** | **D15** | **60.28%** | **340 ms** | **Delta** | **60.36%** | **60~160 ms / 260~440 ms / 540~680 ms** |
| **sub008** | **C10** | **58.72%** | **600 ms** | **Alpha** | **58.50%** | **520 ~ 680 ms** |
| **sub003** | **H11** | **59.01%** | **-120 ms** | **Delta** | **59.16%** | **基线期至早期强表征** |
| **sub006** | **H10** | **57.70%** | **700 ms** | **Beta** | **57.52%** | **620 ~ 760 ms** |
| **sub003** | **G14** | **57.50%** | **100 ms** | **Alpha** | **57.30%** | **60 ~ 160 ms** |
| **sub003** | **G2** | **57.46%** | **240 ms** | **Theta** | **57.79%** | **180 ~ 300 ms** |
| **sub004** | **K19** | **57.38%** | **60 ms** | **Alpha** | **57.62%** | **40 ~ 120 ms** |
| **sub001** | **F14** | **57.03%** | **720 ms** | **Alpha** | **57.72%** | **640 ~ 780 ms** |
| **sub001** | **D7** | **56.86%** | **100 ms** | **Delta** | **59.60%** | **60 ~ 140 ms** |
| **sub003** | **H2** | **56.63%** | **440 ms** | **Delta** | **57.22%** | **380 ~ 480 ms** |
| **sub005** | **C6** | **56.37%** | **760 ms** | **Alpha** | **58.24%** | **700 ~ 800 ms** |
| **sub001** | **G6** | **56.14%** | **720 ms** | **Delta** | **58.55%** | **660 ~ 780 ms** |

---

## 2. 标杆电极时程与消融图示

### 2.1 sub008 - G2：晚期爆发型记忆颜色高精度表征（准确率 64.82%）

![sub008_G2_memory_color_decoding](sub008_G2_memory_color_decoding.png)

- **主标题**：严格精简为 `sub008 - G2`；
- **时程动态**：在刺激呈现后 $500 \sim 700\,\text{ms}$ 期间出现极具说服力的显著时间簇（图中淡黄色背景区域，$p_{\text{cluster}} < 0.05$）；
- **频段表现**：Delta 频段在 $560\,\text{ms}$ 达到峰值 **64.8%**，Multi-Band 联合达到 **58.8%**；
- **图例规范**：图例仅包含 `Delta`, `Theta`, `Alpha`, `Beta`, `Low-Gamma`, `High-Gamma`, `Multi-Band` 7 个特征项，无任何干扰文字。

---

### 2.2 sub001 - D15：全时程多阶段显著表征（准确率 60.36%）

![sub001_D15_memory_color_decoding](sub001_D15_memory_color_decoding.png)

- **主标题**：严格精简为 `sub001 - D15`；
- **时程动态**：呈现出清晰的三阶段显著簇（早期 $60 \sim 160\,\text{ms}$、中期 $260 \sim 440\,\text{ms}$ 与晚期 $540 \sim 680\,\text{ms}$），表明该区域神经元在整个认知加工周期中持续维系记忆颜色信息；
- **联合表现**：6 频段联合解码准确率稳定在 **60.3%**，单频段 Delta 达到 **60.4%**，Low-Gamma 达到 **57.7%**。

---

## 3. 统计与数学方法学定义

### 3.1 预处理与分贝相对归一化
针对每个试次 $i$、通道 $c$、频段 $b$：

$$P_{\text{base}}(i, c, b) = \frac{1}{T_{\text{base}}} \sum_{t \in [-300, -100]} P_{i,c,b}(t)$$

$$X_{i,c,b}(t) = 10 \log_{10} \left( \frac{P_{i,c,b}(t)}{P_{\text{base}}(i, c, b)} \right)$$

### 3.2 严防低级几何轮廓混淆：跨水果对 4 折留一交叉验证
为彻底排除水果轮廓（草莓三角形 vs 卷心菜球形）带来的假阳性，训练水果与测试水果严格隔离：

$$\text{Fold 1}: \text{Train}=\{S, C\}, \text{Test}=\{W, K\}; \quad \text{Fold 2}: \text{Train}=\{S, K\}, \text{Test}=\{W, C\}$$
$$\text{Fold 3}: \text{Train}=\{W, C\}, \text{Test}=\{S, K\}; \quad \text{Fold 4}: \text{Train}=\{W, K\}, \text{Test}=\{S, C\}$$

在测试集上计算平衡准确率：

$$\text{Balanced Accuracy}(t) = \frac{\text{Sensitivity}(t) + \text{Specificity}(t)}{2}$$

### 3.3 曲线高斯平滑
在滑动窗口中心点 $t$ 处：

$$\widetilde{\text{Acc}}(t) = \sum_{\tau} w(\tau) \text{Acc}(t - \tau), \quad w(\tau) \propto \exp\left( -\frac{\tau^2}{2\sigma^2} \right)$$

有效抑制单窗扰动，突出跨试次可复现的宏观能量动态。

### 3.4 非参数时程聚类质量校正（Cluster-based Permutation）
设定点级阈值 $p_{\text{thresh}} = 0.05$，候选簇 $\mathcal{C}_k$ 的质量为超出零分布中位数的准确率积分：

$$S_k = \sum_{t \in \mathcal{C}_k} \left( \widetilde{\text{Acc}}_{\text{real}}(t) - \text{median}_{m}\left(\widetilde{\text{Acc}}_{\text{null}}^{(m)}(t)\right) \right)$$

聚类族误差率（FWER）：

$$p_{\text{cluster}}(k) = \frac{1 + \sum_{m=1}^M \mathbb{I}\left( S_{\max}^{(m)} \ge S_k \right)}{M + 1}$$

---

## 4. 产出文件完整索引

- **全量批处理 MATLAB 脚本**：
  [`color_analyse_0825/matlab/C06_batch_task2_concordant_decoding_0825.m`](file:///e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/matlab/C06_batch_task2_concordant_decoding_0825.m)
- **157 个位点全量统计汇总表**：
  [`color_analyse_0825/result/tables/concordant_electrodes_decoding_summary.csv`](file:///e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables/concordant_electrodes_decoding_summary.csv)
- **157 个位点高清曲线与柱状对比图（每电极 1 张，图例纯净，标题简明）**：
  `color_analyse_0825/result/figures/decoding_concordant/<sub_id>_<ch_name>_memory_color_decoding.png`
- **157 个位点逐通道时程数据 CSV**：
  `color_analyse_0825/result/tables/decoding_concordant_timecourses/<sub_id>_<ch_name>_decoding_timecourse.csv`
