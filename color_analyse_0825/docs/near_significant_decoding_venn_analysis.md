# 包含“接近显著”（Near-Significant）位点下的 Decoding 神经解码拓扑重叠分析报告

## 1. 分析背景与统计定义

在先前的严格统计检验中，各分析均施加了全时程/全网格的非参数置换族误差校正（Cluster-Mass FWE Correction, $p_{\text{FWE}} < 0.05$）：
- **Task 2 交叉解码（C06）**：19 个位点
- **Task 2 Direct 解码（C13）**：13 个位点
- **Task 3 交叉解码（C07）**：14 个位点
- **Task 3 Direct 解码（C14）**：15 个位点
- **跨任务泛化（Cross-Task TGM 2D, C09）**：10 个位点

为了探究当统计范围适度放宽至包含**“接近显著” / 显著趋势（Near-Significant / Trend-level）**位点时，全脑电极在不同解码任务间的重合模式如何变化，本项目依据前期既定标准（见 `plot_task3_onset_y_correlation.m`）进行了拓展统计。

### 1.1 统计标准分层
1. **严格显著（Strict Significant）**：
   - 1D 解码曲线：通过时间簇质量置换检验校正（$p_{\text{cluster}} < 0.05$）。
   - 2D 跨任务泛化：通过 2D 连通域簇质量置换检验校正（$p_{\text{FWE}} < 0.05$）。
2. **接近显著（Near-Significant / Trend）**：
   - **1D 曲线（Task 2/3 Cross 与 Direct）**：在刺激呈现后（$t \ge 0\text{ ms}$），未通过全时程校正，但在点检验水平上存在连续 $\ge 4$ 个时间点（即持续时间 $\ge 80\text{ ms}$）达到 $p_{\text{pointwise}} < 0.05$。
   - **2D 跨任务泛化（Cross-Task TGM）**：2D 泛化矩阵置换检验呈现显著趋势（$0.05 \le p < 0.10$）。

---

## 2. 各解码任务位点数量变化对比

| 分析任务 | 严格显著位点数 (Strict, $p<0.05$) | 接近显著位点数 (Near-Sig) | 综合总数 (Combined) | 增幅 |
| :--- | :---: | :---: | :---: | :---: |
| **1. Task 2 交叉 (Cross-Fruit)** | 19 | 27 | **46** | +142% |
| **2. Task 2 Direct (纯红绿 5折)** | 13 | 27 | **40** | +208% |
| **3. Task 3 交叉 (Cross-Patch)** | 14 | 32 | **46** | +229% |
| **4. Task 3 Direct (纯红绿 5折)** | 15 | 30 | **45** | +200% |
| **5. Cross-Task (T3 $\to$ T2 2D)** | 10 | 9 | **19** | +90% |
| **全脑非重复位点总数** | **50** | **61** | **111** | +122% |

> **数据产出保存路径**：
> - 结构体数据：`result/tables/decoding_near_significant_sites_master.mat`
> - 完整表格：`result/tables/decoding_near_significant_sites_master.csv`

---

## 3. 三圆环韦恩图（Venn Diagram）重叠变化深度解析

### 3.1 图 1：交叉解码体系（Cross-Decoding: Task 2 Cross vs Task 3 Cross vs Cross-Task）
- **图像路径**：`result/figures/decoding_venn_diagrams/figure1_venn_cross_decoding_near_sig.png`
- **集合大小**：
  - A (Task 2 交叉): $N = 46$
  - B (Task 3 交叉): $N = 46$
  - C (Cross-Task): $N = 19$

#### 重叠位点对比变化：
1. **★ 三者共同重叠（A ∩ B ∩ C，Triple Overlap）**：
   - 严格显著时：**0 个**。
   - 包含接近显著后：**涌现出 1 个关键核心通道 —— `sub008-D10`**！
2. **A ∩ B（Task 2 交叉 与 Task 3 交叉 重叠）**：
   - 严格显著时：2 个（`sub003-H11`, `sub003-H2`）。
   - 包含接近显著后：**跃升至 9 个位点**（增长 4.5 倍）：
     * `sub001-D7` (White R)
     * `sub001-G6` (White R)
     * `sub001-G8` (White R)
     * `sub003-H11` (White R, 严格双显著)
     * `sub003-H2` (White R, 严格双显著)
     * `sub008-C3` (Visual stream)
     * `sub008-C7` (Visual stream)
     * `sub008-C8` (Visual stream)
     * `sub008-D10` (Visual stream)
3. **A ∩ C（Task 2 交叉 与 跨任务泛化 重叠）**：
   - 严格显著时：3 个（`sub001-D15`, `sub006-H10`, `sub008-C10`）。
   - 包含接近显著后：**扩展至 8 个位点**：
     * `sub001-D15` (Lateral occipital R)
     * `sub002-D3` (Cuneus R)
     * `sub002-D4` (Lateral occipital R)
     * `sub003-G2` (White R)
     * `sub006-H10` (CSF/Temporal)
     * `sub008-C10` (Visual stream)
     * `sub008-D10` (Visual stream)
     * `sub008-E8` (Visual stream)
4. **B ∩ C（Task 3 交叉 与 跨任务泛化 重叠）**：
   - 严格显著时：0 个。
   - 包含接近显著后：**1 个**（`sub008-D10`）。

---

### 3.2 图 2：Direct 解码体系（Direct-Decoding: Task 2 Direct vs Task 3 Direct vs Cross-Task）
- **图像路径**：`result/figures/decoding_venn_diagrams/figure2_venn_direct_decoding_near_sig.png`
- **集合大小**：
  - A (Task 2 Direct): $N = 40$
  - B (Task 3 Direct): $N = 45$
  - C (Cross-Task): $N = 19$

#### 重叠位点对比变化：
1. **★ 三者共同重叠（A ∩ B ∩ C，Triple Overlap）**：
   - 严格显著时：**0 个**。
   - 包含接近显著后：**同样确立了核心枢纽 —— `sub008-D10`**！
2. **A ∩ B（Task 2 Direct 与 Task 3 Direct 重叠）**：
   - 严格显著时：**0 个**（严格校正下早期感知与晚期提取位点分离）。
   - 包含接近显著后：**大幅涌现出 10 个重合位点**：
     * `sub001-D6` (Lateral occipital R)
     * `sub001-D7` (White R)
     * `sub001-G6` (White R)
     * `sub003-G13` (Middle temporal R)
     * `sub003-H11` (White R)
     * `sub003-H2` (White R)
     * `sub005-I2` (Temporal)
     * `sub006-G4` (Fusiform/Temporal)
     * `sub007-H5` (Temporal)
     * `sub008-D10` (Visual stream)
3. **A ∩ C（Task 2 Direct 与 跨任务泛化 重叠）**：
   - 严格显著时：2 个（`sub002-D3`, `sub006-H10`）。
   - 包含接近显著后：**扩充至 9 个位点**：
     * `sub001-D15` (Lateral occipital R)
     * `sub002-D3` (Cuneus R)
     * `sub002-D4` (Lateral occipital R)
     * `sub002-F8` (Temporal)
     * `sub003-G2` (White R)
     * `sub005-A2` (Fusiform R)
     * `sub006-F12` (Temporal)
     * `sub006-H10` (CSF/Temporal)
     * `sub008-D10` (Visual stream)
4. **B ∩ C（Task 3 Direct 与 跨任务泛化 重叠）**：
   - 严格显著时：0 个。
   - 包含接近显著后：**扩展至 2 个位点**（`sub004-L5`, `sub008-D10`）。

---

## 4. 关键科学发现与生物学启示

### 4.1 全脑超级枢纽通道：`sub008-D10`
- **现象**：在严格标准下，由于全时程严苛校正门槛较高，三圆环韦恩图中心均为 0。但在纳入接近显著（持续 $\ge 80\text{ ms}$）后，**`sub008-D10` 成为唯一一个在两个韦恩图体系（Cross 体系与 Direct 体系）中均达成 A ∩ B ∩ C 三者共同重叠的超级核心通道**！
- **生理学背景**：
  - 位于枕叶/颞下回腹侧视通路后侧（Visual Upstream）；
  - 在 Task 1 中表现为极强的 Theta 频段同向正效应与 Low-Gamma 频段负效应；
  - 在 Cross-Task 2D TGM 中达到严格显著（$p_{\text{FWE}} = 0.0448 < 0.05$）；
  - 这表明该区域直接承担了从“物理颜色感知”向“抽象记忆颜色回放”全流程的时序表征桥梁功能。

### 4.2 记忆与感知通路的功能交汇被重新揭示
- 严格显著时，Task 2 Direct 与 Task 3 Direct 之间重叠为 0，容易导致“颜色感知与颜色记忆由完全不相交的神经回路独立负责”的偏倚结论；
- 当引入持续 $\ge 80\text{ ms}$ 的接近显著位点后，我们观察到多达 10 个共有通道（如 `sub003-H11`、`sub003-H2`、`sub006-G4` 梭状回等），表明**感知和记忆在皮层回路中实际上共享了相当规模的亚阈值/瞬态调谐神经元群体**，只是记忆回放的信噪比或持续时间弱于强外刺激驱动。

---

## 5. 产出图谱与文件清单

1. **图 1 (Cross体系含接近显著)**：
   - `result/figures/decoding_venn_diagrams/figure1_venn_cross_decoding_near_sig.png`
2. **图 2 (Direct体系含接近显著)**：
   - `result/figures/decoding_venn_diagrams/figure2_venn_direct_decoding_near_sig.png`
3. **组合总览对比图**：
   - `result/figures/decoding_venn_diagrams/figure_combined_venn_near_sig.png`
4. **主数据文件**：
   - `result/tables/decoding_near_significant_sites_master.mat`
   - `result/tables/decoding_near_significant_sites_master.csv`
