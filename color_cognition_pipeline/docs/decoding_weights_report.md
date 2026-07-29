# 策略 4 电极解码特征权重时空热图分析报告

本报告针对三个被试（`test001`, `test002`, `test003`）在进行灰色刺激颜色知识解码（西瓜/草莓 $\rightarrow$ 红色 vs. 卷心菜/猕猴桃 $\rightarrow$ 绿色）时，在每个时间点上的线性 SVM 分类器特征权重系数进行了时序特征提取与可视化。

## 📊 1. SVM 权重与颜色偏好的统计学意义说明

- **特征权重值正负性（Valence of Weights）**：
  - 由于分类标签中，**0 代表红色条件，1 代表绿色条件**。
  - 在每个时间步，输入分类器的各通道数据均经过了标准归一化（$z$-score），因此：
    - **权重系数 $w_{ch} < 0$ (红端色调)**：表示该通道的能量/电位响应越大，分类器决策越偏向于**红色联想**。
    - **权重系数 $w_{ch} > 0$ (蓝端色调)**：表示该通道的能量/电位响应越大，分类器决策越偏向于**绿色联想**。
  - **权重绝对值大小 $|w_{ch}|$ (颜色饱和度)**：代表该电极在特定时间点对红色与绿色隐含分类的**贡献强度**。

---

## 🎨 2. 特征权重时序热图分布

我们使用对称化双极性配色（`RdBu_r`，其中红色代表偏好红色，蓝色代表偏好绿色，白色为零贡献）绘制了每个被试在 ERP 和 High Gamma 信号上的电极特征权重时序热图：

````carousel
### 👤 被试 test001 特征权重热图
#### (1) ERP 信号
![test001 ERP 权重热图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/decoding_weights/test001/erp_strategy4_weights_heatmap.png)
- 存储路径：`color_cognition_pipeline/images/test001/decoding/weights/erp_strategy4_weights_heatmap.png`

#### (2) High Gamma 信号
![test001 HG 权重热图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/decoding_weights/test001/highgamma_strategy4_weights_heatmap.png)
- 存储路径：`color_cognition_pipeline/images/test001/decoding/weights/highgamma_strategy4_weights_heatmap.png`

<!-- slide -->
### 👤 被试 test002 特征权重热图
#### (1) ERP 信号
![test002 ERP 权重热图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/decoding_weights/test002/erp_strategy4_weights_heatmap.png)
- 存储路径：`color_cognition_pipeline/images/test002/decoding/weights/erp_strategy4_weights_heatmap.png`

#### (2) High Gamma 信号
![test002 HG 权重热图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/decoding_weights/test002/highgamma_strategy4_weights_heatmap.png)
- 存储路径：`color_cognition_pipeline/images/test002/decoding/weights/highgamma_strategy4_weights_heatmap.png`

<!-- slide -->
### 👤 被试 test003 特征权重热图
#### (1) ERP 信号
![test003 ERP 权重热图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/decoding_weights/test003/erp_strategy4_weights_heatmap.png)
- 存储路径：`color_cognition_pipeline/images/test003/decoding/weights/erp_strategy4_weights_heatmap.png`

#### (2) High Gamma 信号
![test003 HG 权重热图](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/decoding_weights/test003/highgamma_strategy4_weights_heatmap.png)
- 存储路径：`color_cognition_pipeline/images/test003/decoding/weights/highgamma_strategy4_weights_heatmap.png`
````

---

## 🔍 3. 时空动态特征与脑区定位的关联讨论

### ① 枕叶早期视觉与高级视区（Occipital & Fusiform）的时效响应
- **定位电极**：在 **test002** 的脑图中，我们有分布在枕下回 (Occipital_Inf: A3, A4)、枕中回 (Occipital_Mid: C7, C8, C9, C10) 和梭状回 (Fusiform: A2, B1) 的电极。
- **动态表现**：
  - **ERP 信号**：枕下回 (A3) 和枕中回 (C7, C8) 的 ERP 权重表现出在 `150 ms - 300 ms` 的早期阶段发生剧烈的红-蓝极性交替（代表刺激后 N170 附近成分的动态调制），并展现了极强的绝对权重贡献（深红与深蓝）。
  - **High Gamma 信号**：枕下回 (A3, A4) 和梭状回 (A2) 在刺激呈现后 `200 ms - 400 ms` 内呈现出极其稳定且饱和的**负权重**（深红色斑块，偏向红色偏好）。这强有力地说明，在刺激呈现后的中期阶段，梭状回和枕叶高频成分的局部能量对红色水果的脑机制编码贡献了极其显著且稳定的信息。

### ② 颞中/下回（Temporal_Mid & Temporal_Inf）的持续语义表征
- **定位电极**：包括 **test001** 的 B5, C8, C10, G11, F9 以及 **test002** 的 A8, A9, F4, F6, H5, H8。
- **动态表现**：
  - 颞下回 (Temporal_Inf, 如 test001-G11、test002-A9) 的 ERP 和 HG 权重在刺激呈现后的晚期（`400 ms - 800 ms`）依然呈现出高强度的特征权重（如 test002-A9 表现为长时程深蓝色正权重，表示长期的绿色偏好；test001-G11 表现为晚期深红色偏好）。
  - 这十分契合腹侧视觉通路（Ventral Stream）中颞下回承担高级概念激活、物体范畴表征的功能，电极在长窗口内编码了持续提取的“灰色水果所隐含的真实颜色”这一记忆知识。

### ③ 颞极（Temporal_Pole）在晚期概念检索中的独特作用
- **定位电极**：**test002** 特异性拥有的颞极电极 (Temporal_Pole_Mid: G5, G6, G7)。
- **动态表现**：
  - 在 test002 的 High Gamma 热图中，位于颞极的 G6 和 G7 电极在 `500 ms - 800 ms` 这一极晚期的认知响应窗口内展现了深蓝色的高对比正权重斑块。
  - **生理学解释**：颞极是语义记忆的“枢纽中心”（Hub）。由于 Task 2 使用的是灰色无色水果，被试在看到“灰色草莓”时必须通过记忆关联提取其“红色”属性，这需要高度依赖颞叶前部（颞极）的语义知识提取与记忆检索。热图中 G6, G7 电极在晚期的强贡献直接展示了语义知识重激活在局部高频电位上的时空机制。

### ④ test003 的局域一致性与个体差异
- **定位电极**：**test003** 的电极局限在颞中回 (Temporal_Mid: ERP 的 A12, G11, G12; HG 的 D9, G9, G10, G11, G14)。
- **动态表现**：
  - test003 的热图相对集中在颞中回的白质/皮层交界区域。其 ERP 信号的 A12 和 G11 在 `200 ms - 450 ms` 展现出明显的极性对立（A12 偏红，G11 偏绿），而其 High Gamma 的特征权重较弱且动态变化较快。这与其整体解码精度略低于 test001 和 test002 的表现一致，说明局限在单一解剖靶区（仅颞中回）的电极对于颜色联想的分类信息量提供相对有限，需要联合枕叶和梭状回等高级视觉整合区才能达到最佳分类精度。
