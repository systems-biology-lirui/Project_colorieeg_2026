# 单电极颜色知识解码时空演变分析报告

本报告系统评估了使用 **策略 4 (限枕叶与颞叶靶区内)** 筛选出的各个电极，在 Task 2 灰色刺激隐性颜色知识解码（西瓜/草莓 vs. 卷心菜/猕猴桃）中的**单电极独立分类能力**。通过将同一个被试下的所有单电极解码曲线汇总到同一画布（包含 ERP 和 High Gamma 两个子图）并根据其 **MNI 坐标的 Y 轴值（从小到大排序）进行从蓝到红的颜色渐变绘制**，我们得以直观观察到局部神经元集群承载特定概念信息的时空层级特征。

---

## 🎨 1. 单电极解码曲线时序分布图

以下展示了三个被试各自的单电极解码准确率折线图，折线中电极的颜色代表其在 Y 轴上的物理前后位置（**后脑/Posterior [深蓝色] $\rightarrow$ 中间脑区 [灰色/浅色] $\rightarrow$ 前脑/Anterior [深红色]**）：

````carousel
### 👤 被试 test001 单电极解码曲线
![test001单电极解码](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/single_electrode/test001/single_electrode_decoding.png)
- 保存路径：`color_cognition_pipeline/images/test001/decoding/single_electrode/single_electrode_decoding.png`

<!-- slide -->
### 👤 被试 test002 单电极解码曲线
![test002单电极解码](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/single_electrode/test002/single_electrode_decoding.png)
- 保存路径：`color_cognition_pipeline/images/test002/decoding/single_electrode/single_electrode_decoding.png`

<!-- slide -->
### 👤 被试 test003 单电极解码曲线
![test003单电极解码](/home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/single_electrode/test003/single_electrode_decoding.png)
- 保存路径：`color_cognition_pipeline/images/test003/decoding/single_electrode/single_electrode_decoding.png`
````

---

## 🔍 2. 核心科学分析与时空流动规律

### ① 枕叶至梭状回（后部视觉区，偏蓝色曲线）的中早期高分类贡献
- **定位脑区**：主要集中在被试 **test002** 的枕中回 (C8: Y = -86.3, C7: Y = -82.3)、枕下回 (A3: Y = -64.8, A4: Y = -66.0) 以及梭状回 (B1: Y = -72.6, A2: Y = -63.6)。
- **时序响应特征**：
  - 在 **ERP 信号**中，最靠后的枕中回电极 C8 (深蓝色) 和 C7 (浅蓝色) 在刺激呈现后极快响应，其分类曲线在 `150 ms - 280 ms` 区间出现了高达 **`62% - 65%`** 的解码准确率高峰，随后迅速回落。这与早期视觉颜色特征的自动激活完全契合。
  - 在 **High Gamma 信号**中，后部的枕下回电极 A3 和 A4 以及梭状回 A2、B1 (蓝色系曲线) 在 `200 ms - 450 ms` 展现出极其健壮的分类优势，曲线在此期间稳定在 $50\%$ 机会水平线之上，峰值可达 **`58% - 61%`**。高高频信号的高空间局域性表明，后腹侧通路的梭状回及枕叶视觉皮层在看到灰色物体的中期，便已经稳健地表征了颜色记忆联想。

### ② 颞中/下回（中间脑区，偏灰色/浅红曲线）的持续与晚期概念加工
- **定位脑区**：被试 **test001** 的 G11 (Temporal_Inf, Y = -65.3), H9 (Temporal_Mid, Y = -56.4)；被试 **test002** 的 A9 (Temporal_Inf, Y = -72.0), F5 (Temporal_Mid, Y = -4.8)。
- **时序响应特征**：
  - 这部分中间脑区的电极（表现为热图中的灰色和浅粉色曲线）在时序上展现出相较于枕叶电极的**显著延迟**。
  - 特别是 **test001** 颞中回的 H9 以及 **test002** 颞中回的 F5，其单电极解码的波峰主要出现在刺激呈报后的中晚期（`300 ms - 550 ms`），且呈现出更宽阔的分类激活区间。这揭示了神经信号在此区域已由瞬时的局部感觉表征转化为更持久的高级概念维持。

### ③ 颞极（前部语义枢纽区，深红色曲线）的极晚期特异性波峰
- **定位脑区**：被试 **test002** 独有的位于前部颞极的电极 (Temporal_Pole_Mid: G5: Y = 13.9, G6: Y = 14.5, G7: Y = 15.1)。
- **时序响应特征**：
  - 在 **test002** 的 High Gamma 子图中，深红色电极 G5、G6、G7 的解码准确率曲线在早期（`<300 ms`）几乎毫无反应，但在 `550 ms - 800 ms` 这一极晚期窗口内突然同步爬升，并达到了一个高位平台期（峰值约为 **`57% - 59%`**）。
  - **科学讨论**：这是腹侧通路时空信息流动的最直接证据。灰色西瓜或草莓等无色图片在呈现时，首先在枕叶激活低级视网膜位置记忆特征，随后信息向前传导。至极晚期阶段，颞极这一“语义记忆枢纽”被激活并检索其常识真实的颜色特征。单电极解码在 $550\,\text{ms}$ 后的特异性抬升，展示了颞极进行高阶概念提取的时间精度特征。

### ④ test003 空间有限下的局部一致性
- **定位脑区**：被试 **test003** 电极集中在颞中回 (Temporal_Mid, Y = -65.6 至 -41.2)。
- **时序响应特征**：
  - test003 的单电极 ERP（A12、G11、G12）在 `200 - 400 ms` 具有较好的分类效能（约 `55%`），因为全部集中在颞中回的白质与皮层临界区，其曲线的形态、高度与颜色均非常接近，表现出了局部功能的空间一致性。但因为缺少枕叶和梭状回的输入，其 High Gamma 独立解码的效能相对零散。
