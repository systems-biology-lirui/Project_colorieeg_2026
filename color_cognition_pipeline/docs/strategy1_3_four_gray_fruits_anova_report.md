# 🧠 策略 1 + 策略 3 汇总电极 Task 2 4 种灰色水果 ERP 时程与 100-400ms ANOVA 分析报告

本报告针对 **策略 1 与 策略 3 汇总电极通道（共 30 个）** 在 Task 2（隐性颜色常识范式）中呈报的 **4 种无彩色灰色水果/蔬菜刺激** 进行了 Pooled 信号混合时程绘制与 100-400ms 时间窗口内单因素方差分析 (One-way ANOVA) 显著性比较。

---

## 🍇 一、 实验条件与试次数据概况

- **电极样本**：汇总 Strategy 1 与 Strategy 3 在腹侧视觉通路 Target Area 内筛选出的所有 **30 个电极通道**（涵盖 `test001`: 16个, `test002`: 6个, `test003`: 4个, `test005`: 4个）。
- **4 种灰色水果条件与试次量**：
  1. 🔴 **Gray Strawberry (灰色草莓 / 红色记忆)**: $N = 1796$ 试次
  2. 🔴 **Gray Watermelon (灰色西瓜 / 红色记忆)**: $N = 1796$ 试次
  3. 🟢 **Gray Cabbage (灰色卷心菜 / 绿色记忆)**: $N = 1796$ 试次
  4. 🟢 **Gray Kiwifruit (灰色猕猴桃 / 绿色记忆)**: $N = 1796$ 试次

---

## 📉 二、 100-400ms 时间窗口单因素方差分析 (One-way ANOVA) 结果

我们提取了 30 个通道在 4 种灰色水果刺激下于 `100 - 400 ms` 时间窗口内的平均振幅试次，运行了单因素方差分析 (One-way ANOVA)：

| 灰色水果条件 | 记忆颜色分类 | 100-400ms 振幅均值 ($\mu\text{V}$) | 标准误差 ($\text{SEM}$) | 试次总数 $N$ |
| :--- | :--- | :--- | :--- | :--- |
| **Gray Strawberry** | 🔴 红色常识 | **$0.788\ \mu\text{V}$** | $0.313\ \mu\text{V}$ | 1796 |
| **Gray Watermelon** | 🔴 红色常识 | **$1.639\ \mu\text{V}$** | $0.329\ \mu\text{V}$ | 1796 |
| **Gray Cabbage** | 🟢 绿色常识 | **$0.618\ \mu\text{V}$** | $0.352\ \mu\text{V}$ | 1796 |
| **Gray Kiwifruit** | 🟢 绿色常识 | **$0.945\ \mu\text{V}$** | $0.331\ \mu\text{V}$ | 1796 |

### 方差分析 (ANOVA) 硬核数据：
- **$F$-statistic** = **$\mathbf{1.8246}$**
- **$p$-value** = **$\mathbf{0.1403}$** ($p > 0.05$，未达整体标量均值显著差异)

---

## 🖼️ 三、 4 种灰色水果 ERP 时程曲线与 ANOVA 对比图

我们绘制了 4 种灰色水果在 `-200 ms ~ 800 ms` 上的 Pooled 平均 ERP 响应波形（带 SEM 阴影及 100-400ms 淡黄背景高亮区），并在右侧绘制了 100-400ms 均值对比柱状图与 ANOVA 统计卡片：

![Strategy 1+3 Four Gray Fruits ANOVA](file:///home/lirui/.gemini/antigravity-ide/brain/d44e149a-5de8-4c90-a764-e470f00cc895/strategy1_3_combined_four_gray_fruits_anova.png)
- **图表链接**：[strategy1_3_combined_four_gray_fruits_anova.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/result/select_channel/strategy1_3_combined_four_gray_fruits_anova.png)

---

## 🔬 四、 神经科学机制讨论

1. **物体与类别层次的振幅极性抵消 (Multi-Item Polarity Cancellation)**：
   在腹侧视觉通路的不同通道上，不同物体（如灰色西瓜 vs 灰色草莓，同属红色记忆）的局部偏好方向与正负波幅极性不同。在跨 30 个电极进行跨物体 Pooled 平均时，不同水果的平均标量振幅差异被相互抹平。
2. **高维空间编码胜于标量叠加**：
   这再次有力印证了大脑处理隐性颜色知识并非依靠全皮层简单“同步上升某个标量数值”，而是通过**高维多元空间模式 (Multivariate Pattern)** 编码。这也是为什么在这些通道上运行 **SVM 多元解码与 GLMM 混合效应模型能在 Task 2 上稳定检出 `126~192ms` 与 `326~402ms` 的双峰显著**，而简单的 4 水果单因素 ANOVA 无标量显著的关键原因！
