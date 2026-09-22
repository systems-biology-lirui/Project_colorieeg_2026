# Cross-decoding 中“只解码出一种记忆颜色”的现象与后续分析

## 1. 当前现象

在 `Task3 纯色色块 → Task2 灰度水果` 的 cross-decoding 中，部分电极出现明显的类别不对称：

- 有些电极能较稳定地把 **绿色记忆水果**（白菜、猕猴桃）判为绿色，但对 **红色记忆水果**（草莓、西瓜）判断较差；
- 另一些电极则相反，红色记忆水果判断较好，而绿色记忆水果判断较差；
- 因此总体 balanced accuracy 可能接近 chance，但单类别 accuracy/recall 呈明显分离。

这类结果目前只能描述为：

> **cross-decoding 的类别不对称（class-asymmetric generalization）**

不能直接解释为“该电极只编码红色记忆”或“只编码绿色记忆”。

---

## 2. 首先需要区分的几种可能

### A. 跨任务分类边界偏移

Task3 分类器可写为：

`score = w·x + b`

其中：

- `w`：Task3 中 red-green 的表征方向；
- `b`：分类阈值。

Task2 与 Task3 的整体神经活动分布可能不同，因此即使 Task2 中：

`red-memory > green-memory`

仍可能因为整体 offset 导致两类 trial 都落在 Task3 分类边界的同一侧。

这会产生：

- 一类 recall 很高；
- 另一类 recall 很低；
- overall balanced accuracy 接近 0.5。

### B. 真正的类别不对称表征

也可能确实只有一个记忆颜色类别与对应的真实颜色模式更相似。

例如：

- 白菜、猕猴桃稳定靠近 Task3-green pattern；
- 草莓、西瓜没有稳定靠近 Task3-red pattern。

这种情况需要在排除分类阈值偏移后才能考虑。

### C. 单个物体驱动

表面上看像“绿色记忆”或“红色记忆”效应，但实际上可能只是：

- cabbage 特别强；
- strawberry 特别强；

而同组另一种水果并不一致。

这种情况不能解释为 memory-color generalization。

---

## 3. 后续分析

### 3.1 保存 trial-level decision score

不要只保存 predicted label / accuracy。

对每个 Task2 trial 保存 Task3 classifier 的连续输出：

`decision_score`

统一规定：

- 正值 = Red direction
- 负值 = Green direction

分别画出：

- Strawberry
- Watermelon
- Cabbage
- Kiwi

的 score 分布或时间曲线。

重点检查：

`mean(Strawberry, Watermelon) > mean(Cabbage, Kiwi)`

而不仅仅看是否跨过分类阈值。

---

### 3.2 同时计算 AUC

对 Task3 → Task2 cross-decoding 同时报告：

- Balanced Accuracy
- AUC

解释：

- **Balanced Accuracy**：Task3 的完整分类边界能否直接迁移到 Task2；
- **AUC**：Task3 的 red-green 表征轴能否正确排序 Task2 中的 red-memory 与 green-memory trial。

如果出现：

`Balanced Accuracy ≈ 0.5`

但：

`AUC > 0.5`

说明可能存在：

> **shared representational axis，但存在 cross-task boundary shift。**

---

### 3.3 检查四种水果的一致性

理想 shared memory-color pattern 应至少满足方向一致：

- Strawberry > Cabbage
- Strawberry > Kiwi
- Watermelon > Cabbage
- Watermelon > Kiwi

并且同记忆颜色内部应尽量一致：

- Strawberry ≈ Watermelon
- Cabbage ≈ Kiwi

如果只有一个水果驱动结果，则优先解释为 object-specific effect。

---

### 3.4 检查预测偏置

对每个时间点统计 classifier 在 Task2 上预测：

- Red 的比例
- Green 的比例

如果某段时间几乎所有 trial 都被预测成同一类，那么当前“单侧解码”更可能是 **prediction bias / boundary shift**，而不是单一记忆颜色的真实选择性。

---

### 3.5 检查 Task3 自身分类器是否平衡

在 Task3 内部确认：

- Red recall
- Green recall
- confusion matrix
- decision score distribution

如果 Task3 本身就明显偏向某一类，则 cross-decoding 中的类别不对称可能只是训练分类器偏置的延续。

---

## 4. 最终判断逻辑

### 情况 1

`BA低 + AUC高 + 两组水果方向一致`

→ 更支持：

**shared red-green representation axis + cross-task boundary shift**

### 情况 2

`BA高/AUC高 + 两种同色水果都一致`

→ 更支持：

**稳定的 memory-color / hue shared representation**

### 情况 3

`只有一个水果明显偏向正确方向`

→ 更支持：

**object-specific effect**

### 情况 4

`所有水果都被预测成同一类别`

→ 更支持：

**classifier bias / cross-task distribution shift**

---

## 5. 下一步优先输出

对每个候选电极输出：

1. Task2 四种水果的 trial-level decision score；
2. 四种水果的平均 score 时间曲线；
3. Red-memory vs Green-memory 的 AUC；
4. Balanced Accuracy；
5. 每类 recall；
6. confusion matrix；
7. 每个时间点预测 Red/Green 的比例。

优先用这些结果判断当前看到的“只解码出一种记忆颜色”究竟是：

**真实类别不对称、单物体效应，还是跨任务分类边界偏移。**
