# 记忆颜色多电极解码与组水平 GLMM 显著性分析计划 (Step2_2)

本计划旨在实现 `step2_2_memory_color_decoding_glmm.py` 脚本，对三套不同的电极选择方案进行记忆颜色（红记忆 vs 绿记忆）的多通道分类解码，计算每个时间步的正确率，并运用二项分布广义线性混合模型（GLMM）进行组水平的统计显著性时间窗识别。

## User Review Required

> [!IMPORTANT]
> **分析逻辑与核心设计**：
> 1. **三套电极选择方案**：
>    - **方案 1 (Strategy 4)**：仅使用主要筛选电极表中，ERP 或 HG 匹配了策略 4 的电极。
>    - **方案 2 (Union)**：使用主要筛选电极表中的并集电极（即 `ERP_Selected == True` / `HG_Selected == True` 的通道）。
>    - **方案 3 (Memory Color Sig)**：仅使用上一步（Step 2_1）计算出的在记忆颜色中表现显著（Mean 或 Cont 之一显著）的电极。
> 2. **解码配对设计 (Cross-object Decoding)**：
>    - 红色记忆：`123`（灰色草莓）、`133`（灰色西瓜）。
>    - 绿色记忆：`103`（灰色卷心菜）、`113`（灰色猕猴桃）。
>    - 4 种配对交叉训练测试：
>      - 配对 1：`(123, 103)` 训练，在 `(133, 113)` 上测试；
>      - 配对 2：`(123, 113)` 训练，在 `(133, 103)` 上测试；
>      - 配对 3：`(133, 103)` 训练，在 `(123, 113)` 上测试；
>      - 配对 4：`(133, 113)` 训练，在 `(123, 103)` 上测试。
>    - 在每个时间步，收集 3 个被试、4 种配对中所有测试 trial 的预测正确/错误（1/0）向量，用作 GLMM 输入。
> 3. **多核并行与真实性检测**：
>    - 使用 `joblib.Parallel` 配备 `n_jobs=-1` 对 750 个时间点进行并发 SVM 训练与测试。
>    - 脚本开始时，将获取系统逻辑核心数，并运行一个小规模并行 benchmark 计算并输出所耗时间，向日志报告以检测并证实确实启用了多核运行。
> 4. **二项分布 GLMM 群组统计**：
>    - 在每个时间步，合并 3 被试的全部测试试次（共约数百个对错 0/1 样本），并引入被试作为随机效应：
>      $$\text{logit}(P(Y=1)) = \beta_0 + b_{0,\text{Subject}}$$
>    - 使用 `statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM` 的变分贝叶斯方法 (`fit_vb()`) 快速求解，检验 $\beta_0 > 0$（单尾 $p < 0.05$ 且持续窗口 $>20\,\text{ms}$ 为显著）。
> 5. **绘图设计**：
>    - 三套电极方案各产生一张 ERP 和一张 HG 曲线大图（共 6 张大图）。
>    - 图中用细虚线画出三个被试（test001, test002, test003）的个体解码曲线，用粗实线画出群组平均解码曲线。
>    - 将 GLMM 检验显著（$p < 0.05$ 且持续 $\ge 20$ms）的时间区间以淡橙红色半透明阴影（`ax.axvspan`）在图中标出，并在显著时间区间的顶部画出一条粗红线进行突出标注。
> 6. **数据保存**：在画图前，把每个时间步上所有被试的正确率以及 GLMM 估计均值、z值、p值全部保存为 CSV 和 Excel 表格，存放于 `analyse_0617/doc/`。

---

## Open Questions

> [!NOTE]
> 1. **方案 3 显著电极不足的防错处理**：
>    如果个别被试在某些脑电特征（如 HG）下显著的 memory_color 通道过少（例如仅有 2 个），SVM 依然可以使用 2 个通道特征进行解码，但当为 0 时无法特征化。在目前数据中，我们已经使用 Python 验证过 test001, test002, test003 在 ERP / HG 中最少也有 2 个通道符合方案 3，可以安全运行。若通道数为 0 我们已编写了跳过警告逻辑以防崩溃。
> 2. **SVC 参数的一致性**：
>    我们维持使用 `SVC(kernel='linear', C=1.0)` 和 `StandardScaler` 进行空间通道的特征标准化，确保与项目前人研究方法完全对齐。

---

## Proposed Changes

### 多通道记忆颜色解码与 GLMM 统计脚本

#### [NEW] [step2_2_memory_color_decoding_glmm.py](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/code/step2_2_memory_color_decoding_glmm.py)
1. **并行有效性检测**：
   - 使用 `multiprocessing.cpu_count()` 检测并输出计算资源，运行 200 个模拟 dummy 任务记录用时，提供并行执行的证据。
2. **三套方案电极提取**：
   - 方案 1：筛选 `ERP_Strategies_Matched` / `HG_Strategies_Matched` 中含 `'4'` 的通道。
   - 方案 2：筛选 `ERP_Selected` / `HG_Selected` 为 `True` 的主要筛选电极并集。
   - 方案 3：读取 Step 2_1 的明细表，筛选 `Sig_Category != 'Non_Sig'`（即 `Is_Mean_Sig` 或 `Is_Cont_Sig` 为 `True`）的显著记忆通道。
3. **SVM 解码管线与 Trial 收集**：
   - 载入 `task2` 数据，对 4 种记忆颜色红绿配对循环在每个时间点进行线性 SVM 预测，提取 test trial 的 0/1 准确性向量。
4. **GLMM 统计检验**：
   - 整合所有测试 trials（包含 Subject 标签），在 750 个时间点循环使用 `BinomialBayesMixedGLM` 进行混合截距估计，求出 Wald z-test 单尾 $p$ 值。
5. **曲线与阴影绘制**：
   - 按格式要求绘制个体曲线、均值粗线、GLMM 显著阴影和顶部粗红线（共 6 张图），保存至 `result/select_channel/decoding/`。
6. **历史备份**：
   - 拷贝本实施计划至 `analyse_0617/plan_backups/implementation_plan_step2_2.md`。

---

## Verification Plan

### Automated Tests
- 运行 `/home/lirui/anaconda3/envs/lr2026/bin/python color_cognition_pipeline/analyse_0617/code/step2_2_memory_color_decoding_glmm.py`。
- 检查 `analyse_0617/doc/` 目录下生成的三套方案的 CSV 和 Excel 绘图数据表是否齐全。
- 确认 `analyse_0617/result/select_channel/decoding/` 目录下正确生成 6 张包含 GLMM 阴影和多曲线的大图。

### Manual Verification
- 检查控制台日志中是否输出多核 benchmark 和 CPU 核心检测信息。
- 确认图表中的时间轴范围对齐在 `[-200, 800]` ms 绘图区间，且包含正确的图例（被试虚线、均值实线、GLMM 显著阴影）。
