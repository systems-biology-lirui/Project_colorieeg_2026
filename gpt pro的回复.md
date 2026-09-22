我通读并交叉核对了你上传的 **Introduction、Results、Methods、主图图注和补充图图注**。下面主要审查的是文本内部的论证、分析描述和跨文件一致性；由于没有原始数据、代码和完整图形，我不能判断数值结果本身是否正确。

整体而言，你们目前最有力量、也最能被数据防守的主线是：

> **在无显式平均判断的条件下，平均朝向信息能够很早从猕猴V1群体活动中读取；这种信息可以由固定的朝向选择性群体框架恢复，而不必假设一个时间上明显分离的晚期汇总阶段。**

目前最主要的问题是，正文有时从“V1中可以读取平均值信息”滑向了“V1已经完成平均计算”“这是独立的平均表征”甚至“高级区域只是对其进行后续稳定化”。后一组结论明显强于现有数据。

---

# 一、投稿前必须解决的核心问题

## 1. 当前刺激并不能区分“平均值表征”与“整个朝向分布的重复”

MGv中，同一平均朝向下的12个相对偏转值始终固定为：

−9∘,−7∘,−5∘,−3∘,−2∘,−1∘,1∘,2∘,3∘,5∘,7∘,9∘-9^\circ,-7^\circ,-5^\circ,-3^\circ,-2^\circ,-1^\circ, 1^\circ,2^\circ,3^\circ,5^\circ,7^\circ,9^\circ

不同pattern改变的是这些朝向在空间位置上的分配。因此，当目标帧周期性重复同一平均朝向时，周期性重复的不仅是均值，也是**整个朝向直方图或朝向分布**。

这意味着6.25 Hz信号可以被解释为：

- 平均朝向信息；
    
- 整个朝向分布的中心位置；
    
- 汇聚后的朝向通道活动；
    
- 对同一组绝对组成朝向的周期性响应；
    
- 对跨空间位置保持不变的orientation-energy profile的响应。
    

所以以下表述偏强：

> “V1存在平均朝向表征。”

更准确的是：

> “V1活动中包含跨局部构型保持稳定的平均朝向相关信息。”

或者英文：

> V1 activity contained mean-orientation-related information that generalized across changes in the spatial assignment of constituent orientations.

你们的PVD结果其实与后一种解释更吻合：均值可以作为分布式朝向代码的可读出属性出现，而不一定已经被V1重新编码成一个独立的“mean variable”。

这不是研究的弱点。相反，它可能是最有机制意义的结论：

> **V1不一定需要一套独立的平均值神经元；已有的朝向选择性群体活动在计算上已经足以提供均值信息。**

但整篇论文必须统一采用“information availability / readout sufficiency”的框架，不能一会儿说“可解码”，一会儿又说“V1完成了平均计算”。

---

## 2. SSGv控制不能证明MGv响应不是局部响应的简单组合

Results目前写道，SSGv没有显著的6.25 Hz效应，因此MGv的频率标记响应“不能由各位置单元素序列中已测得的频率标记效应充分解释”；随后又利用SSGnv > SSGv、MGv > MGnv的方向反转，推断MGv响应与集合结构有关。

这个推理仍然过强，原因有三点。

第一，**单个位置不显著不等于该位置的真实效应为零**。多个弱效应可以在群体层面汇聚。

第二，你们比较的是频谱**功率**。功率是傅里叶复数幅值的平方，不具备简单线性可加性：

P(∑ixi)≠∑iP(xi)P\left(\sum_i x_i\right)\neq \sum_i P(x_i)

因此，平均12个单元素功率不能作为完整集合LFP功率的线性预测。

第三，完整多元素刺激还会引入归一化、交叉朝向抑制、空间相互作用和相位同步，这些机制都可能导致SSG与MG比较方向反转，而不必专门诉诸“平均值计算”。

建议把结论收缩为：

> “目标频率响应在孤立呈现各局部元素时不可检测，但在完整异质阵列中稳定出现，说明该信号依赖跨位置的群体刺激结构，而不能由任何单一局部位置携带。”

不要写：

> “排除了局部响应的简单加和。”

除非恢复并完善真正的时域或复频域重构分析。

---

## 3. 核心“平均朝向与单光栅一样快”的统计证据目前不充分

结果中先说四类刺激在约30 ms开始显著解码，随后比较half-peak latency；因为四个条件的bootstrap中位数最多只差2 ms，便得出“平均朝向的可解码起始时间没有明显晚于单个朝向”。

这里存在两个问题。

### Half-peak latency不是onset latency

Methods对half-peak的定义是：解码曲线第一次达到“理论机会水平与峰值之间中点”的时间。

因此，它测量的是：

> 解码信号上升过程中的半峰时间。

它不是：

> 信息首次显著出现的时间。

Results中的“可解码起始时间”应改成：

> “解码信号的half-peak latency”  
> “解码表现的早期上升时间”  
> “the temporal buildup of decodable information”

不能把half-peak直接称为onset。

### 差异不显著或中位数接近，不等于潜伏期等效

如果目前只是发现两个条件差异不显著，或者bootstrap中位数只差2 ms，只能写：

> “未观察到明显的额外延迟。”

还不能严格写：

> “两者一样快”或“具有相同潜伏期”。

要支持后者，建议对

Δt=tMGv−tBSG\Delta t=t_{\mathrm{MGv}}-t_{\mathrm{BSG}}

进行正式的等效性或非劣效检验，并事先定义具有生物学意义的延迟边界，例如5或10 ms。

此外，2 ms本身等于你们的一个采样步长，而解码窗宽为4 ms。因此“最大差2 ms”在现有时间分辨率下，本质上只有一个采样点。

---

## 4. Half-peak bootstrap的重采样单位不正确

Methods写的是：对100次重复解码结果进行有放回重采样，据此构建half-peak latency分布。

但这100次重复解码不是100个独立实验观测，而是：

> 使用同一批trial，只是重新欠采样和划分交叉验证fold得到的100个算法重复。

对这些重复进行bootstrap，只能估计“随机划分CV数据带来的波动”，不能估计：

- trial抽样不确定性；
    
- 神经群体抽样不确定性；
    
- session间变异；
    
- 动物间变异。
    

因此，它不能作为两个神经潜伏期是否相同的生物统计证据。

建议至少改为以下一种：

1. 在trial层面进行bootstrap，并在每次bootstrap内完整重跑解码和latency估计；
    
2. 如果跨多个session采集，以session为重采样单位；
    
3. 对通道和trial进行分层bootstrap；
    
4. 在每只动物内直接构造 Δt\Delta t 的trial-level bootstrap分布。
    

这是你们核心结论最需要优先修复的统计环节。

---

## 5. 解码的统计推断把100次CV重复当成独立样本，存在严重伪重复风险

Methods明确写道，所有解码独立重复100次；随后在真实准确率与shuffle准确率之间做配对t检验和sign-flip permutation，并在最终统计说明中把“重复解码结果”作为固定时间窗检验的配对单位。

这是当前最严重的统计问题之一。

100次CV重复共享同一批神经trial，因此并不是100个独立观测。将其作为t检验或sign-flip的样本，会人为放大有效样本量，产生非常小的P值。

更合适的置换流程是：

1. 在trial层面整体打乱orientation或pattern标签；
    
2. 对每次标签置换完整重跑欠采样、交叉验证和标准化；
    
3. 每个置换只产生一条汇总解码时间程或一个解码矩阵；
    
4. 用这些完整置换结果构建max-cluster零分布。
    

100次交叉验证可以保留，用来降低结果对fold划分的敏感性，但应先在每次标签状态内平均，不能将100次划分当作生物统计样本。

同一问题影响：

- 时间程显著性；
    
- 跨时间泛化矩阵；
    
- within-与cross-pattern比较；
    
- 固定时间窗比较；
    
- half-peak bootstrap；
    
- 图中的SEM。
    

在Neuro、Nature Neuroscience或Neuron审稿中，这一问题很可能被直接指出。

---

## 6. Exp3每个trial包含4次刺激，但交叉验证没有说明按trial分组

Exp3中每个trial连续呈现4张刺激，每张40 ms，间隔480 ms。

解码方法则只说“样本被伪随机划分为五折”，没有说明同一行为trial中的4次刺激是否被放在同一个fold。

如果同一trial中的刺激presentation被分到训练集和测试集，分类器可能利用：

- 同一trial中的慢漂移；
    
- 共同的基线状态；
    
- 唤醒水平；
    
- 眼位；
    
- trial特异噪声；
    
- 刺激序列位置。
    

这会构成group leakage。

建议把“刺激presentation”和“behavioral trial”严格区分，并采用：

> Grouped cross-validation by trial

即一个trial中的4次呈现必须始终进入同一个fold。还应检查不同orientation和stimulus condition在4个serial positions上是否平衡。

---

# 二、“pattern”和“局部细节”的解释目前不准确

## 7. MGv的pattern并不是纯粹的“空间细节”

你之前将pattern解释为：

> 学生和座位不变，但学生与座位之间的对应关系变化。

如果刺激确实只有orientation–position assignment变化，这个比喻是准确的。

但补充图图注写的是：

> MGv不同pattern中，12个子元素的局部朝向和相位均发生变化。

因此，目前的pattern可能同时包含：

- 哪个orientation被分配到哪个位置；
    
- 各局部Gabor的phase；
    
- 完整图像exemplar差异。
    

这意味着pattern decoding不能被直接解释为：

> 个体朝向细节  
> 局部空间组成  
> orientation–position configuration information

除非你确认并明确说明phase在pattern间如何处理。

建议根据真实刺激生成方式二选一：

### 如果MGv中只有orientation–position assignment变化

第一次定义为：

> orientation–position configuration

之后简称：

> configuration

### 如果orientation assignment和phase都变化

则应使用更中性的：

> stimulus-configuration information  
> configuration-specific information  
> image-exemplar information

并明确说明它是orientation–position assignment与local phase的复合因素。

---

## 8. 四种条件的“pattern”不是同一个变量，DI跨条件比较不完全可比

在BSG、Center SSGnv和MGnv中，六种pattern实际上主要对应光栅phase；在MGv中，pattern至少包含orientation–position assignment，补充图注还表明local phase也变化。

因此：

- BSG pattern distance = phase sensitivity；
    
- SSGnv pattern distance = phase sensitivity；
    
- MGnv pattern distance = 多位置共享phase sensitivity；
    
- MGv pattern distance = orientation assignment + phase + exemplar sensitivity。
    

这四者不能统一解释成“局部空间细节”。

所以Results 4中“集合刺激与单个刺激的pattern contribution比较”需要非常谨慎。你们可以比较同一数学定义下的configuration-state distance，但不能直接说：

> 集合刺激比单个刺激更重视局部空间组成。

更稳妥的是：

> “The relative separation of orientation and within-orientation stimulus configurations changed more strongly for multielement arrays than for single gratings.”

---

## 9. pattern标签在不同平均朝向之间没有对应关系，却构建了跨朝向的6×6 pattern RDM

Methods特别说明：

> 10°的pattern 1–6与20°的pattern 1–6之间没有对应关系。

但RSA又将每个平均朝向内的pattern距离按相同pattern编号进行平均，形成6×6 pattern RDM。

如果不同mean orientation下的pattern 1没有共同的结构身份，那么聚合后的：

> pattern 1 versus pattern 2 distance

没有清晰的刺激学意义。编号只是任意标签。

可以保留的量是：

> 每个mean orientation内部所有不同configuration pair的平均距离。

也就是平均within-orientation configuration distance。

但不宜进一步解释聚合后的6×6 pattern RDM中某个特定pattern之间的关系，也不宜对这6个点做具有身份意义的MDS。

---

## 10. within-pattern > cross-pattern不等于“中期平均表征更依赖pattern”

你们的within-pattern训练和测试使用同一套重复图像exemplar，而cross-pattern测试使用新的exemplar。因此二者差异至少包含：

- configuration-specific神经活动；
    
- 图像exemplar重复带来的优势；
    
- phase重复；
    
- 分类器对固定图像特征的记忆；
    
- 真正的均值表征泛化能力。
    

因此，更准确的解释是：

> 中期朝向解码包含了更多configuration-specific information，导致其跨新configuration的泛化下降。

而不是：

> 中期的平均值本身“依赖”pattern。

Methods中已经比较谨慎地说cross-pattern更严格反映相对独立于pattern的朝向信息；Results应保持同样的谨慎。

更重要的是，你们的核心latency比较使用的是普通条件内解码，训练集和测试集都含有相同的六种固定exemplar。要证明“configuration-invariant mean information与BSG一样早”，最有力的分析应当是：

> 用leave-one-configuration-out或cross-configuration decoder计算MGv latency，再与BSG比较。

否则，早期解码只能证明“mean label可预测”，不能证明早期代码已经独立于具体构型。

---

# 三、Results 3–4的时间动态解释偏强

## 11. 跨时间泛化下降不必然代表“表征格式改变”

早期分类器在中期泛化下降，可能来自：

- 神经代码真正旋转或重组；
    
- 中期总体解码信噪比下降；
    
- response gain变化；
    
- configuration信息增强；
    
- 刺激offset response；
    
- 不同神经群体相对贡献变化。
    

所以：

> “中期的神经表征发生变化”

可以保留为工作性解释，但不能仅凭cross-temporal generalization作为确定结论。

你们后续RSA确实提供了补充，但Results中“因为神经距离与真实朝向差异的对应关系减弱，所以跨条件解码下降”仍是因果措辞。

建议把“因为”统一改成：

> “与……相一致”  
> “为……提供了表征几何层面的解释”  
> “was accompanied by”

---

## 12. Exp2A连续刺激数据不适合与Exp3一起进行严格的阶段性时间解释

Exp2A每40 ms出现一张新图，连续呈现72张刺激。

但你们使用Exp2A数据报告pattern decoding在约30 ms出现，并在55–75 ms高于30–50 ms。

在Exp2A中：

- 40 ms之后已经出现下一张图；
    
- 55–75 ms的神经活动同时受到后续刺激影响；
    
- 80–100 ms时已经经历两次图像更新；
    
- 前一帧和后一帧的诱发反应会与目标帧重叠。
    

邻近随机刺激如果与pattern独立，可能主要增加噪声，但仍然使时间阶段的神经解释不再干净。下采样trial数不能解决response overlap。

因此，Exp2A pattern decoding可以用于证明：

> 连续刺激条件下存在可解码的configuration信息。

但不宜将其作为严格的：

> 早期—中期局部细节逐步增强

的主要时间证据。

这一主张最好以Exp3为主。若Exp3 trial不足，宁可将动态细节作为探索性结果，也不要使用连续流数据作强时间因果解释。

---

## 13. “平均朝向和pattern信息此消彼长”并未被当前分析直接证明

Results写道：

> 平均朝向和空间pattern的可解码性在时间上此消彼长。

但当前数据主要显示：

- mean-orientation模型相关或泛化在中期下降；
    
- pattern decoding或pattern distance在某些分析中增强；
    
- DI在中期减小。
    

“此消彼长”暗示二者存在直接资源竞争或负相关，但你们没有直接检验：

Δmean information与Δconfiguration information\Delta \text{mean information} \quad\text{与}\quad \Delta \text{configuration information}

之间的负相关或trade-off。

建议改成：

> “Mean-orientation and configuration-related information followed distinct temporal profiles.”

中文：

> “平均朝向信息与构型相关信息表现出不同的时间进程。”

---

## 14. DI不能被直接解释为两类信息的“相对贡献”

你们把所有orientation pair的平均crossnobis distance与所有pattern pair的平均距离进行比较，并将DI减小解释为pattern信息相对贡献增加。

但两个均值对应的刺激差异不同：

- orientation distance包含10°到90°的多种朝向差异；
    
- pattern distance对应任意configuration exemplar差异；
    
- 二者不是经过等效尺度匹配的两个方差成分。
    

因此DI更准确地表示：

> 在当前距离定义下，orientation-related separation相对于within-orientation configuration separation的优势。

它不是严格的：

> orientation与pattern对神经活动的方差贡献比例。

“贡献”建议统一改为：

> relative representational separation  
> relative distance prominence  
> 相对表征分离程度

而不是“相对信息贡献”。

---

# 四、行为实验的结论需要进一步收敛

## 15. Result 1标题中的“未经训练表现出集合知觉”过强

三只猴虽然没有接受heterogeneous-array专项训练，但此前已经完成了：

- 单线段朝向训练；
    
- 六条同朝向线段的homogeneous-array训练；
    
- 明确的“选择更接近垂直方向刺激”的任务学习。
    

因此不能称为：

> 未经训练  
> 天然能力  
> 先天集合知觉

建议把标题改为：

> **猕猴将朝向判断泛化至未经专项训练的异质阵列**

英文：

> Macaques generalize orientation judgments to heterogeneous arrays without heterogeneous-array training

这既保留亮点，又不会忽略已有任务训练。

---

## 16. 第一个block高于chance不能完全排除block内学习

第一个block包含64个trial。即使第一个block整体为72.1%，动物仍可能在该block前若干十个trial内快速学习。

“不随六个block改变”说明：

> 没有检测到跨block的持续学习趋势。

但不能严格证明：

> 测试一开始的第一批trial就已经掌握。

建议补充：

- 第一个block内按trial bin分析；
    
- 对正确率拟合连续trial-order slope；
    
- 报告前8、16或32个trial的表现；
    
- 或对第一个block前半段和后半段比较。
    

当前表述“在测试开始时已经出现”建议改成：

> “at the level of the first test block”

---

## 17. 极值线索控制只排除了两种策略，不能唯一证明算术平均

四种cue条件结果无显著差异，支持“并非仅依赖最水平或最垂直元素”。这是合理的。

但它不能排除：

- 随机抽样多个元素；
    
- 中位数；
    
- 多数元素的方向；
    
- orientation-energy总和；
    
- 加权平均；
    
- 范围或端点组合；
    
- 只利用中央或显著位置元素。
    

因此：

> “与基于集合整体统计信息的判断一致”

是合适的。

而：

> “动物依赖的是阵列平均值”

则偏强，Introduction最后一段目前用了后者。

建议改成：

> “cue manipulations argued against reliance on either extreme constituent and were consistent with the use of distributed array information.”

要更直接证明算术均值，可以拟合trial-wise item-weight模型，或者设计mean、median、range和extreme cues相互解耦的刺激。

---

## 18. Methods把“最垂直”和“最水平”对应的最大/最小角度写反了

你们定义：

- 水平 = 0°或180°；
    
- 垂直 = 90°；
    
- distractor范围 = 120°–150°。
    

在这个范围内：

- 120°更接近90°，所以更垂直；
    
- 150°更接近180°，所以更水平。
    

但Methods写成：

> 最垂直朝向对应最大角度；最水平朝向对应最小角度。

这是明确错误，应当交换。

---

# 五、PVD部分存在多处逻辑和跨文件不一致

## 19. PVD证明的是条件平均响应的可读性，不是单trial在线读出

Methods显示，在每只动物、每个条件和每个真实mean orientation下，你们先对：

- trial；
    
- pattern/phase；
    
- 30–80 ms时间窗
    

全部平均，然后只进行一次朝向预测。

所以PVD证明的是：

> 条件平均群体响应中包含可由固定朝向权重恢复的mean information。

它不能直接证明：

- 单trial即可准确读出；
    
- 猴子在行为中使用该信号；
    
- 一个下游神经元能在40 ms单次刺激内达到相同精度；
    
- 该机制解释行为正确率。
    

建议把“快速读出”改成：

> “an early population-level readout from condition-averaged responses”

或者只说：

> “computational sufficiency of a fixed orientation-weighted readout.”

---

## 20. “四种条件预测精度相近”不能由差异不显著推出

Results说四种条件间没有显著差异，因此同一群体矢量能够“以相近精度”读出四种条件。

如果没有等效性检验，只能写：

> “we found no significant difference in reconstruction error across conditions.”

不能写：

> “with comparable accuracy”  
> “以相近精度”

否则仍然是“absence of evidence = evidence of equivalence”。

---

## 21. PVD的数据来源在Results、Methods和图注之间不一致

Results说：

> 使用Exp3 BSG估计偏好朝向，预测Exp2A中的BSG、Center SSGnv、MGnv和MGv。

但Exp2A的刺激在前文中只有：

- BSG；
    
- MGnv；
    
- MGv。
    

Center SSGnv是Exp3条件，不是Exp2A条件。

主图图注同样写成用Exp2A反应预测四个条件。

必须明确：

- BSG测试响应来自Exp2A还是Exp3？
    
- CSSGnv来自Exp3还是Exp2B的中心位置？
    
- MGnv和MGv来自哪个实验？
    
- 每个测试数据是否真正独立于偏好朝向估计数据？
    

建议在Methods列出一张清楚的来源表。

---

## 22. PVD缩写与模型名称不一致

你们写的是：

> population vector summation model

但缩写为：

> PVD

如果D代表decoder，应正式命名为：

> population vector decoder

如果名称是summation model，则缩写应更接近：

> PVS或PVSM

全文统一一种。

---

## 23. Introduction中的V1样网络/S1结果在其余文件中完全缺失

Introduction写道：

> 在V1样神经网络中，平均朝向在类简单细胞的S1阶段已经能够被读取。

但当前Results、Methods和主图图注都止于PVD，没有：

- 网络结构；
    
- 输入图像预处理；
    
- S1/C1定义；
    
- 参数来源；
    
- 训练方式；
    
- 读出分析；
    
- 统计检验；
    
- 对应Figure 6。
    

因此目前必须二选一：

1. 暂时从Introduction删除网络结果；
    
2. 补齐完整的Result、Methods、Figure及统计分析。
    

在没有正文支撑的情况下，它不能只存在于Introduction最后一段。

---

# 六、统计与方法描述中的其他高风险点

## 24. SSVEP把电极通道当作独立统计样本

你们先在每个通道拟合模型，再以每只动物的87或84个通道为样本做单样本t检验。

尤其对LFP而言，同一阵列上的通道可能存在明显空间相关和volume conduction，因此87个通道并不等于87个独立复制。

建议优先使用：

- session-level effect；
    
- trial-bin-level permutation，并先跨通道汇总；
    
- spatial block bootstrap；
    
- 或含channel嵌套结构的层级模型。
    

至少在文中不能把通道数等同于独立样本量。

---

## 25. 25 Hz功率作为协变量需要更清楚的统计理由

25 Hz功率本身是刺激条件诱发的结果变量，可能受到target/random条件影响。把它作为协变量相当于控制一个可能位于实验条件影响路径上的变量。

这不一定错误，但需要说明：

- 为什么25 Hz是技术噪声指标而不是实验效应的一部分；
    
- ANCOVA前提是否成立；
    
- target/random × 25 Hz斜率是否相同；
    
- 不加协变量时结论是否保持；
    
- 使用6.25 Hz相对邻频或SNR指标是否得到同样结果。
    

建议至少提供不控制25 Hz的敏感性分析。

---

## 26. MUA滤波和时间平滑描述不足，直接关系到30 ms结论

Methods只写了7阶1000 Hz高通和4 ms移动平均，没有说明：

- 使用单向filter还是`filtfilt`；
    
- 是否为零相位非因果滤波；
    
- spike threshold是正向还是负向；
    
- “信噪比为5.5的电压阈值”究竟是不是5.5倍noise SD；
    
- 阈值事件如何转换为500 Hz MUA；
    
- 4 ms窗口是前向、后向还是中心窗口；
    
- 图上的时间戳表示窗口起点、中心还是终点。
    

如果使用非因果滤波或中心滑窗，刺激后的活动可能被向前扩散，直接影响“约30 ms出现”的结论。必须明确并进行causal-processing敏感性检查。

“应用信噪比为5.5的电压阈值”本身也不是规范说法。通常应写成：

> threshold crossings at −5.5-5.5 times the estimated noise standard deviation

具体取决于你们实际实现。

---

## 27. 缺少session层面的信息

当前Methods没有清楚说明：

- 每只动物每个实验采集多少session；
    
- trial如何跨session合并；
    
- 同一电极通道是否跨天视为同一feature；
    
- 是否对session做标准化；
    
- cross-validation是否跨session平衡；
    
- 是否存在某些orientation只集中于特定session。
    

如果数据跨多天采集，这些信息对解码和PVD都很重要。尤其需要避免分类器把session差异当作orientation差异。

---

## 28. 行为GLMM的文字与模型式不一致

Methods说Block随机截距“嵌套在Monkey内”，但写出的式子是：

(1∣Monkey)+(1∣Block)(1|\mathrm{Monkey})+(1|\mathrm{Block})

这不是显式嵌套模型。如果每只猴的Block编号都是1–6，应使用唯一的Monkey × Block标识，或者类似：

(1∣Monkey)+(1∣Monkey:Block)(1|\mathrm{Monkey})+(1|\mathrm{Monkey:Block})

此外，block分析把只有四个水平的Condition作为随机效应，也需要给出统计理由。

这里建议让熟悉混合模型的统计人员重新检查完整模型矩阵，而不仅是调整文字。

---

## 29. 用P > 0.05支持“没有block效应”和“没有cue效应”仍需谨慎

当前图注和Results多处把P > 0.05解释为：

> 不存在显著变化  
> 不依赖线索

“未检测到差异”可以写，但如果这两个null result承担关键论证，建议增加：

- 效应量与置信区间；
    
- equivalence test；
    
- Bayes factor；
    
- 最小可检测效应。
    

尤其cue manipulation的结论高度依赖null result。

---

# 七、Introduction中需要收回的两处结论

## 30. 开头的“基本神经计算矛盾”可以改为“计算挑战”

“行为依赖分布属性，而早期感觉编码具有局部选择性”并非逻辑矛盾；局部群体编码本身完全可以支持分布统计读出。

建议把：

> 基本的神经计算矛盾

改成：

> 基本的神经计算挑战  
> computational challenge  
> representational mismatch

Introduction当前已提出“新建摘要、直接群体读出、任务转换后出现”三种可能，这个框架是合理的。

---

## 31. 最后一处“高级皮层只是转换早期信号”超出数据

Introduction最后写道：

> 视觉层级是在逐步转换、稳定并赋予行为意义一个已经存在于早期感觉活动中的统计信号，而不是从零开始新建该信号。

你们没有记录V2、V4、LOC或顶叶，也没有比较跨区域信息流，所以不能证明高级层级只是：

- 转换；
    
- 稳定；
    
- 赋予行为意义；
    

更不能证明它们没有新建其他形式的均值表征。

这句话非常适合放在Discussion中作为模型，但在Introduction最后作为主要结论太强。

建议止于：

> “These findings show that mean-orientation information is already available in the early population response of primate V1 and can be recovered by a fixed orientation-selective readout, arguing against the necessity of a temporally segregated late stage for its initial availability.”

然后增加限定：

> “They do not exclude rapid recurrent contributions or further transformations in downstream cortex.”

---

# 八、明确的跨文件错误和编辑残留

这些属于可以直接修改的硬错误。

### 朝向范围不一致

- Results：target mean为10°–170°，步长20°；
    
- Methods：同样为10°–170°；
    
- Figure 2 legend：写成0°–170°，步长20°。
    

“0°–170°、步长20°”本身也不能得到9个等间隔值。应统一为10°、30°……170°。

同时，完整18类刺激有时写10°–180°，有时写0°–170°。两者在180°周期下等价，但标签必须全文统一，建议统一为0°–170°或10°–180°，不要混用。

### Figure 2面板编号错误

主图图注定义：

- I：SSG频谱；
    
- J：SSGnv–SSGv效应比较；
    
- K：MGnv–MGv效应比较。
    

但Results把SSG柱图指向Figure 2K，又把完整集合比较指向Figure 2L；当前图中没有L。

Methods也写“图2I–L”。应全部改为I–K，并逐一核对。

### Figure 2图注保留了编辑问题

主图图注末尾仍有：

> “（有没有多重比较校正）”

必须删除并给出明确答案。

### Supplementary Figure 1误差线自相矛盾

补充图注先说黑点和误差线为EMM及95% CI，后面又说：

> 所有errorbar代表mean ± SEM。

二者不能同时成立。应明确区分观测曲线与模型估计。

### 术语和拼写

- `Muti-grating`应为`multi-grating`；
    
- `Horizon`应为`Horizontal`；
    
- `Center SSGnv`、`CSSGnv`和`center SSGnv`需要统一；
    
- `pattern`、`spatial pattern`、`空间模式`、`空间构型`需要统一；
    
- Figure 3中的`(k)`应改为`(K)`；
    
- Figure 1图注中的“实现圆圈”应为“实线圆圈”；
    
- `PVD`名称需要统一；
    
- Results中仍有“参考大猩猩的文章”“参考”等占位符；
    
- RSA Methods中仍有“参考文献”占位符；
    
- `session`和`block`混用，应根据实际单位统一。
    

---

# 九、我建议你们把全文的结论强度固定成四级

## 数据直接支持

1. 猕猴可以把已学习的朝向判断泛化至未经专项训练的异质阵列。
    
2. V1 LFP能够追踪跨空间重排而周期性复现的朝向分布/平均朝向相关信息。
    
3. MGv的mean-orientation label可以很早从V1 MUA中解码。
    
4. 固定的朝向偏好加权读出可以从条件平均V1响应中恢复阵列均值。
    

## 数据支持，但需要修复统计后才能成为主结论

1. MGv mean information的时间进程与BSG orientation相当。
    
2. 配置相关信息在后续阶段相对增强。
    
3. 单一和集合刺激可以由共同的朝向选择性读出框架解释。
    

## 可以作为Discussion中的解释

1. V1早期群体代码可能已经足以向下游提供统计量。
    
2. 后续皮层区域可能进一步增强、稳定或行为化这一信号。
    
3. 中期变化可能反映局部构型信息的增加或循环加工。
    

## 当前数据不能证明

1. V1单独完成了算术平均计算。
    
2. 完全不存在快速反馈。
    
3. V1中没有专门的mean-selective neurons。
    
4. 行为猴实际使用了你们在另外两只猴中记录到的V1信号。
    
5. pattern代表纯粹的个体朝向细节。
    
6. 高级区域不再新建其他形式的统计表征。
    
7. 四种条件具有统计学等效的潜伏期或PVD精度。
    

---

# 十、优先修改顺序

第一优先级不是语言润色，而是四个分析与定义问题：

1. **重新建立解码置换统计，不能把100次CV重复当独立样本；**
    
2. **用trial/session-level bootstrap和等效性或非劣效检验重新检验MGv–BSG latency；**
    
3. **明确MGv pattern是否包含phase，并重新界定configuration analysis；**
    
4. **解决普通解码共享exemplar的问题，最好用held-out configuration的latency支撑核心结论。**
    

随后处理：

5. 收缩SSGv控制的机制结论；
    
6. 限制Exp2A pattern timecourse的解释；
    
7. 修正PVD数据来源和单trial/平均响应措辞；
    
8. 补齐或删除V1样网络模型；
    
9. 统一图号、角度、术语和统计说明；
    
10. 将高级区域转换、反馈和行为意义等内容放入Discussion。
    

目前的稿件不是“故事不成立”，而是**故事的强度超过了部分分析能够支持的强度**。把主张收敛为“early availability + fixed population readout sufficiency”，并修复解码推断单位和configuration定义后，主线会明显更坚固。