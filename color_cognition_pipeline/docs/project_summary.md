# 🧠 颅内脑电 (iEEG) 颜色认知加工机制项目全流程总结报告 (5 被试全量版)

本报告系统回顾和总结了本项目中关于人脑对“真实物理颜色”与“隐含颜色联想知识”加工的电生理机制研究所进行的所有分析与成果，已完成全量 **5 名被试**（`test001`, `test002`, `test003`, `test005`, `test006`）的完整补齐与组水平二项 GLMM 混合效应拟合。

---

## 📅 一、 项目背景与实验范式

本项目旨在探究人脑在处理**真实物理刺激颜色**（Task 3）与**灰色物体的隐含颜色知识检索**（Task 2）时，其神经加工在**时间尺度上**的动态演变和在**空间大脑皮层**（枕叶-颞叶腹侧视觉通路）的层级传导机制。

### 1. 被试与电极配置
数据来源于 **5 名** 植入颅内深度电极（sEEG）的癫痫患者（`test001`, `test002`, `test003`, `test005`, `test006`），其电极触点主要覆盖了枕中回、枕下回、梭状回、颞下回、颞中回以及前部的颞极（Temporal Pole）等腹侧视觉通路及语义表征的核心脑区。

### 2. 实验任务设计
- **Task 3 (Real Color Task)**：呈现红色、绿色、蓝色、黄色的纯色块。旨在评估视网膜接收到的纯物理颜色刺激在早期视皮层激发的生理电信号。
- **Task 2 (Memory Color Task)**：呈现本身具有强烈颜色属性的灰色水果/蔬菜图片（如灰色草莓、灰色猕猴桃、灰色卷心菜等）。在此范式下，被试眼部接收到的仅为无彩色（灰色）刺激，但其大脑内部会主动检索并联想起这些物体在真实世界中所对应的颜色常识（红色或绿色），即**隐性颜色知识激活**。

---

## 🛠️ 二、 数据预处理与特征提取管线

我们设计了完整、严密的 MATLAB-Python 跨平台信号处理流水线：
1. **重参考 (Re-referencing)**：对 sEEG 原始信号进行**全脑通道平均重参考 (CAR)**，滤除大范围环境工频电磁噪声和电极公共基底漂移。
2. **滤波 (Filtering)**：应用低通和高通双向零相位 Butterworth 滤波器，将信号划分为：
   - **ERP (时域低频成分)**：滤波范围 `0.1 Hz - 30 Hz`，保留经典诱发电位。
   - **High Gamma (高频能量成分)**：利用 `extract_new_highgamma.m` 进行 `60 Hz - 150 Hz` 多带重叠滤波，使用 Hilbert 变换提取信号包络并降采样。
3. **Epoch 划分与清理**：以刺激呈报时刻（$t=0\,\text{ms}$）为基准，截取 **`-500 ms` 至 `1000 ms`** 共 $1500\,\text{ms}$（750 个时间步，降采样率 $500\,\text{Hz}$）的 Epoch 数据，并剔除含有坏通道及异常过大伪迹的试次。
4. **Trial-wise 基线减除**：对每个 trial 每个通道，分别减去该 trial 在基线期（`-500 ms - 0 ms`）的均值，极大消除了通道间的低频漂移与基线期的不平衡。

---

## 🔍 三、 5 被试电极筛选与全脑空间分布表

在全量 5 个被试的 38 个色块选择性敏感电极分布如下：

| 被试 | 选中色块敏感电极通道列表 | 通道总数 | 核心特征脑区覆盖 |
| :--- | :--- | :--- | :--- |
| **test001** | `G11`, `F9`, `B5`, `D7`, `E7`, `E8`, `A5`, `E6` | 8 | 梭状回, 颞中回, 颞极 |
| **test002** | `C7`, `B6`, `C8`, `C1`, `A2`, `A7`, `A4`, `B7`, `C9`, `A9`, `F3`, `B5`, `B1`, `F8` | 14 | 枕下回, 枕中回, 梭状回, 颞下回, 颞极 |
| **test003** | `D11`, `H14`, `G11`, `G10`, `G12`, `G14`, `H15`, `G9`, `G8` | 9 | 梭状回, 颞下回 |
| **test005** | `E10`, `A8`, `E13`, `A7`, `A1`, `E14` | 6 | 梭状回, 颞中回, 颞极 |
| **test006** | `A9` | 1 | 梭状回 |
| **全组汇总** | **5 被试共计 38 个选择性敏感通道** | **38** | **腹侧视觉通路全覆盖** |

---

## 📈 四、 5 被试二项 GLMM 混合效应拟合核心统计发现

通过在 750 个时间点上建立包含被试随机截距（Random Intercept by Subject）的 Binomial GLMM 模型，我们取得了极其明确的显著性时间窗发现（$p < 0.05$）：

1. **Task 2 颜色知识解码 (Red vs Green Memory)**
   - 后部集群 (`POSTERIOR`, Y 轴坐标区间 `[-90.3, -42.3] mm`) 在 **`126 ms ~ 192 ms`** ($p < 0.05$) 表现出显著的早期提取，并于 **`326 ms ~ 402 ms`** ($p < 0.05$) 表现出晚期表征维持。
2. **Task 3 物理纯色块解码 (Red vs Green Block)**
   - 物理颜色刺激提取极快，GLMM 在 **`54 ms ~ 74 ms`** 与 **`118 ms ~ 142 ms`** 均达到显著水平。
3. **跨范式 1D 跨任务表征迁移 (Color Block Train $\rightarrow$ Gray Memory Test)**
   - 在 **`150 ms ~ 194 ms`** 与 **`218 ms ~ 252 ms`** 检出强烈的显著跨范式泛化，证实真实感知颜色与隐性颜色知识共享了底层神经编码。
4. **`color_with_sti` 知识先验电极分析**
   - 包含先验知识提示的通道在 **`152 ms ~ 180 ms`** 与 **`216 ms ~ 242 ms`** 检出显著的颜色检索效应。

---

## 🌐 五、 单文件 5 被试 HTML 交互式汇总报告

我们已成功将 5 个被试的所有脑皮层电极定位、3D 玻璃脑、策略分布饼图、GLMM 混合效应解码曲线及跨范式时间泛化 2D 热图编译为单文件交互式 HTML 报告：

- 交互报告位置：
  - [final_report/color_ieeg_0617_five_subject_interactive_report.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/result/final_report/color_ieeg_0617_five_subject_interactive_report.html)
  - [run_5subjects_original/report/color_ieeg_0617_five_subject_interactive_report.html](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/analyse_0617/run_5subjects_original/report/color_ieeg_0617_five_subject_interactive_report.html)

---

## 📁 六、 官方代码库与文档备份

所有调用的代码与文档均已在项目 `color_cognition_pipeline/docs/` 目录下完成双重备份：
1. `docs/implementation_plan.md`
2. `docs/task.md`
3. `docs/project_summary.md`
4. `docs/walkthrough.md`
