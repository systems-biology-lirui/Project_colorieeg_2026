# Analyse 0720：fMRI Color-Patch—SEEG 独立分析管线

本目录是从 `seegdata` 重新开始的独立管线，不覆盖 `analyse_0617` 或
`processed_data`。当前 20-mm 批处理入口覆盖 `test001`–`test006`，逐被试保存结果，
再进入组水平汇总；没有 20-mm 实际覆盖的被试只保留覆盖状态，不进入信号统计。

## 目录

```text
analyse_0720/
├── config.py                 # 路径、坐标、触发码和计算参数
├── utils/                    # 读取、窗口化、解码、统计和绘图复用函数
├── notebooks/                # 分块标注的执行入口
├── intermediate/             # 旧版 10-mm 中间数据
├── result/                   # 旧版 10-mm 探索结果
├── intermediate_20mm/        # 20-mm ERP/HG clean 数据
├── result_20mm/subjects/     # 20-mm 独立被试结果
├── result_20mm/group/        # 20-mm 组统计和图
└── reports/                  # 自动生成的中文结果报告
```

## 计算效率原则

- 原始 run 按被试/run 逐个处理，不同时载入三个连续文件。
- 预处理结果持久化，后续 notebook 不重复滤波或 Hilbert 提取。
- 时间解码使用向量化时间窗特征、固定 CV 划分和并行 permutation。
- 每个 subject 的结果单独保存；组分析只读取已完成的 subject artifact。
- 所有结果保存配置快照、随机种子和输入文件清单。

旧版 10-mm notebook 保留用于追溯；新分析直接运行下面的 20-mm 批处理入口。

当前环境如果尚未安装 MNE、scikit-learn、pymatreader 等依赖，先完成环境
安装后再运行预处理 notebook；配置和目录本身不依赖这些包。

## 20-mm 批处理入口

1. `20_batch_20mm.ipynb`：覆盖检查、ERP/HG 预处理、坏段 QC、逐被试 decoding。
2. `21_group_statistics_20mm.ipynb`：组曲线、sign-flip 探索性汇总和 publication-style PNG/PDF。

绘图由 `utils/plotting_science.py` 统一控制，保存 600 dpi PNG 和可编辑 PDF；
每张图也在 notebook 执行结果中保留 inline 输出。

## 全通道 functional-selection 分支

新分支不使用 fMRI 坐标预筛选，输出位于 `intermediate_all_channels/` 和
`result_all_channels/`。执行顺序为：

1. `30_preprocess_all_channels.ipynb`
2. `31_color_gray_screening.ipynb`
3. `32_color_select_distribution.ipynb`
4. `33_subject_level_decoding.ipynb`
5. `34_virtual_subject_decoding.ipynb`
6. `35_spatial_group_decoding.ipynb`
7. `36_all_channel_summary_no_tabulate.ipynb`

当前快速模式统一使用 100 次 permutation。由于 100 次置换的 p 值分辨率限制，
筛选表同时保留 permutation p/q 和 Welch 参数检验 p/q；下游 decoding 只使用
功能筛选得到的电极，不会回退到全部通道。
