# Notebook-driven workflow

整个预处理部分只保留两个 Notebook，底层可复用函数统一放在 `../pipeline/`，MATLAB 原始数据转换函数放在 `../matlab/`。

1. `00_rawdata_to_seegdata.ipynb`：查看 rawdata→seegdata 映射；默认不执行转换，只有明确打开开关后才调用通用 MATLAB 函数。
2. `01_preprocess_and_export.ipynb`：一个 Notebook 完成事件/条件检查、电刺激定位表检查、信号与定位交集、全长滤波、坏道决定、epoch 导出、HDF5 验证、Task 2 条件读取和均值/阴影示例图。

重要的可复用入口：

- `pipeline.signal_processing.filter_continuous`：全时间长度的去 DC、重采样、1–200 Hz 带通和 50/100/150 Hz 陷波；
- `pipeline.epoch_plots.plot_epoch_mean_shading`：单条件 epoch 均值与 SEM/SD 阴影；
- `pipeline.epoch_plots.plot_hdf5_conditions`：从 HDF5 直接叠加多个条件；
- `metadata/manual_channel_decisions.csv`：持久化的人工坏道决定；
- `metadata/stimulation_behavioral_annotation.csv`：逐刺激对的行为学证据分类；
- `metadata/stimulation_behavioral_electrodes.csv`：逐接触点的电刺激标注摘要。

Notebook 默认不会删除原始数据，也不会重新覆盖 `.set/.fdt`；只有显式打开对应开关时才会执行转换或重建输出。
