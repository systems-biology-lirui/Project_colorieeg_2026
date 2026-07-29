# CCEP 独立运行目录

这个目录是从原始 `ccepcode/` 复制出来的自包含版本，目标是把代码、输入数据、定位元数据和中间结果都集中到一个目录下，避免再依赖项目根目录中的 `ccepcode/`、`processed_data/`、`feature/` 和 `newanalyse/`。

## 目录结构

- `code/`
  - `Sec1_ccep_preanalyse.m`：CCEP 连续数据预处理，生成 ERP/TFA epoch。
  - `Sec2_ccep_preprocess_roi_features.m`：把 epoch 数据整理成 ROI 特征文件。
  - `Sec3_ccep_electrode_response_stats.py`：做逐电极统计并输出图表、表格和 npz。
  - `Sec4_ccep_batch_run.py`：批处理入口，只调用当前 `code/` 目录中的脚本。
  - `get_roi_map.m`：ROI 映射辅助函数。
  - `runtime_config.py`：运行时配置加载辅助函数。
  - `ccep_batch_config_example.json`：批处理示例配置。
- `data/raw/test1/`
  - 已复制 `ccep.set` 和 `ccep.fdt`。
- `data/metadata/test001/`
  - 已复制 `test001_ieegloc.xlsx`。
- `workspace/processed/test001/ccep/`
  - 已复制现有的 CCEP 预处理输出，方便直接从第二步继续跑。
- `workspace/feature/ccep_erp/test001/`
  - 已复制现有 ERP ROI 特征文件。
- `workspace/feature/ccep_tfa/test001/`
  - 已复制现有 TFA ROI 特征文件。
- `workspace/result/`
  - 用于保存统计结果；如果原项目里已有 `result/ccep/test001/`，也会一起复制进来。

## 路径改动

这个独立版本已经把路径改成相对当前目录的本地路径：

- 原始输入：`data/raw/<raw_subject_dir>/ccep.set`
- 定位文件：`data/metadata/<subject>/<subject>_ieegloc.xlsx`
- 预处理输出：`workspace/processed/<subject>/ccep/`
- ROI 特征：`workspace/feature/ccep_<modality>/<subject>/`
- 统计结果：`workspace/result/ccep/<subject>/<modality>/`

## 运行方式

### 1. MATLAB 单步运行

在 MATLAB 中运行：

- `run('code/Sec1_ccep_preanalyse.m')`
- `run('code/Sec2_ccep_preprocess_roi_features.m')`

### 2. Python 单步运行

在当前目录下运行：

```bash
python code/Sec3_ccep_electrode_response_stats.py
```

### 3. 批处理运行

```bash
python code/Sec4_ccep_batch_run.py --config code/ccep_batch_config_example.json
```

## 额外依赖

- MATLAB 端仍然需要 EEGLAB。
- `Sec1_ccep_preanalyse.m` 会优先尝试 `external_tools/eeglab/`；如果该目录不存在，则退回到当前 MATLAB 已经配置好的 EEGLAB。
- Python 端需要 `numpy`、`scipy`、`pandas`、`matplotlib`。

## 说明

- 原始 `ccepcode/` 没有被修改，这个目录是独立副本。
- 复制的是当前 `test001` / `test1` 这套 CCEP 数据和已有中间结果；如果后续要切换被试，需要把对应的原始数据和定位文件也放到同样的本地结构里。
