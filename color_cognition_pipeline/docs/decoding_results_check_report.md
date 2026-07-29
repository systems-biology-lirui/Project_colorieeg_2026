# 被试解码结果完整性检查报告

本报告对所有被试（`test001`、`test002`、`test003`）在所有解码（decoding）类型下的 ERP 和 Subband 以及其对应的 `colorwithsti` 和 `type1` 电极组合进行详细的完整性检查。

## 1. 电极选择情况汇总
| 被试名称 | `colorwithsti` 电极列表 (数量) | `type1` 电极列表 (数量) | `temporal_pole` 电极列表 (数量) |
| --- | --- | --- | --- |
| **test001** | D4, D5, D6, G5, G6, G7 (6个) | B5, C8, C10, F9, G10, G11, H9 (7个) | E6, E7 (2个) |
| **test002** | D1, D2, D3, B2 (4个) | A3, B1, F3, F5 (4个) | G1, G2, G3, G4, G5, G6, G7, G8 (8个) |
| **test003** | G3, G4, H2, H3, H4, H5, H11, H12 (8个) | A7, A12, G11, G12, H11, H12 (6个) | 无 (0个) |

## 2. 检查结果明细

我们对每个被试在 `images/{subject}/decoding/` 下各个子目录生成的文件进行了扫描，所有组合的文件完整性如下：

### test001
- **Memory Pairs Decoding**
  - [x] ERP x colorwithsti: [erp_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/memory_pairs/erp_colorwithsti.png) / `_acc.npy` (生成时间: 2026-06-15 10:58)
  - [x] ERP x type1: [erp_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/memory_pairs/erp_type1.png) / `_acc.npy` (生成时间: 2026-06-15 10:58)
  - [x] Subband x colorwithsti: [subband_60_150_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/memory_pairs/subband_60_150_colorwithsti.png) / `_acc.npy` (生成时间: 2026-06-15 11:00)
  - [x] Subband x type1: [subband_60_150_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/memory_pairs/subband_60_150_type1.png) / `_acc.npy` (生成时间: 2026-06-15 11:00)
- **True vs False Decoding**
  - [x] ERP x colorwithsti: [erp_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/true_false/erp_colorwithsti.png) / `_acc.npy` (生成时间: 2026-06-15 10:58)
  - [x] ERP x type1: [erp_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/true_false/erp_type1.png) / `_acc.npy` (生成时间: 2026-06-15 10:59)
  - [x] Subband x colorwithsti: [subband_60_150_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/true_false/subband_60_150_colorwithsti.png) / `_acc.npy` (生成时间: 2026-06-15 11:00)
  - [x] Subband x type1: [subband_60_150_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/true_false/subband_60_150_type1.png) / `_acc.npy` (生成时间: 2026-06-15 11:00)
- **Time Generalization Decoding**
  - [x] ERP x colorwithsti: [erp_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/time_generalization/erp_colorwithsti.png) / `_tgm.npy` (生成时间: 2026-06-15 10:59)
  - [x] ERP x type1: [erp_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/time_generalization/erp_type1.png) / `_tgm.npy` (生成时间: 2026-06-15 11:00)
  - [x] Subband x colorwithsti: [subband_60_150_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/time_generalization/subband_60_150_colorwithsti.png) / `_tgm.npy` (生成时间: 2026-06-15 11:01)
  - [x] Subband x type1: [subband_60_150_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test001/decoding/time_generalization/subband_60_150_type1.png) / `_tgm.npy` (生成时间: 2026-06-15 11:01)

### test002
- **Memory Pairs Decoding**
  - [x] ERP x colorwithsti: [erp_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/memory_pairs/erp_colorwithsti.png) / `_acc.npy` (生成时间: 2026-06-15 11:01)
  - [x] ERP x type1: [erp_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/memory_pairs/erp_type1.png) / `_acc.npy` (生成时间: 2026-06-15 11:01)
  - [x] Subband x colorwithsti: [subband_60_150_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/memory_pairs/subband_60_150_colorwithsti.png) / `_acc.npy` (生成时间: 2026-06-15 11:03)
  - [x] Subband x type1: [subband_60_150_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/memory_pairs/subband_60_150_type1.png) / `_acc.npy` (生成时间: 2026-06-15 11:03)
- **True vs False Decoding**
  - [x] ERP x colorwithsti: [erp_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/true_false/erp_colorwithsti.png) / `_acc.npy` (生成时间: 2026-06-15 11:02)
  - [x] ERP x type1: [erp_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/true_false/erp_type1.png) / `_acc.npy` (生成时间: 2026-06-15 11:02)
  - [x] Subband x colorwithsti: [subband_60_150_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/true_false/subband_60_150_colorwithsti.png) / `_acc.npy` (生成时间: 2026-06-15 11:03)
  - [x] Subband x type1: [subband_60_150_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/true_false/subband_60_150_type1.png) / `_acc.npy` (生成时间: 2026-06-15 11:03)
- **Time Generalization Decoding**
  - [x] ERP x colorwithsti: [erp_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/time_generalization/erp_colorwithsti.png) / `_tgm.npy` (生成时间: 2026-06-15 11:02)
  - [x] ERP x type1: [erp_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/time_generalization/erp_type1.png) / `_tgm.npy` (生成时间: 2026-06-15 11:02)
  - [x] Subband x colorwithsti: [subband_60_150_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/time_generalization/subband_60_150_colorwithsti.png) / `_tgm.npy` (生成时间: 2026-06-15 11:04)
  - [x] Subband x type1: [subband_60_150_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test002/decoding/time_generalization/subband_60_150_type1.png) / `_tgm.npy` (生成时间: 2026-06-15 11:04)

### test003
- **Memory Pairs Decoding**
  - [x] ERP x colorwithsti: [erp_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/memory_pairs/erp_colorwithsti.png) / `_acc.npy` (生成时间: 2026-06-15 11:04)
  - [x] ERP x type1: [erp_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/memory_pairs/erp_type1.png) / `_acc.npy` (生成时间: 2026-06-15 11:04)
  - [x] Subband x colorwithsti: [subband_60_150_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/memory_pairs/subband_60_150_colorwithsti.png) / `_acc.npy` (生成时间: 2026-06-15 11:06)
  - [x] Subband x type1: [subband_60_150_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/memory_pairs/subband_60_150_type1.png) / `_acc.npy` (生成时间: 2026-06-15 11:06)
- **True vs False Decoding**
  - [x] ERP x colorwithsti: [erp_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/true_false/erp_colorwithsti.png) / `_acc.npy` (生成时间: 2026-06-15 11:05)
  - [x] ERP x type1: [erp_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/true_false/erp_type1.png) / `_acc.npy` (生成时间: 2026-06-15 11:05)
  - [x] Subband x colorwithsti: [subband_60_150_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/true_false/subband_60_150_colorwithsti.png) / `_acc.npy` (生成时间: 2026-06-15 11:06)
  - [x] Subband x type1: [subband_60_150_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/true_false/subband_60_150_type1.png) / `_acc.npy` (生成时间: 2026-06-15 11:07)
- **Time Generalization Decoding**
  - [x] ERP x colorwithsti: [erp_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/time_generalization/erp_colorwithsti.png) / `_tgm.npy` (生成时间: 2026-06-15 11:05)
  - [x] ERP x type1: [erp_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/time_generalization/erp_type1.png) / `_tgm.npy` (生成时间: 2026-06-15 11:06)
  - [x] Subband x colorwithsti: [subband_60_150_colorwithsti.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/time_generalization/subband_60_150_colorwithsti.png) / `_tgm.npy` (生成时间: 2026-06-15 11:07)
  - [x] Subband x type1: [subband_60_150_type1.png](file:///home/lirui/liulab_project/ieeg/Project_colorieeg_2026/color_cognition_pipeline/images/test003/decoding/time_generalization/subband_60_150_type1.png) / `_tgm.npy` (生成时间: 2026-06-15 11:07)

## 3. 结论
所有被试在三种 Decoding 任务下（Memory Pairs、True vs False、Time Generalization）所需的四种特征-电极组合：
1. **erp_colorwithsti**
2. **erp_type1**
3. **subband_60_150_colorwithsti**
4. **subband_60_150_type1**

其相对应的 `.png` 趋势/时序泛化图和相应的 `.npy` 计算结果数据均已全部存在且完整。检查验证通过。
