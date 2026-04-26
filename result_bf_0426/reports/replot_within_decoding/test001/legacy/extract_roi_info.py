import os
import glob
import numpy as np
import datetime
from collections import defaultdict

# Setup paths
base_dir = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026/analyse_code/newanalyse/fft_decoding_task/batch_within_decoding/test001/perm1000'
output_md = os.path.join(os.path.dirname(base_dir), 'roi_summary.md')

# Exclusion list (lowercase for comparison)
exclude_keywords = ['unknown', 'preceneus', 'precuneus', 'n_a', 'calcarine']

# Log records
log_records = []
def log(msg):
    print(msg)
    log_records.append(msg)

log(f"开始遍历目录: {base_dir}")

# Find all PNG files
png_files = glob.glob(os.path.join(base_dir, '**', '*.png'), recursive=True)
log(f"共找到 {len(png_files)} 个 .png 文件。")

# To keep track of processed ROIs to avoid duplicates if multiple PNGs exist for same ROI
# task -> band -> roi -> info
results = defaultdict(lambda: defaultdict(dict))

processed_count = 0
success_count = 0
skip_count = 0

for png_path in png_files:
    processed_count += 1
    # Example path: .../perm1000/erp/task1_color_vs_gray_pair_cv/roi_curves/Color_patch_curve.png
    parts = png_path.split(os.sep)
    
    try:
        # Find the index of 'perm1000'
        perm_idx = parts.index('perm1000')
        band = parts[perm_idx + 1]
        task_name = parts[perm_idx + 2]
    except ValueError:
        log(f"跳过: {png_path} (路径不符合预期结构)")
        skip_count += 1
        continue
        
    filename = os.path.basename(png_path)
    
    # We are interested in ROI curves to extract ROI specific info
    if 'roi_curves' in parts and filename.endswith('_curve.png'):
        roi_name = filename.replace('_curve.png', '')
        
        # Check exclusion
        roi_lower = roi_name.lower()
        if any(kw in roi_lower for kw in exclude_keywords):
            log(f"跳过: {png_path} (命中排除规则: {roi_name})")
            skip_count += 1
            continue
            
        # Instead of OCR, we read the corresponding exact NPZ file to ensure 100% accurate ms data
        npz_path = os.path.join(os.path.dirname(os.path.dirname(png_path)), 'computed_results', f'{roi_name}_results.npz')
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path, allow_pickle=True)
                # Check if significant
                sig_indices = data.get('sig_indices', [])
                if np.any(sig_indices):
                    latency_earliest = float(data.get('latency_earliest', np.nan))
                    latency_peak = float(data.get('latency_peak', np.nan))
                    
                    if not np.isnan(latency_earliest):
                        time_info = f"Earliest: {latency_earliest:.1f} ms, Peak: {latency_peak:.1f} ms"
                        results[task_name][band][roi_name] = {
                            'time_info': time_info,
                            'remarks': 'Significant'
                        }
                        success_count += 1
                        log(f"成功提取: 任务={task_name}, 频段={band}, ROI={roi_name}, 时间点={time_info}")
                    else:
                        log(f"跳过: {png_path} (无有效显著时间点)")
                        skip_count += 1
                else:
                    log(f"跳过: {png_path} (ROI未达到显著性)")
                    skip_count += 1
            except Exception as e:
                log(f"跳过: {png_path} (读取数据失败: {e})")
                skip_count += 1
        else:
            log(f"跳过: {png_path} (找不到对应的数据文件以提取精确时间)")
            skip_count += 1
    else:
        # Skip overview figures
        log(f"跳过: {png_path} (非单一ROI曲线图)")
        skip_count += 1

log(f"\n--- 解析统计 ---")
log(f"解析文件数: {processed_count}")
log(f"成功提取数: {success_count}")
log(f"跳过数: {skip_count}")

# Generate Markdown
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

md_lines = []
md_lines.append("# ROI 显著性及时间点汇总")
md_lines.append("")
md_lines.append("## 总览")
md_lines.append(f"- **数据来源**: `{base_dir}` 及其子目录下的图像与对应结果文件。为了保证时间点数值精度（1位小数），实际通过图像对应的底层 `.npz` 数据文件提取精确时间点。")
md_lines.append(f"- **排除规则**: 排除 ROI 名称中包含 `unknown`、`preceneus`、`precuneus`、`N_A`、`calcarine`（忽略大小写）的记录。")
md_lines.append(f"- **汇总时间**: {timestamp}")
md_lines.append("")

# Collect stats
task_band_coverage = defaultdict(set)
task_roi_counts = defaultdict(int)

for task, bands in results.items():
    md_lines.append(f"## 任务: {task}")
    md_lines.append("")
    md_lines.append("| 频段 | 显著 ROI | 显著时间点 | 备注 |")
    md_lines.append("| --- | --- | --- | --- |")
    
    for band, rois in bands.items():
        task_band_coverage[task].add(band)
        for roi, info in rois.items():
            task_roi_counts[task] += 1
            md_lines.append(f"| {band} | {roi} | {info['time_info']} | {info['remarks']} |")
            
    md_lines.append("")

md_lines.append("## 任务统计")
md_lines.append("| 任务名称 | 频段覆盖 | 显著 ROI 样本量 |")
md_lines.append("| --- | --- | --- |")
for task in results.keys():
    bands_str = ", ".join(sorted(list(task_band_coverage[task])))
    count = task_roi_counts[task]
    md_lines.append(f"| {task} | {bands_str} | {count} |")

with open(output_md, 'w') as f:
    f.write("\n".join(md_lines))

log(f"\n成功生成汇总文件: {output_md}")

with open('extract_log.txt', 'w') as f:
    f.write("\n".join(log_records))
