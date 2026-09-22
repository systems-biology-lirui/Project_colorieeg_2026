%% test_run_task1_upset_plot.m
% 运行并验证 Task 1 频段耦合 UpSet 集合交集图生成
clear; clc; close all;

proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
addpath(fullfile(proj_root, 'matlab', 'plot_tool'));

fprintf('>>> 开始执行 Task 1 频段耦合 UpSet 图绘制测试 ...\n');

cfg = struct();
cfg.mode = 'both'; % 同时生成总体显著与严格同向两种版本
cfg.show_bar_values = false; % 复刻顶刊无冗余标注风格
cfg.sort_within_degree = 'canonical'; % 典范阶梯排序

plot_task1_band_overlap_upset(cfg);

out_dir = fullfile(proj_root, 'result', 'figures', 'upset_band_overlap');
f1 = fullfile(out_dir, 'color_selective_channels_band_overlap_upset.png');
f2 = fullfile(out_dir, 'color_selective_concordant_channels_band_overlap_upset.png');

assert(isfile(f1), '未找到生成的图像: %s', f1);
assert(isfile(f2), '未找到生成的图像: %s', f2);

info1 = dir(f1);
info2 = dir(f2);
fprintf('\n[✓] 验证通过！生成文件大小:\n');
fprintf('    - 总体显著图: %s (%.2f KB)\n', f1, info1.bytes / 1024);
fprintf('    - 严格同向图: %s (%.2f KB)\n', f2, info2.bytes / 1024);
