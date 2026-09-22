%% test_run_decoding_rank_summary.m
% 运行并验证 Task 1 筛选频段在 Task 2 & Task 3 中的 Decoding Rank 汇总图生成
clear; clc; close all;

proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
addpath(fullfile(proj_root, 'matlab', 'plot_tool'));

fprintf('>>> 开始执行 Task 1 筛选频段 vs Task 2 & Task 3 Decoding Rank 汇总图绘制测试 ...\n');

cfg = struct();
cfg.mode = 'both';         % 生成 all_significant 与 concordant 两种版本
cfg.acc_metric = 'peak';   % 峰值准确率
cfg.invert_y = true;       % 1 (Top) 置于上方，符合直觉

plot_task1_screening_vs_decoding_ranks(cfg);

out_dir = fullfile(proj_root, 'result', 'figures', 'decoding_rank_summary');
f1 = fullfile(out_dir, 'task1_screening_vs_decoding_ranks_all_significant.png');
f2 = fullfile(out_dir, 'task1_screening_vs_decoding_ranks_concordant.png');

assert(isfile(f1), '未找到生成的图像: %s', f1);
assert(isfile(f2), '未找到生成的图像: %s', f2);

info1 = dir(f1);
info2 = dir(f2);
fprintf('\n[✓] 验证通过！生成文件大小:\n');
fprintf('    - 总体显著汇总图: %s (%.2f KB)\n', f1, info1.bytes / 1024);
fprintf('    - 严格同向汇总图: %s (%.2f KB)\n', f2, info2.bytes / 1024);
