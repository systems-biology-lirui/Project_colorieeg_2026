% run_all_direct_decoding.m
% 批处理全量 229 个 Task 1 显著通道的 Task 2 与 Task 3 直接红绿 Decoding

clear; clc;
test_dir   = fileparts(mfilename('fullpath'));
matlab_dir = fileparts(test_dir);
addpath(matlab_dir);

fprintf('========================================================================\n');
fprintf('  【批处理启动: Task 2 与 Task 3 直接红绿 Decoding (全量 229 通道)】\n');
fprintf('========================================================================\n');

total_tic = tic;

% 1. 运行 Task 2 Direct Decoding (全部 229 通道)
fprintf('\n>>> 开始执行阶段 1: Task 2 记忆色彩直接二分类 Decoding ...\n');
cfg_override = struct();
cfg_override.max_elecs     = Inf;
cfg_override.skip_existing = true;
C13_task2_direct_decoding_0825;

% 2. 运行 Task 3 Direct Decoding (全部 229 通道)
fprintf('\n>>> 开始执行阶段 2: Task 3 物理纯色直接二分类 Decoding ...\n');
cfg_override = struct();
cfg_override.max_elecs     = Inf;
cfg_override.skip_existing = true;
C14_task3_direct_decoding_0825;

fprintf('\n========================================================================\n');
fprintf('  【全部任务顺利完成！】总耗时: %.2f 分钟\n', toc(total_tic) / 60);
fprintf('========================================================================\n');
