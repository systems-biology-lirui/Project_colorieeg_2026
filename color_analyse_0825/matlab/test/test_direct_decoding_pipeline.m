% test_direct_decoding_pipeline.m
% 快速验证 C13 (Task 2 Direct) 与 C14 (Task 3 Direct) 的解码与出图逻辑

clear; clc;
test_dir   = fileparts(mfilename('fullpath'));
matlab_dir = fileparts(test_dir);
addpath(matlab_dir);

fprintf('========================================================================\n');
fprintf('>>> 步骤 1: 验证 C13 (Task 2 Direct Decoding) 快速测试 (1 个通道) ...\n');
fprintf('========================================================================\n');

cfg_override = struct();
cfg_override.max_elecs = 1;
cfg_override.skip_existing = false;
C13_task2_direct_decoding_0825;

fprintf('\n========================================================================\n');
fprintf('>>> 步骤 2: 验证 C14 (Task 3 Direct Decoding) 快速测试 (1 个通道) ...\n');
fprintf('========================================================================\n');

cfg_override = struct();
cfg_override.max_elecs = 1;
cfg_override.skip_existing = false;
C14_task3_direct_decoding_0825;

fprintf('\n>>> 测试全部通过！\n');
