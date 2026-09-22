%% run_c09_full_batch.m
% 启动 C09 Task 3 -> Task 2 全量跨任务解码批处理

clear; clc;
script_dir = fileparts(mfilename('fullpath'));
matlab_dir = fileparts(script_dir);
c09_script = fullfile(matlab_dir, 'C09_cross_decoding_task3_to_task2_0825.m');

fprintf('========================================================================\n');
fprintf('  【启动 C09 全量批处理运行】\n');
fprintf('  主脚本路径: %s\n', c09_script);
fprintf('========================================================================\n');

run(c09_script);
