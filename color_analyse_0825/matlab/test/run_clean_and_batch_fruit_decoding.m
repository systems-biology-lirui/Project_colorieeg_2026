% run_clean_and_batch_fruit_decoding.m
% Launcher script to execute/resume full batch fruit cross-decoding across all 157 electrodes

clear; clc; close all;

script_dir = fileparts(mfilename('fullpath'));
color_root = fileparts(fileparts(script_dir));

fprintf('========================================================================\n');
fprintf('  【启动全量 157 电极水果分品类跨任务解码 (断点续跑模式)】\n');
fprintf('========================================================================\n');

run(fullfile(script_dir, 'test_cross_decoding_by_fruit.m'));

fprintf('========================================================================\n');
fprintf('  【全部电极处理完成】\n');
fprintf('========================================================================\n');
