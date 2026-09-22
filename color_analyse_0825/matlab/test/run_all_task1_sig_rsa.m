%% run_all_task1_sig_rsa.m
% 批量执行 Task 1 所有显著电极的跨任务多频段 RSA 与 3D 轨迹分析
clear; clc; close all;

fprintf('========================================================================\n');
fprintf('  【启动全量 Task 1 显著通道跨任务 RSA 与 3D 轨迹出图批处理】\n');
fprintf('========================================================================\n');

script_path = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825\matlab\C12_channel_rsa_two_figures_0825.m';
run(script_path);
