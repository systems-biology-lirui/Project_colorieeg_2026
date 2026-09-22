% test_run_all_electrodes_pipeline.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
addpath(fullfile(proj_root, 'matlab'));

fprintf('运行主脚本 plot_all_recorded_electrodes_glass_brain ...\n');
plot_all_recorded_electrodes_glass_brain();
fprintf('运行成功完成！\n');
