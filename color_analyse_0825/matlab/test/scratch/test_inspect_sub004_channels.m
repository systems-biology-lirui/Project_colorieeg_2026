% test_inspect_sub004_channels.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
d = load(fullfile(proj_root, 'result', 'tables', 'color_effects_summary.mat'), 'all_tbl');
t = d.all_tbl;
t4 = t(strcmp(t.subject, 'sub004') & strcmp(t.freq_band, 'High_Gamma'), :);
fprintf('sub004 在 color_effects_summary 中的通道数: %d\n', height(t4));
disp('sub004 前 20 个通道:');
disp(t4.channel(1:min(20, height(t4))));
disp('sub004 后 20 个通道:');
disp(t4.channel(max(1, height(t4)-19):end));
