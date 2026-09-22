% 批量生成不同统计版本的玻璃脑三视图
clear; clc; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

fprintf('>>> [1/3] 正在生成未校正置换检验 p < 0.05 结果 ...\n');
cfg1 = struct('stat_mode', 'uncorrected', 'p_thresh', 0.05);
plot_glass_brain_3view_0825(cfg1);

fprintf('\n>>> [2/3] 正在生成被试内同频段 FDR q < 0.05 结果 ...\n');
cfg2 = struct('stat_mode', 'fdr_sub', 'fdr_q', 0.05);
plot_glass_brain_3view_0825(cfg2);

fprintf('\n>>> [3/3] 正在生成全频段同频段 FDR q < 0.05 结果 ...\n');
cfg3 = struct('stat_mode', 'fdr_band', 'fdr_q', 0.05);
plot_glass_brain_3view_0825(cfg3);

fprintf('\n========================================================================\n');
fprintf('  【全部版本玻璃脑三视图生成完成！】\n');
fprintf('========================================================================\n');
