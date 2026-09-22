% 批量生成四类别严格同向的未校正与FDR校正玻璃脑三视图
clear; clc; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

fprintf('>>> [1/2] 正在生成【未校正 p < 0.05 + 四类别严格同向】三视图 ...\n');
cfg1 = struct('stat_mode', 'uncorrected', 'p_thresh', 0.05, 'concordant_only', true);
plot_glass_brain_3view_0825(cfg1);

fprintf('\n>>> [2/2] 正在生成【被试内 FDR q < 0.05 + 四类别严格同向】三视图 ...\n');
cfg2 = struct('stat_mode', 'fdr_sub', 'fdr_q', 0.05, 'concordant_only', true);
plot_glass_brain_3view_0825(cfg2);

fprintf('\n========================================================================\n');
fprintf('  【四类别严格同向版本全部生成完成！】\n');
fprintf('========================================================================\n');
