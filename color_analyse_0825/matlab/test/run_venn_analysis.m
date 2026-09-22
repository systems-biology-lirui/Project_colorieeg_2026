% run_venn_analysis.m
% 执行 C15_decoding_significant_venn_0825.m

clear; clc;
test_dir   = fileparts(mfilename('fullpath'));
matlab_dir = fileparts(test_dir);
addpath(matlab_dir);

fprintf('========================================================================\n');
fprintf('>>> 运行 C15: 汇总显著位点到 .mat 并绘制 3-Circle Venn 韦恩图 ...\n');
fprintf('========================================================================\n');

C15_decoding_significant_venn_0825;

fprintf('\n>>> C15 脚本执行完毕！\n');
