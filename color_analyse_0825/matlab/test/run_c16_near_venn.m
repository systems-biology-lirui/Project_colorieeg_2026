% run_c16_near_venn.m
% 简单清晰的运行脚本: 调用并执行 C16 包含接近显著位点的 Venn 分析

clear; clc;
fprintf('>>> 启动 C16: 包含接近显著位点的 5 大 Decoding 统计与 Venn 分析...\n');
c16_path = fullfile('e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/matlab/C16_decoding_near_significant_venn_0825.m');
run(c16_path);
fprintf('>>> C16 运行结束！\n');
