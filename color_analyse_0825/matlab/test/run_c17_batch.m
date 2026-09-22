% run_c17_batch.m
% 启动全量 229 通道的 Task 3 同形状内部解码并平均分析

clear; clc;
fprintf('>>> 启动 C17: Task 3 同形状内部红绿 Decoding 与三形状平均...\n');
c17_path = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/matlab/C17_task3_within_shape_decoding_0825.m';
run(c17_path);
fprintf('>>> C17 运行结束！\n');
