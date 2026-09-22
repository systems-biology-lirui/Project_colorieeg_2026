% inspect_near_sig.m
% 检查各个decoding分析总结表中的字段和统计指标（例如 p值、cluster_p值等）

clear; clc;
tab_dir = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';

fprintf('--- 1. Task 2 Cross (C06) ---\n');
c06_c = load(fullfile(tab_dir, 'concordant_electrodes_decoding_summary.mat'));
disp(c06_c.summary_table.Properties.VariableNames);
disp(head(c06_c.summary_table, 3));

fprintf('--- 2. Task 2 Direct (C13) ---\n');
t2_d = readtable(fullfile(tab_dir, 'task2_direct_decoding_summary.csv'));
disp(t2_d.Properties.VariableNames);
disp(head(t2_d, 3));

fprintf('--- 3. Task 3 Cross (C07) ---\n');
c07 = load(fullfile(tab_dir, 'task3_purecolor_decoding_summary.mat'));
disp(c07.summary_table.Properties.VariableNames);
disp(head(c07.summary_table, 3));

fprintf('--- 4. Task 3 Direct (C14) ---\n');
t3_d = readtable(fullfile(tab_dir, 'task3_direct_decoding_summary.csv'));
disp(t3_d.Properties.VariableNames);
disp(head(t3_d, 3));

fprintf('--- 5. Cross Task TGM Perm200 (C09) ---\n');
tgm = readtable(fullfile(tab_dir, 'cross_decoding_tgm_perm200_summary.csv'));
disp(tgm.Properties.VariableNames);
disp(head(tgm, 3));
