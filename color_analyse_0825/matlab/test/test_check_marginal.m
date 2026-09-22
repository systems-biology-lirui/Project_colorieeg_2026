% test_check_marginal.m
clear; clc;
tab_dir = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';

% 1. 检查 C09
tgm = readtable(fullfile(tab_dir, 'cross_decoding_tgm_perm200_summary.csv'));
fprintf('--- C09 Cross-Task TGM cluster p distribution ---\n');
disp(quantile(tgm.tgm_cluster_p_min, [0, 0.05, 0.1, 0.2, 0.5, 1]));
fprintf('tgm p < 0.05: %d\n', sum(tgm.tgm_has_sig_cluster_2d == 1));
fprintf('tgm p < 0.10: %d\n', sum(tgm.tgm_cluster_p_min < 0.10));
fprintf('tgm diag_has_sig_cluster: %d\n', sum(tgm.diag_has_sig_cluster == 1));

% 2. 检查 C13 Task 2 Direct
t2_d = readtable(fullfile(tab_dir, 'task2_direct_decoding_summary.csv'));
fprintf('\n--- C13 Task 2 Direct ---\n');
fprintf('has_sig_cluster == 1: %d\n', sum(t2_d.has_sig_cluster == 1));
fprintf('peak_p_pointwise < 0.05: %d\n', sum(t2_d.peak_p_pointwise < 0.05));
fprintf('peak_p_pointwise < 0.10: %d\n', sum(t2_d.peak_p_pointwise < 0.10));

% 3. 检查 C14 Task 3 Direct
t3_d = readtable(fullfile(tab_dir, 'task3_direct_decoding_summary.csv'));
fprintf('\n--- C14 Task 3 Direct ---\n');
fprintf('has_sig_cluster == 1: %d\n', sum(t3_d.has_sig_cluster == 1));
fprintf('peak_p_pointwise < 0.05: %d\n', sum(t3_d.peak_p_pointwise < 0.05));
fprintf('peak_p_pointwise < 0.10: %d\n', sum(t3_d.peak_p_pointwise < 0.10));

% 4. 检查 C06 Task 2 Cross
c06_c = load(fullfile(tab_dir, 'concordant_electrodes_decoding_summary.mat'));
c06_n = load(fullfile(tab_dir, 'non_concordant_electrodes_decoding_summary.mat'));
t2_c_all = [c06_c.summary_table; c06_n.summary_table];
fprintf('\n--- C06 Task 2 Cross ---\n');
fprintf('has_sig_cluster == 1: %d\n', sum(t2_c_all.has_sig_cluster == 1));
fprintf('peak_p_pointwise < 0.05: %d\n', sum(t2_c_all.peak_p_pointwise < 0.05));
fprintf('peak_p_pointwise < 0.10: %d\n', sum(t2_c_all.peak_p_pointwise < 0.10));

% 5. 检查 C07 Task 3 Cross
c07 = load(fullfile(tab_dir, 'task3_purecolor_decoding_summary.mat'));
fprintf('\n--- C07 Task 3 Cross ---\n');
fprintf('has_sig_cluster == 1: %d\n', sum(c07.summary_table.has_sig_cluster == 1));
fprintf('peak_p_pointwise < 0.05: %d\n', sum(c07.summary_table.peak_p_pointwise < 0.05));
fprintf('peak_p_pointwise < 0.10: %d\n', sum(c07.summary_table.peak_p_pointwise < 0.10));
