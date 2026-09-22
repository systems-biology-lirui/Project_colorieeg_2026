% test_inspect_sig_lists.m
clear; clc;
test_dir   = fileparts(mfilename('fullpath'));
work_dir   = fileparts(fileparts(test_dir));
res_root   = fullfile(work_dir, 'result');
tab_dir    = fullfile(res_root, 'tables');

% 1. Task 2 Cross (C06)
c06_conc = load(fullfile(tab_dir, 'concordant_electrodes_decoding_summary.mat'));
c06_nonc = load(fullfile(tab_dir, 'non_concordant_electrodes_decoding_summary.mat'));
t2_c_sig1 = c06_conc.summary_table(c06_conc.summary_table.has_sig_cluster == 1, :);
t2_c_sig2 = c06_nonc.summary_table(c06_nonc.summary_table.has_sig_cluster == 1, :);
t2_cross_keys = [strcat(t2_c_sig1.subject, '_', t2_c_sig1.channel); strcat(t2_c_sig2.subject, '_', t2_c_sig2.channel)];
fprintf('1. Task 2 Cross (C06) 显著通道数: %d\n', numel(t2_cross_keys));
disp(t2_cross_keys');

% 2. Task 2 Direct (C13)
t2_d = readtable(fullfile(tab_dir, 'task2_direct_decoding_summary.csv'));
t2_d_sig = t2_d(t2_d.has_sig_cluster == 1, :);
t2_direct_keys = strcat(t2_d_sig.subject, '_', t2_d_sig.channel);
fprintf('2. Task 2 Direct (C13) 显著通道数: %d\n', numel(t2_direct_keys));
disp(t2_direct_keys');

% 3. Task 3 Cross (C07)
c07 = load(fullfile(tab_dir, 'task3_purecolor_decoding_summary.mat'));
t3_c_sig = c07.summary_table(c07.summary_table.has_sig_cluster == 1, :);
t3_cross_keys = strcat(t3_c_sig.subject, '_', t3_c_sig.channel);
fprintf('3. Task 3 Cross (C07) 显著通道数: %d\n', numel(t3_cross_keys));
disp(t3_cross_keys');

% 4. Task 3 Direct (C14)
t3_d = readtable(fullfile(tab_dir, 'task3_direct_decoding_summary.csv'));
t3_d_sig = t3_d(t3_d.has_sig_cluster == 1, :);
t3_direct_keys = strcat(t3_d_sig.subject, '_', t3_d_sig.channel);
fprintf('4. Task 3 Direct (C14) 显著通道数: %d\n', numel(t3_direct_keys));
disp(t3_direct_keys');

% 5. Cross Task (Task 3 <-> Task 2)
% Check both diag and 2D TGM
tgm200 = readtable(fullfile(tab_dir, 'cross_decoding_tgm_perm200_summary.csv'));
tgm_2d_sig = tgm200(tgm200.tgm_has_sig_cluster_2d == 1, :);
cross_2d_keys = strcat(tgm_2d_sig.subject, '_', tgm_2d_sig.channel);
fprintf('5a. Cross-Task 2D TGM Perm200 显著通道数: %d\n', numel(cross_2d_keys));
disp(cross_2d_keys');

diag_sig = tgm200(tgm200.diag_has_sig_cluster == 1, :);
cross_diag_keys = strcat(diag_sig.subject, '_', diag_sig.channel);
fprintf('5b. Cross-Task 1D 对角线 显著通道数: %d\n', numel(cross_diag_keys));
disp(cross_diag_keys');

% by_fruit
by_fr = readtable(fullfile(tab_dir, 'cross_decoding_by_fruit_summary.csv'));
by_fr_sig = by_fr(by_fr.has_sig_cluster == 1, :);
cross_fruit_keys = strcat(by_fr_sig.subject, '_', by_fr_sig.channel);
fprintf('5c. Cross-Task By-Fruit 显著通道数: %d\n', numel(cross_fruit_keys));
disp(cross_fruit_keys');
