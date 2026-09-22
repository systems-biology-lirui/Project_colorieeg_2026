% test_venn_overlaps.m
clear; clc;
test_dir   = fileparts(mfilename('fullpath'));
work_dir   = fileparts(fileparts(test_dir));
tab_dir    = fullfile(work_dir, 'result', 'tables');

% 1. Task 2 Cross
c06_c = load(fullfile(tab_dir, 'concordant_electrodes_decoding_summary.mat'));
c06_n = load(fullfile(tab_dir, 'non_concordant_electrodes_decoding_summary.mat'));
t2_c_sig1 = c06_c.summary_table(c06_c.summary_table.has_sig_cluster == 1, :);
t2_c_sig2 = c06_n.summary_table(c06_n.summary_table.has_sig_cluster == 1, :);
set_t2_cross = unique([strcat(t2_c_sig1.subject, '_', t2_c_sig1.channel); strcat(t2_c_sig2.subject, '_', t2_c_sig2.channel)]);

% 2. Task 2 Direct
t2_d = readtable(fullfile(tab_dir, 'task2_direct_decoding_summary.csv'));
set_t2_direct = unique(strcat(t2_d.subject(t2_d.has_sig_cluster == 1), '_', t2_d.channel(t2_d.has_sig_cluster == 1)));

% 3. Task 3 Cross
c07 = load(fullfile(tab_dir, 'task3_purecolor_decoding_summary.mat'));
set_t3_cross = unique(strcat(c07.summary_table.subject(c07.summary_table.has_sig_cluster == 1), '_', c07.summary_table.channel(c07.summary_table.has_sig_cluster == 1)));

% 4. Task 3 Direct
t3_d = readtable(fullfile(tab_dir, 'task3_direct_decoding_summary.csv'));
set_t3_direct = unique(strcat(t3_d.subject(t3_d.has_sig_cluster == 1), '_', t3_d.channel(t3_d.has_sig_cluster == 1)));

% 5. Cross Task (Task 3 <-> Task 2)
tgm200 = readtable(fullfile(tab_dir, 'cross_decoding_tgm_perm200_summary.csv'));
set_cross_task_2d = unique(strcat(tgm200.subject(tgm200.tgm_has_sig_cluster_2d == 1), '_', tgm200.channel(tgm200.tgm_has_sig_cluster_2d == 1)));
set_cross_task_1d = unique(strcat(tgm200.subject(tgm200.diag_has_sig_cluster == 1), '_', tgm200.channel(tgm200.diag_has_sig_cluster == 1)));
set_cross_task_any = unique([set_cross_task_2d; set_cross_task_1d]);

fprintf('=== 图 1: Task 2 交叉 vs Task 3 交叉 vs Cross-Task (2D TGM) ===\n');
A1 = set_t2_cross;
B1 = set_t3_cross;
C1 = set_cross_task_2d;
fprintf('A (Task 2 Cross): %d\n', numel(A1));
fprintf('B (Task 3 Cross): %d\n', numel(B1));
fprintf('C (Cross-Task 2D): %d\n', numel(C1));
fprintf('A & B: %d -> %s\n', numel(intersect(A1, B1)), strjoin(intersect(A1, B1), ', '));
fprintf('A & C: %d -> %s\n', numel(intersect(A1, C1)), strjoin(intersect(A1, C1), ', '));
fprintf('B & C: %d -> %s\n', numel(intersect(B1, C1)), strjoin(intersect(B1, C1), ', '));
fprintf('A & B & C: %d -> %s\n', numel(intersect(intersect(A1, B1), C1)), strjoin(intersect(intersect(A1, B1), C1), ', '));

fprintf('\n=== 图 2: Task 2 Direct vs Task 3 Direct vs Cross-Task (2D TGM) ===\n');
A2 = set_t2_direct;
B2 = set_t3_direct;
C2 = set_cross_task_2d;
fprintf('A (Task 2 Direct): %d\n', numel(A2));
fprintf('B (Task 3 Direct): %d\n', numel(B2));
fprintf('C (Cross-Task 2D): %d\n', numel(C2));
fprintf('A & B: %d -> %s\n', numel(intersect(A2, B2)), strjoin(intersect(A2, B2), ', '));
fprintf('A & C: %d -> %s\n', numel(intersect(A2, C2)), strjoin(intersect(A2, C2), ', '));
fprintf('B & C: %d -> %s\n', numel(intersect(B2, C2)), strjoin(intersect(B2, C2), ', '));
fprintf('A & B & C: %d -> %s\n', numel(intersect(intersect(A2, B2), C2)), strjoin(intersect(intersect(A2, B2), C2), ', '));
