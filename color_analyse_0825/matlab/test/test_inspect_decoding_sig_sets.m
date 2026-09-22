% test_inspect_decoding_sig_sets.m
% 检查 5 种 decoding 结果中的显著通道定义与存储

clear; clc;
test_dir   = fileparts(mfilename('fullpath'));
work_dir   = fileparts(fileparts(test_dir));
res_root   = fullfile(work_dir, 'result');
tab_dir    = fullfile(res_root, 'tables');

fprintf('=== 检查各类 Decoding 汇总文件 ===\n');

% 1. Task 2 交叉 (cross-fruit) decoding
f_t2_conc = fullfile(tab_dir, 'concordant_electrodes_decoding_summary.mat');
f_t2_nonc = fullfile(tab_dir, 'non_concordant_electrodes_decoding_summary.mat');
fprintf('\n--- 1. Task 2 交叉 Decoding (C06) ---\n');
if isfile(f_t2_conc)
    d = load(f_t2_conc);
    fn = fieldnames(d);
    fprintf('  concordant 文件字段: %s\n', strjoin(fn, ', '));
    t_conc = d.(fn{1});
    fprintf('  concordant 条目数: %d\n', height(t_conc));
    if ismember('has_sig_cluster', t_conc.Properties.VariableNames)
        fprintf('  concordant has_sig_cluster: %d\n', sum(t_conc.has_sig_cluster == 1));
    end
end
if isfile(f_t2_nonc)
    d = load(f_t2_nonc);
    fn = fieldnames(d);
    fprintf('  non_concordant 文件字段: %s\n', strjoin(fn, ', '));
    t_nonc = d.(fn{1});
    fprintf('  non_concordant 条目数: %d\n', height(t_nonc));
    if ismember('has_sig_cluster', t_nonc.Properties.VariableNames)
        fprintf('  non_concordant has_sig_cluster: %d\n', sum(t_nonc.has_sig_cluster == 1));
    end
end

% 2. Task 2 Direct Decoding (C13)
f_t2_dir = fullfile(tab_dir, 'task2_direct_decoding_summary.csv');
fprintf('\n--- 2. Task 2 Direct Decoding (C13) ---\n');
if isfile(f_t2_dir)
    t2_dir = readtable(f_t2_dir);
    fprintf('  总通道数: %d, has_sig_cluster == 1: %d\n', height(t2_dir), sum(t2_dir.has_sig_cluster == 1));
end

% 3. Task 3 交叉 (cross-patch) decoding (C07)
f_t3_cross = fullfile(tab_dir, 'task3_purecolor_decoding_summary.mat');
fprintf('\n--- 3. Task 3 交叉 (Leave-Patch-Out) Decoding (C07) ---\n');
if isfile(f_t3_cross)
    d = load(f_t3_cross);
    fn = fieldnames(d);
    fprintf('  task3 purecolor 字段: %s\n', strjoin(fn, ', '));
    t3_cross = d.(fn{1});
    fprintf('  总通道数: %d\n', height(t3_cross));
    if ismember('has_sig_cluster', t3_cross.Properties.VariableNames)
        fprintf('  has_sig_cluster == 1: %d\n', sum(t3_cross.has_sig_cluster == 1));
    end
end

% 4. Task 3 Direct Decoding (C14)
f_t3_dir = fullfile(tab_dir, 'task3_direct_decoding_summary.csv');
fprintf('\n--- 4. Task 3 Direct Decoding (C14) ---\n');
if isfile(f_t3_dir)
    t3_dir = readtable(f_t3_dir);
    fprintf('  总通道数: %d, has_sig_cluster == 1: %d\n', height(t3_dir), sum(t3_dir.has_sig_cluster == 1));
end

% 5. Task 2 对 Task 3 的 cross decoding (跨任务跨解码)
fprintf('\n--- 5. 跨任务 Cross Decoding (Task 3 <-> Task 2) ---\n');
cross_files = {
    'cross_decoding_tgm_perm200_summary.csv';
    'cross_decoding_concordant_summary.csv';
    'cross_decoding_by_fruit_summary.csv';
    'cross_decoding_asymmetry_summary.csv'
};
for i = 1:numel(cross_files)
    fp = fullfile(tab_dir, cross_files{i});
    if isfile(fp)
        tbl = readtable(fp);
        fprintf('  [%s] 行数: %d, 列名: %s\n', cross_files{i}, height(tbl), strjoin(tbl.Properties.VariableNames(1:min(8, width(tbl))), ', '));
        if ismember('has_sig_cluster_2d', tbl.Properties.VariableNames)
            fprintf('    has_sig_cluster_2d == 1: %d\n', sum(tbl.has_sig_cluster_2d == 1));
        elseif ismember('has_sig_cluster', tbl.Properties.VariableNames)
            fprintf('    has_sig_cluster == 1: %d\n', sum(tbl.has_sig_cluster == 1));
        elseif ismember('is_sig', tbl.Properties.VariableNames)
            fprintf('    is_sig == 1: %d\n', sum(tbl.is_sig == 1));
        end
    end
end
