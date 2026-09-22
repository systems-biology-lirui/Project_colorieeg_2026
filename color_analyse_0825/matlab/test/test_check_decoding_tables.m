%% test_check_decoding_tables.m
clear; clc;

res_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';

f_t2_conc = fullfile(res_root, 'concordant_electrodes_decoding_summary.mat');
f_t2_nonc = fullfile(res_root, 'non_concordant_electrodes_decoding_summary.mat');
f_t3_pure = fullfile(res_root, 'task3_purecolor_decoding_summary.mat');
f_t1_c04  = fullfile(res_root, 'color_effects_summary.mat');

fprintf('=== 检查 Task 2 Concordant ===\n');
if isfile(f_t2_conc)
    d = load(f_t2_conc);
    disp(fieldnames(d));
    if isfield(d, 'summary_table')
        t = d.summary_table;
    elseif isfield(d, 'master_tbl')
        t = d.master_tbl;
    end
    fprintf('Size: %d x %d\n', size(t,1), size(t,2));
    disp(t.Properties.VariableNames);
    disp(head(t, 2));
else
    fprintf('Not found: %s\n', f_t2_conc);
end

fprintf('\n=== 检查 Task 2 Non-Concordant ===\n');
if isfile(f_t2_nonc)
    d = load(f_t2_nonc);
    disp(fieldnames(d));
    if isfield(d, 'summary_table')
        t = d.summary_table;
    elseif isfield(d, 'master_tbl')
        t = d.master_tbl;
    end
    fprintf('Size: %d x %d\n', size(t,1), size(t,2));
    disp(t.Properties.VariableNames);
    disp(head(t, 2));
else
    fprintf('Not found: %s\n', f_t2_nonc);
end

fprintf('\n=== 检查 Task 3 Pure Color ===\n');
if isfile(f_t3_pure)
    d = load(f_t3_pure);
    disp(fieldnames(d));
    if isfield(d, 'summary_table')
        t = d.summary_table;
    elseif isfield(d, 'master_tbl')
        t = d.master_tbl;
    end
    fprintf('Size: %d x %d\n', size(t,1), size(t,2));
    disp(t.Properties.VariableNames);
    disp(head(t, 2));
else
    fprintf('Not found: %s\n', f_t3_pure);
end

fprintf('\n=== 检查 Task 1 C04 Table ===\n');
if isfile(f_t1_c04)
    d = load(f_t1_c04);
    if isfield(d, 'all_tbl')
        t = d.all_tbl;
    else
        t = d.res_table;
    end
    fprintf('Size: %d x %d\n', size(t,1), size(t,2));
    disp(t.Properties.VariableNames);
    disp(head(t, 2));
end
