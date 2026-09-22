% test_inspect_task3_summary.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
tab_file  = fullfile(proj_root, 'result', 'tables', 'task3_purecolor_decoding_summary.mat');

m = load(tab_file);
disp('Loaded fields:');
disp(fieldnames(m));

tbl = m.summary_table;
disp('Table columns:');
disp(tbl.Properties.VariableNames);

fprintf('Total rows in summary_table: %d\n', height(tbl));
fprintf('has_sig_cluster == 1 rows: %d\n', sum(tbl.has_sig_cluster == 1));

% Show channels where has_sig_cluster == 1
sig_idx = find(tbl.has_sig_cluster == 1);
disp('Significant channels:');
disp(tbl(sig_idx, {'subject', 'channel', 'peak_acc_joint', 'peak_time_ms', 'has_sig_cluster', 'best_single_band'}));

% Check sub006 G3 specifically
g3_idx = find(strcmp(tbl.subject, 'sub006') & strcmp(tbl.channel, 'G3'));
if ~isempty(g3_idx)
    disp('sub006 G3 record:');
    disp(tbl(g3_idx, :));
end
