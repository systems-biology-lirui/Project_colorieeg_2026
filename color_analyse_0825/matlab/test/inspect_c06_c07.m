% inspect_c06_c07.m
clear; clc;
tab_dir = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';

c06_c = load(fullfile(tab_dir, 'concordant_electrodes_decoding_summary.mat'));
fprintf('--- C06 Concordant fields ---\n');
disp(c06_c.summary_table.Properties.VariableNames);
if ismember('cluster_p', c06_c.summary_table.Properties.VariableNames)
    disp(c06_c.summary_table(:, {'subject','channel','has_sig_cluster','cluster_p'}));
elseif ismember('min_cluster_p', c06_c.summary_table.Properties.VariableNames)
    disp(c06_c.summary_table(:, {'subject','channel','has_sig_cluster','min_cluster_p'}));
else
    disp(c06_c.summary_table(1:5, :));
end

c07 = load(fullfile(tab_dir, 'task3_purecolor_decoding_summary.mat'));
fprintf('--- C07 fields ---\n');
disp(c07.summary_table.Properties.VariableNames);
disp(c07.summary_table(1:5, :));
