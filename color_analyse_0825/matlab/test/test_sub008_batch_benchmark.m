%% test_sub008_batch_benchmark.m
clear; clc; close all;
root_dir = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825';
data_root = fullfile(root_dir, 'process_data_new');
task_info = fullfile(root_dir, 'task_info');
res_table = fullfile(root_dir, 'result', 'tables', 'color_effects_summary.mat');

sub_id = 'sub008';
d_c04 = load(res_table);
c04_tbl = d_c04.all_tbl;
concord_mask = (c04_tbl.is_significant == 1) & ...
    (strcmp(c04_tbl.concordance_type, 'Concordant_Positive') | ...
     strcmp(c04_tbl.concordance_type, 'Concordant_Negative')) & ...
    strcmp(c04_tbl.subject, sub_id);
sub_tbl = c04_tbl(concord_mask, :);
elecs = unique(sub_tbl.channel, 'stable');
n_elecs = numel(elecs);
fprintf('Found %d concordant electrodes for %s\n', n_elecs, sub_id);

t3_mat = fullfile(data_root, sub_id, 'task3_multiband_epoched.mat');
t2_mat = fullfile(data_root, sub_id, 'task2_multiband_epoched.mat');

t_axis = double(h5read(t3_mat, '/epoched_data/time_ms')); t_axis = t_axis(:)';
ch_list3 = h5read(t3_mat, '/epoched_data/channels');
ch_list2 = h5read(t2_mat, '/epoched_data/channels');
if iscell(ch_list3), ch_list3 = cellfun(@(x) char(x(:)'), ch_list3, 'UniformOutput', false); end
if iscell(ch_list2), ch_list2 = cellfun(@(x) char(x(:)'), ch_list2, 'UniformOutput', false); end

d3 = load(fullfile(task_info, sub_id, 'task3_trial_info.mat')); ti3 = d3.trial_info;
d2 = load(fullfile(task_info, sub_id, 'task2_trial_info.mat')); ti2 = d2.trial_info;

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
n_bands = numel(bands);

t_load = tic;
raw2_bands = cell(n_bands, 1);
raw3_bands = cell(n_bands, 1);
for b = 1:n_bands
    raw2_bands{b} = h5read(t2_mat, ['/epoched_data/' bands{b}]);
    raw3_bands{b} = h5read(t3_mat, ['/epoched_data/' bands{b}]);
end
fprintf('Loaded all 6 bands into RAM in %.2f seconds.\n', toc(t_load));
