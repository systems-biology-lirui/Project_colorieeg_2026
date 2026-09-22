%% test_single_channel_rsa.m
clear; clc; close all;
root_dir = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825';
data_root = fullfile(root_dir, 'process_data_new');
task_info = fullfile(root_dir, 'task_info');

sub_id = 'sub008';
elec   = 'C10';

t3_mat = fullfile(data_root, sub_id, 'task3_multiband_epoched.mat');
t2_mat = fullfile(data_root, sub_id, 'task2_multiband_epoched.mat');

t_axis = double(h5read(t3_mat, '/epoched_data/time_ms')); t_axis = t_axis(:)';
ch_list3 = h5read(t3_mat, '/epoched_data/channels');
ch_list2 = h5read(t2_mat, '/epoched_data/channels');
if iscell(ch_list3), ch_list3 = cellfun(@(x) char(x(:)'), ch_list3, 'UniformOutput', false); end
if iscell(ch_list2), ch_list2 = cellfun(@(x) char(x(:)'), ch_list2, 'UniformOutput', false); end

e_idx3 = find(strcmp(ch_list3, elec), 1);
e_idx2 = find(strcmp(ch_list2, elec), 1);

d3 = load(fullfile(task_info, sub_id, 'task3_trial_info.mat')); ti3 = d3.trial_info;
d2 = load(fullfile(task_info, sub_id, 'task2_trial_info.mat')); ti2 = d2.trial_info;

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
n_bands = numel(bands);

% 10 conditions
cond_names_10 = {
    'Gray Strawberry', 'Gray Watermelon', 'Gray Kiwi', 'Gray Cabbage', ...
    'Red Shape 1', 'Red Shape 2', 'Red Shape 3', ...
    'Green Shape 1', 'Green Shape 2', 'Green Shape 3'
};
cond_masks_10 = cell(10, 1);
cond_masks_10{1} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'strawberry');
cond_masks_10{2} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'watermelon');
cond_masks_10{3} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'kiwi');
cond_masks_10{4} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'cabbage');
cond_masks_10{5} = strcmp(ti3.color, 'red') & (ti3.pic_id == 1);
cond_masks_10{6} = strcmp(ti3.color, 'red') & (ti3.pic_id == 2);
cond_masks_10{7} = strcmp(ti3.color, 'red') & (ti3.pic_id == 3);
cond_masks_10{8} = strcmp(ti3.color, 'green') & (ti3.pic_id == 1);
cond_masks_10{9} = strcmp(ti3.color, 'green') & (ti3.pic_id == 2);
cond_masks_10{10} = strcmp(ti3.color, 'green') & (ti3.pic_id == 3);

% 4 collapsed conditions (相同灰色记忆平均, 相同纯色色块平均)
cond_names_4 = {
    'Red Memory (Gray Fruit)', ...
    'Green Memory (Gray Fruit)', ...
    'Red Physical (Pure Patch)', ...
    'Green Physical (Pure Patch)'
};
cond_masks_4 = cell(4, 1);
cond_masks_4{1} = strcmp(ti2.state, 'gray') & (strcmp(ti2.fruit, 'strawberry') | strcmp(ti2.fruit, 'watermelon'));
cond_masks_4{2} = strcmp(ti2.state, 'gray') & (strcmp(ti2.fruit, 'kiwi') | strcmp(ti2.fruit, 'cabbage'));
cond_masks_4{3} = strcmp(ti3.color, 'red');
cond_masks_4{4} = strcmp(ti3.color, 'green');

% 载入该通道的 6 频段数据 [n_trials x n_time]
X3_all = zeros(height(ti3), n_bands, numel(t_axis));
X2_all = zeros(height(ti2), n_bands, numel(t_axis));
for b = 1:n_bands
    raw3 = h5read(t3_mat, ['/epoched_data/' bands{b}]);
    X3_all(:, b, :) = raw3(:, e_idx3, :);
    raw2 = h5read(t2_mat, ['/epoched_data/' bands{b}]);
    X2_all(:, b, :) = raw2(:, e_idx2, :);
end

% 核心时间窗 [100, 600] ms
t_win = [100, 600];
t_mask = (t_axis >= t_win(1)) & (t_axis <= t_win(2));

% --- 1. 时间窗平均特征 (Option A: 6频段均值; Option B: 6频段 x 时间点) ---
% 10 类别
feat_10_mean = zeros(10, n_bands);
feat_10_tf   = zeros(10, n_bands * sum(t_mask));
for c = 1:10
    if c <= 4
        m = cond_masks_10{c};
        sig = squeeze(mean(X2_all(m, :, :), 1)); % [n_bands x n_time]
    else
        m = cond_masks_10{c};
        sig = squeeze(mean(X3_all(m, :, :), 1));
    end
    feat_10_mean(c, :) = mean(sig(:, t_mask), 2)';
    sig_win = sig(:, t_mask);
    feat_10_tf(c, :) = sig_win(:)';
end

rdm_10_mean = 1 - corr(feat_10_mean');
rdm_10_tf   = 1 - corr(feat_10_tf');

fprintf('Channel %s-%s:\n', sub_id, elec);
fprintf('  RDM 10 (mean 6 bands) min/max: [%.3f, %.3f]\n', min(rdm_10_mean(:)), max(rdm_10_mean(:)));
fprintf('  RDM 10 (TF 6xTime) min/max:    [%.3f, %.3f]\n', min(rdm_10_tf(:)), max(rdm_10_tf(:)));

% 4 类别
feat_4_mean = zeros(4, n_bands);
feat_4_tf   = zeros(4, n_bands * sum(t_mask));
for c = 1:4
    if c <= 2
        m = cond_masks_4{c};
        sig = squeeze(mean(X2_all(m, :, :), 1));
    else
        m = cond_masks_4{c};
        sig = squeeze(mean(X3_all(m, :, :), 1));
    end
    feat_4_mean(c, :) = mean(sig(:, t_mask), 2)';
    sig_win = sig(:, t_mask);
    feat_4_tf(c, :) = sig_win(:)';
end

rdm_4_mean = 1 - corr(feat_4_mean');
rdm_4_tf   = 1 - corr(feat_4_tf');
disp('4x4 RDM (TF):');
disp(array2table(rdm_4_tf, 'RowNames', cond_names_4, 'VariableNames', {'RedGray','GrnGray','RedPure','GrnPure'}));
