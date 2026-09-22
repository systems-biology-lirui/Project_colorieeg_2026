%% test_vectorized_predict.m
clear; clc;
root_dir = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825';
data_root = fullfile(root_dir, 'process_data_new');
task_info = fullfile(root_dir, 'task_info');

sub_id = 'sub008'; elec = 'C10';
win_len = 20; win_step = 20; t_range = [-200, 800];
bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
n_bands = numel(bands);
svm_lambda = 0.01;

t_starts  = t_range(1) : win_step : (t_range(2) - win_len);
n_win     = numel(t_starts);
t_centers = t_starts + win_len / 2;

t3_mat = fullfile(data_root, sub_id, 'task3_multiband_epoched.mat');
t2_mat = fullfile(data_root, sub_id, 'task2_multiband_epoched.mat');

t_axis = double(h5read(t3_mat, '/epoched_data/time_ms')); t_axis = t_axis(:)';
ch_list3 = h5read(t3_mat, '/epoched_data/channels');
ch_list2 = h5read(t2_mat, '/epoched_data/channels');
if iscell(ch_list3), ch_list3 = cellfun(@(x) char(x(:)'), ch_list3, 'UniformOutput', false); end
if iscell(ch_list2), ch_list2 = cellfun(@(x) char(x(:)'), ch_list2, 'UniformOutput', false); end

e_idx3 = find(strcmp(ch_list3, elec), 1);
e_idx2 = find(strcmp(ch_list2, elec), 1);

d3 = load(fullfile(task_info, sub_id, 'task3_trial_info.mat'), 'trial_info');
ti3 = d3.trial_info;
tr_mask = strcmp(ti3.color, 'red') | strcmp(ti3.color, 'green');
ti3_use = ti3(tr_mask, :);
y_tr = zeros(height(ti3_use), 1);
y_tr(strcmp(ti3_use.color, 'red')) = 1;
n_tr = height(ti3_use);

d2 = load(fullfile(task_info, sub_id, 'task2_trial_info.mat'), 'trial_info');
ti2 = d2.trial_info;
te_mask = strcmp(ti2.state, 'gray');
ti2_use = ti2(te_mask, :);
fruit_list = ti2_use.fruit;
y_te = double(strcmp(fruit_list, 'strawberry') | strcmp(fruit_list, 'watermelon'));
n_te = height(ti2_use);

X3_bands = zeros(n_tr, n_bands, numel(t_axis));
X2_bands = zeros(n_te, n_bands, numel(t_axis));
for b = 1:n_bands
    raw3 = h5read(t3_mat, ['/epoched_data/' bands{b}]);
    X3_bands(:, b, :) = raw3(tr_mask, e_idx3, :);
    raw2 = h5read(t2_mat, ['/epoched_data/' bands{b}]);
    X2_bands(:, b, :) = raw2(te_mask, e_idx2, :);
end

X3_3d = zeros(n_tr, n_bands, n_win);
X2_3d = zeros(n_te, n_bands, n_win);
for w = 1:n_win
    w_t1 = t_starts(w); w_t2 = w_t1 + win_len;
    t_mask = (t_axis >= w_t1) & (t_axis < w_t2);
    for b = 1:n_bands
        X3_3d(:, b, w) = mean(X3_bands(:, b, t_mask), 3);
        X2_3d(:, b, w) = mean(X2_bands(:, b, t_mask), 3);
    end
end

% --- 方法 1: 原循环方式 ---
tic;
tgm1 = zeros(n_win, n_win);
pred1 = zeros(n_te, n_win, n_win);
for w3 = 1:n_win
    X_tr = double(squeeze(X3_3d(:, :, w3)));
    mu_w3 = mean(X_tr, 1); sig_w3 = std(X_tr, 0, 1); sig_w3(sig_w3 < 1e-6) = 1;
    X_tr_n = (X_tr - mu_w3) ./ sig_w3;
    mdl = fitclinear(X_tr_n, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', svm_lambda);
    for w2 = 1:n_win
        X_te = (double(squeeze(X2_3d(:, :, w2))) - mu_w3) ./ sig_w3;
        yp = predict(mdl, X_te);
        pred1(:, w3, w2) = yp;
        sens = sum(y_te == 1 & yp == 1) / max(1, sum(y_te == 1));
        spec = sum(y_te == 0 & yp == 0) / max(1, sum(y_te == 0));
        tgm1(w3, w2) = (sens + spec) / 2;
    end
end
t1 = toc;
fprintf('Method 1 (nested loop): %.3f s\n', t1);

% --- 方法 2: 批量预测 ---
tic;
tgm2 = zeros(n_win, n_win);
pred2 = zeros(n_te, n_win, n_win);
pos_te = (y_te == 1);
neg_te = (y_te == 0);
for w3 = 1:n_win
    X_tr = double(squeeze(X3_3d(:, :, w3)));
    mu_w3 = mean(X_tr, 1); sig_w3 = std(X_tr, 0, 1); sig_w3(sig_w3 < 1e-6) = 1;
    X_tr_n = (X_tr - mu_w3) ./ sig_w3;
    mdl = fitclinear(X_tr_n, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', svm_lambda);
    
    % 将所有 w2 的测试样本一次性缩放并预测
    X2_scaled = zeros(n_te, n_bands, n_win);
    for w2 = 1:n_win
        X2_scaled(:, :, w2) = (double(squeeze(X2_3d(:, :, w2))) - mu_w3) ./ sig_w3;
    end
    X2_flat = reshape(permute(X2_scaled, [1, 3, 2]), [n_te * n_win, n_bands]);
    yp_flat = predict(mdl, X2_flat);
    yp_mat  = reshape(yp_flat, [n_te, n_win]);
    
    pred2(:, w3, :) = reshape(yp_mat, [n_te, 1, n_win]);
    sens_w3 = mean(yp_mat(pos_te, :) == 1, 1);
    spec_w3 = mean(yp_mat(neg_te, :) == 0, 1);
    tgm2(w3, :) = (sens_w3 + spec_w3) / 2;
end
t2 = toc;
fprintf('Method 2 (batched predict): %.3f s\n', t2);
fprintf('Max diff in pred: %d\n', max(abs(pred1(:) - pred2(:))));
fprintf('Max diff in TGM: %e\n', max(abs(tgm1(:) - tgm2(:))));
