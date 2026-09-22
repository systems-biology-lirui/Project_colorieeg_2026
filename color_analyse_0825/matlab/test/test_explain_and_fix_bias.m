% test_explain_and_fix_bias.m
% Script to demonstrate why red and green accuracies are mirrored without bias correction,
% and how zero-centering decision values resolves the artifact.

script_dir = fileparts(mfilename('fullpath'));
color_root = fileparts(fileparts(script_dir));
data_root  = fullfile(color_root, 'process_data_new');
task_info_dir = fullfile(color_root, 'task_info');

sub_id = 'sub007'; elec = 'C4';
t3_mat = fullfile(data_root, sub_id, 'task3_multiband_epoched.mat');
t2_mat = fullfile(data_root, sub_id, 'task2_multiband_epoched.mat');

time_ms = h5read(t3_mat, '/epoched_data/time_ms'); time_ms = time_ms(:)';
ch3 = cellfun(@(x) char(x(:)'), h5read(t3_mat, '/epoched_data/channels'), 'UniformOutput', false);
ch2 = cellfun(@(x) char(x(:)'), h5read(t2_mat, '/epoched_data/channels'), 'UniformOutput', false);

d3 = load(fullfile(task_info_dir, sub_id, 'task3_trial_info.mat'), 'trial_info');
ti3 = d3.trial_info;
tr_mask = strcmp(ti3.color, 'red') | strcmp(ti3.color, 'green');
y_tr = double(strcmp(ti3.color(tr_mask), 'red'));

d2 = load(fullfile(task_info_dir, sub_id, 'task2_trial_info.mat'), 'trial_info');
ti2 = d2.trial_info;
te_mask = strcmp(ti2.state, 'gray');
ti2_gray = ti2(te_mask, :);
fruit_list = ti2_gray.fruit;
m_straw = strcmp(fruit_list, 'strawberry');
m_water = strcmp(fruit_list, 'watermelon');
m_cabb  = strcmp(fruit_list, 'cabbage');
m_kiwi  = strcmp(fruit_list, 'kiwi');

idx3 = find(strcmp(ch3, elec));
idx2 = find(strcmp(ch2, elec));

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
X3 = zeros(sum(tr_mask), numel(bands));
X2 = zeros(sum(te_mask), numel(bands));

% Test at a specific time window: t = 200 ms (where sub007-C4 showed massive divergence)
tc = 200;
t_m = (time_ms >= tc - 10) & (time_ms < tc + 10);

for b = 1:numel(bands)
    r3 = h5read(t3_mat, ['/epoched_data/' bands{b}]);
    r2 = h5read(t2_mat, ['/epoched_data/' bands{b}]);
    X3(:, b) = mean(squeeze(r3(tr_mask, idx3, t_m)), 2);
    X2(:, b) = mean(squeeze(r2(te_mask, idx2, t_m)), 2);
end

mu = mean(X3, 1); sig = std(X3, 0, 1); sig(sig<1e-6) = 1;
X3_n = (X3 - mu) ./ sig;
X2_n = (X2 - mu) ./ sig;

mdl = fitclinear(X3_n, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', 0.01);
[y_pred, scores] = predict(mdl, X2_n);
d = scores(:, 2); % decision values

fprintf('=== SUB007-C4 at t = 200 ms ===\n');
fprintf('Proportion of trials predicted as RED: %.1f%%\n', mean(y_pred == 1)*100);
fprintf('Proportion of trials predicted as GREEN: %.1f%%\n', mean(y_pred == 0)*100);
fprintf('\n--- RAW ACCURACIES (without bias correction) ---\n');
fprintf('Strawberry (wants RED):   %.1f%%\n', mean(y_pred(m_straw) == 1)*100);
fprintf('Watermelon (wants RED):   %.1f%%\n', mean(y_pred(m_water) == 1)*100);
fprintf('Cabbage    (wants GREEN): %.1f%%\n', mean(y_pred(m_cabb) == 0)*100);
fprintf('Kiwi       (wants GREEN): %.1f%%\n', mean(y_pred(m_kiwi) == 0)*100);
fprintf('Balanced Overall Mean:    %.1f%%\n', ...
    (mean(y_pred(m_straw)==1) + mean(y_pred(m_water)==1) + mean(y_pred(m_cabb)==0) + mean(y_pred(m_kiwi)==0))/4 * 100);

% Now with bias correction: center d across all 240 gray trials
d_corr = d - median(d);
y_corr = double(d_corr > 0);

fprintf('\n--- BIAS-CORRECTED ACCURACIES (Centered Decision Boundary) ---\n');
fprintf('Strawberry: %.1f%%\n', mean(y_corr(m_straw) == 1)*100);
fprintf('Watermelon: %.1f%%\n', mean(y_corr(m_water) == 1)*100);
fprintf('Cabbage:    %.1f%%\n', mean(y_corr(m_cabb) == 0)*100);
fprintf('Kiwi:       %.1f%%\n', mean(y_corr(m_kiwi) == 0)*100);
fprintf('Balanced Overall Mean:    %.1f%%\n', ...
    (mean(y_corr(m_straw)==1) + mean(y_corr(m_water)==1) + mean(y_corr(m_cabb)==0) + mean(y_corr(m_kiwi)==0))/4 * 100);
