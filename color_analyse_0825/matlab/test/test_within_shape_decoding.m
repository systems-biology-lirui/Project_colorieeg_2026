% test_within_shape_decoding.m
% 单通道测试: 同形状内红绿decoding，然后对三种形状的准确率曲线取平均

clear; clc; close all;

sub_id = 'sub001';
ch_name = 'D15';

data_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new';
mat_file = fullfile(data_root, sub_id, 'task3_multiband_epoched.mat');
d = load(mat_file, 'epoched_data');
ep = d.epoched_data;
time_ms = ep.time_ms(:)';
ti = ep.trial_info;

rg_mask = strcmp(ti.color, 'red') | strcmp(ti.color, 'green');
ti_rg = ti(rg_mask, :);
n_rg = height(ti_rg);

ch_idx = find(strcmp(ep.channels, ch_name), 1);

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
n_bands = numel(bands);

win_len = 20; win_step = 20; t_range = [-200, 800];
t_centers = t_range(1):win_step:t_range(2);
n_win = numel(t_centers);

% 提取滑动窗特征 [n_rg x n_bands x n_win]
X_3d = zeros(n_rg, n_bands, n_win, 'single');
for b = 1:n_bands
    raw_b = squeeze(ep.(bands{b})(rg_mask, ch_idx, :));
    for w = 1:n_win
        tc = t_centers(w);
        tm = (time_ms >= (tc - win_len/2)) & (time_ms < (tc + win_len/2));
        X_3d(:, b, w) = mean(raw_b(:, tm), 2);
    end
end

% 识别三种形状
shapes = unique(ti_rg.pic_id);
n_shapes = numel(shapes);
n_folds = 5;

fprintf('测试通道 %s-%s: 包含 %d 种形状\n', sub_id, ch_name, n_shapes);

% 建立每个形状内部的 5 折分层交叉验证掩码
shape_folds = cell(n_shapes, 1);
for s = 1:n_shapes
    s_id = shapes(s);
    s_idx = find(ti_rg.pic_id == s_id);
    y_s = strcmp(ti_rg.color(s_idx), 'red'); % 1: red, 0: green
    
    rng(42 + s);
    r_sub = find(y_s == 1);
    g_sub = find(y_s == 0);
    cv_r = cvpartition(numel(r_sub), 'KFold', n_folds);
    cv_g = cvpartition(numel(g_sub), 'KFold', n_folds);
    
    f_info = struct();
    f_info.s_idx = s_idx;
    f_info.y_s = double(y_s);
    f_info.tr_masks = false(numel(s_idx), n_folds);
    f_info.te_masks = false(numel(s_idx), n_folds);
    
    for f_i = 1:n_folds
        te_r = r_sub(test(cv_r, f_i));
        te_g = g_sub(test(cv_g, f_i));
        te_all = [te_r; te_g];
        f_info.te_masks(te_all, f_i) = true;
        f_info.tr_masks(:, f_i) = ~f_info.te_masks(:, f_i);
    end
    shape_folds{s} = f_info;
end

% 真实解码计算
t_start = tic;
real_acc_shapes = zeros(n_shapes, n_win);

for s = 1:n_shapes
    f_info = shape_folds{s};
    X_s = X_3d(f_info.s_idx, :, :);
    y_s = f_info.y_s;
    
    for w = 1:n_win
        X_w = double(squeeze(X_s(:, :, w)));
        fold_accs = zeros(1, n_folds);
        for f_i = 1:n_folds
            tr_m = f_info.tr_masks(:, f_i);
            te_m = f_info.te_masks(:, f_i);
            
            mu = mean(X_w(tr_m, :), 1);
            sig = std(X_w(tr_m, :), 0, 1);
            sig(sig < 1e-6) = 1;
            X_tr = (X_w(tr_m, :) - mu) ./ sig;
            X_te = (X_w(te_m, :) - mu) ./ sig;
            
            mdl = fitclinear(X_tr, y_s(tr_m), 'Learner', 'svm', ...
                'Regularization', 'ridge', 'Lambda', 1e-2, 'Solver', 'dual');
            y_pred = predict(mdl, X_te);
            
            sens = sum(y_s(te_m) == 1 & y_pred == 1) / max(1, sum(y_s(te_m) == 1));
            spec = sum(y_s(te_m) == 0 & y_pred == 0) / max(1, sum(y_s(te_m) == 0));
            fold_accs(f_i) = (sens + spec) / 2;
        end
        real_acc_shapes(s, w) = mean(fold_accs);
    end
end

real_acc_avg = mean(real_acc_shapes, 1);
fprintf('真实解码完成，耗时: %.3f 秒, 平均峰值: %.2f%%\n', toc(t_start), max(real_acc_avg)*100);

% 平滑曲线
smooth_pts = 5; smooth_typ = 'gaussian';
real_acc_avg_s = smoothdata(real_acc_avg, smooth_typ, smooth_pts);
real_acc_shapes_s = zeros(size(real_acc_shapes));
for s = 1:n_shapes
    real_acc_shapes_s(s, :) = smoothdata(real_acc_shapes(s, :), smooth_typ, smooth_pts);
end

% 快速测试 50 次置换以评估速度
n_perm = 50;
t_perm = tic;
null_dist = zeros(n_perm, n_win);

% 预标准化每个 shape 的折数据
precomputed = cell(n_shapes, n_win, n_folds);
for s = 1:n_shapes
    f_info = shape_folds{s};
    X_s = X_3d(f_info.s_idx, :, :);
    for w = 1:n_win
        X_w = double(squeeze(X_s(:, :, w)));
        for f_i = 1:n_folds
            tr_m = f_info.tr_masks(:, f_i);
            te_m = f_info.te_masks(:, f_i);
            mu = mean(X_w(tr_m, :), 1);
            sig = std(X_w(tr_m, :), 0, 1);
            sig(sig < 1e-6) = 1;
            p_data = struct();
            p_data.X_tr = (X_w(tr_m, :) - mu) ./ sig;
            p_data.X_te = (X_w(te_m, :) - mu) ./ sig;
            precomputed{s, w, f_i} = p_data;
        end
    end
end

parfor p_i = 1:n_perm
    p_shapes_acc = zeros(n_shapes, n_win);
    for s = 1:n_shapes
        f_info = shape_folds{s};
        % 在形状内部置换标签
        y_perm = f_info.y_s(randperm(numel(f_info.y_s)));
        
        for w = 1:n_win
            fold_accs = zeros(1, n_folds);
            for f_i = 1:n_folds
                tr_m = f_info.tr_masks(:, f_i);
                te_m = f_info.te_masks(:, f_i);
                p_d = precomputed{s, w, f_i};
                
                mdl = fitclinear(p_d.X_tr, y_perm(tr_m), 'Learner', 'svm', ...
                    'Regularization', 'ridge', 'Lambda', 1e-2, 'Solver', 'dual');
                y_pred = predict(mdl, p_d.X_te);
                
                sens = sum(y_perm(te_m) == 1 & y_pred == 1) / max(1, sum(y_perm(te_m) == 1));
                spec = sum(y_perm(te_m) == 0 & y_pred == 0) / max(1, sum(y_perm(te_m) == 0));
                fold_accs(f_i) = (sens + spec) / 2;
            end
            p_shapes_acc(s, w) = mean(fold_accs);
        end
    end
    null_dist(p_i, :) = mean(p_shapes_acc, 1);
end

fprintf('%d 次置换完成，耗时: %.2f 秒 (平均每置换 %.3fs)\n', n_perm, toc(t_perm), toc(t_perm)/n_perm);
