% test_compare_svm_ridge.m
clear; clc;
mat_file = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub001/task3_multiband_epoched.mat';
d = load(mat_file, 'epoched_data');
ep = d.epoched_data;
ti = ep.trial_info;
rg_mask = strcmp(ti.color, 'red') | strcmp(ti.color, 'green');
ti_rg = ti(rg_mask, :);
ch_idx = find(strcmp(ep.channels, 'D15'), 1);

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
t_centers = -200:20:800; n_win = numel(t_centers);

X_3d = zeros(height(ti_rg), 6, n_win, 'single');
for b = 1:6
    raw_b = squeeze(ep.(bands{b})(rg_mask, ch_idx, :));
    for w = 1:n_win
        tm = (ep.time_ms >= (t_centers(w) - 10)) & (ep.time_ms < (t_centers(w) + 10));
        X_3d(:, b, w) = mean(raw_b(:, tm), 2);
    end
end

shapes = unique(ti_rg.pic_id);
n_shapes = numel(shapes); n_folds = 5;

% 预构建折
shape_data = cell(n_shapes, 1);
for s = 1:n_shapes
    s_idx = find(ti_rg.pic_id == shapes(s));
    y_s = strcmp(ti_rg.color(s_idx), 'red');
    rng(42 + s);
    cv_r = cvpartition(sum(y_s==1), 'KFold', n_folds);
    cv_g = cvpartition(sum(y_s==0), 'KFold', n_folds);
    r_sub = find(y_s==1); g_sub = find(y_s==0);
    
    tr_m = false(numel(s_idx), n_folds);
    te_m = false(numel(s_idx), n_folds);
    for f = 1:n_folds
        te = [r_sub(test(cv_r, f)); g_sub(test(cv_g, f))];
        te_m(te, f) = true;
        tr_m(:, f) = ~te_m(:, f);
    end
    s_st = struct('s_idx', s_idx, 'y', double(y_s), 'tr_m', tr_m, 'te_m', te_m);
    shape_data{s} = s_st;
end

% 1. 测试 SVM fitclinear 耗时与结果
t_svm = tic;
acc_svm_shapes = zeros(n_shapes, n_win);
for s = 1:n_shapes
    st = shape_data{s};
    X_s = double(X_3d(st.s_idx, :, :));
    for w = 1:n_win
        X_w = squeeze(X_s(:, :, w));
        f_acc = zeros(1, n_folds);
        for f = 1:n_folds
            mu = mean(X_w(st.tr_m(:, f), :), 1);
            sig = std(X_w(st.tr_m(:, f), :), 0, 1); sig(sig < 1e-6) = 1;
            Xtr = (X_w(st.tr_m(:, f), :) - mu) ./ sig;
            Xte = (X_w(st.te_m(:, f), :) - mu) ./ sig;
            mdl = fitclinear(Xtr, st.y(st.tr_m(:, f)), 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', 0.01);
            pred = predict(mdl, Xte);
            sens = sum(st.y(st.te_m(:, f))==1 & pred==1)/sum(st.y(st.te_m(:, f))==1);
            spec = sum(st.y(st.te_m(:, f))==0 & pred==0)/sum(st.y(st.te_m(:, f))==0);
            f_acc(f) = (sens+spec)/2;
        end
        acc_svm_shapes(s, w) = mean(f_acc);
    end
end
acc_svm = mean(acc_svm_shapes, 1);
fprintf('SVM 耗时: %.3f 秒, 峰值: %.2f%%\n', toc(t_svm), max(acc_svm)*100);

% 2. 测试闭式解 Ridge (带有预计算矩阵加速)
t_ridge = tic;
lambda = 0.1;
% 预计算投影矩阵 P: [n_shapes, n_win, n_folds]
P_cell = cell(n_shapes, n_win, n_folds);
Xte_cell = cell(n_shapes, n_win, n_folds);
for s = 1:n_shapes
    st = shape_data{s};
    X_s = double(X_3d(st.s_idx, :, :));
    for w = 1:n_win
        X_w = squeeze(X_s(:, :, w));
        for f = 1:n_folds
            mu = mean(X_w(st.tr_m(:, f), :), 1);
            sig = std(X_w(st.tr_m(:, f), :), 0, 1); sig(sig < 1e-6) = 1;
            Xtr = [(X_w(st.tr_m(:, f), :) - mu) ./ sig, ones(sum(st.tr_m(:, f)), 1)];
            Xte = [(X_w(st.te_m(:, f), :) - mu) ./ sig, ones(sum(st.te_m(:, f)), 1)];
            I_reg = eye(size(Xtr, 2)); I_reg(end, end) = 0;
            P_cell{s, w, f} = (Xtr' * Xtr + lambda * I_reg) \ Xtr';
            Xte_cell{s, w, f} = Xte;
        end
    end
end

acc_ridge_shapes = zeros(n_shapes, n_win);
for s = 1:n_shapes
    st = shape_data{s};
    y_pm1 = 2 * st.y - 1;
    for w = 1:n_win
        f_acc = zeros(1, n_folds);
        for f = 1:n_folds
            w_vec = P_cell{s, w, f} * y_pm1(st.tr_m(:, f));
            pred = Xte_cell{s, w, f} * w_vec;
            y_te = y_pm1(st.te_m(:, f));
            sens = sum(y_te == 1 & pred > 0) / sum(y_te == 1);
            spec = sum(y_te == -1 & pred <= 0) / sum(y_te == -1);
            f_acc(f) = (sens + spec) / 2;
        end
        acc_ridge_shapes(s, w) = mean(f_acc);
    end
end
acc_ridge = mean(acc_ridge_shapes, 1);
fprintf('Ridge 预计算+求解总耗时: %.3f 秒, 峰值: %.2f%%\n', toc(t_ridge), max(acc_ridge)*100);
fprintf('SVM 与 Ridge 曲线相关系数: r = %.4f\n', corr(acc_svm', acc_ridge'));

% 测试 200 次置换速度
t_null = tic;
n_perm = 200;
null_dist = zeros(n_perm, n_win);
for p = 1:n_perm
    p_shapes = zeros(n_shapes, n_win);
    for s = 1:n_shapes
        st = shape_data{s};
        y_perm = 2 * st.y(randperm(numel(st.y))) - 1;
        for w = 1:n_win
            f_acc = zeros(1, n_folds);
            for f = 1:n_folds
                w_vec = P_cell{s, w, f} * y_perm(st.tr_m(:, f));
                pred = Xte_cell{s, w, f} * w_vec;
                y_te = y_perm(st.te_m(:, f));
                sens = sum(y_te == 1 & pred > 0) / sum(y_te == 1);
                spec = sum(y_te == -1 & pred <= 0) / sum(y_te == -1);
                f_acc(f) = (sens + spec) / 2;
            end
            p_shapes(s, w) = mean(f_acc);
        end
    end
    null_dist(p, :) = mean(p_shapes, 1);
end
fprintf('200 次置换耗时: %.3f 秒 (平均每通道全部置换只需要 %.3f 秒)!\n', toc(t_null), toc(t_null));
