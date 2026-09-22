% test_benchmark_speed.m
% 测试特征预标准化与不同 Solver 对 SVM 置换检验速度的提升

clear; clc;
n_trials = 240;
n_dim    = 6;
n_win    = 51;
n_folds  = 5;
n_perm   = 200;

X = randn(n_trials, n_dim, n_win);
y = randi([0, 1], n_trials, 1);

cv_part = cvpartition(y, 'KFold', n_folds);
fold_tr = false(n_trials, n_folds);
fold_te = false(n_trials, n_folds);
for f = 1:n_folds
    fold_tr(:, f) = training(cv_part, f);
    fold_te(:, f) = test(cv_part, f);
end

% 方案 A: 原始方法 (每次在循环内标准化)
fprintf('--- 方案 A: 原始方法 (每次在循环内计算均值标准差) ---\n');
tic;
parfor perm_i = 1:20
    y_p = y(randperm(n_trials));
    for w = 1:n_win
        X_w = squeeze(X(:, :, w));
        for f = 1:n_folds
            tr_m = fold_tr(:, f);
            te_m = fold_te(:, f);
            mu  = mean(X_w(tr_m, :), 1);
            sig = std(X_w(tr_m, :), 0, 1);
            sig(sig < 1e-6) = 1;
            X_tr_s = (X_w(tr_m, :) - mu) ./ sig;
            X_te_s = (X_w(te_m, :) - mu) ./ sig;
            mdl = fitclinear(X_tr_s, y_p(tr_m), 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', 0.01);
            y_pred = predict(mdl, X_te_s);
        end
    end
end
tA = toc;
fprintf('20 次置换耗时: %.2f 秒 (折算 200 次: %.2f 秒)\n', tA, tA * 10);

% 方案 B: 预先标准化特征矩阵
fprintf('\n--- 方案 B: 预先对特征标准化，循环内直接索引 ---\n');
tic;
% 预计算每个窗口每个折的标准化矩阵
X_tr_cell = cell(n_win, n_folds);
X_te_cell = cell(n_win, n_folds);
for w = 1:n_win
    X_w = squeeze(X(:, :, w));
    for f = 1:n_folds
        tr_m = fold_tr(:, f);
        te_m = fold_te(:, f);
        mu  = mean(X_w(tr_m, :), 1);
        sig = std(X_w(tr_m, :), 0, 1);
        sig(sig < 1e-6) = 1;
        X_tr_cell{w, f} = (X_w(tr_m, :) - mu) ./ sig;
        X_te_cell{w, f} = (X_w(te_m, :) - mu) ./ sig;
    end
end

parfor perm_i = 1:20
    y_p = y(randperm(n_trials));
    for w = 1:n_win
        for f = 1:n_folds
            tr_m = fold_tr(:, f);
            te_m = fold_te(:, f);
            mdl = fitclinear(X_tr_cell{w, f}, y_p(tr_m), 'Learner', 'svm', ...
                'Regularization', 'ridge', 'Lambda', 0.01, 'Solver', 'dual');
            y_pred = predict(mdl, X_te_cell{w, f});
        end
    end
end
tB = toc;
fprintf('20 次置换耗时: %.2f 秒 (折算 200 次: %.2f 秒)\n', tB, tB * 10);
