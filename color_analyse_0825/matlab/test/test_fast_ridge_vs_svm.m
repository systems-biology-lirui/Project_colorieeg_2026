% test_fast_ridge_vs_svm.m
clear; clc;
run('test_within_shape_decoding.m');

% 对比快速解析解 (Closed-form Ridge) 与 fitclinear
acc_svm = real_acc_avg;

% 快速解析解
acc_ridge_shapes = zeros(n_shapes, n_win);
lambda = 0.1;

for s = 1:n_shapes
    f_info = shape_folds{s};
    y_pm1 = 2 * f_info.y_s - 1; % 转换为 +1 / -1
    
    for w = 1:n_win
        fold_accs = zeros(1, n_folds);
        for f_i = 1:n_folds
            p_d = precomputed{s, w, f_i};
            X_tr = [p_d.X_tr, ones(size(p_d.X_tr, 1), 1)]; % 加截距项
            X_te = [p_d.X_te, ones(size(p_d.X_te, 1), 1)];
            
            % 闭式解权重: w = (X'*X + lambda*I) \ (X'*y)
            I_reg = eye(size(X_tr, 2)); I_reg(end, end) = 0; % 截距不正则化
            w = (X_tr' * X_tr + lambda * I_reg) \ (X_tr' * y_pm1(f_info.tr_masks(:, f_i)));
            
            y_pred = X_te * w;
            y_te = y_pm1(f_info.te_masks(:, f_i));
            sens = sum(y_te == 1 & y_pred > 0) / max(1, sum(y_te == 1));
            spec = sum(y_te == -1 & y_pred <= 0) / max(1, sum(y_te == -1));
            fold_accs(f_i) = (sens + spec) / 2;
        end
        acc_ridge_shapes(s, w) = mean(fold_accs);
    end
end
acc_ridge = mean(acc_ridge_shapes, 1);

fprintf('SVM vs Ridge 相关性: r = %.4f\n', corr(acc_svm', acc_ridge'));
fprintf('SVM 峰值: %.2f%%, Ridge 峰值: %.2f%%\n', max(acc_svm)*100, max(acc_ridge)*100);

% 测试 200 次置换在解析解下的耗时
t_fast_perm = tic;
n_perm = 200;

% 预计算投影矩阵: P_mat = (X_tr' * X_tr + lambda * I_reg) \ X_tr'
P_cell = cell(n_shapes, n_win, n_folds);
Xte_cell = cell(n_shapes, n_win, n_folds);
for s = 1:n_shapes
    f_info = shape_folds{s};
    for w = 1:n_win
        for f_i = 1:n_folds
            p_d = precomputed{s, w, f_i};
            X_tr = [p_d.X_tr, ones(size(p_d.X_tr, 1), 1)];
            X_te = [p_d.X_te, ones(size(p_d.X_te, 1), 1)];
            I_reg = eye(size(X_tr, 2)); I_reg(end, end) = 0;
            P_cell{s, w, f_i} = (X_tr' * X_tr + lambda * I_reg) \ X_tr';
            Xte_cell{s, w, f_i} = X_te;
        end
    end
end

null_fast = zeros(n_perm, n_win);
for p = 1:n_perm
    p_shapes = zeros(n_shapes, n_win);
    for s = 1:n_shapes
        f_info = shape_folds{s};
        y_pm1 = 2 * f_info.y_s(randperm(numel(f_info.y_s))) - 1;
        for w = 1:n_win
            f_acc = zeros(1, n_folds);
            for f_i = 1:n_folds
                tr_m = f_info.tr_masks(:, f_i);
                te_m = f_info.te_masks(:, f_i);
                w_vec = P_cell{s, w, f_i} * y_pm1(tr_m);
                pred = Xte_cell{s, w, f_i} * w_vec;
                y_te = y_pm1(te_m);
                sens = sum(y_te == 1 & pred > 0) / max(1, sum(y_te == 1));
                spec = sum(y_te == -1 & pred <= 0) / max(1, sum(y_te == -1));
                f_acc(f_i) = (sens + spec) / 2;
            end
            p_shapes(s, w) = mean(f_acc);
        end
    end
    null_fast(p, :) = mean(p_shapes, 1);
end
fprintf('200 次解析置换耗时: %.3f 秒 (单次置换 %.4f ms)!\n', toc(t_fast_perm), toc(t_fast_perm)/n_perm*1000);
