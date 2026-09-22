%% test_thread_pool.m
clear; clc;
try
    pool = gcp('nocreate');
    if ~isempty(pool), delete(pool); end
    
    t_start = tic;
    fprintf('>>> Testing parpool("threads") ...\n');
    pool = parpool('threads');
    fprintf('[+] Thread pool started successfully! NumWorkers = %d (耗时 %.2f 秒)\n', ...
        pool.NumWorkers, toc(t_start));
    
    % 测试并行矩阵运算与 fitclinear
    X = randn(120, 6);
    y = randi([0, 1], 120, 1);
    acc = zeros(1, 20);
    parfor i = 1:20
        mdl = fitclinear(X, y, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', 0.01);
        y_pred = predict(mdl, X);
        acc(i) = mean(y == y_pred);
    end
    fprintf('[+] Parfor with fitclinear executed successfully on threads! Mean acc: %.2f%%\n', mean(acc)*100);
    delete(pool);
catch ME
    fprintf('[-] Error with thread pool: %s\n', ME.message);
end
