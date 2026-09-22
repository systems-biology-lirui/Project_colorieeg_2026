% test_solvers.m
clear; clc;
X_tr = randn(192, 6);
y_tr = randi([0, 1], 192, 1);
X_te = randn(48, 6);

solvers = {'dual', 'lbfgs', 'sgd', 'asgd'};
for s = 1:numel(solvers)
    sol = solvers{s};
    try
        tic;
        for i = 1:500
            mdl = fitclinear(X_tr, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', 0.01, 'Solver', sol);
            p = predict(mdl, X_te);
        end
        fprintf('Solver %s: %.3f 秒 (500 次)\n', sol, toc);
    catch ME
        fprintf('Solver %s 报错: %s\n', sol, ME.message);
    end
end
