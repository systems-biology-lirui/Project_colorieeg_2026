%% test_intercept_calibration.m
% -------------------------------------------------------------------------
% 验证无监督截距校准 (Unsupervised Intercept Calibration) 效果:
% 1. 严格固定 Task 3 学到的特征权重向量 w (不碰 Task 2 标签)
% 2. 仅在 Task 2 投影空间进行无监督截距中心化 (消除跨任务全局漂移)
% 3. 对比校准前与校准后的分品类正确率曲线 (Strawberry, Watermelon, Cabbage, Kiwi)
% 4. 包含 100 次置换检验与 95% 置信区间
% -------------------------------------------------------------------------
clear; clc; close all;

%% 1. 参数与路径直观配置
cfg = struct();
cfg.root_dir     = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825';
cfg.data_root    = fullfile(cfg.root_dir, 'process_data_new');
cfg.task_info    = fullfile(cfg.root_dir, 'task_info');
cfg.out_fig_dir  = fullfile(cfg.root_dir, 'result', 'figures', 'intercept_calibration_test');

if ~exist(cfg.out_fig_dir, 'dir'), mkdir(cfg.out_fig_dir); end

% 时间窗与频段
cfg.win_len      = 20;                             % ms
cfg.win_step     = 20;                             % ms
cfg.t_range      = [-200, 800];                    % ms
cfg.bands        = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.n_bands      = numel(cfg.bands);
cfg.svm_lambda   = 0.01;                           % 岭正则化参数
cfg.smooth_pts   = 5;                              % 平滑点数
cfg.n_perm       = 100;                            % 置换检验次数

% 测试典型电极:
% 1. sub007-C4: 此前红绿镜像倒挂 (35% vs 65%) 的纯偏移位点
% 2. sub008-C10: 经典强响应真实记忆色位点
% 3. sub004-L5: 具有轻度基线偏移的显著位点
test_targets = {
    'sub001', 'B3';
};
n_targets = size(test_targets, 1);

fprintf('========================================================================\n');
fprintf('>>> 启动无监督截距校准 (Intercept Calibration) 验证测试\n');
fprintf('>>> 测试电极数: %d\n', n_targets);
fprintf('========================================================================\n');

% 开启并行池
p_pool = gcp('nocreate');
if isempty(p_pool)
    parpool('local', 16);
end

%% 2. 逐通道对比校准前后效果
t_starts  = cfg.t_range(1) : cfg.win_step : (cfg.t_range(2) - cfg.win_len);
n_win     = numel(t_starts);
t_centers = t_starts + cfg.win_len / 2;

for t_i = 1:n_targets
    sub_id = test_targets{t_i, 1};
    elec   = test_targets{t_i, 2};
    fprintf('\n>>> [%d/%d] 正在处理: %s-%s ...\n', t_i, n_targets, sub_id, elec);
    
    % --- 2.1 加载数据 ---
    t3_mat = fullfile(cfg.data_root, sub_id, 'task3_multiband_epoched.mat');
    t2_mat = fullfile(cfg.data_root, sub_id, 'task2_multiband_epoched.mat');
    
    t_axis = double(h5read(t3_mat, '/epoched_data/time_ms'));
    t_axis = t_axis(:)';
    ch_list3 = h5read(t3_mat, '/epoched_data/channels');
    ch_list2 = h5read(t2_mat, '/epoched_data/channels');
    if iscell(ch_list3), ch_list3 = cellfun(@(x) char(x(:)'), ch_list3, 'UniformOutput', false); end
    if iscell(ch_list2), ch_list2 = cellfun(@(x) char(x(:)'), ch_list2, 'UniformOutput', false); end
    
    e_idx3 = find(strcmp(ch_list3, elec), 1);
    e_idx2 = find(strcmp(ch_list2, elec), 1);
    
    d3 = load(fullfile(cfg.task_info, sub_id, 'task3_trial_info.mat'), 'trial_info');
    ti3 = d3.trial_info;
    tr_mask = strcmp(ti3.color, 'red') | strcmp(ti3.color, 'green');
    ti3_use = ti3(tr_mask, :);
    y_tr = zeros(height(ti3_use), 1);
    y_tr(strcmp(ti3_use.color, 'red')) = 1; % 1: Red, 0: Green
    n_tr = height(ti3_use);
    
    d2 = load(fullfile(cfg.task_info, sub_id, 'task2_trial_info.mat'), 'trial_info');
    ti2 = d2.trial_info;
    te_mask = strcmp(ti2.state, 'gray');
    ti2_use = ti2(te_mask, :);
    
    fruit_list = ti2_use.fruit;
    m_straw = strcmp(fruit_list, 'strawberry');
    m_water = strcmp(fruit_list, 'watermelon');
    m_cabb  = strcmp(fruit_list, 'cabbage');
    m_kiwi  = strcmp(fruit_list, 'kiwi');
    n_te    = height(ti2_use);
    
    % 读取频段数据
    X3_bands = zeros(n_tr, cfg.n_bands, numel(t_axis));
    X2_bands = zeros(n_te, cfg.n_bands, numel(t_axis));
    for b = 1:cfg.n_bands
        raw3 = h5read(t3_mat, ['/epoched_data/' cfg.bands{b}]);
        X3_bands(:, b, :) = raw3(tr_mask, e_idx3, :);
        raw2 = h5read(t2_mat, ['/epoched_data/' cfg.bands{b}]);
        X2_bands(:, b, :) = raw2(te_mask, e_idx2, :);
        clear raw3 raw2;
    end
    
    % 构建滑动窗
    X3_3d = zeros(n_tr, cfg.n_bands, n_win);
    X2_3d = zeros(n_te, cfg.n_bands, n_win);
    for w = 1:n_win
        w_t1 = t_starts(w);
        w_t2 = w_t1 + cfg.win_len;
        t_mask = (t_axis >= w_t1) & (t_axis < w_t2);
        for b = 1:cfg.n_bands
            X3_3d(:, b, w) = mean(X3_bands(:, b, t_mask), 3);
            X2_3d(:, b, w) = mean(X2_bands(:, b, t_mask), 3);
        end
    end
    clear X3_bands X2_bands;
    
    % --- 2.2 计算校准前 (Raw) 与 校准后 (Calibrated) 正确率 ---
    % 校准前 (Raw)
    raw_acc_straw = zeros(1, n_win);
    raw_acc_water = zeros(1, n_win);
    raw_acc_cabb  = zeros(1, n_win);
    raw_acc_kiwi  = zeros(1, n_win);
    raw_acc_mean  = zeros(1, n_win);
    
    % 校准后 (Calibrated)
    cal_acc_straw = zeros(1, n_win);
    cal_acc_water = zeros(1, n_win);
    cal_acc_cabb  = zeros(1, n_win);
    cal_acc_kiwi  = zeros(1, n_win);
    cal_acc_mean  = zeros(1, n_win);
    
    X3_norm_all = cell(n_win, 1);
    X2_norm_all = cell(n_win, 1);
    
    for w = 1:n_win
        X_tr_w = double(squeeze(X3_3d(:, :, w)));
        X_te_w = double(squeeze(X2_3d(:, :, w)));
        
        mu_w  = mean(X_tr_w, 1);
        sig_w = std(X_tr_w, 0, 1);
        sig_w(sig_w < 1e-6) = 1;
        
        X_tr_norm = (X_tr_w - mu_w) ./ sig_w;
        X_te_norm = (X_te_w - mu_w) ./ sig_w;
        
        X3_norm_all{w} = X_tr_norm;
        X2_norm_all{w} = X_te_norm;
        
        % 训练 Task 3 线性 SVM (学得权重向量 w 和原始截距 b3)
        mdl = fitclinear(X_tr_norm, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', cfg.svm_lambda);
        
        % 提取权重向量 w 和原始截距 b
        w_vec = mdl.Beta;
        b_raw = mdl.Bias;
        
        % 1. 原始未校准决策值与预测 (Raw)
        score_raw = X_te_norm * w_vec + b_raw;
        y_pred_raw = double(score_raw > 0);
        
        raw_acc_straw(w) = sum(y_pred_raw(m_straw) == 1) / sum(m_straw);
        raw_acc_water(w) = sum(y_pred_raw(m_water) == 1) / sum(m_water);
        raw_acc_cabb(w)  = sum(y_pred_raw(m_cabb)  == 0) / sum(m_cabb);
        raw_acc_kiwi(w)  = sum(y_pred_raw(m_kiwi)  == 0) / sum(m_kiwi);
        raw_acc_mean(w)  = (raw_acc_straw(w) + raw_acc_water(w) + raw_acc_cabb(w) + raw_acc_kiwi(w)) / 4;
        
        % 2. 无监督中位数/均值截距校准 (Calibrated)
        % 纯无监督操作: 将 Task 2 全体试次的投影值做中位数/零均值校准，彻底消除截距偏移
        b_calib = -median(X_te_norm * w_vec); % 也可以用 -mean
        score_cal = X_te_norm * w_vec + b_calib;
        y_pred_cal = double(score_cal > 0);
        
        cal_acc_straw(w) = sum(y_pred_cal(m_straw) == 1) / sum(m_straw);
        cal_acc_water(w) = sum(y_pred_cal(m_water) == 1) / sum(m_water);
        cal_acc_cabb(w)  = sum(y_pred_cal(m_cabb)  == 0) / sum(m_cabb);
        cal_acc_kiwi(w)  = sum(y_pred_cal(m_kiwi)  == 0) / sum(m_kiwi);
        cal_acc_mean(w)  = (cal_acc_straw(w) + cal_acc_water(w) + cal_acc_cabb(w) + cal_acc_kiwi(w)) / 4;
    end
    
    % --- 2.3 校准后正确率的 100 次置换检验 ---
    perm_cal_mat = zeros(cfg.n_perm, n_win);
    svm_lambda = cfg.svm_lambda;
    
    parfor p = 1:cfg.n_perm
        y_perm = y_tr(randperm(n_tr));
        row_cal = zeros(1, n_win);
        for w = 1:n_win
            mdl_p = fitclinear(X3_norm_all{w}, y_perm, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', svm_lambda);
            w_p = mdl_p.Beta;
            X_te_w = X2_norm_all{w};
            b_cal_p = -median(X_te_w * w_p);
            sc_p = X_te_w * w_p + b_cal_p;
            yp_p = double(sc_p > 0);
            
            s_p = sum(yp_p(m_straw) == 1) / sum(m_straw);
            w_p_acc = sum(yp_p(m_water) == 1) / sum(m_water);
            c_p = sum(yp_p(m_cabb)  == 0) / sum(m_cabb);
            k_p = sum(yp_p(m_kiwi)  == 0) / sum(m_kiwi);
            row_cal(w) = (s_p + w_p_acc + c_p + k_p) / 4;
        end
        perm_cal_mat(p, :) = row_cal;
    end
    
    cal_p95 = prctile(perm_cal_mat, 95, 1);
    
    % --- 2.4 对比画图 (左右双面板: 校准前 vs 校准后) ---
    h_fig = figure('Visible', 'off', 'Units', 'pixels', 'Position', [100, 100, 1100, 420], 'Color', 'w');
    
    % 配色定义
    c_straw = [0.84, 0.19, 0.15]; % 深红 (Strawberry)
    c_water = [0.94, 0.50, 0.50]; % 浅红 (Watermelon)
    c_cabb  = [0.13, 0.59, 0.45]; % 翠绿 (Cabbage)
    c_kiwi  = [0.45, 0.76, 0.46]; % 浅绿 (Kiwi)
    c_mean  = [0.85, 0.35, 0.05]; % 焦橙色加粗 (Overall Mean)
    
    % 左子图: 校准前 (Before Calibration - Raw Task 3 Intercept)
    subplot(1, 2, 1);
    hold on;
    yline(0.5, 'k--', 'LineWidth', 1.0, 'Alpha', 0.6);
    xline(0, 'k:', 'LineWidth', 1.0, 'Alpha', 0.5);
    p1 = plot(t_centers, smoothdata(raw_acc_straw, 'gaussian', cfg.smooth_pts), 'Color', c_straw, 'LineWidth', 1.3);
    p2 = plot(t_centers, smoothdata(raw_acc_water, 'gaussian', cfg.smooth_pts), 'Color', c_water, 'LineWidth', 1.3);
    p3 = plot(t_centers, smoothdata(raw_acc_cabb,  'gaussian', cfg.smooth_pts), 'Color', c_cabb,  'LineWidth', 1.3);
    p4 = plot(t_centers, smoothdata(raw_acc_kiwi,  'gaussian', cfg.smooth_pts), 'Color', c_kiwi,  'LineWidth', 1.3);
    p5 = plot(t_centers, smoothdata(raw_acc_mean,  'gaussian', cfg.smooth_pts), 'Color', c_mean,  'LineWidth', 2.3);
    hold off;
    box off; set(gca, 'TickDir', 'out', 'FontSize', 10, 'LineWidth', 1.0);
    xlim([-200, 800]); ylim([0.15, 0.85]);
    xlabel('Time (ms)', 'FontSize', 11); ylabel('Decoding Accuracy', 'FontSize', 11);
    title(sprintf('A. Before Calibration (%s-%s: Raw Intercept)', sub_id, elec), 'FontSize', 12, 'FontWeight', 'bold');
    legend([p1, p2, p3, p4, p5], {'Strawberry', 'Watermelon', 'Cabbage', 'Kiwi', 'Overall Mean'}, ...
           'Location', 'best', 'FontSize', 8, 'Box', 'off');
    
    % 右子图: 校准后 (After Calibration - Unsupervised Median Centering)
    subplot(1, 2, 2);
    hold on;
    yline(0.5, 'k--', 'LineWidth', 1.0, 'Alpha', 0.6);
    xline(0, 'k:', 'LineWidth', 1.0, 'Alpha', 0.5);
    % 填充置换检验 95% 置信带
    fill([t_centers, fliplr(t_centers)], [cal_p95, fliplr(ones(1, n_win)*0.5)], ...
         [0.85, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    p1 = plot(t_centers, smoothdata(cal_acc_straw, 'gaussian', cfg.smooth_pts), 'Color', c_straw, 'LineWidth', 1.3);
    p2 = plot(t_centers, smoothdata(cal_acc_water, 'gaussian', cfg.smooth_pts), 'Color', c_water, 'LineWidth', 1.3);
    p3 = plot(t_centers, smoothdata(cal_acc_cabb,  'gaussian', cfg.smooth_pts), 'Color', c_cabb,  'LineWidth', 1.3);
    p4 = plot(t_centers, smoothdata(cal_acc_kiwi,  'gaussian', cfg.smooth_pts), 'Color', c_kiwi,  'LineWidth', 1.3);
    p5 = plot(t_centers, smoothdata(cal_acc_mean,  'gaussian', cfg.smooth_pts), 'Color', c_mean,  'LineWidth', 2.3);
    hold off;
    box off; set(gca, 'TickDir', 'out', 'FontSize', 10, 'LineWidth', 1.0);
    xlim([-200, 800]); ylim([0.15, 0.85]);
    xlabel('Time (ms)', 'FontSize', 11); ylabel('Decoding Accuracy', 'FontSize', 11);
    title(sprintf('B. After Calibration (Unsupervised Zero-Center)', sub_id, elec), 'FontSize', 12, 'FontWeight', 'bold');
    legend([p1, p2, p3, p4, p5], {'Strawberry', 'Watermelon', 'Cabbage', 'Kiwi', 'Overall Mean'}, ...
           'Location', 'best', 'FontSize', 8, 'Box', 'off');
    
    fig_png = fullfile(cfg.out_fig_dir, sprintf('%s_%s_calibration_comparison.png', sub_id, elec));
    exportgraphics(h_fig, fig_png, 'Resolution', 300);
    close(h_fig);
    
    fprintf('  - %s-%s 处理完成! 图表保存至: %s\n', sub_id, elec, fig_png);
end

fprintf('\n========================================================================\n');
fprintf('>>> 截距校准对比测试完成! 请查看生成的对比图表。\n');
fprintf('========================================================================\n');
