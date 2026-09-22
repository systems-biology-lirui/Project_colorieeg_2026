%% test_asymmetry_and_auc.m
% -------------------------------------------------------------------------
% 验证 cross_decoding_asymmetry_analysis.md 中提出的方案:
% 1. 提取 Task2 连续决策值 (Decision Score: w*x + b)
% 2. 计算红绿记忆 AUC (与分类截距 b 无关的真正排序能力)
% 3. 统计全试次预测偏置 (Prediction Bias: 预测为 Red vs Green 的比例)
% 4. 验证四种水果的一致性与真假效应判定逻辑
% 5. 测算单通道及全量 157 通道耗时评估
% -------------------------------------------------------------------------
clear; clc; close all;

%% 1. 参数与路径配置 (结构体直观可调)
cfg = struct();
cfg.root_dir     = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825';
cfg.data_root    = fullfile(cfg.root_dir, 'process_data_new');
cfg.task_info    = fullfile(cfg.root_dir, 'task_info');
cfg.out_fig_dir  = fullfile(cfg.root_dir, 'result', 'figures', 'asymmetry_validation');
cfg.out_tab_dir  = fullfile(cfg.root_dir, 'result', 'tables', 'asymmetry_validation');

if ~exist(cfg.out_fig_dir, 'dir'), mkdir(cfg.out_fig_dir); end
if ~exist(cfg.out_tab_dir, 'dir'), mkdir(cfg.out_tab_dir); end

% 时间窗与频段 (与成熟流水线严格对齐)
cfg.win_len    = 20;    % ms (20ms滑动窗)
cfg.win_step   = 20;    % ms
cfg.time_range = [-200, 800]; % ms (共51个时间窗)
cfg.bands      = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.n_bands    = numel(cfg.bands);
cfg.svm_lambda = 0.01;
cfg.n_perm     = 100;   % 置换次数 (用于评估置换耗时)
cfg.smooth_pts = 5;     % 5点高斯平滑

% 选择典型验证电极: 
% 1. sub007-C4: 疑似"只有绿色"、红绿相反的纯偏移位点
% 2. sub008-C10: 四种水果同时显著的真记忆色位点
% 3. sub004-L5: 另一具有显著时间簇的高响应位点
test_targets = {
    'sub001', 'D7';
    'sub001', 'D8';
};
n_targets = size(test_targets, 1);

fprintf('========================================================================\n');
fprintf('>>> 启动跨任务不对称性与 AUC 验证测试 (共 %d 个典型通道)\n', n_targets);
fprintf('========================================================================\n');

%% 2. 循环测试每个目标电极
target_timings = zeros(n_targets, 2); % [无置换耗时, 带置换耗时]
results_summary = cell(n_targets, 9);

for t_i = 1:n_targets
    sub_id = test_targets{t_i, 1};
    elec   = test_targets{t_i, 2};
    fprintf('\n>>> [%d/%d] 正在分析: %s-%s ...\n', t_i, n_targets, sub_id, elec);
    
    t_start_total = tic;
    
    % --- 2.1 加载 Task 3 (训练集) ---
    t3_mat = fullfile(cfg.data_root, sub_id, 'task3_multiband_epoched.mat');
    ch_list3 = h5read(t3_mat, '/epoched_data/channels');
    if iscell(ch_list3)
        ch_list3 = cellfun(@(x) char(x(:)'), ch_list3, 'UniformOutput', false);
    end
    t_axis = double(h5read(t3_mat, '/epoched_data/time_ms'));
    t_axis = t_axis(:)';
    
    d3 = load(fullfile(cfg.task_info, sub_id, 'task3_trial_info.mat'), 'trial_info');
    ti3 = d3.trial_info;
    tr_mask = strcmp(ti3.color, 'red') | strcmp(ti3.color, 'green');
    ti3_use = ti3(tr_mask, :);
    y_tr = zeros(height(ti3_use), 1);
    y_tr(strcmp(ti3_use.color, 'red')) = 1; % 1: Red, 0: Green
    
    e_idx3 = find(strcmp(ch_list3, elec), 1);
    if isempty(e_idx3)
        error('通道 %s 不存在于 %s 的 Task 3 数据中!', elec, sub_id);
    end
    
    % 读取 Task 3 特征 (目标通道切片)
    n_tr = height(ti3_use);
    n_tp = numel(t_axis);
    X3_bands = zeros(n_tr, cfg.n_bands, n_tp);
    for b = 1:cfg.n_bands
        raw = h5read(t3_mat, ['/epoched_data/' cfg.bands{b}]);
        X3_bands(:, b, :) = raw(tr_mask, e_idx3, :);
        clear raw;
    end
    
    % --- 2.2 加载 Task 2 (测试集) ---
    t2_mat = fullfile(cfg.data_root, sub_id, 'task2_multiband_epoched.mat');
    ch_list2 = h5read(t2_mat, '/epoched_data/channels');
    if iscell(ch_list2)
        ch_list2 = cellfun(@(x) char(x(:)'), ch_list2, 'UniformOutput', false);
    end
    d2 = load(fullfile(cfg.task_info, sub_id, 'task2_trial_info.mat'), 'trial_info');
    ti2 = d2.trial_info;
    te_mask = strcmp(ti2.state, 'gray');
    ti2_use = ti2(te_mask, :);
    
    fruit_list = ti2_use.fruit;
    m_straw = strcmp(fruit_list, 'strawberry');
    m_water = strcmp(fruit_list, 'watermelon');
    m_cabb  = strcmp(fruit_list, 'cabbage');
    m_kiwi  = strcmp(fruit_list, 'kiwi');
    y_te    = double(m_straw | m_water); % 1: Red, 0: Green
    n_te    = height(ti2_use);
    
    e_idx2 = find(strcmp(ch_list2, elec), 1);
    if isempty(e_idx2)
        error('通道 %s 不存在于 %s 的 Task 2 数据中!', elec, sub_id);
    end
    
    X2_bands = zeros(n_te, cfg.n_bands, n_tp);
    for b = 1:cfg.n_bands
        raw = h5read(t2_mat, ['/epoched_data/' cfg.bands{b}]);
        X2_bands(:, b, :) = raw(te_mask, e_idx2, :);
        clear raw;
    end
    
    % --- 2.3 构建滑动时间窗特征 ---
    t_starts = cfg.time_range(1):cfg.win_step:(cfg.time_range(2) - cfg.win_len);
    n_win    = numel(t_starts);
    t_centers = t_starts + cfg.win_len / 2;
    
    X3_3d = zeros(n_tr, cfg.n_bands, n_win);
    X2_3d = zeros(n_te, cfg.n_bands, n_win);
    for w = 1:n_win
        w_t1 = t_starts(w);
        w_t2 = w_t1 + cfg.win_len;
        t_mask = (t_axis >= w_t1) & (t_axis < w_t2);
        X3_3d(:, :, w) = mean(X3_bands(:, :, t_mask), 3);
        X2_3d(:, :, w) = mean(X2_bands(:, :, t_mask), 3);
    end
    clear X3_bands X2_bands;
    
    % --- 2.4 计算连续决策值、AUC、平衡准确率与预测偏倚 ---
    t_calc_no_perm = tic;
    
    dec_score_straw = zeros(1, n_win);
    dec_score_water = zeros(1, n_win);
    dec_score_cabb  = zeros(1, n_win);
    dec_score_kiwi  = zeros(1, n_win);
    dec_score_diff  = zeros(1, n_win); % Red_mean - Green_mean
    
    auc_timecourse  = zeros(1, n_win);
    bal_acc         = zeros(1, n_win);
    pred_red_prop   = zeros(1, n_win);
    
    straw_acc = zeros(1, n_win);
    water_acc = zeros(1, n_win);
    cabb_acc  = zeros(1, n_win);
    kiwi_acc  = zeros(1, n_win);
    
    X3_norm_all = cell(n_win, 1);
    X2_norm_all = cell(n_win, 1);
    
    for w = 1:n_win
        X_tr_w = double(squeeze(X3_3d(:, :, w)));
        X_te_w = double(squeeze(X2_3d(:, :, w)));
        
        % 标准化: 基于 Task 3 均值与标准差
        mu_w  = mean(X_tr_w, 1);
        sig_w = std(X_tr_w, 0, 1);
        sig_w(sig_w < 1e-6) = 1;
        
        X_tr_norm = (X_tr_w - mu_w) ./ sig_w;
        X_te_norm = (X_te_w - mu_w) ./ sig_w;
        
        X3_norm_all{w} = X_tr_norm;
        X2_norm_all{w} = X_te_norm;
        
        % 训练 Task 3 线性 SVM (0: Green, 1: Red)
        mdl = fitclinear(X_tr_norm, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', cfg.svm_lambda);
        
        % 预测 Task 2: 获取离散预测与连续得分 (scores(:, 2) 为正类 Red 的判决距离 w*x + b)
        [y_pred, scores] = predict(mdl, X_te_norm);
        trial_scores = scores(:, 2); % 正值代表偏向 Red，负值代表偏向 Green
        
        % 1. 四类水果各自的连续决策得分均值
        dec_score_straw(w) = mean(trial_scores(m_straw));
        dec_score_water(w) = mean(trial_scores(m_water));
        dec_score_cabb(w)  = mean(trial_scores(m_cabb));
        dec_score_kiwi(w)  = mean(trial_scores(m_kiwi));
        % 真正的四线总平均 (Overall Mean: 真实展现跨任务基线整体偏移)
        dec_score_mean(w)  = (dec_score_straw(w) + dec_score_water(w) + ...
                              dec_score_cabb(w)  + dec_score_kiwi(w)) / 4;
        dec_score_diff(w)  = ((dec_score_straw(w) + dec_score_water(w)) - ...
                              (dec_score_cabb(w)  + dec_score_kiwi(w))) / 2;
        
        % 2. 红绿 AUC (基于 Mann-Whitney U 公式，完全与分类截距 b 无关)
        pos_scores = trial_scores(y_te == 1);
        neg_scores = trial_scores(y_te == 0);
        auc_val = (sum(sum(pos_scores > neg_scores')) + 0.5 * sum(sum(pos_scores == neg_scores'))) / ...
                  (numel(pos_scores) * numel(neg_scores));
        auc_timecourse(w) = auc_val;
        
        % 3. 预测偏置: 无论真标签为何，模型将试次预测为 Red 的百分比
        pred_red_prop(w) = mean(y_pred == 1);
        
        % 4. 分品类正确率与平衡正确率
        straw_acc(w) = sum(y_pred(m_straw) == 1) / max(1, sum(m_straw));
        water_acc(w) = sum(y_pred(m_water) == 1) / max(1, sum(m_water));
        cabb_acc(w)  = sum(y_pred(m_cabb)  == 0) / max(1, sum(m_cabb));
        kiwi_acc(w)  = sum(y_pred(m_kiwi)  == 0) / max(1, sum(m_kiwi));
        bal_acc(w)   = (straw_acc(w) + water_acc(w) + cabb_acc(w) + kiwi_acc(w)) / 4;
    end
    t_no_perm = toc(t_calc_no_perm);
    
    % --- 2.5 测算 100 次置换检验的时间 (用于全量耗时评估) ---
    t_perm_start = tic;
    perm_auc_mat = zeros(cfg.n_perm, n_win);
    
    % 检查是否有 parpool
    p_pool = gcp('nocreate');
    if isempty(p_pool)
        parpool('local', 16);
    end
    
    svm_lambda = cfg.svm_lambda;
    parfor p = 1:cfg.n_perm
        y_perm = y_tr(randperm(n_tr));
        row_auc = zeros(1, n_win);
        for w = 1:n_win
            mdl_p = fitclinear(X3_norm_all{w}, y_perm, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', svm_lambda);
            [~, sc_p] = predict(mdl_p, X2_norm_all{w});
            pos_p = sc_p(y_te == 1, 2);
            neg_p = sc_p(y_te == 0, 2);
            row_auc(w) = (sum(sum(pos_p > neg_p')) + 0.5 * sum(sum(pos_p == neg_p'))) / (numel(pos_p) * numel(neg_p));
        end
        perm_auc_mat(p, :) = row_auc;
    end
    t_perm = toc(t_perm_start);
    t_with_perm = t_no_perm + t_perm;
    
    target_timings(t_i, :) = [t_no_perm, t_with_perm];
    
    % 计算 AUC 95% 置信上限
    auc_p95 = prctile(perm_auc_mat, 95, 1);
    
    % --- 2.6 保存时程 CSV ---
    tc_tbl = table(t_centers', dec_score_straw', dec_score_water', dec_score_cabb', dec_score_kiwi', ...
                   dec_score_mean', dec_score_diff', auc_timecourse', bal_acc', pred_red_prop', ...
                   straw_acc', water_acc', cabb_acc', kiwi_acc', ...
                   'VariableNames', {'time_ms', 'score_straw', 'score_water', 'score_cabb', 'score_kiwi', ...
                                     'score_overall_mean', 'score_diff', 'auc', 'bal_acc', 'pred_red_prop', ...
                                     'straw_acc', 'water_acc', 'cabb_acc', 'kiwi_acc'});
    csv_out = fullfile(cfg.out_tab_dir, sprintf('%s_%s_asymmetry_analysis.csv', sub_id, elec));
    writetable(tc_tbl, csv_out);
    
    % 记录汇总指标
    [max_auc, max_auc_idx] = max(smoothdata(auc_timecourse, 'gaussian', cfg.smooth_pts));
    max_auc_t = t_centers(max_auc_idx);
    [max_ba, max_ba_idx]   = max(smoothdata(bal_acc, 'gaussian', cfg.smooth_pts));
    max_ba_t = t_centers(max_ba_idx);
    
    % 判断归类情况 (按照文档标准):
    if max_auc >= 0.56 && max_ba < 0.54
        classified_case = 'Case 1: Boundary Shift (BA Low, AUC High)';
    elseif max_auc >= 0.56 && max_ba >= 0.55
        classified_case = 'Case 2: True Color Memory (BA High, AUC High)';
    elseif max_auc < 0.54 && max_ba < 0.54
        classified_case = 'Case 4: Classifier Bias / No Signal';
    else
        classified_case = 'Ambiguous / Needs Inspection';
    end
    
    results_summary(t_i, :) = {sub_id, elec, max_ba, max_ba_t, max_auc, max_auc_t, ...
                               mean(pred_red_prop), classified_case, t_with_perm};
    
    fprintf('  - %s-%s 完成! Peak BA: %.2f%% (%d ms), Peak AUC: %.3f (%d ms), 判定: %s (耗时: %.2f秒)\n', ...
        sub_id, elec, max_ba*100, max_ba_t, max_auc, max_auc_t, classified_case, t_with_perm);
    
    % --- 2.7 绘制专业三面板学术图 ---
    h_fig = figure('Visible', 'off', 'Units', 'pixels', 'Position', [100, 100, 1200, 380], 'Color', 'w');
    
    % 配色定义 (经典 Nature 风格)
    c_straw = [0.84, 0.19, 0.15]; % 深红 (Strawberry)
    c_water = [0.94, 0.50, 0.50]; % 浅珊瑚红 (Watermelon)
    c_cabb  = [0.13, 0.59, 0.45]; % 翠绿 (Cabbage)
    c_kiwi  = [0.45, 0.76, 0.46]; % 浅草绿 (Kiwi)
    c_mean  = [0.20, 0.20, 0.20]; % 炭黑加粗 (Overall Mean: 四线真正的平均值)
    c_auc   = [0.12, 0.47, 0.71]; % 经典蓝 (AUC)
    c_ba    = [0.87, 0.49, 0.00]; % 暖橙色 (Balanced Acc)
    
    % 子图 1: 连续决策值 (Decision Score: w*x + b)
    subplot(1, 3, 1);
    hold on;
    yline(0, 'k--', 'LineWidth', 1.0, 'Alpha', 0.6); % 决策分界线 (0 轴)
    xline(0, 'k:', 'LineWidth', 1.0, 'Alpha', 0.5);
    p_st = plot(t_centers, smoothdata(dec_score_straw, 'gaussian', cfg.smooth_pts), 'Color', c_straw, 'LineWidth', 1.4);
    p_wt = plot(t_centers, smoothdata(dec_score_water, 'gaussian', cfg.smooth_pts), 'Color', c_water, 'LineWidth', 1.4);
    p_cb = plot(t_centers, smoothdata(dec_score_cabb,  'gaussian', cfg.smooth_pts), 'Color', c_cabb,  'LineWidth', 1.4);
    p_kw = plot(t_centers, smoothdata(dec_score_kiwi,  'gaussian', cfg.smooth_pts), 'Color', c_kiwi,  'LineWidth', 1.4);
    % 加粗黑线为四条线真正的总平均 (必定老老实实穿行在四条线正中间)
    p_mn = plot(t_centers, smoothdata(dec_score_mean,  'gaussian', cfg.smooth_pts), 'Color', c_mean, 'LineWidth', 2.2);
    hold off;
    box off; set(gca, 'TickDir', 'out', 'FontSize', 10, 'LineWidth', 1.0);
    xlim([-200, 800]); xlabel('Time (ms)', 'FontSize', 11);
    ylabel('Decision Score (w\cdot x + b)', 'FontSize', 11);
    title(sprintf('A. Continuous Scores (%s-%s)', sub_id, elec), 'FontSize', 12, 'FontWeight', 'bold');
    legend([p_st, p_wt, p_cb, p_kw, p_mn], ...
           {'Strawberry', 'Watermelon', 'Cabbage', 'Kiwi', 'Overall Mean (4 Fruits)'}, ...
           'Location', 'best', 'FontSize', 8, 'Box', 'off');
    
    % 子图 2: AUC vs Balanced Accuracy (解耦门槛检验)
    subplot(1, 3, 2);
    hold on;
    yline(0.5, 'k--', 'LineWidth', 1.0, 'Alpha', 0.6);
    xline(0, 'k:', 'LineWidth', 1.0, 'Alpha', 0.5);
    % 填充 AUC 95% 置信上限阴影
    fill([t_centers, fliplr(t_centers)], [auc_p95, fliplr(ones(1, n_win)*0.5)], ...
         [0.85, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.45);
    p_auc_line = plot(t_centers, smoothdata(auc_timecourse, 'gaussian', cfg.smooth_pts), ...
                      'Color', c_auc, 'LineWidth', 2.2);
    p_ba_line  = plot(t_centers, smoothdata(bal_acc, 'gaussian', cfg.smooth_pts), ...
                      'Color', c_ba, 'LineWidth', 1.8, 'LineStyle', '-.');
    hold off;
    box off; set(gca, 'TickDir', 'out', 'FontSize', 10, 'LineWidth', 1.0);
    xlim([-200, 800]); ylim([0.35, 0.80]);
    xlabel('Time (ms)', 'FontSize', 11);
    ylabel('Performance Index', 'FontSize', 11);
    title('B. AUC (Threshold-Free) vs BA', 'FontSize', 12, 'FontWeight', 'bold');
    legend([p_auc_line, p_ba_line], {'AUC (Red vs Green)', 'Balanced Accuracy'}, ...
           'Location', 'best', 'FontSize', 8, 'Box', 'off');
       
    % 子图 3: 预测偏置 (Prediction Bias: 全试次预测为 Red 的比例)
    subplot(1, 3, 3);
    hold on;
    yline(0.5, 'k--', 'LineWidth', 1.0, 'Alpha', 0.6);
    xline(0, 'k:', 'LineWidth', 1.0, 'Alpha', 0.5);
    p_bias_red = plot(t_centers, smoothdata(pred_red_prop, 'gaussian', cfg.smooth_pts), ...
                      'Color', [0.8, 0.2, 0.2], 'LineWidth', 2.0);
    p_bias_grn = plot(t_centers, smoothdata(1 - pred_red_prop, 'gaussian', cfg.smooth_pts), ...
                      'Color', [0.2, 0.6, 0.3], 'LineWidth', 2.0);
    hold off;
    box off; set(gca, 'TickDir', 'out', 'FontSize', 10, 'LineWidth', 1.0);
    xlim([-200, 800]); ylim([0, 1.0]);
    xlabel('Time (ms)', 'FontSize', 11);
    ylabel('Prediction Proportion', 'FontSize', 11);
    title('C. Task 2 Prediction Bias', 'FontSize', 12, 'FontWeight', 'bold');
    legend([p_bias_red, p_bias_grn], {'Predicted Red (%)', 'Predicted Green (%)'}, ...
           'Location', 'best', 'FontSize', 8, 'Box', 'off');
    
    % 保存图片
    fig_png = fullfile(cfg.out_fig_dir, sprintf('%s_%s_asymmetry_validation.png', sub_id, elec));
    exportgraphics(h_fig, fig_png, 'Resolution', 300);
    close(h_fig);
end

%% 3. 输出耗时测算与全量预估
avg_t_no_perm   = mean(target_timings(:, 1));
avg_t_with_perm = mean(target_timings(:, 2));
n_total_elecs   = 157;

fprintf('\n========================================================================\n');
fprintf('>>> 典型通道测试完成! 耗时统计与全量运行预估:\n');
fprintf('------------------------------------------------------------------------\n');
fprintf('  单通道纯计算 (无置换检验): 平均 %.2f 秒 / 通道\n', avg_t_no_perm);
fprintf('  单通道全流程 (含 100 次置换): 平均 %.2f 秒 / 通道\n', avg_t_with_perm);
fprintf('------------------------------------------------------------------------\n');
fprintf('>>> 全量 157 个通道耗时预估:\n');
fprintf('  1. 模式 A (快速模式: 仅输出 Score/AUC/BA 曲线, 无置换): 约 %.1f 秒 (~%.1f 分钟)\n', ...
    avg_t_no_perm * n_total_elecs, (avg_t_no_perm * n_total_elecs)/60);
fprintf('  2. 模式 B (完整统计模式: 16 核并行, 含 100 次置换检验):\n');
fprintf('     - 若单通道内置换并行: 约 %.1f 分钟\n', ...
    (avg_t_with_perm * n_total_elecs)/60);
fprintf('     - 若通道级 parfor 并行 (推荐): 约 %.1f ~ %.1f 分钟\n', ...
    (avg_t_with_perm * n_total_elecs)/(60*14), (avg_t_with_perm * n_total_elecs)/(60*12));
fprintf('========================================================================\n');

% 保存汇总简表
sum_tab = cell2table(results_summary, 'VariableNames', { ...
    'subject', 'channel', 'peak_ba', 'peak_ba_time_ms', 'peak_auc', 'peak_auc_time_ms', ...
    'mean_pred_red_prop', 'diagnosis', 'elapsed_sec'});
disp(sum_tab);
writetable(sum_tab, fullfile(cfg.out_tab_dir, 'validation_target_summary.csv'));
