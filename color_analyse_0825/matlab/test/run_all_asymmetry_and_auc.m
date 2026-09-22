%% run_all_asymmetry_and_auc.m
% -------------------------------------------------------------------------
% 全量 157 个电极的跨任务不对称性、连续决策得分、AUC 及其统计置换检验分析
%
% 核心分析维度:
% 1. 连续决策值 (Decision Score: w*x + b): 包含4种水果及加粗的红绿差值线 Diff (Red - Green)
% 2. 统计置换检验 AUC: 无阈值依赖的红绿排序能力，配合 100 次置换检验与 Cluster-Mass FWE 校正
% 3. 预测偏置 (Prediction Bias): 预测为 Red vs Green 的比例时程
% 4. 自动化分型诊断 (Case 1 边界偏移, Case 2 稳定颜色记忆, Case 4 纯偏置/无信号)
% -------------------------------------------------------------------------
clear; clc; close all;

%% 1. 参数与路径直观配置
cfg = struct();
cfg.root_dir     = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825';
cfg.data_root    = fullfile(cfg.root_dir, 'process_data_new');
cfg.task_info    = fullfile(cfg.root_dir, 'task_info');
cfg.c04_table    = fullfile(cfg.root_dir, 'result', 'tables', 'color_effects_summary.mat');

cfg.out_fig_dir  = fullfile(cfg.root_dir, 'result', 'figures', 'cross_decoding_asymmetry_all');
cfg.out_tc_dir   = fullfile(cfg.root_dir, 'result', 'tables', 'cross_decoding_asymmetry_timecourses');
cfg.out_tab_dir  = fullfile(cfg.root_dir, 'result', 'tables');

if ~exist(cfg.out_fig_dir, 'dir'), mkdir(cfg.out_fig_dir); end
if ~exist(cfg.out_tc_dir, 'dir'),  mkdir(cfg.out_tc_dir);  end
if ~exist(cfg.out_tab_dir, 'dir'), mkdir(cfg.out_tab_dir); end

% 时间窗与时程参数 (20ms 滑动窗，与成熟流水线一致)
cfg.win_len      = 20;                             % ms
cfg.win_step     = 20;                             % ms
cfg.t_range      = [-200, 800];                    % ms
cfg.smooth_pts   = 5;                              % 5 点高斯平滑

% 频段与分类参数
cfg.bands        = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.n_bands      = numel(cfg.bands);
cfg.svm_lambda   = 0.01;                           % 岭正则化参数 Lambda

% 置换检验与多核并行
cfg.n_perm       = 100;                            % 置换检验次数 (用于零分布置信带与 Cluster-Mass)
cfg.n_workers    = 16;                             % 16 线程并行加速
cfg.skip_exist   = true;                           % 是否跳过已存在的结果 (true 为断点续跑)

%% 2. 加载 157 个同向显著色觉电极列表 (C04)
if ~isfile(cfg.c04_table)
    error('未找到 C04 结果表: %s', cfg.c04_table);
end
c04_data = load(cfg.c04_table);
if isfield(c04_data, 'all_tbl'), c04_tbl = c04_data.all_tbl; else, c04_tbl = c04_data.res_table; end

concord_mask = (c04_tbl.is_significant == 1) & ...
    (strcmp(c04_tbl.concordance_type, 'Concordant_Positive') | ...
     strcmp(c04_tbl.concordance_type, 'Concordant_Negative'));
c04_valid = c04_tbl(concord_mask, :);

elec_keys = strcat(c04_valid.subject, '_', c04_valid.channel);
[~, u_ia] = unique(elec_keys, 'stable');
all_subs  = c04_valid.subject(u_ia);
all_elecs = c04_valid.channel(u_ia);
n_total   = numel(all_subs);
unique_subs = unique(all_subs, 'stable');

fprintf('========================================================================\n');
fprintf('>>> 启动全量 157 个色觉电极不对称性、决策得分与 AUC 置换检验分析\n');
fprintf('>>> 目标总电极数: %d, 涉及被试数: %d\n', n_total, numel(unique_subs));
fprintf('========================================================================\n');

%% 3. 初始化并行池
if cfg.n_workers > 1
    p_pool = gcp('nocreate');
    if isempty(p_pool)
        parpool('local', cfg.n_workers);
    elseif p_pool.NumWorkers ~= cfg.n_workers
        delete(p_pool);
        parpool('local', cfg.n_workers);
    end
end

%% 4. 时间窗参数生成
t_starts  = cfg.t_range(1) : cfg.win_step : (cfg.t_range(2) - cfg.win_len);
n_win     = numel(t_starts);
t_centers = t_starts + cfg.win_len / 2;

summary_rows = cell(n_total, 13);
global_idx   = 0;
t_batch_all  = tic;

%% 5. 按被试流式加载与逐通道计算
for s_i = 1:numel(unique_subs)
    sub_id = unique_subs{s_i};
    sub_mask = strcmp(all_subs, sub_id);
    sub_elecs = all_elecs(sub_mask);
    n_sub_elecs = numel(sub_elecs);
    
    fprintf('\n------------------------------------------------------------------------\n');
    fprintf('>>> [被试 %d/%d: %s] 批量流式加载 %d 个电极 ...\n', s_i, numel(unique_subs), sub_id, n_sub_elecs);
    fprintf('------------------------------------------------------------------------\n');
    
    t3_mat = fullfile(cfg.data_root, sub_id, 'task3_multiband_epoched.mat');
    t2_mat = fullfile(cfg.data_root, sub_id, 'task2_multiband_epoched.mat');
    
    if ~isfile(t3_mat) || ~isfile(t2_mat)
        warning('被试 %s 缺少 Task 2 或 Task 3 数据，跳过。', sub_id);
        continue;
    end
    
    t_load = tic;
    
    % --- 加载 Task 3 训练数据 ---
    t_axis = double(h5read(t3_mat, '/epoched_data/time_ms'));
    t_axis = t_axis(:)';
    ch_list3_all = h5read(t3_mat, '/epoched_data/channels');
    if iscell(ch_list3_all)
        ch_list3_all = cellfun(@(x) char(x(:)'), ch_list3_all, 'UniformOutput', false);
    end
    
    d3 = load(fullfile(cfg.task_info, sub_id, 'task3_trial_info.mat'), 'trial_info');
    ti3 = d3.trial_info;
    tr_mask = strcmp(ti3.color, 'red') | strcmp(ti3.color, 'green');
    ti3_use = ti3(tr_mask, :);
    y_tr = zeros(height(ti3_use), 1);
    y_tr(strcmp(ti3_use.color, 'red')) = 1; % 1: Red, 0: Green
    n_tr = height(ti3_use);
    
    [ch_match3, ch_sub_idx3] = ismember(sub_elecs, ch_list3_all);
    valid_elecs = sub_elecs(ch_match3);
    target_idx3 = ch_sub_idx3(ch_match3);
    
    ep3_bands = struct();
    for b = 1:cfg.n_bands
        raw = h5read(t3_mat, ['/epoched_data/' cfg.bands{b}]);
        ep3_bands.(cfg.bands{b}) = raw(tr_mask, target_idx3, :);
        clear raw;
    end
    
    % --- 加载 Task 2 测试数据 ---
    ch_list2_all = h5read(t2_mat, '/epoched_data/channels');
    if iscell(ch_list2_all)
        ch_list2_all = cellfun(@(x) char(x(:)'), ch_list2_all, 'UniformOutput', false);
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
    
    [ch_match2, ch_sub_idx2] = ismember(valid_elecs, ch_list2_all);
    final_elecs = valid_elecs(ch_match2);
    target_idx2 = ch_sub_idx2(ch_match2);
    
    if ~all(ch_match2)
        for b = 1:cfg.n_bands
            ep3_bands.(cfg.bands{b}) = ep3_bands.(cfg.bands{b})(:, ch_match2, :);
        end
    end
    
    ep2_bands = struct();
    for b = 1:cfg.n_bands
        raw = h5read(t2_mat, ['/epoched_data/' cfg.bands{b}]);
        ep2_bands.(cfg.bands{b}) = raw(te_mask, target_idx2, :);
        clear raw;
    end
    
    fprintf('  [流式加载完成] 耗时: %.2f 秒\n', toc(t_load));
    
    % --- 逐通道进行连续值提取、AUC置换检验与画图 ---
    for e_i = 1:numel(final_elecs)
        elec = final_elecs{e_i};
        global_idx = global_idx + 1;
        t_elec = tic;
        
        fig_png = fullfile(cfg.out_fig_dir, sprintf('%s_%s_asymmetry_validation.png', sub_id, elec));
        csv_tc  = fullfile(cfg.out_tc_dir,  sprintf('%s_%s_asymmetry_timecourse.csv', sub_id, elec));
        
        if cfg.skip_exist && isfile(fig_png) && isfile(csv_tc)
            try
                tc_ex = readtable(csv_tc);
                [p_auc, p_auc_i] = max(smoothdata(tc_ex.auc, 'gaussian', cfg.smooth_pts));
                [p_ba, p_ba_i]   = max(smoothdata(tc_ex.bal_acc, 'gaussian', cfg.smooth_pts));
                [p_df, p_df_i]   = max(smoothdata(tc_ex.score_diff, 'gaussian', cfg.smooth_pts));
                has_sig = any(tc_ex.is_sig_auc_cluster);
                [~, n_sig_c] = bwlabel(tc_ex.is_sig_auc_cluster);
                if has_sig && p_ba < 0.54
                    diag_ex = 'Case 1: Boundary Shift (BA Low, AUC Sig)';
                elseif has_sig && p_ba >= 0.54
                    diag_ex = 'Case 2: True Color Memory (BA High, AUC Sig)';
                elseif p_auc < 0.54 && p_ba < 0.54
                    diag_ex = 'Case 4: Classifier Bias / No Signal';
                else
                    diag_ex = 'Case 3/Exploratory: Moderate Signal';
                end
                summary_rows(global_idx, :) = {sub_id, elec, p_auc, t_centers(p_auc_i), double(has_sig), n_sig_c, ...
                                               p_ba, t_centers(p_ba_i), p_df, t_centers(p_df_i), ...
                                               mean(tc_ex.pred_red_prop), diag_ex, 0};
                fprintf('  - [%d/%d] %s-%s 已存在，复用现有结果 (Peak AUC: %.3f, Sig: %d)。\n', ...
                    global_idx, n_total, sub_id, elec, p_auc, n_sig_c);
                continue;
            catch
                % 读取异常则重新计算
            end
        end
        
        % 构建本通道滑动时间窗特征
        X3_3d = zeros(n_tr, cfg.n_bands, n_win);
        X2_3d = zeros(n_te, cfg.n_bands, n_win);
        for w = 1:n_win
            w_t1 = t_starts(w);
            w_t2 = w_t1 + cfg.win_len;
            t_mask = (t_axis >= w_t1) & (t_axis < w_t2);
            for b = 1:cfg.n_bands
                X3_3d(:, b, w) = mean(ep3_bands.(cfg.bands{b})(:, e_i, t_mask), 3);
                X2_3d(:, b, w) = mean(ep2_bands.(cfg.bands{b})(:, e_i, t_mask), 3);
            end
        end
        
        % 真实解码与连续得分计算
        dec_score_straw = zeros(1, n_win);
        dec_score_water = zeros(1, n_win);
        dec_score_cabb  = zeros(1, n_win);
        dec_score_kiwi  = zeros(1, n_win);
        dec_score_mean  = zeros(1, n_win);
        dec_score_diff  = zeros(1, n_win); % Red_mean - Green_mean (按要求计算差值)
        
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
            
            mu_w  = mean(X_tr_w, 1);
            sig_w = std(X_tr_w, 0, 1);
            sig_w(sig_w < 1e-6) = 1;
            
            X_tr_norm = (X_tr_w - mu_w) ./ sig_w;
            X_te_norm = (X_te_w - mu_w) ./ sig_w;
            
            X3_norm_all{w} = X_tr_norm;
            X2_norm_all{w} = X_te_norm;
            
            mdl = fitclinear(X_tr_norm, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', cfg.svm_lambda);
            [y_pred, scores] = predict(mdl, X_te_norm);
            trial_scores = scores(:, 2); % 正值代表偏向 Red，负值代表偏向 Green
            
            % 1. 四类水果各自的连续决策得分均值
            dec_score_straw(w) = mean(trial_scores(m_straw));
            dec_score_water(w) = mean(trial_scores(m_water));
            dec_score_cabb(w)  = mean(trial_scores(m_cabb));
            dec_score_kiwi(w)  = mean(trial_scores(m_kiwi));
            dec_score_mean(w)  = (dec_score_straw(w) + dec_score_water(w) + ...
                                  dec_score_cabb(w)  + dec_score_kiwi(w)) / 4;
            % 差值 (Red - Green)
            dec_score_diff(w)  = ((dec_score_straw(w) + dec_score_water(w)) - ...
                                  (dec_score_cabb(w)  + dec_score_kiwi(w))) / 2;
            
            % 2. 连续 AUC (Mann-Whitney U 快速向量化实现，完全不受截距偏移影响)
            pos_scores = trial_scores(y_te == 1);
            neg_scores = trial_scores(y_te == 0);
            auc_timecourse(w) = (sum(sum(pos_scores > neg_scores')) + 0.5 * sum(sum(pos_scores == neg_scores'))) / ...
                                (numel(pos_scores) * numel(neg_scores));
            
            % 3. 预测偏置
            pred_red_prop(w) = mean(y_pred == 1);
            
            % 4. 品类正确率与平衡正确率
            straw_acc(w) = sum(y_pred(m_straw) == 1) / max(1, sum(m_straw));
            water_acc(w) = sum(y_pred(m_water) == 1) / max(1, sum(m_water));
            cabb_acc(w)  = sum(y_pred(m_cabb)  == 0) / max(1, sum(m_cabb));
            kiwi_acc(w)  = sum(y_pred(m_kiwi)  == 0) / max(1, sum(m_kiwi));
            bal_acc(w)   = (straw_acc(w) + water_acc(w) + cabb_acc(w) + kiwi_acc(w)) / 4;
        end
        
        % --- 置换检验: 100 次置换评估 AUC 零分布与基于聚类质量的 FWE 校正 ---
        n_perm = cfg.n_perm;
        perm_auc_mat = zeros(n_perm, n_win);
        svm_lambda = cfg.svm_lambda;
        
        parfor p = 1:n_perm
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
        
        % AUC 零分布 95% 置信上限
        auc_p95 = prctile(perm_auc_mat, 95, 1);
        
        % Cluster-Mass FWE 显著检验 (专门针对 AUC 进行统计显著性推断)
        raw_p_auc = mean(perm_auc_mat >= auc_timecourse, 1);
        auc_pt_sig = raw_p_auc < 0.05;
        
        [L_auc, num_clusters_auc] = bwlabel(auc_pt_sig);
        real_masses_auc = zeros(1, num_clusters_auc);
        for c = 1:num_clusters_auc
            real_masses_auc(c) = sum(auc_timecourse(L_auc == c) - 0.50);
        end
        
        % 计算零假设最大簇质量分布
        max_null_masses_auc = zeros(1, n_perm);
        for p = 1:n_perm
            null_pt_sig = perm_auc_mat(p, :) >= auc_p95;
            [L_null, num_null_c] = bwlabel(null_pt_sig);
            if num_null_c > 0
                null_m = zeros(1, num_null_c);
                for c = 1:num_null_c
                    null_m(c) = sum(perm_auc_mat(p, L_null == c) - 0.50);
                end
                max_null_masses_auc(p) = max(null_m);
            else
                max_null_masses_auc(p) = 0;
            end
        end
        
        % 筛选通过 FWE 校正的显著时间簇 (p_fwe < 0.05)
        sig_clusters_auc = [];
        for c = 1:num_clusters_auc
            p_cluster = mean(max_null_masses_auc >= real_masses_auc(c));
            if p_cluster < 0.05
                sig_clusters_auc(end+1) = c; %#ok<AGROW>
            end
        end
        n_sig_auc_clusters = numel(sig_clusters_auc);
        auc_has_sig = double(n_sig_auc_clusters > 0);
        
        is_sig_auc_time = ismember(L_auc, sig_clusters_auc);
        
        % --- 保存本通道 CSV 时程数据 ---
        tc_tbl = table(t_centers', dec_score_straw', dec_score_water', dec_score_cabb', dec_score_kiwi', ...
                       dec_score_mean', dec_score_diff', auc_timecourse', auc_p95', double(is_sig_auc_time'), ...
                       bal_acc', pred_red_prop', straw_acc', water_acc', cabb_acc', kiwi_acc', ...
                       'VariableNames', {'time_ms', 'score_straw', 'score_water', 'score_cabb', 'score_kiwi', ...
                                         'score_overall_mean', 'score_diff', 'auc', 'auc_p95_null', 'is_sig_auc_cluster', ...
                                         'bal_acc', 'pred_red_prop', 'straw_acc', 'water_acc', 'cabb_acc', 'kiwi_acc'});
        writetable(tc_tbl, csv_tc);
        
        % 汇总统计指标
        [peak_auc, peak_auc_idx] = max(smoothdata(auc_timecourse, 'gaussian', cfg.smooth_pts));
        peak_auc_t = t_centers(peak_auc_idx);
        [peak_ba, peak_ba_idx]   = max(smoothdata(bal_acc, 'gaussian', cfg.smooth_pts));
        peak_ba_t = t_centers(peak_ba_idx);
        [peak_diff, peak_diff_idx] = max(smoothdata(dec_score_diff, 'gaussian', cfg.smooth_pts));
        peak_diff_t = t_centers(peak_diff_idx);
        
        % 自动化分类诊断
        if auc_has_sig && peak_ba < 0.54
            diagnosis = 'Case 1: Boundary Shift (BA Low, AUC Sig)';
        elseif auc_has_sig && peak_ba >= 0.54
            diagnosis = 'Case 2: True Color Memory (BA High, AUC Sig)';
        elseif peak_auc < 0.54 && peak_ba < 0.54
            diagnosis = 'Case 4: Classifier Bias / No Signal';
        else
            diagnosis = 'Case 3/Exploratory: Moderate Signal';
        end
        
        summary_rows(global_idx, :) = {sub_id, elec, peak_auc, peak_auc_t, auc_has_sig, n_sig_auc_clusters, ...
                                       peak_ba, peak_ba_t, peak_diff, peak_diff_t, ...
                                       mean(pred_red_prop), diagnosis, toc(t_elec)};
        
        fprintf('  - [%d/%d] %s-%s 完成! Peak AUC: %.3f (%d ms, 显著簇: %d), Peak BA: %.2f%%, 判定: %s (耗时: %.2f秒)\n', ...
            global_idx, n_total, sub_id, elec, peak_auc, peak_auc_t, n_sig_auc_clusters, peak_ba*100, diagnosis, toc(t_elec));
        
        % --- 绘制三面板学术图 (符合 Nature 发表标准，严格无中文) ---
        h_fig = figure('Visible', 'off', 'Units', 'pixels', 'Position', [100, 100, 1200, 380], 'Color', 'w');
        
        c_straw = [0.84, 0.19, 0.15]; % 深红 (Strawberry)
        c_water = [0.94, 0.50, 0.50]; % 浅珊瑚红 (Watermelon)
        c_cabb  = [0.13, 0.59, 0.45]; % 翠绿 (Cabbage)
        c_kiwi  = [0.45, 0.76, 0.46]; % 浅草绿 (Kiwi)
        c_diff  = [0.15, 0.15, 0.15]; % 炭黑加粗 (Diff: Red - Green)
        c_auc   = [0.12, 0.47, 0.71]; % 经典蓝 (AUC)
        c_ba    = [0.87, 0.49, 0.00]; % 暖橙色 (Balanced Acc)
        
        % 子图 1: 连续决策值 (保留用户指定的 Diff 加粗差值线)
        subplot(1, 3, 1);
        hold on;
        yline(0, 'k--', 'LineWidth', 1.0, 'Alpha', 0.6);
        xline(0, 'k:', 'LineWidth', 1.0, 'Alpha', 0.5);
        p_st = plot(t_centers, smoothdata(dec_score_straw, 'gaussian', cfg.smooth_pts), 'Color', c_straw, 'LineWidth', 1.4);
        p_wt = plot(t_centers, smoothdata(dec_score_water, 'gaussian', cfg.smooth_pts), 'Color', c_water, 'LineWidth', 1.4);
        p_cb = plot(t_centers, smoothdata(dec_score_cabb,  'gaussian', cfg.smooth_pts), 'Color', c_cabb,  'LineWidth', 1.4);
        p_kw = plot(t_centers, smoothdata(dec_score_kiwi,  'gaussian', cfg.smooth_pts), 'Color', c_kiwi,  'LineWidth', 1.4);
        p_df = plot(t_centers, smoothdata(dec_score_diff,  'gaussian', cfg.smooth_pts), 'Color', c_diff,  'LineWidth', 2.2);
        hold off;
        box off; set(gca, 'TickDir', 'out', 'FontSize', 10, 'LineWidth', 1.0);
        xlim([-200, 800]); xlabel('Time (ms)', 'FontSize', 11);
        ylabel('Decision Score (w\cdot x + b)', 'FontSize', 11);
        title(sprintf('A. Continuous Scores (%s-%s)', sub_id, elec), 'FontSize', 12, 'FontWeight', 'bold');
        legend([p_st, p_wt, p_cb, p_kw, p_df], ...
               {'Strawberry', 'Watermelon', 'Cabbage', 'Kiwi', 'Diff (Red - Green)'}, ...
               'Location', 'best', 'FontSize', 8, 'Box', 'off');
        
        % 子图 2: AUC 置换检验显著性 vs 平衡准确率
        subplot(1, 3, 2);
        hold on;
        yline(0.5, 'k--', 'LineWidth', 1.0, 'Alpha', 0.6);
        xline(0, 'k:', 'LineWidth', 1.0, 'Alpha', 0.5);
        
        % 填充置换检验 95% 置信零分布阴影
        fill([t_centers, fliplr(t_centers)], [auc_p95, fliplr(ones(1, n_win)*0.5)], ...
             [0.85, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        
        % 绘制 AUC 显著时间簇标记条 (横向红色粗条)
        if any(is_sig_auc_time)
            sig_y = 0.38 * ones(1, sum(is_sig_auc_time));
            scatter(t_centers(is_sig_auc_time), sig_y, 25, [0.85, 0.15, 0.15], 'filled', 's');
        end
        
        p_auc_line = plot(t_centers, smoothdata(auc_timecourse, 'gaussian', cfg.smooth_pts), ...
                          'Color', c_auc, 'LineWidth', 2.2);
        p_ba_line  = plot(t_centers, smoothdata(bal_acc, 'gaussian', cfg.smooth_pts), ...
                          'Color', c_ba, 'LineWidth', 1.8, 'LineStyle', '-.');
        hold off;
        box off; set(gca, 'TickDir', 'out', 'FontSize', 10, 'LineWidth', 1.0);
        xlim([-200, 800]); ylim([0.35, 0.80]);
        xlabel('Time (ms)', 'FontSize', 11);
        ylabel('Performance Index', 'FontSize', 11);
        title('B. AUC Permutation Test (Cluster FWE) vs BA', 'FontSize', 12, 'FontWeight', 'bold');
        legend([p_auc_line, p_ba_line], {'AUC (Red vs Green)', 'Balanced Accuracy'}, ...
               'Location', 'best', 'FontSize', 8, 'Box', 'off');
           
        % 子图 3: 任务 2 预测偏置 (检验是否整体偏向某色)
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
        
        exportgraphics(h_fig, fig_png, 'Resolution', 300);
        close(h_fig);
        
        clear X3_3d X2_3d X3_norm_all X2_norm_all perm_auc_mat;
    end
    
    clear ep3_bands ep2_bands;
end

%% 6. 保存全量汇总总表
if global_idx > 0
    actual_rows = summary_rows(1:global_idx, :);
    sum_table = cell2table(actual_rows, 'VariableNames', { ...
        'subject', 'channel', 'peak_auc', 'peak_auc_time_ms', 'auc_has_sig_cluster', 'auc_n_sig_clusters', ...
        'peak_ba', 'peak_ba_time_ms', 'peak_diff', 'peak_diff_time_ms', ...
        'mean_pred_red_prop', 'diagnosis', 'elapsed_sec'});
    
    out_csv = fullfile(cfg.out_tab_dir, 'cross_decoding_asymmetry_summary.csv');
    out_mat = fullfile(cfg.out_tab_dir, 'cross_decoding_asymmetry_summary.mat');
    
    writetable(sum_table, out_csv);
    save(out_mat, 'sum_table', 'cfg');
    
    total_time_min = toc(t_batch_all) / 60;
    
    fprintf('\n========================================================================\n');
    fprintf('>>> 全量 157 个电极批处理全部完成!\n');
    fprintf('>>> 总耗时: %.2f 分钟 (平均 %.2f 秒 / 通道)\n', total_time_min, toc(t_batch_all) / global_idx);
    fprintf('>>> 汇总 CSV: %s\n', out_csv);
    fprintf('>>> 汇总 MAT: %s\n', out_mat);
    fprintf('>>> 图表目录: %s\n', cfg.out_fig_dir);
    fprintf('========================================================================\n');
end
