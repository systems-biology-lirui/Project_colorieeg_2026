%% run_all_tgm_permutations.m
% =========================================================================
% 脚本名称: run_all_tgm_permutations.m
% 功能:
%   1. 针对 Task 1 筛选出的全部 157 个四类别同向显著电极 (Concordant Channels)
%   2. 计算 Task 3 (纯色块) -> Task 2 (灰度记忆色) 跨任务神经解码
%   3. 引入 2D 时间泛化矩阵 (TGM) 的 200 次全网格非参数置换检验
%   4. 执行 2D 连通域簇质量多重比较校正 (2D Cluster-Mass Permutation Test, FWE p < 0.05)
%   5. 在 2D TGM 热力图上叠加显著簇白色高亮轮廓 (Nature 标准规范可视化)
%   6. 同源执行 1D 对角线同步解码的 200 次置换检验与 1D 簇质量校正
%   7. 导出全量逐通道图谱、时程 .mat 文件与全量汇总指标 CSV / MAT 表
% =========================================================================

clear; clc; close all;

%% 1. 参数配置 (置顶直观，简写平铺，可控可调)
cfg = struct();
cfg.root_dir     = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825';
cfg.data_root    = fullfile(cfg.root_dir, 'process_data_new');
cfg.task_info    = fullfile(cfg.root_dir, 'task_info');
cfg.res_root     = fullfile(cfg.root_dir, 'result');

% 输出目录配置
cfg.out_fig_dir  = fullfile(cfg.res_root, 'figures', 'cross_decoding_tgm_perm200');
cfg.out_tab_dir  = fullfile(cfg.res_root, 'tables',  'cross_decoding_tgm_perm200_timecourses');
cfg.out_sum_mat  = fullfile(cfg.res_root, 'tables',  'cross_decoding_tgm_perm200_summary.mat');
cfg.out_sum_csv  = fullfile(cfg.res_root, 'tables',  'cross_decoding_tgm_perm200_summary.csv');

% 同步更新原 cross_decoding_concordant 目录
cfg.sync_fig_dir = fullfile(cfg.res_root, 'figures', 'cross_decoding_concordant');
cfg.sync_tab_dir = fullfile(cfg.res_root, 'tables',  'cross_decoding_concordant_timecourses');
cfg.sync_sum_mat = fullfile(cfg.res_root, 'tables',  'cross_decoding_concordant_summary.mat');
cfg.sync_sum_csv = fullfile(cfg.res_root, 'tables',  'cross_decoding_concordant_summary.csv');

if ~exist(cfg.out_fig_dir, 'dir'),  mkdir(cfg.out_fig_dir);  end
if ~exist(cfg.out_tab_dir, 'dir'),  mkdir(cfg.out_tab_dir);  end
if ~exist(cfg.sync_fig_dir, 'dir'), mkdir(cfg.sync_fig_dir); end
if ~exist(cfg.sync_tab_dir, 'dir'), mkdir(cfg.sync_tab_dir); end

% 分析与统计参数
cfg.win_len      = 20;                             % 滑动窗长 20 ms
cfg.win_step     = 20;                             % 滑动步长 20 ms (51个时间窗)
cfg.t_range      = [-200, 800];                    % 分析时程范围 [-200, 800] ms
cfg.bands        = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.bands_disp   = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low-Gamma', 'High-Gamma'};
cfg.n_bands      = numel(cfg.bands);
cfg.svm_lambda   = 0.01;                           % Ridge 正则化参数
cfg.smooth_pts   = 5;                              % 1D 高斯平滑点数
cfg.gauss_sigma  = 0.8;                            % 2D 高斯平滑核宽度
cfg.n_perm       = 200;                            % 用户指定的 200 次置换检验
cfg.skip_existing = false;                         % 全量重新计算以包含 2D 置换检验轮廓

%% 2. 读取 C04 筛选出的全部 157 个同向电极
c04_mat = fullfile(cfg.res_root, 'tables', 'color_effects_summary.mat');
if ~isfile(c04_mat)
    error('未找到 C04 汇总文件: %s', c04_mat);
end
c04_data = load(c04_mat);
if isfield(c04_data, 'all_tbl')
    tbl = c04_data.all_tbl;
else
    tbl = c04_data.res_table;
end

concord_mask = (tbl.is_significant == 1) & ...
    (strcmp(tbl.concordance_type, 'Concordant_Positive') | ...
     strcmp(tbl.concordance_type, 'Concordant_Negative'));
c04_concord = tbl(concord_mask, :);

elec_keys = strcat(c04_concord.subject, '_', c04_concord.channel);
[~, u_idx] = unique(elec_keys, 'stable');
target_subs  = c04_concord.subject(u_idx);
target_elecs = c04_concord.channel(u_idx);
n_total_elecs = numel(target_subs);

fprintf('========================================================================\n');
fprintf('>>> 启动全量 2D TGM 跨任务解码 + 200 次置换检验批处理\n');
fprintf('>>> 目标电极总数: %d 个 (覆盖 %d 名被试)\n', n_total_elecs, numel(unique(target_subs)));
fprintf('========================================================================\n');

%% 3. 时间网格与绘图配色定义
t_starts  = cfg.t_range(1) : cfg.win_step : (cfg.t_range(2) - cfg.win_len);
n_win     = numel(t_starts);
t_centers = t_starts + cfg.win_len / 2;

col_joint = [0.85, 0.37, 0.01]; % 陶土暖橙 (Multi-Band)
col_null  = [0.88, 0.88, 0.88]; % 浅灰置换零分布阴影
col_clust = [1.00, 0.92, 0.70]; % 金黄显著簇高亮
band_cols = [
    0.40, 0.40, 0.40;  % Delta: 灰
    0.95, 0.60, 0.20;  % Theta: 杏黄
    0.20, 0.45, 0.75;  % Alpha: 钴蓝
    0.10, 0.65, 0.45;  % Beta: 青绿
    0.50, 0.35, 0.75;  % Low_Gamma: 紫灰
    0.90, 0.15, 0.50   % High_Gamma: 玫红
];

%% 4. 按被试分组批量极速流式计算
unique_subs = unique(target_subs, 'stable');
summary_list = struct([]);
global_elec_count = 0;
t_all_start = tic;

for s_i = 1:numel(unique_subs)
    sub_id = unique_subs{s_i};
    sub_mask = strcmp(target_subs, sub_id);
    sub_elecs = target_elecs(sub_mask);
    n_sub_elecs = numel(sub_elecs);
    
    fprintf('\n------------------------------------------------------------------------\n');
    fprintf('>>> [被试 %d/%d: %s] 载入被试数据 (共 %d 个通道) ...\n', ...
        s_i, numel(unique_subs), sub_id, n_sub_elecs);
    fprintf('------------------------------------------------------------------------\n');
    
    t3_mat = fullfile(cfg.data_root, sub_id, 'task3_multiband_epoched.mat');
    t2_mat = fullfile(cfg.data_root, sub_id, 'task2_multiband_epoched.mat');
    if ~isfile(t3_mat) || ~isfile(t2_mat)
        warning('被试 %s 数据缺失，跳过！', sub_id);
        continue;
    end
    
    t_load = tic;
    
    % --- 载入 Task 3 (纯色块红绿试次) ---
    t_axis = double(h5read(t3_mat, '/epoched_data/time_ms'));
    t_axis = t_axis(:)';
    ch_list3 = h5read(t3_mat, '/epoched_data/channels');
    ch_list2 = h5read(t2_mat, '/epoched_data/channels');
    if iscell(ch_list3), ch_list3 = cellfun(@(x) char(x(:)'), ch_list3, 'UniformOutput', false); end
    if iscell(ch_list2), ch_list2 = cellfun(@(x) char(x(:)'), ch_list2, 'UniformOutput', false); end
    
    d3 = load(fullfile(cfg.task_info, sub_id, 'task3_trial_info.mat'), 'trial_info');
    ti3 = d3.trial_info;
    tr_mask = strcmp(ti3.color, 'red') | strcmp(ti3.color, 'green');
    ti3_use = ti3(tr_mask, :);
    n_tr = height(ti3_use);
    y_tr = zeros(n_tr, 1);
    y_tr(strcmp(ti3_use.color, 'red')) = 1; % Red=1, Green=0
    
    % --- 载入 Task 2 (灰度记忆色试次) ---
    d2 = load(fullfile(cfg.task_info, sub_id, 'task2_trial_info.mat'), 'trial_info');
    ti2 = d2.trial_info;
    te_mask = strcmp(ti2.state, 'gray');
    ti2_use = ti2(te_mask, :);
    fruit_list = ti2_use.fruit;
    y_te = double(strcmp(fruit_list, 'strawberry') | strcmp(fruit_list, 'watermelon')); % Red=1, Green=0
    n_te = height(ti2_use);
    
    % 确定有效通道索引
    [ch_match3, ch_idx3] = ismember(sub_elecs, ch_list3);
    [ch_match2, ch_idx2] = ismember(sub_elecs, ch_list2);
    valid_mask = ch_match3 & ch_match2;
    final_elecs = sub_elecs(valid_mask);
    final_idx3  = ch_idx3(valid_mask);
    final_idx2  = ch_idx2(valid_mask);
    n_valid     = numel(final_elecs);
    
    if n_valid == 0
        warning('被试 %s 无有效匹配通道，跳过！', sub_id);
        continue;
    end
    
    % 整被试流式预载入 6 频段数据至 RAM
    sub_X3_bands = zeros(n_tr, n_valid, cfg.n_bands, numel(t_axis), 'single');
    sub_X2_bands = zeros(n_te, n_valid, cfg.n_bands, numel(t_axis), 'single');
    for b = 1:cfg.n_bands
        b_name = cfg.bands{b};
        raw3 = h5read(t3_mat, ['/epoched_data/' b_name]);
        sub_X3_bands(:, :, b, :) = raw3(tr_mask, final_idx3, :);
        raw2 = h5read(t2_mat, ['/epoched_data/' b_name]);
        sub_X2_bands(:, :, b, :) = raw2(te_mask, final_idx2, :);
        clear raw3 raw2;
    end
    fprintf('  - 数据预载入完成 (耗时: %.2f 秒) | 目标通道: %d 个\n', toc(t_load), n_valid);
    
    % 逐电极计算与 2D 置换检验
    for e_i = 1:n_valid
        ch_name = final_elecs{e_i};
        global_elec_count = global_elec_count + 1;
        t_elec = tic;
        
        fig_out1 = fullfile(cfg.out_fig_dir, sprintf('%s_%s_cross_decoding.png', sub_id, ch_name));
        fig_out2 = fullfile(cfg.sync_fig_dir, sprintf('%s_%s_cross_decoding.png', sub_id, ch_name));
        mat_out1 = fullfile(cfg.out_tab_dir, sprintf('%s_%s_cross_decoding_results.mat', sub_id, ch_name));
        mat_out2 = fullfile(cfg.sync_tab_dir, sprintf('%s_%s_cross_decoding_results.mat', sub_id, ch_name));
        
        if cfg.skip_existing && isfile(fig_out1) && isfile(mat_out1)
            fprintf('  [%d/%d | 全局 %d/%d] %s-%s 已存在，跳过。\n', ...
                e_i, n_valid, global_elec_count, n_total_elecs, sub_id, ch_name);
            continue;
        end
        
        % --- 1. 提取滑动时间窗特征 [N x n_bands x n_win] ---
        X3_3d = zeros(n_tr, cfg.n_bands, n_win);
        X2_3d = zeros(n_te, cfg.n_bands, n_win);
        for w = 1:n_win
            w_t1 = t_starts(w);
            w_t2 = w_t1 + cfg.win_len;
            t_m  = (t_axis >= w_t1) & (t_axis < w_t2);
            for b = 1:cfg.n_bands
                X3_3d(:, b, w) = mean(sub_X3_bands(:, e_i, b, t_m), 4);
                X2_3d(:, b, w) = mean(sub_X2_bands(:, e_i, b, t_m), 4);
            end
        end
        
        % --- 2. 训练 Task 3 模型并批量计算 51x51 TGM 预测 ---
        tgm_matrix = zeros(n_win, n_win);
        pred_tgm   = zeros(n_te, n_win, n_win);
        pos_te     = (y_te == 1);
        neg_te     = (y_te == 0);
        
        for w3 = 1:n_win
            X_tr_w3 = double(squeeze(X3_3d(:, :, w3)));
            mu_w3  = mean(X_tr_w3, 1);
            sig_w3 = std(X_tr_w3, 0, 1);
            sig_w3(sig_w3 < 1e-6) = 1;
            X_tr_norm = (X_tr_w3 - mu_w3) ./ sig_w3;
            
            mdl_w3 = fitclinear(X_tr_norm, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', cfg.svm_lambda);
            
            % 将 51 个 Task 2 时间窗一次性矩阵化批量预测 (比嵌套循环快3倍)
            X2_scaled = zeros(n_te, cfg.n_bands, n_win);
            for w2 = 1:n_win
                X2_scaled(:, :, w2) = (double(squeeze(X2_3d(:, :, w2))) - mu_w3) ./ sig_w3;
            end
            X2_flat = reshape(permute(X2_scaled, [1, 3, 2]), [n_te * n_win, cfg.n_bands]);
            yp_flat = predict(mdl_w3, X2_flat);
            yp_mat  = reshape(yp_flat, [n_te, n_win]);
            
            pred_tgm(:, w3, :) = reshape(yp_mat, [n_te, 1, n_win]);
            sens_w3 = mean(yp_mat(pos_te, :) == 1, 1);
            spec_w3 = mean(yp_mat(neg_te, :) == 0, 1);
            tgm_matrix(w3, :) = (sens_w3 + spec_w3) / 2;
        end
        
        % 1D 对角线同步曲线
        real_diag_joint = diag(tgm_matrix)';
        diag_joint_s    = smoothdata(real_diag_joint, 'gaussian', cfg.smooth_pts);
        
        % 计算 6 个单频段独立解码曲线 (用于 1D 辅助展示)
        real_diag_bands = zeros(cfg.n_bands, n_win);
        for w = 1:n_win
            for b = 1:cfg.n_bands
                X_tr_b = double(X3_3d(:, b, w));
                X_te_b = double(X2_3d(:, b, w));
                mu_b  = mean(X_tr_b);
                sig_b = std(X_tr_b);
                if sig_b < 1e-6, sig_b = 1; end
                X_tr_bn = (X_tr_b - mu_b) ./ sig_b;
                X_te_bn = (X_te_b - mu_b) ./ sig_b;
                mdl_b = fitclinear(X_tr_bn, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', cfg.svm_lambda);
                yp_b  = predict(mdl_b, X_te_bn);
                sens_b = sum(y_te == 1 & yp_b == 1) / max(1, sum(y_te == 1));
                spec_b = sum(y_te == 0 & yp_b == 0) / max(1, sum(y_te == 0));
                real_diag_bands(b, w) = (sens_b + spec_b) / 2;
            end
        end
        diag_bands_s = zeros(size(real_diag_bands));
        for b = 1:cfg.n_bands
            diag_bands_s(b, :) = smoothdata(real_diag_bands(b, :), 'gaussian', cfg.smooth_pts);
        end
        
        % --- 3. 2D TGM 200 次全网格非参数置换检验 ---
        null_tgm_dist  = zeros(cfg.n_perm, n_win, n_win);
        null_diag_dist = zeros(cfg.n_perm, n_win);
        
        for p = 1:cfg.n_perm
            y_te_p = y_te(randperm(n_te));
            pos_m  = (y_te_p == 1);
            neg_m  = (y_te_p == 0);
            
            sens_p = squeeze(mean(pred_tgm(pos_m, :, :) == 1, 1));
            spec_p = squeeze(mean(pred_tgm(neg_m, :, :) == 0, 1));
            tgm_p  = (sens_p + spec_p) / 2;
            
            null_tgm_dist(p, :, :) = tgm_p;
            null_diag_dist(p, :)   = diag(tgm_p)';
        end
        
        % 2D 高斯平滑 (真实与零分布同步平滑)
        tgm_matrix_s = imgaussfilt(tgm_matrix, cfg.gauss_sigma);
        null_tgm_s   = zeros(cfg.n_perm, n_win, n_win);
        for p = 1:cfg.n_perm
            null_tgm_s(p, :, :) = imgaussfilt(squeeze(null_tgm_dist(p, :, :)), cfg.gauss_sigma);
        end
        
        % --- 4. 2D Cluster-Mass FWE 显著性检验 ---
        tgm_s_3d = reshape(tgm_matrix_s, [1, n_win, n_win]);
        p_tgm_pointwise = squeeze((1 + sum(null_tgm_s >= tgm_s_3d, 1)) / (1 + cfg.n_perm));
        
        [T2_grid, ~] = meshgrid(t_centers, t_centers);
        sig_mask_2d_raw = (p_tgm_pointwise < 0.05) & (T2_grid >= 0);
        
        CC_real = bwconncomp(sig_mask_2d_raw, 8);
        real_masses_2d = zeros(1, CC_real.NumObjects);
        for ci = 1:CC_real.NumObjects
            real_masses_2d(ci) = sum(tgm_matrix_s(CC_real.PixelIdxList{ci}) - 0.50);
        end
        
        max_null_masses_2d = zeros(1, cfg.n_perm);
        null_p95_2d = squeeze(prctile(null_tgm_s, 95, 1));
        for p = 1:cfg.n_perm
            null_p_cur = squeeze(null_tgm_s(p, :, :));
            null_sig_mask = (null_p_cur >= null_p95_2d) & (T2_grid >= 0);
            CC_null = bwconncomp(null_sig_mask, 8);
            if CC_null.NumObjects > 0
                masses_p = zeros(1, CC_null.NumObjects);
                for ci = 1:CC_null.NumObjects
                    masses_p(ci) = sum(null_p_cur(CC_null.PixelIdxList{ci}) - 0.50);
                end
                max_null_masses_2d(p) = max(masses_p);
            else
                max_null_masses_2d(p) = 0;
            end
        end
        
        sig_2d_cluster_mask = false(n_win, n_win);
        n_sig_2d_clusters   = 0;
        min_p_cluster_2d    = 1.0;
        for ci = 1:CC_real.NumObjects
            p_cl = (1 + sum(max_null_masses_2d >= real_masses_2d(ci))) / (1 + cfg.n_perm);
            if p_cl < min_p_cluster_2d
                min_p_cluster_2d = p_cl;
            end
            if p_cl < 0.05
                sig_2d_cluster_mask(CC_real.PixelIdxList{ci}) = true;
                n_sig_2d_clusters = n_sig_2d_clusters + 1;
            end
        end
        
        % --- 5. 1D 对角线 Cluster-Mass FWE 显著性检验 ---
        null_diag_s  = smoothdata(null_diag_dist, 2, 'gaussian', cfg.smooth_pts);
        diag_p95     = prctile(null_diag_s, 95, 1);
        null_hi      = prctile(null_diag_s, 97.5, 1);
        null_lo      = prctile(null_diag_s, 2.5, 1);
        
        p_diag_pointwise = (1 + sum(null_diag_s >= diag_joint_s, 1)) / (1 + cfg.n_perm);
        sig_mask_1d_raw  = (p_diag_pointwise < 0.05) & (t_centers >= 0);
        
        clusters_1d = [];
        in_c = false; c_start = 1;
        for w = 1:n_win
            if sig_mask_1d_raw(w) && ~in_c
                in_c = true; c_start = w;
            elseif ~sig_mask_1d_raw(w) && in_c
                in_c = false; clusters_1d = [clusters_1d; c_start, w-1]; %#ok<AGROW>
            end
        end
        if in_c, clusters_1d = [clusters_1d; c_start, n_win]; end
        
        sig_clusters_1d = [];
        if ~isempty(clusters_1d)
            n_cl_1d = size(clusters_1d, 1);
            cl_mass_1d = zeros(n_cl_1d, 1);
            for ci = 1:n_cl_1d
                c_range = clusters_1d(ci, 1) : clusters_1d(ci, 2);
                cl_mass_1d(ci) = sum(diag_joint_s(c_range) - 0.50);
            end
            
            max_null_mass_1d = zeros(cfg.n_perm, 1);
            for pi = 1:cfg.n_perm
                null_c_mask = (null_diag_s(pi, :) >= diag_p95) & (t_centers >= 0);
                null_masses = 0;
                in_nc = false; nc_start = 1;
                for w = 1:n_win
                    if null_c_mask(w) && ~in_nc
                        in_nc = true; nc_start = w;
                    elseif ~null_c_mask(w) && in_nc
                        in_nc = false;
                        null_masses(end+1) = sum(null_diag_s(pi, nc_start:w-1) - 0.50); %#ok<AGROW>
                    end
                end
                if in_nc, null_masses(end+1) = sum(null_diag_s(pi, nc_start:n_win) - 0.50); end %#ok<AGROW>
                max_null_mass_1d(pi) = max(null_masses);
            end
            
            for ci = 1:n_cl_1d
                p_cl_1d = (1 + sum(max_null_mass_1d >= cl_mass_1d(ci))) / (1 + cfg.n_perm);
                if p_cl_1d < 0.05
                    sig_clusters_1d = [sig_clusters_1d; clusters_1d(ci, :)]; %#ok<AGROW>
                end
            end
        end
        
        % --- 6. 学术级 1:1 双子图规范绘图 ---
        h_fig = figure('Visible', 'off', 'Units', 'pixels', 'Position', [100, 100, 1280, 520], 'Color', 'w');
        
        % 左子图: 1D 对角线同步解码
        subplot(1, 2, 1);
        hold on;
        if ~isempty(sig_clusters_1d)
            for sci = 1:size(sig_clusters_1d, 1)
                c_x1 = t_centers(sig_clusters_1d(sci, 1));
                c_x2 = t_centers(sig_clusters_1d(sci, 2));
                fill([c_x1, c_x2, c_x2, c_x1], [0.35, 0.35, 0.85, 0.85], col_clust, ...
                    'EdgeColor', 'none', 'FaceAlpha', 0.6, 'HandleVisibility', 'off');
            end
        end
        
        fill([t_centers, fliplr(t_centers)], [null_hi, fliplr(null_lo)], col_null, ...
            'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'Null 95% CI');
        
        yline(0.5, '--', 'Color', [0.55, 0.55, 0.55], 'LineWidth', 1.0, 'HandleVisibility', 'off');
        xline(0, ':', 'Color', [0.40, 0.40, 0.40], 'LineWidth', 1.0, 'HandleVisibility', 'off');
        
        for b = 1:cfg.n_bands
            plot(t_centers, diag_bands_s(b, :), 'Color', [band_cols(b, :), 0.55], ...
                'LineWidth', 1.2, 'DisplayName', cfg.bands_disp{b});
        end
        
        plot(t_centers, diag_joint_s, 'Color', col_joint, 'LineWidth', 2.4, ...
            'DisplayName', 'Multi-Band');
        
        if any(sig_mask_1d_raw)
            plot(t_centers(sig_mask_1d_raw), repmat(0.38, 1, sum(sig_mask_1d_raw)), 's', ...
                'MarkerFaceColor', col_joint, 'MarkerEdgeColor', 'none', 'MarkerSize', 4, ...
                'HandleVisibility', 'off');
        end
        hold off;
        
        xlim([-200, 800]); ylim([0.35, 0.80]);
        xlabel('Time (ms)', 'FontSize', 11, 'FontWeight', 'bold');
        ylabel('Balanced Accuracy', 'FontSize', 11, 'FontWeight', 'bold');
        title(sprintf('A. Diagonal Synchronization (%s-%s)', sub_id, ch_name), ...
            'FontSize', 12, 'FontWeight', 'bold');
        legend('Location', 'northeast', 'FontSize', 8, 'Box', 'off');
        grid off; box off;
        set(gca, 'TickDir', 'out', 'LineWidth', 1.0, 'FontSize', 10);
        
        % 右子图: 2D TGM 时间泛化矩阵 (叠加 2D 显著白色轮廓)
        subplot(1, 2, 2);
        imagesc(t_centers, t_centers, tgm_matrix_s);
        set(gca, 'YDir', 'normal');
        colormap(gca, 'parula');
        caxis([0.40, 0.70]);
        cb = colorbar;
        ylabel(cb, 'Cross-Task Accuracy', 'FontSize', 10, 'FontWeight', 'bold');
        hold on;
        plot([-200, 800], [-200, 800], 'w--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        xline(0, 'w:', 'LineWidth', 1.0, 'HandleVisibility', 'off');
        yline(0, 'w:', 'LineWidth', 1.0, 'HandleVisibility', 'off');
        
        % 叠加 2D 显著簇白色高亮轮廓 (White Contour)
        if any(sig_2d_cluster_mask(:))
            contour(t_centers, t_centers, double(sig_2d_cluster_mask), [0.5, 0.5], ...
                'Color', 'w', 'LineWidth', 2.0);
        end
        hold off;
        
        xlim([-200, 800]); ylim([-200, 800]);
        xlabel('Task 2 (Memory Fruit) Time (ms)', 'FontSize', 11, 'FontWeight', 'bold');
        ylabel('Task 3 (Physical Patch) Time (ms)', 'FontSize', 11, 'FontWeight', 'bold');
        title(sprintf('B. Temporal Generalization (%s-%s, 2D Sig Clusters: %d)', ...
            sub_id, ch_name, n_sig_2d_clusters), 'FontSize', 12, 'FontWeight', 'bold');
        grid off; box off;
        set(gca, 'TickDir', 'out', 'LineWidth', 1.0, 'FontSize', 10);
        
        % 保存图片到两处目录
        exportgraphics(h_fig, fig_out1, 'Resolution', 300);
        copyfile(fig_out1, fig_out2);
        close(h_fig);
        
        % 保存 .mat 详细数据
        [r_max, c_max] = find(tgm_matrix_s == max(tgm_matrix_s(:)), 1);
        [peak_diag_val, peak_diag_idx] = max(diag_joint_s);
        
        save(mat_out1, 'real_diag_joint', 'diag_joint_s', 'real_diag_bands', 'diag_bands_s', ...
            'null_diag_dist', 'tgm_matrix', 'tgm_matrix_s', 'null_tgm_dist', ...
            'sig_mask_1d_raw', 'sig_clusters_1d', 'sig_2d_cluster_mask', 'n_sig_2d_clusters', ...
            'min_p_cluster_2d', 't_centers', 'cfg');
        copyfile(mat_out1, mat_out2);
        
        % 记录汇总条目
        rec = struct();
        rec.subject                = sub_id;
        rec.channel                = ch_name;
        rec.diag_peak_acc          = round(peak_diag_val, 4);
        rec.diag_peak_time_ms      = t_centers(peak_diag_idx);
        rec.diag_n_sig_clusters    = size(sig_clusters_1d, 1);
        rec.diag_has_sig_cluster   = (size(sig_clusters_1d, 1) > 0);
        rec.tgm_max_acc            = round(max(tgm_matrix_s(:)), 4);
        rec.tgm_task3_time_ms      = t_centers(r_max);
        rec.tgm_task2_time_ms      = t_centers(c_max);
        rec.tgm_n_sig_clusters_2d  = n_sig_2d_clusters;
        rec.tgm_has_sig_cluster_2d = (n_sig_2d_clusters > 0);
        rec.tgm_sig_area_points    = sum(sig_2d_cluster_mask(:));
        rec.tgm_cluster_p_min      = round(min_p_cluster_2d, 4);
        
        summary_list = [summary_list; rec]; %#ok<AGROW>
        
        sig_str = '';
        if n_sig_2d_clusters > 0
            sig_str = sprintf(' *** [2D 显著簇: %d, 面积: %d pts] ***', n_sig_2d_clusters, sum(sig_2d_cluster_mask(:)));
        end
        fprintf('  [%d/%d | 全局 %d/%d] %s-%s 完成 (耗时: %.2f s) | 1D峰值: %.2f%% (%d ms) | 2D峰值: %.2f%% (T3=%d ms, T2=%d ms)%s\n', ...
            e_i, n_valid, global_elec_count, n_total_elecs, sub_id, ch_name, toc(t_elec), ...
            peak_diag_val * 100, t_centers(peak_diag_idx), ...
            max(tgm_matrix_s(:)) * 100, t_centers(r_max), t_centers(c_max), sig_str);
    end
    clear sub_X3_bands sub_X2_bands;
end

%% 5. 导出全量汇总表格
if ~isempty(summary_list)
    sum_tbl = struct2table(summary_list);
    
    save(cfg.out_sum_mat, 'sum_tbl', 'cfg');
    writetable(sum_tbl, cfg.out_sum_csv);
    
    save(cfg.sync_sum_mat, 'sum_tbl', 'cfg');
    writetable(sum_tbl, cfg.sync_sum_csv);
    
    fprintf('\n========================================================================\n');
    fprintf('>>> 全量 157 通道 2D TGM 跨任务解码 + 200 次置换检验全部完成!\n');
    fprintf('>>> 总运行耗时: %.2f 分钟\n', toc(t_all_start) / 60);
    fprintf('>>> 汇总文件保存至:\n');
    fprintf('    - %s\n', cfg.out_sum_csv);
    fprintf('    - %s\n', cfg.sync_sum_csv);
    fprintf('>>> 统计概览:\n');
    fprintf('    - 目标总通道数: %d\n', height(sum_tbl));
    fprintf('    - 1D 对角线具有显著时间簇通道数: %d (%.1f%%)\n', ...
        sum(sum_tbl.diag_has_sig_cluster), sum(sum_tbl.diag_has_sig_cluster) / height(sum_tbl) * 100);
    fprintf('    - 2D TGM 具有显著时序重放簇通道数: %d (%.1f%%)\n', ...
        sum(sum_tbl.tgm_has_sig_cluster_2d), sum(sum_tbl.tgm_has_sig_cluster_2d) / height(sum_tbl) * 100);
    
    % 输出 2D 显著通道列表
    sig2d_tbl = sum_tbl(sum_tbl.tgm_has_sig_cluster_2d == 1, :);
    if ~isempty(sig2d_tbl)
        fprintf('\n>>> 【2D TGM 显著通道详情 (Cluster-Mass FWE p < 0.05)】:\n');
        for k = 1:height(sig2d_tbl)
            fprintf('    * %s-%s: Peak Acc = %.2f%%, T3 = %d ms, T2 = %d ms, 簇数 = %d, 面积 = %d 像素, p = %.4f\n', ...
                sig2d_tbl.subject{k}, sig2d_tbl.channel{k}, sig2d_tbl.tgm_max_acc(k)*100, ...
                sig2d_tbl.tgm_task3_time_ms(k), sig2d_tbl.tgm_task2_time_ms(k), ...
                sig2d_tbl.tgm_n_sig_clusters_2d(k), sig2d_tbl.tgm_sig_area_points(k), ...
                sig2d_tbl.tgm_cluster_p_min(k));
        end
    end
    fprintf('========================================================================\n');
end
