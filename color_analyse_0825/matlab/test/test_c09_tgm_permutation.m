%% test_c09_tgm_permutation.m
% -------------------------------------------------------------------------
% 验证 2D 时间泛化矩阵 (TGM) 加上 200 次统计置换检验 (2D Cluster-Mass FWE)
%
% 核心升级点:
% 1. 2D TGM 全网格 200 次非参数置换检验 (Null 51x51 TGM 分布)
% 2. 2D 连通域簇质量校正 (2D Cluster-Mass Permutation Test, FWE p < 0.05)
% 3. 在 2D TGM 热力图上叠加显著簇轮廓线 (白色高亮轮廓, Nature 顶级发表标准)
% 4. 左图 1D 对角线同步解码与右图 2D TGM 共享同源置换检验零分布
% -------------------------------------------------------------------------
clear; clc; close all;

%% 1. 参数配置
cfg = struct();
cfg.root_dir     = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825';
cfg.data_root    = fullfile(cfg.root_dir, 'process_data_new');
cfg.task_info    = fullfile(cfg.root_dir, 'task_info');
cfg.out_fig_dir  = fullfile(cfg.root_dir, 'result', 'figures', 'tgm_permutation_test');
cfg.out_tab_dir  = fullfile(cfg.root_dir, 'result', 'tables', 'tgm_permutation_test');

if ~exist(cfg.out_fig_dir, 'dir'), mkdir(cfg.out_fig_dir); end
if ~exist(cfg.out_tab_dir, 'dir'), mkdir(cfg.out_tab_dir); end

% 时间窗与频段
cfg.win_len      = 20;                             % ms
cfg.win_step     = 20;                             % ms
cfg.t_range      = [-200, 800];                    % ms (51个时间窗)
cfg.bands        = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.bands_disp   = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low-Gamma', 'High-Gamma'};
cfg.n_bands      = numel(cfg.bands);
cfg.svm_lambda   = 0.01;
cfg.smooth_pts   = 5;                              % 1D 平滑
cfg.n_perm       = 200;                            % 用户指定的 200 次置换检验

% 测试两个典型电极:
% 1. sub008-C10 (显著真记忆色位点)
% 2. sub007-C4 (纯偏置对照位点)
test_targets = {
    'sub008', 'C10';
    'sub007', 'C4'
};
n_targets = size(test_targets, 1);

fprintf('========================================================================\n');
fprintf('>>> 启动 2D TGM 时间泛化 200 次置换检验测试 (共 %d 个典型通道)\n', n_targets);
fprintf('========================================================================\n');

%% 2. 逐通道计算与绘图
t_starts  = cfg.t_range(1) : cfg.win_step : (cfg.t_range(2) - cfg.win_len);
n_win     = numel(t_starts);
t_centers = t_starts + cfg.win_len / 2;

for t_i = 1:n_targets
    sub_id = test_targets{t_i, 1};
    elec   = test_targets{t_i, 2};
    fprintf('\n>>> [%d/%d] 正在分析: %s-%s ...\n', t_i, n_targets, sub_id, elec);
    t_start = tic;
    
    % --- 加载数据 ---
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
    y_te = double(strcmp(fruit_list, 'strawberry') | strcmp(fruit_list, 'watermelon')); % 1: Red, 0: Green
    n_te = height(ti2_use);
    
    X3_bands = zeros(n_tr, cfg.n_bands, numel(t_axis));
    X2_bands = zeros(n_te, cfg.n_bands, numel(t_axis));
    for b = 1:cfg.n_bands
        raw3 = h5read(t3_mat, ['/epoched_data/' cfg.bands{b}]);
        X3_bands(:, b, :) = raw3(tr_mask, e_idx3, :);
        raw2 = h5read(t2_mat, ['/epoched_data/' cfg.bands{b}]);
        X2_bands(:, b, :) = raw2(te_mask, e_idx2, :);
        clear raw3 raw2;
    end
    
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
    
    % --- 训练 Task 3 模型并计算 2D TGM 真实准确率与试次预测矩阵 ---
    tgm_matrix = zeros(n_win, n_win);
    pred_tgm   = zeros(n_te, n_win, n_win); % [试次 x Task3窗口 x Task2窗口]
    
    % 预存 Out-of-sample 标准化特征
    norm_X3 = cell(n_win, 1);
    norm_X2_for_w3 = cell(n_win, 1);
    
    for w3 = 1:n_win
        X_tr_w3 = double(squeeze(X3_3d(:, :, w3)));
        mu_w3  = mean(X_tr_w3, 1);
        sig_w3 = std(X_tr_w3, 0, 1);
        sig_w3(sig_w3 < 1e-6) = 1;
        norm_X3{w3} = (X_tr_w3 - mu_w3) ./ sig_w3;
        
        X2_scaled = zeros(n_te, cfg.n_bands, n_win);
        for w2 = 1:n_win
            X_te_w2 = double(squeeze(X2_3d(:, :, w2)));
            X2_scaled(:, :, w2) = (X_te_w2 - mu_w3) ./ sig_w3;
        end
        norm_X2_for_w3{w3} = X2_scaled;
    end
    
    for w3 = 1:n_win
        mdl_w3 = fitclinear(norm_X3{w3}, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', cfg.svm_lambda);
        X2_all_w2 = norm_X2_for_w3{w3};
        for w2 = 1:n_win
            y_pred = predict(mdl_w3, squeeze(X2_all_w2(:, :, w2)));
            pred_tgm(:, w3, w2) = y_pred;
            
            sens = sum(y_te == 1 & y_pred == 1) / max(1, sum(y_te == 1));
            spec = sum(y_te == 0 & y_pred == 0) / max(1, sum(y_te == 0));
            tgm_matrix(w3, w2) = (sens + spec) / 2;
        end
    end
    
    % 对角线 1D 同步曲线
    diag_joint = diag(tgm_matrix)';
    diag_joint_s = smoothdata(diag_joint, 'gaussian', cfg.smooth_pts);
    tgm_matrix_s = imgaussfilt(tgm_matrix, 0.8);
    
    % --- 2D TGM 200 次极速向量化置换检验 ---
    fprintf('  - 正在执行 2D TGM 的 %d 次置换检验与 2D 簇质量统计推断 ...\n', cfg.n_perm);
    t_perm = tic;
    null_tgm_dist  = zeros(cfg.n_perm, n_win, n_win);
    null_diag_dist = zeros(cfg.n_perm, n_win);
    
    for p = 1:cfg.n_perm
        y_te_p = y_te(randperm(n_te));
        pos_m = (y_te_p == 1);
        neg_m = (y_te_p == 0);
        
        % 极速向量化计算 51x51 矩阵的平衡正确率
        sens_p = squeeze(mean(pred_tgm(pos_m, :, :) == 1, 1));
        spec_p = squeeze(mean(pred_tgm(neg_m, :, :) == 0, 1));
        tgm_p  = (sens_p + spec_p) / 2;
        
        null_tgm_dist(p, :, :) = tgm_p;
        null_diag_dist(p, :)   = diag(tgm_p)';
    end
    fprintf('  - 2D 置换检验完成! 耗时: %.2f 秒\n', toc(t_perm));
    
    % 平滑真实 2D TGM
    tgm_matrix_s = imgaussfilt(tgm_matrix, 0.8);
    
    % 平滑 200 次置换的 2D TGM
    null_tgm_s = zeros(cfg.n_perm, n_win, n_win);
    for p = 1:cfg.n_perm
        null_tgm_s(p, :, :) = imgaussfilt(squeeze(null_tgm_dist(p, :, :)), 0.8);
    end
    
    % --- 2D Cluster-Mass FWE 显著性检验 ---
    % 逐像素经验 p 值 (基于平滑后的真实矩阵与平滑后的零分布对比)
    tgm_s_3d = reshape(tgm_matrix_s, [1, n_win, n_win]);
    p_tgm_pointwise = squeeze((1 + sum(null_tgm_s >= tgm_s_3d, 1)) / (1 + cfg.n_perm));
    
    % 2D 候选显著像素掩码 (p < 0.05 且 Task2 反应时间 >= 0)
    [T2_grid, ~] = meshgrid(t_centers, t_centers);
    sig_mask_2d_raw = (p_tgm_pointwise < 0.05) & (T2_grid >= 0);
    
    % 连通域分析 (8-连通网格)
    CC_real = bwconncomp(sig_mask_2d_raw, 8);
    real_masses_2d = zeros(1, CC_real.NumObjects);
    for ci = 1:CC_real.NumObjects
        real_masses_2d(ci) = sum(tgm_matrix_s(CC_real.PixelIdxList{ci}) - 0.50);
    end
    
    % 计算 200 次置换中零分布的最大 2D 簇质量
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
    
    % 筛选通过 2D FWE 校正的显著簇 (p_fwe < 0.05)
    sig_2d_cluster_mask = false(n_win, n_win);
    n_sig_2d_clusters   = 0;
    for ci = 1:CC_real.NumObjects
        p_cluster_2d = (1 + sum(max_null_masses_2d >= real_masses_2d(ci))) / (1 + cfg.n_perm);
        if p_cluster_2d < 0.05
            sig_2d_cluster_mask(CC_real.PixelIdxList{ci}) = true;
            n_sig_2d_clusters = n_sig_2d_clusters + 1;
        end
    end
    
    % 1D 对角线置换检验 (Cluster-Mass FWE)
    null_diag_s = smoothdata(null_diag_dist, 2, 'gaussian', cfg.smooth_pts);
    diag_p95    = prctile(null_diag_s, 95, 1);
    raw_p_diag  = mean(null_diag_dist >= diag_joint, 1);
    pt_sig_diag = (raw_p_diag < 0.05) & (t_centers >= 0);
    
    % --- 学术级 1:1 双子图排版 (带 2D 显著轮廓线) ---
    h_fig = figure('Visible', 'off', 'Units', 'pixels', 'Position', [100, 100, 1280, 520], 'Color', 'w');
    
    % 左子图: 1D 对角线同步解码
    subplot(1, 2, 1);
    hold on;
    fill([t_centers, fliplr(t_centers)], [diag_p95, fliplr(ones(1, n_win)*0.5)], ...
         [0.85, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'Null 95% CI');
    yline(0.5, 'k--', 'LineWidth', 1.0, 'Alpha', 0.6, 'HandleVisibility', 'off');
    xline(0, 'k:', 'LineWidth', 1.0, 'Alpha', 0.5, 'HandleVisibility', 'off');
    
    p_diag = plot(t_centers, diag_joint_s, 'Color', [0.85, 0.35, 0.05], 'LineWidth', 2.4, ...
                  'DisplayName', 'Multi-Band (Diagonal)');
    if any(pt_sig_diag)
        sig_y = 0.38 * ones(1, sum(pt_sig_diag));
        scatter(t_centers(pt_sig_diag), sig_y, 25, [0.85, 0.35, 0.05], 'filled', 's', ...
                'HandleVisibility', 'off');
    end
    hold off;
    box off; set(gca, 'TickDir', 'out', 'FontSize', 10, 'LineWidth', 1.0);
    xlim([-200, 800]); ylim([0.35, 0.80]);
    xlabel('Time (ms)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Balanced Accuracy', 'FontSize', 11, 'FontWeight', 'bold');
    title(sprintf('A. Diagonal Synchronization (%s-%s)', sub_id, elec), 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'northeast', 'FontSize', 8, 'Box', 'off');
    
    % 右子图: 2D TGM 时间泛化矩阵热力图 (叠加显著轮廓线)
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
    
    % 如果有显著 2D 簇，使用白色轮廓线高亮标记 (White Contour)
    if any(sig_2d_cluster_mask(:))
        contour(t_centers, t_centers, double(sig_2d_cluster_mask), [0.5, 0.5], ...
                'Color', 'w', 'LineWidth', 2.0);
    end
    hold off;
    box off; set(gca, 'TickDir', 'out', 'FontSize', 10, 'LineWidth', 1.0);
    xlim([-200, 800]); ylim([-200, 800]);
    xlabel('Task 2 (Memory Fruit) Time (ms)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Task 3 (Physical Patch) Time (ms)', 'FontSize', 11, 'FontWeight', 'bold');
    title(sprintf('B. Temporal Generalization (%s-%s, 2D Sig Clusters: %d)', sub_id, elec, n_sig_2d_clusters), ...
          'FontSize', 12, 'FontWeight', 'bold');
    
    % 保存图片
    fig_png = fullfile(cfg.out_fig_dir, sprintf('%s_%s_tgm_permutation_test.png', sub_id, elec));
    exportgraphics(h_fig, fig_png, 'Resolution', 300);
    close(h_fig);
    
    fprintf('  - [%d/%d] %s-%s 处理完成! Peak TGM Acc: %.2f%%, 显著 2D 簇: %d (总耗时: %.2f 秒)\n', ...
        t_i, n_targets, sub_id, elec, max(tgm_matrix_s(:))*100, n_sig_2d_clusters, toc(t_start));
end

fprintf('\n========================================================================\n');
fprintf('>>> 2D TGM 200 次置换检验测试成功结束!\n');
fprintf('========================================================================\n');
