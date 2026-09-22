%% ========================================================================
% 脚本名称: test_cross_decoding_by_fruit.m
% 定位: 测试/探索脚本 (放在 matlab/test/ 中，不污染主分析流水线)
% 功能:
%   1. 【Task 3 物理知觉 -> Task 2 灰度物体记忆色彩 跨任务神经解码】
%   2. 【灰色水果分品类独立正确率统计】:
%      分别统计 Task 2 灰度条件下各品类水果的解码正确率:
%        - 草莓 (Strawberry): 60 试次, 隐含色为红 (+1)
%        - 西瓜 (Watermelon): 60 试次, 隐含色为红 (+1)
%        - 卷心菜 (Cabbage):  60 试次, 隐含色为绿 (0)
%        - 猕猴桃 (Kiwi):     60 试次, 隐含色为绿 (0)
%        - 全体平均 (Overall Mean): 240 试次平衡平均
%   3. 【单面板纯正确率折线图】:
%      移除 2D TGM 热力图，单图绘制 5 条曲线 (4 种水果 + 全体平均)
%      附带 50% 机会水平基准线与置换检验显著性标记
%   4. 【轻量 HDF5 流式切片读取与 16 线程极速并行】
% ========================================================================

clear; clc; close all;

%% 1. 主参数配置 (置顶直观，简写平铺，方便审阅调整)
cfg = struct();

% -------------------------------------------------------------------------
% 目标电极模式:
%   'significant' : C09 中检出的 7 个统计显著电极 (默认极速验证)
%   'concordant'  : C04 筛选出的全部 157 个同向显著电极
%   'custom'      : 自定义指定电极
% -------------------------------------------------------------------------
cfg.target_mode    = 'concordant'; 

% 当 cfg.target_mode = 'custom' 时生效:
cfg.custom_subs    = {'sub007'};
cfg.custom_elecs   = {'C4'};

% 时间窗与时程参数
cfg.win_len        = 20;                             % 滑动窗长 20 ms
cfg.win_step       = 20;                             % 滑动步长 20 ms
cfg.t_range        = [-200, 800];                    % 解码时程范围 (ms, 51个时间窗)

% 平滑参数
cfg.smooth_pts     = 5;                              % 平滑点数 (5点高斯平滑)

% 并行与统计参数
cfg.n_perm         = 200;                            % 置换检验次数
cfg.n_workers      = 16;                             % 16 线程极速并行
cfg.svm_lambda     = 0.01;                           % 岭正则化参数 Lambda

% 频段定义 (联合 6 频段特征 Multi-Band)
cfg.bands          = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.n_bands        = numel(cfg.bands);

% 调试与运行开关
cfg.max_elecs      = Inf;                            % 最大处理电极数
cfg.skip_existing  = true;                           % 跳过已完成电极 (支持断点续跑)

%% 2. 路径初始化
script_path = mfilename('fullpath');
c_idx = strfind(script_path, 'color_analyse_0825');
if ~isempty(c_idx)
    color_root = script_path(1 : c_idx(1) + length('color_analyse_0825') - 1);
else
    color_root = fullfile(fileparts(fileparts(fileparts(script_path))), 'color_analyse_0825');
end

data_root      = fullfile(color_root, 'process_data_new');
task_info_dir  = fullfile(color_root, 'task_info');
res_root       = fullfile(color_root, 'result');
c04_table      = fullfile(res_root, 'tables', 'color_effects_summary.mat');

tab_dir    = fullfile(res_root, 'tables');
tc_tab_dir = fullfile(tab_dir, 'cross_decoding_by_fruit_timecourses');
fig_dir    = fullfile(res_root, 'figures', 'cross_decoding_by_fruit');

if ~exist(tab_dir, 'dir'),    mkdir(tab_dir);    end
if ~exist(tc_tab_dir, 'dir'), mkdir(tc_tab_dir); end
if ~exist(fig_dir, 'dir'),    mkdir(fig_dir);    end

%% 3. 确定目标电极列表
switch cfg.target_mode
    case 'significant'
        % C09 中发现的 7 个 Cluster-Mass 显著电极
        target_subs  = {'sub001', 'sub003', 'sub006', 'sub006', 'sub008', 'sub008', 'sub008'};
        target_elecs = {'B3',     'C13',    'E8',     'H10',    'C2',     'C10',    'E8'};
        fprintf('>>> [目标筛选: significant] 处理 C09 检出的 7 个统计显著电极 ...\n');
        
    case 'concordant'
        if ~isfile(c04_table)
            error('未找到 C04 结果表: %s', c04_table);
        end
        loaded_c04 = load(c04_table);
        if isfield(loaded_c04, 'all_tbl'), c04_tbl = loaded_c04.all_tbl; else, c04_tbl = loaded_c04.res_table; end
        
        concord_mask = (c04_tbl.is_significant == 1) & ...
            (strcmp(c04_tbl.concordance_type, 'Concordant_Positive') | ...
             strcmp(c04_tbl.concordance_type, 'Concordant_Negative'));
        c04_sub_tbl = c04_tbl(concord_mask, :);
        
        elec_keys = strcat(c04_sub_tbl.subject, '_', c04_sub_tbl.channel);
        [~, u_ia] = unique(elec_keys, 'stable');
        
        target_subs  = c04_sub_tbl.subject(u_ia);
        target_elecs = c04_sub_tbl.channel(u_ia);
        fprintf('>>> [目标筛选: concordant] 共加载 %d 个同向显著色觉电极 ...\n', numel(target_subs));
        
    case 'custom'
        target_subs  = cfg.custom_subs;
        target_elecs = cfg.custom_elecs;
        fprintf('>>> [目标筛选: custom] 自定义 %d 个指定电极 ...\n', numel(target_subs));
end

n_total_elecs = min(numel(target_subs), cfg.max_elecs);
unique_subs = unique(target_subs(1:n_total_elecs), 'stable');

%% 4. 初始化多核并行池
if cfg.n_workers > 1
    curr_pool = gcp('nocreate');
    if isempty(curr_pool)
        parpool('local', cfg.n_workers);
    elseif curr_pool.NumWorkers ~= cfg.n_workers
        delete(curr_pool);
        parpool('local', cfg.n_workers);
    end
end

%% 5. 时间窗定义
t_centers = cfg.t_range(1) : cfg.win_step : cfg.t_range(2);
n_win = numel(t_centers);

% 汇总结果存储
summary_rows = {};
global_elec_count = 0;

%% 6. 主循环: 按被试批量流式加载与逐通道计算
for s_i = 1:numel(unique_subs)
    sub_id = unique_subs{s_i};
    
    sub_mask = strcmp(target_subs(1:n_total_elecs), sub_id);
    sub_elecs = target_elecs(sub_mask);
    n_sub_elecs = numel(sub_elecs);
    
    fprintf('\n========================================================================\n');
    fprintf('>>> [被试 %d/%d: %s] 批量处理本被试 %d 个电极 ...\n', ...
        s_i, numel(unique_subs), sub_id, n_sub_elecs);
    fprintf('========================================================================\n');
    
    t3_mat = fullfile(data_root, sub_id, 'task3_multiband_epoched.mat');
    t2_mat = fullfile(data_root, sub_id, 'task2_multiband_epoched.mat');
    
    if ~isfile(t3_mat) || ~isfile(t2_mat)
        warning('被试 %s 缺少 Task 2 或 Task 3 数据，跳过。', sub_id);
        continue;
    end
    
    t_load = tic;
    
    % 1. 读取 Task 3 元数据与流式频段 (零内存开销，彻底杜绝 OOM)
    time_ms = h5read(t3_mat, '/epoched_data/time_ms');
    time_ms = time_ms(:)';
    ch_list3_all = h5read(t3_mat, '/epoched_data/channels');
    if iscell(ch_list3_all)
        ch_list3_all = cellfun(@(x) char(x(:)'), ch_list3_all, 'UniformOutput', false);
    end
    sub_elecs = cellstr(sub_elecs);
    
    d3 = load(fullfile(task_info_dir, sub_id, 'task3_trial_info.mat'), 'trial_info');
    ti3 = d3.trial_info;
    tr_mask = strcmp(ti3.color, 'red') | strcmp(ti3.color, 'green');
    ti3_use = ti3(tr_mask, :);
    n_tr    = height(ti3_use);
    y_tr    = zeros(n_tr, 1);
    y_tr(strcmp(ti3_use.color, 'red')) = 1; % Red=1, Green=0
    
    [ch_match3, ch_sub_idx3] = ismember(sub_elecs, ch_list3_all);
    valid_elecs = sub_elecs(ch_match3);
    target_idx3 = ch_sub_idx3(ch_match3);
    
    ep3_bands = struct();
    for b = 1:cfg.n_bands
        b_name = cfg.bands{b};
        raw = h5read(t3_mat, ['/epoched_data/' b_name]);
        ep3_bands.(b_name) = raw(tr_mask, target_idx3, :);
        clear raw;
    end
    clear d3 ti3;
    
    % 2. 读取 Task 2 元数据与流式频段 (零内存开销，彻底杜绝 OOM)
    ch_list2_all = h5read(t2_mat, '/epoched_data/channels');
    if iscell(ch_list2_all)
        ch_list2_all = cellfun(@(x) char(x(:)'), ch_list2_all, 'UniformOutput', false);
    end
    d2 = load(fullfile(task_info_dir, sub_id, 'task2_trial_info.mat'), 'trial_info');
    ti2 = d2.trial_info;
    te_mask = strcmp(ti2.state, 'gray');
    ti2_use = ti2(te_mask, :);
    n_te    = height(ti2_use);
    
    fruit_list = ti2_use.fruit;
    m_straw = strcmp(fruit_list, 'strawberry');
    m_water = strcmp(fruit_list, 'watermelon');
    m_cabb  = strcmp(fruit_list, 'cabbage');
    m_kiwi  = strcmp(fruit_list, 'kiwi');
    y_te = double(m_straw | m_water); % 1: Red, 0: Green
    
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
        b_name = cfg.bands{b};
        raw = h5read(t2_mat, ['/epoched_data/' b_name]);
        ep2_bands.(b_name) = raw(te_mask, target_idx2, :);
        clear raw;
    end
    clear d2 ti2;
    
    fprintf('  [I/O 优化] 完成 %s 的 %d 个目标通道流式切片加载 (耗时: %.2f 秒)\n', ...
        sub_id, numel(final_elecs), toc(t_load));
    
    %% 7. 逐通道计算跨任务解码
    for e_i = 1:numel(final_elecs)
        elec = final_elecs{e_i};
        global_elec_count = global_elec_count + 1;
        
        fig_png = fullfile(fig_dir, sprintf('%s_%s_cross_decoding_by_fruit.png', sub_id, elec));
        csv_tc  = fullfile(tc_tab_dir, sprintf('%s_%s_cross_decoding_by_fruit.csv', sub_id, elec));
        
        if cfg.skip_existing && isfile(fig_png) && isfile(csv_tc)
            try
                t_ex = readtable(csv_tc);
                mean_sm_ex = smoothdata(t_ex.acc_overall, 'gaussian', cfg.smooth_pts);
                [p_m_ex, p_idx_ex] = max(mean_sm_ex);
                p_t_ex = t_centers(p_idx_ex);
                summary_rows(end+1, :) = {sub_id, elec, p_m_ex, p_t_ex, ...
                    max(smoothdata(t_ex.acc_strawberry, 'gaussian', cfg.smooth_pts)), ...
                    max(smoothdata(t_ex.acc_watermelon, 'gaussian', cfg.smooth_pts)), ...
                    max(smoothdata(t_ex.acc_cabbage, 'gaussian', cfg.smooth_pts)), ...
                    max(smoothdata(t_ex.acc_kiwi, 'gaussian', cfg.smooth_pts)), ...
                    0, double(any(t_ex.is_sig_cluster))};
                fprintf('  - [%d/%d] %s-%s 已存在，成功复用已存结果。\n', global_elec_count, n_total_elecs, sub_id, elec);
                continue;
            catch
                % 若读取出错则重新计算
            end
        end
        
        t_elec = tic;
        
        % --- 步骤 1: 提取 51 个时间窗的多频段特征 ---
        X3_3d = zeros(n_tr, cfg.n_bands, n_win);
        X2_3d = zeros(n_te, cfg.n_bands, n_win);
        
        for b = 1:cfg.n_bands
            b_name = cfg.bands{b};
            raw3 = squeeze(ep3_bands.(b_name)(:, e_i, :));
            raw2 = squeeze(ep2_bands.(b_name)(:, e_i, :));
            
            for w = 1:n_win
                tc = t_centers(w);
                t_m = (time_ms >= (tc - cfg.win_len / 2)) & (time_ms < (tc + cfg.win_len / 2));
                X3_3d(:, b, w) = mean(raw3(:, t_m), 2);
                X2_3d(:, b, w) = mean(raw2(:, t_m), 2);
            end
        end
        
        % --- 步骤 2: 真实解码与水果分品类正确率统计 ---
        acc_straw = zeros(1, n_win);
        acc_water = zeros(1, n_win);
        acc_cabb  = zeros(1, n_win);
        acc_kiwi  = zeros(1, n_win);
        acc_mean  = zeros(1, n_win);
        
        % 预存标准化特征，供置换检验快速复用
        X3_diag_norm = cell(n_win, 1);
        X2_diag_norm = cell(n_win, 1);
        
        for w = 1:n_win
            X_tr_w = double(squeeze(X3_3d(:, :, w)));
            X_te_w = double(squeeze(X2_3d(:, :, w)));
            
            % Out-of-Sample 标准化: 严格基于 Task 3 均值与标准差
            mu_w  = mean(X_tr_w, 1);
            sig_w = std(X_tr_w, 0, 1);
            sig_w(sig_w < 1e-6) = 1;
            
            X_tr_norm = (X_tr_w - mu_w) ./ sig_w;
            X_te_norm = (X_te_w - mu_w) ./ sig_w;
            
            X3_diag_norm{w} = X_tr_norm;
            X2_diag_norm{w} = X_te_norm;
            
            % 训练线性 SVM 模型
            mdl = fitclinear(X_tr_norm, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', cfg.svm_lambda);
            y_pred = predict(mdl, X_te_norm);
            
            % 统计各水果品类独立准确率
            acc_straw(w) = sum(y_pred(m_straw) == 1) / max(1, sum(m_straw));
            acc_water(w) = sum(y_pred(m_water) == 1) / max(1, sum(m_water));
            acc_cabb(w)  = sum(y_pred(m_cabb)  == 0) / max(1, sum(m_cabb));
            acc_kiwi(w)  = sum(y_pred(m_kiwi)  == 0) / max(1, sum(m_kiwi));
            
            % 全体平均准确率 (四品类均衡平均)
            acc_mean(w)  = (acc_straw(w) + acc_water(w) + acc_cabb(w) + acc_kiwi(w)) / 4;
        end
        
        % --- 步骤 3: 16 线程并行置换检验与统计显著簇校正 ---
        n_perm = cfg.n_perm;
        perm_acc_matrix = zeros(n_perm, n_win);
        svm_lambda = cfg.svm_lambda;
        
        parfor p = 1:n_perm
            y_tr_perm = y_tr(randperm(n_tr));
            row_p = zeros(1, n_win);
            for w = 1:n_win
                X_tr_p = X3_diag_norm{w};
                X_te_p = X2_diag_norm{w};
                mdl_p = fitclinear(X_tr_p, y_tr_perm, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', svm_lambda);
                y_pred_p = predict(mdl_p, X_te_p);
                
                s_p = sum(y_pred_p(m_straw) == 1) / max(1, sum(m_straw));
                w_p = sum(y_pred_p(m_water) == 1) / max(1, sum(m_water));
                c_p = sum(y_pred_p(m_cabb)  == 0) / max(1, sum(m_cabb));
                k_p = sum(y_pred_p(m_kiwi)  == 0) / max(1, sum(m_kiwi));
                row_p(w) = (s_p + w_p + c_p + k_p) / 4;
            end
            perm_acc_matrix(p, :) = row_p;
        end
        
        % 计算零分布 95% 置信上限
        p95_thresh = prctile(perm_acc_matrix, 95, 1);
        
        % 基于聚类质量的 FWE 校正 (Cluster-Mass Permutation Test)
        raw_p = mean(perm_acc_matrix >= acc_mean, 1);
        pt_sig = raw_p < 0.05;
        
        [L, num_clusters] = bwlabel(pt_sig);
        real_masses = zeros(1, num_clusters);
        for c = 1:num_clusters
            real_masses(c) = sum(acc_mean(L == c) - 0.50);
        end
        
        % 计算零分布的最大簇质量
        max_null_masses = zeros(1, n_perm);
        for p = 1:n_perm
            p_curve = perm_acc_matrix(p, :);
            p_pt_sig = p_curve > p95_thresh;
            [L_p, n_c_p] = bwlabel(p_pt_sig);
            if n_c_p > 0
                m_list = zeros(1, n_c_p);
                for cp = 1:n_c_p
                    m_list(cp) = sum(p_curve(L_p == cp) - 0.50);
                end
                max_null_masses(p) = max(m_list);
            else
                max_null_masses(p) = 0;
            end
        end
        cluster_fwe_thresh = prctile(max_null_masses, 95);
        
        sig_cluster_mask = false(1, n_win);
        for c = 1:num_clusters
            if real_masses(c) >= cluster_fwe_thresh && cluster_fwe_thresh > 0
                sig_cluster_mask(L == c) = true;
            end
        end
        
        % --- 步骤 4: 曲线平滑处理 ---
        straw_sm = smoothdata(acc_straw, 'gaussian', cfg.smooth_pts);
        water_sm = smoothdata(acc_water, 'gaussian', cfg.smooth_pts);
        cabb_sm  = smoothdata(acc_cabb,  'gaussian', cfg.smooth_pts);
        kiwi_sm  = smoothdata(acc_kiwi,  'gaussian', cfg.smooth_pts);
        mean_sm  = smoothdata(acc_mean,  'gaussian', cfg.smooth_pts);
        
        null_sm_mat = smoothdata(perm_acc_matrix, 2, 'gaussian', cfg.smooth_pts);
        null_hi = prctile(null_sm_mat, 97.5, 1);
        null_lo = prctile(null_sm_mat, 2.5, 1);
        
        % 统计峰值与潜伏期
        [peak_mean, p_idx] = max(mean_sm);
        peak_time = t_centers(p_idx);
        peak_straw = max(straw_sm);
        peak_water = max(water_sm);
        peak_cabb  = max(cabb_sm);
        peak_kiwi  = max(kiwi_sm);
        has_sig = any(sig_cluster_mask);
        
        % --- 步骤 5: 单面板纯正确率折线图绘制 (完全依照 C09 学术规范风格) ---
        col_joint = [0.85, 0.37, 0.01]; % 陶土暖橙 (Overall Mean)
        col_null  = [0.90, 0.90, 0.90]; % 浅灰阴影
        col_clust = [1.00, 0.88, 0.88]; % 显著簇粉红高亮
        
        % 4 种水果配色 (柔和雅致学术色)
        col_straw = [0.85, 0.22, 0.22]; % Strawberry: Crimson
        col_water = [0.90, 0.50, 0.50]; % Watermelon: Salmon / Rose
        col_cabb  = [0.15, 0.65, 0.35]; % Cabbage: Emerald Green
        col_kiwi  = [0.45, 0.70, 0.25]; % Kiwi: Olive Green
        
        h_fig = figure('Position', [150, 150, 640, 480], 'Color', 'w', 'Visible', 'off');
        hold on;
        
        % 1. 显著时间簇阴影
        if has_sig
            [L_sig, n_sig_c] = bwlabel(sig_cluster_mask);
            for sci = 1:n_sig_c
                c_idx_pts = find(L_sig == sci);
                c_x1 = t_centers(c_idx_pts(1));
                c_x2 = t_centers(c_idx_pts(end));
                fill([c_x1, c_x2, c_x2, c_x1], [0.35, 0.35, 0.80, 0.80], col_clust, ...
                    'EdgeColor', 'none', 'FaceAlpha', 0.6, 'HandleVisibility', 'off');
            end
        end
        
        % 2. 绘制置换检验 95% 经验零分布浅灰色阴影带 (2.5% ~ 97.5%)
        fill([t_centers, fliplr(t_centers)], [null_hi, fliplr(null_lo)], col_null, ...
            'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'Null 95% CI');
        
        % 3. 绘制 50% 机会水平基准线与 0 ms 刺激起始线
        yline(0.50, '--', 'Color', [0.55, 0.55, 0.55], 'LineWidth', 1.0, 'HandleVisibility', 'off');
        xline(0, ':', 'Color', [0.40, 0.40, 0.40], 'LineWidth', 1.0, 'HandleVisibility', 'off');
        
        % 4. 绘制 4 种水果曲线
        plot(t_centers, straw_sm, 'Color', [col_straw, 0.60], 'LineWidth', 1.3, 'DisplayName', 'Strawberry');
        plot(t_centers, water_sm, 'Color', [col_water, 0.60], 'LineWidth', 1.3, 'DisplayName', 'Watermelon');
        plot(t_centers, cabb_sm,  'Color', [col_cabb,  0.60], 'LineWidth', 1.3, 'DisplayName', 'Cabbage');
        plot(t_centers, kiwi_sm,  'Color', [col_kiwi,  0.60], 'LineWidth', 1.3, 'DisplayName', 'Kiwi');
        
        % 5. 绘制全体平均加粗陶土橙线 (Multi-Band 主线)
        plot(t_centers, mean_sm,  'Color', col_joint, 'LineWidth', 2.6, 'DisplayName', 'Overall Mean');
        
        % 6. 显著时间点方块标记
        if any(pt_sig)
            plot(t_centers(pt_sig), repmat(0.38, 1, sum(pt_sig)), 's', ...
                'MarkerFaceColor', col_joint, 'MarkerEdgeColor', 'none', 'MarkerSize', 4, ...
                'HandleVisibility', 'off');
        end
        
        % 7. 坐标轴美化 (box off，学术极简风，严禁中文)
        xlim(cfg.t_range);
        ylim([0.35, 0.80]);
        xlabel('Time (ms)', 'FontSize', 11, 'FontWeight', 'bold');
        ylabel('Balanced Accuracy', 'FontSize', 11, 'FontWeight', 'bold');
        title(sprintf('%s - %s', sub_id, elec), 'FontSize', 13, 'FontWeight', 'bold');
        legend('Location', 'northeast', 'FontSize', 8.5, 'Box', 'off');
        grid off;
        box off;
        set(gca, 'TickDir', 'out', 'LineWidth', 1.0, 'FontSize', 10);
        
        % 保存图片
        exportgraphics(h_fig, fig_png, 'Resolution', 300);
        close(h_fig);
        
        % --- 步骤 6: 导出单通道时程 CSV ---
        tc_table = table(t_centers(:), acc_mean(:), acc_straw(:), acc_water(:), acc_cabb(:), acc_kiwi(:), ...
            p95_thresh(:), sig_cluster_mask(:), ...
            'VariableNames', {'time_ms', 'acc_overall', 'acc_strawberry', 'acc_watermelon', ...
                              'acc_cabbage', 'acc_kiwi', 'null_p95', 'is_sig_cluster'});
        writetable(tc_table, csv_tc);
        
        % 记录汇总
        summary_rows(end+1, :) = {sub_id, elec, peak_mean, peak_time, ...
            peak_straw, peak_water, peak_cabb, peak_kiwi, num_clusters, double(has_sig)};
        
        fprintf('  - [%d/%d] %s-%s 处理完成! 全体峰值: %.2f%% (%d ms), 草莓: %.2f%%, 西瓜: %.2f%%, 卷心菜: %.2f%%, 猕猴桃: %.2f%%, 显著簇: %d (耗时: %.2f秒)\n', ...
            global_elec_count, n_total_elecs, sub_id, elec, peak_mean*100, peak_time, ...
            peak_straw*100, peak_water*100, peak_cabb*100, peak_kiwi*100, num_clusters, toc(t_elec));
        
        clear X3_3d X2_3d X3_diag_norm X2_diag_norm perm_acc_matrix tc_table;
    end
    
    % 清理中间大变量
    clear ep3_bands ep2_bands;
end

%% 8. 导出总汇总表
if ~isempty(summary_rows)
    sum_table = cell2table(summary_rows, 'VariableNames', { ...
        'subject', 'channel', 'mean_peak_acc', 'mean_peak_time_ms', ...
        'strawberry_peak_acc', 'watermelon_peak_acc', 'cabbage_peak_acc', 'kiwi_peak_acc', ...
        'n_sig_clusters', 'has_sig_cluster'});
    
    sum_csv = fullfile(tab_dir, 'cross_decoding_by_fruit_summary.csv');
    sum_mat = fullfile(tab_dir, 'cross_decoding_by_fruit_summary.mat');
    
    writetable(sum_table, sum_csv);
    save(sum_mat, 'sum_table', 'cfg');
    
    fprintf('\n========================================================================\n');
    fprintf('>>> 批量测试全部完成!\n');
    fprintf('>>> 汇总 CSV: %s\n', sum_csv);
    fprintf('>>> 汇总 MAT: %s\n', sum_mat);
    fprintf('>>> 图表目录: %s\n', fig_dir);
    fprintf('========================================================================\n');
end
