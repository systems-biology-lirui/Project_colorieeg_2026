%% ========================================================================
% 脚本名称: C06_task2_memory_color_decoding_0825.m
% 功能:
%   1. 【Task 2 记忆颜色多频段 Decoding 与聚类置换检验 (纯 MATLAB 实现)】
%   2. 【支持全量同向显著电极或自定义指定电极】
%      - cfg.target_mode = 'concordant': 自动读取 C04 筛选出的全部【总体显著且四类别同向】电极 (默认, 157个位点)
%      - cfg.target_mode = 'custom'    : 指定特定被试与电极列表 (如 sub001-G13, sub007-C4)
%   3. 【按被试批量极速加载】
%      每个被试数据仅载入一次，避免反复读取大文件，大幅节省 I/O 耗时与内存开销
%   4. 【20ms 紧凑时间窗与滑动特征】
%      滑动窗长 20ms，步长 20ms，覆盖 [-200, 800] ms
%   5. 【杜绝形状混淆】
%      严格执行跨水果配对 4 折交叉验证 (Leave-One-Fruit-Pair-Out)
%   6. 【多频段联合与各单频段消融】
%      同时计算 6 频段联合特征与 6 个单频段独立特征
%   7. 【20 线程并行置换检验与统计校正】
%      parpool 调度 20 线程执行标签置换与聚类检验
%   8. 【曲线平滑与规范绘图】
%      - 对真实 decoding 曲线进行高斯平滑
%      - 图例只保留每种频段 (Multi-Band 及 Delta~High-Gamma)
%      - 主标题仅包含被试与电极编号 (如: sub001 - G13)
%      - 无背景方格 (grid off)
%   9. 【导出汇总表与逐通道时程 CSV】
% ========================================================================

clear; clc; close all;

%% 1. 主参数配置 (置顶直观，简写平铺，方便审阅调整)
cfg = struct();

% -------------------------------------------------------------------------
% 目标电极选择模式:
%   'concordant'     : 自动读取 C04 筛选出的全部【总体显著且四类别同向】电极 (157个)
%   'non_concordant' : 自动读取 C04 筛选出的全部【总体显著但不同向】电极 (72个)
%   'custom'         : 指定目标电极列表 (如指定的关注电极: sub001-G13, sub007-C4)
% -------------------------------------------------------------------------
cfg.target_mode    = 'non_concordant'; 

% 当 cfg.target_mode = 'custom' 时生效:
cfg.custom_subs    = {'sub001', 'sub007'};
cfg.custom_elecs   = {'G13', 'C4'};

% 时间窗与时程参数 (按要求: 窗口改为 20ms)
cfg.win_len        = 20;                             % 滑动窗长 20 ms
cfg.win_step       = 20;                             % 滑动步长 20 ms (无重叠覆盖)
cfg.t_range        = [-200, 800];                    % 解码时程范围 (ms)

% 平滑参数 (按要求: 对 decoding 曲线进行平滑)
cfg.smooth_pts     = 5;                              % 平滑点数 (5点平滑)
cfg.smooth_typ     = 'gaussian';                     % 平滑方式: gaussian 或 movmean

% 并行与统计参数
cfg.n_perm         = 200;                            % 置换检验次数 (批处理推荐 200 次，兼顾统计效力与运行速度)
cfg.n_workers      = 20;                             % 20 线程极速 CPU 并行
cfg.cnd_state      = 'gray';                         % 仅分析灰色水果 (零物理颜色偏倚)

% 频段定义与规范学术显示标签 (避免下划线被识别为下标)
cfg.bands          = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.bands_disp     = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low-Gamma', 'High-Gamma'};
cfg.n_bands        = numel(cfg.bands);

% 调试或运行范围 ('all': 运行全部目标电极; 或指定数量)
cfg.max_elecs      = Inf;                            % Inf: 全量批处理运行全部目标电极
cfg.skip_existing  = true;                           % 若已完成出图与CSV则跳过，支持断点续跑

% 路径设置
script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(script_dir));
data_root  = fullfile(proj_root, 'color_analyse_0825', 'process_data_new');
res_root   = fullfile(proj_root, 'color_analyse_0825', 'result');
c04_table  = fullfile(res_root, 'tables', 'color_effects_summary.mat');

tab_dir    = fullfile(res_root, 'tables');
tc_tab_dir = fullfile(tab_dir, sprintf('decoding_%s_timecourses', cfg.target_mode));
fig_dir    = fullfile(res_root, 'figures', sprintf('decoding_%s', cfg.target_mode));

if ~exist(tab_dir, 'dir'),    mkdir(tab_dir);    end
if ~exist(tc_tab_dir, 'dir'), mkdir(tc_tab_dir); end
if ~exist(fig_dir, 'dir'),    mkdir(fig_dir);    end

%% 2. 确定目标电极通道列表
fprintf('========================================================================\n');
fprintf('  【C07: Task 2 记忆颜色多频段 Decoding 与聚类置换检验】  \n');
fprintf('========================================================================\n');

if strcmp(cfg.target_mode, 'custom')
    target_subs   = cfg.custom_subs(:);
    target_elecs  = cfg.custom_elecs(:);
    n_total_elecs = min(numel(target_subs), cfg.max_elecs);
    fprintf('[+] 当前为【自定义指定电极】模式，共 %d 个通道待分析。\n', n_total_elecs);
elseif strcmp(cfg.target_mode, 'non_concordant')
    if ~isfile(c04_table)
        error('未找到 C04 筛选汇总表: %s\n请先执行 C04_screen_color_channels_0825.m！', c04_table);
    end
    loaded_c04 = load(c04_table);
    if isfield(loaded_c04, 'all_tbl'), c04_tbl = loaded_c04.all_tbl; else, c04_tbl = loaded_c04.res_table; end
    
    % 同向显著电极集合 (用于排除)
    concord_mask = (c04_tbl.is_significant == 1) & ...
        (strcmp(c04_tbl.concordance_type, 'Concordant_Positive') | ...
         strcmp(c04_tbl.concordance_type, 'Concordant_Negative'));
    concord_keys = unique(strcat(c04_tbl.subject(concord_mask), '_', c04_tbl.channel(concord_mask)));
    
    % 全部总体显著电极
    sig_mask = (c04_tbl.is_significant == 1);
    all_sig_tbl = c04_tbl(sig_mask, :);
    all_sig_keys = strcat(all_sig_tbl.subject, '_', all_sig_tbl.channel);
    [~, u_ia] = unique(all_sig_keys, 'stable');
    
    cand_subs  = all_sig_tbl.subject(u_ia);
    cand_elecs = all_sig_tbl.channel(u_ia);
    cand_keys  = all_sig_keys(u_ia);
    
    % 排除同向电极，保留纯不同向(Category_Biased)显著电极
    non_concord_mask = ~ismember(cand_keys, concord_keys);
    target_subs   = cand_subs(non_concord_mask);
    target_elecs  = cand_elecs(non_concord_mask);
    n_total_elecs = min(numel(target_subs), cfg.max_elecs);
    fprintf('[+] 当前为【总体显著但不同向】模式，共筛选出 %d 个目标电极。\n', n_total_elecs);
else
    if ~isfile(c04_table)
        error('未找到 C04 筛选汇总表: %s\n请先执行 C04_screen_color_channels_0825.m！', c04_table);
    end
    loaded_c04 = load(c04_table);
    if isfield(loaded_c04, 'all_tbl'), c04_tbl = loaded_c04.all_tbl; else, c04_tbl = loaded_c04.res_table; end
    concord_mask = (c04_tbl.is_significant == 1) & ...
        (strcmp(c04_tbl.concordance_type, 'Concordant_Positive') | ...
         strcmp(c04_tbl.concordance_type, 'Concordant_Negative'));
    c04_concord = c04_tbl(concord_mask, :);
    
    % 提取唯一的 [被试_电极] 组合
    elec_keys = strcat(c04_concord.subject, '_', c04_concord.channel);
    [~, u_ia] = unique(elec_keys, 'stable');
    
    target_subs   = c04_concord.subject(u_ia);
    target_elecs  = c04_concord.channel(u_ia);
    n_total_elecs = min(numel(target_subs), cfg.max_elecs);
    fprintf('[+] 当前为【全量同向显著电极】模式，共筛选出 %d 个目标电极。\n', n_total_elecs);
end

fprintf('[+] 时间窗口: %d ms | 步长: %d ms | 曲线平滑: %s (%d点) | 置换次数: %d\n', ...
    cfg.win_len, cfg.win_step, cfg.smooth_typ, cfg.smooth_pts, cfg.n_perm);

%% 3. 启动 20 线程并行池
curr_pool = gcp('nocreate');
if isempty(curr_pool)
    fprintf('[+] 正在启动 %d 线程并行池 ...\n', cfg.n_workers);
    parpool('local', cfg.n_workers);
elseif curr_pool.NumWorkers ~= cfg.n_workers
    fprintf('[+] 调整并行池 Worker 数量至 %d ...\n', cfg.n_workers);
    delete(curr_pool);
    parpool('local', cfg.n_workers);
else
    fprintf('[+] 并行池已就绪 (NumWorkers = %d)\n', curr_pool.NumWorkers);
end

%% 4. 按被试分组批量解码 (避免反复加载大文件)
unique_subs = unique(target_subs(1:n_total_elecs), 'stable');
summary_list = struct([]);

% 配色定义
col_joint = [0.85, 0.37, 0.01]; % 陶土暖橙 (Multi-Band)
col_null  = [0.85, 0.85, 0.85]; % 浅灰阴影
band_cols = [
    0.40, 0.40, 0.40;  % Delta: 灰
    0.95, 0.60, 0.20;  % Theta: 杏黄
    0.20, 0.45, 0.75;  % Alpha: 钴蓝
    0.10, 0.65, 0.45;  % Beta: 青绿
    0.50, 0.35, 0.75;  % Low_Gamma: 紫灰
    0.90, 0.15, 0.50   % High_Gamma: 玫红
];

% 4 折跨水果配对 (Leave-One-Fruit-Pair-Out)
folds = {
    {'strawberry', 'cabbage'}, {'watermelon', 'kiwi'};
    {'strawberry', 'kiwi'},    {'watermelon', 'cabbage'};
    {'watermelon', 'cabbage'}, {'strawberry', 'kiwi'};
    {'watermelon', 'kiwi'},    {'strawberry', 'cabbage'}
};
n_folds = size(folds, 1);

t_centers = cfg.t_range(1) : cfg.win_step : cfg.t_range(2);
n_win = numel(t_centers);

global_elec_count = 0;

for s_i = 1:numel(unique_subs)
    sub_id = unique_subs{s_i};
    
    % 当前被试的所有目标电极
    sub_mask = strcmp(target_subs(1:n_total_elecs), sub_id);
    sub_elecs = target_elecs(sub_mask);
    n_sub_elecs = numel(sub_elecs);
    
    fprintf('\n========================================================================\n');
    fprintf('>>> [被试 %d/%d: %s] 开始批量处理本被试 %d 个显著电极 ...\n', ...
        s_i, numel(unique_subs), sub_id, n_sub_elecs);
    fprintf('========================================================================\n');
    
    % 加载该被试 Task 2 多频段数据 (整被试只载入一次)
    mat_file = fullfile(data_root, sub_id, 'task2_multiband_epoched.mat');
    if ~isfile(mat_file)
        warning('未找到被试 %s 数据文件，跳过。', sub_id);
        continue;
    end
    
    t_load = tic;
    ep_data = load(mat_file, 'epoched_data');
    ep = ep_data.epoched_data;
    time_ms = ep.time_ms(:)';
    ti = ep.trial_info;
    
    % 仅保留灰色水果试次
    gray_mask = strcmp(ti.state, cfg.cnd_state);
    ti_gray = ti(gray_mask, :);
    fruit_list = ti_gray.fruit;
    mem_color_list = ti_gray.memory_color;
    n_gray = height(ti_gray);
    
    % 二分类标签: red = 1, green = 0
    y = zeros(n_gray, 1);
    y(strcmp(mem_color_list, 'red')) = 1;
    
    % 构建 4 折掩码
    fold_tr_masks = false(n_gray, n_folds);
    fold_te_masks = false(n_gray, n_folds);
    for f_i = 1:n_folds
        fold_tr_masks(:, f_i) = ismember(fruit_list, folds{f_i, 1});
        fold_te_masks(:, f_i) = ismember(fruit_list, folds{f_i, 2});
    end
    
    fprintf('    [+] 数据载入完成 (耗时 %.2f 秒)，灰色试次数: %d (Red: %d, Green: %d)\n', ...
        toc(t_load), n_gray, sum(y==1), sum(y==0));
    
    % 逐电极循环解码
    for e_i = 1:n_sub_elecs
        ch_name = sub_elecs{e_i};
        global_elec_count = global_elec_count + 1;
        
        fig_png = fullfile(fig_dir, sprintf('%s_%s_memory_color_decoding.png', sub_id, ch_name));
        tc_csv  = fullfile(tc_tab_dir, sprintf('%s_%s_decoding_timecourse.csv', sub_id, ch_name));
        if cfg.skip_existing && isfile(fig_png) && isfile(tc_csv)
            fprintf('    [%d/%d | 全局 %d/%d] 电极 [%s - %s] 结果已存在，跳过。\n', ...
                e_i, n_sub_elecs, global_elec_count, n_total_elecs, sub_id, ch_name);
            continue;
        end
        
        t_elec = tic;
        fprintf('    [%d/%d | 全局 %d/%d] 正在解码电极: [%s - %s] ...\n', ...
            e_i, n_sub_elecs, global_elec_count, n_total_elecs, sub_id, ch_name);
        
        ch_idx = find(strcmp(ep.channels, ch_name), 1);
        if isempty(ch_idx)
            warning('通道 %s 不在数据通道列表中，跳过。', ch_name);
            continue;
        end
        
        % --- 步骤 1: 提取 20ms 滑动时间窗特征张量 [n_gray x n_bands x n_win] ---
        X_3d = zeros(n_gray, cfg.n_bands, n_win, 'single');
        for b = 1:cfg.n_bands
            b_name = cfg.bands{b};
            raw_b_2d = squeeze(ep.(b_name)(gray_mask, ch_idx, :)); % [n_gray x 750]
            for w = 1:n_win
                tc = t_centers(w);
                t_win_m = (time_ms >= (tc - cfg.win_len / 2)) & (time_ms < (tc + cfg.win_len / 2));
                X_3d(:, b, w) = mean(raw_b_2d(:, t_win_m), 2);
            end
        end
        
        % --- 步骤 2: 真实多频段联合与单频段解码 ---
        real_acc_joint = zeros(1, n_win);
        for w = 1:n_win
            X_w = double(squeeze(X_3d(:, :, w)));
            f_accs = zeros(1, n_folds);
            for f_i = 1:n_folds
                tr_m = fold_tr_masks(:, f_i);
                te_m = fold_te_masks(:, f_i);
                
                mu  = mean(X_w(tr_m, :), 1);
                sig = std(X_w(tr_m, :), 0, 1);
                sig(sig < 1e-6) = 1;
                
                X_tr_s = (X_w(tr_m, :) - mu) ./ sig;
                X_te_s = (X_w(te_m, :) - mu) ./ sig;
                
                mdl = fitclinear(X_tr_s, y(tr_m), 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', 0.01);
                y_pred = predict(mdl, X_te_s);
                
                sens = sum(y(te_m) == 1 & y_pred == 1) / max(1, sum(y(te_m) == 1));
                spec = sum(y(te_m) == 0 & y_pred == 0) / max(1, sum(y(te_m) == 0));
                f_accs(f_i) = (sens + spec) / 2;
            end
            real_acc_joint(w) = mean(f_accs);
        end
        
        % 6 个单频段独立解码
        single_band_accs = zeros(cfg.n_bands, n_win);
        for b = 1:cfg.n_bands
            for w = 1:n_win
                X_wb = double(X_3d(:, b, w));
                f_accs = zeros(1, n_folds);
                for f_i = 1:n_folds
                    tr_m = fold_tr_masks(:, f_i);
                    te_m = fold_te_masks(:, f_i);
                    
                    mu  = mean(X_wb(tr_m));
                    sig = std(X_wb(tr_m));
                    if sig < 1e-6, sig = 1; end
                    
                    X_tr_s = (X_wb(tr_m) - mu) ./ sig;
                    X_te_s = (X_wb(te_m) - mu) ./ sig;
                    
                    mdl = fitclinear(X_tr_s, y(tr_m), 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', 0.01);
                    y_pred = predict(mdl, X_te_s);
                    
                    sens = sum(y(te_m) == 1 & y_pred == 1) / max(1, sum(y(te_m) == 1));
                    spec = sum(y(te_m) == 0 & y_pred == 0) / max(1, sum(y(te_m) == 0));
                    f_accs(f_i) = (sens + spec) / 2;
                end
                single_band_accs(b, w) = mean(f_accs);
            end
        end
        
        % --- 步骤 3: 20 线程并行置换检验 ---
        X_3d_double = double(X_3d);
        null_dist = zeros(cfg.n_perm, n_win);
        
        parfor perm_i = 1:cfg.n_perm
            y_perm = y(randperm(n_gray));
            perm_acc_curve = zeros(1, n_win);
            for w = 1:n_win
                X_w = squeeze(X_3d_double(:, :, w));
                f_accs = zeros(1, n_folds);
                for f_i = 1:n_folds
                    tr_m = fold_tr_masks(:, f_i);
                    te_m = fold_te_masks(:, f_i);
                    
                    mu  = mean(X_w(tr_m, :), 1);
                    sig = std(X_w(tr_m, :), 0, 1);
                    sig(sig < 1e-6) = 1;
                    
                    X_tr_s = (X_w(tr_m, :) - mu) ./ sig;
                    X_te_s = (X_w(te_m, :) - mu) ./ sig;
                    
                    mdl = fitclinear(X_tr_s, y_perm(tr_m), 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', 0.01);
                    y_pred = predict(mdl, X_te_s);
                    
                    sens = sum(y_perm(te_m) == 1 & y_pred == 1) / max(1, sum(y_perm(te_m) == 1));
                    spec = sum(y_perm(te_m) == 0 & y_pred == 0) / max(1, sum(y_perm(te_m) == 0));
                    f_accs(f_i) = (sens + spec) / 2;
                end
                perm_acc_curve(w) = mean(f_accs);
            end
            null_dist(perm_i, :) = perm_acc_curve;
        end
        
        % --- 步骤 4: 曲线平滑处理 (按照指示平滑曲线) ---
        real_acc_joint_s = smoothdata(real_acc_joint, cfg.smooth_typ, cfg.smooth_pts);
        single_band_accs_s = zeros(size(single_band_accs));
        for b = 1:cfg.n_bands
            single_band_accs_s(b, :) = smoothdata(single_band_accs(b, :), cfg.smooth_typ, cfg.smooth_pts);
        end
        null_dist_s = smoothdata(null_dist, 2, cfg.smooth_typ, cfg.smooth_pts);
        
        % --- 步骤 5: 统计显著性计算与时间簇检验 ---
        p_pointwise = (1 + sum(null_dist_s >= real_acc_joint_s, 1)) / (1 + cfg.n_perm);
        cluster_alpha = 0.05;
        sig_mask = (p_pointwise < cluster_alpha) & (t_centers >= 0);
        
        % 寻找连续显著簇
        clusters = [];
        in_c = false;
        c_start = 1;
        for w = 1:n_win
            if sig_mask(w) && ~in_c
                in_c = true; c_start = w;
            elseif ~sig_mask(w) && in_c
                in_c = false; clusters = [clusters; c_start, w-1]; %#ok<AGROW>
            end
        end
        if in_c, clusters = [clusters; c_start, n_win]; end
        
        n_cl = size(clusters, 1);
        real_masses = zeros(n_cl, 1);
        for c_i = 1:n_cl
            idx_r = clusters(c_i, 1) : clusters(c_i, 2);
            real_masses(c_i) = sum(real_acc_joint_s(idx_r) - 0.5);
        end
        
        % 零分布最大簇质量
        null_max_mass = zeros(cfg.n_perm, 1);
        for p_i = 1:cfg.n_perm
            curve_p = null_dist_s(p_i, :);
            p_pt = (1 + sum(null_dist_s >= curve_p, 1)) / (1 + cfg.n_perm);
            p_sig = (p_pt < cluster_alpha) & (t_centers >= 0);
            
            c_list = []; in_nc = false; nc_s = 1;
            for w = 1:n_win
                if p_sig(w) && ~in_nc
                    in_nc = true; nc_s = w;
                elseif ~p_sig(w) && in_nc
                    in_nc = false; c_list = [c_list; nc_s, w-1]; %#ok<AGROW>
                end
            end
            if in_nc, c_list = [c_list; nc_s, n_win]; end
            
            if isempty(c_list)
                null_max_mass(p_i) = 0;
            else
                m_arr = zeros(size(c_list, 1), 1);
                for ncl_i = 1:size(c_list, 1)
                    idx_r = c_list(ncl_i, 1) : c_list(ncl_i, 2);
                    m_arr(ncl_i) = sum(curve_p(idx_r) - 0.5);
                end
                null_max_mass(p_i) = max(m_arr);
            end
        end
        
        cluster_pvals = zeros(n_cl, 1);
        for c_i = 1:n_cl
            cluster_pvals(c_i) = (1 + sum(null_max_mass >= real_masses(c_i))) / (1 + cfg.n_perm);
        end
        
        % --- 步骤 6: 导出单通道时程 CSV ---
        tc_tbl = table(repmat(string(sub_id), n_win, 1), repmat(string(ch_name), n_win, 1), ...
            t_centers', real_acc_joint_s', p_pointwise', mean(null_dist_s, 1)', ...
            prctile(null_dist_s, 97.5, 1)', prctile(null_dist_s, 2.5, 1)', ...
            'VariableNames', {'subject', 'channel', 'time_ms', 'acc_joint_smoothed', ...
                              'p_pointwise', 'null_mean', 'null_ci_upper', 'null_ci_lower'});
        for b = 1:cfg.n_bands
            tc_tbl.(['acc_' cfg.bands{b} '_smoothed']) = single_band_accs_s(b, :)';
        end
        tc_csv = fullfile(tc_tab_dir, sprintf('%s_%s_decoding_timecourse.csv', sub_id, ch_name));
        writetable(tc_tbl, tc_csv);
        
        % --- 步骤 7: 记录汇总条目 ---
        [max_joint, best_w_idx] = max(real_acc_joint_s);
        [max_sb, max_sb_b]     = max(max(single_band_accs_s, [], 2));
        
        rec = struct();
        rec.subject             = string(sub_id);
        rec.channel             = string(ch_name);
        rec.n_trials_gray       = n_gray;
        rec.peak_acc_joint      = max_joint;
        rec.peak_time_ms        = t_centers(best_w_idx);
        rec.peak_p_pointwise    = p_pointwise(best_w_idx);
        rec.has_sig_cluster     = any(cluster_pvals < 0.05);
        rec.best_single_band    = string(cfg.bands{max_sb_b});
        rec.best_single_band_acc= max_sb;
        for b = 1:cfg.n_bands
            rec.(['peak_acc_' cfg.bands{b}]) = max(single_band_accs_s(b, :));
        end
        summary_list = [summary_list; rec]; %#ok<AGROW>
        
        % --- 步骤 8: 严格按照要求绘图 (无方格 grid off) ---
        % 要求:
        %   1. 主标题只包括被试和电极 (如: sub001 - G13)
        %   2. 图例只保留每种频段 (Multi-Band 及 Delta~High-Gamma)
        %   3. 不加方格背景 (grid off)
        fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1200, 480]);
        
        % (1) 左子图: 时程解码曲线 (不加方格: grid off)
        subplot(1, 2, 1); hold on; grid off;
        set(gca, 'Box', 'off', 'FontSize', 11, 'LineWidth', 1.1);
        
        % 零假设置信带 (HandleVisibility off, 不进入图例)
        null_up  = prctile(null_dist_s, 97.5, 1);
        null_low = prctile(null_dist_s, 2.5, 1);
        fill([t_centers, fliplr(t_centers)], [null_low, fliplr(null_up)], ...
            col_null, 'EdgeColor', 'none', 'FaceAlpha', 0.6, 'HandleVisibility', 'off');
        
        % 参考线 (HandleVisibility off)
        yline(0.5, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.2, 'HandleVisibility', 'off');
        xline(0, '-', 'Color', [0.3, 0.3, 0.3], 'LineWidth', 1.0, 'HandleVisibility', 'off');
        
        % 显著时间簇阴影 (HandleVisibility off)
        for c_i = 1:n_cl
            if cluster_pvals(c_i) < 0.05
                c_s = clusters(c_i, 1);
                c_e = clusters(c_i, 2);
                fill([t_centers(c_s), t_centers(c_e), t_centers(c_e), t_centers(c_s)], ...
                     [0.35, 0.35, 0.85, 0.85], [1.0, 0.9, 0.7], 'EdgeColor', 'none', ...
                     'FaceAlpha', 0.35, 'HandleVisibility', 'off');
            end
        end
        
        % 显著时间点小方块标记 (HandleVisibility off)
        sig_p_idx = find(p_pointwise < 0.05 & t_centers >= 0);
        if ~isempty(sig_p_idx)
            plot(t_centers(sig_p_idx), repmat(0.38, 1, numel(sig_p_idx)), 's', ...
                'MarkerFaceColor', col_joint, 'MarkerEdgeColor', 'none', ...
                'MarkerSize', 4, 'HandleVisibility', 'off');
        end
        
        % 绘制 6 大单频段曲线 (图例保留)
        for b = 1:cfg.n_bands
            plot(t_centers, single_band_accs_s(b, :), 'Color', band_cols(b, :), ...
                'LineWidth', 1.3, 'LineStyle', '-', 'DisplayName', cfg.bands_disp{b});
        end
        
        % 绘制 Multi-Band 联合曲线 (图例保留)
        plot(t_centers, real_acc_joint_s, 'Color', col_joint, 'LineWidth', 2.6, ...
            'DisplayName', 'Multi-Band');
        
        xlim(cfg.t_range);
        ylim([0.35, 0.85]);
        xlabel('Time from stimulus onset (ms)', 'FontWeight', 'bold');
        ylabel('Balanced Accuracy', 'FontWeight', 'bold');
        
        % 图例严格只保留每种频段 (设置 Interpreter none 避免字符下划线被转义)
        legend('Location', 'northwest', 'NumColumns', 2, 'Box', 'off', 'FontSize', 9, 'Interpreter', 'none');
        
        % (2) 右子图: 频段峰值柱状对比图 (不加方格: grid off)
        subplot(1, 2, 2); hold on; grid off;
        set(gca, 'Box', 'off', 'FontSize', 11, 'LineWidth', 1.1);
        
        bar_names = [{'Multi-Band'}, cfg.bands_disp];
        bar_vals  = [max_joint, max(single_band_accs_s, [], 2)'];
        all_cols  = [col_joint; band_cols];
        
        b_h = bar(1:numel(bar_names), bar_vals, 0.65, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.9);
        for k = 1:numel(bar_names)
            b_h.CData(k, :) = all_cols(k, :);
        end
        yline(0.5, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.2);
        
        for k = 1:numel(bar_names)
            text(k, bar_vals(k) + 0.015, sprintf('%.1f%%', bar_vals(k) * 100), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
        end
        
        set(gca, 'XTick', 1:numel(bar_names), 'XTickLabel', bar_names, 'XTickLabelRotation', 30, 'TickLabelInterpreter', 'none');
        ylim([0.40, 0.85]);
        ylabel('Peak Balanced Accuracy', 'FontWeight', 'bold');
        
        % 主标题严格只包括被试和电极！
        sgtitle(sprintf('%s - %s', sub_id, ch_name), 'FontSize', 15, 'FontWeight', 'bold', 'Interpreter', 'none');
        
        fig_png = fullfile(fig_dir, sprintf('%s_%s_memory_color_decoding.png', sub_id, ch_name));
        saveas(fig, fig_png);
        close(fig);
        
        fprintf('        [完成] 耗时 %.2f 秒 | Multi-Band 峰值: %.2f%% (在 %d ms) | 最优单频段: %s (%.2f%%)\n', ...
            toc(t_elec), max_joint * 100, t_centers(best_w_idx), cfg.bands{max_sb_b}, max_sb * 100);
    end
end

%% 5. 导出总览汇总表
if ~isempty(summary_list)
    master_tbl = struct2table(summary_list);
    % 浮点数规范化为 4 位小数
    for v = 1:numel(master_tbl.Properties.VariableNames)
        vn = master_tbl.Properties.VariableNames{v};
        if isnumeric(master_tbl.(vn)), master_tbl.(vn) = round(master_tbl.(vn), 4); end
    end
    summary_table  = master_tbl;
    out_master_mat = fullfile(tab_dir, sprintf('%s_electrodes_decoding_summary.mat', cfg.target_mode));
    save(out_master_mat, 'summary_table', 'master_tbl');
    fprintf('\n========================================================================\n');
    fprintf('  【批量 Decoding 任务全部顺利完成！】\n');
    fprintf('  汇总表格已保存至: %s\n', out_master_mat);
    fprintf('  高清曲线图保存在: %s\n', fig_dir);
    fprintf('========================================================================\n');
end
