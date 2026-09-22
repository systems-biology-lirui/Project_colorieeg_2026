%% ========================================================================
% 脚本名称: C08_multisite_band_decoding_0825.m
% 功能:
%   1. 【多位点空间模式解码 (Multi-Site Decoding)】
%      以 Task 1 筛选出的总体显著电极为多维空间特征向量，评估空间群体表征。
%   2. 【5 大电极集合全面对比】
%      - Set 1: 同正集合 (Concordant Positive, 4 类别均为正色彩效应)
%      - Set 2: 同负集合 (Concordant Negative, 4 类别均为负色彩效应/抑制)
%      - Set 3: 不同向集合 (Category Biased, 类别间效应方向不一致)
%      - Set 4: 同向合一集合 (Concordant Positive + Concordant Negative)
%      - Set 5: 三种合一集合 (Task 1 全部总体显著电极 All Significant)
%   3. 【6 大生理频段独立解码】
%      Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz),
%      Beta (13-30 Hz), Low-Gamma (30-70 Hz), High-Gamma (70-150 Hz)
%   4. 【防混淆严格交叉验证】
%      Task 2 执行跨水果配对 4 折交叉验证 (Leave-One-Fruit-Pair-Out)
%   5. 【20 线程并行置换检验与时间簇质量检验 (Cluster-Mass Test)】
%      被试内标签打乱，群体级均值统计与全时程多重比较校正 (FWE < 0.05)
%   6. 【标准学术绘图与汇总表格导出】
% ========================================================================

clear; clc; close all;

%% 1. 主参数配置 (置顶直观，简写平铺，方便审阅调整)
cfg = struct();

% -------------------------------------------------------------------------
% 任务与数据配置
% -------------------------------------------------------------------------
cfg.task_name     = 'task2';              % 目标任务: 'task2' (记忆色彩) 或 'task3' (纯色色块)
cfg.cnd_state     = 'gray';               % 仅分析灰色水果试次 (排除低阶颜色干扰)

% -------------------------------------------------------------------------
% 时间窗与时程参数 (20ms 滑动窗，覆盖 -200 到 800 ms)
% -------------------------------------------------------------------------
cfg.win_len       = 20;                   % 滑动窗长 20 ms
cfg.win_step      = 20;                   % 滑动步长 20 ms (无重叠覆盖)
cfg.t_range       = [-200, 800];          % 时程范围 (ms)

% -------------------------------------------------------------------------
% 平滑与统计参数
% -------------------------------------------------------------------------
cfg.smooth_pts    = 5;                    % 平滑点数 (5点高斯平滑)
cfg.smooth_typ    = 'gaussian';           % 平滑方式
cfg.n_perm        = 100;                  % 置换检验次数 (标准置换检验)
cfg.n_workers     = 20;                   % 极速并行核数
cfg.cluster_alpha = 0.05;                 % 形成显著簇的逐点阈值 (p < 0.05)

% -------------------------------------------------------------------------
% 5 大目标电极集合与显示配置
% -------------------------------------------------------------------------
cfg.sets       = {'pos', 'neg', 'diff', 'concord', 'all_sig'};
cfg.set_names  = {'Concordant Positive', 'Concordant Negative', ...
                  'Category Biased', 'All Concordant', 'All Significant'};
cfg.set_short  = {'Pos', 'Neg', 'Diff', 'Concord', 'All'};
cfg.set_cols   = [
    0.85, 0.20, 0.20;  % Pos: 鲜红
    0.15, 0.45, 0.85;  % Neg: 钴蓝
    0.60, 0.20, 0.70;  % Diff: 罗兰紫
    0.10, 0.65, 0.35;  % Concord: 翡翠绿
    0.20, 0.20, 0.20   % All: 深灰炭黑
];

% -------------------------------------------------------------------------
% 6 大生理频段与规范显示名
% -------------------------------------------------------------------------
cfg.bands      = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.bands_disp = {'Delta (1-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-13 Hz)', ...
                  'Beta (13-30 Hz)', 'Low-Gamma (30-70 Hz)', 'High-Gamma (70-150 Hz)'};
cfg.n_bands    = numel(cfg.bands);
cfg.n_sets     = numel(cfg.sets);

% -------------------------------------------------------------------------
% 路径设置
% -------------------------------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(script_dir));
data_root  = fullfile(proj_root, 'color_analyse_0825', 'process_data_new');
res_root   = fullfile(proj_root, 'color_analyse_0825', 'result');
c04_table  = fullfile(res_root, 'tables', 'color_effects_summary.mat');

tab_dir     = fullfile(res_root, 'tables');
fig_dir     = fullfile(res_root, 'figures', 'multisite_decoding');
sub_fig_dir = fullfile(fig_dir, 'subjects');
if ~exist(tab_dir, 'dir'), mkdir(tab_dir); end
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist(sub_fig_dir, 'dir'), mkdir(sub_fig_dir); end

fprintf('========================================================================\n');
fprintf('  【C10: Task 1 总体显著电极多位点频段解码与置换检验】  \n');
fprintf('========================================================================\n');
fprintf('[+] 解码任务: %s | 时间范围: [%d, %d] ms | 窗口: %d ms | 步长: %d ms\n', ...
    cfg.task_name, cfg.t_range(1), cfg.t_range(2), cfg.win_len, cfg.win_step);
fprintf('[+] 5 大电极集合: Pos, Neg, Diff, Concord, All\n');
fprintf('[+] 6 大生理频段: Delta, Theta, Alpha, Beta, Low-Gamma, High-Gamma\n');
fprintf('[+] 置换次数: %d | 平滑点数: %d (%s)\n', cfg.n_perm, cfg.smooth_pts, cfg.smooth_typ);

%% 2. 提取 Task 1 总体显著电极的 5 大集合
if ~isfile(c04_table)
    error('未找到 C04 筛选汇总表: %s\n请先确认 C04 运行完毕！', c04_table);
end
loaded_c04 = load(c04_table);
if isfield(loaded_c04, 'all_tbl'), c04_tbl = loaded_c04.all_tbl; else, c04_tbl = loaded_c04.res_table; end
sig_tbl = c04_tbl(c04_tbl.is_significant == 1, :);

% 提取 8 名被试列表
all_subs = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008'};
n_subs   = numel(all_subs);

% 结构体存储每个被试在 5 个集合中的电极列表
sub_elecs = struct();
for s_i = 1:n_subs
    sub = all_subs{s_i};
    st = sig_tbl(strcmp(sig_tbl.subject, sub), :);
    
    % 同正 (Concordant Positive)
    m_pos = strcmp(st.concordance_type, 'Concordant_Positive');
    ch_pos = unique(st.channel(m_pos));
    
    % 同负 (Concordant Negative)
    m_neg = strcmp(st.concordance_type, 'Concordant_Negative');
    ch_neg = unique(st.channel(m_neg));
    
    % 同向合一 (Positive + Negative)
    ch_concord = unique([ch_pos; ch_neg]);
    
    % 不同向 (Category Biased, 排除与同向重叠)
    m_diff = strcmp(st.concordance_type, 'Category_Biased');
    ch_diff_raw = unique(st.channel(m_diff));
    ch_diff = setdiff(ch_diff_raw, ch_concord);
    
    % 三种合一 (全部总体显著)
    ch_all = unique(st.channel);
    
    sub_elecs.(sub).pos     = ch_pos;
    sub_elecs.(sub).neg     = ch_neg;
    sub_elecs.(sub).diff    = ch_diff;
    sub_elecs.(sub).concord = ch_concord;
    sub_elecs.(sub).all_sig = ch_all;
end

fprintf('\n[+] 8 名被试在各集合中的电极数量分布统计:\n');
fprintf('    %-8s | %-6s | %-6s | %-6s | %-8s | %-8s\n', 'Subject', 'Pos', 'Neg', 'Diff', 'Concord', 'All-Sig');
for s_i = 1:n_subs
    sub = all_subs{s_i};
    fprintf('    %-8s | %-6d | %-6d | %-6d | %-8d | %-8d\n', sub, ...
        numel(sub_elecs.(sub).pos), numel(sub_elecs.(sub).neg), ...
        numel(sub_elecs.(sub).diff), numel(sub_elecs.(sub).concord), ...
        numel(sub_elecs.(sub).all_sig));
end

%% 3. 启动并行计算池
curr_pool = gcp('nocreate');
if isempty(curr_pool)
    fprintf('\n[+] 正在启动 %d 线程并行池 ...\n', cfg.n_workers);
    parpool('local', cfg.n_workers);
else
    fprintf('\n[+] 并行池已就绪 (NumWorkers = %d)\n', curr_pool.NumWorkers);
end

%% 4. 被试级多位点特征提取与分类解码
% 时间窗格定义
t_centers = cfg.t_range(1) : cfg.win_step : cfg.t_range(2);
n_win     = numel(t_centers);

% 4 折跨水果配对 (Leave-One-Fruit-Pair-Out)
folds_t2 = {
    {'strawberry', 'cabbage'}, {'watermelon', 'kiwi'};
    {'strawberry', 'kiwi'},    {'watermelon', 'cabbage'};
    {'watermelon', 'cabbage'}, {'strawberry', 'kiwi'};
    {'watermelon', 'kiwi'},    {'strawberry', 'cabbage'}
};
n_folds_t2 = size(folds_t2, 1);

% 预分配结构体存储 8 个被试的结果
% dim: sub_real_acc(n_subs, n_sets, n_bands, n_win)
% dim: sub_null_acc(n_subs, n_sets, n_bands, n_perm, n_win)
sub_real_acc = zeros(n_subs, cfg.n_sets, cfg.n_bands, n_win);
sub_null_acc = zeros(n_subs, cfg.n_sets, cfg.n_bands, cfg.n_perm, n_win);

% 检查点缓存目录 (防止意外中断重复耗时计算)
cache_dir = fullfile(tab_dir, 'cache_c10');
if ~exist(cache_dir, 'dir'), mkdir(cache_dir); end

% 预分配单被试显著性汇总记录
sub_summary_rows = struct([]);

for s_i = 1:n_subs
    sub_id = all_subs{s_i};
    sub_cache_file = fullfile(cache_dir, sprintf('cache_%s_%s.mat', cfg.task_name, sub_id));
    sub_fig_png    = fullfile(sub_fig_dir, sprintf('%s_multisite_decoding.png', sub_id));
    
    if isfile(sub_cache_file)
        fprintf('\n------------------------------------------------------------------------\n');
        fprintf('>>> [被试 %d/%d: %s] 发现已有缓存结果，直接快速载入！\n', s_i, n_subs, sub_id);
        fprintf('------------------------------------------------------------------------\n');
        cached = load(sub_cache_file, 'sub_real_i', 'sub_null_i');
        sub_real_acc(s_i, :, :, :) = cached.sub_real_i;
        sub_null_acc(s_i, :, :, :, :) = cached.sub_null_i;
    else
        fprintf('\n------------------------------------------------------------------------\n');
        fprintf('>>> [被试 %d/%d: %s] 正在载入数据并构建多位点时频特征 ...\n', s_i, n_subs, sub_id);
        fprintf('------------------------------------------------------------------------\n');
    
    mat_file = fullfile(data_root, sub_id, sprintf('%s_multiband_epoched.mat', cfg.task_name));
    if ~isfile(mat_file)
        warning('未找到被试 %s 数据: %s，跳过。', sub_id, mat_file);
        continue;
    end
    
    t_load = tic;
    mat_data = load(mat_file, 'epoched_data');
    ep = mat_data.epoched_data;
    time_ms = ep.time_ms(:)';
    ti = ep.trial_info;
    n_chans = numel(ep.channels);
    
    % 试次筛选与标签
    if strcmp(cfg.task_name, 'task2')
        gray_mask = strcmp(ti.state, cfg.cnd_state);
        ti_use = ti(gray_mask, :);
        n_trials = height(ti_use);
        y = zeros(n_trials, 1);
        y(strcmp(ti_use.memory_color, 'red')) = 1; % red=1, green=0
        
        fold_tr = false(n_trials, n_folds_t2);
        fold_te = false(n_trials, n_folds_t2);
        for f = 1:n_folds_t2
            fold_tr(:, f) = ismember(ti_use.fruit, folds_t2{f, 1});
            fold_te(:, f) = ismember(ti_use.fruit, folds_t2{f, 2});
        end
        n_cv_folds = n_folds_t2;
    else
        rg_mask = ismember(ti.color, {'red', 'green'});
        ti_use = ti(rg_mask, :);
        n_trials = height(ti_use);
        y = zeros(n_trials, 1);
        y(strcmp(ti_use.color, 'red')) = 1;
        cv_part = cvpartition(y, 'KFold', 5);
        n_cv_folds = 5;
        fold_tr = false(n_trials, n_cv_folds);
        fold_te = false(n_trials, n_cv_folds);
        for f = 1:n_cv_folds
            fold_tr(:, f) = training(cv_part, f);
            fold_te(:, f) = test(cv_part, f);
        end
    end
    
    % 快速预计算该被试 6 大频段在 51 个时间窗内的窗口均值特征
    % win_feat.(band) = [n_trials x n_chans x n_win]
    win_feat = struct();
    for b = 1:cfg.n_bands
        b_name = cfg.bands{b};
        raw_b = ep.(b_name);
        if strcmp(cfg.task_name, 'task2')
            raw_b = raw_b(gray_mask, :, :);
        else
            raw_b = raw_b(rg_mask, :, :);
        end
        
        wf = zeros(n_trials, n_chans, n_win, 'single');
        for w = 1:n_win
            t_c = t_centers(w);
            t_m = (time_ms >= (t_c - cfg.win_len/2)) & (time_ms < (t_c + cfg.win_len/2));
            wf(:, :, w) = mean(raw_b(:, :, t_m), 3);
        end
        win_feat.(b_name) = wf;
    end
    fprintf('    [+] 特征预计算完毕 (耗时 %.2f 秒)，有效试次数: %d (Red: %d, Green: %d)\n', ...
        toc(t_load), n_trials, sum(y==1), sum(y==0));
    
    % 循环 5 大电极集合
    for set_i = 1:cfg.n_sets
        set_key = cfg.sets{set_i};
        ch_list = sub_elecs.(sub_id).(set_key);
        [~, ch_idx] = ismember(ch_list, ep.channels);
        ch_idx(ch_idx == 0) = [];
        n_sub_elecs = numel(ch_idx);
        
        if n_sub_elecs == 0
            warning('被试 %s 在集合 [%s] 中没有电极，跳过。', sub_id, set_key);
            continue;
        end
        
        % 循环 6 大频段执行多位点解码与置换检验
        for b = 1:cfg.n_bands
            b_name = cfg.bands{b};
            t_dec = tic;
            
            % 提取多位点特征矩阵: [n_trials x n_sub_elecs x n_win]
            X_all = double(win_feat.(b_name)(:, ch_idx, :));
            
            % 1. 真实多位点解码
            real_curve = zeros(1, n_win);
            for w = 1:n_win
                X_w = squeeze(X_all(:, :, w));
                f_accs = zeros(1, n_cv_folds);
                for f = 1:n_cv_folds
                    tr_m = fold_tr(:, f);
                    te_m = fold_te(:, f);
                    
                    mu  = mean(X_w(tr_m, :), 1);
                    sig = std(X_w(tr_m, :), 0, 1);
                    sig(sig < 1e-6) = 1;
                    
                    X_tr_s = (X_w(tr_m, :) - mu) ./ sig;
                    X_te_s = (X_w(te_m, :) - mu) ./ sig;
                    
                    mdl = fitclinear(X_tr_s, y(tr_m), 'Learner', 'svm', ...
                        'Regularization', 'ridge', 'Lambda', 0.01, 'Solver', 'dual');
                    y_pred = predict(mdl, X_te_s);
                    
                    sens = sum(y(te_m) == 1 & y_pred == 1) / max(1, sum(y(te_m) == 1));
                    spec = sum(y(te_m) == 0 & y_pred == 0) / max(1, sum(y(te_m) == 0));
                    f_accs(f) = (sens + spec) / 2;
                end
                real_curve(w) = mean(f_accs);
            end
            sub_real_acc(s_i, set_i, b, :) = real_curve;
            
            % 2. 20 线程极速并行置换检验
            null_mat = zeros(cfg.n_perm, n_win);
            parfor perm_i = 1:cfg.n_perm
                y_perm = y(randperm(n_trials));
                p_curve = zeros(1, n_win);
                for w = 1:n_win
                    X_w = squeeze(X_all(:, :, w));
                    f_accs = zeros(1, n_cv_folds);
                    for f = 1:n_cv_folds
                        tr_m = fold_tr(:, f);
                        te_m = fold_te(:, f);
                        
                        mu  = mean(X_w(tr_m, :), 1);
                        sig = std(X_w(tr_m, :), 0, 1);
                        sig(sig < 1e-6) = 1;
                        
                        X_tr_s = (X_w(tr_m, :) - mu) ./ sig;
                        X_te_s = (X_w(te_m, :) - mu) ./ sig;
                        
                        mdl = fitclinear(X_tr_s, y_perm(tr_m), 'Learner', 'svm', ...
                            'Regularization', 'ridge', 'Lambda', 0.01, 'Solver', 'dual');
                        y_pred = predict(mdl, X_te_s);
                        
                        sens = sum(y_perm(te_m) == 1 & y_pred == 1) / max(1, sum(y_perm(te_m) == 1));
                        spec = sum(y_perm(te_m) == 0 & y_pred == 0) / max(1, sum(y_perm(te_m) == 0));
                        f_accs(f) = (sens + spec) / 2;
                    end
                    p_curve(w) = mean(f_accs);
                end
                null_mat(perm_i, :) = p_curve;
            end
            sub_null_acc(s_i, set_i, b, :, :) = null_mat;
            
            [max_acc, max_idx] = max(real_curve);
            fprintf('    [集合: %-7s (%2d 电极) | 频段: %-10s] 耗时: %5.2f s | 峰值: %5.2f%% (在 %3d ms)\n', ...
                set_key, n_sub_elecs, b_name, toc(t_dec), max_acc * 100, t_centers(max_idx));
        end
    end
    
        % 保存当前被试的计算结果到缓存，避免中断后重跑
        sub_real_i = sub_real_acc(s_i, :, :, :);
        sub_null_i = sub_null_acc(s_i, :, :, :, :);
        save(sub_cache_file, 'sub_real_i', 'sub_null_i');
        fprintf('    [+] 被试 %s 结果已成功写入缓存: %s\n', sub_id, sub_cache_file);
    end
    
    % ---------------------------------------------------------------------
    % 即时生成该被试 6 频段 × 5 集合解码时程图 (含单被试置换显著性检验横条)
    % ---------------------------------------------------------------------
    f_sub = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1450, 840]);
    for b_idx = 1:cfg.n_bands
        subplot(2, 3, b_idx);
        hold on; grid off;
        set(gca, 'Box', 'off', 'FontSize', 10, 'LineWidth', 1.0);
        
        % 绘制 95% 置换检验零分布灰色阴影
        sub_null_b = reshape(sub_null_acc(s_i, :, b_idx, :, :), cfg.n_sets * cfg.n_perm, n_win);
        sub_null_b_sm = smoothdata(sub_null_b, 2, cfg.smooth_typ, cfg.smooth_pts);
        ci_up  = prctile(sub_null_b_sm, 97.5, 1);
        ci_low = prctile(sub_null_b_sm, 2.5, 1);
        fill([t_centers, fliplr(t_centers)], [ci_up, fliplr(ci_low)], ...
            [0.85, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
        
        yline(0.50, '--', 'Color', [0.55, 0.55, 0.55], 'LineWidth', 1.0, 'HandleVisibility', 'off');
        xline(0, '-', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 0.8, 'HandleVisibility', 'off');
        
        h_lines = [];
        for s_idx = 1:cfg.n_sets
            n_elecs = numel(sub_elecs.(sub_id).(cfg.sets{s_idx}));
            if n_elecs == 0, continue; end
            
            c_raw = reshape(sub_real_acc(s_i, s_idx, b_idx, :), 1, n_win);
            c_sm  = smoothdata(c_raw, 2, cfg.smooth_typ, cfg.smooth_pts);
            lw = 1.8;
            if s_idx == 5, lw = 2.4; end
            hl = plot(t_centers, c_sm, 'Color', cfg.set_cols(s_idx, :), 'LineWidth', lw, ...
                'DisplayName', sprintf('%s (N=%d)', cfg.set_short{s_idx}, n_elecs));
            h_lines = [h_lines, hl]; %#ok<AGROW>
            
            % --- 单被试内置换显著性检验 (Permutation & Cluster Mass) ---
            null_raw = reshape(sub_null_acc(s_i, s_idx, b_idx, :, :), cfg.n_perm, n_win);
            null_sm  = smoothdata(null_raw, 2, cfg.smooth_typ, cfg.smooth_pts);
            
            % 逐点显著性 (单侧检验: real >= null)
            p_pt_sub  = (1 + sum(null_sm >= c_sm, 1)) / (1 + cfg.n_perm);
            sig_m_sub = (p_pt_sub < cfg.cluster_alpha) & (t_centers >= 0);
            
            % 识别连续时间簇
            s_cls = []; in_sc = false; sc_start = 1;
            for w = 1:n_win
                if sig_m_sub(w) && ~in_sc
                    in_sc = true; sc_start = w;
                elseif ~sig_m_sub(w) && in_sc
                    in_sc = false; s_cls = [s_cls; sc_start, w-1]; %#ok<AGROW>
                end
            end
            if in_sc, s_cls = [s_cls; sc_start, n_win]; end
            
            % 计算真实簇质量
            n_sc = size(s_cls, 1);
            s_real_mass = zeros(n_sc, 1);
            for sc_i = 1:n_sc
                s_real_mass(sc_i) = sum(c_sm(s_cls(sc_i, 1) : s_cls(sc_i, 2)) - 0.50);
            end
            
            % 计算零分布最大簇质量
            s_null_max = zeros(cfg.n_perm, 1);
            for p_i = 1:cfg.n_perm
                p_curve = null_sm(p_i, :);
                p_pt_null = (1 + sum(null_sm >= p_curve, 1)) / (1 + cfg.n_perm);
                p_sig_null = (p_pt_null < cfg.cluster_alpha) & (t_centers >= 0);
                
                c_null = []; in_nc = false; nc_s = 1;
                for w = 1:n_win
                    if p_sig_null(w) && ~in_nc
                        in_nc = true; nc_s = w;
                    elseif ~p_sig_null(w) && in_nc
                        in_nc = false; c_null = [c_null; nc_s, w-1]; %#ok<AGROW>
                    end
                end
                if in_nc, c_null = [c_null; nc_s, n_win]; end
                
                if isempty(c_null)
                    s_null_max(p_i) = 0;
                else
                    m_arr = zeros(size(c_null, 1), 1);
                    for ncl_i = 1:size(c_null, 1)
                        m_arr(ncl_i) = sum(p_curve(c_null(ncl_i, 1) : c_null(ncl_i, 2)) - 0.50);
                    end
                    s_null_max(p_i) = max([0; m_arr]);
                end
            end
            
            % 计算每个时间簇的置换 p 值
            s_pvals = zeros(n_sc, 1);
            for sc_i = 1:n_sc
                s_pvals(sc_i) = (1 + sum(s_null_max >= s_real_mass(sc_i))) / (1 + cfg.n_perm);
            end
            
            % 在子图底部绘制显著性横条 (置于 0.406 ~ 0.422)
            y_bar = 0.422 - (s_idx - 1) * 0.004;
            for sc_i = 1:n_sc
                t_s = t_centers(s_cls(sc_i, 1));
                t_e = t_centers(s_cls(sc_i, 2));
                dur = t_e - t_s;
                if s_pvals(sc_i) < 0.05
                    plot([t_s, t_e], [y_bar, y_bar], 'LineWidth', 3.2, ...
                        'Color', cfg.set_cols(s_idx, :), 'HandleVisibility', 'off');
                elseif dur >= 40
                    plot([t_s, t_e], [y_bar, y_bar], 'LineWidth', 1.5, 'LineStyle', ':', ...
                        'Color', cfg.set_cols(s_idx, :), 'HandleVisibility', 'off');
                end
            end
            
            % 记录单被试汇总结果
            [peak_v, best_w] = max(c_sm);
            has_sig = any(s_pvals < 0.05);
            s_rec = struct();
            s_rec.subject         = string(sub_id);
            s_rec.set_key         = string(cfg.sets{s_idx});
            s_rec.set_name        = string(cfg.set_names{s_idx});
            s_rec.band_name       = string(cfg.bands{b_idx});
            s_rec.n_electrodes    = n_elecs;
            s_rec.peak_acc        = peak_v;
            s_rec.peak_time_ms    = t_centers(best_w);
            s_rec.has_sig_cluster = has_sig;
            if has_sig
                first_sig = find(s_pvals < 0.05, 1);
                s_rec.cluster_start_ms = t_centers(s_cls(first_sig, 1));
                s_rec.cluster_end_ms   = t_centers(s_cls(first_sig, 2));
                s_rec.min_cluster_p    = min(s_pvals);
            else
                s_rec.cluster_start_ms = NaN;
                s_rec.cluster_end_ms   = NaN;
                s_rec.min_cluster_p    = NaN;
            end
            sub_summary_rows = [sub_summary_rows; s_rec]; %#ok<AGROW>
        end
        xlim(cfg.t_range);
        ylim([0.40, 0.70]);
        xlabel('Time (ms)', 'FontSize', 10);
        ylabel('Balanced Accuracy', 'FontSize', 10);
        title(cfg.bands{b_idx}, 'FontSize', 12, 'FontWeight', 'bold');
        if b_idx == 1
            legend(h_lines, 'Location', 'northwest', 'Box', 'off', 'FontSize', 8.5);
        end
    end
    sgtitle(sprintf('Subject %s', sub_id), 'FontSize', 14, 'FontWeight', 'bold');
    if isfile(sub_fig_png)
        try, delete(sub_fig_png); catch, end
    end
    try
        exportgraphics(f_sub, sub_fig_png, 'Resolution', 200);
    catch
        saveas(f_sub, sub_fig_png);
    end
    close(f_sub);
    fprintf('    [+] 被试 %s 单被试图谱(含置换检验横条)已即时生成: %s\n', sub_id, sub_fig_png);
end

% 整理单被试统计汇总表
if ~isempty(sub_summary_rows)
    sub_summary_tbl = struct2table(sub_summary_rows);
end

%% 5. 群体水平 (Group-Level) 统计与 Cluster-Mass 显著性检验
fprintf('\n========================================================================\n');
fprintf('  【计算 8 名被试群体平均解码时程与基于 Cluster-Mass 的全时程统计检验】  \n');
fprintf('========================================================================\n');

summary_rows = struct([]);
tc_export_rows = struct([]);

% 群体平滑曲线存储
group_real_smoothed = zeros(cfg.n_sets, cfg.n_bands, n_win);
group_null_smoothed = zeros(cfg.n_sets, cfg.n_bands, cfg.n_perm, n_win);
group_sig_masks     = false(cfg.n_sets, cfg.n_bands, n_win);
group_cluster_info  = cell(cfg.n_sets, cfg.n_bands);

for set_i = 1:cfg.n_sets
    set_key  = cfg.sets{set_i};
    set_name = cfg.set_names{set_i};
    
    for b = 1:cfg.n_bands
        b_name = cfg.bands{b};
        
        % 8 名被试求群体平均 (严格保持行向量与矩阵形状)
        g_real_raw = reshape(mean(sub_real_acc(:, set_i, b, :), 1), 1, n_win);               % [1 x n_win]
        g_null_raw = reshape(mean(sub_null_acc(:, set_i, b, :, :), 1), cfg.n_perm, n_win);    % [n_perm x n_win]
        
        % 高斯平滑处理 (沿着第 2 维时间轴平滑)
        g_real_s = smoothdata(g_real_raw, 2, cfg.smooth_typ, cfg.smooth_pts);
        g_null_s = smoothdata(g_null_raw, 2, cfg.smooth_typ, cfg.smooth_pts);
        
        group_real_smoothed(set_i, b, :) = reshape(g_real_s, [1, 1, n_win]);
        group_null_smoothed(set_i, b, :, :) = reshape(g_null_s, [1, 1, cfg.n_perm, n_win]);
        
        % 逐点显著性检验 (单侧检验: real >= null)
        % g_null_s: [100 x 51], g_real_s: [1 x 51], 自动行广播对比
        p_pt = (1 + sum(g_null_s >= g_real_s, 1)) / (1 + cfg.n_perm);
        sig_mask = (p_pt < cfg.cluster_alpha) & (t_centers >= 0);
        group_sig_masks(set_i, b, :) = reshape(sig_mask, [1, 1, n_win]);
        
        % 寻找连续时间簇
        clusters = [];
        in_c = false;
        c_s = 1;
        for w = 1:n_win
            if sig_mask(w) && ~in_c
                in_c = true; c_s = w;
            elseif ~sig_mask(w) && in_c
                in_c = false; clusters = [clusters; c_s, w-1]; %#ok<AGROW>
            end
        end
        if in_c, clusters = [clusters; c_s, n_win]; end
        
        % 计算真实簇质量 (Cluster Mass: 超出 50% 机会水平的积分)
        n_cl = size(clusters, 1);
        real_masses = zeros(n_cl, 1);
        for c_i = 1:n_cl
            idx_range = clusters(c_i, 1) : clusters(c_i, 2);
            real_masses(c_i) = sum(g_real_s(idx_range) - 0.50);
        end
        
        % 计算零分布的最大簇质量
        null_max_mass = zeros(cfg.n_perm, 1);
        for p_i = 1:cfg.n_perm
            curve_p = g_null_s(p_i, :);
            p_pt_null = (1 + sum(g_null_s >= curve_p, 1)) / (1 + cfg.n_perm);
            p_sig_null = (p_pt_null < cfg.cluster_alpha) & (t_centers >= 0);
            
            c_null = []; in_nc = false; nc_s = 1;
            for w = 1:n_win
                if p_sig_null(w) && ~in_nc
                    in_nc = true; nc_s = w;
                elseif ~p_sig_null(w) && in_nc
                    in_nc = false; c_null = [c_null; nc_s, w-1]; %#ok<AGROW>
                end
            end
            if in_nc, c_null = [c_null; nc_s, n_win]; end
            
            if isempty(c_null)
                null_max_mass(p_i) = 0;
            else
                m_arr = zeros(size(c_null, 1), 1);
                for ncl_i = 1:size(c_null, 1)
                    idx_r = c_null(ncl_i, 1) : c_null(ncl_i, 2);
                    m_arr(ncl_i) = sum(curve_p(idx_r) - 0.50);
                end
                null_max_mass(p_i) = max([0; m_arr]);
            end
        end
        
        % 计算每个簇的 FWE 校正 p 值
        cl_pvals = zeros(n_cl, 1);
        for c_i = 1:n_cl
            cl_pvals(c_i) = (1 + sum(null_max_mass >= real_masses(c_i))) / (1 + cfg.n_perm);
        end
        
        group_cluster_info{set_i, b} = struct('clusters', clusters, 'pvals', cl_pvals);
        
        % 记录汇总指标
        [peak_val, best_w] = max(g_real_s);
        has_sig_cl = any(cl_pvals < 0.05);
        cl_start_ms = NaN; cl_end_ms = NaN; min_cl_p = NaN;
        if has_sig_cl
            sig_cl_idx = find(cl_pvals < 0.05, 1);
            cl_start_ms = t_centers(clusters(sig_cl_idx, 1));
            cl_end_ms   = t_centers(clusters(sig_cl_idx, 2));
            min_cl_p    = min(cl_pvals);
        end
        
        rec = struct();
        rec.task_name        = string(cfg.task_name);
        rec.set_key          = string(set_key);
        rec.set_name         = string(set_name);
        rec.band_name        = string(b_name);
        rec.peak_acc         = peak_val;
        rec.peak_time_ms     = t_centers(best_w);
        rec.peak_p_pointwise = p_pt(best_w);
        rec.has_sig_cluster  = has_sig_cl;
        rec.cluster_start_ms = cl_start_ms;
        rec.cluster_end_ms   = cl_end_ms;
        rec.cluster_p_fwe    = min_cl_p;
        summary_rows = [summary_rows; rec]; %#ok<AGROW>
        
        % 导出逐点时程数据
        null_ci_up  = prctile(g_null_s, 97.5, 1);
        null_ci_low = prctile(g_null_s, 2.5, 1);
        null_mean   = mean(g_null_s, 1);
        for w = 1:n_win
            tc_rec = struct();
            tc_rec.task_name     = string(cfg.task_name);
            tc_rec.set_key       = string(set_key);
            tc_rec.band_name     = string(b_name);
            tc_rec.time_ms       = t_centers(w);
            tc_rec.accuracy      = g_real_s(w);
            tc_rec.p_pointwise   = p_pt(w);
            tc_rec.null_mean     = null_mean(w);
            tc_rec.null_ci_upper = null_ci_up(w);
            tc_rec.null_ci_lower = null_ci_low(w);
            tc_rec.is_sig_point  = sig_mask(w);
            tc_export_rows = [tc_export_rows; tc_rec]; %#ok<AGROW>
        end
        
        fprintf('  [集合: %-7s | 频段: %-10s] 群体峰值: %5.2f%% (在 %3d ms) | 显著簇: %d (FWE p = %.4f)\n', ...
            set_key, b_name, peak_val * 100, t_centers(best_w), has_sig_cl, min_cl_p);
    end
end

% 导出结果文件 (.mat 纯净结构体存储)
summary_tbl = struct2table(summary_rows);
tc_tbl      = struct2table(tc_export_rows);

% 浮点数截断规范化 (4 位小数)
for v = 1:numel(summary_tbl.Properties.VariableNames)
    vn = summary_tbl.Properties.VariableNames{v};
    if isnumeric(summary_tbl.(vn)), summary_tbl.(vn) = round(summary_tbl.(vn), 4); end
end
for v = 1:numel(tc_tbl.Properties.VariableNames)
    vn = tc_tbl.Properties.VariableNames{v};
    if isnumeric(tc_tbl.(vn)), tc_tbl.(vn) = round(tc_tbl.(vn), 4); end
end

res_mat = fullfile(tab_dir, sprintf('multisite_decoding_%s_results.mat', cfg.task_name));
ws_mat  = fullfile(tab_dir, sprintf('multisite_decoding_%s_workspace.mat', cfg.task_name));

summary = summary_tbl;
timecourses = tc_tbl;
if exist('sub_summary_tbl', 'var') && ~isempty(sub_summary_tbl)
    for v = 1:numel(sub_summary_tbl.Properties.VariableNames)
        vn = sub_summary_tbl.Properties.VariableNames{v};
        if isnumeric(sub_summary_tbl.(vn)), sub_summary_tbl.(vn) = round(sub_summary_tbl.(vn), 4); end
    end
    subjects_summary = sub_summary_tbl;
    save(res_mat, 'summary', 'timecourses', 'subjects_summary');
else
    save(res_mat, 'summary', 'timecourses');
end

save(ws_mat, 'cfg', 'sub_real_acc', 'sub_null_acc', 'group_real_smoothed', ...
     'group_null_smoothed', 'group_sig_masks', 'group_cluster_info', 't_centers', '-v7.3');

fprintf('\n[+] 结果数据保存成功:\n    - 结构化MAT: %s\n    - 完整工作区MAT: %s\n', ...
    res_mat, ws_mat);

%% 6. 生成学术级 6 频段多子图与峰值对比大图 (300 DPI 印刷级)
fprintf('\n========================================================================\n');
fprintf('  【正在生成 6 大生理频段时程曲线大图与 5 大集合性能对比图】  \n');
fprintf('========================================================================\n');

% -------------------------------------------------------------------------
% 图 1: 6 大频段群体解码时程大图 (2 行 3 列，无网格，极简自然期刊风格)
% -------------------------------------------------------------------------
fig1 = figure('Visible', 'off', 'Color', 'w', 'Position', [80, 80, 1500, 880]);

for b = 1:cfg.n_bands
    subplot(2, 3, b);
    hold on; grid off;
    set(gca, 'Box', 'off', 'FontSize', 11, 'LineWidth', 1.1, ...
        'XColor', [0.15, 0.15, 0.15], 'YColor', [0.15, 0.15, 0.15]);
    
    % 绘制 95% 置换检验零分布灰色阴影
    g_null_b = reshape(group_null_smoothed(:, b, :, :), cfg.n_sets * cfg.n_perm, n_win);
    ci_up  = prctile(g_null_b, 97.5, 1);
    ci_low = prctile(g_null_b, 2.5, 1);
    fill([t_centers, fliplr(t_centers)], [ci_up, fliplr(ci_low)], ...
        [0.85, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % 绘制参考线 (机会水平 50% 与 刺激呈现 0ms)
    yline(0.50, '--', 'Color', [0.55, 0.55, 0.55], 'LineWidth', 1.0, 'HandleVisibility', 'off');
    xline(0, '-', 'Color', [0.35, 0.35, 0.35], 'LineWidth', 0.8, 'HandleVisibility', 'off');
    
    % 遍历 5 个电极集合绘制曲线
    h_lines = zeros(cfg.n_sets, 1);
    for set_i = 1:cfg.n_sets
        c_curve = squeeze(group_real_smoothed(set_i, b, :));
        c_col   = cfg.set_cols(set_i, :);
        
        if set_i == 5
            lw = 2.4; % All-Sig 稍粗更醒目
        else
            lw = 1.8;
        end
        h_lines(set_i) = plot(t_centers, c_curve, 'Color', c_col, 'LineWidth', lw, ...
            'DisplayName', cfg.set_names{set_i});
        
        % 显著时间簇高亮横条 (置于底部 y = 0.406 ~ 0.422)
        cl_info = group_cluster_info{set_i, b};
        if ~isempty(cl_info) && ~isempty(cl_info.clusters)
            for c_k = 1:size(cl_info.clusters, 1)
                if cl_info.pvals(c_k) < 0.05
                    c_start = t_centers(cl_info.clusters(c_k, 1));
                    c_end   = t_centers(cl_info.clusters(c_k, 2));
                    y_bar   = 0.422 - (set_i - 1) * 0.004;
                    plot([c_start, c_end], [y_bar, y_bar], 'LineWidth', 3.2, ...
                        'Color', c_col, 'HandleVisibility', 'off');
                end
            end
        end
    end
    
    xlim(cfg.t_range);
    ylim([0.40, 0.70]);
    xlabel('Time (ms)', 'FontWeight', 'bold', 'FontSize', 11);
    ylabel('Balanced Accuracy', 'FontWeight', 'bold', 'FontSize', 11);
    title(cfg.bands{b}, 'FontSize', 13, 'FontWeight', 'bold');
    
    if b == 1
        legend(h_lines, cfg.set_names, 'Location', 'northwest', 'Box', 'off', ...
            'FontSize', 9.0, 'Interpreter', 'none');
    end
end

sgtitle('Group Decoding (Task 2)', 'FontSize', 15, 'FontWeight', 'bold');

fig1_png = fullfile(fig_dir, sprintf('multisite_decoding_%s_timecourses_6bands.png', cfg.task_name));
if isfile(fig1_png), try, delete(fig1_png); catch, end; end
try
    exportgraphics(fig1, fig1_png, 'Resolution', 300);
catch
    saveas(fig1, fig1_png);
end
close(fig1);
fprintf('[+] 6 频段多子图时程大图已保存: %s\n', fig1_png);

% -------------------------------------------------------------------------
% 图 2: 5 大集合 × 6 大频段 峰值解码平衡准确率对比柱状图
% -------------------------------------------------------------------------
fig2 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1200, 560]);
hold on; grid off;
set(gca, 'Box', 'off', 'FontSize', 12, 'LineWidth', 1.2, ...
    'XColor', [0.15, 0.15, 0.15], 'YColor', [0.15, 0.15, 0.15]);

% 提取 6 频段 × 5 集合的峰值矩阵
peak_mat = zeros(cfg.n_bands, cfg.n_sets);
for b = 1:cfg.n_bands
    for set_i = 1:cfg.n_sets
        peak_mat(b, set_i) = max(squeeze(group_real_smoothed(set_i, b, :)));
    end
end

% 分组柱状图
b_bars = bar(1:cfg.n_bands, peak_mat, 0.85, 'EdgeColor', 'k', 'LineWidth', 0.8);
for set_i = 1:cfg.n_sets
    b_bars(set_i).FaceColor = cfg.set_cols(set_i, :);
end

yline(0.50, '--', 'Color', [0.65, 0.65, 0.65], 'LineWidth', 1.2, 'HandleVisibility', 'off');
ylim([0.45, 0.60]);
set(gca, 'XTick', 1:cfg.n_bands, 'XTickLabel', cfg.bands, 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Peak Balanced Accuracy', 'FontSize', 13, 'FontWeight', 'bold');
title('Peak Decoding Accuracy across Bands', 'FontSize', 14, 'FontWeight', 'bold');
legend(cfg.set_names, 'Location', 'northeast', 'Box', 'off', 'FontSize', 10.5);

% 在柱状图上方标注最高数值
for b = 1:cfg.n_bands
    for set_i = 1:cfg.n_sets
        val = peak_mat(b, set_i);
        x_offset = b_bars(set_i).XEndPoints(b);
        text(x_offset, val + 0.005, sprintf('%.1f%%', val * 100), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold', ...
            'Rotation', 45);
    end
end

fig2_png = fullfile(fig_dir, sprintf('multisite_decoding_%s_peak_comparison.png', cfg.task_name));
if isfile(fig2_png), try, delete(fig2_png); catch, end; end
try
    exportgraphics(fig2, fig2_png, 'Resolution', 300);
catch
    saveas(fig2, fig2_png);
end
close(fig2);
fprintf('[+] 峰值性能对比柱状图已保存: %s\n', fig2_png);

fprintf('\n========================================================================\n');
fprintf('  【C10 多位点频段解码与置换检验分析全部圆满完成！】\n');
fprintf('========================================================================\n');
