%% ========================================================================
% 脚本名称: C09_cross_decoding_task3_to_task2_0825.m
% 功能:
%   1. 【Task 3 (纯色块) -> Task 2 (灰度记忆色) 跨任务跨表征神经解码 (纯 MATLAB 实现)】
%   2. 【电极筛选基底: Task 1 独立筛选电极，杜绝双重浸入偏差】
%      - cfg.target_mode = 'concordant': 读取 C04 四类别同向显著电极 (默认推荐, 157个)
%      - cfg.target_mode = 'all_sig'   : 读取 C04 全部总体显著电极 (229个)
%      - cfg.target_mode = 'custom'    : 指定特定重点电极 (如: sub001-G13, sub007-C4)
%   3. 【按被试批量极速加载】
%      每个被试的 Task 3 与 Task 2 数据仅载入一次，大幅降低 I/O 开销与内存占用
%   4. 【零信息泄露严格外推与标准化】
%      - 训练集: Task 3 全部 120 个红绿试次 (消除特定色块偶然特征，信噪比最高)
%      - 测试集: Task 2 全部 240 个灰度水果试次 (无物理颜色，纯自上而下记忆色)
%      - 特征标准化: 严格基于 Task 3 计算均值与方差，零泄露缩放 Task 2
%   5. 【两大时间维度分析】
%      - 1D 对角线同步解码: 考察同一潜伏期下的表征重合，执行 200 次置换检验与时间簇校正
%      - 2D TGM 时间泛化矩阵: 51x51 网格考察时序延迟重放，仅输出经验准确率热力图 (免置换检验，秒级出图)
%   6. 【多频段联合与 6 单频段消融】
%      同时计算 Multi-Band 联合特征与 6 个单频段独立特征
%   7. 【规范学术可视化绘图】
%      - 1:1 左右双子图排版 (左图: 1D 对角线时程 + 显著簇; 右图: 2D TGM 热力图)
%      - 无背景方格 (grid off)，遵循 Nature 规范配色
%   8. 【全量结果与汇总表格导出】
%      输出逐通道时程数据、汇总统计表与高分辨率图谱
% ========================================================================

clear; clc; close all;

%% 1. 主参数配置 (置顶直观，简写平铺，方便审阅调整)
cfg = struct();

% -------------------------------------------------------------------------
% 目标电极选择模式:
%   'concordant'     : 自动读取 C04 筛选出的全部【总体显著且四类别同向】电极 (157个, 默认推荐)
%   'all_sig'        : 自动读取 C04 筛选出的全部【总体显著】电极 (229个)
%   'custom'         : 指定特定关注电极 (如: sub001-G13, sub007-C4)
% -------------------------------------------------------------------------
cfg.target_mode = 'custom'; 

% 当 cfg.target_mode = 'custom' 时生效:
cfg.custom_subs    = {'sub001', 'sub007'};
cfg.custom_elecs   = {'G13', 'C4'};

% 测试集条件配置
cfg.test_state     = 'gray';                         % 仅分析灰度水果 (无物理颜色偏倚)
cfg.split_mode     = 'full';                         % 'full': 100%全量外推 (默认推荐)

% 时间窗与时程参数
cfg.win_len        = 20;                             % 滑动窗长 20 ms
cfg.win_step       = 20;                             % 滑动步长 20 ms (无重叠覆盖)
cfg.t_range        = [-200, 800];                    % 解码时程范围 (ms，共 51 个窗口)

% 平滑参数
cfg.smooth_pts     = 5;                              % 平滑点数 (5点平滑)
cfg.smooth_typ     = 'gaussian';                     % 平滑方式: gaussian

% 并行与统计参数
cfg.n_perm = 50;                            % 对角线置换检验次数 (批处理推荐 200 次)
cfg.n_workers      = 20;                             % 20 线程极速并行池 (用户指定 20 核并行)
cfg.svm_lambda     = 0.01;                           % 岭正则化参数 Lambda

% 频段定义与规范学术显示标签
cfg.bands          = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.bands_disp     = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low-Gamma', 'High-Gamma'};
cfg.n_bands        = numel(cfg.bands);

% 调试或运行范围 ('all': 运行全部目标电极; 或指定数量)
cfg.max_elecs = 2;                            % Inf: 全量批处理运行全部目标电极
cfg.skip_existing = false;                           % 若已完成出图与结果则跳过，支持断点续跑

% 路径设置 (自适应定位 color_analyse_0825 根目录)
script_path = mfilename('fullpath');
c_idx = strfind(script_path, 'color_analyse_0825');
if ~isempty(c_idx)
    color_root = script_path(1 : c_idx(1) + length('color_analyse_0825') - 1);
else
    color_root = fullfile(fileparts(fileparts(fileparts(script_path))), 'color_analyse_0825');
end
data_root  = fullfile(color_root, 'process_data_new');
res_root   = fullfile(color_root, 'result');
c04_table  = fullfile(res_root, 'tables', 'color_effects_summary.mat');

tab_dir    = fullfile(res_root, 'tables');
tc_tab_dir = fullfile(tab_dir, sprintf('cross_decoding_%s_timecourses', cfg.target_mode));
fig_dir    = fullfile(res_root, 'figures', sprintf('cross_decoding_%s', cfg.target_mode));

if ~exist(tab_dir, 'dir'),    mkdir(tab_dir);    end
if ~exist(tc_tab_dir, 'dir'), mkdir(tc_tab_dir); end
if ~exist(fig_dir, 'dir'),    mkdir(fig_dir);    end

%% 2. 读取 C04 汇总表并确定目标电极通道列表
fprintf('========================================================================\n');
fprintf('  【C09: Task 3 (纯色块) -> Task 2 (灰度记忆色) 跨任务跨表征神经解码】  \n');
fprintf('========================================================================\n');

if strcmp(cfg.target_mode, 'custom')
    target_subs   = cfg.custom_subs(:);
    target_elecs  = cfg.custom_elecs(:);
    n_total_elecs = min(numel(target_subs), cfg.max_elecs);
    fprintf('[+] 当前为【自定义指定电极】模式，共 %d 个通道待分析。\n', n_total_elecs);
else
    if ~isfile(c04_table)
        error('未找到 C04 筛选汇总表: %s\n请先确认 C04_screen_color_channels_0825.m 结果！', c04_table);
    end
    loaded_c04 = load(c04_table);
    if isfield(loaded_c04, 'all_tbl'), c04_tbl = loaded_c04.all_tbl; else, c04_tbl = loaded_c04.res_table; end
    
    if strcmp(cfg.target_mode, 'concordant')
        concord_mask = (c04_tbl.is_significant == 1) & ...
            (strcmp(c04_tbl.concordance_type, 'Concordant_Positive') | ...
             strcmp(c04_tbl.concordance_type, 'Concordant_Negative'));
        c04_sub_tbl = c04_tbl(concord_mask, :);
        fprintf('[+] 模式: 【Task 1 四类别同向显著电极】\n');
    else
        % 默认全量显著
        sig_mask = (c04_tbl.is_significant == 1);
        c04_sub_tbl = c04_tbl(sig_mask, :);
        fprintf('[+] 模式: 【Task 1 全部总体显著电极 (is_significant == 1)】\n');
    end
    
    % 提取唯一的 [被试_电极] 组合
    elec_keys = strcat(c04_sub_tbl.subject, '_', c04_sub_tbl.channel);
    [~, u_ia] = unique(elec_keys, 'stable');
    
    target_subs   = c04_sub_tbl.subject(u_ia);
    target_elecs  = c04_sub_tbl.channel(u_ia);
    n_total_elecs = min(numel(target_subs), cfg.max_elecs);
    fprintf('[+] 共筛选出 %d 个目标电极待处理。\n', n_total_elecs);
end

fprintf('[+] 分折模式: %s | 窗长: %d ms | 步长: %d ms | 对角线置换: %d 次 | TGM: 经验热力图\n', ...
    cfg.split_mode, cfg.win_len, cfg.win_step, cfg.n_perm);

%% 3. 启动 CPU 并行池
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

%% 4. 按被试分组批量跨任务解码 (整被试仅载入一次数据)
unique_subs = unique(target_subs(1:n_total_elecs), 'stable');
summary_list = struct([]);

% 配色定义 (遵循 Nature 科研标准)
col_joint = [0.85, 0.37, 0.01]; % 陶土暖橙 (Multi-Band)
col_null  = [0.88, 0.88, 0.88]; % 浅灰阴影
col_clust = [1.00, 0.92, 0.70]; % 金黄显著簇背景
band_cols = [
    0.40, 0.40, 0.40;  % Delta: 灰
    0.95, 0.60, 0.20;  % Theta: 杏黄
    0.20, 0.45, 0.75;  % Alpha: 钴蓝
    0.10, 0.65, 0.45;  % Beta: 青绿
    0.50, 0.35, 0.75;  % Low_Gamma: 紫灰
    0.90, 0.15, 0.50   % High_Gamma: 玫红
];

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
    fprintf('>>> [被试 %d/%d: %s] 开始批量处理本被试 %d 个电极 ...\n', ...
        s_i, numel(unique_subs), sub_id, n_sub_elecs);
    fprintf('========================================================================\n');
    
    % 加载 Task 3 与 Task 2 数据 (整被试只载入一次)
    t3_mat = fullfile(data_root, sub_id, 'task3_multiband_epoched.mat');
    t2_mat = fullfile(data_root, sub_id, 'task2_multiband_epoched.mat');
    if ~isfile(t3_mat) || ~isfile(t2_mat)
        warning('未找到被试 %s 数据文件，跳过。', sub_id);
        continue;
    end
    
    t_load = tic;
    % 1. 先载入 Task 3，只保留红绿试次，立即清空大变量释放内存
    m3 = load(t3_mat, 'epoched_data');
    ti3 = m3.epoched_data.trial_info;
    tr_mask = strcmp(ti3.color, 'red') | strcmp(ti3.color, 'green');
    ti3_use = ti3(tr_mask, :);
    n_tr    = height(ti3_use);
    y_tr    = zeros(n_tr, 1);
    y_tr(strcmp(ti3_use.color, 'red')) = 1; % Red=1, Green=0
    time_ms = m3.epoched_data.time_ms(:)';
    ch_list3 = m3.epoched_data.channels;
    
    ep3_bands = struct();
    for b = 1:cfg.n_bands
        ep3_bands.(cfg.bands{b}) = m3.epoched_data.(cfg.bands{b})(tr_mask, :, :);
    end
    clear m3 ti3;
    
    % 2. 再载入 Task 2，只保留灰度水果试次，立即清空大变量释放内存
    m2 = load(t2_mat, 'epoched_data');
    ti2 = m2.epoched_data.trial_info;
    te_mask = strcmp(ti2.state, cfg.test_state);
    ti2_use = ti2(te_mask, :);
    n_te    = height(ti2_use);
    y_te    = zeros(n_te, 1);
    y_te(strcmp(ti2_use.memory_color, 'red')) = 1; % Red Memory=1, Green Memory=0
    ch_list2 = m2.epoched_data.channels;
    
    ep2_bands = struct();
    for b = 1:cfg.n_bands
        ep2_bands.(cfg.bands{b}) = m2.epoched_data.(cfg.bands{b})(te_mask, :, :);
    end
    clear m2 ti2;
    
    fprintf('    [+] 数据载入完成 (耗时 %.2f 秒) | Task 3 试次: %d (Red: %d, Green: %d) | Task 2 Gray: %d (Red: %d, Green: %d)\n', ...
        toc(t_load), n_tr, sum(y_tr==1), sum(y_tr==0), n_te, sum(y_te==1), sum(y_te==0));
    
    % 逐电极循环解码
    for e_i = 1:n_sub_elecs
        ch_name = sub_elecs{e_i};
        global_elec_count = global_elec_count + 1;
        
        fig_png = fullfile(fig_dir, sprintf('%s_%s_cross_decoding.png', sub_id, ch_name));
        res_mat = fullfile(tc_tab_dir, sprintf('%s_%s_cross_decoding_results.mat', sub_id, ch_name));
        
        if cfg.skip_existing && isfile(fig_png) && isfile(res_mat)
            fprintf('    [%d/%d | 全局 %d/%d] 电极 [%s - %s] 结果已存在，载入缓存汇总。\n', ...
                e_i, n_sub_elecs, global_elec_count, n_total_elecs, sub_id, ch_name);
            try
                saved_res = load(res_mat, 'diag_joint_s', 'tgm_matrix_s', 'sig_clusters', 't_centers');
                [r_max, c_max] = find(saved_res.tgm_matrix_s == max(saved_res.tgm_matrix_s(:)), 1);
                [peak_diag_val, peak_diag_idx] = max(saved_res.diag_joint_s);
                rec = struct();
                rec.subject            = sub_id;
                rec.channel            = ch_name;
                rec.diag_peak_acc      = round(peak_diag_val, 4);
                rec.diag_peak_time_ms  = saved_res.t_centers(peak_diag_idx);
                rec.tgm_max_acc        = round(max(saved_res.tgm_matrix_s(:)), 4);
                rec.tgm_task3_time_ms  = saved_res.t_centers(r_max);
                rec.tgm_task2_time_ms  = saved_res.t_centers(c_max);
                rec.n_sig_clusters     = size(saved_res.sig_clusters, 1);
                rec.has_sig_cluster    = (size(saved_res.sig_clusters, 1) > 0);
                summary_list = [summary_list; rec]; %#ok<AGROW>
            catch
            end
            continue;
        end
        
        t_elec = tic;
        fprintf('    [%d/%d | 全局 %d/%d] 正在跨任务解码电极: [%s - %s] ...\n', ...
            e_i, n_sub_elecs, global_elec_count, n_total_elecs, sub_id, ch_name);
        
        ch_idx3 = find(strcmp(ch_list3, ch_name), 1);
        ch_idx2 = find(strcmp(ch_list2, ch_name), 1);
        if isempty(ch_idx3) || isempty(ch_idx2)
            warning('通道 %s 在数据中不存在，跳过。', ch_name);
            continue;
        end
        
        % --- 步骤 1: 提取 20ms 滑动时间窗特征张量 [N x n_bands x n_win] ---
        X3_3d = zeros(n_tr, cfg.n_bands, n_win, 'single');
        X2_3d = zeros(n_te, cfg.n_bands, n_win, 'single');
        
        for b = 1:cfg.n_bands
            b_name = cfg.bands{b};
            raw3 = squeeze(ep3_bands.(b_name)(:, ch_idx3, :));
            raw2 = squeeze(ep2_bands.(b_name)(:, ch_idx2, :));
            
            for w = 1:n_win
                tc = t_centers(w);
                t_m = (time_ms >= (tc - cfg.win_len / 2)) & (time_ms < (tc + cfg.win_len / 2));
                X3_3d(:, b, w) = mean(raw3(:, t_m), 2);
                X2_3d(:, b, w) = mean(raw2(:, t_m), 2);
            end
        end
        
        % --- 步骤 2: 真实对角线同步解码与 6 单频段消融 ---
        real_diag_joint = zeros(1, n_win);
        real_diag_bands = zeros(cfg.n_bands, n_win);
        
        % 预先标准化对角线特征
        X3_diag_norm = cell(n_win, 1);
        X2_diag_norm = cell(n_win, 1);
        
        for w = 1:n_win
            X_tr_w = double(squeeze(X3_3d(:, :, w)));
            X_te_w = double(squeeze(X2_3d(:, :, w)));
            
            mu_w  = mean(X_tr_w, 1);
            sig_w = std(X_tr_w, 0, 1);
            sig_w(sig_w < 1e-6) = 1;
            
            X_tr_norm = (X_tr_w - mu_w) ./ sig_w;
            X_te_norm = (X_te_w - mu_w) ./ sig_w;
            
            X3_diag_norm{w} = X_tr_norm;
            X2_diag_norm{w} = X_te_norm;
            
            % Multi-Band
            mdl = fitclinear(X_tr_norm, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', cfg.svm_lambda);
            y_pred = predict(mdl, X_te_norm);
            sens = sum(y_te == 1 & y_pred == 1) / max(1, sum(y_te == 1));
            spec = sum(y_te == 0 & y_pred == 0) / max(1, sum(y_te == 0));
            real_diag_joint(w) = (sens + spec) / 2;
            
            % 6 个单频段独立解码
            for b = 1:cfg.n_bands
                X_tr_b = double(X3_3d(:, b, w));
                X_te_b = double(X2_3d(:, b, w));
                mu_b  = mean(X_tr_b);
                sig_b = std(X_tr_b);
                if sig_b < 1e-6, sig_b = 1; end
                
                X_tr_bn = (X_tr_b - mu_b) ./ sig_b;
                X_te_bn = (X_te_b - mu_b) ./ sig_b;
                
                mdl_b = fitclinear(X_tr_bn, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', cfg.svm_lambda);
                y_pred_b = predict(mdl_b, X_te_bn);
                sens_b = sum(y_te == 1 & y_pred_b == 1) / max(1, sum(y_te == 1));
                spec_b = sum(y_te == 0 & y_pred_b == 0) / max(1, sum(y_te == 0));
                real_diag_bands(b, w) = (sens_b + spec_b) / 2;
            end
        end
        
        % --- 步骤 3: 计算 2D TGM 时间泛化矩阵 (51x51 网格，Multi-Band) ---
        tgm_matrix = zeros(n_win, n_win);
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
                sens = sum(y_te == 1 & y_pred == 1) / max(1, sum(y_te == 1));
                spec = sum(y_te == 0 & y_pred == 0) / max(1, sum(y_te == 0));
                tgm_matrix(w3, w2) = (sens + spec) / 2;
            end
        end
        
        % --- 步骤 4: 对角线非参数置换检验 (仅 1D 曲线跑 200 次置换，极速并行) ---
        null_diag_dist = zeros(cfg.n_perm, n_win);
        svm_lam = cfg.svm_lambda;
        
        parfor perm_i = 1:cfg.n_perm
            y_te_perm = y_te(randperm(n_te));
            perm_curve = zeros(1, n_win);
            for w = 1:n_win
                mdl_p = fitclinear(X3_diag_norm{w}, y_tr, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', svm_lam);
                y_pred = predict(mdl_p, X2_diag_norm{w});
                sens = sum(y_te_perm == 1 & y_pred == 1) / max(1, sum(y_te_perm == 1));
                spec = sum(y_te_perm == 0 & y_pred == 0) / max(1, sum(y_te_perm == 0));
                perm_curve(w) = (sens + spec) / 2;
            end
            null_diag_dist(perm_i, :) = perm_curve;
        end
        
        % --- 步骤 5: 曲线平滑与时间簇质量检验 ---
        diag_joint_s = smoothdata(real_diag_joint, cfg.smooth_typ, cfg.smooth_pts);
        diag_bands_s = zeros(size(real_diag_bands));
        for b = 1:cfg.n_bands
            diag_bands_s(b, :) = smoothdata(real_diag_bands(b, :), cfg.smooth_typ, cfg.smooth_pts);
        end
        null_diag_s  = smoothdata(null_diag_dist, 2, cfg.smooth_typ, cfg.smooth_pts);
        tgm_matrix_s = imgaussfilt(tgm_matrix, 0.8);
        
        % 逐点 p 值与显著掩码
        p_pointwise = (1 + sum(null_diag_s >= diag_joint_s, 1)) / (1 + cfg.n_perm);
        alpha_pt = 0.05;
        sig_mask = (p_pointwise < alpha_pt) & (t_centers >= 0);
        
        % 提取连续显著簇
        clusters = [];
        in_c = false; c_start = 1;
        for w = 1:n_win
            if sig_mask(w) && ~in_c
                in_c = true; c_start = w;
            elseif ~sig_mask(w) && in_c
                in_c = false; clusters = [clusters; c_start, w-1]; %#ok<AGROW>
            end
        end
        if in_c, clusters = [clusters; c_start, n_win]; end
        
        % 簇质量 FWE 校正
        sig_clusters = [];
        if ~isempty(clusters)
            n_cl = size(clusters, 1);
            cl_mass = zeros(n_cl, 1);
            for ci = 1:n_cl
                c_range = clusters(ci, 1) : clusters(ci, 2);
                cl_mass(ci) = sum(diag_joint_s(c_range) - 0.5);
            end
            
            max_null_mass = zeros(cfg.n_perm, 1);
            for pi = 1:cfg.n_perm
                null_c_mask = (null_diag_s(pi, :) >= prctile(null_diag_s, 95, 1)) & (t_centers >= 0);
                null_masses = [0];
                in_nc = false; nc_start = 1;
                for w = 1:n_win
                    if null_c_mask(w) && ~in_nc
                        in_nc = true; nc_start = w;
                    elseif ~null_c_mask(w) && in_nc
                        in_nc = false;
                        null_masses(end+1) = sum(null_diag_s(pi, nc_start:w-1) - 0.5); %#ok<AGROW>
                    end
                end
                if in_nc, null_masses(end+1) = sum(null_diag_s(pi, nc_start:n_win) - 0.5); end %#ok<AGROW>
                max_null_mass(pi) = max(null_masses);
            end
            
            for ci = 1:n_cl
                p_cl = (1 + sum(max_null_mass >= cl_mass(ci))) / (1 + cfg.n_perm);
                if p_cl < 0.05
                    sig_clusters = [sig_clusters; clusters(ci, :)]; %#ok<AGROW>
                end
            end
        end
        
        % --- 步骤 6: 学术级规范出图 (1:1 左右双子图) ---
        fig = figure('Position', [100, 100, 1280, 520], 'Color', 'w', 'Visible', 'off');
        
        % 左子图: 1D 对角线同步时程
        subplot(1, 2, 1);
        hold on;
        
        if ~isempty(sig_clusters)
            for sci = 1:size(sig_clusters, 1)
                c_x1 = t_centers(sig_clusters(sci, 1));
                c_x2 = t_centers(sig_clusters(sci, 2));
                fill([c_x1, c_x2, c_x2, c_x1], [0.35, 0.35, 0.85, 0.85], col_clust, ...
                    'EdgeColor', 'none', 'FaceAlpha', 0.6, 'HandleVisibility', 'off');
            end
        end
        
        null_hi = prctile(null_diag_s, 97.5, 1);
        null_lo = prctile(null_diag_s, 2.5, 1);
        fill([t_centers, fliplr(t_centers)], [null_hi, fliplr(null_lo)], col_null, ...
            'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'Null 95% CI');
        
        yline(0.5, '--', 'Color', [0.55, 0.55, 0.55], 'LineWidth', 1.0, 'HandleVisibility', 'off');
        xline(0, ':', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1.0, 'HandleVisibility', 'off');
        
        for b = 1:cfg.n_bands
            plot(t_centers, diag_bands_s(b, :), 'Color', [band_cols(b, :), 0.55], ...
                'LineWidth', 1.2, 'DisplayName', cfg.bands_disp{b});
        end
        
        plot(t_centers, diag_joint_s, 'Color', col_joint, 'LineWidth', 2.6, ...
            'DisplayName', 'Multi-Band');
        
        if any(sig_mask)
            plot(t_centers(sig_mask), repmat(0.38, 1, sum(sig_mask)), 's', ...
                'MarkerFaceColor', col_joint, 'MarkerEdgeColor', 'none', 'MarkerSize', 4, ...
                'HandleVisibility', 'off');
        end
        
        xlim([-200, 800]);
        ylim([0.35, 0.80]);
        xlabel('Time (ms)', 'FontSize', 11, 'FontWeight', 'bold');
        ylabel('Balanced Accuracy', 'FontSize', 11, 'FontWeight', 'bold');
        title(sprintf('%s - %s  (Diagonal Sync)', sub_id, ch_name), ...
            'FontSize', 13, 'FontWeight', 'bold');
        legend('Location', 'northeast', 'FontSize', 8, 'Box', 'off');
        grid off; box off;
        set(gca, 'TickDir', 'out', 'LineWidth', 1.0, 'FontSize', 10);
        
        % 右子图: 2D TGM 时间泛化矩阵热力图
        subplot(1, 2, 2);
        imagesc(t_centers, t_centers, tgm_matrix_s);
        set(gca, 'YDir', 'normal');
        colormap(gca, 'parula');
        caxis([0.40, 0.70]);
        cb = colorbar;
        ylabel(cb, 'Cross-Task Accuracy', 'FontSize', 10, 'FontWeight', 'bold');
        
        hold on;
        plot([-200, 800], [-200, 800], 'w--', 'LineWidth', 1.2);
        xline(0, 'w:', 'LineWidth', 1.0);
        yline(0, 'w:', 'LineWidth', 1.0);
        
        xlim([-200, 800]);
        ylim([-200, 800]);
        xlabel('Task 2 (Memory Fruit) Time [ms]', 'FontSize', 11, 'FontWeight', 'bold');
        ylabel('Task 3 (Physical Patch) Time [ms]', 'FontSize', 11, 'FontWeight', 'bold');
        title(sprintf('%s - %s  (Temporal Generalization)', sub_id, ch_name), ...
            'FontSize', 13, 'FontWeight', 'bold');
        grid off; box off;
        set(gca, 'TickDir', 'out', 'LineWidth', 1.0, 'FontSize', 10);
        
        exportgraphics(fig, fig_png, 'Resolution', 300);
        close(fig);
        
        % 导出单通道结果 mat 文件
        save(res_mat, 'real_diag_joint', 'diag_joint_s', 'real_diag_bands', 'diag_bands_s', ...
            'null_diag_dist', 'tgm_matrix', 'tgm_matrix_s', 'p_pointwise', 'sig_clusters', ...
            't_centers', 'cfg');
        
        % 收集汇总指标
        [r_max, c_max] = find(tgm_matrix_s == max(tgm_matrix_s(:)), 1);
        [peak_diag_val, peak_diag_idx] = max(diag_joint_s);
        
        rec = struct();
        rec.subject            = sub_id;
        rec.channel            = ch_name;
        rec.diag_peak_acc      = round(peak_diag_val, 4);
        rec.diag_peak_time_ms  = t_centers(peak_diag_idx);
        rec.tgm_max_acc        = round(max(tgm_matrix_s(:)), 4);
        rec.tgm_task3_time_ms  = t_centers(r_max);
        rec.tgm_task2_time_ms  = t_centers(c_max);
        rec.n_sig_clusters     = size(sig_clusters, 1);
        rec.has_sig_cluster    = (size(sig_clusters, 1) > 0);
        summary_list = [summary_list; rec]; %#ok<AGROW>
        
        fprintf('    [+] 完成电极 [%s - %s] (耗时 %.2f 秒) | 对角线峰值: %.2f%% (%d ms) | TGM峰值: %.2f%% (T3=%d ms, T2=%d ms)\n', ...
            sub_id, ch_name, toc(t_elec), peak_diag_val * 100, t_centers(peak_diag_idx), ...
            max(tgm_matrix_s(:)) * 100, t_centers(r_max), t_centers(c_max));
    end
    clear ep3_bands ep2_bands;
end

%% 5. 导出全量跨任务解码汇总表
if ~isempty(summary_list)
    sum_tbl = struct2table(summary_list);
    out_sum_mat = fullfile(tab_dir, sprintf('cross_decoding_%s_summary.mat', cfg.target_mode));
    out_sum_csv = fullfile(tab_dir, sprintf('cross_decoding_%s_summary.csv', cfg.target_mode));
    save(out_sum_mat, 'sum_tbl', 'cfg');
    writetable(sum_tbl, out_sum_csv);
    fprintf('\n========================================================================\n');
    fprintf('  【C09 全量汇总表导出完成】\n');
    fprintf('  - MAT 路径: %s\n', out_sum_mat);
    fprintf('  - CSV 路径: %s\n', out_sum_csv);
    fprintf('  - 处理总电极数: %d | 具有显著时间簇电极数: %d (占比 %.1f%%)\n', ...
        height(sum_tbl), sum(sum_tbl.has_sig_cluster), ...
        (sum(sum_tbl.has_sig_cluster) / max(1, height(sum_tbl))) * 100);
    fprintf('========================================================================\n');
end
