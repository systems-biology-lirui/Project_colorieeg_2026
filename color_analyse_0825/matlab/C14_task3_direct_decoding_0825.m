%% ========================================================================
% 脚本名称: C14_task3_direct_decoding_0825.m
% 功能:
%   1. 【Task 3 物理纯色直接二分类 Decoding (不考虑形状色块)】
%      将纯色试次直接划分为单纯的两组：物理红色 (全部形状) vs 物理绿色 (全部形状)
%   2. 【分层 5 折交叉验证 (Stratified 5-Fold CV)】
%      严格保证训练集/测试集红绿平衡，无跨色块限制
%   3. 【多频段联合与 6 单频段独立解码】
%      提取 20ms 滑动窗口特征，评估 Multi-Band 及 6 个单频段
%   4. 【200 次置换检验与时间簇质量族误差校正 (Cluster-Mass FWE < 0.05)】
%   5. 【规范绘图与交互保存】
%      - 5 点高斯平滑
%      - 无背景方格 (grid off)
%      - 保存 300 DPI PNG 与可交互 FIG (Visible=on)
%   6. 【导出独立文件夹结果，绝不影响已有历史数据】
% ========================================================================

if exist('cfg_override', 'var') && isstruct(cfg_override)
    saved_cfg = cfg_override;
else
    saved_cfg = struct();
end
clc; close all;

%% 1. 主参数配置 (结构体简写，直观可调)
cfg = struct();

% 时间窗与时程参数
cfg.win_len        = 20;              % 滑动窗长 20 ms
cfg.win_step       = 20;              % 滑动步长 20 ms
cfg.t_range        = [-200, 800];     % 解码时程范围 (ms)
cfg.smooth_pts     = 5;               % 平滑点数
cfg.smooth_typ     = 'gaussian';      % 平滑方式

% 交叉验证与统计
cfg.n_folds        = 5;               % 分层 5 折交叉验证
cfg.n_perm         = 200;             % 200 次标签置换检验
cfg.n_workers      = 20;              % 并行线程数
cfg.svm_lambda     = 0.01;            % 岭正则化参数
cfg.cluster_alpha  = 0.05;            % 逐点显著性阈值

% 频段定义
cfg.bands          = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.bands_disp     = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low-Gamma', 'High-Gamma'};
cfg.n_bands        = numel(cfg.bands);

% 运行控制
cfg.max_elecs      = Inf;             % 默认运行全部 229 个显著电极
cfg.skip_existing  = true;            % 支持断点续跑

% 应用外部传入参数覆盖
if ~isempty(fieldnames(saved_cfg))
    flds = fieldnames(saved_cfg);
    for fi = 1:numel(flds)
        cfg.(flds{fi}) = saved_cfg.(flds{fi});
    end
end

% 路径配置 (独立全新目录)
script_dir = fileparts(mfilename('fullpath'));
work_dir   = fileparts(script_dir);
data_root  = fullfile(work_dir, 'process_data_new');
res_root   = fullfile(work_dir, 'result');
c04_table  = fullfile(res_root, 'tables', 'color_effects_summary.mat');

fig_dir    = fullfile(res_root, 'figures', 'decoding_task3_direct');
tc_tab_dir = fullfile(res_root, 'tables', 'decoding_task3_direct_timecourses');
sum_csv    = fullfile(res_root, 'tables', 'task3_direct_decoding_summary.csv');
sum_mat    = fullfile(res_root, 'tables', 'task3_direct_decoding_summary.mat');

if ~exist(fig_dir, 'dir'),    mkdir(fig_dir);    end
if ~exist(tc_tab_dir, 'dir'), mkdir(tc_tab_dir); end

%% 2. 提取 Task 1 总体显著电极列表 (共 229 个)
if ~isfile(c04_table)
    error('未找到 C04 筛选表: %s', c04_table);
end
c04_data = load(c04_table);
if isfield(c04_data, 'all_tbl'), c04_tbl = c04_data.all_tbl; else, c04_tbl = c04_data.res_table; end

sig_mask = (c04_tbl.is_significant == 1);
c04_sig  = c04_tbl(sig_mask, :);
[~, u_ia] = unique(strcat(c04_sig.subject, '_', c04_sig.channel), 'stable');
c04_sig_u    = c04_sig(u_ia, :);
target_subs  = c04_sig_u.subject;
target_elecs = c04_sig_u.channel;
n_total_elecs = min(numel(target_subs), cfg.max_elecs);

% 构建解剖与坐标映射字典 (兼容不同字段名)
anat_map = containers.Map();
for r = 1:height(c04_tbl)
    k_item = sprintf('%s_%s', char(c04_tbl.subject{r}), char(c04_tbl.channel{r}));
    if ~isKey(anat_map, k_item)
        s_info = struct();
        if ismember('dkt_anatomy', c04_tbl.Properties.VariableNames), s_info.dkt = char(c04_tbl.dkt_anatomy(r)); else, s_info.dkt = ''; end
        if ismember('aal_anatomy', c04_tbl.Properties.VariableNames), s_info.aal = char(c04_tbl.aal_anatomy(r)); else, s_info.aal = ''; end
        if ismember('stream_hierarchy', c04_tbl.Properties.VariableNames)
            s_info.stream = char(c04_tbl.stream_hierarchy(r));
        elseif ismember('stream', c04_tbl.Properties.VariableNames)
            s_info.stream = char(c04_tbl.stream(r));
        else
            s_info.stream = '';
        end
        if ismember('mni_x', c04_tbl.Properties.VariableNames), s_info.x = c04_tbl.mni_x(r); else, s_info.x = NaN; end
        if ismember('mni_y', c04_tbl.Properties.VariableNames), s_info.y = c04_tbl.mni_y(r); else, s_info.y = NaN; end
        if ismember('mni_z', c04_tbl.Properties.VariableNames), s_info.z = c04_tbl.mni_z(r); else, s_info.z = NaN; end
        anat_map(k_item) = s_info;
    end
end

fprintf('========================================================================\n');
fprintf('  【C14: Task 3 物理纯色直接二分类 Decoding (不考虑形状色块)】\n');
fprintf('  目标显著电极总数: %d | 交叉验证: 分层 %d 折 | 置换次数: %d\n', ...
    n_total_elecs, cfg.n_folds, cfg.n_perm);
fprintf('  结果保存目录: %s\n', fig_dir);
fprintf('========================================================================\n');

%% 3. 启动 CPU 并行池
curr_pool = gcp('nocreate');
if isempty(curr_pool)
    parpool('local', cfg.n_workers);
elseif curr_pool.NumWorkers ~= cfg.n_workers
    delete(curr_pool);
    parpool('local', cfg.n_workers);
end

%% 4. 按被试批量解码
unique_subs = unique(target_subs(1:n_total_elecs), 'stable');
summary_list = struct([]);

% 配色定义
col_joint = [0.85, 0.37, 0.01]; % Multi-Band 暖橙
col_null  = [0.85, 0.85, 0.85]; % 置信区间浅灰
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
global_idx = 0;

for s_i = 1:numel(unique_subs)
    sub_id = unique_subs{s_i};
    sub_mask = strcmp(target_subs(1:n_total_elecs), sub_id);
    sub_elecs = target_elecs(sub_mask);
    n_sub_elecs = numel(sub_elecs);
    
    mat_file = fullfile(data_root, sub_id, 'task3_multiband_epoched.mat');
    if ~isfile(mat_file)
        warning('未找到被试 %s 数据，跳过。', sub_id);
        continue;
    end
    
    t_load = tic;
    ep_data = load(mat_file, 'epoched_data');
    ep = ep_data.epoched_data;
    time_ms = ep.time_ms(:)';
    ti = ep.trial_info;
    
    % 仅提取红绿纯色试次 (所有形状合并)
    rg_mask = strcmp(ti.color, 'red') | strcmp(ti.color, 'green');
    ti_rg   = ti(rg_mask, :);
    n_rg    = height(ti_rg);
    
    % 二分类标签: 全部物理红 = 1, 全部物理绿 = 0
    y = zeros(n_rg, 1);
    y(strcmp(ti_rg.color, 'red')) = 1;
    
    % 构建分层 5 折交叉验证掩码 (直接两组数据，不管形状)
    rng(42);
    idx_red   = find(y == 1);
    idx_green = find(y == 0);
    cv_r = cvpartition(numel(idx_red), 'KFold', cfg.n_folds);
    cv_g = cvpartition(numel(idx_green), 'KFold', cfg.n_folds);
    
    fold_tr_masks = false(n_rg, cfg.n_folds);
    fold_te_masks = false(n_rg, cfg.n_folds);
    for f_i = 1:cfg.n_folds
        te_r = idx_red(test(cv_r, f_i));
        te_g = idx_green(test(cv_g, f_i));
        te_idx = [te_r; te_g];
        fold_te_masks(te_idx, f_i) = true;
        fold_tr_masks(:, f_i) = ~fold_te_masks(:, f_i);
    end
    
    fprintf('>>> [被试 %d/%d: %s] 载入完成 (%.2fs), 纯色试次: %d (红: %d, 绿: %d), 包含 %d 个显著通道\n', ...
        s_i, numel(unique_subs), sub_id, toc(t_load), n_rg, sum(y==1), sum(y==0), n_sub_elecs);
    
    for e_i = 1:n_sub_elecs
        ch_name = sub_elecs{e_i};
        global_idx = global_idx + 1;
        
        fig_png = fullfile(fig_dir, sprintf('%s_%s_task3_direct_decoding.png', sub_id, ch_name));
        tc_csv  = fullfile(tc_tab_dir, sprintf('%s_%s_task3_direct_timecourse.csv', sub_id, ch_name));
        
        need_calc = ~(cfg.skip_existing && isfile(fig_png) && isfile(tc_csv));
        
        ch_idx = find(strcmp(ep.channels, ch_name), 1);
        if isempty(ch_idx)
            warning('通道 %s 不在被试 %s 的通道列表中，跳过。', ch_name, sub_id);
            continue;
        end
        
        % 提取滑动时间窗特征 [n_rg x n_bands x n_win]
        X_3d = zeros(n_rg, cfg.n_bands, n_win, 'single');
        for b = 1:cfg.n_bands
            b_name = cfg.bands{b};
            raw_b = squeeze(ep.(b_name)(rg_mask, ch_idx, :));
            for w = 1:n_win
                tc = t_centers(w);
                t_m = (time_ms >= (tc - cfg.win_len/2)) & (time_ms < (tc + cfg.win_len/2));
                X_3d(:, b, w) = mean(raw_b(:, t_m), 2);
            end
        end
        
        if need_calc
            t_elec = tic;
            
            % 预计算每个窗口和交叉验证折的标准化特征 (大幅提速)
            X_tr_cell = cell(n_win, cfg.n_folds);
            X_te_cell = cell(n_win, cfg.n_folds);
            for w = 1:n_win
                X_w = double(squeeze(X_3d(:, :, w)));
                for f_i = 1:cfg.n_folds
                    tr_m = fold_tr_masks(:, f_i);
                    te_m = fold_te_masks(:, f_i);
                    mu  = mean(X_w(tr_m, :), 1);
                    sig = std(X_w(tr_m, :), 0, 1);
                    sig(sig < 1e-6) = 1;
                    X_tr_cell{w, f_i} = (X_w(tr_m, :) - mu) ./ sig;
                    X_te_cell{w, f_i} = (X_w(te_m, :) - mu) ./ sig;
                end
            end
            
            % 1. 真实 Multi-Band 联合解码
            real_acc_joint = zeros(1, n_win);
            for w = 1:n_win
                f_accs = zeros(1, cfg.n_folds);
                for f_i = 1:cfg.n_folds
                    tr_m = fold_tr_masks(:, f_i);
                    te_m = fold_te_masks(:, f_i);
                    mdl = fitclinear(X_tr_cell{w, f_i}, y(tr_m), 'Learner', 'svm', ...
                        'Regularization', 'ridge', 'Lambda', cfg.svm_lambda, 'Solver', 'dual');
                    y_pred = predict(mdl, X_te_cell{w, f_i});
                    sens = sum(y(te_m) == 1 & y_pred == 1) / max(1, sum(y(te_m) == 1));
                    spec = sum(y(te_m) == 0 & y_pred == 0) / max(1, sum(y(te_m) == 0));
                    f_accs(f_i) = (sens + spec) / 2;
                end
                real_acc_joint(w) = mean(f_accs);
            end
            
            % 2. 6 个单频段独立解码
            single_band_accs = zeros(cfg.n_bands, n_win);
            for b = 1:cfg.n_bands
                for w = 1:n_win
                    f_accs = zeros(1, cfg.n_folds);
                    for f_i = 1:cfg.n_folds
                        tr_m = fold_tr_masks(:, f_i);
                        te_m = fold_te_masks(:, f_i);
                        X_tr_1d = X_tr_cell{w, f_i}(:, b);
                        X_te_1d = X_te_cell{w, f_i}(:, b);
                        mdl = fitclinear(X_tr_1d, y(tr_m), 'Learner', 'svm', ...
                            'Regularization', 'ridge', 'Lambda', cfg.svm_lambda, 'Solver', 'dual');
                        y_pred = predict(mdl, X_te_1d);
                        sens = sum(y(te_m) == 1 & y_pred == 1) / max(1, sum(y(te_m) == 1));
                        spec = sum(y(te_m) == 0 & y_pred == 0) / max(1, sum(y(te_m) == 0));
                        f_accs(f_i) = (sens + spec) / 2;
                    end
                    single_band_accs(b, w) = mean(f_accs);
                end
            end
            
            % 3. 200 次并行置换检验
            null_dist = zeros(cfg.n_perm, n_win);
            n_folds_loc = cfg.n_folds;
            svm_lambda_loc = cfg.svm_lambda;
            
            parfor perm_i = 1:cfg.n_perm
                y_perm = y(randperm(n_rg));
                perm_curve = zeros(1, n_win);
                for w = 1:n_win
                    f_accs = zeros(1, n_folds_loc);
                    for f_i = 1:n_folds_loc
                        tr_m = fold_tr_masks(:, f_i);
                        te_m = fold_te_masks(:, f_i);
                        mdl = fitclinear(X_tr_cell{w, f_i}, y_perm(tr_m), 'Learner', 'svm', ...
                            'Regularization', 'ridge', 'Lambda', svm_lambda_loc, 'Solver', 'dual');
                        y_pred = predict(mdl, X_te_cell{w, f_i});
                        sens = sum(y_perm(te_m) == 1 & y_pred == 1) / max(1, sum(y_perm(te_m) == 1));
                        spec = sum(y_perm(te_m) == 0 & y_pred == 0) / max(1, sum(y_perm(te_m) == 0));
                        f_accs(f_i) = (sens + spec) / 2;
                    end
                    perm_curve(w) = mean(f_accs);
                end
                null_dist(perm_i, :) = perm_curve;
            end
            
            % 4. 曲线高斯平滑
            real_acc_joint_s = smoothdata(real_acc_joint, cfg.smooth_typ, cfg.smooth_pts);
            single_band_accs_s = zeros(size(single_band_accs));
            for b = 1:cfg.n_bands
                single_band_accs_s(b, :) = smoothdata(single_band_accs(b, :), cfg.smooth_typ, cfg.smooth_pts);
            end
            null_dist_s = smoothdata(null_dist, 2, cfg.smooth_typ, cfg.smooth_pts);
            
            % 5. 时间簇质量检验 (Cluster-Mass FWE)
            p_pointwise = (1 + sum(null_dist_s >= real_acc_joint_s, 1)) / (1 + cfg.n_perm);
            sig_mask_pts = (p_pointwise < cfg.cluster_alpha) & (t_centers >= 0);
            
            clusters = []; in_c = false; c_s = 1;
            for w = 1:n_win
                if sig_mask_pts(w) && ~in_c
                    in_c = true; c_s = w;
                elseif ~sig_mask_pts(w) && in_c
                    in_c = false; clusters = [clusters; c_s, w-1]; %#ok<AGROW>
                end
            end
            if in_c, clusters = [clusters; c_s, n_win]; end
            
            n_cl = size(clusters, 1);
            real_masses = zeros(n_cl, 1);
            for c_i = 1:n_cl
                idx_r = clusters(c_i, 1) : clusters(c_i, 2);
                real_masses(c_i) = sum(real_acc_joint_s(idx_r) - 0.5);
            end
            
            null_max_mass = zeros(cfg.n_perm, 1);
            for p_i = 1:cfg.n_perm
                curve_p = null_dist_s(p_i, :);
                p_pt = (1 + sum(null_dist_s >= curve_p, 1)) / (1 + cfg.n_perm);
                p_sig = (p_pt < cfg.cluster_alpha) & (t_centers >= 0);
                
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
            
            % 6. 导出时程 CSV
            tc_tbl = table(repmat(string(sub_id), n_win, 1), repmat(string(ch_name), n_win, 1), ...
                t_centers', real_acc_joint_s', p_pointwise', mean(null_dist_s, 1)', ...
                prctile(null_dist_s, 97.5, 1)', prctile(null_dist_s, 2.5, 1)', ...
                'VariableNames', {'subject', 'channel', 'time_ms', 'acc_joint_smoothed', ...
                                  'p_pointwise', 'null_mean', 'null_ci_upper', 'null_ci_lower'});
            for b = 1:cfg.n_bands
                tc_tbl.(['acc_' cfg.bands{b} '_smoothed']) = single_band_accs_s(b, :)';
            end
            writetable(tc_tbl, tc_csv);
            
            % 7. 绘制规范图 (无方格 grid off, 主标题简洁)
            fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1200, 480]);
            
            % 左图: 时程曲线
            subplot(1, 2, 1); hold on; grid off;
            set(gca, 'Box', 'off', 'FontSize', 11, 'LineWidth', 1.1);
            
            null_up  = prctile(null_dist_s, 97.5, 1);
            null_low = prctile(null_dist_s, 2.5, 1);
            fill([t_centers, fliplr(t_centers)], [null_low, fliplr(null_up)], ...
                col_null, 'EdgeColor', 'none', 'FaceAlpha', 0.6, 'HandleVisibility', 'off');
            
            yline(0.5, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.2, 'HandleVisibility', 'off');
            xline(0, '-', 'Color', [0.3, 0.3, 0.3], 'LineWidth', 1.0, 'HandleVisibility', 'off');
            
            % 显著簇阴影
            for c_i = 1:n_cl
                if cluster_pvals(c_i) < 0.05
                    fill([t_centers(clusters(c_i,1)), t_centers(clusters(c_i,2)), ...
                          t_centers(clusters(c_i,2)), t_centers(clusters(c_i,1))], ...
                         [0.35, 0.35, 0.85, 0.85], [1.0, 0.9, 0.7], 'EdgeColor', 'none', ...
                         'FaceAlpha', 0.35, 'HandleVisibility', 'off');
                end
            end
            
            % 逐点显著标记
            sig_pts = find(p_pointwise < 0.05 & t_centers >= 0);
            if ~isempty(sig_pts)
                plot(t_centers(sig_pts), repmat(0.38, 1, numel(sig_pts)), 's', ...
                    'MarkerFaceColor', col_joint, 'MarkerEdgeColor', 'none', ...
                    'MarkerSize', 4, 'HandleVisibility', 'off');
            end
            
            for b = 1:cfg.n_bands
                plot(t_centers, single_band_accs_s(b, :), 'Color', band_cols(b, :), ...
                    'LineWidth', 1.3, 'LineStyle', '-', 'DisplayName', cfg.bands_disp{b});
            end
            plot(t_centers, real_acc_joint_s, 'Color', col_joint, 'LineWidth', 2.6, 'DisplayName', 'Multi-Band');
            
            xlim(cfg.t_range); ylim([0.35, 0.85]);
            xlabel('Time from stimulus onset (ms)', 'FontWeight', 'bold');
            ylabel('Balanced Accuracy', 'FontWeight', 'bold');
            legend('Location', 'northwest', 'NumColumns', 2, 'Box', 'off', 'FontSize', 9, 'Interpreter', 'none');
            
            % 右图: 柱状图
            subplot(1, 2, 2); hold on; grid off;
            set(gca, 'Box', 'off', 'FontSize', 11, 'LineWidth', 1.1);
            
            bar_names = [{'Multi-Band'}, cfg.bands_disp];
            bar_vals  = [max(real_acc_joint_s), max(single_band_accs_s, [], 2)'];
            all_cols  = [col_joint; band_cols];
            
            b_h = bar(1:numel(bar_names), bar_vals, 0.65, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.9);
            for k = 1:numel(bar_names), b_h.CData(k, :) = all_cols(k, :); end
            yline(0.5, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.2);
            for k = 1:numel(bar_names)
                text(k, bar_vals(k) + 0.015, sprintf('%.1f%%', bar_vals(k) * 100), ...
                    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
            end
            set(gca, 'XTick', 1:numel(bar_names), 'XTickLabel', bar_names, 'XTickLabelRotation', 30, 'TickLabelInterpreter', 'none');
            ylim([0.40, 0.85]);
            ylabel('Peak Balanced Accuracy', 'FontWeight', 'bold');
            
            sgtitle(sprintf('%s - %s (Task 3 Direct Pure Color Decoding)', sub_id, ch_name), ...
                'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
            
            % 导出 PNG (不需要 FIG)
            saveas(fig, fig_png);
            close(fig);
            
            fprintf('  [%d/%d] %s-%s 完成 (峰值: %.1f%%, 耗时 %.2fs)\n', ...
                global_idx, n_total_elecs, sub_id, ch_name, max(real_acc_joint_s)*100, toc(t_elec));
        else
            % 读取已有数据组装汇总
            tc_data = readtable(tc_csv);
            real_acc_joint_s = tc_data.acc_joint_smoothed';
            p_pointwise = tc_data.p_pointwise';
            cluster_pvals = 1; % 占位
            single_band_accs_s = zeros(cfg.n_bands, n_win);
            for b = 1:cfg.n_bands
                single_band_accs_s(b, :) = tc_data.(['acc_' cfg.bands{b} '_smoothed'])';
            end
        end
        
        % 记录条目
        [max_joint, best_w] = max(real_acc_joint_s);
        rec = struct();
        rec.subject          = string(sub_id);
        rec.channel          = string(ch_name);
        
        % 匹配解剖与 MNI 坐标
        k_elec = sprintf('%s_%s', sub_id, ch_name);
        if isKey(anat_map, k_elec)
            ainfo = anat_map(k_elec);
            rec.dkt_anatomy  = string(ainfo.dkt);
            rec.aal_anatomy  = string(ainfo.aal);
            rec.stream       = string(ainfo.stream);
            rec.mni_x        = ainfo.x;
            rec.mni_y        = ainfo.y;
            rec.mni_z        = ainfo.z;
        else
            rec.dkt_anatomy  = ""; rec.aal_anatomy = ""; rec.stream = "";
            rec.mni_x        = NaN; rec.mni_y = NaN; rec.mni_z = NaN;
        end
        
        rec.peak_acc_joint   = max_joint;
        rec.peak_time_ms     = t_centers(best_w);
        rec.peak_p_pointwise = p_pointwise(best_w);
        rec.has_sig_cluster  = any(cluster_pvals < 0.05);
        
        % 记录 6 单频段峰值
        for b = 1:cfg.n_bands
            rec.(['peak_acc_' cfg.bands{b}]) = max(single_band_accs_s(b, :));
        end
        summary_list = [summary_list; rec]; %#ok<AGROW>
    end
end

%% 5. 导出总表
if ~isempty(summary_list)
    sum_tbl = struct2table(summary_list);
    writetable(sum_tbl, sum_csv);
    save(sum_mat, 'sum_tbl', 'cfg');
    fprintf('\n[+] Task 3 总表导出成功: %s (共 %d 个通道)\n', sum_csv, height(sum_tbl));
end
