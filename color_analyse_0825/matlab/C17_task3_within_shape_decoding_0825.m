%% ========================================================================
% 脚本名称: C17_task3_within_shape_decoding_0825.m
% 功能:
%   1. 【Task 3 同形状内部红绿 Decoding 与三形状曲线平均】
%      - 对每个电极，分别在三种独立几何形状内部进行物理红 vs 物理绿二分类解码
%        * Shape 1 (pic_id == 1): 红 vs 绿 (5折分层交叉验证)
%        * Shape 2 (pic_id == 2): 红 vs 绿 (5折分层交叉验证)
%        * Shape 3 (pic_id == 3): 红 vs 绿 (5折分层交叉验证)
%      - 将三种形状的解码时程曲线逐点算术平均:
%        Acc_avg(t) = [Acc_shape1(t) + Acc_shape2(t) + Acc_shape3(t)] / 3
%   2. 【200 次置换检验与时间簇统计校正】
%      - 在每个形状内部独立置换红绿标签，计算零分布下的三形状平均曲线
%      - 计算逐时间点经验 p 值与全时程 Cluster-Mass FWE 校正 (p < 0.05)
%   3. 【独立目录导出，不改动已有结果】
%      - figures: result/figures/decoding_task3_within_shape/ (*.png, 300 DPI, 无 FIG)
%      - tables : result/tables/decoding_task3_within_shape_timecourses/ (*.csv)
%      - 汇总表 : result/tables/task3_within_shape_decoding_summary.csv / .mat
% ========================================================================

clear; clc; close all;

%% 1. 主参数配置 (置顶易调，简写平铺)
cfg = struct();
cfg.win_len        = 20;             % 滑动时间窗长 (ms)
cfg.win_step       = 20;             % 滑动时间步长 (ms)
cfg.t_range        = [-200, 800];    % 分析时程范围 (ms)

cfg.n_folds        = 5;              % 每个形状内部 5 折分层交叉验证
cfg.n_perm         = 200;            % 200 次非参数置换检验
cfg.cluster_alpha  = 0.05;           % 聚类形成门槛 (p < 0.05)
cfg.smooth_pts     = 5;              % 曲线平滑点数
cfg.smooth_typ     = 'gaussian';     % 高斯平滑
cfg.ridge_lambda   = 0.10;           % 正则化参数

cfg.n_workers      = 12;             % 并行核心数
cfg.skip_existing  = false;          % 是否跳过已完成位点
cfg.max_elecs      = Inf;            % 运行通道数 (Inf 为全量 229)

% 频段定义
cfg.bands          = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.n_bands        = numel(cfg.bands);

% 路径配置
script_dir = fileparts(mfilename('fullpath'));
work_dir   = fileparts(script_dir);
data_root  = fullfile(work_dir, 'process_data_new');
res_root   = fullfile(work_dir, 'result');
tab_dir    = fullfile(res_root, 'tables');

fig_dir    = fullfile(res_root, 'figures', 'decoding_task3_within_shape');
tc_tab_dir = fullfile(tab_dir, 'decoding_task3_within_shape_timecourses');
sum_csv    = fullfile(tab_dir, 'task3_within_shape_decoding_summary.csv');
sum_mat    = fullfile(tab_dir, 'task3_within_shape_decoding_summary.mat');

if ~exist(fig_dir, 'dir'),    mkdir(fig_dir);    end
if ~exist(tc_tab_dir, 'dir'), mkdir(tc_tab_dir); end

%% 2. 载入筛选电极列表 (全量 229 个显著电极)
c04_table = fullfile(tab_dir, 'color_effects_summary.mat');
if ~isfile(c04_table), error('未找到 C04 文件: %s', c04_table); end
c04_data = load(c04_table);
if isfield(c04_data, 'all_tbl'), c04_tbl = c04_data.all_tbl; else, c04_tbl = c04_data.res_table; end

sig_mask = (c04_tbl.is_significant == 1);
c04_sig  = c04_tbl(sig_mask, :);
[~, u_ia] = unique(strcat(c04_sig.subject, '_', c04_sig.channel), 'stable');
c04_sig_u    = c04_sig(u_ia, :);
target_subs  = c04_sig_u.subject;
target_elecs = c04_sig_u.channel;
n_total_elecs = min(numel(target_subs), cfg.max_elecs);

% 构建解剖与坐标字典
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
fprintf('  【C17: Task 3 同形状内部红绿 Decoding 与三形状平均】\n');
fprintf('  目标电极总数: %d | 内部交叉验证: %d 折 | 置换次数: %d\n', ...
    n_total_elecs, cfg.n_folds, cfg.n_perm);
fprintf('  图片保存目录: %s\n', fig_dir);
fprintf('  时程保存目录: %s\n', tc_tab_dir);
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

t_centers = cfg.t_range(1) : cfg.win_step : cfg.t_range(2);
n_win = numel(t_centers);
global_idx = 0;

shape_cols = [
    0.20, 0.55, 0.85;  % Shape 1: 钴蓝
    0.22, 0.68, 0.42;  % Shape 2: 翠绿
    0.65, 0.35, 0.75   % Shape 3: 罗兰紫
];
col_avg = [0.88, 0.35, 0.05]; % 平均曲线: 陶土橙红

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
    
    % 仅提取红绿纯色试次
    rg_mask = strcmp(ti.color, 'red') | strcmp(ti.color, 'green');
    ti_rg   = ti(rg_mask, :);
    n_rg    = height(ti_rg);
    
    % 形状索引 (pic_id == 1, 2, 3)
    shapes = unique(ti_rg.pic_id);
    n_shapes = numel(shapes);
    
    % 构建各形状内部的 5 折分层交叉验证掩码
    shape_meta = cell(n_shapes, 1);
    for s = 1:n_shapes
        s_id = shapes(s);
        s_idx = find(ti_rg.pic_id == s_id);
        y_s = strcmp(ti_rg.color(s_idx), 'red'); % 1: red, 0: green
        
        rng(42 + s);
        r_sub = find(y_s == 1);
        g_sub = find(y_s == 0);
        cv_r = cvpartition(numel(r_sub), 'KFold', cfg.n_folds);
        cv_g = cvpartition(numel(g_sub), 'KFold', cfg.n_folds);
        
        tr_m = false(numel(s_idx), cfg.n_folds);
        te_m = false(numel(s_idx), cfg.n_folds);
        for f = 1:cfg.n_folds
            te_all = [r_sub(test(cv_r, f)); g_sub(test(cv_g, f))];
            te_m(te_all, f) = true;
            tr_m(:, f) = ~te_m(:, f);
        end
        
        sm = struct();
        sm.s_id = s_id;
        sm.s_idx = s_idx;
        sm.y = double(y_s);
        sm.y_pm1 = 2 * double(y_s) - 1; % +1/-1
        sm.tr_m = tr_m;
        sm.te_m = te_m;
        shape_meta{s} = sm;
    end
    
    fprintf('\n>>> [被试 %d/%d: %s] 载入完成 (%.2fs), 包含 %d 纯色试次 (3种形状), 共 %d 个显著通道\n', ...
        s_i, numel(unique_subs), sub_id, toc(t_load), n_rg, n_sub_elecs);
    
    for e_i = 1:n_sub_elecs
        ch_name = sub_elecs{e_i};
        global_idx = global_idx + 1;
        
        fig_png = fullfile(fig_dir, sprintf('%s_%s_task3_within_shape_decoding.png', sub_id, ch_name));
        tc_csv  = fullfile(tc_tab_dir, sprintf('%s_%s_task3_within_shape_timecourse.csv', sub_id, ch_name));
        
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
            
            % 预计算每个形状、每个窗口、每个折的标准化投影矩阵
            P_cell = cell(n_shapes, n_win, cfg.n_folds);
            Xte_cell = cell(n_shapes, n_win, cfg.n_folds);
            for s = 1:n_shapes
                sm = shape_meta{s};
                X_s = double(X_3d(sm.s_idx, :, :));
                for w = 1:n_win
                    X_w = squeeze(X_s(:, :, w));
                    for f = 1:cfg.n_folds
                        tr_idx = sm.tr_m(:, f);
                        te_idx = sm.te_m(:, f);
                        mu = mean(X_w(tr_idx, :), 1);
                        sig = std(X_w(tr_idx, :), 0, 1); sig(sig < 1e-6) = 1;
                        
                        Xtr = [(X_w(tr_idx, :) - mu) ./ sig, ones(sum(tr_idx), 1)];
                        Xte = [(X_w(te_idx, :) - mu) ./ sig, ones(sum(te_idx), 1)];
                        
                        I_reg = eye(size(Xtr, 2)); I_reg(end, end) = 0; % 截距不正则
                        P_cell{s, w, f} = (Xtr' * Xtr + cfg.ridge_lambda * I_reg) \ Xtr';
                        Xte_cell{s, w, f} = Xte;
                    end
                end
            end
            
            % 1. 计算真实三形状独立解码与平均解码曲线
            real_acc_shapes = zeros(n_shapes, n_win);
            for s = 1:n_shapes
                sm = shape_meta{s};
                for w = 1:n_win
                    f_acc = zeros(1, cfg.n_folds);
                    for f = 1:cfg.n_folds
                        w_vec = P_cell{s, w, f} * sm.y_pm1(sm.tr_m(:, f));
                        pred = Xte_cell{s, w, f} * w_vec;
                        y_te = sm.y_pm1(sm.te_m(:, f));
                        
                        sens_d = sum(y_te == 1);  sens = sum(y_te == 1 & pred > 0) / max(1, sens_d);
                        spec_d = sum(y_te == -1); spec = sum(y_te == -1 & pred <= 0) / max(1, spec_d);
                        f_acc(f) = (sens + spec) / 2;
                    end
                    real_acc_shapes(s, w) = mean(f_acc);
                end
            end
            real_acc_avg = mean(real_acc_shapes, 1);
            
            % 2. 200 次置换检验 (在各形状内部独立打乱标签)
            null_dist = zeros(cfg.n_perm, n_win);
            parfor p_i = 1:cfg.n_perm
                p_shapes = zeros(n_shapes, n_win);
                for s = 1:n_shapes
                    sm = shape_meta{s};
                    % 形状内部置换标签
                    y_perm = sm.y_pm1(randperm(numel(sm.y_pm1)));
                    for w = 1:n_win
                        f_acc = zeros(1, cfg.n_folds);
                        for f = 1:cfg.n_folds
                            w_vec = P_cell{s, w, f} * y_perm(sm.tr_m(:, f));
                            pred = Xte_cell{s, w, f} * w_vec;
                            y_te = y_perm(sm.te_m(:, f));
                            
                            sens_d = sum(y_te == 1);  sens = sum(y_te == 1 & pred > 0) / max(1, sens_d);
                            spec_d = sum(y_te == -1); spec = sum(y_te == -1 & pred <= 0) / max(1, spec_d);
                            f_acc(f) = (sens + spec) / 2;
                        end
                        p_shapes(s, w) = mean(f_acc);
                    end
                end
                null_dist(p_i, :) = mean(p_shapes, 1);
            end
            
            % 3. 曲线高斯平滑
            real_acc_avg_s = smoothdata(real_acc_avg, cfg.smooth_typ, cfg.smooth_pts);
            real_acc_shapes_s = zeros(size(real_acc_shapes));
            for s = 1:n_shapes
                real_acc_shapes_s(s, :) = smoothdata(real_acc_shapes(s, :), cfg.smooth_typ, cfg.smooth_pts);
            end
            null_dist_s = smoothdata(null_dist, 2, cfg.smooth_typ, cfg.smooth_pts);
            
            % 4. 统计检验与时间簇质量检验 (Cluster-Mass FWE)
            p_pointwise = (1 + sum(null_dist_s >= real_acc_avg_s, 1)) / (1 + cfg.n_perm);
            sig_mask_pts = (p_pointwise < cfg.cluster_alpha) & (t_centers >= 0);
            
            clusters = []; in_c = false; c_s = 1;
            for w = 1:n_win
                if sig_mask_pts(w) && ~in_c, in_c = true; c_s = w;
                elseif ~sig_mask_pts(w) && in_c, in_c = false; clusters = [clusters; c_s, w-1]; %#ok<AGROW>
                end
            end
            if in_c, clusters = [clusters; c_s, n_win]; end
            
            n_cl = size(clusters, 1);
            real_masses = zeros(n_cl, 1);
            for c_i = 1:n_cl
                real_masses(c_i) = sum(real_acc_avg_s(clusters(c_i,1):clusters(c_i,2)) - 0.5);
            end
            
            null_max_m = zeros(cfg.n_perm, 1);
            for p = 1:cfg.n_perm
                c_p = null_dist_s(p, :);
                p_p = (1 + sum(null_dist_s >= c_p, 1)) / (1 + cfg.n_perm);
                s_m = (p_p < cfg.cluster_alpha) & (t_centers >= 0);
                cls = []; in_cc = false; cc_s = 1;
                for w = 1:n_win
                    if s_m(w) && ~in_cc, in_cc = true; cc_s = w;
                    elseif ~s_m(w) && in_cc, in_cc = false; cls = [cls; cc_s, w-1]; %#ok<AGROW>
                    end
                end
                if in_cc, cls = [cls; cc_s, n_win]; end
                if isempty(cls), null_max_m(p) = 0;
                else
                    m_arr = zeros(size(cls, 1), 1);
                    for ci = 1:size(cls, 1), m_arr(ci) = sum(c_p(cls(ci,1):cls(ci,2)) - 0.5); end
                    null_max_m(p) = max(m_arr);
                end
            end
            
            cluster_pvals = zeros(n_cl, 1);
            for c_i = 1:n_cl
                cluster_pvals(c_i) = (1 + sum(null_max_m >= real_masses(c_i))) / (1 + cfg.n_perm);
            end
            
            % 5. 导出时程 CSV
            tc_tbl = table(repmat(string(sub_id), n_win, 1), repmat(string(ch_name), n_win, 1), ...
                t_centers', real_acc_avg_s', p_pointwise', ...
                real_acc_shapes_s(1,:)', real_acc_shapes_s(2,:)', real_acc_shapes_s(3,:)', ...
                mean(null_dist_s, 1)', prctile(null_dist_s, 97.5, 1)', prctile(null_dist_s, 2.5, 1)', ...
                'VariableNames', {'subject', 'channel', 'time_ms', 'acc_avg_smoothed', 'p_pointwise', ...
                                  'acc_shape1_smoothed', 'acc_shape2_smoothed', 'acc_shape3_smoothed', ...
                                  'null_mean', 'null_ci_upper', 'null_ci_lower'});
            writetable(tc_tbl, tc_csv);
            
            % 6. 绘制规范图 (左图: 3种形状 + 平均曲线; 右图: 峰值柱状对比图)
            fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1200, 480]);
            
            % (1) 左图: 时程解码曲线
            subplot(1, 2, 1); hold on; grid off;
            set(gca, 'Box', 'off', 'FontSize', 11, 'LineWidth', 1.1);
            
            null_up  = prctile(null_dist_s, 97.5, 1);
            null_low = prctile(null_dist_s, 2.5, 1);
            fill([t_centers, fliplr(t_centers)], [null_low, fliplr(null_up)], ...
                [0.85, 0.85, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.6, 'HandleVisibility', 'off');
            
            yline(0.5, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.2, 'HandleVisibility', 'off');
            xline(0, '-', 'Color', [0.3, 0.3, 0.3], 'LineWidth', 1.0, 'HandleVisibility', 'off');
            
            % 显著簇高亮
            for c_i = 1:n_cl
                if cluster_pvals(c_i) < 0.05
                    fill([t_centers(clusters(c_i,1)), t_centers(clusters(c_i,2)), ...
                          t_centers(clusters(c_i,2)), t_centers(clusters(c_i,1))], ...
                         [0.35, 0.35, 0.85, 0.85], [1.0, 0.9, 0.7], 'EdgeColor', 'none', ...
                         'FaceAlpha', 0.35, 'HandleVisibility', 'off');
                end
            end
            
            % 逐点显著标记
            sig_p_idx = find(p_pointwise < 0.05 & t_centers >= 0);
            if ~isempty(sig_p_idx)
                plot(t_centers(sig_p_idx), repmat(0.38, 1, numel(sig_p_idx)), 's', ...
                    'MarkerFaceColor', col_avg, 'MarkerEdgeColor', 'none', ...
                    'MarkerSize', 4, 'HandleVisibility', 'off');
            end
            
            % 绘制 3 种形状的单条虚线
            for s = 1:n_shapes
                plot(t_centers, real_acc_shapes_s(s, :), 'Color', shape_cols(s, :), ...
                    'LineWidth', 1.3, 'LineStyle', ':', 'DisplayName', sprintf('Shape %d (pic0%d)', s, s));
            end
            
            % 绘制 3 种形状平均线 (加粗实线)
            plot(t_centers, real_acc_avg_s, 'Color', col_avg, 'LineWidth', 2.8, ...
                'DisplayName', 'Shape-Averaged (Mean)');
            
            xlim(cfg.t_range); ylim([0.35, 0.85]);
            xlabel('Time from stimulus onset (ms)', 'FontWeight', 'bold');
            ylabel('Balanced Accuracy', 'FontWeight', 'bold');
            legend('Location', 'northwest', 'Box', 'off', 'FontSize', 9, 'Interpreter', 'none');
            
            % (2) 右图: 峰值对比柱状图
            subplot(1, 2, 2); hold on; grid off;
            set(gca, 'Box', 'off', 'FontSize', 11, 'LineWidth', 1.1);
            
            b_names = {'Shape-Averaged', 'Shape 1', 'Shape 2', 'Shape 3'};
            b_vals  = [max(real_acc_avg_s), max(real_acc_shapes_s(1,:)), max(real_acc_shapes_s(2,:)), max(real_acc_shapes_s(3,:))];
            b_cols  = [col_avg; shape_cols];
            
            bh = bar(1:4, b_vals, 0.65, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.9);
            for k = 1:4, bh.CData(k, :) = b_cols(k, :); end
            yline(0.5, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.2);
            
            for k = 1:4
                text(k, b_vals(k) + 0.015, sprintf('%.1f%%', b_vals(k) * 100), ...
                    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9.5);
            end
            set(gca, 'XTick', 1:4, 'XTickLabel', b_names, 'XTickLabelRotation', 20, 'TickLabelInterpreter', 'none');
            ylim([0.40, 0.85]); ylabel('Peak Balanced Accuracy', 'FontWeight', 'bold');
            
            sgtitle(sprintf('%s - %s (Task 3 Within-Shape Color Decoding)', sub_id, ch_name), ...
                'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
            
            % 导出 PNG (不需要 FIG)
            saveas(fig, fig_png);
            close(fig);
            
            fprintf('  [%d/%d] %s-%s 完成 (三形状平均峰值: %.1f%%, 耗时 %.2fs)\n', ...
                global_idx, n_total_elecs, sub_id, ch_name, max(real_acc_avg_s)*100, toc(t_elec));
        else
            % 读取已有数据组装汇总
            tc_data = readtable(tc_csv);
            real_acc_avg_s = tc_data.acc_avg_smoothed';
            real_acc_shapes_s = [tc_data.acc_shape1_smoothed'; tc_data.acc_shape2_smoothed'; tc_data.acc_shape3_smoothed'];
            p_pointwise = tc_data.p_pointwise';
            cluster_pvals = 1;
        end
        
        % 记录汇总条目
        [max_avg, best_w] = max(real_acc_avg_s);
        rec = struct();
        rec.subject = string(sub_id);
        rec.channel = string(ch_name);
        
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
        
        rec.peak_acc_avg     = max_avg;
        rec.peak_time_ms     = t_centers(best_w);
        rec.peak_p_pointwise = p_pointwise(best_w);
        rec.has_sig_cluster  = any(cluster_pvals < 0.05);
        rec.peak_acc_shape1  = max(real_acc_shapes_s(1, :));
        rec.peak_acc_shape2  = max(real_acc_shapes_s(2, :));
        rec.peak_acc_shape3  = max(real_acc_shapes_s(3, :));
        
        summary_list = [summary_list; rec]; %#ok<AGROW>
    end
end

%% 5. 导出总览汇总表
if ~isempty(summary_list)
    sum_tbl = struct2table(summary_list);
    writetable(sum_tbl, sum_csv);
    save(sum_mat, 'sum_tbl', 'cfg');
    fprintf('\n========================================================================\n');
    fprintf('  【Task 3 同形状内解码全量批处理完成！】\n');
    fprintf('  汇总 CSV: %s (共 %d 个通道)\n', sum_csv, height(sum_tbl));
    fprintf('  汇总 MAT: %s\n', sum_mat);
    fprintf('  严格显著通道数: %d\n', sum(sum_tbl.has_sig_cluster == 1));
    fprintf('========================================================================\n');
end
