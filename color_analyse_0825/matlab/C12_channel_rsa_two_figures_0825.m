%% ========================================================================
% 脚本名称: C12_channel_rsa_two_figures_0825.m
% 功能:
%   1. 【单电极跨任务多频段 RSA 与 3D 动态轨迹分析】
%   2. 对每个目标位点绘制两个核心图表 (同时保存为 .fig 与 .png):
%      - 图 1 (平均图):
%          * 左图: 10 类别 RDM (4 种灰色水果 + 2 种纯色 * 3 种形状)
%          * 中间图: 4 类别 RDM (相同灰色记忆平均, 相同纯色色块平均)
%          * 右图: 3D MDS 空间流形分布 (交互式 3D 散点图)
%      - 图 2 (时间轨迹图 - 3D 空间动态演化):
%          * 左图: 10 类别 3D 神经状态空间时间轨迹 (-200 到 800 ms)
%          * 中间图: 4 类别 3D 神经状态空间时间轨迹 (红绿记忆色 vs 红绿物理色)
%          * 右图: 4 类别 2D 主平面投影轨迹 (PC1 vs PC2 俯视图带时间流向箭头)
%   3. 支持断点续跑 (cfg.skip_existing = true)
%   4. 按被试批量流式加载 (每个被试 6 频段仅读入内存一次，极速处理)
% ========================================================================

clear; clc; close all;

%% 1. 参数与路径配置 (置顶直观，简写平铺)
cfg = struct();
cfg.subjects       = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008'};
cfg.target_mode    = 'all_significant'; % 'all_significant' (Task 1 所有显著电极, 229个), 'concordant' (157个), 或 'custom'
cfg.custom_subs    = {'sub008'};
cfg.custom_elecs   = {'C10'};
cfg.max_elecs      = 500;               % 最多处理通道数上限

% 时间分析参数
cfg.avg_win_ms     = [100, 600];   % 时间窗平均的时间范围 (ms)
cfg.slide_win_len  = 50;           % 滑动时间窗长度 (ms)
cfg.slide_win_step = 20;           % 滑动时间窗步长 (ms)
cfg.t_range        = [-200, 800];  % 时程分析总范围 (ms)
cfg.smooth_pts     = 3;            % 时程平滑点数 (高斯平滑)

% 输出控制
cfg.skip_existing  = true;         % 跳过已处理通道 (断点续跑)
cfg.save_fig       = true;         % 保存 .fig 交互式图表 (用户明确要求)
cfg.save_png       = true;         % 保存 300 DPI 高清学术图
cfg.dpi            = 300;

% 频段定义
cfg.bands   = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.n_bands = numel(cfg.bands);

% 路径设置
script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(script_dir);
data_root  = fullfile(proj_root, 'process_data_new');
task_info  = fullfile(proj_root, 'task_info');
res_table  = fullfile(proj_root, 'result', 'tables', 'color_effects_summary.mat');
out_fig_dir = fullfile(proj_root, 'result', 'figures', 'rsa_per_electrode');
out_tab_dir = fullfile(proj_root, 'result', 'tables');

if ~exist(out_fig_dir, 'dir'), mkdir(out_fig_dir); end
if ~exist(out_tab_dir, 'dir'), mkdir(out_tab_dir); end

fprintf('========================================================================\n');
fprintf('  【C12: 单位点多频段 RSA 与 3D 神经状态空间时间轨迹分析】  \n');
fprintf('========================================================================\n');
fprintf('[+] 目标模式: %s | 平均时间窗: [%d, %d] ms\n', cfg.target_mode, cfg.avg_win_ms(1), cfg.avg_win_ms(2));
fprintf('[+] 滑动窗口: %d ms | 步长: %d ms | 总范围: [%d, %d] ms\n', ...
    cfg.slide_win_len, cfg.slide_win_step, cfg.t_range(1), cfg.t_range(2));

%% 2. 筛选目标通道
if strcmp(cfg.target_mode, 'custom')
    target_subs  = cfg.custom_subs(:);
    target_elecs = cfg.custom_elecs(:);
    n_targets    = numel(target_subs);
    fprintf('[+] 自定义电极模式，共 %d 个通道待分析。\n', n_targets);
else
    if ~isfile(res_table)
        error('未找到筛选汇总表: %s', res_table);
    end
    d_c04 = load(res_table);
    c04_tbl = d_c04.all_tbl;
    
    if strcmp(cfg.target_mode, 'concordant')
        concord_mask = (c04_tbl.is_significant == 1) & ...
            (strcmp(c04_tbl.concordance_type, 'Concordant_Positive') | ...
             strcmp(c04_tbl.concordance_type, 'Concordant_Negative'));
        sel_tbl = c04_tbl(concord_mask, :);
        fprintf('[+] 筛选模式: 【Task 1 四类别同向显著电极】\n');
    else
        % 默认: all_significant (Task 1 所有总体显著电极, is_significant == 1)
        sig_mask = (c04_tbl.is_significant == 1);
        sel_tbl  = c04_tbl(sig_mask, :);
        fprintf('[+] 筛选模式: 【Task 1 所有显著电极 (is_significant == 1)】\n');
    end
    
    elec_keys = strcat(sel_tbl.subject, '_', sel_tbl.channel);
    [~, u_ia] = unique(elec_keys, 'stable');
    
    target_subs  = sel_tbl.subject(u_ia);
    target_elecs = sel_tbl.channel(u_ia);
    n_targets    = min(numel(target_subs), cfg.max_elecs);
    fprintf('[+] 共筛选出 %d 个目标通道待分析。\n', n_targets);
    
    % 构建解剖与坐标映射字典
    anat_map = containers.Map();
    for r = 1:height(c04_tbl)
        k = sprintf('%s_%s', c04_tbl.subject{r}, c04_tbl.channel{r});
        if ~isKey(anat_map, k)
            s_info = struct();
            if ismember('dkt_anatomy', c04_tbl.Properties.VariableNames), s_info.dkt = char(c04_tbl.dkt_anatomy(r)); else, s_info.dkt = ''; end
            if ismember('aal_anatomy', c04_tbl.Properties.VariableNames), s_info.aal = char(c04_tbl.aal_anatomy(r)); else, s_info.aal = ''; end
            if ismember('stream_hierarchy', c04_tbl.Properties.VariableNames), s_info.stream = char(c04_tbl.stream_hierarchy(r)); else, s_info.stream = ''; end
            if ismember('mni_x', c04_tbl.Properties.VariableNames), s_info.x = c04_tbl.mni_x(r); else, s_info.x = NaN; end
            if ismember('mni_y', c04_tbl.Properties.VariableNames), s_info.y = c04_tbl.mni_y(r); else, s_info.y = NaN; end
            if ismember('mni_z', c04_tbl.Properties.VariableNames), s_info.z = c04_tbl.mni_z(r); else, s_info.z = NaN; end
            anat_map(k) = s_info;
        end
    end
end

% 类别名称与配色定义
cond_names_10 = {
    'Gray Strawberry', 'Gray Watermelon', 'Gray Kiwi', 'Gray Cabbage', ...
    'Red Shape 1', 'Red Shape 2', 'Red Shape 3', ...
    'Green Shape 1', 'Green Shape 2', 'Green Shape 3'
};
cond_names_4 = {
    'Red Memory (Gray)', ...
    'Green Memory (Gray)', ...
    'Red Patch (Pure)', ...
    'Green Patch (Pure)'
};

col_red = [0.85, 0.15, 0.15];
col_grn = [0.15, 0.70, 0.25];
colors_10 = {
    [0.92, 0.40, 0.35], [0.85, 0.20, 0.20], ... % Straw, Water (Red Memo)
    [0.45, 0.80, 0.40], [0.15, 0.65, 0.30], ... % Kiwi, Cabb (Grn Memo)
    col_red, [0.95, 0.30, 0.10], [0.75, 0.10, 0.10], ... % Red Shapes 1, 2, 3
    col_grn, [0.10, 0.60, 0.35], [0.10, 0.50, 0.20]  ... % Grn Shapes 1, 2, 3
};
markers_10 = {'o', 'o', 'o', 'o', 's', 'd', '^', 's', 'd', '^'};

colors_4 = {
    [0.90, 0.35, 0.30], ... % Red Memory (Gray Fruit)
    [0.30, 0.75, 0.35], ... % Green Memory (Gray Fruit)
    [0.85, 0.10, 0.10], ... % Red Pure (Patch)
    [0.10, 0.65, 0.20]      % Green Pure (Patch)
};

%% 3. 按被试分组批量流式处理
unique_subs = unique(target_subs(1:n_targets), 'stable');
summary_list = struct([]);
global_count = 0;

for s_i = 1:numel(unique_subs)
    sub_id = unique_subs{s_i};
    sub_mask = strcmp(target_subs(1:n_targets), sub_id);
    sub_elecs = target_elecs(sub_mask);
    n_sub_elecs = numel(sub_elecs);
    
    fprintf('\n>>> [被试 %d/%d: %s] 开始处理本被试 %d 个目标通道 ...\n', ...
        s_i, numel(unique_subs), sub_id, n_sub_elecs);
    
    t3_mat = fullfile(data_root, sub_id, 'task3_multiband_epoched.mat');
    t2_mat = fullfile(data_root, sub_id, 'task2_multiband_epoched.mat');
    if ~isfile(t3_mat) || ~isfile(t2_mat)
        warning('被试 %s 数据文件缺失，跳过。', sub_id);
        continue;
    end
    
    % 加载元数据与条件掩码
    t_axis = double(h5read(t3_mat, '/epoched_data/time_ms')); t_axis = t_axis(:)';
    ch_list3 = h5read(t3_mat, '/epoched_data/channels');
    ch_list2 = h5read(t2_mat, '/epoched_data/channels');
    if iscell(ch_list3), ch_list3 = cellfun(@(x) char(x(:)'), ch_list3, 'UniformOutput', false); end
    if iscell(ch_list2), ch_list2 = cellfun(@(x) char(x(:)'), ch_list2, 'UniformOutput', false); end
    
    d3 = load(fullfile(task_info, sub_id, 'task3_trial_info.mat')); ti3 = d3.trial_info;
    d2 = load(fullfile(task_info, sub_id, 'task2_trial_info.mat')); ti2 = d2.trial_info;
    
    % 10 类别掩码
    cond_masks_10 = cell(10, 1);
    cond_masks_10{1} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'strawberry');
    cond_masks_10{2} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'watermelon');
    cond_masks_10{3} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'kiwi');
    cond_masks_10{4} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'cabbage');
    cond_masks_10{5} = strcmp(ti3.color, 'red') & (ti3.pic_id == 1);
    cond_masks_10{6} = strcmp(ti3.color, 'red') & (ti3.pic_id == 2);
    cond_masks_10{7} = strcmp(ti3.color, 'red') & (ti3.pic_id == 3);
    cond_masks_10{8} = strcmp(ti3.color, 'green') & (ti3.pic_id == 1);
    cond_masks_10{9} = strcmp(ti3.color, 'green') & (ti3.pic_id == 2);
    cond_masks_10{10} = strcmp(ti3.color, 'green') & (ti3.pic_id == 3);
    
    % 4 类别掩码
    cond_masks_4 = cell(4, 1);
    cond_masks_4{1} = strcmp(ti2.state, 'gray') & (strcmp(ti2.fruit, 'strawberry') | strcmp(ti2.fruit, 'watermelon'));
    cond_masks_4{2} = strcmp(ti2.state, 'gray') & (strcmp(ti2.fruit, 'kiwi') | strcmp(ti2.fruit, 'cabbage'));
    cond_masks_4{3} = strcmp(ti3.color, 'red');
    cond_masks_4{4} = strcmp(ti3.color, 'green');
    
    % 一次性预载入该被试 6 频段数据至 RAM
    t_load = tic;
    raw2_bands = cell(cfg.n_bands, 1);
    raw3_bands = cell(cfg.n_bands, 1);
    for b = 1:cfg.n_bands
        raw2_bands{b} = h5read(t2_mat, ['/epoched_data/' cfg.bands{b}]);
        raw3_bands{b} = h5read(t3_mat, ['/epoched_data/' cfg.bands{b}]);
    end
    fprintf('    [+] 数据载入完成 (耗时 %.2f 秒)\n', toc(t_load));
    
    % 时间窗口划分
    t_starts = cfg.t_range(1) : cfg.slide_win_step : (cfg.t_range(2) - cfg.slide_win_len);
    n_win = numel(t_starts);
    t_centers = t_starts + cfg.slide_win_len / 2;
    t_avg_mask = (t_axis >= cfg.avg_win_ms(1)) & (t_axis <= cfg.avg_win_ms(2));
    
    idx_start = 1;
    idx_0ms   = find(t_centers >= 0, 1);
    idx_200ms = find(t_centers >= 200, 1);
    idx_400ms = find(t_centers >= 400, 1);
    idx_end   = n_win;
    
    % 逐通道分析
    for e_i = 1:n_sub_elecs
        elec = sub_elecs{e_i};
        global_count = global_count + 1;
        
        fig1_fig = fullfile(out_fig_dir, sprintf('%s_%s_rsa_window_avg.fig', sub_id, elec));
        fig1_png = fullfile(out_fig_dir, sprintf('%s_%s_rsa_window_avg.png', sub_id, elec));
        fig2_fig = fullfile(out_fig_dir, sprintf('%s_%s_rsa_3d_trajectory.fig', sub_id, elec));
        fig2_png = fullfile(out_fig_dir, sprintf('%s_%s_rsa_3d_trajectory.png', sub_id, elec));
        
        need_render = ~(cfg.skip_existing && isfile(fig1_png) && isfile(fig2_png) && isfile(fig1_fig) && isfile(fig2_fig));
        
        t_ch = tic;
        e_idx3 = find(strcmp(ch_list3, elec), 1);
        e_idx2 = find(strcmp(ch_list2, elec), 1);
        if isempty(e_idx3) || isempty(e_idx2), continue; end
        
        % 提取通道数据
        X3 = zeros(height(ti3), cfg.n_bands, numel(t_axis));
        X2 = zeros(height(ti2), cfg.n_bands, numel(t_axis));
        for b = 1:cfg.n_bands
            X3(:, b, :) = raw3_bands{b}(:, e_idx3, :);
            X2(:, b, :) = raw2_bands{b}(:, e_idx2, :);
        end
        
        % -----------------------------------------------------------------
        % 1. 计算时间窗平均特征与 RDM
        % -----------------------------------------------------------------
        f10_avg = zeros(10, cfg.n_bands * sum(t_avg_mask));
        for c = 1:10
            if c <= 4
                sig = squeeze(mean(X2(cond_masks_10{c}, :, t_avg_mask), 1));
            else
                sig = squeeze(mean(X3(cond_masks_10{c}, :, t_avg_mask), 1));
            end
            f10_avg(c, :) = sig(:)';
        end
        rdm_10_avg = 1 - corr(f10_avg');
        
        f4_avg = zeros(4, cfg.n_bands * sum(t_avg_mask));
        for c = 1:4
            if c <= 2
                sig = squeeze(mean(X2(cond_masks_4{c}, :, t_avg_mask), 1));
            else
                sig = squeeze(mean(X3(cond_masks_4{c}, :, t_avg_mask), 1));
            end
            f4_avg(c, :) = sig(:)';
        end
        rdm_4_avg = 1 - corr(f4_avg');
        
        % -----------------------------------------------------------------
        % 2. 计算滑动时间窗特征与 3D 轨迹
        % -----------------------------------------------------------------
        feat_10 = zeros(10, n_win, cfg.n_bands);
        for w = 1:n_win
            w_t1 = t_starts(w); w_t2 = w_t1 + cfg.slide_win_len;
            t_m = (t_axis >= w_t1) & (t_axis < w_t2);
            for c = 1:10
                if c <= 4
                    sig = mean(mean(X2(cond_masks_10{c}, :, t_m), 1), 3);
                else
                    sig = mean(mean(X3(cond_masks_10{c}, :, t_m), 1), 3);
                end
                feat_10(c, w, :) = sig(:)';
            end
        end
        
        feat_4 = zeros(4, n_win, cfg.n_bands);
        for w = 1:n_win
            w_t1 = t_starts(w); w_t2 = w_t1 + cfg.slide_win_len;
            t_m = (t_axis >= w_t1) & (t_axis < w_t2);
            for c = 1:4
                if c <= 2
                    sig = mean(mean(X2(cond_masks_4{c}, :, t_m), 1), 3);
                else
                    sig = mean(mean(X3(cond_masks_4{c}, :, t_m), 1), 3);
                end
                feat_4(c, w, :) = sig(:)';
            end
        end
        
        % 统一 PCA 构建 3D 状态空间
        mat_10_flat = reshape(permute(feat_10, [2, 1, 3]), [n_win * 10, cfg.n_bands]);
        mu_feat  = mean(mat_10_flat, 1);
        std_feat = std(mat_10_flat, 0, 1);
        std_feat(std_feat < 1e-6) = 1;
        mat_10_norm = (mat_10_flat - mu_feat) ./ std_feat;
        
        [coeff, ~, ~, ~, explained] = pca(mat_10_norm);
        var_exp = explained(1:3);
        
        % 10 类别与 4 类别 3D 轨迹计算及平滑
        traj_10 = zeros(10, n_win, 3);
        for c = 1:10
            c_feat = squeeze(feat_10(c, :, :));
            c_norm = (c_feat - mu_feat) ./ std_feat;
            pts_3d = c_norm * coeff(:, 1:3);
            for d = 1:3
                traj_10(c, :, d) = smoothdata(pts_3d(:, d), 'gaussian', cfg.smooth_pts);
            end
        end
        
        traj_4 = zeros(4, n_win, 3);
        for c = 1:4
            c_feat = squeeze(feat_4(c, :, :));
            c_norm = (c_feat - mu_feat) ./ std_feat;
            pts_3d = c_norm * coeff(:, 1:3);
            for d = 1:3
                traj_4(c, :, d) = smoothdata(pts_3d(:, d), 'gaussian', cfg.smooth_pts);
            end
        end
        
        % -----------------------------------------------------------------
        % 3. 绘制图 1: 平均图 (左 10 RDM, 中 4 RDM, 右 3D MDS)
        % -----------------------------------------------------------------
        if need_render
            h1 = figure('Units', 'pixels', 'Position', [50, 100, 1650, 500], 'Color', 'w', 'Visible', 'off');
            
            subplot(1, 3, 1);
            imagesc(rdm_10_avg); colormap(gca, 'parula');
            cb = colorbar; ylabel(cb, 'Distance (1 - r)', 'FontSize', 9, 'FontWeight', 'bold');
            caxis([0, max(rdm_10_avg(:))]); axis square;
            set(gca, 'XTick', 1:10, 'XTickLabel', cond_names_10, 'XTickLabelRotation', 45, ...
                     'YTick', 1:10, 'YTickLabel', cond_names_10, 'FontSize', 8, 'TickDir', 'out');
            title({'A. 10-Condition RDM (Averaged [100, 600] ms)', ...
                   '(4 Gray Fruits + 2 Colors * 3 Shapes)'}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
            hold on;
            xline(4.5, 'w-', 'LineWidth', 2.0); yline(4.5, 'w-', 'LineWidth', 2.0);
            xline(7.5, 'w--', 'LineWidth', 1.2); yline(7.5, 'w--', 'LineWidth', 1.2);
            hold off;
            
            subplot(1, 3, 2);
            imagesc(rdm_4_avg); colormap(gca, 'parula');
            cb = colorbar; ylabel(cb, 'Distance (1 - r)', 'FontSize', 9, 'FontWeight', 'bold');
            caxis([0, max(rdm_4_avg(:))]); axis square;
            set(gca, 'XTick', 1:4, 'XTickLabel', cond_names_4, 'XTickLabelRotation', 35, ...
                     'YTick', 1:4, 'YTickLabel', cond_names_4, 'FontSize', 8.5, 'TickDir', 'out');
            title({'B. Collapsed 4-Condition RDM', ...
                   '(Mean Gray Memory & Mean Pure Color)'}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
            hold on;
            xline(2.5, 'w-', 'LineWidth', 2.0); yline(2.5, 'w-', 'LineWidth', 2.0);
            hold off;
            
            subplot(1, 3, 3);
            [Y_3d, eigvals] = cmdscale(rdm_10_avg, 3);
            if size(Y_3d, 2) < 3
                Y_3d = [Y_3d, zeros(10, 3 - size(Y_3d, 2))];
                var_exp_mds = [100, 0, 0];
            else
                var_exp_mds = 100 * eigvals(1:3) / max(1e-6, sum(abs(eigvals(eigvals > 0))));
            end
            
            hold on; grid on; box on;
            z_base = min(Y_3d(:,3)) - 0.10;
            for i = 1:10
                plot3([Y_3d(i,1), Y_3d(i,1)], [Y_3d(i,2), Y_3d(i,2)], [z_base, Y_3d(i,3)], ':', ...
                    'Color', [0.75, 0.75, 0.75], 'LineWidth', 0.8, 'HandleVisibility', 'off');
                scatter3(Y_3d(i,1), Y_3d(i,2), Y_3d(i,3), 110, markers_10{i}, 'filled', ...
                    'MarkerFaceColor', colors_10{i}, 'MarkerEdgeColor', [0.2, 0.2, 0.2], 'LineWidth', 1.0);
                text(Y_3d(i,1)+0.015, Y_3d(i,2)+0.015, Y_3d(i,3)+0.015, cond_names_10{i}, ...
                    'FontSize', 7.5, 'FontWeight', 'bold', 'Color', colors_10{i}*0.8, 'Interpreter', 'none');
            end
            view(38, 22);
            xlabel(sprintf('Dim 1 (%.1f%%)', var_exp_mds(1)), 'FontSize', 9, 'FontWeight', 'bold');
            ylabel(sprintf('Dim 2 (%.1f%%)', var_exp_mds(2)), 'FontSize', 9, 'FontWeight', 'bold');
            zlabel(sprintf('Dim 3 (%.1f%%)', var_exp_mds(3)), 'FontSize', 9, 'FontWeight', 'bold');
            title(sprintf('C. 3D Representational Space (%s-%s)', sub_id, elec), 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
            hold off;
            
            if cfg.save_fig
                set(h1, 'Visible', 'on');
                savefig(h1, fig1_fig);
            end
            if cfg.save_png
                try
                    exportgraphics(h1, fig1_png, 'Resolution', cfg.dpi);
                catch
                    pause(0.2);
                    try
                        exportgraphics(h1, fig1_png, 'Resolution', cfg.dpi);
                    catch
                        print(h1, fig1_png, '-dpng', sprintf('-r%d', cfg.dpi));
                    end
                end
            end
            close(h1);
        
        % -----------------------------------------------------------------
        % 4. 绘制图 2: 3D 神经状态空间时间轨迹图
        % -----------------------------------------------------------------
        h2 = figure('Units', 'pixels', 'Position', [50, 100, 1650, 520], 'Color', 'w', 'Visible', 'off');
        
        % 左图: 10 类别 3D 时间轨迹
        subplot(1, 3, 1);
        hold on; grid on; box on;
        for c = 1:10
            x = squeeze(traj_10(c, :, 1));
            y = squeeze(traj_10(c, :, 2));
            z = squeeze(traj_10(c, :, 3));
            
            if c <= 4
                plot3(x, y, z, 'LineWidth', 1.8, 'Color', colors_10{c}, 'LineStyle', '--', ...
                    'DisplayName', cond_names_10{c});
            else
                plot3(x, y, z, 'LineWidth', 2.0, 'Color', colors_10{c}, 'LineStyle', '-', ...
                    'DisplayName', cond_names_10{c});
            end
            
            scatter3(x(idx_start), y(idx_start), z(idx_start), 35, colors_10{c}, 'o', 'filled', ...
                'MarkerEdgeColor', [0.3, 0.3, 0.3], 'HandleVisibility', 'off');
            scatter3(x(idx_0ms), y(idx_0ms), z(idx_0ms), 55, colors_10{c}, '^', 'filled', ...
                'MarkerEdgeColor', [0.2, 0.2, 0.2], 'HandleVisibility', 'off');
            scatter3(x(idx_end), y(idx_end), z(idx_end), 45, colors_10{c}, 's', 'filled', ...
                'MarkerEdgeColor', [0.3, 0.3, 0.3], 'HandleVisibility', 'off');
        end
        view(38, 24);
        xlabel(sprintf('PC 1 (%.1f%%)', var_exp(1)), 'FontSize', 9, 'FontWeight', 'bold');
        ylabel(sprintf('PC 2 (%.1f%%)', var_exp(2)), 'FontSize', 9, 'FontWeight', 'bold');
        zlabel(sprintf('PC 3 (%.1f%%)', var_exp(3)), 'FontSize', 9, 'FontWeight', 'bold');
        title({'A. 10-Condition 3D Neural Trajectories', ...
               'Circle: -200ms | Triangle: 0ms | Square: 800ms'}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
        legend('Location', 'northeast', 'FontSize', 6.5, 'Box', 'off');
        hold off;
        
        % 中间图: 4 类别 3D 时间轨迹 (相同记忆平均, 相同纯色平均)
        subplot(1, 3, 2);
        hold on; grid on; box on;
        for c = 1:4
            x = squeeze(traj_4(c, :, 1));
            y = squeeze(traj_4(c, :, 2));
            z = squeeze(traj_4(c, :, 3));
            
            if c <= 2
                plot3(x, y, z, 'LineWidth', 2.8, 'Color', colors_4{c}, 'LineStyle', '--', ...
                    'DisplayName', cond_names_4{c});
            else
                plot3(x, y, z, 'LineWidth', 3.2, 'Color', colors_4{c}, 'LineStyle', '-', ...
                    'DisplayName', cond_names_4{c});
            end
            
            scatter3(x(idx_start), y(idx_start), z(idx_start), 45, colors_4{c}, 'o', 'filled', ...
                'MarkerEdgeColor', [0.2, 0.2, 0.2], 'HandleVisibility', 'off');
            scatter3(x(idx_0ms), y(idx_0ms), z(idx_0ms), 70, colors_4{c}, '^', 'filled', ...
                'MarkerEdgeColor', [0.1, 0.1, 0.1], 'HandleVisibility', 'off');
            scatter3(x(idx_200ms), y(idx_200ms), z(idx_200ms), 60, colors_4{c}, 'd', 'filled', ...
                'MarkerEdgeColor', [0.1, 0.1, 0.1], 'HandleVisibility', 'off');
            scatter3(x(idx_400ms), y(idx_400ms), z(idx_400ms), 60, colors_4{c}, 'p', 'filled', ...
                'MarkerEdgeColor', [0.1, 0.1, 0.1], 'HandleVisibility', 'off');
            scatter3(x(idx_end), y(idx_end), z(idx_end), 55, colors_4{c}, 's', 'filled', ...
                'MarkerEdgeColor', [0.2, 0.2, 0.2], 'HandleVisibility', 'off');
        end
        
        x_rp = squeeze(traj_4(3, :, 1));
        y_rp = squeeze(traj_4(3, :, 2));
        z_rp = squeeze(traj_4(3, :, 3));
        text(x_rp(idx_0ms)+0.1, y_rp(idx_0ms), z_rp(idx_0ms), ' 0ms', 'FontSize', 8, 'FontWeight', 'bold', 'Color', [0.2, 0.2, 0.2], 'Interpreter', 'none');
        text(x_rp(idx_200ms)+0.1, y_rp(idx_200ms), z_rp(idx_200ms), ' 200ms', 'FontSize', 8, 'FontWeight', 'bold', 'Color', [0.2, 0.2, 0.2], 'Interpreter', 'none');
        text(x_rp(idx_400ms)+0.1, y_rp(idx_400ms), z_rp(idx_400ms), ' 400ms', 'FontSize', 8, 'FontWeight', 'bold', 'Color', [0.2, 0.2, 0.2], 'Interpreter', 'none');
        
        view(38, 24);
        xlabel(sprintf('PC 1 (%.1f%%)', var_exp(1)), 'FontSize', 9, 'FontWeight', 'bold');
        ylabel(sprintf('PC 2 (%.1f%%)', var_exp(2)), 'FontSize', 9, 'FontWeight', 'bold');
        zlabel(sprintf('PC 3 (%.1f%%)', var_exp(3)), 'FontSize', 9, 'FontWeight', 'bold');
        title({'B. Collapsed 4-Category 3D Neural Trajectories', ...
               '0ms: Triangle | 200ms: Diamond | 400ms: Star | 800ms: Square'}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
        legend('Location', 'northeast', 'FontSize', 7.5, 'Box', 'off');
        hold off;
        
        % 右图: 4 类别 2D 主平面投影轨迹 (PC1 vs PC2 俯视图)
        subplot(1, 3, 3);
        hold on; grid on; box on;
        for c = 1:4
            x = squeeze(traj_4(c, :, 1));
            y = squeeze(traj_4(c, :, 2));
            
            if c <= 2
                plot(x, y, 'LineWidth', 2.5, 'Color', colors_4{c}, 'LineStyle', '--', ...
                    'DisplayName', cond_names_4{c});
            else
                plot(x, y, 'LineWidth', 3.0, 'Color', colors_4{c}, 'LineStyle', '-', ...
                    'DisplayName', cond_names_4{c});
            end
            
            scatter(x(idx_start), y(idx_start), 40, colors_4{c}, 'o', 'filled', 'MarkerEdgeColor', [0.2, 0.2, 0.2], 'HandleVisibility', 'off');
            scatter(x(idx_0ms), y(idx_0ms), 65, colors_4{c}, '^', 'filled', 'MarkerEdgeColor', [0.1, 0.1, 0.1], 'HandleVisibility', 'off');
            scatter(x(idx_200ms), y(idx_200ms), 55, colors_4{c}, 'd', 'filled', 'MarkerEdgeColor', [0.1, 0.1, 0.1], 'HandleVisibility', 'off');
            scatter(x(idx_400ms), y(idx_400ms), 55, colors_4{c}, 'p', 'filled', 'MarkerEdgeColor', [0.1, 0.1, 0.1], 'HandleVisibility', 'off');
            scatter(x(idx_end), y(idx_end), 50, colors_4{c}, 's', 'filled', 'MarkerEdgeColor', [0.2, 0.2, 0.2], 'HandleVisibility', 'off');
        end
        
        for c = 1:4
            x = squeeze(traj_4(c, :, 1));
            y = squeeze(traj_4(c, :, 2));
            for a_i = [round(n_win*0.35), round(n_win*0.65)]
                dx = x(a_i+1) - x(a_i-1);
                dy = y(a_i+1) - y(a_i-1);
                quiver(x(a_i), y(a_i), dx, dy, 0, 'Color', colors_4{c}*0.8, 'LineWidth', 1.5, ...
                    'MaxHeadSize', 2.0, 'HandleVisibility', 'off');
            end
        end
        
        xlabel(sprintf('PC 1 (%.1f%%)', var_exp(1)), 'FontSize', 9, 'FontWeight', 'bold');
        ylabel(sprintf('PC 2 (%.1f%%)', var_exp(2)), 'FontSize', 9, 'FontWeight', 'bold');
        title({'C. 2D Projection (PC1 vs PC2 Top-down View)', sprintf('(%s-%s State Evolution)', sub_id, elec)}, ...
            'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'none');
        legend('Location', 'northeast', 'FontSize', 7.5, 'Box', 'off');
        hold off;
        
            if cfg.save_fig
                set(h2, 'Visible', 'on');
                savefig(h2, fig2_fig);
            end
            if cfg.save_png
                try
                    exportgraphics(h2, fig2_png, 'Resolution', cfg.dpi);
                catch
                    pause(0.2);
                    try
                        exportgraphics(h2, fig2_png, 'Resolution', cfg.dpi);
                    catch
                        print(h2, fig2_png, '-dpng', sprintf('-r%d', cfg.dpi));
                    end
                end
            end
            close(h2);
        end % if need_render
        
        % 记录汇总指标
        d_phys_avg = rdm_4_avg(3, 4);
        d_memo_avg = rdm_4_avg(1, 2);
        d_same_avg = (rdm_4_avg(1, 3) + rdm_4_avg(2, 4)) / 2;
        d_diff_avg = (rdm_4_avg(1, 4) + rdm_4_avg(2, 3)) / 2;
        
        cur_idx = numel(summary_list) + 1;
        summary_list(cur_idx).subject    = sub_id;
        summary_list(cur_idx).channel    = elec;
        
        k_ch = sprintf('%s_%s', sub_id, elec);
        if exist('anat_map', 'var') && isKey(anat_map, k_ch)
            an = anat_map(k_ch);
            summary_list(cur_idx).dkt_anatomy = an.dkt;
            summary_list(cur_idx).aal_anatomy = an.aal;
            summary_list(cur_idx).stream      = an.stream;
            summary_list(cur_idx).mni_x       = an.x;
            summary_list(cur_idx).mni_y       = an.y;
            summary_list(cur_idx).mni_z       = an.z;
        else
            summary_list(cur_idx).dkt_anatomy = '';
            summary_list(cur_idx).aal_anatomy = '';
            summary_list(cur_idx).stream      = '';
            summary_list(cur_idx).mni_x       = NaN;
            summary_list(cur_idx).mni_y       = NaN;
            summary_list(cur_idx).mni_z       = NaN;
        end
        
        summary_list(cur_idx).d_phys_avg = d_phys_avg;
        summary_list(cur_idx).d_memo_avg = d_memo_avg;
        summary_list(cur_idx).d_same_avg = d_same_avg;
        summary_list(cur_idx).d_diff_avg = d_diff_avg;
        summary_list(cur_idx).net_replay = d_diff_avg - d_same_avg;
        
        if need_render
            fprintf('    [%d/%d | 全局 %d/%d] 通道 [%s-%s] 出图完成 (耗时 %.2f 秒) | Replay Net = %.4f\n', ...
                e_i, n_sub_elecs, global_count, n_targets, sub_id, elec, toc(t_ch), d_diff_avg - d_same_avg);
        else
            fprintf('    [%d/%d | 全局 %d/%d] 通道 [%s-%s] 图表已存在，已记录指标 (耗时 %.3f 秒) | Replay Net = %.4f\n', ...
                e_i, n_sub_elecs, global_count, n_targets, sub_id, elec, toc(t_ch), d_diff_avg - d_same_avg);
        end
    end
end

% 导出全局汇总表
if ~isempty(summary_list)
    tbl_out = struct2table(summary_list);
    csv_path = fullfile(out_tab_dir, 'rsa_channel_two_figures_summary.csv');
    writetable(tbl_out, csv_path);
    fprintf('\n========================================================================\n');
    fprintf('  【C12 处理完成】 汇总表已保存至: %s\n', csv_path);
    fprintf('========================================================================\n');
end
