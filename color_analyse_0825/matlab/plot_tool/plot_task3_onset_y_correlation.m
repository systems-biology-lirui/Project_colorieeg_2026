%% ========================================================================
% 脚本名称: plot_task3_onset_y_correlation.m
% 功能:
%   1. 【Task 3 纯色解码: 显著时间簇起始时间点 (Cluster Onset) 与 MNI Y 坐标相关性】
%   2. 【两种纳入标准出图 (按用户明确需求)】:
%      - 图 1: 严格模式 (Strict) —— 仅纳入置换检验全时程校正显著的簇 (has_sig_cluster == 1, N = 12)
%      - 图 2: 宽松模式 (Relaxed) —— 在严格模式基础上，纳入连续 >= 4 个时间点逐点显著 (p < 0.05)
%              但未过全时程校正的接近显著簇 (例如 sub006-G3, 共 N = 42)
%      - 图 3: 双子图对照大图 (Strict vs Relaxed 左右并排)
%      - 图 4 & 5: 三维空间全景图 (MNI X / Y / Z 各一子图)
%   3. 【丰富视觉编码】:
%      - 散点横轴: MNI Y (mm, 后侧 < 0 < 前侧)
%      - 散点纵轴: 时间簇起始时间点 (Cluster Onset Time, ms)
%      - 垂直淡色柱: 展现时间簇从 Onset 到 Offset 的持续时间范围
%      - 频段颜色: 最优单频段 (Delta ~ High-Gamma 学术配色)
%      - 点大小: Sigmoid 映射峰值解码正确率
%      - 线性回归虚线与统计检验标注 (Pearson r, p, Spearman rho)
%   4. 【导出详细汇总表】:
%      - result/tables/task3_cluster_onset_y_correlation.csv
% ========================================================================

clear; clc; close all;

%% 1. 参数直观配置 (置顶易调，简写平铺)
cfg = struct();
cfg.t_lim        = [0, 800];                     % 纵轴时间范围 (ms)
cfg.t_step       = 100;                          % 纵轴时间刻度步长 (ms)
cfg.y_lim        = [-95, 45];                    % 横轴 MNI Y 范围 (mm, 右侧预留充裕空间避免图例压点)
cfg.min_pts_near = 4;                            % 接近显著连续时间点数门槛 (4 点 = 80ms)

% 视觉效果参数
cfg.dur_alpha    = 0.35;                         % 簇持续时间柱透明度
cfg.dur_width    = 2.8;                          % 簇持续时间柱宽度
cfg.corr_col     = [0.20, 0.20, 0.20];           % 回归拟合线颜色
cfg.corr_width   = 2.0;                          % 回归拟合线线宽

% Sigmoid 散点大小映射: Size = S_min + (S_max - S_min) / (1 + exp(-k * (acc - mid)))
cfg.acc_mid      = 0.58;                         % Sigmoid 中点正确率 (58%)
cfg.acc_k        = 35;                           % Sigmoid 增长斜率
cfg.size_min     = 70;                           % 最小点大小
cfg.size_max     = 360;                          % 最大点大小

% 6 大生理频段配色方案 (Nature 规范学术配色)
cfg.bands        = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low-Gamma', 'High-Gamma'};
cfg.band_keys    = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.band_cols    = [
    0.45, 0.45, 0.45;  % Delta: 石板灰
    0.95, 0.60, 0.15;  % Theta: 琥珀橙
    0.15, 0.45, 0.80;  % Alpha: 经典蓝
    0.10, 0.65, 0.45;  % Beta: 翡翠绿
    0.50, 0.30, 0.75;  % Low-Gamma: 罗兰紫
    0.90, 0.15, 0.45   % High-Gamma: 玫瑰红
];

% 路径配置
script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(script_dir));
res_root   = fullfile(proj_root, 'result');
tab_dir    = fullfile(res_root, 'tables');
fig_dir    = fullfile(res_root, 'figures', 'spatiotemporal');
loc_dir    = fullfile(proj_root, 'metadata', 'ieeg_location');

c04_file   = fullfile(tab_dir, 'color_effects_summary.mat');
tc_file    = fullfile(tab_dir, 'single_channel_decoding_timecourses.mat');
sum_file   = fullfile(tab_dir, 'task3_purecolor_decoding_summary.mat');

if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% 2. 加载数据并提取每个电极的簇时程与 MNI 坐标
fprintf('========================================================================\n');
fprintf('  【Task 3: 显著簇起始时间点与 MNI Y 坐标相关性分析】  \n');
fprintf('========================================================================\n');

load(sum_file, 'summary_table');
load(tc_file, 'task3_purecolor', 'time_ms');

c04_data = load(c04_file);
if isfield(c04_data, 'all_tbl'), c04_tbl = c04_data.all_tbl; else, c04_tbl = c04_data.res_table; end

n_ch = height(summary_table);
records = struct([]);

for i = 1:n_ch
    sub = char(summary_table.subject{i});
    ch  = char(summary_table.channel{i});
    key = sprintf('%s_%s', sub, ch);
    
    % 1. 获取三维 MNI 坐标
    mx = NaN; my = NaN; mz = NaN;
    m_c04 = strcmp(string(c04_tbl.subject), string(sub)) & strcmp(string(c04_tbl.channel), string(ch));
    idx_c04 = find(m_c04, 1);
    if ~isempty(idx_c04) && ~isnan(c04_tbl.mni_y(idx_c04))
        mx = c04_tbl.mni_x(idx_c04);
        my = c04_tbl.mni_y(idx_c04);
        mz = c04_tbl.mni_z(idx_c04);
    end
    
    % 若 C04 表缺失，尝试从 ieegloc.xlsx 或 .tsv 读取 (如 sub008)
    if isnan(my)
        f_xlsx = fullfile(loc_dir, sprintf('%s_ieegloc.xlsx', sub));
        f_tsv  = fullfile(loc_dir, sprintf('%s.tsv', sub));
        if isfile(f_xlsx)
            t_loc = readtable(f_xlsx, 'VariableNamingRule', 'preserve');
            c_m = strcmp(string(table2cell(t_loc(:, 1))), string(ch));
            c_idx = find(c_m, 1);
            if ~isempty(c_idx)
                mni_col = find(strcmpi(t_loc.Properties.VariableNames, 'MNI'), 1);
                if ~isempty(mni_col)
                    mni_raw = string(t_loc{c_idx, mni_col});
                    nums = sscanf(char(strrep(strrep(mni_raw, '[', ''), ']', '')), '%f,%f,%f');
                    if numel(nums) == 3, mx = nums(1); my = nums(2); mz = nums(3); end
                elseif width(t_loc) >= 4
                    vals = table2array(t_loc(c_idx, 2:4));
                    if isnumeric(vals) && ~any(isnan(vals)), mx = vals(1); my = vals(2); mz = vals(3); end
                end
            end
        elseif isfile(f_tsv)
            t_tsv = readtable(f_tsv, 'FileType', 'text', 'Delimiter', '\t');
            c_m = strcmp(string(t_tsv.Channel), string(ch));
            c_idx = find(c_m, 1);
            if ~isempty(c_idx) && ismember('MNI', t_tsv.Properties.VariableNames)
                mni_raw = string(t_tsv.MNI(c_idx));
                nums = sscanf(char(strrep(strrep(mni_raw, '[', ''), ']', '')), '%f,%f,%f');
                if numel(nums) == 3, mx = nums(1); my = nums(2); mz = nums(3); end
            end
        end
    end
    
    % 2. 匹配解码时程
    tc_idx = find(strcmp({task3_purecolor.key}, key), 1);
    if isempty(tc_idx), continue; end
    entry = task3_purecolor(tc_idx);
    
    % 寻找刺激后 (time >= 0) 逐点显著 (p < 0.05) 的时间片段
    p_pt = entry.p_pointwise;
    sig_mask = (p_pt < 0.05) & (time_ms >= 0);
    
    in_c = false; cls = []; c_s = 1;
    for w = 1:numel(time_ms)
        if sig_mask(w) && ~in_c
            in_c = true; c_s = w;
        elseif ~sig_mask(w) && in_c
            in_c = false; cls = [cls; c_s, w-1]; %#ok<AGROW>
        end
    end
    if in_c, cls = [cls; c_s, numel(time_ms)]; end
    
    n_cl = size(cls, 1);
    cl_lens = []; cl_starts = []; cl_ends = []; cl_masses = [];
    for c_i = 1:n_cl
        idx_r = cls(c_i, 1):cls(c_i, 2);
        cl_lens(c_i) = numel(idx_r);
        cl_starts(c_i) = time_ms(cls(c_i, 1));
        cl_ends(c_i) = time_ms(cls(c_i, 2));
        cl_masses(c_i) = sum(entry.acc_joint(idx_r) - 0.5);
    end
    
    % 最优频段解析
    raw_band = char(summary_table.best_single_band{i});
    clean_band = strrep(raw_band, '_', '-');
    b_idx = find(strcmp(cfg.bands, clean_band), 1);
    if isempty(b_idx), b_idx = 1; end
    
    % Sigmoid 散点大小
    p_acc = summary_table.peak_acc_joint(i);
    sigm_val = 1 / (1 + exp(-cfg.acc_k * (p_acc - cfg.acc_mid)));
    pt_size  = cfg.size_min + (cfg.size_max - cfg.size_min) * sigm_val;
    
    r = struct();
    r.subject         = string(sub);
    r.channel         = string(ch);
    r.label           = string(sprintf('%s-%s', sub, ch));
    r.mni_x           = mx;
    r.mni_y           = my;
    r.mni_z           = mz;
    r.has_coords      = ~isnan(mx) && ~isnan(my) && ~isnan(mz);
    r.has_sig_cluster = summary_table.has_sig_cluster(i);
    r.peak_acc_joint  = p_acc;
    r.peak_time_ms    = summary_table.peak_time_ms(i);
    r.best_band       = string(cfg.bands{b_idx});
    r.band_color      = cfg.band_cols(b_idx, :);
    r.pt_size         = pt_size;
    
    % 判断归属类别与提取起始时间点
    if r.has_sig_cluster == 1 && ~isempty(cl_starts)
        [~, max_m] = max(cl_masses);
        r.t_onset   = cl_starts(max_m);
        r.t_offset  = cl_ends(max_m);
        r.dur_ms    = cl_ends(max_m) - cl_starts(max_m);
        r.n_points  = cl_lens(max_m);
        r.group     = "Strict (Significant Cluster)";
        r.is_strict = 1;
        r.included  = 1;
    elseif any(cl_lens >= cfg.min_pts_near)
        c_cand = find(cl_lens >= cfg.min_pts_near);
        [~, best_cand] = max(cl_masses(c_cand));
        idx_m = c_cand(best_cand);
        r.t_onset   = cl_starts(idx_m);
        r.t_offset  = cl_ends(idx_m);
        r.dur_ms    = cl_ends(idx_m) - cl_starts(idx_m);
        r.n_points  = cl_lens(idx_m);
        r.group     = "Near-Significant (>=4 points)";
        r.is_strict = 0;
        r.included  = 1;
    else
        r.t_onset   = NaN;
        r.t_offset  = NaN;
        r.dur_ms    = NaN;
        r.n_points  = 0;
        r.group     = "Non-Significant";
        r.is_strict = 0;
        r.included  = 0;
    end
    
    records = [records; r]; %#ok<AGROW>
end

%% 3. 保存详细对照表格
tab_export = struct2table(records);
csv_file   = fullfile(tab_dir, 'task3_cluster_onset_y_correlation.csv');
writetable(tab_export, csv_file);
fprintf('[+] 详细统计明细已导出至: %s\n', csv_file);

% 提取有效子集
strict_recs = records([records.is_strict] == 1 & [records.has_coords] == 1);
comb_recs   = records([records.included] == 1  & [records.has_coords] == 1);

fprintf('  - 严格显著电极数 (有效 MNI): N = %d\n', numel(strict_recs));
fprintf('  - 宽松/合并电极数 (有效 MNI): N = %d (其中严格显著 %d, 接近显著 %d)\n', ...
    numel(comb_recs), sum([comb_recs.is_strict] == 1), sum([comb_recs.is_strict] == 0));

%% 4. 绘图 1: 严格显著模式 (Strict: has_sig_cluster == 1)
fig1 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 680, 580]);
ax1 = axes('Position', [0.14, 0.16, 0.80, 0.76]);
hold(ax1, 'on'); grid(ax1, 'off');
set(ax1, 'Box', 'off', 'FontSize', 12, 'LineWidth', 1.2, ...
    'XColor', [0.15, 0.15, 0.15], 'YColor', [0.15, 0.15, 0.15]);

plot_scatter_and_fit(ax1, strict_recs, cfg, 'Strict Significant Clusters');

png1 = fullfile(fig_dir, 'task3_cluster_onset_vs_mni_y_strict.png');
if isfile(png1), try, delete(png1); catch, end; end
try, exportgraphics(fig1, png1, 'Resolution', 300); catch, saveas(fig1, png1); end
close(fig1);
fprintf('[+] 图 1 (严格显著模式) 已保存: %s\n', png1);

%% 5. 绘图 2: 宽松模式 (Relaxed: 纳入连续 >= 4 点接近显著电极)
fig2 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 680, 580]);
ax2 = axes('Position', [0.14, 0.16, 0.80, 0.76]);
hold(ax2, 'on'); grid(ax2, 'off');
set(ax2, 'Box', 'off', 'FontSize', 12, 'LineWidth', 1.2, ...
    'XColor', [0.15, 0.15, 0.15], 'YColor', [0.15, 0.15, 0.15]);

plot_scatter_and_fit(ax2, comb_recs, cfg, 'Strict + Near-Significant (>= 4 Points)');

png2 = fullfile(fig_dir, 'task3_cluster_onset_vs_mni_y_relaxed.png');
if isfile(png2), try, delete(png2); catch, end; end
try, exportgraphics(fig2, png2, 'Resolution', 300); catch, saveas(fig2, png2); end
close(fig2);
fprintf('[+] 图 2 (宽松模式/含接近显著) 已保存: %s\n', png2);

%% 6. 绘图 3: 左右双子图横向对比大图 (Strict vs Relaxed)
fig3 = figure('Visible', 'off', 'Color', 'w', 'Position', [80, 80, 1300, 580]);

% 左子图: 严格显著
ax_l = subplot(1, 2, 1);
hold(ax_l, 'on'); grid(ax_l, 'off');
set(ax_l, 'Box', 'off', 'FontSize', 12, 'LineWidth', 1.2, ...
    'XColor', [0.15, 0.15, 0.15], 'YColor', [0.15, 0.15, 0.15]);
plot_scatter_and_fit(ax_l, strict_recs, cfg, 'Strict Significant (N = 12)');

% 右子图: 宽松合并
ax_r = subplot(1, 2, 2);
hold(ax_r, 'on'); grid(ax_r, 'off');
set(ax_r, 'Box', 'off', 'FontSize', 12, 'LineWidth', 1.2, ...
    'XColor', [0.15, 0.15, 0.15], 'YColor', [0.15, 0.15, 0.15]);
plot_scatter_and_fit(ax_r, comb_recs, cfg, 'Strict + Near-Significant (N = 42)');

png3 = fullfile(fig_dir, 'task3_cluster_onset_vs_mni_y_comparison.png');
if isfile(png3), try, delete(png3); catch, end; end
try, exportgraphics(fig3, png3, 'Resolution', 300); catch, saveas(fig3, png3); end
close(fig3);
fprintf('[+] 图 3 (双子图对比大图) 已保存: %s\n', png3);

%% 7. 绘图 4 & 5: 三维空间全景图 (MNI X / Y / Z 各一列)
render_3view_plot(strict_recs, cfg, 'Strict Significant Clusters (Task 3)', ...
    fullfile(fig_dir, 'task3_cluster_onset_XYZ_strict.png'));
render_3view_plot(comb_recs, cfg, 'Strict + Near-Significant (>= 4 Points) (Task 3)', ...
    fullfile(fig_dir, 'task3_cluster_onset_XYZ_relaxed.png'));

fprintf('\n========================================================================\n');
fprintf('  【全部相关性图表生成完毕！】\n');
fprintf('========================================================================\n');


%% ========================================================================
% 辅助函数 1: 单子图散点、持续时间柱与拟合线绘制
% ========================================================================
function plot_scatter_and_fit(ax, recs, cfg, subtitle_text)
    xlim(ax, cfg.y_lim);
    ylim(ax, cfg.t_lim);
    set(ax, 'YTick', cfg.t_lim(1):cfg.t_step:cfg.t_lim(2));
    
    xlabel(ax, sprintf('MNI Y (mm)\n[Posterior < 0 < Anterior]'), 'FontSize', 12.5, 'FontWeight', 'bold');
    ylabel(ax, 'Cluster Onset Time (ms)', 'FontSize', 12.5, 'FontWeight', 'bold');
    title(ax, subtitle_text, 'FontSize', 13.5, 'FontWeight', 'bold');
    
    if isempty(recs), return; end
    
    % 1. 绘制时间簇持续范围柱 (Onset 到 Offset)
    for k = 1:numel(recs)
        r = recs(k);
        if r.is_strict == 1
            line_style = '-';
            l_width    = cfg.dur_width;
            l_alpha    = cfg.dur_alpha;
        else
            line_style = ':'; % 接近显著用虚线柱以直观区分
            l_width    = cfg.dur_width * 0.9;
            l_alpha    = cfg.dur_alpha * 0.85;
        end
        line(ax, [r.mni_y, r.mni_y], [r.t_onset, r.t_offset], ...
            'Color', [r.band_color, l_alpha], 'LineWidth', l_width, ...
            'LineStyle', line_style, 'HandleVisibility', 'off');
    end
    
    % 2. 线性回归拟合虚线与相关系数标注
    x_val = [recs.mni_y]';
    y_val = [recs.t_onset]';
    
    if numel(x_val) >= 3 && std(x_val) > 1e-4
        [r_pear, p_pear] = corr(x_val, y_val);
        [r_spear, p_spear] = corr(x_val, y_val, 'type', 'Spearman');
        
        p_fit  = polyfit(x_val, y_val, 1);
        x_grid = linspace(min(x_val), max(x_val), 200);
        y_grid = polyval(p_fit, x_grid);
        valid_fit = (y_grid >= cfg.t_lim(1)) & (y_grid <= cfg.t_lim(2));
        
        plot(ax, x_grid(valid_fit), y_grid(valid_fit), '--', ...
            'Color', cfg.corr_col, 'LineWidth', cfg.corr_width, 'HandleVisibility', 'off');
        
        if p_pear < 0.001
            p_str = '***';
        elseif p_pear < 0.01
            p_str = '**';
        elseif p_pear < 0.05
            p_str = '*';
        elseif p_pear < 0.10
            p_str = ' (trend)';
        else
            p_str = ' (n.s.)';
        end
        
        stat_text = sprintf('r = %.2f, p = %.3f%s\n\\rho = %.2f, p = %.3f\nN = %d', ...
            r_pear, p_pear, p_str, r_spear, p_spear, numel(x_val));
        
        text(ax, cfg.y_lim(1) + 4, cfg.t_lim(2) - 40, stat_text, ...
            'FontSize', 10.5, 'FontWeight', 'bold', 'VerticalAlignment', 'top', ...
            'BackgroundColor', [1, 1, 1, 0.88], 'EdgeColor', [0.75, 0.75, 0.75], 'LineWidth', 0.8);
    end
    
    % 3. 绘制散点 (严格显著为实心圆，接近显著为带标记圆)
    for k = 1:numel(recs)
        r = recs(k);
        if r.is_strict == 1
            scatter(ax, r.mni_y, r.t_onset, r.pt_size, ...
                'MarkerFaceColor', r.band_color, ...
                'MarkerEdgeColor', [0.15, 0.15, 0.15], ...
                'MarkerFaceAlpha', 0.95, ...
                'LineWidth', 1.3, ...
                'HandleVisibility', 'off');
        else
            % 接近显著: 菱形标记，与严格显著清晰区分
            scatter(ax, r.mni_y, r.t_onset, r.pt_size * 0.9, ...
                'Marker', 'd', ...
                'MarkerFaceColor', r.band_color, ...
                'MarkerEdgeColor', [0.25, 0.25, 0.25], ...
                'MarkerFaceAlpha', 0.75, ...
                'LineWidth', 1.1, ...
                'HandleVisibility', 'off');
        end
    end
    
    % 4. 频段与标记图例
    h_leg = [];
    leg_names = {};
    for b = 1:numel(cfg.bands)
        h_leg(b) = plot(ax, NaN, NaN, 'o', 'LineStyle', 'none', ...
            'MarkerFaceColor', cfg.band_cols(b, :), ...
            'MarkerEdgeColor', [0.2, 0.2, 0.2], 'MarkerSize', 7, 'LineWidth', 0.8); %#ok<AGROW>
        leg_names{b} = cfg.bands{b}; %#ok<AGROW>
    end
    if any([recs.is_strict] == 0)
        h_leg(end+1) = plot(ax, NaN, NaN, 'o', 'LineStyle', 'none', ...
            'MarkerFaceColor', [0.55, 0.55, 0.55], ...
            'MarkerEdgeColor', [0.2, 0.2, 0.2], 'MarkerSize', 7, 'LineWidth', 0.8); %#ok<AGROW>
        leg_names{end+1} = 'Strict (p < 0.05)'; %#ok<AGROW>
        h_leg(end+1) = plot(ax, NaN, NaN, 'd', 'LineStyle', 'none', ...
            'MarkerFaceColor', [0.55, 0.55, 0.55], ...
            'MarkerEdgeColor', [0.2, 0.2, 0.2], 'MarkerSize', 7, 'LineWidth', 0.8); %#ok<AGROW>
        leg_names{end+1} = 'Near-Sig (>= 4 pts)'; %#ok<AGROW>
    end
    legend(ax, h_leg, leg_names, 'Location', 'northeast', 'Box', 'off', 'FontSize', 8.5, 'Interpreter', 'none');
end


%% ========================================================================
% 辅助函数 2: 三维空间全景图 (X / Y / Z 各一子图)
% ========================================================================
function render_3view_plot(plot_recs, cfg, super_title, out_png)
    if isempty(plot_recs), return; end
    
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80, 80, 1500, 600]);
    
    coord_fields  = {'mni_x', 'mni_y', 'mni_z'};
    coord_xlabels = {
        sprintf('MNI X (mm)\n[Left < 0 < Right]'), ...
        sprintf('MNI Y (mm)\n[Posterior < 0 < Anterior]'), ...
        sprintf('MNI Z (mm)\n[Inferior < 0 < Superior]')
    };
    
    min_x = min([plot_recs.mni_x]); max_x = max([plot_recs.mni_x]);
    min_y = min([plot_recs.mni_y]); max_y = max([plot_recs.mni_y]);
    min_z = min([plot_recs.mni_z]); max_z = max([plot_recs.mni_z]);
    
    coord_xlimits = {
        [min(-45, min_x - 10), max(65, max_x + 10)], ...
        [min(-95, min_y - 10), max(25, max_y + 10)], ...
        [min(-35, min_z - 10), max(65, max_z + 10)]
    };
    
    sub_pos = {
        [0.06, 0.22, 0.28, 0.68], ...
        [0.38, 0.22, 0.28, 0.68], ...
        [0.70, 0.22, 0.28, 0.68]
    };
    
    for p_i = 1:3
        f_name = coord_fields{p_i};
        ax = axes('Position', sub_pos{p_i});
        hold(ax, 'on'); grid(ax, 'off');
        set(ax, 'Box', 'off', 'FontSize', 12, 'LineWidth', 1.2, ...
            'XColor', [0.15, 0.15, 0.15], 'YColor', [0.15, 0.15, 0.15]);
        
        xlim(ax, coord_xlimits{p_i});
        ylim(ax, cfg.t_lim);
        set(ax, 'YTick', cfg.t_lim(1):cfg.t_step:cfg.t_lim(2));
        
        xlabel(ax, coord_xlabels{p_i}, 'FontSize', 12.5, 'FontWeight', 'bold');
        if p_i == 1
            ylabel(ax, 'Cluster Onset Time (ms)', 'FontSize', 13, 'FontWeight', 'bold');
        else
            set(ax, 'YTickLabel', []);
        end
        
        % 1. 持续时间柱
        for k = 1:numel(plot_recs)
            r = plot_recs(k);
            c_val = r.(f_name);
            if r.is_strict == 1, ls = '-'; lw = cfg.dur_width;
            else, ls = ':'; lw = cfg.dur_width * 0.9; end
            line(ax, [c_val, c_val], [r.t_onset, r.t_offset], ...
                'Color', [r.band_color, cfg.dur_alpha], 'LineWidth', lw, ...
                'LineStyle', ls, 'HandleVisibility', 'off');
        end
        
        % 2. 拟合与检验
        x_all = [plot_recs.(f_name)]';
        y_all = [plot_recs.t_onset]';
        if numel(x_all) >= 3 && std(x_all) > 1e-4
            [r_val, p_val] = corr(x_all, y_all);
            p_fit  = polyfit(x_all, y_all, 1);
            x_grid = linspace(min(x_all), max(x_all), 200);
            y_grid = polyval(p_fit, x_grid);
            valid_fit = (y_grid >= cfg.t_lim(1)) & (y_grid <= cfg.t_lim(2));
            plot(ax, x_grid(valid_fit), y_grid(valid_fit), '--', ...
                'Color', cfg.corr_col, 'LineWidth', cfg.corr_width, 'HandleVisibility', 'off');
            
            if p_val < 0.001, sig_s = '***'; elseif p_val < 0.01, sig_s = '**';
            elseif p_val < 0.05, sig_s = '*'; elseif p_val < 0.10, sig_s = ' (trend)';
            else, sig_s = ' (n.s.)'; end
            
            stat_t = sprintf('r = %.2f, p = %.3f%s', r_val, p_val, sig_s);
            box_x  = coord_xlimits{p_i}(1) + (coord_xlimits{p_i}(2) - coord_xlimits{p_i}(1)) * 0.05;
            box_y  = cfg.t_lim(2) - 40;
            text(ax, box_x, box_y, stat_t, 'FontSize', 11, 'FontWeight', 'bold', ...
                'BackgroundColor', [1, 1, 1, 0.85], 'EdgeColor', [0.8, 0.8, 0.8], 'LineWidth', 0.8);
        end
        
        % 3. 散点
        for k = 1:numel(plot_recs)
            r = plot_recs(k);
            c_val = r.(f_name);
            if r.is_strict == 1, mk = 'o'; m_alpha = 0.95; sz = r.pt_size;
            else, mk = 'd'; m_alpha = 0.75; sz = r.pt_size * 0.9; end
            scatter(ax, c_val, r.t_onset, sz, ...
                'Marker', mk, 'MarkerFaceColor', r.band_color, ...
                'MarkerEdgeColor', [0.15, 0.15, 0.15], 'MarkerFaceAlpha', m_alpha, ...
                'LineWidth', 1.2, 'HandleVisibility', 'off');
        end
    end
    
    sgtitle(super_title, 'FontSize', 15, 'FontWeight', 'bold');
    if isfile(out_png), try, delete(out_png); catch, end; end
    try, exportgraphics(fig, out_png, 'Resolution', 300); catch, saveas(fig, out_png); end
    close(fig);
    fprintf('[+] 三维空间全景图已保存: %s\n', out_png);
end
