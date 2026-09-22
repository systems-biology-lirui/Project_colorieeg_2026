function plot_task1_band_overlap_upset(user_cfg)
%% ========================================================================
% 脚本名称: plot_task1_band_overlap_upset.m
% 功能说明:
%   复刻顶刊经典的 UpSet 集合交集重叠图 (UpSet Plot):
%   1. 上半部分柱状图 (Upper Bar Plot):
%      - 纵轴: Intersection size (特定频段组合对应的电极数量)
%      - 最左侧独立黑色柱: 全频段均无显著响应的基底电极总数 (Degree = 0)
%      - 柱子色彩渐变: 颜色映射该组合包含的频段数量 (Degree: 1, 2, 3...)
%   2. 下半部分点阵图 (Lower Dot Matrix Plot):
%      - 纵轴: Frequency Band (从下到上: Delta, Theta, Alpha, Beta, Gamma, High-Gamma)
%      - 每行配有柔和浅灰水平条带
%      - 浅灰小圆点: 未激活频段
%      - 彩色实心大圆点: 激活频段 (与上方柱子同色)
%      - 垂直连线: 当 >=2 个频段共激活时，垂直贯穿连接各激活圆点，展现频段耦合
%   3. 布局与排列逻辑:
%      - 先按频段组合阶数 (Degree) 分组，组内按电极数量降序排列
%      - 上下两个子图 X 轴范围严格对齐联动
%   4. 支持模式:
%      - 'all_significant': Task 1 色彩总体显著电极 (is_significant == 1)
%      - 'concordant'     : Task 1 四类别严格同向显著电极 (Concordant Positive/Negative)
%      - 'both'           : 两种模式均生成并导出 300 DPI 印刷级高清图 (默认)
% ========================================================================

if nargin < 1
    user_cfg = struct();
end

%% 1. 参数与路径配置 (平铺直观，可在此直接调整)
cfg = struct();
cfg.mode                = 'both';       % 可选: 'both', 'all_significant', 'concordant'
cfg.include_degree_zero = false;        % 是否绘制左侧全频段无显著响应的基底电极 (false: 仅绘制显著位点; true: 包含无响应基底)
cfg.sort_within_degree  = 'canonical';  % 组内排序: 'canonical' (典范频段阶梯, 严格复刻顶刊), 或 'count' (按数量降序)
cfg.show_bar_values     = false;        % 是否在柱顶显示精确数字 (false 为顶刊默认纯粹学术风; true 则标注)
cfg.bar_width           = 0.65;         % 柱状图柱子宽度
cfg.line_width          = 2.2;          % 点阵图中垂直连接线线宽
cfg.active_dot_sz       = 68;           % 激活圆点大小 (散点面积)
cfg.inactive_dot_sz     = 28;           % 未激活浅灰圆点大小 (散点面积)

% 频段定义 (从低频到高频排列，与下半部点阵自底向上顺序一致)
cfg.bands        = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.band_labels  = {'\delta', '\theta', '\alpha', '\beta', '\gamma', 'High \gamma'};
cfg.n_bands      = numel(cfg.bands);

% 阶数 (Degree 0 ~ 6) 配色方案 (完全复刻参考图的自然渐变配色)
% Degree 0: 黑色基底柱
% Degree 1: 深墨绿/深海蓝绿
% Degree 2: 翡翠绿/青绿
% Degree 3: 浅青绿/薄荷绿
% Degree 4: 橄榄绿
% Degree 5: 琥珀金绿
% Degree 6: 暖金橙
cfg.degree_colors = [
    0.10, 0.10, 0.10;  % Degree 0: Black
    0.02, 0.32, 0.35;  % Degree 1: Dark Teal
    0.08, 0.52, 0.50;  % Degree 2: Medium Teal
    0.18, 0.68, 0.52;  % Degree 3: Light Teal-Green
    0.45, 0.70, 0.35;  % Degree 4: Olive Green
    0.72, 0.68, 0.22;  % Degree 5: Olive Gold
    0.88, 0.56, 0.15   % Degree 6: Amber Orange
];

% 浅灰背景条与点阵配色
cfg.row_bg_color     = [0.93, 0.93, 0.94];
cfg.inactive_dot_col = [0.80, 0.80, 0.82];

% 用户参数覆盖
fields = fieldnames(user_cfg);
for i = 1:numel(fields)
    cfg.(fields{i}) = user_cfg.(fields{i});
end

% 路径配置
script_dir  = fileparts(mfilename('fullpath'));
proj_root   = fileparts(fileparts(script_dir)); % color_analyse_0825
tab_file    = fullfile(proj_root, 'result', 'tables', 'color_effects_summary.mat');
out_fig_dir = fullfile(proj_root, 'result', 'figures', 'upset_band_overlap');

if ~exist(out_fig_dir, 'dir'), mkdir(out_fig_dir); end

if ~isfile(tab_file)
    error('未找到 Task 1 色彩效应主表: %s', tab_file);
end

%% 2. 加载数据
loaded = load(tab_file);
if isfield(loaded, 'all_tbl')
    tbl = loaded.all_tbl;
else
    tbl = loaded.res_table;
end

% 确定需要生成的模式列表
if strcmp(cfg.mode, 'both')
    modes_to_run = {'all_significant', 'concordant'};
else
    modes_to_run = {cfg.mode};
end

%% 3. 逐模式绘制 UpSet 图
for m_idx = 1:numel(modes_to_run)
    curr_mode = modes_to_run{m_idx};
    render_single_upset(tbl, curr_mode, cfg, out_fig_dir);
end

end

%% ========================================================================
% 单模式 UpSet 核心渲染子函数
% ========================================================================
function render_single_upset(tbl, mode_name, cfg, out_fig_dir)

fprintf('\n========================================================================\n');
fprintf('>>> 正在绘制 UpSet 频段耦合交集图 [模式: %s] ...\n', mode_name);
fprintf('========================================================================\n');

% 1. 提取唯一电极列表 (被试 + 通道)
ch_keys = unique(strcat(tbl.subject, '_', tbl.channel), 'stable');
n_chs   = numel(ch_keys);

% 构建 0/1 响应矩阵 [n_chs x n_bands]
sig_mat = false(n_chs, cfg.n_bands);

for i = 1:n_chs
    key = ch_keys{i};
    parts = strsplit(key, '_');
    sub = parts{1}; ch = parts{2};
    sub_tbl = tbl(strcmp(tbl.subject, sub) & strcmp(tbl.channel, ch), :);
    
    for b = 1:cfg.n_bands
        b_name = cfg.bands{b};
        r = sub_tbl(strcmp(sub_tbl.freq_band, b_name), :);
        if ~isempty(r) && (r.is_significant(1) == 1)
            if strcmp(mode_name, 'concordant')
                % 严格同向限定 (Concordant Positive / Negative)
                if ismember(r.concordance_type{1}, {'Concordant_Positive', 'Concordant_Negative'})
                    sig_mat(i, b) = true;
                end
            else
                % 总体显著 (包含偏好位点)
                sig_mat(i, b) = true;
            end
        end
    end
end

% 2. 统计唯一存在的组合及其频次
[u_comb, ~, ic] = unique(sig_mat, 'rows');
counts = accumarray(ic, 1);
degrees = sum(u_comb, 2);

% 如果不包含无显著响应的基底位点 (Degree 0)，则剔除
if ~cfg.include_degree_zero
    deg0_idx = (degrees == 0);
    deg0_count = sum(counts(deg0_idx));
    fprintf('[i] 已排除未响应基底电极 (Degree 0): %d (%.2f%%)\n', deg0_count, deg0_count/n_chs*100);
    valid_mask = (degrees > 0);
    u_comb  = u_comb(valid_mask, :);
    counts  = counts(valid_mask);
    degrees = degrees(valid_mask);
end

% 排序规则 (严格复刻顶刊论文的 Canonical Staircase 阶梯规范):
% 1. 先按频段组合阶数 (degree: 1, 2, 3...) 升序
% 2. 阶数内排序:
%    - 'canonical' (默认): 频段自然顺序 (Delta -> Theta -> Alpha -> Beta -> Gamma -> High-Gamma) 典范字典序
%      使得 Degree 1 呈现自左下向右上攀升的完美阶梯，高阶呈现结构化频段扫描
%    - 'count': 按该组合的电极数量降序
if strcmp(cfg.sort_within_degree, 'canonical')
    sort_mat = [degrees, -double(u_comb)];
else
    sort_mat = [degrees, -counts];
end
[~, sort_idx] = sortrows(sort_mat);
u_comb  = u_comb(sort_idx, :);
counts  = counts(sort_idx);
degrees = degrees(sort_idx);

n_combs = numel(counts);

fprintf('[+] 总电极数: %d | 绘制的有效频段组合数: %d 种\n', n_chs, n_combs);
fprintf('[+] 显著响应电极总数 (Degree >= 1): %d (%.2f%%)\n', sum(counts), sum(counts)/n_chs*100);

% 3. 构造画布与子图布局
% 上半部分: 柱状图 [left, bottom, width, height]
% 下半部分: 点阵图 [left, bottom, width, height]
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80, 80, 1100, 520]);

ax_bar = axes('Position', [0.09, 0.44, 0.86, 0.44]);
ax_mat = axes('Position', [0.09, 0.12, 0.86, 0.28]);

%% 4. 绘制上半部分柱状图 (Upper Bar Plot)
axes(ax_bar); %#ok<LAXES>
hold(ax_bar, 'on');

% 循环逐列绘制柱子并设置专属颜色
for c = 1:n_combs
    deg = degrees(c);
    c_col = cfg.degree_colors(deg + 1, :);
    
    % 绘制单个 Bar
    bar(ax_bar, c, counts(c), cfg.bar_width, ...
        'FaceColor', c_col, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % 标注柱顶数字 (极简学术风格，小字号清晰呈现)
    if cfg.show_bar_values
        if counts(c) > 0
            text(ax_bar, c, counts(c) + max(counts)*0.02, sprintf('%d', counts(c)), ...
                'HorizontalAlignment', 'center', 'FontSize', 8.5, 'Color', [0.2, 0.2, 0.2]);
        end
    end
end

% 轴外观与刻度配置
y_max = max(counts);
if y_max > 200
    y_step = 100;
elseif y_max > 50
    y_step = 20;
elseif y_max > 20
    y_step = 10;
elseif y_max > 10
    y_step = 5;
else
    y_step = 2;
end
y_top = ceil(y_max / y_step) * y_step;
y_ticks = 0:y_step:y_top;

xlim(ax_bar, [0.4, n_combs + 0.6]);
if cfg.show_bar_values
    ylim(ax_bar, [0, y_top * 1.08]);
else
    ylim(ax_bar, [0, y_top]);
end
set(ax_bar, 'Box', 'off', 'Color', 'none', 'FontSize', 10, 'LineWidth', 1.0, ...
    'XTick', [], 'XColor', 'none', 'YTick', y_ticks);
ylabel(ax_bar, 'Intersection size', 'FontSize', 11, 'FontWeight', 'bold');
grid(ax_bar, 'on');
set(ax_bar, 'YGrid', 'on', 'XGrid', 'off', 'GridColor', [0.88, 0.88, 0.88], 'GridAlpha', 0.8);

% 标题与副标题设置 (使用原生 title / subtitle，排版端正清爽)
if strcmp(mode_name, 'concordant')
    main_title = 'Color Selective (Concordant) Channels Shared in Common Across Bands';
    sub_title  = 'Color selective electrodes (category-concordant) common across bands';
else
    main_title = 'Color Selective Channels Shared in Common Across Bands';
    sub_title  = 'Color selective electrodes common across bands';
end

title(ax_bar, main_title, 'FontSize', 13.5, 'FontWeight', 'bold');
subtitle(ax_bar, sub_title, 'FontSize', 10, 'FontWeight', 'normal', 'Color', [0.35, 0.35, 0.35]);

%% 5. 绘制下半部分点阵图 (Lower Dot Matrix Plot)
axes(ax_mat); %#ok<LAXES>
hold(ax_mat, 'on');

% 1) 绘制每行的浅灰背景带
for b = 1:cfg.n_bands
    rectangle(ax_mat, 'Position', [0.4, b - 0.38, n_combs + 0.2, 0.76], ...
        'FaceColor', cfg.row_bg_color, 'EdgeColor', 'none');
end

% 2) 绘制点阵与垂直连线
for c = 1:n_combs
    deg   = degrees(c);
    c_col = cfg.degree_colors(deg + 1, :);
    
    active_bands = find(u_comb(c, :));
    
    % 如果该列包含 2 个及以上频段，先画垂直穿引实线
    if numel(active_bands) >= 2
        min_b = min(active_bands);
        max_b = max(active_bands);
        plot(ax_mat, [c, c], [min_b, max_b], '-', ...
            'Color', c_col, 'LineWidth', cfg.line_width, 'HandleVisibility', 'off');
    end
    
    % 画该列的所有圆点
    for b = 1:cfg.n_bands
        if ismember(b, active_bands)
            % 激活频段点 (彩色实心大圆)
            scatter(ax_mat, c, b, cfg.active_dot_sz, ...
                'MarkerFaceColor', c_col, 'MarkerEdgeColor', c_col, 'HandleVisibility', 'off');
        else
            % 未激活频段点 (浅灰小圆)
            scatter(ax_mat, c, b, cfg.inactive_dot_sz, ...
                'MarkerFaceColor', cfg.inactive_dot_col, 'MarkerEdgeColor', 'none', 'HandleVisibility', 'off');
        end
    end
end

% 轴外观配置
xlim(ax_mat, [0.4, n_combs + 0.6]);
ylim(ax_mat, [0.5, cfg.n_bands + 0.5]);
set(ax_mat, 'Box', 'off', 'Color', 'none', 'FontSize', 10.5, 'LineWidth', 1.0, ...
    'YTick', 1:cfg.n_bands, 'YTickLabel', cfg.band_labels, ...
    'XTick', [], 'XColor', [0.4, 0.4, 0.4]);

ylabel(ax_mat, 'Frequency Band', 'FontSize', 11, 'FontWeight', 'bold');
xlabel(ax_mat, 'Band Combination', 'FontSize', 11, 'FontWeight', 'bold');

%% 6. 保存高清科研图像 (300 DPI 印刷级)
if strcmp(mode_name, 'concordant')
    out_file_base = 'color_selective_concordant_channels_band_overlap_upset';
else
    out_file_base = 'color_selective_channels_band_overlap_upset';
end

out_png = fullfile(out_fig_dir, [out_file_base, '.png']);
exportgraphics(fig, out_png, 'Resolution', 300);
fprintf('  [✓] 300 DPI 高清科研图已保存至:\n      %s\n', out_png);

close(fig);

end
