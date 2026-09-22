function plot_task1_screening_vs_decoding_ranks(user_cfg)
%% ========================================================================
% 脚本名称: plot_task1_screening_vs_decoding_ranks.m
% 功能说明:
%   评估 Task 1 色彩显著性位点的【筛选频段】与其在 Task 2 (记忆颜色) 和 Task 3 (纯色红绿) 
%   中各单频段 Decoding 表现的相关性:
%   1. 对 Task 1 筛选出的每个显著电极及其筛选频段:
%      - 提取该电极在 Task 2 和 Task 3 中的 6 个独立单频段 Balanced Accuracy (不含 Multi-Band)
%      - 计算该筛选频段在 6 个频段中的排名 Rank (1 = Top 最高, 6 = Lowest 最低)
%   2. 假设检验 (Null Hypothesis vs 3.5):
%      - 若 Task 1 筛选频段与后续 Decoding 不相关，理论平均 Rank 应服从均匀分布期望值 3.5:
%        E[Rank] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5
%      - 若显著相关 (具备频段特异性传承)，平均 Rank 应显著优于 3.5 (Mean < 3.5, p < 0.05)
%   3. 视觉呈现 (2x2 科研出版级多子图):
%      - 上排: Task 2 & Task 3 的散点、均值、95% CI 误差棒与 vs 3.5 统计检验标注
%      - 下排: 各频段在 Task 2 & Task 3 中 Rank 1st ~ 6th 的构成比堆叠柱状图
% ========================================================================

if nargin < 1
    user_cfg = struct();
end

%% 1. 参数与路径配置 (平铺直观，可在此直接调整)
cfg = struct();
cfg.mode             = 'both';           % 可选: 'both', 'all_significant', 'concordant'
cfg.acc_metric       = 'peak';           % 准确率指标: 'peak' (全时程峰值), 或 'post_stim_peak' (50-500ms)
cfg.invert_y         = true;             % 是否反转纵轴使 Rank 1 (Top) 置于上方 (更符合"向上为优"直觉)
cfg.dot_sz           = 32;               % 散点大小
cfg.dot_alpha        = 0.40;             % 散点透明度

% 频段定义与规范学术标签
cfg.bands        = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.bands_disp   = {'\delta', '\theta', '\alpha', '\beta', 'Low \gamma', 'High \gamma'};
cfg.n_bands      = numel(cfg.bands);

% 配色方案 (统一学术规范)
cfg.band_cols = [
    0.40, 0.40, 0.40;  % Delta: 灰
    0.92, 0.58, 0.18;  % Theta: 杏黄
    0.18, 0.45, 0.75;  % Alpha: 钴蓝
    0.08, 0.62, 0.42;  % Beta: 青绿
    0.50, 0.32, 0.72;  % Low_Gamma: 紫灰
    0.88, 0.15, 0.48   % High_Gamma: 玫红
];
cfg.col_overall = [0.20, 0.20, 0.20];  % Overall: 深黑灰

% 用户参数覆盖
fields = fieldnames(user_cfg);
for i = 1:numel(fields)
    cfg.(fields{i}) = user_cfg.(fields{i});
end

% 路径配置
script_dir  = fileparts(mfilename('fullpath'));
proj_root   = fileparts(fileparts(script_dir)); % color_analyse_0825
tab_dir     = fullfile(proj_root, 'result', 'tables');
out_fig_dir = fullfile(proj_root, 'result', 'figures', 'decoding_rank_summary');

if ~exist(out_fig_dir, 'dir'), mkdir(out_fig_dir); end

f_t1    = fullfile(tab_dir, 'color_effects_summary.mat');
f_t2_c  = fullfile(tab_dir, 'concordant_electrodes_decoding_summary.mat');
f_t2_nc = fullfile(tab_dir, 'non_concordant_electrodes_decoding_summary.mat');
f_t3    = fullfile(tab_dir, 'task3_purecolor_decoding_summary.mat');
f_tc    = fullfile(tab_dir, 'single_channel_decoding_timecourses.mat');

%% 2. 数据载入与整合
fprintf('========================================================================\n');
fprintf('  【Task 1 筛选频段 vs Task 2 & Task 3 Decoding Rank 汇总分析】  \n');
fprintf('========================================================================\n');

% 载入 Task 1
d1 = load(f_t1);
if isfield(d1, 'all_tbl'), t1 = d1.all_tbl; else, t1 = d1.res_table; end

% 载入 Task 2 汇总表
d2_c = load(f_t2_c); t2_c = d2_c.summary_table;
d2_nc = load(f_t2_nc); t2_nc = d2_nc.summary_table;
t2_tbl = [t2_c; t2_nc];

% 载入 Task 3 汇总表
d3 = load(f_t3);
t3_tbl = d3.summary_table;

% 若采用时间窗指标，载入完整时程
d_tc = [];
if strcmp(cfg.acc_metric, 'post_stim_peak')
    if isfile(f_tc)
        d_tc = load(f_tc);
    else
        warning('未找到时程文件 %s，自动回退至全时程 peak 模式。', f_tc);
        cfg.acc_metric = 'peak';
    end
end

% 确定待运行模式
if strcmp(cfg.mode, 'both')
    modes_to_run = {'all_significant', 'concordant'};
else
    modes_to_run = {cfg.mode};
end

for m_idx = 1:numel(modes_to_run)
    curr_mode = modes_to_run{m_idx};
    render_rank_summary_figure(t1, t2_tbl, t3_tbl, d_tc, curr_mode, cfg, out_fig_dir);
end

end

%% ========================================================================
% 单模式绘图核心函数
% ========================================================================
function render_rank_summary_figure(t1, t2_tbl, t3_tbl, d_tc, mode_name, cfg, out_fig_dir)

fprintf('\n>>> 正在生成 Rank 汇总图 [模式: %s | 指标: %s] ...\n', mode_name, cfg.acc_metric);

% 1. 筛选目标电极条目
if strcmp(mode_name, 'concordant')
    sig_mask = (t1.is_significant == 1) & ...
        (strcmp(t1.concordance_type, 'Concordant_Positive') | strcmp(t1.concordance_type, 'Concordant_Negative'));
    mode_title_suffix = ' (Concordant Electrodes Only)';
    out_file_suffix   = '_concordant';
else
    sig_mask = (t1.is_significant == 1);
    mode_title_suffix = ' (All Significant Electrodes)';
    out_file_suffix   = '_all_significant';
end

t1_sub = t1(sig_mask, :);
n_items = height(t1_sub);
fprintf('[+] 纳入分析的 Task 1 筛选条目数: %d 条 (跨 %d 个唯一电极)\n', ...
    n_items, numel(unique(strcat(t1_sub.subject, '_', t1_sub.channel))));

% 2. 逐条目提取各频段准确率并计算 Rank
t1_sub.rank_t2 = nan(n_items, 1);
t1_sub.rank_t3 = nan(n_items, 1);

if strcmp(cfg.acc_metric, 'post_stim_peak') && ~isempty(d_tc)
    t2_tc_all = [d_tc.concordant; d_tc.non_concordant];
    t3_tc_all = d_tc.task3_purecolor;
    w_post = (d_tc.time_ms >= 50 & d_tc.time_ms <= 500);
end

for i = 1:n_items
    sub = t1_sub.subject{i};
    ch  = t1_sub.channel{i};
    b_screen = t1_sub.freq_band{i};
    b_idx = find(strcmp(cfg.bands, b_screen));
    
    % Task 2
    if strcmp(cfg.acc_metric, 'post_stim_peak') && ~isempty(d_tc)
        idx2 = find(strcmp({t2_tc_all.subject}, sub) & strcmp({t2_tc_all.channel}, ch), 1);
        if ~isempty(idx2)
            accs2 = nan(1, 6);
            for b = 1:6
                tc = t2_tc_all(idx2).(sprintf('acc_%s', cfg.bands{b}));
                accs2(b) = max(tc(w_post));
            end
            rnk2 = tiedrank(-accs2);
            t1_sub.rank_t2(i) = rnk2(b_idx);
        end
    else
        idx2 = find(strcmp(t2_tbl.subject, sub) & strcmp(t2_tbl.channel, ch), 1);
        if ~isempty(idx2)
            r2 = t2_tbl(idx2, :);
            accs2 = [r2.peak_acc_Delta, r2.peak_acc_Theta, r2.peak_acc_Alpha, ...
                     r2.peak_acc_Beta, r2.peak_acc_Low_Gamma, r2.peak_acc_High_Gamma];
            rnk2 = tiedrank(-accs2);
            t1_sub.rank_t2(i) = rnk2(b_idx);
        end
    end
    
    % Task 3
    if strcmp(cfg.acc_metric, 'post_stim_peak') && ~isempty(d_tc)
        idx3 = find(strcmp({t3_tc_all.subject}, sub) & strcmp({t3_tc_all.channel}, ch), 1);
        if ~isempty(idx3)
            accs3 = nan(1, 6);
            for b = 1:6
                tc = t3_tc_all(idx3).(sprintf('acc_%s', cfg.bands{b}));
                accs3(b) = max(tc(w_post));
            end
            rnk3 = tiedrank(-accs3);
            t1_sub.rank_t3(i) = rnk3(b_idx);
        end
    else
        idx3 = find(strcmp(t3_tbl.subject, sub) & strcmp(t3_tbl.channel, ch), 1);
        if ~isempty(idx3)
            r3 = t3_tbl(idx3, :);
            accs3 = [r3.peak_acc_Delta, r3.peak_acc_Theta, r3.peak_acc_Alpha, ...
                     r3.peak_acc_Beta, r3.peak_acc_Low_Gamma, r3.peak_acc_High_Gamma];
            rnk3 = tiedrank(-accs3);
            t1_sub.rank_t3(i) = rnk3(b_idx);
        end
    end
end

% 打印统计检验结果
print_stats_table(t1_sub, cfg.bands);

% 3. 构造画布与 2x2 TiledLayout 布局
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50, 50, 1400, 880]);
tiled = tiledlayout(fig, 2, 2, 'TileSpacing', 'loose', 'Padding', 'compact');

% Tile 1: Task 2 均值检验与散点图
ax1 = nexttile(tiled, 1);
render_scatter_tile(ax1, t1_sub, 'rank_t2', cfg, ...
    ['Task 2 (Memory Color Decoding)', mode_title_suffix]);

% Tile 2: Task 3 均值检验与散点图
ax2 = nexttile(tiled, 2);
render_scatter_tile(ax2, t1_sub, 'rank_t3', cfg, ...
    ['Task 3 (Pure Color Decoding)', mode_title_suffix]);

% Tile 3: Task 2 排名构成比堆叠图
ax3 = nexttile(tiled, 3);
render_dist_tile(ax3, t1_sub, 'rank_t2', cfg, ...
    'Task 2: Rank Proportion by Band', false);

% Tile 4: Task 3 排名构成比堆叠图 (带共享图例)
ax4 = nexttile(tiled, 4);
lgd_dist = render_dist_tile(ax4, t1_sub, 'rank_t3', cfg, ...
    'Task 3: Rank Proportion by Band', true);
if ~isempty(lgd_dist)
    lgd_dist.Layout.Tile = 'east'; % 置于整个布局最右侧，两侧子图宽度严格对齐
end

% 导出图像
out_png = fullfile(out_fig_dir, sprintf('task1_screening_vs_decoding_ranks%s.png', out_file_suffix));
exportgraphics(fig, out_png, 'Resolution', 300);
fprintf('  [✓] 300 DPI 高清科研图已保存至:\n      %s\n', out_png);
close(fig);

end

%% ========================================================================
% 散点与统计检验子图渲染
% ========================================================================
function render_scatter_tile(ax, tbl, col_name, cfg, title_str)
axes(ax); hold(ax, 'on'); grid(ax, 'on');
set(ax, 'Box', 'off', 'LineWidth', 1.1, 'FontSize', 10.5, ...
    'YGrid', 'on', 'XGrid', 'off', 'GridColor', [0.88, 0.88, 0.88], 'GridAlpha', 0.8);

n_b = cfg.n_bands;

% 构建横轴频段标签 (保持单行无换行符，避免被 MATLAB 拆分为多重 Tick)
all_x_labels = [cfg.bands_disp, {'Overall'}];

% 1. 理论无相关参考线 (y = 3.5)
h_null = yline(ax, 3.5, '--', 'Color', [0.85, 0.22, 0.18], 'LineWidth', 1.8, ...
    'DisplayName', 'Chance = 3.5');

rng(100); % 固定随机数抖动

for b = 1:(n_b + 1)
    if b <= n_b
        sub_mask = strcmp(tbl.freq_band, cfg.bands{b});
        c_col = cfg.band_cols(b, :);
    else
        sub_mask = true(height(tbl), 1);
        c_col = cfg.col_overall;
    end
    
    vals = tbl.(col_name)(sub_mask);
    vals = vals(~isnan(vals));
    n_v  = numel(vals);
    if n_v == 0, continue; end
    
    % Jitter 散点
    jit_x = b + (rand(n_v, 1) - 0.5) * 0.36;
    scatter(ax, jit_x, vals, cfg.dot_sz, 'MarkerFaceColor', c_col, ...
        'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', cfg.dot_alpha, 'HandleVisibility', 'off');
    
    % 统计值
    m_val   = mean(vals);
    sd_val  = std(vals);
    sem_val = sd_val / sqrt(n_v);
    ci_95   = tinv(0.975, n_v - 1) * sem_val;
    [~, p_t] = ttest(vals - 3.5);
    
    % 95% CI 误差棒
    errorbar(ax, b, m_val, ci_95, 'LineStyle', 'none', 'Color', [0.15, 0.15, 0.15], ...
        'LineWidth', 2.2, 'CapSize', 8, 'HandleVisibility', 'off');
    
    % 均值菱形标记
    plot(ax, b, m_val, 'd', 'MarkerFaceColor', c_col, 'MarkerEdgeColor', 'k', ...
        'MarkerSize', 9.5, 'LineWidth', 1.2, 'HandleVisibility', 'off');
    
    % 格式化显著性标记
    if p_t < 0.001
        sig_str = sprintf('p<.001 ***');
        m_str   = sprintf('%.2f ***', m_val);
    elseif p_t < 0.01
        sig_str = sprintf('p=%.3f **', p_t);
        m_str   = sprintf('%.2f **', m_val);
    elseif p_t < 0.05
        sig_str = sprintf('p=%.3f *', p_t);
        m_str   = sprintf('%.2f *', m_val);
    else
        sig_str = sprintf('p=%.3f (ns)', p_t);
        m_str   = sprintf('%.2f', m_val);
    end
    
    % 均值标注 (顶部) 与 p 值标注 (底部)
    if cfg.invert_y
        % 顶部标注均值
        text(ax, b, 0.65, m_str, ...
            'HorizontalAlignment', 'center', 'FontSize', 8.5, 'FontWeight', 'bold', 'Color', [0.15, 0.15, 0.15]);
        % 底部标注 p 值
        text(ax, b, 6.45, sig_str, ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', [0.4, 0.4, 0.4]);
    else
        text(ax, b, 6.35, m_str, ...
            'HorizontalAlignment', 'center', 'FontSize', 8.5, 'FontWeight', 'bold', 'Color', [0.15, 0.15, 0.15]);
        text(ax, b, 0.65, sig_str, ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', [0.4, 0.4, 0.4]);
    end
end

xlim(ax, [0.4, n_b + 1.6]);
ylim(ax, [0.2, 6.8]);

if cfg.invert_y
    set(ax, 'YDir', 'reverse'); % 1 (Top) 置于上方
end

set(ax, 'YTick', 1:6, 'YTickLabel', {'1 (Top)', '2', '3', '4', '5', '6 (Low)'}, ...
    'XTick', 1:(n_b + 1), 'XTickLabel', all_x_labels, 'TickLength', [0.015, 0.025]);
xtickangle(ax, 0);

% 标注样本量 N 与 x 轴标题
if cfg.invert_y
    y_n  = 7.55;
    y_xl = 8.25;
else
    y_n  = -0.55;
    y_xl = -1.25;
end
for b = 1:(n_b + 1)
    if b <= n_b
        n_v = sum(strcmp(tbl.freq_band, cfg.bands{b}) & ~isnan(tbl.(col_name)));
    else
        n_v = sum(~isnan(tbl.(col_name)));
    end
    text(ax, b, y_n, sprintf('(N=%d)', n_v), 'HorizontalAlignment', 'center', ...
        'FontSize', 8.5, 'Color', [0.45, 0.45, 0.45], 'Clipping', 'off');
end

xl = xlabel(ax, 'Task 1 Screening Frequency Band', 'FontWeight', 'bold');
xl.Units = 'data';
xl.Position = [(n_b + 2)/2, y_xl, 0];
ylabel(ax, 'Decoding Rank of Screening Band', 'FontWeight', 'bold');
title(ax, title_str, 'FontSize', 12, 'FontWeight', 'bold');

% 水平图例置于上方 (northoutside)，彻底避免与数据点重叠
h_mean_dummy = plot(ax, NaN, NaN, 'd', 'MarkerFaceColor', [0.4 0.4 0.4], ...
    'MarkerEdgeColor', 'k', 'MarkerSize', 8, 'LineWidth', 1.0, 'DisplayName', 'Mean ± 95% CI');
legend([h_null, h_mean_dummy], {'Theoretical Chance = 3.5', 'Mean ± 95% CI'}, ...
    'Orientation', 'horizontal', 'Location', 'northoutside', 'Box', 'off', 'FontSize', 9);

end

%% ========================================================================
% 排名构成比堆叠图渲染
% ========================================================================
function lgd = render_dist_tile(ax, tbl, col_name, cfg, title_str, show_legend)
axes(ax); hold(ax, 'on'); grid(ax, 'on');
set(ax, 'Box', 'off', 'LineWidth', 1.1, 'FontSize', 10.5, ...
    'YGrid', 'on', 'XGrid', 'off', 'GridColor', [0.88, 0.88, 0.88], 'GridAlpha', 0.8);

n_b = cfg.n_bands;

% 构建横轴频段标签 (保持单行无换行符，避免被 MATLAB 拆分为多重 Tick)
all_x_labels = [cfg.bands_disp, {'Overall'}];

dist_mat = zeros(n_b + 1, 6);

for b = 1:(n_b + 1)
    if b <= n_b
        sub_mask = strcmp(tbl.freq_band, cfg.bands{b});
    else
        sub_mask = true(height(tbl), 1);
    end
    vals = tbl.(col_name)(sub_mask);
    vals = vals(~isnan(vals));
    n_v  = numel(vals);
    if n_v > 0
        r_round = max(1, min(6, round(vals)));
        for r = 1:6
            dist_mat(b, r) = sum(r_round == r) / n_v * 100;
        end
    end
end

% 6 种排名的色阶配色 (Rank 1: 深蓝绿 -> Rank 6: 浅灰)
rank_palette = [
    0.05, 0.36, 0.40;  % Rank 1: 深海蓝绿 (Top 1)
    0.12, 0.58, 0.52;  % Rank 2: 翡翠绿
    0.35, 0.72, 0.56;  % Rank 3: 青绿
    0.68, 0.76, 0.50;  % Rank 4: 浅橄榄
    0.85, 0.72, 0.45;  % Rank 5: 浅黄褐
    0.82, 0.82, 0.85   % Rank 6: 浅灰
];

b_h = bar(ax, 1:(n_b + 1), dist_mat, 0.62, 'stacked', 'EdgeColor', 'w', 'LineWidth', 0.8);
for r = 1:6
    b_h(r).FaceColor = rank_palette(r, :);
    b_h(r).DisplayName = sprintf('Rank %d', r);
end

% 理论均分基准线 (16.67%)
yline(ax, 100/6, ':', 'Color', [0.25, 0.25, 0.25], 'LineWidth', 1.5, ...
    'DisplayName', 'Chance (16.7%)');

xlim(ax, [0.4, n_b + 1.6]);
ylim(ax, [0, 100]);
set(ax, 'YTick', 0:20:100, 'XTick', 1:(n_b + 1), 'XTickLabel', all_x_labels, 'TickLength', [0.015, 0.025]);
xtickangle(ax, 0);

% 标注样本量 N 与 x 轴标题
for b = 1:(n_b + 1)
    if b <= n_b
        n_v = sum(strcmp(tbl.freq_band, cfg.bands{b}) & ~isnan(tbl.(col_name)));
    else
        n_v = sum(~isnan(tbl.(col_name)));
    end
    text(ax, b, -13, sprintf('(N=%d)', n_v), 'HorizontalAlignment', 'center', ...
        'FontSize', 8.5, 'Color', [0.45, 0.45, 0.45], 'Clipping', 'off');
end

xl = xlabel(ax, 'Task 1 Screening Frequency Band', 'FontWeight', 'bold');
xl.Units = 'data';
xl.Position = [(n_b + 2)/2, -24, 0];
ylabel(ax, 'Proportion of Channels (%)', 'FontWeight', 'bold');
title(ax, title_str, 'FontSize', 12, 'FontWeight', 'bold');

lgd = [];
if show_legend
    lgd = legend(ax, 'Box', 'off', 'FontSize', 9);
end

end

%% ========================================================================
% 辅助函数: 控制台打印统计表
% ========================================================================
function print_stats_table(tbl, bands)
fprintf('------------------------------------------------------------------------\n');
fprintf('%-11s | %4s | %-24s | %-24s\n', 'Band', 'N', 'Task 2 (Mean±SD, t, p)', 'Task 3 (Mean±SD, t, p)');
fprintf('------------------------------------------------------------------------\n');
for b = 1:numel(bands)
    b_name = bands{b};
    sub_t = tbl(strcmp(tbl.freq_band, b_name), :);
    v2 = sub_t.rank_t2(~isnan(sub_t.rank_t2));
    v3 = sub_t.rank_t3(~isnan(sub_t.rank_t3));
    [~, p2, ~, st2] = ttest(v2 - 3.5);
    [~, p3, ~, st3] = ttest(v3 - 3.5);
    fprintf('%-11s | %4d | %4.2f ± %4.2f (t=%+5.2f, p=%.3f) | %4.2f ± %4.2f (t=%+5.2f, p=%.3f)\n', ...
        b_name, height(sub_t), mean(v2), std(v2), st2.tstat, p2, mean(v3), std(v3), st3.tstat, p3);
end
v2_all = tbl.rank_t2(~isnan(tbl.rank_t2));
v3_all = tbl.rank_t3(~isnan(tbl.rank_t3));
[~, p2_all, ~, st2_all] = ttest(v2_all - 3.5);
[~, p3_all, ~, st3_all] = ttest(v3_all - 3.5);
fprintf('------------------------------------------------------------------------\n');
fprintf('%-11s | %4d | %4.2f ± %4.2f (t=%+5.2f, p=%.3f) | %4.2f ± %4.2f (t=%+5.2f, p=%.3f)\n', ...
    'Overall', height(tbl), mean(v2_all), std(v2_all), st2_all.tstat, p2_all, ...
    mean(v3_all), std(v3_all), st3_all.tstat, p3_all);
fprintf('------------------------------------------------------------------------\n');
end
