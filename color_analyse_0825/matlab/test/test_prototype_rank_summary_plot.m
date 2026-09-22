%% test_prototype_rank_summary_plot.m
% =========================================================================
% 功能: 原型测试 Task 1 筛选频段在 Task 2 & Task 3 中的 Decoding Rank 汇总分析与出图
% =========================================================================
clear; clc; close all;

proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
res_root  = fullfile(proj_root, 'result', 'tables');
out_fig_dir = fullfile(proj_root, 'result', 'figures', 'decoding_rank_summary');
if ~exist(out_fig_dir, 'dir'), mkdir(out_fig_dir); end

% 1. 加载数据
f_t1 = fullfile(res_root, 'color_effects_summary.mat');
f_t2_c = fullfile(res_root, 'concordant_electrodes_decoding_summary.mat');
f_t2_nc = fullfile(res_root, 'non_concordant_electrodes_decoding_summary.mat');
f_t3 = fullfile(res_root, 'task3_purecolor_decoding_summary.mat');

d1 = load(f_t1);
t1 = d1.all_tbl;
t1_sig = t1(t1.is_significant == 1, :);

d2_c = load(f_t2_c); t2_c = d2_c.summary_table;
d2_nc = load(f_t2_nc); t2_nc = d2_nc.summary_table;
t2 = [t2_c; t2_nc];

d3 = load(f_t3); t3 = d3.summary_table;

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
bands_disp = {'\delta', '\theta', '\alpha', '\beta', '\gamma', 'High \gamma'};
n_bands = numel(bands);

% 配色 (与全项目保持一致)
band_cols = [
    0.40, 0.40, 0.40;  % Delta: 灰
    0.92, 0.58, 0.18;  % Theta: 杏黄
    0.18, 0.45, 0.75;  % Alpha: 钴蓝
    0.08, 0.62, 0.42;  % Beta: 青绿
    0.50, 0.32, 0.72;  % Low_Gamma: 紫灰
    0.88, 0.15, 0.48   % High_Gamma: 玫红
];
col_overall = [0.20, 0.20, 0.20]; % Overall: 深黑灰

% 2. 逐条目提取 Rank
n_items = height(t1_sig);
ranks_t2 = nan(n_items, 1);
ranks_t3 = nan(n_items, 1);

for i = 1:n_items
    sub = t1_sig.subject{i};
    ch  = t1_sig.channel{i};
    b_screen = t1_sig.freq_band{i};
    b_idx = find(strcmp(bands, b_screen));
    
    % Task 2
    idx2 = find(strcmp(t2.subject, sub) & strcmp(t2.channel, ch), 1);
    if ~isempty(idx2)
        row2 = t2(idx2, :);
        accs2 = [row2.peak_acc_Delta, row2.peak_acc_Theta, row2.peak_acc_Alpha, ...
                 row2.peak_acc_Beta, row2.peak_acc_Low_Gamma, row2.peak_acc_High_Gamma];
        rnk2 = tiedrank(-accs2); % 降序: 1 = 最高acc, 6 = 最低acc
        ranks_t2(i) = rnk2(b_idx);
    end
    
    % Task 3
    idx3 = find(strcmp(t3.subject, sub) & strcmp(t3.channel, ch), 1);
    if ~isempty(idx3)
        row3 = t3(idx3, :);
        accs3 = [row3.peak_acc_Delta, row3.peak_acc_Theta, row3.peak_acc_Alpha, ...
                 row3.peak_acc_Beta, row3.peak_acc_Low_Gamma, row3.peak_acc_High_Gamma];
        rnk3 = tiedrank(-accs3);
        ranks_t3(i) = rnk3(b_idx);
    end
end

t1_sig.rank_t2 = ranks_t2;
t1_sig.rank_t3 = ranks_t3;

% 3. 构造 2x2 高清综合汇总图
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50, 50, 1300, 780]);

% 子图 1: Task 2 均值与散点对比 (vs 3.5)
ax1 = subplot(2, 2, 1); hold(ax1, 'on');
render_rank_scatter_panel(ax1, t1_sig, 'rank_t2', bands, bands_disp, band_cols, col_overall, ...
    'Task 2 (Memory Color Decoding)', 'Decoding Rank of Task 1 Screening Band');

% 子图 2: Task 3 均值与散点对比 (vs 3.5)
ax2 = subplot(2, 2, 2); hold(ax2, 'on');
render_rank_scatter_panel(ax2, t1_sig, 'rank_t3', bands, bands_disp, band_cols, col_overall, ...
    'Task 3 (Pure Color Decoding)', 'Decoding Rank of Task 1 Screening Band');

% 子图 3: Task 2 排名构成比 (Rank 1st ~ 6th 堆叠柱状图)
ax3 = subplot(2, 2, 3); hold(ax3, 'on');
render_rank_dist_panel(ax3, t1_sig, 'rank_t2', bands, bands_disp, 'Task 2: Rank Proportion by Band');

% 子图 4: Task 3 排名构成比 (Rank 1st ~ 6th 堆叠柱状图)
ax4 = subplot(2, 2, 4); hold(ax4, 'on');
render_rank_dist_panel(ax4, t1_sig, 'rank_t3', bands, bands_disp, 'Task 3: Rank Proportion by Band');

% 导出图像
out_png = fullfile(out_fig_dir, 'task1_screening_vs_task2_task3_decoding_ranks.png');
exportgraphics(fig, out_png, 'Resolution', 300);
fprintf('[✓] 汇总图已成功导出至:\n    %s\n', out_png);
close(fig);

% =========================================================================
% 子函数: 绘制上半行散点与均值检验子图
% =========================================================================
function render_rank_scatter_panel(ax, tbl, rank_col, bands, bands_disp, band_cols, col_overall, title_str, y_label_str)
axes(ax); grid(ax, 'on');
set(ax, 'Box', 'off', 'LineWidth', 1.1, 'FontSize', 10, ...
    'YGrid', 'on', 'XGrid', 'off', 'GridColor', [0.88, 0.88, 0.88], 'GridAlpha', 0.8);

n_b = numel(bands);
all_x_names = [bands_disp, {'Overall'}];

% 1. 理论无相关基线 (Rank = 3.5)
yline(ax, 3.5, '--', 'Color', [0.85, 0.25, 0.20], 'LineWidth', 1.8, ...
    'DisplayName', 'Chance / Unrelated (Mean = 3.5)');

rng(42); % 固定抖动随机种子

% 逐频段绘制散点与均值
for b = 1:(n_b + 1)
    if b <= n_b
        sub_mask = strcmp(tbl.freq_band, bands{b});
        c_col = band_cols(b, :);
    else
        sub_mask = true(height(tbl), 1);
        c_col = col_overall;
    end
    
    vals = tbl.(rank_col)(sub_mask);
    vals = vals(~isnan(vals));
    n_v  = numel(vals);
    
    if n_v == 0, continue; end
    
    % Jitter 散点
    jitter_x = b + (rand(n_v, 1) - 0.5) * 0.38;
    scatter(ax, jitter_x, vals, 28, 'MarkerFaceColor', c_col, ...
        'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.35, 'HandleVisibility', 'off');
    
    % 统计值
    m_val   = mean(vals);
    sd_val  = std(vals);
    sem_val = sd_val / sqrt(n_v);
    ci_95   = tinv(0.975, n_v - 1) * sem_val;
    [~, p_t] = ttest(vals - 3.5);
    
    % 绘制 95% CI 误差棒
    errorbar(ax, b, m_val, ci_95, 'LineStyle', 'none', 'Color', [0.15, 0.15, 0.15], ...
        'LineWidth', 2.0, 'CapSize', 8, 'HandleVisibility', 'off');
    
    % 绘制均值中心菱形点
    plot(ax, b, m_val, 'd', 'MarkerFaceColor', c_col, 'MarkerEdgeColor', 'k', ...
        'MarkerSize', 9, 'LineWidth', 1.2, 'HandleVisibility', 'off');
    
    % 顶部标注均值与 p 值
    if p_t < 0.001
        sig_tag = 'p<.001 ***';
    elseif p_t < 0.01
        sig_tag = sprintf('p=%.3f **', p_t);
    elseif p_t < 0.05
        sig_tag = sprintf('p=%.3f *', p_t);
    else
        sig_tag = sprintf('p=%.3f (ns)', p_t);
    end
    
    text(ax, b, 6.25, sprintf('%.2f', m_val), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', 'Color', [0.15, 0.15, 0.15]);
    text(ax, b, 0.75, sprintf('N=%d\n%s', n_v, sig_tag), ...
        'HorizontalAlignment', 'center', 'FontSize', 7.5, 'Color', [0.35, 0.35, 0.35]);
end

xlim(ax, [0.4, n_b + 1.6]);
ylim(ax, [0.5, 6.5]);
set(ax, 'YTick', 1:6, 'YTickLabel', {'1 (Top)', '2', '3', '4', '5', '6 (Low)'}, ...
    'XTick', 1:(n_b + 1), 'XTickLabel', all_x_names);

xlabel(ax, 'Task 1 Screening Frequency Band', 'FontWeight', 'bold');
ylabel(ax, y_label_str, 'FontWeight', 'bold');
title(ax, title_str, 'FontSize', 12, 'FontWeight', 'bold');
legend(ax, 'Location', 'northeast', 'Box', 'off', 'FontSize', 8.5);
end

% =========================================================================
% 子函数: 绘制下半行排名构成比 (Stacked Bar Plot)
% =========================================================================
function render_rank_dist_panel(ax, tbl, rank_col, bands, bands_disp, title_str)
axes(ax); grid(ax, 'on');
set(ax, 'Box', 'off', 'LineWidth', 1.1, 'FontSize', 10, ...
    'YGrid', 'on', 'XGrid', 'off', 'GridColor', [0.88, 0.88, 0.88], 'GridAlpha', 0.8);

n_b = numel(bands);
all_x_names = [bands_disp, {'Overall'}];

% 计算每个类别各 Rank (1~6) 的百分比
dist_mat = zeros(n_b + 1, 6);

for b = 1:(n_b + 1)
    if b <= n_b
        sub_mask = strcmp(tbl.freq_band, bands{b});
    else
        sub_mask = true(height(tbl), 1);
    end
    vals = tbl.(rank_col)(sub_mask);
    vals = vals(~isnan(vals));
    n_v  = numel(vals);
    if n_v > 0
        % 统计四舍五入后的秩 (避免 half-ties)
        r_rounded = round(vals);
        r_rounded = max(1, min(6, r_rounded));
        for r = 1:6
            dist_mat(b, r) = sum(r_rounded == r) / n_v * 100;
        end
    end
end

% 6 种排名的色阶配色 (从 Rank 1 顶尖深色 到 Rank 6 浅灰)
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
yline(ax, 100/6, ':', 'Color', [0.3, 0.3, 0.3], 'LineWidth', 1.5, ...
    'DisplayName', 'Chance (16.7%)');

xlim(ax, [0.4, n_b + 1.6]);
ylim(ax, [0, 100]);
set(ax, 'YTick', 0:20:100, 'XTick', 1:(n_b + 1), 'XTickLabel', all_x_names);
xlabel(ax, 'Task 1 Screening Frequency Band', 'FontWeight', 'bold');
ylabel(ax, 'Proportion of Channels (%)', 'FontWeight', 'bold');
title(ax, title_str, 'FontSize', 12, 'FontWeight', 'bold');

if contains(title_str, 'Task 2')
    legend(ax, 'Location', 'eastoutside', 'Box', 'off', 'FontSize', 8);
else
    legend(ax, 'off');
end
end
