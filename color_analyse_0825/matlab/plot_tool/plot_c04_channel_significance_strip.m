%% plot_c04_channel_significance_strip.m
% =========================================================================
% 功能: 绘制 C04 筛选电极在各频段的显著性横向色带图 (Strip Chart)
% 结构特点:
%   1. X 轴: 被试所有电极横向排列，组成无填充色、不间断的电极条带；
%   2. Y 轴: 6 行分别对应 6 个频段 (从高频到低频排列)；
%   3. 填充规则: 无边框 (EdgeColor = 'none')；
%      - 总体正向增强 -> 红色
%      - 总体负向抑制 -> 蓝色
%      - 类别偏向不同向 -> 橙色
%      - 未显著 -> 无填充色 (透明/纯白)
%   4. 先行支持 sub001，参数在 pipeline 顶部直观调节，便于后续全被试横向扩展。
% =========================================================================

clear; clc; close all;

%% 1. 核心参数配置 (简单直观可调)
cfg = struct();
cfg.sub_id      = 'sub001';          % 目标被试
cfg.alpha_sig   = 0.05;              % 显著性判定阈值
cfg.fig_size    = [1650, 480];       % 画布尺寸 [宽, 高]
cfg.dpi         = 300;               % 图像导出分辨率

% 三种总体显著类型的颜色定义 (顶刊标准配色，无边框纯色填充)
cfg.col_pos     = [0.86, 0.20, 0.20];  % 总体正向增强: 纯正鲜红
cfg.col_neg     = [0.15, 0.45, 0.75];  % 总体负向抑制: 深邃宝石蓝
cfg.col_bias    = [0.98, 0.55, 0.10];  % 类别偏向不同向: 温暖琥珀橙
cfg.col_empty   = [1.00, 1.00, 1.00];  % 未显著: 无填充 (纯白)

%% 2. 路径配置
script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(script_dir)); % 定位到 color_analyse_0825
proc_new   = fullfile(proj_root, 'process_data_new');
res_root   = fullfile(proj_root, 'result');
res_table  = fullfile(res_root, 'tables', 'color_effects_summary.mat');
out_fig_dir= fullfile(res_root, 'figures', 'c04_screening');

if ~exist(out_fig_dir, 'dir'), mkdir(out_fig_dir); end

%% 3. 读取被试全通道与频段数据
% 读取 Task 1 多频段特征文件以获取自然排列的全部电极列表
ep_file = fullfile(proc_new, cfg.sub_id, 'task1_multiband_epoched.mat');
if ~isfile(ep_file)
    error('未找到被试特征缓存文件: %s', ep_file);
end
ep_mat = load(ep_file);
channels = ep_mat.epoched_data.triplet_info.center_channel; % 82 个电极
n_ch     = numel(channels);

% 频段由高频到低频依次排列成 6 行
band_keys   = {'High_Gamma', 'Low_Gamma', 'Beta', 'Alpha', 'Theta', 'Delta'};
band_labels = {'High \gamma (60-140Hz)', 'Low \gamma (30-60Hz)', ...
               '\beta (13-30Hz)', '\alpha (8-13Hz)', '\theta (4-8Hz)', '\delta (1-4Hz)'};
n_bands     = numel(band_keys);

% 读取 C04 导出的全量汇总明细表
if ~isfile(res_table)
    error('未找到 C04 汇总明细表: %s\n请先执行 C04_screen_color_channels_0825.m！', res_table);
end
loaded_c04 = load(res_table);
if isfield(loaded_c04, 'all_tbl'), tbl = loaded_c04.all_tbl; else, tbl = loaded_c04.res_table; end
sub_tbl = tbl(strcmp(tbl.subject, cfg.sub_id), :);

%% 4. 构建显著性分类矩阵 (n_bands x n_ch)
% 状态码: 0 = 未显著, 1 = 总体正向, 2 = 总体负向, 3 = 不同向
sig_mat = zeros(n_bands, n_ch);

for ch = 1:n_ch
    c_name = channels{ch};
    for b = 1:n_bands
        b_name = band_keys{b};
        
        row_mask = strcmp(sub_tbl.channel, c_name) & strcmp(sub_tbl.freq_band, b_name);
        if any(row_mask)
            r = sub_tbl(row_mask, :);
            if r.p_perm_100_400ms(1) < cfg.alpha_sig
                c_type = r.concordance_type{1};
                if strcmp(c_type, 'Concordant_Positive')
                    sig_mat(b, ch) = 1;
                elseif strcmp(c_type, 'Concordant_Negative')
                    sig_mat(b, ch) = 2;
                else
                    sig_mat(b, ch) = 3;
                end
            end
        end
    end
end

% 提取电极柄 (Shaft) 分组信息用于在顶部条带划分
shaft_letters = cell(n_ch, 1);
for ch = 1:n_ch
    tok = regexp(channels{ch}, '^([A-Za-z]+)', 'tokens');
    if ~isempty(tok)
        shaft_letters{ch} = tok{1}{1};
    else
        shaft_letters{ch} = '?';
    end
end

%% 5. 创建画布与绘图
fig = figure('Color', 'w', 'Position', [80, 200, cfg.fig_size(1), cfg.fig_size(2)], 'Visible', 'off');

% 主绘图坐标系: 留出顶部条带和底部图例的空间
ax = axes('Position', [0.08, 0.22, 0.90, 0.60]);
hold(ax, 'on');

% (1) 绘制整个条带的外轮廓与浅色无填充背景 (不间断连续条带)
rectangle('Position', [0.5, 0.5, n_ch, n_bands], ...
          'FaceColor', 'w', 'EdgeColor', [0.82, 0.82, 0.85], 'LineWidth', 1.0);

% (2) 逐单元格绘制显著填充块 (严格无边框 EdgeColor = 'none')
for b = 1:n_bands
    for ch = 1:n_ch
        val = sig_mat(b, ch);
        if val > 0
            switch val
                case 1, cur_col = cfg.col_pos;
                case 2, cur_col = cfg.col_neg;
                case 3, cur_col = cfg.col_bias;
            end
            % 无边框单块贴合填充
            patch([ch-0.5, ch+0.5, ch+0.5, ch-0.5], ...
                  [b-0.5, b-0.5, b+0.5, b+0.5], ...
                  cur_col, 'EdgeColor', 'none', 'Parent', ax);
        end
    end
end

% (3) 绘制电极柄 (Shaft) 之间的垂直微弱分割线 (贯穿整列)
shaft_bounds = [1];
for ch = 2:n_ch
    if ~strcmp(shaft_letters{ch}, shaft_letters{ch-1})
        shaft_bounds = [shaft_bounds, ch];
        x_sep = ch - 0.5;
        line([x_sep, x_sep], [-0.65, n_bands + 0.5], 'Color', [0.82, 0.82, 0.86], ...
             'LineStyle', '-', 'LineWidth', 1.0, 'Parent', ax);
    end
end
shaft_bounds = [shaft_bounds, n_ch + 1];

% (4) 绘制每行频段之间的微细水平分隔线 (增强条带横向流线感)
for b = 1:n_bands-1
    line([0.5, n_ch + 0.5], [b + 0.5, b + 0.5], 'Color', [0.92, 0.92, 0.94], ...
         'LineStyle', '-', 'LineWidth', 0.6, 'Parent', ax);
end

% (5) 顶部电极条带 (一条没有填充色的不间断条带，标注各电极柄 Shaft)
rectangle('Position', [0.5, -0.65, n_ch, 0.55], ...
          'FaceColor', 'none', 'EdgeColor', [0.65, 0.65, 0.72], 'LineWidth', 1.2, 'Parent', ax);

for s = 1:numel(shaft_bounds)-1
    st_ch = shaft_bounds(s);
    ed_ch = shaft_bounds(s+1) - 1;
    mid_ch = (st_ch + ed_ch) / 2;
    s_label = shaft_letters{st_ch};
    
    % 在顶部条带内标明电极柄名称
    text(mid_ch, -0.38, sprintf('Shaft %s', s_label), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
         'FontWeight', 'bold', 'FontSize', 10.5, 'Color', [0.20, 0.20, 0.25], 'Parent', ax);
end

% (6) 坐标轴与标签配置
set(ax, 'XLim', [0.5, n_ch + 0.5], 'YLim', [-0.75, n_bands + 0.5]);
set(ax, 'YDir', 'reverse'); % 顶端为 High_Gamma，底端为 Delta
set(ax, 'YTick', 1:n_bands, 'YTickLabel', band_labels, ...
        'FontSize', 9.5, 'FontWeight', 'bold', 'TickLength', [0, 0]);

set(ax, 'XTick', 1:n_ch, 'XTickLabel', channels, ...
        'XTickLabelRotation', 90, 'FontSize', 8, 'TickLength', [0.003, 0]);
set(ax, 'Box', 'off');

% 主标题
n_sig_tot = sum(sig_mat(:) > 0);
n_pos_tot = sum(sig_mat(:) == 1);
n_neg_tot = sum(sig_mat(:) == 2);
n_bias_tot= sum(sig_mat(:) == 3);

title_str = sprintf('\\fontsize{12.5}{\\bf %s 全电极色彩效应显著性分布色带图}  \\fontsize{9.5}\\color[rgb]{0.35,0.35,0.35}(总电极数: %d | 显著事件: %d: 正向增强 %d, 负向抑制 %d, 不同向 %d)', ...
                    cfg.sub_id, n_ch, n_sig_tot, n_pos_tot, n_neg_tot, n_bias_tot);
title(ax, title_str, 'Interpreter', 'tex', 'Position', [n_ch/2 + 0.5, -1.05, 0]);

% (7) 底部居中图例 (无边框纯色色块指示)
leg_ax = axes('Position', [0.15, 0.03, 0.70, 0.08]);
hold(leg_ax, 'on');
set(leg_ax, 'XLim', [0, 100], 'YLim', [0, 1], 'Visible', 'off');

% 4 个图例项的位置与标签
leg_items = {
    cfg.col_pos,   sprintf('总体正向增强 (Concordant Positive, n=%d)', n_pos_tot);
    cfg.col_neg,   sprintf('总体负向抑制 (Concordant Negative, n=%d)', n_neg_tot);
    cfg.col_bias,  sprintf('类别偏向不同向 (Category Biased, n=%d)', n_bias_tot);
    cfg.col_empty, '未显著 (p \geq 0.05, 无填充色)'
};

x_pos = [5, 30, 58, 83];
for i = 1:4
    col = leg_items{i, 1};
    txt = leg_items{i, 2};
    % 色块
    if i == 4
        patch([x_pos(i), x_pos(i)+3, x_pos(i)+3, x_pos(i)], [0.2, 0.2, 0.8, 0.8], ...
              col, 'EdgeColor', [0.75, 0.75, 0.75], 'LineWidth', 0.8, 'Parent', leg_ax);
    else
        patch([x_pos(i), x_pos(i)+3, x_pos(i)+3, x_pos(i)], [0.2, 0.2, 0.8, 0.8], ...
              col, 'EdgeColor', 'none', 'Parent', leg_ax);
    end
    % 文本
    text(x_pos(i) + 4.2, 0.5, txt, 'FontSize', 9.5, 'VerticalAlignment', 'middle', ...
         'FontWeight', 'normal', 'Interpreter', 'tex', 'Parent', leg_ax);
end

%% 6. 保存高质量图像
out_png = fullfile(out_fig_dir, sprintf('%s_channel_significance_strip.png', cfg.sub_id));
if isfile(out_png)
    try delete(out_png); catch, end
end
exportgraphics(fig, out_png, 'Resolution', cfg.dpi);
fprintf('\n[+] 高清色带图已导出至: %s\n', out_png);

% 同时保存一份到 metadata 便于索引展示
meta_fig_dir = fullfile(proj_root, 'metadata', 'figures');
if ~exist(meta_fig_dir, 'dir'), mkdir(meta_fig_dir); end
meta_png = fullfile(meta_fig_dir, sprintf('%s_channel_significance_strip.png', cfg.sub_id));
if isfile(meta_png)
    try delete(meta_png); catch, end
end
copyfile(out_png, meta_png);

fprintf('[+] 绘图完成！\n');
