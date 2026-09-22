function fig = plot_channel_timecourse(sub_id, ch_name, band_name, save_dir, is_visible)
% PLOT_CHANNEL_TIMECOURSE 独立通用科研绘图工具函数 (Nature 级双子图规范)
% 
% 布局与功能:
%   - 主标题: 仅包含被试、电极和频段 (例如: test002  D4  High-Gamma)
%   - 左子图 (1/2 宽): 总体时程差异
%       * 彩色信号: 橙棕色线 + SEM 浅色阴影 (无红绿)
%       * 灰度信号: 灰色线 + SEM 浅色阴影
%       * 图例只标两条主线，不包含阴影
%       * 横轴刻度严格为: -300, 0, 300, 600, 900
%       * 100ms 和 400ms 添加虚线 xline，且在两线之间的顶部标示总体显著性星号
%   - 右子图 (1/2 宽): 四大类别柱状图 (Face, Object, Body, Place)
%       * 每个类别包含: 左柱彩色 (橙棕色), 右柱灰度 (灰色)
%       * 柱顶带 SEM 误差棒，并在每对柱子上方标注类别内显著性连线与星号
%
% 输入参数:
%   sub_id     - 被试编号 (如 'test001', 'test002')
%   ch_name    - 电极通道名称 (如 'D4', 'D14', 'G13')
%   band_name  - (可选) 频段名称: 'Delta','Theta','Alpha','Beta','Low_Gamma','High_Gamma' (默认 'High_Gamma')
%   save_dir   - (可选) 保存图片目录路径，若为空或不传则仅展示不保存
%   is_visible - (可选) 窗口是否可见: 'on' 或 'off' (默认 'on')

if nargin < 3 || isempty(band_name)
    band_name = 'High_Gamma';
end
if nargin < 4
    save_dir = '';
end
if nargin < 5 || isempty(is_visible)
    is_visible = 'on';
end

% -------------------------------------------------------------------------
% 1. 路径与数据加载
% -------------------------------------------------------------------------
% 统一标准化被试编号: 兼容输入 'test002' 或 'sub002' -> 统一规范为 'sub002'
sub_id     = regexprep(sub_id, '^test', 'sub');

script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(script_dir)); % 定位到 color_analyse_0825
data_file  = fullfile(proj_root, 'process_data_new', sub_id, 'task1_multiband_epoched.mat');

if ~isfile(data_file)
    error('未找到数据文件: %s\n请先执行 C03 提取特征！', data_file);
end

mat_obj = load(data_file);
ep_data = mat_obj.epoched_data;

if ~isfield(ep_data, band_name)
    error('数据中不存在频段 [%s]', band_name);
end

ch_idx = find(strcmp(ep_data.channels, ch_name), 1);
if isempty(ch_idx)
    error('被试 [%s] 中未找到通道 [%s]！', sub_id, ch_name);
end

% 提取单通道试次矩阵 [n_trials x n_timepoints] (dB)
epoch_db   = double(squeeze(ep_data.(band_name)(:, ch_idx, :)));
time_ms    = ep_data.time_ms;
trial_info = ep_data.trial_info;
n_pts      = numel(time_ms);

% 核心主窗口 [100, 400] ms
main_win_ms = [100, 400];
win_mask    = (time_ms >= main_win_ms(1)) & (time_ms <= main_win_ms(2));

% -------------------------------------------------------------------------
% 2. 经典科研配色 (Nature 级别: 温暖橙棕 vs 石墨冷灰，严格无红无绿)
% -------------------------------------------------------------------------
col_orange     = [0.82, 0.42, 0.16];       % 主橙棕色 (Terracotta Orange-Brown)
col_orange_sem = [0.94, 0.83, 0.74];       % 橙棕色浅阴影
col_gray       = [0.42, 0.46, 0.50];       % 主灰色 (Slate Graphite Gray)
col_gray_sem   = [0.85, 0.87, 0.89];       % 灰色浅阴影
col_line_dash  = [0.35, 0.35, 0.35];       % 窗口虚线灰

% -------------------------------------------------------------------------
% 3. 数据统计计算 (总体与分类别)
% -------------------------------------------------------------------------
cat_trig_col = [11, 21, 31, 41];
cat_trig_gry = [12, 22, 32, 42];
cat_names    = {'Face', 'Object', 'Body', 'Place'};

is_col = ismember(trial_info.trigger, cat_trig_col);
is_gry = ismember(trial_info.trigger, cat_trig_gry);

% 总体时程均值与 SEM
col_curve_m   = mean(epoch_db(is_col, :), 1, 'omitnan');
col_curve_sem = std(epoch_db(is_col, :), 0, 1, 'omitnan') ./ sqrt(sum(is_col));
gry_curve_m   = mean(epoch_db(is_gry, :), 1, 'omitnan');
gry_curve_sem = std(epoch_db(is_gry, :), 0, 1, 'omitnan') ./ sqrt(sum(is_gry));

% 总体配对显著性 (100-400ms 内单试次均值检验，各类别内分别严格配对)
col_trials_win = mean(epoch_db(is_col, win_mask), 2);
gry_trials_win = mean(epoch_db(is_gry, win_mask), 2);

all_col_pairs = [];
all_gry_pairs = [];
for c = 1:4
    tc = find(trial_info.trigger == cat_trig_col(c));
    tg = find(trial_info.trigger == cat_trig_gry(c));
    [~, ic, ig] = intersect(trial_info.pic_id(tc), trial_info.pic_id(tg));
    all_col_pairs = [all_col_pairs; mean(epoch_db(tc(ic), win_mask), 2)];
    all_gry_pairs = [all_gry_pairs; mean(epoch_db(tg(ig), win_mask), 2)];
end

if numel(all_col_pairs) >= 20
    [~, p_overall] = ttest(all_col_pairs, all_gry_pairs);
else
    [~, p_overall] = ttest2(col_trials_win, gry_trials_win);
end
star_overall = get_star_str(p_overall);

% 四类别柱状图统计 (100-400ms 内各类别 Color vs Gray 均值、SEM 及 p 值)
cat_bar_col_m   = zeros(1, 4);
cat_bar_col_sem = zeros(1, 4);
cat_bar_gry_m   = zeros(1, 4);
cat_bar_gry_sem = zeros(1, 4);
cat_p_vals      = zeros(1, 4);
cat_stars       = cell(1, 4);

for c = 1:4
    tc = (trial_info.trigger == cat_trig_col(c));
    tg = (trial_info.trigger == cat_trig_gry(c));
    
    vals_c = mean(epoch_db(tc, win_mask), 2);
    vals_g = mean(epoch_db(tg, win_mask), 2);
    
    cat_bar_col_m(c)   = mean(vals_c, 'omitnan');
    cat_bar_col_sem(c) = std(vals_c, 0, 'omitnan') ./ sqrt(sum(~isnan(vals_c)));
    cat_bar_gry_m(c)   = mean(vals_g, 'omitnan');
    cat_bar_gry_sem(c) = std(vals_g, 0, 'omitnan') ./ sqrt(sum(~isnan(vals_g)));
    
    % 优先采用同图片配对检验
    tc_idx = find(tc);
    tg_idx = find(tg);
    [c_pics, ic, ig] = intersect(trial_info.pic_id(tc_idx), trial_info.pic_id(tg_idx));
    if ~isempty(c_pics) && numel(c_pics) >= 10
        [~, p_c] = ttest(vals_c(ic), vals_g(ig));
    else
        [~, p_c] = ttest2(vals_c, vals_g);
    end
    
    cat_p_vals(c) = p_c;
    cat_stars{c}  = get_star_str(p_c);
end

% -------------------------------------------------------------------------
% 4. 创建规范图窗与子图绘制 (左右两图各占 1/2 水平宽度)
% -------------------------------------------------------------------------
fig = figure('Color', 'w', 'Position', [100, 150, 1180, 480], 'Visible', is_visible);

% 主标题: 严格仅保留被试、电极和频段
clean_band = strrep(band_name, '_', '-');
sgtitle(sprintf('%s  %s  %s', sub_id, ch_name, clean_band), ...
    'FontSize', 15, 'FontWeight', 'bold', 'FontName', 'Helvetica');

% =========================================================================
% 左子图: 总体时程响应 (1/2 宽，使用精确对称布局 Position)
% =========================================================================
subplot('Position', [0.08, 0.14, 0.38, 0.74]);
hold on;

% 绘制 SEM 阴影 (设置 HandleVisibility 为 off，图例不包含阴影)
fill([time_ms, fliplr(time_ms)], ...
     [col_curve_m + col_curve_sem, fliplr(col_curve_m - col_curve_sem)], ...
     col_orange_sem, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill([time_ms, fliplr(time_ms)], ...
     [gry_curve_m + gry_curve_sem, fliplr(gry_curve_m - gry_curve_sem)], ...
     col_gray_sem, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% 绘制主曲线 (橙棕色线与灰色线，图例只标这两条线)
h_col = plot(time_ms, col_curve_m, 'LineWidth', 2.3, 'Color', col_orange, 'DisplayName', 'Color');
h_gry = plot(time_ms, gry_curve_m, 'LineWidth', 2.3, 'Color', col_gray, 'DisplayName', 'Gray');

% 辅助基线
xline(0, '-', 'Color', [0.75, 0.75, 0.75], 'LineWidth', 0.8, 'HandleVisibility', 'off');
yline(0, ':', 'Color', [0.75, 0.75, 0.75], 'LineWidth', 0.8, 'HandleVisibility', 'off');

% 100ms 和 400ms 虚线
xline(100, '--', 'Color', col_line_dash, 'LineWidth', 1.2, 'HandleVisibility', 'off');
xline(400, '--', 'Color', col_line_dash, 'LineWidth', 1.2, 'HandleVisibility', 'off');

% 横轴刻度严格限制为: -300, 0, 300, 600, 900
xlim([-300, 900]);
set(gca, 'XTick', [-300, 0, 300, 600, 900], 'FontSize', 10.5, 'FontName', 'Helvetica', 'TickDir', 'out');
xlabel('Time (ms)', 'FontSize', 11.5, 'FontWeight', 'bold');
ylabel('Power (dB)', 'FontSize', 11.5, 'FontWeight', 'bold');
box off;

% 动态计算左图 Y 轴上限并在 100-400ms 顶部标记显著性 Bracket
y_curve_max = max(max(col_curve_m + col_curve_sem), max(gry_curve_m + gry_curve_sem));
y_curve_min = min(min(col_curve_m - col_curve_sem), min(gry_curve_m - gry_curve_sem));
y_span_raw  = y_curve_max - y_curve_min;
if y_span_raw <= 0, y_span_raw = 1.0; end

y_top_line = y_curve_max + 0.16 * y_span_raw;
tick_len   = 0.03 * y_span_raw;
y_top_text = y_top_line + 0.02 * y_span_raw;

% 绘制顶部显著性 Bracket 与星号
plot([100, 400], [y_top_line, y_top_line], '-', 'Color', [0.25, 0.25, 0.25], 'LineWidth', 1.1, 'HandleVisibility', 'off');
plot([100, 100], [y_top_line - tick_len, y_top_line], '-', 'Color', [0.25, 0.25, 0.25], 'LineWidth', 1.1, 'HandleVisibility', 'off');
plot([400, 400], [y_top_line - tick_len, y_top_line], '-', 'Color', [0.25, 0.25, 0.25], 'LineWidth', 1.1, 'HandleVisibility', 'off');
text(250, y_top_text, star_overall, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
     'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');

% 设置 Y 轴范围，为顶部星号留出呼吸空间
ylim([y_curve_min - 0.08 * y_span_raw, y_top_text + 0.20 * y_span_raw]);

% 仅标注两条主线
legend([h_col, h_gry], {'Color', 'Gray'}, 'Location', 'northwest', 'Box', 'off', 'FontSize', 11);
title('Overall Difference', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');

% =========================================================================
% 右子图: 四大类别柱状图对比 (1/2 宽，使用精确对称布局 Position)
% =========================================================================
subplot('Position', [0.55, 0.14, 0.38, 0.74]);
hold on;

% 组织柱状图数据: 4个类别 x 2种条件 (左柱彩色，右柱灰度)
bar_data = [cat_bar_col_m(:), cat_bar_gry_m(:)]; % [4 x 2]
b_plot = bar([1, 2, 3, 4], bar_data, 0.72, 'grouped');
b_plot(1).FaceColor = col_orange;
b_plot(1).EdgeColor = 'none';
b_plot(1).DisplayName = 'Color';
b_plot(2).FaceColor = col_gray;
b_plot(2).EdgeColor = 'none';
b_plot(2).DisplayName = 'Gray';

% 获取各柱中心坐标并添加误差棒
x_col = b_plot(1).XEndPoints;
x_gry = b_plot(2).XEndPoints;

errorbar(x_col, cat_bar_col_m, cat_bar_col_sem, 'k.', 'LineWidth', 1.1, 'CapSize', 4, 'HandleVisibility', 'off');
errorbar(x_gry, cat_bar_gry_m, cat_bar_gry_sem, 'k.', 'LineWidth', 1.1, 'CapSize', 4, 'HandleVisibility', 'off');

yline(0, ':', 'Color', [0.75, 0.75, 0.75], 'LineWidth', 0.8, 'HandleVisibility', 'off');

% 在每个类别上方标注显著性 Bracket 与星号
all_bars_top = max(cat_bar_col_m + cat_bar_col_sem, cat_bar_gry_m + cat_bar_gry_sem);
all_bars_bot = min([cat_bar_col_m - cat_bar_col_sem, cat_bar_gry_m - cat_bar_gry_sem, 0]);
y_r_span     = max(all_bars_top) - min(all_bars_bot);
if y_r_span <= 0, y_r_span = 1.0; end

max_annot_y = max(all_bars_top);
tick_r      = 0.025 * y_r_span;

for c = 1:4
    top_c   = max([cat_bar_col_m(c) + cat_bar_col_sem(c), cat_bar_gry_m(c) + cat_bar_gry_sem(c), 0]);
    y_bar_h = top_c + 0.10 * y_r_span;
    y_bar_t = y_bar_h + 0.015 * y_r_span;
    
    % 绘制类别上方 bracket 横线与端点短竖线
    plot([x_col(c), x_gry(c)], [y_bar_h, y_bar_h], '-', 'Color', [0.25, 0.25, 0.25], 'LineWidth', 1.0, 'HandleVisibility', 'off');
    plot([x_col(c), x_col(c)], [y_bar_h - tick_r, y_bar_h], '-', 'Color', [0.25, 0.25, 0.25], 'LineWidth', 1.0, 'HandleVisibility', 'off');
    plot([x_gry(c), x_gry(c)], [y_bar_h - tick_r, y_bar_h], '-', 'Color', [0.25, 0.25, 0.25], 'LineWidth', 1.0, 'HandleVisibility', 'off');
    
    text(mean([x_col(c), x_gry(c)]), y_bar_t, cat_stars{c}, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 11, 'FontWeight', 'bold', 'FontName', 'Helvetica');
    
    max_annot_y = max(max_annot_y, y_bar_t + 0.06 * y_r_span);
end

% 留足顶部空间，彻底避免图例与第4类别产生任何重叠
ylim([all_bars_bot - 0.05 * y_r_span, max_annot_y + 0.35 * y_r_span]);
xlim([0.4, 4.6]);
set(gca, 'XTick', 1:4, 'XTickLabel', cat_names, 'FontSize', 10.5, 'FontName', 'Helvetica', 'TickDir', 'out');
xlabel('Category', 'FontSize', 11.5, 'FontWeight', 'bold');
ylabel('Power (dB)', 'FontSize', 11.5, 'FontWeight', 'bold');
box off;

legend({'Color', 'Gray'}, 'Location', 'northeast', 'Box', 'off', 'FontSize', 11);
title('Category Responses (100–400 ms)', 'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Helvetica');

% -------------------------------------------------------------------------
% 5. 保存图片 (若指定 save_dir)
% -------------------------------------------------------------------------
if ~isempty(save_dir)
    if ~exist(save_dir, 'dir'), mkdir(save_dir); end
    out_file = fullfile(save_dir, sprintf('%s_%s_%s_timecourse.png', sub_id, ch_name, band_name));
    exportgraphics(fig, out_file, 'Resolution', 300);
    fprintf('[+] 精美学术时程图已保存: %s\n', out_file);
end

end

% =========================================================================
% 辅助函数: 显著性星号转换
% =========================================================================
function s = get_star_str(p)
    if isnan(p)
        s = 'n.s.';
    elseif p < 0.001
        s = '***';
    elseif p < 0.01
        s = '**';
    elseif p < 0.05
        s = '*';
    else
        s = 'n.s.';
    end
end
