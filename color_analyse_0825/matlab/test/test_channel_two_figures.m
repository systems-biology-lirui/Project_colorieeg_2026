%% test_channel_two_figures.m
clear; clc; close all;
root_dir = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825';
data_root = fullfile(root_dir, 'process_data_new');
task_info = fullfile(root_dir, 'task_info');
out_fig_dir = fullfile(root_dir, 'result', 'figures', 'rsa_per_electrode');
if ~exist(out_fig_dir, 'dir'), mkdir(out_fig_dir); end

sub_id = 'sub008';
elec   = 'C10';

t3_mat = fullfile(data_root, sub_id, 'task3_multiband_epoched.mat');
t2_mat = fullfile(data_root, sub_id, 'task2_multiband_epoched.mat');

t_axis = double(h5read(t3_mat, '/epoched_data/time_ms')); t_axis = t_axis(:)';
ch_list3 = h5read(t3_mat, '/epoched_data/channels');
ch_list2 = h5read(t2_mat, '/epoched_data/channels');
if iscell(ch_list3), ch_list3 = cellfun(@(x) char(x(:)'), ch_list3, 'UniformOutput', false); end
if iscell(ch_list2), ch_list2 = cellfun(@(x) char(x(:)'), ch_list2, 'UniformOutput', false); end

e_idx3 = find(strcmp(ch_list3, elec), 1);
e_idx2 = find(strcmp(ch_list2, elec), 1);

d3 = load(fullfile(task_info, sub_id, 'task3_trial_info.mat')); ti3 = d3.trial_info;
d2 = load(fullfile(task_info, sub_id, 'task2_trial_info.mat')); ti2 = d2.trial_info;

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
n_bands = numel(bands);

% 10 类别定义
cond_names_10 = {
    'Gray Strawberry', 'Gray Watermelon', 'Gray Kiwi', 'Gray Cabbage', ...
    'Red Shape 1', 'Red Shape 2', 'Red Shape 3', ...
    'Green Shape 1', 'Green Shape 2', 'Green Shape 3'
};
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

% 4 类别定义 (相同灰色记忆平均, 相同纯色色块平均)
cond_names_4 = {
    'Red Memory (Gray)', ...
    'Green Memory (Gray)', ...
    'Red Patch (Pure)', ...
    'Green Patch (Pure)'
};
cond_masks_4 = cell(4, 1);
cond_masks_4{1} = strcmp(ti2.state, 'gray') & (strcmp(ti2.fruit, 'strawberry') | strcmp(ti2.fruit, 'watermelon'));
cond_masks_4{2} = strcmp(ti2.state, 'gray') & (strcmp(ti2.fruit, 'kiwi') | strcmp(ti2.fruit, 'cabbage'));
cond_masks_4{3} = strcmp(ti3.color, 'red');
cond_masks_4{4} = strcmp(ti3.color, 'green');

% 载入多频段数据
X3 = zeros(height(ti3), n_bands, numel(t_axis));
X2 = zeros(height(ti2), n_bands, numel(t_axis));
for b = 1:n_bands
    raw3 = h5read(t3_mat, ['/epoched_data/' bands{b}]);
    X3(:, b, :) = raw3(:, e_idx3, :);
    raw2 = h5read(t2_mat, ['/epoched_data/' bands{b}]);
    X2(:, b, :) = raw2(:, e_idx2, :);
end

% 滑动时间窗设置
win_len = 50; win_step = 20;
t_starts = -200 : win_step : (800 - win_len);
n_win = numel(t_starts);
t_centers = t_starts + win_len / 2;

% 计算逐个时间点的 RDM (50ms 滑动时间窗)
rdms_10_time = zeros(10, 10, n_win);
rdms_4_time  = zeros(4, 4, n_win);

for w = 1:n_win
    w_t1 = t_starts(w); w_t2 = w_t1 + win_len;
    t_m = (t_axis >= w_t1) & (t_axis < w_t2);
    
    % 10 类别特征 (6 bands x 时间点)
    f10 = zeros(10, n_bands * sum(t_m));
    for c = 1:10
        if c <= 4
            sig = squeeze(mean(X2(cond_masks_10{c}, :, t_m), 1));
        else
            sig = squeeze(mean(X3(cond_masks_10{c}, :, t_m), 1));
        end
        f10(c, :) = sig(:)';
    end
    rdms_10_time(:, :, w) = 1 - corr(f10');
    
    % 4 类别特征
    f4 = zeros(4, n_bands * sum(t_m));
    for c = 1:4
        if c <= 2
            sig = squeeze(mean(X2(cond_masks_4{c}, :, t_m), 1));
        else
            sig = squeeze(mean(X3(cond_masks_4{c}, :, t_m), 1));
        end
        f4(c, :) = sig(:)';
    end
    rdms_4_time(:, :, w) = 1 - corr(f4');
end

% 计算时间窗平均特征 [100, 600] ms
t_avg_mask = (t_axis >= 100) & (t_axis <= 600);
f10_avg = zeros(10, n_bands * sum(t_avg_mask));
for c = 1:10
    if c <= 4
        sig = squeeze(mean(X2(cond_masks_10{c}, :, t_avg_mask), 1));
    else
        sig = squeeze(mean(X3(cond_masks_10{c}, :, t_avg_mask), 1));
    end
    f10_avg(c, :) = sig(:)';
end
rdm_10_avg = 1 - corr(f10_avg');

f4_avg = zeros(4, n_bands * sum(t_avg_mask));
for c = 1:4
    if c <= 2
        sig = squeeze(mean(X2(cond_masks_4{c}, :, t_avg_mask), 1));
    else
        sig = squeeze(mean(X3(cond_masks_4{c}, :, t_avg_mask), 1));
    end
    f4_avg(c, :) = sig(:)';
end
rdm_4_avg = 1 - corr(f4_avg');

% =========================================================================
%% 图 1: 时间窗平均图 (平均图: 左图 10 类别 RDM, 中间图 4 类别 RDM, 右图 3D MDS)
% =========================================================================
h_fig1 = figure('Units', 'pixels', 'Position', [50, 100, 1600, 480], 'Color', 'w');

% --- 左图: 10 类别 RDM ---
subplot(1, 3, 1);
imagesc(rdm_10_avg);
colormap(gca, 'parula');
cb = colorbar; ylabel(cb, 'Distance (1 - r)', 'FontSize', 9, 'FontWeight', 'bold');
caxis([0, max(rdm_10_avg(:))]);
axis square;
set(gca, 'XTick', 1:10, 'XTickLabel', cond_names_10, 'XTickLabelRotation', 45, ...
         'YTick', 1:10, 'YTickLabel', cond_names_10, 'FontSize', 8, 'TickDir', 'out');
title({'A. 10-Condition RDM (Averaged [100, 600] ms)', ...
       '(4 Gray Fruits + 2 Colors * 3 Shapes)'}, 'FontSize', 10, 'FontWeight', 'bold');
hold on;
xline(4.5, 'w-', 'LineWidth', 2.0); yline(4.5, 'w-', 'LineWidth', 2.0);
xline(7.5, 'w--', 'LineWidth', 1.2); yline(7.5, 'w--', 'LineWidth', 1.2);
hold off;

% --- 中间图: 4 类别 RDM (相同记忆平均, 相同纯色平均) ---
subplot(1, 3, 2);
imagesc(rdm_4_avg);
colormap(gca, 'parula');
cb = colorbar; ylabel(cb, 'Distance (1 - r)', 'FontSize', 9, 'FontWeight', 'bold');
caxis([0, max(rdm_4_avg(:))]);
axis square;
set(gca, 'XTick', 1:4, 'XTickLabel', cond_names_4, 'XTickLabelRotation', 35, ...
         'YTick', 1:4, 'YTickLabel', cond_names_4, 'FontSize', 8.5, 'TickDir', 'out');
title({'B. Collapsed 4-Condition RDM', ...
       '(Mean Gray Memory & Mean Pure Color)'}, 'FontSize', 10, 'FontWeight', 'bold');
hold on;
xline(2.5, 'w-', 'LineWidth', 2.0); yline(2.5, 'w-', 'LineWidth', 2.0);
hold off;

% --- 右图: 3D 空间 MDS 降维 ---
subplot(1, 3, 3);
[Y_3d, eigvals] = cmdscale(rdm_10_avg, 3);
var_exp = 100 * eigvals(1:3) / sum(abs(eigvals(eigvals > 0)));

col_red = [0.85, 0.20, 0.20];
col_grn = [0.20, 0.70, 0.25];
colors = {
    [0.92, 0.40, 0.35], [0.85, 0.15, 0.15], ... % Straw, Water
    [0.45, 0.80, 0.40], [0.15, 0.65, 0.30], ... % Kiwi, Cabb
    col_red, col_red, col_red, ...              % Red S1, S2, S3
    col_grn, col_grn, col_grn                   % Grn S1, S2, S3
};
markers = {'o', 'o', 'o', 'o', 's', 'd', '^', 's', 'd', '^'};

hold on; grid on; box on;
z_base = min(Y_3d(:,3)) - 0.10;
for i = 1:10
    plot3([Y_3d(i,1), Y_3d(i,1)], [Y_3d(i,2), Y_3d(i,2)], [z_base, Y_3d(i,3)], ':', ...
        'Color', [0.75, 0.75, 0.75], 'LineWidth', 0.8, 'HandleVisibility', 'off');
    scatter3(Y_3d(i,1), Y_3d(i,2), Y_3d(i,3), 110, markers{i}, 'filled', ...
        'MarkerFaceColor', colors{i}, 'MarkerEdgeColor', [0.2, 0.2, 0.2], 'LineWidth', 1.0);
    text(Y_3d(i,1)+0.015, Y_3d(i,2)+0.015, Y_3d(i,3)+0.015, cond_names_10{i}, ...
        'FontSize', 7.5, 'FontWeight', 'bold', 'Color', colors{i}*0.8, 'Interpreter', 'none');
end
view(38, 22);
xlabel(sprintf('Dim 1 (%.1f%%)', var_exp(1)), 'FontSize', 9, 'FontWeight', 'bold');
ylabel(sprintf('Dim 2 (%.1f%%)', var_exp(2)), 'FontSize', 9, 'FontWeight', 'bold');
zlabel(sprintf('Dim 3 (%.1f%%)', var_exp(3)), 'FontSize', 9, 'FontWeight', 'bold');
title(sprintf('C. 3D Representational Space (%s-%s)', sub_id, elec), 'FontSize', 10, 'FontWeight', 'bold');
hold off;

fig1_path = fullfile(out_fig_dir, sprintf('%s_%s_rsa_window_avg.fig', sub_id, elec));
png1_path = fullfile(out_fig_dir, sprintf('%s_%s_rsa_window_avg.png', sub_id, elec));
savefig(h_fig1, fig1_path);
exportgraphics(h_fig1, png1_path, 'Resolution', 300);
close(h_fig1);

% =========================================================================
%% 图 2: 随时间的变化图 (时间动态演化)
% =========================================================================
h_fig2 = figure('Units', 'pixels', 'Position', [50, 100, 1600, 480], 'Color', 'w');

% 提取 4 类别的核心时程
d_phys_time = squeeze(rdms_4_time(3, 4, :))'; % Red Pure vs Green Pure
d_memo_time = squeeze(rdms_4_time(1, 2, :))'; % Red Gray vs Green Gray
d_same_time = (squeeze(rdms_4_time(1, 3, :))' + squeeze(rdms_4_time(2, 4, :))') / 2; % Same Color (Cross-task)
d_diff_time = (squeeze(rdms_4_time(1, 4, :))' + squeeze(rdms_4_time(2, 3, :))') / 2; % Diff Color (Cross-task)
color_effect = d_diff_time - d_same_time;

% 平滑时程
smooth_pts = 3;
d_phys_s = smoothdata(d_phys_time, 'gaussian', smooth_pts);
d_memo_s = smoothdata(d_memo_time, 'gaussian', smooth_pts);
d_same_s = smoothdata(d_same_time, 'gaussian', smooth_pts);
d_diff_s = smoothdata(d_diff_time, 'gaussian', smooth_pts);
effect_s = smoothdata(color_effect, 'gaussian', smooth_pts);

% --- 左图: 10 类别细粒度成对距离时程 ---
subplot(1, 3, 1);
hold on;
xline(0, 'k:', 'LineWidth', 1.0, 'HandleVisibility', 'off');
% 水果间同记忆色与异记忆色
d_straw_water = smoothdata(squeeze(rdms_10_time(1, 2, :))', 'gaussian', smooth_pts);
d_kiwi_cabb   = smoothdata(squeeze(rdms_10_time(3, 4, :))', 'gaussian', smooth_pts);
d_straw_kiwi  = smoothdata(squeeze(rdms_10_time(1, 3, :))', 'gaussian', smooth_pts);
d_water_cabb  = smoothdata(squeeze(rdms_10_time(2, 4, :))', 'gaussian', smooth_pts);

plot(t_centers, d_straw_water, 'Color', [0.85, 0.20, 0.20], 'LineWidth', 1.6, 'DisplayName', 'Strawberry vs Watermelon (Red Memo)');
plot(t_centers, d_kiwi_cabb,   'Color', [0.20, 0.70, 0.25], 'LineWidth', 1.6, 'DisplayName', 'Kiwi vs Cabbage (Green Memo)');
plot(t_centers, d_straw_kiwi,  'Color', [0.45, 0.45, 0.85], 'LineWidth', 1.6, 'DisplayName', 'Strawberry vs Kiwi (Opposite Memo)');
plot(t_centers, d_water_cabb,  'Color', [0.70, 0.40, 0.80], 'LineWidth', 1.6, 'DisplayName', 'Watermelon vs Cabbage (Opposite Memo)');
hold off;
box off; grid off; set(gca, 'TickDir', 'out', 'FontSize', 9, 'LineWidth', 1.0);
xlim([-200, 800]); ylim([0, 1.0]);
xlabel('Time (ms)', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Correlation Distance (1 - r)', 'FontSize', 10, 'FontWeight', 'bold');
title({'A. 10-Condition Pairwise Dynamics', sprintf('(%s-%s)', sub_id, elec)}, 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 7, 'Box', 'off');

% --- 中间图: 4 类别大类距离演化 (相同记忆平均, 相同纯色平均) ---
subplot(1, 3, 2);
hold on;
xline(0, 'k:', 'LineWidth', 1.0, 'HandleVisibility', 'off');
plot(t_centers, d_phys_s, 'Color', [0.10, 0.55, 0.85], 'LineWidth', 2.0, 'DisplayName', 'Physical Color: Red vs Green Patch');
plot(t_centers, d_memo_s, 'Color', [0.20, 0.75, 0.40], 'LineWidth', 2.0, 'DisplayName', 'Memory Color: Red vs Green Gray');
plot(t_centers, d_same_s, 'Color', [0.60, 0.20, 0.70], 'LineWidth', 2.2, 'DisplayName', 'Cross-Task Same: Gray Fruit → Matched Patch');
plot(t_centers, d_diff_s, 'Color', [0.90, 0.45, 0.10], 'LineWidth', 2.2, 'DisplayName', 'Cross-Task Diff: Gray Fruit → Opposite Patch');
hold off;
box off; grid off; set(gca, 'TickDir', 'out', 'FontSize', 9, 'LineWidth', 1.0);
xlim([-200, 800]); ylim([0, 0.8]);
xlabel('Time (ms)', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Correlation Distance (1 - r)', 'FontSize', 10, 'FontWeight', 'bold');
title({'B. Collapsed 4-Category Dynamics', '(Physical, Memory, Cross-Task)'}, 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 7.5, 'Box', 'off');

% --- 右图: 跨任务颜色表征提取效应时程 (Diff - Same) ---
subplot(1, 3, 3);
hold on;
yline(0, 'k--', 'LineWidth', 1.0, 'Alpha', 0.6, 'HandleVisibility', 'off');
xline(0, 'k:', 'LineWidth', 1.0, 'Alpha', 0.6, 'HandleVisibility', 'off');

% 正效应阴影填充 (记忆色对齐)
pos_effect = effect_s; pos_effect(pos_effect < 0) = 0;
neg_effect = effect_s; neg_effect(neg_effect > 0) = 0;
fill([t_centers, fliplr(t_centers)], [pos_effect, zeros(1, n_win)], [0.85, 0.35, 0.10], ...
    'FaceAlpha', 0.25, 'EdgeColor', 'none', 'DisplayName', 'Color Replay (Diff > Same)');
fill([t_centers, fliplr(t_centers)], [neg_effect, zeros(1, n_win)], [0.5, 0.5, 0.5], ...
    'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');

plot(t_centers, effect_s, 'Color', [0.85, 0.30, 0.05], 'LineWidth', 2.5, 'DisplayName', 'Net Replay Effect (Diff - Same)');
hold off;
box off; grid off; set(gca, 'TickDir', 'out', 'FontSize', 9, 'LineWidth', 1.0);
xlim([-200, 800]);
xlabel('Time (ms)', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Distance Difference (Diff - Same)', 'FontSize', 10, 'FontWeight', 'bold');
title({'C. Cross-Task Memory Color Replay Effect', 'Positive = Memory Pattern Matches Physical Color'}, 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 8, 'Box', 'off');

fig2_path = fullfile(out_fig_dir, sprintf('%s_%s_rsa_timecourse.fig', sub_id, elec));
png2_path = fullfile(out_fig_dir, sprintf('%s_%s_rsa_timecourse.png', sub_id, elec));
savefig(h_fig2, fig2_path);
exportgraphics(h_fig2, png2_path, 'Resolution', 300);
close(h_fig2);

fprintf('Generated figures for %s-%s:\n', sub_id, elec);
fprintf('  - Fig 1 (Avg):        %s\n', png1_path);
fprintf('  - Fig 2 (Timecourse): %s\n', png2_path);
