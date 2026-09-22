%% test_prototype_3d_trajectory.m
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

% 提取单通道数据
X3 = zeros(height(ti3), n_bands, numel(t_axis));
X2 = zeros(height(ti2), n_bands, numel(t_axis));
for b = 1:n_bands
    raw3 = h5read(t3_mat, ['/epoched_data/' bands{b}]);
    X3(:, b, :) = raw3(:, e_idx3, :);
    raw2 = h5read(t2_mat, ['/epoched_data/' bands{b}]);
    X2(:, b, :) = raw2(:, e_idx2, :);
end

% 滑动时间窗设置 (50ms 窗长, 20ms 步长, [-200, 800] ms)
win_len = 50; win_step = 20;
t_starts = -200 : win_step : (800 - win_len);
n_win = numel(t_starts);
t_centers = t_starts + win_len / 2;

% 构建特征张量: 10 条件在各时间窗的 6 频段特征
feat_10 = zeros(10, n_win, n_bands);
for w = 1:n_win
    w_t1 = t_starts(w); w_t2 = w_t1 + win_len;
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

% 4 类别特征
feat_4 = zeros(4, n_win, n_bands);
for w = 1:n_win
    w_t1 = t_starts(w); w_t2 = w_t1 + win_len;
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

% 基于 10 条件全时段数据进行统一 PCA 构建 3D 状态空间
mat_10_flat = reshape(permute(feat_10, [2, 1, 3]), [n_win * 10, n_bands]);
mu_feat  = mean(mat_10_flat, 1);
std_feat = std(mat_10_flat, 0, 1);
std_feat(std_feat < 1e-6) = 1;
mat_10_norm = (mat_10_flat - mu_feat) ./ std_feat;

[coeff, ~, ~, ~, explained] = pca(mat_10_norm);
var_exp = explained(1:3);

% 计算 10 类别 3D 轨迹 (平滑处理)
traj_10 = zeros(10, n_win, 3);
smooth_pts = 3;
for c = 1:10
    c_feat = squeeze(feat_10(c, :, :));
    c_norm = (c_feat - mu_feat) ./ std_feat;
    pts_3d = c_norm * coeff(:, 1:3);
    for d = 1:3
        traj_10(c, :, d) = smoothdata(pts_3d(:, d), 'gaussian', smooth_pts);
    end
end

% 计算 4 类别 3D 轨迹
traj_4 = zeros(4, n_win, 3);
for c = 1:4
    c_feat = squeeze(feat_4(c, :, :));
    c_norm = (c_feat - mu_feat) ./ std_feat;
    pts_3d = c_norm * coeff(:, 1:3);
    for d = 1:3
        traj_4(c, :, d) = smoothdata(pts_3d(:, d), 'gaussian', smooth_pts);
    end
end

% 配色定义
col_red = [0.85, 0.15, 0.15];
col_grn = [0.15, 0.70, 0.25];
colors_10 = {
    [0.92, 0.40, 0.35], [0.85, 0.20, 0.20], ... % Straw, Water (Red Memo)
    [0.45, 0.80, 0.40], [0.15, 0.65, 0.30], ... % Kiwi, Cabb (Grn Memo)
    col_red, [0.95, 0.30, 0.10], [0.75, 0.10, 0.10], ... % Red Shapes 1, 2, 3
    col_grn, [0.10, 0.60, 0.35], [0.10, 0.50, 0.20]  ... % Grn Shapes 1, 2, 3
};

% 4 类别配色
colors_4 = {
    [0.90, 0.35, 0.30], ... % Red Memory (Gray Fruit)
    [0.30, 0.75, 0.35], ... % Green Memory (Gray Fruit)
    [0.85, 0.10, 0.10], ... % Red Pure (Patch)
    [0.10, 0.65, 0.20]      % Green Pure (Patch)
};

% 标记关键时间点索引
idx_start = 1;                         % -200 ms (起点)
idx_0ms   = find(t_centers >= 0, 1);   % 0 ms (刺激呈现)
idx_200ms = find(t_centers >= 200, 1); % 200 ms
idx_400ms = find(t_centers >= 400, 1); % 400 ms
idx_end   = n_win;                     % 800 ms (终点)

%% =========================================================================
% 绘制图 2: 3D 神经状态空间时间轨迹图 (1x3 规范学术全景图)
% =========================================================================
h_fig2 = figure('Units', 'pixels', 'Position', [50, 100, 1650, 520], 'Color', 'w');

% --- 左图: 10 类别 3D 状态空间时间轨迹 ---
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

% --- 中间图: 4 类别 3D 状态空间时间轨迹 (相同记忆平均, 相同纯色平均) ---
subplot(1, 3, 2);
hold on; grid on; box on;
z_min = min(traj_4(:));
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

% 标注时间节点文字
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

% --- 右图: 4 类别 2D 主平面投影轨迹 (PC1 vs PC2 俯视图) ---
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

fig2_path = fullfile(out_fig_dir, sprintf('%s_%s_rsa_3d_trajectory.fig', sub_id, elec));
png2_path = fullfile(out_fig_dir, sprintf('%s_%s_rsa_3d_trajectory.png', sub_id, elec));
set(h_fig2, 'Visible', 'on');
savefig(h_fig2, fig2_path);
exportgraphics(h_fig2, png2_path, 'Resolution', 300);
close(h_fig2);

fprintf('\n[+] Generated 3D Trajectory Figure:\n    - FIG: %s\n    - PNG: %s\n', fig2_path, png2_path);
