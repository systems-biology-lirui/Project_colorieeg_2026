%% C11_cross_task_rsa_3d_0825.m
% =========================================================================
% 脚本名称: C11_cross_task_rsa_3d_0825.m
% 功能:
%   1. 【数据提取】提取 Task 2 灰色水果 (4类) 与 Task 3 纯色色块 (红绿各3形状, 共6类)
%      - Task 2: strawberry, watermelon (红记忆色), kiwi, cabbage (绿记忆色)
%      - Task 3: red shape 1/2/3, green shape 1/2/3
%   2. 【同类别试次平均】在同类别下对多频段/时程信号进行 trial average
%   3. 【RSA 表征相似度分析】构建 10x10 表征相异度矩阵 (RDM, Correlation Distance)
%   4. 【3D 空间降维投影】经典多维尺度分析 (MDS) 降维至 3D 空间
%   5. 【交互式 3D 可视化并保存 .fig 图】
%      - 保存为 MATLAB 原生 .fig 图 (支持鼠标自由三维旋转、缩放、数据游标拾取)
%      - 同步导出 300 DPI 高清 .png 预览图
% =========================================================================

clear; clc; close all;

%% 1. 参数与路径配置 (平铺直观，可控可调)
cfg = struct();
cfg.root_dir     = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825';
cfg.data_root    = fullfile(cfg.root_dir, 'process_data_new');
cfg.task_info    = fullfile(cfg.root_dir, 'task_info');
cfg.res_root     = fullfile(cfg.root_dir, 'result');

% 输出路径
cfg.out_fig_dir  = fullfile(cfg.res_root, 'figures', 'rsa_3d');
cfg.out_tab_dir  = fullfile(cfg.res_root, 'tables', 'rsa_3d');
if ~exist(cfg.out_fig_dir, 'dir'), mkdir(cfg.out_fig_dir); end
if ~exist(cfg.out_tab_dir, 'dir'), mkdir(cfg.out_tab_dir); end

% 分析时间窗与频段
cfg.t_win        = [100, 600];                      % 认知分析时间窗 [100, 600] ms
cfg.bands        = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.n_bands      = numel(cfg.bands);

% 10 大类别定义
cond_names = {
    'Gray Strawberry', 'Gray Watermelon', 'Gray Kiwi', 'Gray Cabbage', ...
    'Red Shape 1', 'Red Shape 2', 'Red Shape 3', ...
    'Green Shape 1', 'Green Shape 2', 'Green Shape 3'
};
n_cond = numel(cond_names);

% 读取 C04 157 个同向显著电极
c04_mat = fullfile(cfg.res_root, 'tables', 'color_effects_summary.mat');
c04_d   = load(c04_mat);
tbl     = c04_d.all_tbl;
concord_mask = (tbl.is_significant == 1) & ...
    (strcmp(tbl.concordance_type, 'Concordant_Positive') | ...
     strcmp(tbl.concordance_type, 'Concordant_Negative'));
c04_concord = tbl(concord_mask, :);

elec_keys = strcat(c04_concord.subject, '_', c04_concord.channel);
[~, u_idx] = unique(elec_keys, 'stable');
all_subs  = c04_concord.subject(u_idx);
all_elecs = c04_concord.channel(u_idx);
unique_subs = unique(all_subs, 'stable');

fprintf('========================================================================\n');
fprintf('>>> 启动跨任务 10 类别 RSA 分析 (Task 2 灰色水果 + Task 3 纯色块)\n');
fprintf('>>> 覆盖被试: %d 名 | 目标电极: %d 个 | 时间窗: [%d, %d] ms\n', ...
    numel(unique_subs), numel(all_subs), cfg.t_win(1), cfg.t_win(2));
fprintf('========================================================================\n');

%% 2. 逐被试提取 10 类别的试次平均特征向量并计算各被试 RDM
sub_rdms = zeros(n_cond, n_cond, numel(unique_subs));
sub_vectors = cell(numel(unique_subs), 1);

for s_i = 1:numel(unique_subs)
    sub_id = unique_subs{s_i};
    sub_mask = strcmp(all_subs, sub_id);
    sub_elecs = all_elecs(sub_mask);
    n_e = numel(sub_elecs);
    
    t3_mat = fullfile(cfg.data_root, sub_id, 'task3_multiband_epoched.mat');
    t2_mat = fullfile(cfg.data_root, sub_id, 'task2_multiband_epoched.mat');
    
    t_axis = double(h5read(t3_mat, '/epoched_data/time_ms'));
    t_axis = t_axis(:)';
    ch_list3 = h5read(t3_mat, '/epoched_data/channels');
    ch_list2 = h5read(t2_mat, '/epoched_data/channels');
    if iscell(ch_list3), ch_list3 = cellfun(@(x) char(x(:)'), ch_list3, 'UniformOutput', false); end
    if iscell(ch_list2), ch_list2 = cellfun(@(x) char(x(:)'), ch_list2, 'UniformOutput', false); end
    
    [~, idx3] = ismember(sub_elecs, ch_list3);
    [~, idx2] = ismember(sub_elecs, ch_list2);
    
    d3 = load(fullfile(cfg.task_info, sub_id, 'task3_trial_info.mat'));
    ti3 = d3.trial_info;
    d2 = load(fullfile(cfg.task_info, sub_id, 'task2_trial_info.mat'));
    ti2 = d2.trial_info;
    
    t_mask = (t_axis >= cfg.t_win(1)) & (t_axis <= cfg.t_win(2));
    n_time_pts = sum(t_mask);
    
    % 构建 10 个类别的试次筛选掩码
    cond_masks = cell(n_cond, 1);
    % Task 2 gray fruits (4 类)
    cond_masks{1} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'strawberry');
    cond_masks{2} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'watermelon');
    cond_masks{3} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'kiwi');
    cond_masks{4} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'cabbage');
    % Task 3 color shapes (红绿各3形状, 共6类)
    cond_masks{5} = strcmp(ti3.color, 'red') & (ti3.pic_id == 1);
    cond_masks{6} = strcmp(ti3.color, 'red') & (ti3.pic_id == 2);
    cond_masks{7} = strcmp(ti3.color, 'red') & (ti3.pic_id == 3);
    cond_masks{8} = strcmp(ti3.color, 'green') & (ti3.pic_id == 1);
    cond_masks{9} = strcmp(ti3.color, 'green') & (ti3.pic_id == 2);
    cond_masks{10} = strcmp(ti3.color, 'green') & (ti3.pic_id == 3);
    
    % 载入多频段数据并进行同类别试次平均
    feat_mat = zeros(n_cond, n_e, cfg.n_bands, n_time_pts);
    for b = 1:cfg.n_bands
        b_name = cfg.bands{b};
        raw3 = h5read(t3_mat, ['/epoched_data/' b_name]);
        raw2 = h5read(t2_mat, ['/epoched_data/' b_name]);
        
        % Task 2 平均
        for c_idx = 1:4
            m = cond_masks{c_idx};
            dat_c = raw2(m, idx2, t_mask);
            feat_mat(c_idx, :, b, :) = squeeze(mean(dat_c, 1));
        end
        % Task 3 平均
        for c_idx = 5:10
            m = cond_masks{c_idx};
            dat_c = raw3(m, idx3, t_mask);
            feat_mat(c_idx, :, b, :) = squeeze(mean(dat_c, 1));
        end
        clear raw3 raw2;
    end
    
    % 展平成条件特征向量 [n_cond x (n_e * n_bands * n_time)]
    vec_c = reshape(feat_mat, [n_cond, n_e * cfg.n_bands * n_time_pts]);
    sub_vectors{s_i} = vec_c;
    
    % 计算被试级别 RDM (相关距离: 1 - r)
    rdm_s = 1 - corr(vec_c');
    sub_rdms(:, :, s_i) = rdm_s;
    
    fprintf('  - [%d/%d] 被试 %s: %d 个电极, 10 类别特征已提取完成\n', ...
        s_i, numel(unique_subs), sub_id, n_e);
end

%% 3. 群体平均 RDM 与 3D MDS 降维
group_rdm = mean(sub_rdms, 3);

% 保存 RDM 数据表格
save(fullfile(cfg.out_tab_dir, 'cross_task_rdm_data.mat'), 'group_rdm', 'sub_rdms', 'cond_names', 'cfg');
rdm_tbl = array2table(group_rdm, 'RowNames', cond_names, 'VariableNames', cond_names);
writetable(rdm_tbl, fullfile(cfg.out_tab_dir, 'cross_task_rdm_matrix.csv'), 'WriteRowNames', true);

% 经典多维尺度分析 (Classical MDS) 降至 3D 空间
[Y_3d, eigvals] = cmdscale(group_rdm, 3);
var_explained = 100 * eigvals(1:3) / sum(abs(eigvals(eigvals > 0)));

fprintf('\n>>> 3D MDS 降维完成:\n');
fprintf('    - Dim 1 解释率: %.1f%%\n', var_explained(1));
fprintf('    - Dim 2 解释率: %.1f%%\n', var_explained(2));
fprintf('    - Dim 3 解释率: %.1f%%\n', var_explained(3));
fprintf('    - 累计解释率: %.1f%%\n', sum(var_explained(1:3)));

%% 4. 学术级 3D 空间可视化绘图 (含三维投影辅助线与智能文本避让)
h_fig = figure('Name', 'Cross-Task 3D RSA Space', 'Units', 'pixels', ...
               'Position', [80, 80, 1100, 850], 'Color', 'w');

% 配色定义
col_red_straw  = [0.92, 0.30, 0.25]; % 珊瑚红 (草莓)
col_red_water  = [0.85, 0.10, 0.10]; % 鲜红 (西瓜)
col_red_patch  = [0.65, 0.05, 0.05]; % 浓红 (纯色块)

col_grn_kiwi   = [0.45, 0.78, 0.35]; % 浅青绿 (猕猴桃)
col_grn_cabb   = [0.15, 0.62, 0.30]; % 鲜翠绿 (卷心菜)
col_grn_patch  = [0.05, 0.42, 0.12]; % 浓墨绿 (纯色块)

colors = {
    col_red_straw;   % 1: Gray Strawberry
    col_red_water;   % 2: Gray Watermelon
    col_grn_kiwi;    % 3: Gray Kiwi
    col_grn_cabb;    % 4: Gray Cabbage
    col_red_patch;   % 5: Red Shape 1
    col_red_patch;   % 6: Red Shape 2
    col_red_patch;   % 7: Red Shape 3
    col_grn_patch;   % 8: Green Shape 1
    col_grn_patch;   % 9: Green Shape 2
    col_grn_patch    % 10: Green Shape 3
};

markers = {'o', 'o', 'o', 'o', 's', 'd', '^', 's', 'd', '^'};
marker_sizes = [150, 150, 150, 150, 170, 170, 170, 170, 170, 170];

hold on;
grid on;
box on;

% 设置坐标轴与底面参考深度
z_base = min(Y_3d(:, 3)) - 0.12;

% 1. 绘制三维空间垂落线 (Drop-lines 到底面)，极大增强 3D 深度感知
for i = 1:n_cond
    plot3([Y_3d(i, 1), Y_3d(i, 1)], [Y_3d(i, 2), Y_3d(i, 2)], [z_base, Y_3d(i, 3)], ...
        ':', 'Color', [0.75, 0.75, 0.75], 'LineWidth', 1.0, 'HandleVisibility', 'off');
    % 底面轻微阴影点
    scatter3(Y_3d(i, 1), Y_3d(i, 2), z_base, 35, [0.85, 0.85, 0.85], 'filled', ...
        'MarkerEdgeColor', 'none', 'HandleVisibility', 'off');
end

% 2. 绘制原点中心十字参考线
plot3([0, 0], [0, 0], [z_base, max(Y_3d(:,3))*1.15], 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');
plot3([-1, 1]*max(abs(Y_3d(:,1)))*1.15, [0, 0], [0, 0], 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');
plot3([0, 0], [-1, 1]*max(abs(Y_3d(:,2)))*1.15, [0, 0], 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');

% 3. 逐个类别绘制 3D 实体散点
h_sc = gobjects(n_cond, 1);
for i = 1:n_cond
    h_sc(i) = scatter3(Y_3d(i, 1), Y_3d(i, 2), Y_3d(i, 3), marker_sizes(i), ...
        'Marker', markers{i}, 'MarkerFaceColor', colors{i}, 'MarkerEdgeColor', [0.15, 0.15, 0.15], ...
        'LineWidth', 1.3, 'DisplayName', cond_names{i});
end

% 4. 智能文本微调避让偏移向量 (避免重叠碰撞)
text_offsets = [
     0.030, -0.035, -0.045;  % 1: Gray Strawberry (右下前)
    -0.090, -0.055, -0.040;  % 2: Gray Watermelon (左下前)
     0.030,  0.025,  0.035;  % 3: Gray Kiwi (右上后)
    -0.105,  0.030,  0.035;  % 4: Gray Cabbage (左上后)
     0.030,  0.015,  0.010;  % 5: Red Shape 1
    -0.040, -0.030, -0.035;  % 6: Red Shape 2
     0.025,  0.010,  0.040;  % 7: Red Shape 3
     0.020, -0.035, -0.025;  % 8: Green Shape 1
     0.025,  0.020,  0.035;  % 9: Green Shape 2
     0.025, -0.010,  0.030   % 10: Green Shape 3
];

for i = 1:n_cond
    t_pos = Y_3d(i, :) + text_offsets(i, :);
    text(t_pos(1), t_pos(2), t_pos(3), cond_names{i}, ...
        'FontSize', 10, 'FontWeight', 'bold', 'Color', colors{i} * 0.75, ...
        'Interpreter', 'none');
end

% 5. 绘制红绿两大家族的聚类几何引线 (向心弱虚线)
red_centroid = mean(Y_3d([1, 2, 5, 6, 7], :), 1);
grn_centroid = mean(Y_3d([3, 4, 8, 9, 10], :), 1);

for idx = [1, 2, 5, 6, 7]
    plot3([Y_3d(idx, 1), red_centroid(1)], [Y_3d(idx, 2), red_centroid(2)], [Y_3d(idx, 3), red_centroid(3)], ...
        '--', 'Color', [0.85, 0.30, 0.30, 0.35], 'LineWidth', 1.0, 'HandleVisibility', 'off');
end
for idx = [3, 4, 8, 9, 10]
    plot3([Y_3d(idx, 1), grn_centroid(1)], [Y_3d(idx, 2), grn_centroid(2)], [Y_3d(idx, 3), grn_centroid(3)], ...
        '--', 'Color', [0.25, 0.70, 0.25, 0.35], 'LineWidth', 1.0, 'HandleVisibility', 'off');
end

hold off;

% 6. 视角美化与学术标签
view(38, 22);
set(gca, 'FontSize', 11, 'LineWidth', 1.0, 'TickDir', 'out');
zlim([z_base - 0.05, max(Y_3d(:,3)) + 0.15]);
xlabel(sprintf('MDS Dimension 1 (%.1f%%)', var_explained(1)), 'FontSize', 12, 'FontWeight', 'bold');
ylabel(sprintf('MDS Dimension 2 (%.1f%%)', var_explained(2)), 'FontSize', 12, 'FontWeight', 'bold');
zlabel(sprintf('MDS Dimension 3 (%.1f%%)', var_explained(3)), 'FontSize', 12, 'FontWeight', 'bold');

title({'3D Representational Similarity Analysis (RSA)', ...
       'Task 2 Gray Fruits vs. Task 3 Pure Color Shapes'}, ...
       'FontSize', 14, 'FontWeight', 'bold');

legend(h_sc, cond_names, 'Location', 'eastoutside', 'FontSize', 9.5, 'Box', 'off');

% 保存用户特别指定的 .fig 文件与伴生 .png
out_fig_file = fullfile(cfg.out_fig_dir, 'cross_task_rsa_3d.fig');
out_png_file = fullfile(cfg.out_fig_dir, 'cross_task_rsa_3d.png');
savefig(h_fig, out_fig_file);
exportgraphics(h_fig, out_png_file, 'Resolution', 300);

fprintf('\n>>> 3D 核心图谱导出成功:\n');
fprintf('    - [FIG 原生三维文件]: %s\n', out_fig_file);
fprintf('    - [PNG 高清预览文件]: %s\n', out_png_file);

%% 5. 伴生输出: 10x10 RDM 表征相异度矩阵热力图
h_rdm = figure('Name', 'Cross-Task 10x10 RDM Matrix', 'Units', 'pixels', ...
               'Position', [150, 150, 780, 680], 'Color', 'w');
imagesc(group_rdm);
colormap('parula');
cb = colorbar;
ylabel(cb, 'Dissimilarity (1 - Pearson r)', 'FontSize', 10, 'FontWeight', 'bold');
caxis([0, max(group_rdm(:))]);
axis square;
set(gca, 'XTick', 1:10, 'XTickLabel', cond_names, 'XTickLabelRotation', 45, ...
         'YTick', 1:10, 'YTickLabel', cond_names, 'FontSize', 9.5, 'TickDir', 'out');
title({'Cross-Task Representational Dissimilarity Matrix (RDM)', ...
       'Task 2 (4 Gray Fruits) vs. Task 3 (6 Pure Color Shapes)'}, ...
       'FontSize', 12, 'FontWeight', 'bold');

hold on;
% 任务边界分界实线
xline(4.5, 'w-', 'LineWidth', 2.2);
yline(4.5, 'w-', 'LineWidth', 2.2);
% 任务内部红绿分界虚线
xline(2.5, 'w:', 'LineWidth', 1.2);
yline(2.5, 'w:', 'LineWidth', 1.2);
xline(7.5, 'w--', 'LineWidth', 1.5);
yline(7.5, 'w--', 'LineWidth', 1.5);
hold off;

out_rdm_fig = fullfile(cfg.out_fig_dir, 'cross_task_rdm_matrix.fig');
out_rdm_png = fullfile(cfg.out_fig_dir, 'cross_task_rdm_matrix.png');
savefig(h_rdm, out_rdm_fig);
exportgraphics(h_rdm, out_rdm_png, 'Resolution', 300);

fprintf('>>> RDM 矩阵图谱导出成功:\n');
fprintf('    - [RDM FIG]: %s\n', out_rdm_fig);
fprintf('    - [RDM PNG]: %s\n', out_rdm_png);

close(h_fig);
close(h_rdm);
fprintf('\n========================================================================\n');
fprintf('>>> 全部 RSA 3D 空间建模与出图已圆满完成!\n');
fprintf('========================================================================\n');
