%% test_cross_task_rsa.m
% =========================================================================
% 测试: Task 2 灰色水果 (4种) 与 Task 3 纯色色块 (红绿各3形状, 共6种)
% 同类别试次平均 -> 计算 10x10 RSA RDM -> 3D 空间 MDS 降维投影与 .fig 保存
% =========================================================================
clear; clc; close all;

%% 1. 参数设置
cfg = struct();
cfg.root_dir     = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825';
cfg.data_root    = fullfile(cfg.root_dir, 'process_data_new');
cfg.task_info    = fullfile(cfg.root_dir, 'task_info');
cfg.res_root     = fullfile(cfg.root_dir, 'result');
cfg.out_fig_dir  = fullfile(cfg.res_root, 'figures', 'rsa_3d');
if ~exist(cfg.out_fig_dir, 'dir'), mkdir(cfg.out_fig_dir); end

% 分析时间窗 (ms)
cfg.t_win        = [100, 600];                      % 核心认知加工时间窗 [100, 600] ms
cfg.bands        = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.n_bands      = numel(cfg.bands);

% 10 大类别定义
% 1-4: Task 2 灰色水果
% 5-7: Task 3 红色色块 (3种形状)
% 8-10: Task 3 绿色色块 (3种形状)
cond_names = {
    'Gray Strawberry', 'Gray Watermelon', 'Gray Kiwi', 'Gray Cabbage', ...
    'Red Shape 1', 'Red Shape 2', 'Red Shape 3', ...
    'Green Shape 1', 'Green Shape 2', 'Green Shape 3'
};
n_cond = numel(cond_names);

% 读取 C04 157 个同向电极
c04_mat = fullfile(cfg.res_root, 'tables', 'color_effects_summary.mat');
c04_d = load(c04_mat);
tbl = c04_d.all_tbl;
concord_mask = (tbl.is_significant == 1) & ...
    (strcmp(tbl.concordance_type, 'Concordant_Positive') | ...
     strcmp(tbl.concordance_type, 'Concordant_Negative'));
c04_concord = tbl(concord_mask, :);
elec_keys = strcat(c04_concord.subject, '_', c04_concord.channel);
[~, u_idx] = unique(elec_keys, 'stable');
all_subs  = c04_concord.subject(u_idx);
all_elecs = c04_concord.channel(u_idx);
unique_subs = unique(all_subs, 'stable');

fprintf('>>> 启动跨任务 10 类别 RSA 测试 (被试数: %d, 同向电极数: %d)\n', ...
    numel(unique_subs), numel(all_subs));

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
    
    % 构建 10 个类别的试次掩码
    cond_masks = cell(n_cond, 1);
    % Task 2 gray fruits
    cond_masks{1} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'strawberry');
    cond_masks{2} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'watermelon');
    cond_masks{3} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'kiwi');
    cond_masks{4} = strcmp(ti2.state, 'gray') & strcmp(ti2.fruit, 'cabbage');
    % Task 3 color shapes
    cond_masks{5} = strcmp(ti3.color, 'red') & (ti3.pic_id == 1);
    cond_masks{6} = strcmp(ti3.color, 'red') & (ti3.pic_id == 2);
    cond_masks{7} = strcmp(ti3.color, 'red') & (ti3.pic_id == 3);
    cond_masks{8} = strcmp(ti3.color, 'green') & (ti3.pic_id == 1);
    cond_masks{9} = strcmp(ti3.color, 'green') & (ti3.pic_id == 2);
    cond_masks{10} = strcmp(ti3.color, 'green') & (ti3.pic_id == 3);
    
    % 载入 6 频段数据
    feat_mat = zeros(n_cond, n_e, cfg.n_bands, n_time_pts);
    for b = 1:cfg.n_bands
        b_name = cfg.bands{b};
        raw3 = h5read(t3_mat, ['/epoched_data/' b_name]);
        raw2 = h5read(t2_mat, ['/epoched_data/' b_name]);
        
        % Task 2 (1-4)
        for c_idx = 1:4
            m = cond_masks{c_idx};
            dat_c = raw2(m, idx2, t_mask); % [trials x elecs x time]
            feat_mat(c_idx, :, b, :) = squeeze(mean(dat_c, 1));
        end
        % Task 3 (5-10)
        for c_idx = 5:10
            m = cond_masks{c_idx};
            dat_c = raw3(m, idx3, t_mask); % [trials x elecs x time]
            feat_mat(c_idx, :, b, :) = squeeze(mean(dat_c, 1));
        end
        clear raw3 raw2;
    end
    
    % 展平成条件特征向量 [n_cond x (n_e * n_bands * n_time)]
    vec_c = reshape(feat_mat, [n_cond, n_e * cfg.n_bands * n_time_pts]);
    sub_vectors{s_i} = vec_c;
    
    % 计算被试 RDM (1 - Pearson correlation)
    rdm_s = 1 - corr(vec_c');
    sub_rdms(:, :, s_i) = rdm_s;
    
    fprintf('  - 被试 %s (%d 电极): 特征维度 = %d\n', sub_id, n_e, size(vec_c, 2));
end

%% 3. 群体平均 RDM 与全电极池化 RDM
group_rdm = mean(sub_rdms, 3);

% 池化拼接所有被试特征向量
pooled_vec = cat(2, sub_vectors{:}); % [10 x total_features]
pooled_rdm = 1 - corr(pooled_vec');

fprintf('\n>>> RDM 计算完成! 检查群体平均 RDM 对称性与距离范围: [%.4f, %.4f]\n', ...
    min(group_rdm(:)), max(group_rdm(:)));

%% 4. 经典多维尺度分析 (Classical MDS) 降维至 3D 空间
% 对群体平均 RDM 做 MDS
[Y_3d, eigvals] = cmdscale(group_rdm, 3);
var_explained = 100 * eigvals(1:3) / sum(abs(eigvals(eigvals > 0)));
fprintf('>>> 3D MDS 前三维度方差解释率: Dim1=%.1f%%, Dim2=%.1f%%, Dim3=%.1f%% (累计: %.1f%%)\n', ...
    var_explained(1), var_explained(2), var_explained(3), sum(var_explained(1:3)));

%% 5. 绘制精美 3D 学术图谱并保存为 .fig 与 .png
h_fig = figure('Units', 'pixels', 'Position', [100, 100, 1000, 800], 'Color', 'w');

% 配色与 Marker 定义
% 红色系:
% 1: Strawberry (珊瑚红, 浅红实心圆)
% 2: Watermelon (鲜红, 深红实心圆)
% 5: Red Shape 1 (正方, 深红实心方块)
% 6: Red Shape 2 (菱形, 深红实心菱形)
% 7: Red Shape 3 (三角, 深红实心三角)
% 绿色系:
% 3: Kiwi (浅青绿实心圆)
% 4: Cabbage (深翠绿实心圆)
% 8: Green Shape 1 (正方, 深绿实心方块)
% 9: Green Shape 2 (菱形, 深绿实心菱形)
% 10: Green Shape 3 (三角, 深绿实心三角)

col_red_fruit  = [0.90, 0.40, 0.35]; % 浅珊瑚红
col_red_water  = [0.85, 0.15, 0.15]; % 鲜红
col_red_patch  = [0.70, 0.05, 0.05]; % 浓红

col_grn_kiwi   = [0.45, 0.80, 0.40]; % 浅青绿
col_grn_cabb   = [0.15, 0.65, 0.35]; % 鲜绿
col_grn_patch  = [0.05, 0.45, 0.15]; % 浓墨绿

colors = {
    col_red_fruit;   % 1: Strawberry
    col_red_water;   % 2: Watermelon
    col_grn_kiwi;    % 3: Kiwi
    col_grn_cabb;    % 4: Cabbage
    col_red_patch;   % 5: Red S1
    col_red_patch;   % 6: Red S2
    col_red_patch;   % 7: Red S3
    col_grn_patch;   % 8: Green S1
    col_grn_patch;   % 9: Green S2
    col_grn_patch    % 10: Green S3
};

markers = {'o', 'o', 'o', 'o', 's', 'd', '^', 's', 'd', '^'};
marker_sizes = [130, 130, 130, 130, 150, 150, 150, 150, 150, 150];

hold on;
grid on;
box on;

% 绘制原点辅助虚线
plot3([0, 0], [0, 0], [-1, 1]*max(abs(Y_3d(:,3)))*1.2, 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');
plot3([-1, 1]*max(abs(Y_3d(:,1)))*1.2, [0, 0], [0, 0], 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');
plot3([0, 0], [-1, 1]*max(abs(Y_3d(:,2)))*1.2, [0, 0], 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');

% 逐个类别绘制 3D 散点与文字
h_sc = gobjects(n_cond, 1);
for i = 1:n_cond
    h_sc(i) = scatter3(Y_3d(i, 1), Y_3d(i, 2), Y_3d(i, 3), marker_sizes(i), ...
        'Marker', markers{i}, 'MarkerFaceColor', colors{i}, 'MarkerEdgeColor', [0.2, 0.2, 0.2], ...
        'LineWidth', 1.2, 'DisplayName', cond_names{i});
    
    % 文字标注
    text(Y_3d(i, 1) + 0.015, Y_3d(i, 2) + 0.015, Y_3d(i, 3) + 0.015, ...
        sprintf(' %s', cond_names{i}), 'FontSize', 10, 'FontWeight', 'bold', ...
        'Color', colors{i} * 0.8, 'Interpreter', 'none');
end

% 绘制红/绿聚类连线 (水果与对应色块之间的几何关系)
% 红色中心与绿色中心
red_mean = mean(Y_3d([1, 2, 5, 6, 7], :), 1);
grn_mean = mean(Y_3d([3, 4, 8, 9, 10], :), 1);

% 绘制红色组内部弱虚线
for idx = [1, 2, 5, 6, 7]
    plot3([Y_3d(idx,1), red_mean(1)], [Y_3d(idx,2), red_mean(2)], [Y_3d(idx,3), red_mean(3)], ...
        '--', 'Color', [0.85, 0.35, 0.35, 0.4], 'LineWidth', 1.0, 'HandleVisibility', 'off');
end
% 绘制绿色组内部弱虚线
for idx = [3, 4, 8, 9, 10]
    plot3([Y_3d(idx,1), grn_mean(1)], [Y_3d(idx,2), grn_mean(2)], [Y_3d(idx,3), grn_mean(3)], ...
        '--', 'Color', [0.35, 0.70, 0.35, 0.4], 'LineWidth', 1.0, 'HandleVisibility', 'off');
end

hold off;

% 美化学术视角
view(40, 24);
set(gca, 'FontSize', 11, 'LineWidth', 1.0, 'TickDir', 'out');
xlabel(sprintf('MDS Dimension 1 (%.1f%%)', var_explained(1)), 'FontSize', 12, 'FontWeight', 'bold');
ylabel(sprintf('MDS Dimension 2 (%.1f%%)', var_explained(2)), 'FontSize', 12, 'FontWeight', 'bold');
zlabel(sprintf('MDS Dimension 3 (%.1f%%)', var_explained(3)), 'FontSize', 12, 'FontWeight', 'bold');
title({'3D Representational Similarity Analysis (RSA)', ...
       'Task 2 Gray Fruits vs. Task 3 Pure Color Shapes'}, ...
       'FontSize', 14, 'FontWeight', 'bold');

% 格式良好的图例
legend(h_sc, cond_names, 'Location', 'eastoutside', 'FontSize', 9, 'Box', 'off');

% 保存 .fig 与 .png
out_fig = fullfile(cfg.out_fig_dir, 'cross_task_rsa_3d.fig');
out_png = fullfile(cfg.out_fig_dir, 'cross_task_rsa_3d.png');
savefig(h_fig, out_fig);
exportgraphics(h_fig, out_png, 'Resolution', 300);
fprintf('\n>>> 3D 图像已成功保存!\n  - FIG: %s\n  - PNG: %s\n', out_fig, out_png);

%% 6. 同时绘制 10x10 RDM 辅助热力图
h_rdm = figure('Units', 'pixels', 'Position', [150, 150, 750, 650], 'Color', 'w');
imagesc(group_rdm);
colormap('parula');
colorbar;
axis square;
set(gca, 'XTick', 1:10, 'XTickLabel', cond_names, 'XTickLabelRotation', 45, ...
         'YTick', 1:10, 'YTickLabel', cond_names, 'FontSize', 9, 'TickDir', 'out');
title('Cross-Task Representational Dissimilarity Matrix (10x10 RDM)', 'FontSize', 12, 'FontWeight', 'bold');
% 划分线
hold on;
xline(4.5, 'w-', 'LineWidth', 2.0);
yline(4.5, 'w-', 'LineWidth', 2.0);
xline(7.5, 'w--', 'LineWidth', 1.2);
yline(7.5, 'w--', 'LineWidth', 1.2);
hold off;

out_rdm_fig = fullfile(cfg.out_fig_dir, 'cross_task_rdm_matrix.fig');
out_rdm_png = fullfile(cfg.out_fig_dir, 'cross_task_rdm_matrix.png');
savefig(h_rdm, out_rdm_fig);
exportgraphics(h_rdm, out_rdm_png, 'Resolution', 300);
fprintf('  - RDM FIG: %s\n  - RDM PNG: %s\n', out_rdm_fig, out_rdm_png);

close(h_fig);
close(h_rdm);
