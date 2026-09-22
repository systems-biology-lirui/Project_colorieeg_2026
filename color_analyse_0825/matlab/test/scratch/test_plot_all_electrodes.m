% test_plot_all_electrodes.m
clear; clc; close all;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';

mesh_file = fullfile(proj_root, 'metadata', 'brain_mesh_fsaverage5.mat');
data_file = fullfile(proj_root, 'result', 'tables', 'all_recorded_electrodes.mat');

mesh = load(mesh_file);
d = load(data_file, 'all_elec_tbl');
tbl = d.all_elec_tbl;

subs = unique(tbl.subject);
n_subs = numel(subs);
colors = [
    0.85, 0.22, 0.12;  % sub001: 陶土红
    0.90, 0.49, 0.13;  % sub002: 暖琥珀金
    0.15, 0.68, 0.38;  % sub003: 森林翡翠绿
    0.56, 0.27, 0.68;  % sub004: 皇家紫
    0.09, 0.63, 0.52;  % sub005: 湖水青蓝
    0.16, 0.50, 0.73;  % sub006: 深海靛蓝
    0.88, 0.30, 0.55;  % sub007: 玫瑰珊瑚粉
    0.35, 0.40, 0.45   % sub008: 碳深灰
];

out_dir = fullfile(proj_root, 'result', 'figures', 'glass_brain', 'all_electrodes_coverage');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

views = [90 0; 0 0; 0 90];
v_names = {'Sagittal', 'Coronal', 'Axial'};
sub_pos = [
    0.03, 0.16, 0.28, 0.72;  % Sagittal
    0.35, 0.16, 0.28, 0.72;  % Coronal
    0.67, 0.16, 0.28, 0.72   % Axial
];

%% 1. 生成按被试区分颜色的三视图
fig1 = figure('Visible', 'off', 'Position', [100 100 1850 700], 'Color', 'w');

for v = 1:3
    ax = subplot('Position', sub_pos(v, :));
    hold(ax, 'on');
    
    patch('Vertices', mesh.l_vert, 'Faces', mesh.l_face, ...
          'FaceColor', [0.88 0.90 0.92], 'EdgeColor', 'none', ...
          'FaceAlpha', 0.10, 'FaceLighting', 'gouraud');
    patch('Vertices', mesh.r_vert, 'Faces', mesh.r_face, ...
          'FaceColor', [0.88 0.90 0.92], 'EdgeColor', 'none', ...
          'FaceAlpha', 0.10, 'FaceLighting', 'gouraud');
          
    for s = 1:n_subs
        sub_name = subs{s};
        idx = strcmp(tbl.subject, sub_name);
        scatter3(ax, tbl.mni_x(idx), tbl.mni_y(idx), tbl.mni_z(idx), 38, ...
                 'filled', 'MarkerFaceColor', colors(s, :), ...
                 'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.6);
    end
    
    view(ax, views(v, :));
    axis(ax, 'equal');
    axis(ax, 'off');
    camlight(ax, 'headlight');
    material(ax, 'dull');
    title(ax, v_names{v}, 'FontSize', 15, 'FontWeight', 'bold', 'Color', [0.2 0.25 0.3]);
end

sgtitle(sprintf('All Recorded Electrodes Anatomical Coverage (N = %d)', height(tbl)), ...
        'FontSize', 22, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.1]);

% 底部被试图例 (精致2行排版)
lgd_handles = [];
lgd_labels  = {};
for s = 1:n_subs
    h = plot(NaN, NaN, 'o', 'MarkerFaceColor', colors(s, :), ...
             'MarkerEdgeColor', [0.2 0.2 0.2], 'MarkerSize', 8, 'LineWidth', 0.6);
    lgd_handles = [lgd_handles, h];
    n_pts_sub = sum(strcmp(tbl.subject, subs{s}));
    lgd_labels{end+1} = sprintf('%s (n=%d)', subs{s}, n_pts_sub);
end

legend(lgd_handles, lgd_labels, ...
       'Position', [0.05, 0.025, 0.54, 0.07], ...
       'FontSize', 11, 'FontWeight', 'bold', ...
       'NumColumns', 4, 'Box', 'off');

% 底部方向轴
ax_orient = axes('Position', [0.65, 0.04, 0.28, 0.04], 'Visible', 'off');
hold(ax_orient, 'on');
plot(ax_orient, [-0.80, 0.80], [0, 0], 'k-', 'LineWidth', 2.0);
plot(ax_orient, -0.80, 0, '<k', 'MarkerFaceColor', 'k', 'MarkerSize', 9);
plot(ax_orient, 0.80, 0, '>k', 'MarkerFaceColor', 'k', 'MarkerSize', 9);
text(ax_orient, -0.85, 0, 'Left', 'FontSize', 14, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
text(ax_orient, 0.85, 0, 'Right', 'FontSize', 14, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
xlim(ax_orient, [-1.2, 1.2]);
ylim(ax_orient, [-1, 1]);

out_sub_png = fullfile(out_dir, 'all_recorded_electrodes_by_subject_3view.png');
exportgraphics(fig1, out_sub_png, 'Resolution', 150);
close(fig1);
fprintf('[✓] 成功输出按被试区分颜色图片: %s\n', out_sub_png);

%% 2. 生成全脑统一单色经典版
fig2 = figure('Visible', 'off', 'Position', [100 100 1850 700], 'Color', 'w');
uni_color = [0.15, 0.45, 0.68]; % 雅致深天青蓝

for v = 1:3
    ax = subplot('Position', sub_pos(v, :));
    hold(ax, 'on');
    
    patch('Vertices', mesh.l_vert, 'Faces', mesh.l_face, ...
          'FaceColor', [0.88 0.90 0.92], 'EdgeColor', 'none', ...
          'FaceAlpha', 0.10, 'FaceLighting', 'gouraud');
    patch('Vertices', mesh.r_vert, 'Faces', mesh.r_face, ...
          'FaceColor', [0.88 0.90 0.92], 'EdgeColor', 'none', ...
          'FaceAlpha', 0.10, 'FaceLighting', 'gouraud');
          
    scatter3(ax, tbl.mni_x, tbl.mni_y, tbl.mni_z, 36, ...
             'filled', 'MarkerFaceColor', uni_color, ...
             'MarkerEdgeColor', [0.1 0.15 0.2], 'LineWidth', 0.6);
    
    view(ax, views(v, :));
    axis(ax, 'equal');
    axis(ax, 'off');
    camlight(ax, 'headlight');
    material(ax, 'dull');
    title(ax, v_names{v}, 'FontSize', 15, 'FontWeight', 'bold', 'Color', [0.2 0.25 0.3]);
end

sgtitle(sprintf('All Recorded Electrodes Anatomical Coverage (N = %d)', height(tbl)), ...
        'FontSize', 22, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.1]);

% 图例
h_uni = plot(NaN, NaN, 'o', 'MarkerFaceColor', uni_color, ...
             'MarkerEdgeColor', [0.1 0.15 0.2], 'MarkerSize', 9, 'LineWidth', 0.6);
legend(h_uni, {sprintf('Recorded Electrode Contact (N = %d)', height(tbl))}, ...
       'Position', [0.08, 0.04, 0.30, 0.05], ...
       'FontSize', 12, 'FontWeight', 'bold', 'Box', 'off');

% 底部方向轴
ax_orient2 = axes('Position', [0.55, 0.04, 0.36, 0.04], 'Visible', 'off');
hold(ax_orient2, 'on');
plot(ax_orient2, [-0.80, 0.80], [0, 0], 'k-', 'LineWidth', 2.0);
plot(ax_orient2, -0.80, 0, '<k', 'MarkerFaceColor', 'k', 'MarkerSize', 9);
plot(ax_orient2, 0.80, 0, '>k', 'MarkerFaceColor', 'k', 'MarkerSize', 9);
text(ax_orient2, -0.85, 0, 'Left', 'FontSize', 14, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
text(ax_orient2, 0.85, 0, 'Right', 'FontSize', 14, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
xlim(ax_orient2, [-1.2, 1.2]);
ylim(ax_orient2, [-1, 1]);

out_uni_png = fullfile(out_dir, 'all_recorded_electrodes_uniform_3view.png');
exportgraphics(fig2, out_uni_png, 'Resolution', 150);
close(fig2);
fprintf('[✓] 成功输出统一单色图片: %s\n', out_uni_png);
