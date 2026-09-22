% test_plot_venn_preview.m
% 测试 3-Circle Venn 图的几何排版与显示效果

clear; clc; close all;

% 1. 几何参数定义
R = 1.0;
cx = [-0.6, 0.6, 0];
cy = [0.4, 0.4, -0.5];

% 配色: 经典高雅学术配色 (淡红、淡蓝、淡绿)
colors = [
    0.85, 0.35, 0.25;  % Circle A: 珊瑚红
    0.20, 0.55, 0.85;  % Circle B: 天空蓝
    0.25, 0.70, 0.45   % Circle C: 翡翠绿
];

th = linspace(0, 2*pi, 300);

fig = figure('Color', 'w', 'Position', [100, 100, 700, 600]);
ax = axes('Position', [0.1, 0.1, 0.8, 0.8]);
hold on; axis equal; box off; axis off;

% 绘制三个填充圆
for i = 1:3
    x = cx(i) + R * cos(th);
    y = cy(i) + R * sin(th);
    fill(x, y, colors(i, :), 'FaceAlpha', 0.35, 'EdgeColor', colors(i, :) * 0.7, 'LineWidth', 2.0);
end

% 文字标签坐标测试
text(-0.95, 0.5, 'Only A', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
text(0.95, 0.5, 'Only B', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
text(0, -0.95, 'Only C', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');

text(0, 0.65, 'A & B', 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.2, 0.2, 0.5]);
text(-0.45, -0.15, 'A & C', 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.4, 0.2, 0.2]);
text(0.45, -0.15, 'B & C', 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.1, 0.4, 0.2]);

text(0, 0.12, 'A & B & C', 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');

% 集合标题
text(-0.9, 1.55, 'Task 2 Decoding', 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold', 'Color', colors(1,:)*0.6);
text(0.9, 1.55, 'Task 3 Decoding', 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold', 'Color', colors(2,:)*0.6);
text(0, -1.65, 'Cross-Task (T2 vs T3)', 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold', 'Color', colors(3,:)*0.6);

xlim([-2.0, 2.0]);
ylim([-2.0, 2.0]);

test_dir = fileparts(mfilename('fullpath'));
png_out = fullfile(test_dir, 'test_venn.png');
exportgraphics(fig, png_out, 'Resolution', 300);
close(fig);
fprintf('Venn 几何测试图已导出: %s\n', png_out);
