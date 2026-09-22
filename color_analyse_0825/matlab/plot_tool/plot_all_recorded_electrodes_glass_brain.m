function plot_all_recorded_electrodes_glass_brain(user_cfg)
%% ========================================================================
% 脚本名称: plot_all_recorded_electrodes_glass_brain.m
% 功能说明:
%   1. 读取全部已记录的原始电极触点数据 (all_recorded_electrodes.mat, 共 862 个触点)
%   2. 在 MNI 标准玻璃脑 (fsaverage5) 上绘制全脑解剖覆盖三视图 (Sagittal, Coronal, Axial)
%   3. 不分析任何特征 (无频段划分、无显著性门槛、无效应量大小缩放、无激活抑制区分)
%   4. 提供两种经典展现模式:
%      - 'by_subject' : 8 位被试使用各自专属颜色区分显示，并附带各被试触点数图例
%      - 'uniform'    : 全脑统一经典深蓝单色显示，呈现最纯粹的空间植入密度
%      - 'both'       : 两种模式均生成 (默认)
%   5. 输出高清科研图片至:
%      color_analyse_0825/result/figures/glass_brain/all_electrodes_coverage/
% ========================================================================

if nargin < 1
    user_cfg = struct();
end

%% 1. 参数与路径配置 (平铺直观，可在此直接修改)
cfg = struct();
cfg.mode        = 'both';        % 可选: 'both', 'by_subject', 'uniform'
cfg.point_size  = 38;            % 电极点大小 (像素面积)
cfg.line_width  = 0.6;           % 电极点黑色外边框线宽
cfg.uni_color   = [0.15, 0.45, 0.68]; % 统一单色模式下的专属颜色 (深天青蓝)

% 8 位被试专属配色方案 (高区分度、色彩协调)
cfg.sub_colors = [
    0.85, 0.22, 0.12;  % sub001: 陶土红
    0.90, 0.49, 0.13;  % sub002: 暖琥珀金
    0.15, 0.68, 0.38;  % sub003: 森林翡翠绿
    0.56, 0.27, 0.68;  % sub004: 皇家紫
    0.09, 0.63, 0.52;  % sub005: 湖水青蓝
    0.16, 0.50, 0.73;  % sub006: 深海靛蓝
    0.88, 0.30, 0.55;  % sub007: 玫瑰珊瑚粉
    0.35, 0.40, 0.45   % sub008: 碳深灰
];

% 用户自定义参数覆盖
fields = fieldnames(user_cfg);
for i = 1:numel(fields)
    cfg.(fields{i}) = user_cfg.(fields{i});
end

% 路径配置
script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(script_dir));
mesh_file  = fullfile(proj_root, 'metadata', 'brain_mesh_fsaverage5.mat');
data_file  = fullfile(proj_root, 'result', 'tables', 'all_recorded_electrodes.mat');
out_dir    = fullfile(proj_root, 'result', 'figures', 'glass_brain', 'all_electrodes_coverage');

if ~exist(out_dir, 'dir'), mkdir(out_dir); end

fprintf('========================================================================\n');
fprintf('  【MATLAB 全脑已记录电极覆盖图谱绘制 (不分析任何特征)】\n');
fprintf('========================================================================\n');

%% 2. 加载标准脑网格与全部已记录电极数据 (.mat 极速加载)
if ~isfile(mesh_file), error('未找到玻璃脑表面网格文件: %s', mesh_file); end
mesh = load(mesh_file);

if ~isfile(data_file), error('未找到全部电极数据文件: %s', data_file); end
loaded = load(data_file, 'all_elec_tbl');
tbl = loaded.all_elec_tbl;

subs = unique(tbl.subject);
n_subs = numel(subs);
n_total = height(tbl);

fprintf('[+] 成功加载全部记录电极: 共 %d 位被试，合计 %d 个空间触点。\n\n', n_subs, n_total);

%% 3. 三视图公共布局定义
views   = [90 0; 0 0; 0 90];
v_names = {'Sagittal', 'Coronal', 'Axial'};
sub_pos = [
    0.03, 0.16, 0.28, 0.72;  % Sagittal (矢状面)
    0.35, 0.16, 0.28, 0.72;  % Coronal  (冠状面)
    0.67, 0.16, 0.28, 0.72   % Axial    (水平面)
];

%% 4. 生成【按被试区分颜色版】(可选: 'both' 或 'by_subject')
if ismember(cfg.mode, {'both', 'by_subject'})
    fprintf('>>> 正在绘制 [被试专属颜色版] 三视图 ...\n');
    fig1 = figure('Visible', 'off', 'Position', [100 100 1850 700], 'Color', 'w');
    
    for v = 1:3
        ax = subplot('Position', sub_pos(v, :));
        hold(ax, 'on');
        
        % 绘制半透明玻璃脑
        patch('Vertices', mesh.l_vert, 'Faces', mesh.l_face, ...
              'FaceColor', [0.88 0.90 0.92], 'EdgeColor', 'none', ...
              'FaceAlpha', 0.10, 'FaceLighting', 'gouraud');
        patch('Vertices', mesh.r_vert, 'Faces', mesh.r_face, ...
              'FaceColor', [0.88 0.90 0.92], 'EdgeColor', 'none', ...
              'FaceAlpha', 0.10, 'FaceLighting', 'gouraud');
              
        % 逐个被试绘制电极散点
        for s = 1:n_subs
            sub_name = subs{s};
            idx = strcmp(tbl.subject, sub_name);
            scatter3(ax, tbl.mni_x(idx), tbl.mni_y(idx), tbl.mni_z(idx), cfg.point_size, ...
                     'filled', 'MarkerFaceColor', cfg.sub_colors(s, :), ...
                     'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', cfg.line_width);
        end
        
        view(ax, views(v, :));
        axis(ax, 'equal');
        axis(ax, 'off');
        camlight(ax, 'headlight');
        material(ax, 'dull');
        title(ax, v_names{v}, 'FontSize', 15, 'FontWeight', 'bold', 'Color', [0.2 0.25 0.3]);
    end
    
    % 大标题
    sgtitle(sprintf('All Recorded Electrodes Anatomical Coverage (N = %d)', n_total), ...
            'FontSize', 22, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.1]);
            
    % 底部被试图例 (2行4列精美排版)
    lgd_h = [];
    lgd_txt = {};
    for s = 1:n_subs
        h = plot(NaN, NaN, 'o', 'MarkerFaceColor', cfg.sub_colors(s, :), ...
                 'MarkerEdgeColor', [0.2 0.2 0.2], 'MarkerSize', 8, 'LineWidth', 0.6);
        lgd_h = [lgd_h, h];
        n_pts_sub = sum(strcmp(tbl.subject, subs{s}));
        lgd_txt{end+1} = sprintf('%s (n=%d)', subs{s}, n_pts_sub);
    end
    legend(lgd_h, lgd_txt, ...
           'Position', [0.05, 0.025, 0.54, 0.07], ...
           'FontSize', 11, 'FontWeight', 'bold', ...
           'NumColumns', 4, 'Box', 'off');
           
    % 底部左右方向指示轴
    draw_orientation_axis([0.65, 0.04, 0.28, 0.04]);
    
    out_sub_png = fullfile(out_dir, 'all_recorded_electrodes_by_subject_3view.png');
    exportgraphics(fig1, out_sub_png, 'Resolution', 150);
    close(fig1);
    fprintf('  [✓] 已生成: %s\n', out_sub_png);
end

%% 5. 生成【全脑统一单色经典版】(可选: 'both' 或 'uniform')
if ismember(cfg.mode, {'both', 'uniform'})
    fprintf('>>> 正在绘制 [全脑统一单色版] 三视图 ...\n');
    fig2 = figure('Visible', 'off', 'Position', [100 100 1850 700], 'Color', 'w');
    
    for v = 1:3
        ax = subplot('Position', sub_pos(v, :));
        hold(ax, 'on');
        
        patch('Vertices', mesh.l_vert, 'Faces', mesh.l_face, ...
              'FaceColor', [0.88 0.90 0.92], 'EdgeColor', 'none', ...
              'FaceAlpha', 0.10, 'FaceLighting', 'gouraud');
        patch('Vertices', mesh.r_vert, 'Faces', mesh.r_face, ...
              'FaceColor', [0.88 0.90 0.92], 'EdgeColor', 'none', ...
              'FaceAlpha', 0.10, 'FaceLighting', 'gouraud');
              
        scatter3(ax, tbl.mni_x, tbl.mni_y, tbl.mni_z, cfg.point_size, ...
                 'filled', 'MarkerFaceColor', cfg.uni_color, ...
                 'MarkerEdgeColor', [0.1 0.15 0.2], 'LineWidth', cfg.line_width);
                 
        view(ax, views(v, :));
        axis(ax, 'equal');
        axis(ax, 'off');
        camlight(ax, 'headlight');
        material(ax, 'dull');
        title(ax, v_names{v}, 'FontSize', 15, 'FontWeight', 'bold', 'Color', [0.2 0.25 0.3]);
    end
    
    sgtitle(sprintf('All Recorded Electrodes Anatomical Coverage (N = %d)', n_total), ...
            'FontSize', 22, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.1]);
            
    % 简明单色图例
    h_uni = plot(NaN, NaN, 'o', 'MarkerFaceColor', cfg.uni_color, ...
                 'MarkerEdgeColor', [0.1 0.15 0.2], 'MarkerSize', 9, 'LineWidth', 0.6);
    legend(h_uni, {sprintf('Recorded Electrode Contact (N = %d)', n_total)}, ...
           'Position', [0.08, 0.04, 0.30, 0.05], ...
           'FontSize', 12, 'FontWeight', 'bold', 'Box', 'off');
           
    draw_orientation_axis([0.55, 0.04, 0.36, 0.04]);
    
    out_uni_png = fullfile(out_dir, 'all_recorded_electrodes_uniform_3view.png');
    exportgraphics(fig2, out_uni_png, 'Resolution', 150);
    close(fig2);
    fprintf('  [✓] 已生成: %s\n', out_uni_png);
end

fprintf('\n========================================================================\n');
fprintf('  【全部记录电极三视图绘制完成！】\n  保存目录: %s\n', out_dir);
fprintf('========================================================================\n');
end

%% 辅助函数: 绘制底部左右方向指示轴
function draw_orientation_axis(pos)
    ax = axes('Position', pos, 'Visible', 'off');
    hold(ax, 'on');
    plot(ax, [-0.80, 0.80], [0, 0], 'k-', 'LineWidth', 2.0);
    plot(ax, -0.80, 0, '<k', 'MarkerFaceColor', 'k', 'MarkerSize', 9);
    plot(ax, 0.80, 0, '>k', 'MarkerFaceColor', 'k', 'MarkerSize', 9);
    text(ax, -0.85, 0, 'Left', 'FontSize', 14, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
    text(ax, 0.85, 0, 'Right', 'FontSize', 14, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
    xlim(ax, [-1.2, 1.2]);
    ylim(ax, [-1, 1]);
end
