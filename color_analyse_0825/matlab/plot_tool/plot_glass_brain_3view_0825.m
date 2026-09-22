function plot_glass_brain_3view_0825(user_cfg)
%% ========================================================================
% 脚本名称: plot_glass_brain_3view_0825.m
% 功能说明:
%   1. 读取 MNI 标准玻璃脑皮层表面 (fsaverage5) 与显著通道三维主数据表
%   2. 支持两种统计筛选模式:
%      - 'uncorrected': 未校正置换检验 p 值 (例如 p < 0.05 或 p < 0.005)
%      - 'fdr_sub'    : 被试内同频段 Benjamini-Hochberg FDR 校正 (例如 q < 0.05)
%      - 'fdr_band'   : 全频段同频段 Benjamini-Hochberg FDR 校正 (例如 q < 0.05)
%   3. 在【每个频段内部各自独立】执行 Sigmoid 非线性对比度拉大，效应差异醒目
%   4. 逐频段绘制学术级三视图 (Sagittal, Coronal, Axial):
%      - 主图顶部仅标注频段名称 (如 High-Gamma)
%      - 实心点: Color > Gray (ERS 激活)
%      - 空心实线圆: Gray > Color (ERD 抑制)
%      - 图例精简: 仅标注 "Color > Gray" 与 "Gray > Color"
%      - 三视图下方附带全局左右方向轴: Left <--------> Right
%   5. 自动按版本输出至独立文件夹 (如 uncorrected_p0.05 / fdr_sub_q0.05)
% ========================================================================

%% 1. 参数与路径配置 (直观平铺，可在此直接修改)
if nargin < 1
    user_cfg = struct();
end

cfg = struct();

% --- 统计模式与阈值控制 ---
% 可选模式: 'uncorrected' (未校正p值), 'fdr_sub' (被试内FDR), 'fdr_band' (全频段FDR)
cfg.stat_mode       = 'uncorrected'; 
cfg.p_thresh        = 0.05;          % 未校正 p 值阈值 (当 stat_mode 为 'uncorrected' 时生效)
cfg.fdr_q           = 0.05;          % FDR 校正 q 值阈值 (当 stat_mode 为 'fdr_sub' 或 'fdr_band' 时生效)
cfg.concordant_only = true;          % 是否仅保留四类别严格同向位点 (true: 剔除类别偏好位点; false: 保留全部)

% --- 单频段独立 Sigmoid 参数 ---
cfg.sig_k       = 6.0;           % Sigmoid 陡度斜率 (越大对比度越强烈)
cfg.min_size    = 50;            % 实心点最小面积 (像素)
cfg.max_size    = 380;           % 实心点最大面积 (像素)
cfg.min_rad     = 2.8;           % 空心实线圆最小几何半径 (mm)
cfg.max_rad     = 8.8;           % 空心实线圆最大几何半径 (mm)

% 目标频段与配色
cfg.bands       = {'High_Gamma', 'Low_Gamma', 'Beta', 'Alpha', 'Theta', 'Delta'};
cfg.band_titles = {'High-Gamma', 'Low-Gamma', 'Beta', 'Alpha', 'Theta', 'Delta'};
cfg.band_colors = {
    [0.85, 0.22, 0.12];  % High_Gamma: 暖陶土红
    [0.90, 0.49, 0.13];  % Low_Gamma:  暖琥珀金
    [0.15, 0.68, 0.38];  % Beta:       森林翡翠绿
    [0.56, 0.27, 0.68];  % Alpha:      皇家紫
    [0.09, 0.63, 0.52];  % Theta:      湖水青蓝
    [0.16, 0.50, 0.73];  % Delta:      深海靛蓝
};

% 用户自定义参数覆盖
fields = fieldnames(user_cfg);
for i = 1:numel(fields)
    cfg.(fields{i}) = user_cfg.(fields{i});
end

% 路径配置
script_dir   = fileparts(mfilename('fullpath'));
proj_root    = fileparts(fileparts(script_dir)); % 定位到 color_analyse_0825
mesh_file    = fullfile(proj_root, 'metadata', 'brain_mesh_fsaverage5.mat');
data_file    = fullfile(proj_root, 'result', 'tables', 'color_effects_summary.mat');
loc_dir      = fullfile(proj_root, 'metadata', 'ieeg_location');

% 根据统计版本自动创建独立输出文件夹
if strcmp(cfg.stat_mode, 'uncorrected')
    base_folder = sprintf('uncorrected_p%.3f', cfg.p_thresh);
elseif strcmp(cfg.stat_mode, 'bonferroni')
    cfg.p_thresh = 0.05 / 6;
    base_folder  = 'bonferroni_p0.0083';
elseif strcmp(cfg.stat_mode, 'fdr_sub')
    base_folder = sprintf('fdr_sub_q%.3f', cfg.fdr_q);
elseif strcmp(cfg.stat_mode, 'fdr_band')
    base_folder = sprintf('fdr_band_q%.3f', cfg.fdr_q);
else
    error('未知的统计模式 cfg.stat_mode = %s', cfg.stat_mode);
end
% 去除末尾多余的零以保持文件夹名简明，例如 p0.05 而非 p0.050
base_folder = regexprep(base_folder, '0+$', '');
base_folder = regexprep(base_folder, '\.$', '');

if cfg.concordant_only
    folder_name = [base_folder, '_concordant'];
else
    folder_name = base_folder;
end

out_fig_dir  = fullfile(proj_root, 'result', 'figures', 'glass_brain', folder_name);
if ~exist(out_fig_dir, 'dir'), mkdir(out_fig_dir); end

fprintf('========================================================================\n');
fprintf('  【MATLAB 玻璃脑电极三视图绘制】\n');
fprintf('  当前统计模式: %s | 四类别同向限定: %d | 输出目录: %s\n', ...
        cfg.stat_mode, cfg.concordant_only, folder_name);
fprintf('========================================================================\n');

%% 2. 加载标准脑网格与主数据表 (.mat 原生极速加载并动态匹配 MNI 坐标)
if ~isfile(mesh_file), error('未找到网格文件: %s', mesh_file); end
mesh = load(mesh_file);

if ~isfile(data_file), error('未找到主数据文件: %s', data_file); end
loaded = load(data_file);
if isfield(loaded, 'all_tbl')
    tbl = loaded.all_tbl;
else
    tbl = loaded.res_table;
end

% 统一字段名映射
if ~ismember('general_effect_dB', tbl.Properties.VariableNames) && ismember('general_effect_100_400ms', tbl.Properties.VariableNames)
    tbl.general_effect_dB = tbl.general_effect_100_400ms;
end
if ~ismember('p_perm', tbl.Properties.VariableNames) && ismember('p_perm_100_400ms', tbl.Properties.VariableNames)
    tbl.p_perm = tbl.p_perm_100_400ms;
end
if ~ismember('has_mni_coord', tbl.Properties.VariableNames)
    tbl.has_mni_coord = ~isnan(tbl.mni_x) & ~isnan(tbl.mni_y) & ~isnan(tbl.mni_z);
end

% 动态解析缺失的 MNI 坐标 (适配 sub008 等以字符串存储坐标的情况)
missing_idx = find(~tbl.has_mni_coord);
for mi = 1:numel(missing_idx)
    idx = missing_idx(mi);
    sub = char(tbl.subject{idx});
    ch  = char(tbl.channel{idx});
    f_xlsx = fullfile(loc_dir, sprintf('%s_ieegloc.xlsx', sub));
    f_tsv  = fullfile(loc_dir, sprintf('%s.tsv', sub));
    if isfile(f_xlsx)
        t_loc = readtable(f_xlsx, 'VariableNamingRule', 'preserve');
        c_m = strcmp(string(table2cell(t_loc(:, 1))), string(ch));
        c_idx = find(c_m, 1);
        if ~isempty(c_idx)
            mni_col = find(strcmpi(t_loc.Properties.VariableNames, 'MNI'), 1);
            if ~isempty(mni_col)
                mni_raw = string(t_loc{c_idx, mni_col});
                nums = sscanf(char(strrep(strrep(mni_raw, '[', ''), ']', '')), '%f,%f,%f');
                if numel(nums) == 3
                    tbl.mni_x(idx) = nums(1);
                    tbl.mni_y(idx) = nums(2);
                    tbl.mni_z(idx) = nums(3);
                    tbl.has_mni_coord(idx) = true;
                end
            end
        end
    elseif isfile(f_tsv)
        t_tsv = readtable(f_tsv, 'FileType', 'text', 'Delimiter', '\t');
        c_m = strcmp(string(t_tsv.Channel), string(ch));
        c_idx = find(c_m, 1);
        if ~isempty(c_idx) && ismember('MNI', t_tsv.Properties.VariableNames)
            mni_raw = string(t_tsv.MNI(c_idx));
            nums = sscanf(char(strrep(strrep(mni_raw, '[', ''), ']', '')), '%f,%f,%f');
            if numel(nums) == 3
                tbl.mni_x(idx) = nums(1);
                tbl.mni_y(idx) = nums(2);
                tbl.mni_z(idx) = nums(3);
                tbl.has_mni_coord(idx) = true;
            end
        end
    end
end

% 统计筛选逻辑 (逻辑极其直观清晰，方便检查)
if strcmp(cfg.stat_mode, 'uncorrected')
    valid_idx = (tbl.has_mni_coord == 1) & (tbl.p_perm < cfg.p_thresh);
    desc_str  = sprintf('未校正置换检验 p < %g', cfg.p_thresh);
elseif strcmp(cfg.stat_mode, 'bonferroni')
    cfg.p_thresh = 0.05 / 6;
    valid_idx = (tbl.has_mni_coord == 1) & (tbl.p_perm < cfg.p_thresh);
    desc_str  = sprintf('Bonferroni 校正 (p < 0.05/6 = %.4f)', cfg.p_thresh);
elseif strcmp(cfg.stat_mode, 'fdr_sub')
    valid_idx = (tbl.has_mni_coord == 1) & (tbl.q_fdr_sub < cfg.fdr_q);
    desc_str  = sprintf('被试内同频段 FDR q < %g', cfg.fdr_q);
elseif strcmp(cfg.stat_mode, 'fdr_band')
    valid_idx = (tbl.has_mni_coord == 1) & (tbl.q_fdr_band < cfg.fdr_q);
    desc_str  = sprintf('全频段同频段 FDR q < %g', cfg.fdr_q);
end

% 四类别同向性限定 (过滤类别偏好位点)
if cfg.concordant_only
    is_concord = strcmp(tbl.concordance_type, 'Concordant_Positive') | ...
                 strcmp(tbl.concordance_type, 'Concordant_Negative');
    valid_idx  = valid_idx & is_concord;
    desc_str   = [desc_str, ' + 四类别严格同向'];
else
    desc_str   = [desc_str, ' + 包含类别偏好位点'];
end

tbl_sig = tbl(valid_idx, :);
fprintf('[+] 筛选条件: %s\n', desc_str);
fprintf('[+] 筛选后共有 %d 个具备 MNI 空间坐标的显著位点。\n\n', height(tbl_sig));

%% 3. 逐频段绘制三视图 (频段内独立 Sigmoid)
views = [90 0; 0 0; 0 90];
v_names = {'Sagittal', 'Coronal', 'Axial'};
th = linspace(0, 2*pi, 40);

for b = 1:numel(cfg.bands)
    band_name  = cfg.bands{b};
    band_title = cfg.band_titles{b};
    band_color = cfg.band_colors{b};
    
    b_mask = strcmp(tbl_sig.freq_band, band_name);
    b_sub  = tbl_sig(b_mask, :);
    n_pts  = height(b_sub);
    
    % 创建画布 (1800 x 680 像素)
    fig = figure('Visible', 'off', 'Position', [100 100 1800 680], 'Color', 'w');
    sub_pos = [
        0.03, 0.16, 0.28, 0.72;  % Sagittal
        0.35, 0.16, 0.28, 0.72;  % Coronal
        0.67, 0.16, 0.28, 0.72   % Axial
    ];
    
    if n_pts > 0
        x = b_sub.mni_x;
        y = b_sub.mni_y;
        z = b_sub.mni_z;
        diff_val = b_sub.general_effect_dB;
        abs_diff = abs(diff_val);
        
        % 频段内独立 Sigmoid 非线性拉大
        d_min = min(abs_diff);
        d_max = max(abs_diff);
        if d_max > d_min
            z_norm = (abs_diff - d_min) / (d_max - d_min);
        else
            z_norm = zeros(size(abs_diff)) + 0.5;
        end
        sig_raw = 1 ./ (1 + exp(-cfg.sig_k * (z_norm - 0.5)));
        sig_min = 1 / (1 + exp(cfg.sig_k * 0.5));
        sig_max = 1 / (1 + exp(-cfg.sig_k * 0.5));
        sig_w   = (sig_raw - sig_min) / (sig_max - sig_min);
        
        pt_sizes = cfg.min_size + sig_w .* (cfg.max_size - cfg.min_size);
        pt_radii = cfg.min_rad  + sig_w .* (cfg.max_rad - cfg.min_rad);
        
        is_col = (diff_val > 0);
        is_gry = (diff_val < 0);
    else
        is_col = [];
        is_gry = [];
        d_min = 0; d_max = 0;
    end
    
    for v = 1:3
        ax = subplot('Position', sub_pos(v, :));
        hold(ax, 'on');
        
        % 1) 绘制半透明玻璃脑表面
        patch('Vertices', mesh.l_vert, 'Faces', mesh.l_face, ...
              'FaceColor', [0.88 0.90 0.92], 'EdgeColor', 'none', ...
              'FaceAlpha', 0.10, 'FaceLighting', 'gouraud');
        patch('Vertices', mesh.r_vert, 'Faces', mesh.r_face, ...
              'FaceColor', [0.88 0.90 0.92], 'EdgeColor', 'none', ...
              'FaceAlpha', 0.10, 'FaceLighting', 'gouraud');
        
        % 2) 绘制实心点 (Color > Gray)
        if any(is_col)
            scatter3(ax, x(is_col), y(is_col), z(is_col), pt_sizes(is_col), ...
                     'filled', 'MarkerFaceColor', band_color, ...
                     'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.8);
        end
        
        % 3) 绘制空心实线圆 (Gray > Color)
        if any(is_gry)
            gry_idx = find(is_gry);
            for i = 1:numel(gry_idx)
                k = gry_idx(i);
                cx0 = x(k); cy0 = y(k); cz0 = z(k); r0 = pt_radii(k);
                if v == 1      % Sagittal
                    plot3(ax, cx0*ones(size(th)), cy0 + r0*cos(th), cz0 + r0*sin(th), ...
                          '-', 'Color', band_color, 'LineWidth', 2.0);
                elseif v == 2  % Coronal
                    plot3(ax, cx0 + r0*cos(th), cy0*ones(size(th)), cz0 + r0*sin(th), ...
                          '-', 'Color', band_color, 'LineWidth', 2.0);
                elseif v == 3  % Axial
                    plot3(ax, cx0 + r0*cos(th), cy0 + r0*sin(th), cz0*ones(size(th)), ...
                          '-', 'Color', band_color, 'LineWidth', 2.0);
                end
            end
        end
        
        % 视角与光学属性
        view(ax, views(v, :));
        axis(ax, 'equal');
        axis(ax, 'off');
        camlight(ax, 'headlight');
        material(ax, 'dull');
        title(ax, v_names{v}, 'FontSize', 15, 'FontWeight', 'bold', 'Color', [0.2 0.25 0.3]);
    end
    
    % 主图标题: 仅标注频段名称 (若位点数为0则加注说明)
    if n_pts > 0
        sgtitle(band_title, 'FontSize', 22, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.1]);
    else
        sgtitle(sprintf('%s (N = 0)', band_title), 'FontSize', 22, 'FontWeight', 'bold', 'Color', [0.4 0.4 0.4]);
    end
    
    % 图例设置: 仅标 "Color > Gray" 与 "Gray > Color"
    h_col = plot(NaN, NaN, 'o', 'MarkerFaceColor', band_color, 'MarkerEdgeColor', [0.2 0.2 0.2], 'MarkerSize', 11);
    h_gry = plot(NaN, NaN, 'o', 'Color', band_color, 'MarkerFaceColor', 'none', 'LineWidth', 2.0, 'MarkerSize', 11);
    legend([h_col, h_gry], {'Color > Gray', 'Gray > Color'}, ...
           'Position', [0.06, 0.04, 0.22, 0.05], ...
           'FontSize', 13, 'FontWeight', 'bold', ...
           'Orientation', 'horizontal', 'Box', 'off');
    
    % 总体三视图底部方向轴: 加 Left 和 Right 表示左右
    ax_orient = axes('Position', [0.42, 0.04, 0.44, 0.04], 'Visible', 'off');
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
    
    % 保存图片
    out_file = fullfile(out_fig_dir, sprintf('matlab_glass_brain_%s_3view.png', band_name));
    exportgraphics(fig, out_file, 'Resolution', 150);
    close(fig);
    if n_pts > 0
        fprintf('  [✓] 频段 [%s] 绘制完成 (N = %d, 效应: %.2f ~ %.2f dB) -> %s\n', ...
                band_name, n_pts, d_min, d_max, out_file);
    else
        fprintf('  [✓] 频段 [%s] 绘制完成 (N = 0, 空白玻璃脑) -> %s\n', ...
                band_name, out_file);
    end
end

fprintf('\n========================================================================\n');
fprintf('  【当前版本生成完毕】保存目录: %s\n', out_fig_dir);
fprintf('========================================================================\n');
end
