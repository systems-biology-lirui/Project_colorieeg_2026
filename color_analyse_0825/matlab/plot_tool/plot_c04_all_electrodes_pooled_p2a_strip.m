%% plot_c04_all_electrodes_pooled_p2a_strip.m
% =========================================================================
% Function: Plot significant color electrodes pooled across all subjects
%           strictly sorted from Posterior to Anterior (MNI Y ascending)
%
% Visual Design:
%   - All 793 electrodes from 8 subjects pooled together (no subject division)
%   - Horizontal axis: Sorted continuously from Posterior to Anterior (left to right)
%   - Top solid gray bar: Represents all 793 electrode sites
%   - Vertical axis: 6 frequency bands from High Gamma down to Delta
%   - Horizontal axis ticks: Anatomical MNI Y coordinates (mm)
%   - Solid borderless patches (EdgeColor = 'none')
%   - Exports 4 figures: Combined, Positive-only, Negative-only, Biased-only
% =========================================================================

clear; clc; close all;

%% 1. Configuration (Simple, intuitive, adjustable)
cfg = struct();
cfg.subjects    = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008'};
cfg.alpha_sig   = 0.05;              % Significance threshold
cfg.fig_size    = [2000, 520];       % Figure size [Width, Height]
cfg.dpi         = 300;               % Resolution for export

% Colors for the 3 significance types (Standard journal palette, no borders)
cfg.col_pos     = [0.86, 0.20, 0.20];  % Concordant Positive: Red
cfg.col_neg     = [0.15, 0.45, 0.75];  % Concordant Negative: Blue
cfg.col_bias    = [0.98, 0.55, 0.10];  % Category Biased: Orange
cfg.col_gray    = [0.75, 0.75, 0.78];  % Electrode bar: Solid Gray

%% 2. Paths
script_dir  = fileparts(mfilename('fullpath'));
proj_root   = fileparts(fileparts(script_dir)); % Root: color_analyse_0825
proc_new    = fullfile(proj_root, 'process_data_new');
res_root    = fullfile(proj_root, 'result');
res_table   = fullfile(res_root, 'tables', 'color_effects_summary.mat');
loc_dir     = fullfile(proj_root, 'metadata', 'ieeg_location');
out_fig_dir = fullfile(res_root, 'figures', 'c04_screening');
meta_fig_dir= fullfile(proj_root, 'metadata', 'figures');

if ~exist(out_fig_dir, 'dir'),  mkdir(out_fig_dir);  end
if ~exist(meta_fig_dir, 'dir'), mkdir(meta_fig_dir); end

%% 3. Frequency bands setup (High to Low frequency)
band_keys   = {'High_Gamma', 'Low_Gamma', 'Beta', 'Alpha', 'Theta', 'Delta'};
band_labels = {'High \gamma (60-140 Hz)', 'Low \gamma (30-60 Hz)', ...
               '\beta (13-30 Hz)', '\alpha (8-13 Hz)', '\theta (4-8 Hz)', '\delta (1-4 Hz)'};
n_bands     = numel(band_keys);

% Load C04 detailed summary table
if ~isfile(res_table)
    error('C04 summary table not found: %s\nPlease run C04_screen_color_channels_0825.m first.', res_table);
end
loaded_c04 = load(res_table);
if isfield(loaded_c04, 'all_tbl'), c04_tbl = loaded_c04.all_tbl; else, c04_tbl = loaded_c04.res_table; end

%% 4. Load all electrodes and MNI Y coordinates across all 8 subjects
pool_sub  = {};
pool_ch   = {};
pool_y    = [];
pool_sig  = []; % (n_bands x n_elecs)

for s = 1:numel(cfg.subjects)
    s_id = cfg.subjects{s};
    mat_f = fullfile(proc_new, s_id, 'task1_multiband_epoched.mat');
    if ~isfile(mat_f)
        error('Feature cache not found for %s: %s', s_id, mat_f);
    end
    ep_obj = load(mat_f);
    chs = ep_obj.epoched_data.triplet_info.center_channel;
    n_chs = numel(chs);
    
    % Read localization table
    xlsx_f = fullfile(loc_dir, sprintf('%s_ieegloc.xlsx', s_id));
    tsv_f  = fullfile(loc_dir, sprintf('%s.tsv', s_id));
    t_loc  = [];
    if isfile(xlsx_f)
        t_loc = readtable(xlsx_f, 'VariableNamingRule', 'preserve');
    elseif isfile(tsv_f)
        t_loc = readtable(tsv_f, 'FileType', 'text', 'Delimiter', '\t');
    end
    
    mni_y = nan(n_chs, 1);
    if ~isempty(t_loc)
        mni_col = find(strcmpi(t_loc.Properties.VariableNames, 'MNI'), 1);
        ch_col  = find(strcmpi(t_loc.Properties.VariableNames, 'Channel'), 1);
        if isempty(ch_col), ch_col = 1; end
        
        for i = 1:n_chs
            ch_name = chs{i};
            idx = find(strcmp(string(t_loc{:, ch_col}), string(ch_name)), 1);
            if ~isempty(idx) && ~isempty(mni_col)
                raw_str = string(t_loc{idx, mni_col});
                nums = sscanf(char(strrep(strrep(raw_str, '[', ''), ']', '')), '%f,%f,%f');
                if numel(nums) == 3
                    mni_y(i) = nums(2);
                end
            end
        end
    end
    
    % 同轴 fallback (末梢触点取同轴已知触点平均 Y)
    for i = 1:n_chs
        if isnan(mni_y(i))
            shaft = regexprep(chs{i}, '\d+', '');
            same_shaft = find(strncmp(chs, shaft, length(shaft)) & ~isnan(mni_y));
            if ~isempty(same_shaft)
                mni_y(i) = mean(mni_y(same_shaft));
            end
        end
    end
    
    % 未定位电极 (如 sub004 E~L 轴) 赋予 +100，排在最前端
    mni_y(isnan(mni_y)) = 100;
    
    % 获取该被试每个电极在 6 频段上的显著性类型
    s_tbl = c04_tbl(strcmp(c04_tbl.subject, s_id), :);
    s_sig = zeros(n_bands, n_chs);
    for ch_idx = 1:n_chs
        ch_name = chs{ch_idx};
        for b = 1:n_bands
            b_name = band_keys{b};
            m = strcmp(s_tbl.channel, ch_name) & strcmp(s_tbl.freq_band, b_name);
            if any(m)
                row = s_tbl(m, :);
                if row.p_perm_100_400ms(1) < cfg.alpha_sig
                    c_type = row.concordance_type{1};
                    if strcmp(c_type, 'Concordant_Positive')
                        s_sig(b, ch_idx) = 1;
                    elseif strcmp(c_type, 'Concordant_Negative')
                        s_sig(b, ch_idx) = 2;
                    else
                        s_sig(b, ch_idx) = 3;
                    end
                end
            end
        end
    end
    
    % 存入池中
    pool_sub = [pool_sub; repmat({s_id}, n_chs, 1)];
    pool_ch  = [pool_ch; chs(:)];
    pool_y   = [pool_y; mni_y(:)];
    pool_sig = [pool_sig, s_sig];
end

n_total = numel(pool_y);

%% 5. 全局按 MNI Y 从后往前 (Posterior -> Anterior) 升序排序
[sorted_y, sort_idx] = sort(pool_y, 'ascend');
sorted_sub = pool_sub(sort_idx);
sorted_ch  = pool_ch(sort_idx);
sorted_sig = pool_sig(:, sort_idx);

tot_pos   = sum(sorted_sig(:) == 1);
tot_neg   = sum(sorted_sig(:) == 2);
tot_bias  = sum(sorted_sig(:) == 3);

% 构建 X 轴坐标刻度 (按 MNI Y 实际解剖位置设置刻度，保证间距充足无重叠)
target_ys = [-100, -60, -40, -20, 0, 20, 40];
x_ticks   = [];
x_labels  = {};

for ty = target_ys
    idx = find(sorted_y >= ty & sorted_y < 90, 1);
    if ~isempty(idx)
        if isempty(x_ticks) || (idx - x_ticks(end) >= 30)
            x_ticks(end+1) = idx; %#ok<AGROW>
            if ty > 0
                x_labels{end+1} = sprintf('+%d mm', ty); %#ok<AGROW>
            else
                x_labels{end+1} = sprintf('%d mm', ty); %#ok<AGROW>
            end
        end
    end
end

% 最右端未定位电极标记 (sub004 等缺乏定位轴)
unloc_start = find(sorted_y >= 90, 1);
if ~isempty(unloc_start)
    x_ticks(end+1) = (unloc_start + n_total) / 2;
    x_labels{end+1} = 'Unlocalized';
end

%% 6. 循环绘制 4 张全景图: 1 张合一图 + 3 张分类型独立图
modes = {'combined', 'positive', 'negative', 'biased'};

for m_idx = 1:numel(modes)
    cur_mode = modes{m_idx};
    
    switch cur_mode
        case 'combined'
            target_vals  = [1, 2, 3];
            title_prefix = 'Color-Sensitive Electrodes Across All Subjects (Pooled, Posterior to Anterior)';
            out_name     = 'all_electrodes_pooled_strip_p2a_combined.png';
            leg_items = {
                cfg.col_pos,   sprintf('Concordant Positive (n=%d)', tot_pos);
                cfg.col_neg,   sprintf('Concordant Negative (n=%d)', tot_neg);
                cfg.col_bias,  sprintf('Category Biased (n=%d)', tot_bias);
                cfg.col_gray,  sprintf('All Electrode Sites (n=%d)', n_total)
            };
            leg_x = [6, 29, 53, 76];
            
        case 'positive'
            target_vals  = [1];
            title_prefix = 'Concordant Positive Color Electrodes (Pooled, Posterior to Anterior)';
            out_name     = 'all_electrodes_pooled_strip_p2a_positive.png';
            leg_items = {
                cfg.col_pos,   sprintf('Concordant Positive (n=%d)', tot_pos);
                cfg.col_gray,  sprintf('All Electrode Sites (n=%d)', n_total)
            };
            leg_x = [25, 55];
            
        case 'negative'
            target_vals  = [2];
            title_prefix = 'Concordant Negative Color Electrodes (Pooled, Posterior to Anterior)';
            out_name     = 'all_electrodes_pooled_strip_p2a_negative.png';
            leg_items = {
                cfg.col_neg,   sprintf('Concordant Negative (n=%d)', tot_neg);
                cfg.col_gray,  sprintf('All Electrode Sites (n=%d)', n_total)
            };
            leg_x = [25, 55];
            
        case 'biased'
            target_vals  = [3];
            title_prefix = 'Category Biased Color Electrodes (Pooled, Posterior to Anterior)';
            out_name     = 'all_electrodes_pooled_strip_p2a_biased.png';
            leg_items = {
                cfg.col_bias,  sprintf('Category Biased (n=%d)', tot_bias);
                cfg.col_gray,  sprintf('All Electrode Sites (n=%d)', n_total)
            };
            leg_x = [25, 55];
    end
    
    % 创建画布
    fig = figure('Color', 'w', 'Position', [50, 250, cfg.fig_size(1), cfg.fig_size(2)], 'Visible', 'off');
    ax  = axes('Position', [0.07, 0.22, 0.91, 0.60]);
    hold(ax, 'on');
    
    % (A) 底框 (纯白底色，浅灰边框)
    rectangle('Position', [0.5, 0.5, n_total, n_bands], ...
              'FaceColor', 'w', 'EdgeColor', [0.85, 0.85, 0.88], 'LineWidth', 0.8, 'Parent', ax);
    
    % (B) 顶部实心灰色条: 代表全部 793 个电极位点，不分被试不间断
    rectangle('Position', [0.5, -0.65, n_total, 0.50], ...
              'FaceColor', cfg.col_gray, 'EdgeColor', 'none', 'Parent', ax);
    
    % 顶部左右方向指示
    text(1, -0.90, '\leftarrow Posterior (Occipital)', ...
         'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', ...
         'FontWeight', 'bold', 'FontSize', 10, 'Color', [0.30, 0.30, 0.35], 'Parent', ax);
    text(n_total, -0.90, 'Anterior (Frontal) \rightarrow', ...
         'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
         'FontWeight', 'bold', 'FontSize', 10, 'Color', [0.30, 0.30, 0.35], 'Parent', ax);
    
    % (C) 频段间细分隔线
    for b = 1:n_bands-1
        line([0.5, n_total + 0.5], [b + 0.5, b + 0.5], ...
             'Color', [0.93, 0.93, 0.95], 'LineWidth', 0.5, 'Parent', ax);
    end
    
    % (D) 绘制显著色块 (无边框紧凑贴合 EdgeColor = 'none')
    for b = 1:n_bands
        for i = 1:n_total
            val = sorted_sig(b, i);
            if ismember(val, target_vals)
                switch val
                    case 1, cur_col = cfg.col_pos;
                    case 2, cur_col = cfg.col_neg;
                    case 3, cur_col = cfg.col_bias;
                end
                patch([i-0.5, i+0.5, i+0.5, i-0.5], ...
                      [b-0.5, b-0.5, b+0.5, b+0.5], ...
                      cur_col, 'EdgeColor', 'none', 'Parent', ax);
            end
        end
    end
    
    % 坐标轴设置
    set(ax, 'XLim', [0, n_total + 1], 'YLim', [-1.4, n_bands + 0.5]);
    set(ax, 'YDir', 'reverse'); % 顶端为 High Gamma，底端为 Delta
    set(ax, 'YTick', 1:n_bands, 'YTickLabel', band_labels, ...
            'FontSize', 9.5, 'FontWeight', 'bold', 'TickLength', [0, 0]);
    
    % X 轴无具体电极文本，显示 MNI Y 坐标刻度及解剖梯度
    set(ax, 'XTick', x_ticks, 'XTickLabel', x_labels, ...
            'FontSize', 9.5, 'FontWeight', 'normal', 'TickLength', [0.005, 0.01]);
    xlabel(ax, 'Posterior \rightarrow Anterior Gradient (MNI Y coordinate)', ...
           'FontSize', 10.5, 'FontWeight', 'bold', 'Color', [0.20, 0.20, 0.25]);
    set(ax, 'Box', 'off');
    
    % 极简英文标题
    title_str = sprintf('\\fontsize{12.5}{\\bf %s}   \\fontsize{9.5}\\color[rgb]{0.35,0.35,0.35}(Total: %d electrodes pooled across 8 subjects)', ...
                        title_prefix, n_total);
    title(ax, title_str, 'Interpreter', 'tex', 'Position', [n_total / 2, -1.55, 0]);
    
    % 极简英文图例
    leg_ax = axes('Position', [0.15, 0.03, 0.70, 0.08]);
    hold(leg_ax, 'on');
    set(leg_ax, 'XLim', [0, 100], 'YLim', [0, 1], 'Visible', 'off');
    
    for li = 1:size(leg_items, 1)
        col = leg_items{li, 1};
        txt = leg_items{li, 2};
        patch([leg_x(li), leg_x(li)+2.8, leg_x(li)+2.8, leg_x(li)], [0.2, 0.2, 0.8, 0.8], ...
              col, 'EdgeColor', 'none', 'Parent', leg_ax);
        text(leg_x(li) + 3.8, 0.5, txt, 'FontSize', 9.5, 'VerticalAlignment', 'middle', ...
             'FontWeight', 'normal', 'Parent', leg_ax);
    end
    
    % 导出图像
    out_file = fullfile(out_fig_dir, out_name);
    if isfile(out_file), try delete(out_file); catch, end; end
    exportgraphics(fig, out_file, 'Resolution', cfg.dpi);
    fprintf('[+] 已生成图像: %s\n', out_name);
    
    % 复制到 metadata/figures/
    meta_file = fullfile(meta_fig_dir, out_name);
    if isfile(meta_file), try delete(meta_file); catch, end; end
    copyfile(out_file, meta_file);
    
    close(fig);
end

fprintf('\n[+] 全电极不分被试后-前全景图绘制完成！\n');
