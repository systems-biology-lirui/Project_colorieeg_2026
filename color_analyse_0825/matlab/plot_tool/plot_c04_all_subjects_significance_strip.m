%% plot_c04_all_subjects_significance_strip.m
% =========================================================================
% Function: Plot significant color electrodes across all subjects (8 subjects)
% Outputs:
%   1. Combined figure: All 3 significance types (Positive, Negative, Biased)
%   2. Positive-only figure: Concordant Positive electrodes (Red)
%   3. Negative-only figure: Concordant Negative electrodes (Blue)
%   4. Biased-only figure: Category Biased electrodes (Orange)
%
% Visual Design:
%   - Horizontal axis: All subjects arranged from left to right (sub001 - sub008)
%   - Gray bar above each subject: Represents all electrode sites of that subject
%   - No individual electrode text on X-axis (clean subject labels only)
%   - Vertical axis: 6 frequency bands from High Gamma down to Delta
%   - Color fill: Borderless solid patches (EdgeColor = 'none')
%   - Simple, concise English title, legend, and axis labels.
% =========================================================================

clear; clc; close all;

%% 1. Configuration (Simple, intuitive, adjustable)
cfg = struct();
cfg.subjects    = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008'};
cfg.alpha_sig   = 0.05;              % Significance threshold
cfg.gap         = 6;                 % Spacing between consecutive subjects (in electrode units)
cfg.fig_size    = [2000, 520];       % Figure size [Width, Height]
cfg.dpi         = 300;               % Resolution for export

% 电极排序模式:
%   'p2a'     : 按 MNI Y 坐标从后往前排列 (Posterior -> Anterior, 左至右)
%   'default' : 原始通道编号顺序 (A1..A10, B1..B10 等)
cfg.sort_mode   = 'p2a';

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

%% 4. Load electrode channels and build coordinates for all subjects
sub_data = struct();
cur_x    = 1;

for s = 1:numel(cfg.subjects)
    s_id = cfg.subjects{s};
    mat_f = fullfile(proc_new, s_id, 'task1_multiband_epoched.mat');
    if ~isfile(mat_f)
        error('Feature cache not found for %s: %s', s_id, mat_f);
    end
    ep_obj = load(mat_f);
    chs = ep_obj.epoched_data.triplet_info.center_channel;
    n_chs = numel(chs);
    
    % 若开启从后往前排序 (Posterior -> Anterior)
    if strcmp(cfg.sort_mode, 'p2a')
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
        
        % 同轴电极 fallback: 末梢触点若未记录坐标，使用同轴已知触点平均 Y
        for i = 1:n_chs
            if isnan(mni_y(i))
                shaft = regexprep(chs{i}, '\d+', '');
                same_shaft = find(strncmp(chs, shaft, length(shaft)) & ~isnan(mni_y));
                if ~isempty(same_shaft)
                    mni_y(i) = mean(mni_y(same_shaft));
                end
            end
        end
        
        % 未定位轴 (如 sub004 E~L 轴) 赋予 +100，排在最前端
        mni_y(isnan(mni_y)) = 100;
        
        % 按 MNI Y 升序排列 (从后往前 Posterior -> Anterior)
        [mni_y_sorted, sort_idx] = sort(mni_y, 'ascend');
        chs = chs(sort_idx);
        sub_data(s).mni_y = mni_y_sorted;
    end
    
    % Store subject information
    sub_data(s).id      = s_id;
    sub_data(s).chs     = chs;
    sub_data(s).n_ch    = n_chs;
    sub_data(s).x_start = cur_x;
    sub_data(s).x_end   = cur_x + n_chs - 1;
    sub_data(s).x_mid   = (sub_data(s).x_start + sub_data(s).x_end) / 2;
    
    % Build significance matrix for this subject (n_bands x n_chs)
    s_tbl = c04_tbl(strcmp(c04_tbl.subject, s_id), :);
    sig_mat = zeros(n_bands, n_chs);
    
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
                        sig_mat(b, ch_idx) = 1;
                    elseif strcmp(c_type, 'Concordant_Negative')
                        sig_mat(b, ch_idx) = 2;
                    else
                        sig_mat(b, ch_idx) = 3;
                    end
                end
            end
        end
    end
    sub_data(s).sig_mat = sig_mat;
    
    % Advance x coordinate for next subject (including gap)
    cur_x = sub_data(s).x_end + cfg.gap + 1;
end

total_x_span = cur_x - cfg.gap - 1;

% Count overall totals
tot_elecs = 0;
tot_pos   = 0;
tot_neg   = 0;
tot_bias  = 0;
for s = 1:numel(cfg.subjects)
    tot_elecs = tot_elecs + sub_data(s).n_ch;
    tot_pos   = tot_pos + sum(sub_data(s).sig_mat(:) == 1);
    tot_neg   = tot_neg + sum(sub_data(s).sig_mat(:) == 2);
    tot_bias  = tot_bias + sum(sub_data(s).sig_mat(:) == 3);
end

%% 5. 循环绘制 4 张图: 1 张合一图 + 3 张分类型独立图
modes = {'combined', 'positive', 'negative', 'biased'};

if strcmp(cfg.sort_mode, 'p2a')
    file_tag  = '_p2a';
    sort_desc = ' (Posterior to Anterior)';
else
    file_tag  = '';
    sort_desc = '';
end

for m_idx = 1:numel(modes)
    cur_mode = modes{m_idx};
    
    switch cur_mode
        case 'combined'
            target_vals  = [1, 2, 3];
            title_prefix = sprintf('Color-Sensitive Electrodes Across All Subjects%s (Combined)', sort_desc);
            out_name     = sprintf('all_subjects_channel_significance_strip%s_combined.png', file_tag);
            leg_items = {
                cfg.col_pos,   sprintf('Concordant Positive (n=%d)', tot_pos);
                cfg.col_neg,   sprintf('Concordant Negative (n=%d)', tot_neg);
                cfg.col_bias,  sprintf('Category Biased (n=%d)', tot_bias);
                cfg.col_gray,  sprintf('All Electrode Sites (n=%d)', tot_elecs)
            };
            leg_x = [6, 29, 53, 76];
            
        case 'positive'
            target_vals  = [1];
            title_prefix = sprintf('Concordant Positive Color Electrodes Across All Subjects%s', sort_desc);
            out_name     = sprintf('all_subjects_channel_significance_strip%s_positive.png', file_tag);
            leg_items = {
                cfg.col_pos,   sprintf('Concordant Positive (n=%d)', tot_pos);
                cfg.col_gray,  sprintf('All Electrode Sites (n=%d)', tot_elecs)
            };
            leg_x = [25, 55];
            
        case 'negative'
            target_vals  = [2];
            title_prefix = sprintf('Concordant Negative Color Electrodes Across All Subjects%s', sort_desc);
            out_name     = sprintf('all_subjects_channel_significance_strip%s_negative.png', file_tag);
            leg_items = {
                cfg.col_neg,   sprintf('Concordant Negative (n=%d)', tot_neg);
                cfg.col_gray,  sprintf('All Electrode Sites (n=%d)', tot_elecs)
            };
            leg_x = [25, 55];
            
        case 'biased'
            target_vals  = [3];
            title_prefix = sprintf('Category Biased Color Electrodes Across All Subjects%s', sort_desc);
            out_name     = sprintf('all_subjects_channel_significance_strip%s_biased.png', file_tag);
            leg_items = {
                cfg.col_bias,  sprintf('Category Biased (n=%d)', tot_bias);
                cfg.col_gray,  sprintf('All Electrode Sites (n=%d)', tot_elecs)
            };
            leg_x = [25, 55];
    end
    
    % 创建画布
    fig = figure('Color', 'w', 'Position', [50, 250, cfg.fig_size(1), cfg.fig_size(2)], 'Visible', 'off');
    ax  = axes('Position', [0.07, 0.22, 0.91, 0.60]);
    hold(ax, 'on');
    
    for s = 1:numel(cfg.subjects)
        n_ch_s  = sub_data(s).n_ch;
        x_st    = sub_data(s).x_start;
        x_ed    = sub_data(s).x_end;
        x_md    = sub_data(s).x_mid;
        s_mat   = sub_data(s).sig_mat;
        
        % (A) 绘制被试底框 (淡灰轮廓，纯白底色)
        rectangle('Position', [x_st - 0.5, 0.5, n_ch_s, n_bands], ...
                  'FaceColor', 'w', 'EdgeColor', [0.85, 0.85, 0.88], 'LineWidth', 0.8, 'Parent', ax);
        
        % (B) 顶部实心灰色条: 严格代表该被试的所有电极位点
        rectangle('Position', [x_st - 0.5, -0.65, n_ch_s, 0.50], ...
                  'FaceColor', cfg.col_gray, 'EdgeColor', 'none', 'Parent', ax);
        
        % 被试标签
        text(x_md, -0.90, sprintf('%s (n=%d)', sub_data(s).id, n_ch_s), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontWeight', 'bold', 'FontSize', 10, 'Color', [0.20, 0.20, 0.25], 'Parent', ax);
        
        % (C) 频段间的细分隔线
        for b = 1:n_bands-1
            line([x_st - 0.5, x_ed + 0.5], [b + 0.5, b + 0.5], ...
                 'Color', [0.93, 0.93, 0.95], 'LineWidth', 0.5, 'Parent', ax);
        end
        
        % (D) 绘制显著色块 (无边框贴合填充 EdgeColor = 'none')
        for b = 1:n_bands
            for ch_idx = 1:n_ch_s
                val = s_mat(b, ch_idx);
                if ismember(val, target_vals)
                    switch val
                        case 1, cur_col = cfg.col_pos;
                        case 2, cur_col = cfg.col_neg;
                        case 3, cur_col = cfg.col_bias;
                    end
                    x_pos = x_st + ch_idx - 1;
                    patch([x_pos-0.5, x_pos+0.5, x_pos+0.5, x_pos-0.5], ...
                          [b-0.5, b-0.5, b+0.5, b+0.5], ...
                          cur_col, 'EdgeColor', 'none', 'Parent', ax);
                end
            end
        end
    end
    
    % 坐标轴设置
    set(ax, 'XLim', [0, total_x_span + 1], 'YLim', [-1.4, n_bands + 0.5]);
    set(ax, 'YDir', 'reverse'); % 顶端为 High Gamma，底端为 Delta
    set(ax, 'YTick', 1:n_bands, 'YTickLabel', band_labels, ...
            'FontSize', 9.5, 'FontWeight', 'bold', 'TickLength', [0, 0]);
    
    % X 轴无具体电极文本，仅标注被试
    set(ax, 'XTick', [sub_data.x_mid], 'XTickLabel', {sub_data.id}, ...
            'FontSize', 10, 'FontWeight', 'bold', 'TickLength', [0, 0]);
    set(ax, 'Box', 'off');
    
    % 极简英文标题
    if strcmp(cfg.sort_mode, 'p2a')
        title_str = sprintf('\\fontsize{12.5}{\\bf %s}   \\fontsize{9.5}\\color[rgb]{0.35,0.35,0.35}(Total: %d electrodes, 8 subjects | Left to Right: Posterior \\rightarrow Anterior)', ...
                            title_prefix, tot_elecs);
    else
        title_str = sprintf('\\fontsize{12.5}{\\bf %s}   \\fontsize{9.5}\\color[rgb]{0.35,0.35,0.35}(Total: %d electrodes, 8 subjects)', ...
                            title_prefix, tot_elecs);
    end
    title(ax, title_str, 'Interpreter', 'tex', 'Position', [total_x_span / 2, -1.55, 0]);
    
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
    
    % 若为 combined 且非 p2a 则覆盖原通用命名文件
    if strcmp(cur_mode, 'combined') && ~strcmp(cfg.sort_mode, 'p2a')
        comb_legacy = fullfile(out_fig_dir, 'all_subjects_channel_significance_strip.png');
        if isfile(comb_legacy), try delete(comb_legacy); catch, end; end
        copyfile(out_file, comb_legacy);
    elseif strcmp(cur_mode, 'combined') && strcmp(cfg.sort_mode, 'p2a')
        comb_legacy = fullfile(out_fig_dir, 'all_subjects_channel_significance_strip_p2a.png');
        if isfile(comb_legacy), try delete(comb_legacy); catch, end; end
        copyfile(out_file, comb_legacy);
    end
    
    close(fig);
end

fprintf('\n[+] 全部 4 张全景图绘制完成！\n');
