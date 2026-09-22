function plot_cluster_spatiotemporal_distribution(target_mode)
%% ========================================================================
% 脚本名称: plot_cluster_spatiotemporal_distribution.m
% 功能:
%   1. 【时空分布多子图映射 (MNI X / Y / Z 各一)】
%      纵轴: 刺激呈现后显著时程与峰值时间 (ms, 0~800 ms)
%      横轴: MNI 空间立体定向坐标 (mm, X:左右, Y:前后, Z:上下)
%   2. 【频段颜色编码】
%      不同色彩标示最优解码单频段 (Delta, Theta, Alpha, Beta, Low-Gamma, High-Gamma)
%   3. 【Sigmoid 非线性映射点大小】
%      以 Sigmoid (sigma) 函数将解码峰值正确率映射为散点面积
%   4. 【时程跨度淡色线条 (更淡更柔和)】
%      采用柔和半透明色柱 (dur_alpha = 0.30, dur_width = 2.4) 竖向绘制显著时间跨度，
%      使背景线条更淡，突出前景峰值散点
%   5. 【线性回归曲线与显著性检验】
%      绘制深灰虚线回归拟合曲线，并标注 Pearson 相关系数 r 与 p 值
%   6. 【极简纯净排版 (无点标签，无网格)】
%      去除散点文字标注与背景方格，图例仅保留频段点与尺寸标尺
%   7. 【支持三种模式出图】
%      - 'concordant': 总体显著且同向电极 (N = 11)
%      - 'non_concordant': 总体显著但不同向电极 (N = 5)
%      - 'all_significant': 总体显著不论同向与否的所有电极 (N = 16)
% ========================================================================

if nargin < 1 || isempty(target_mode)
    % 默认顺序执行全部三种模式: 同向、不同向、总体全部显著
    modes_to_run = {'concordant', 'non_concordant', 'all_significant'};
else
    modes_to_run = {target_mode};
end

for m_idx = 1:numel(modes_to_run)
    curr_mode = modes_to_run{m_idx};
    fprintf('\n========================================================================\n');
    fprintf('>>> 正在生成 [%s] 模式的时空多子图分布图 ...\n', curr_mode);
    fprintf('========================================================================\n');
    render_spatiotemporal_plot(curr_mode);
end

end

function render_spatiotemporal_plot(data_mode)
%% 1. 参数配置 (平铺直观，参数置顶易调)
cfg = struct();
cfg.t_lim        = [0, 800];                     % 纵轴时间范围 (刺激呈现后 0~800 ms)
cfg.y_step       = 100;                          % 纵轴时间刻度步长 (ms)

% 散点文字标注控制 (按用户要求: 彻底去除点文字标注，保持画面纯净整洁)
cfg.show_labels  = false;

% 显著时程柱透明度 (按用户明确指示: 线条更淡一些，半透明背景色柱，突出前景散点)
cfg.dur_alpha    = 0.30;                         % 显著时程线段更淡更柔和 (0.30)
cfg.dur_width    = 2.4;                          % 显著时程线宽度
cfg.corr_color   = [0.20, 0.20, 0.20];           % 相关性拟合线颜色
cfg.corr_width   = 2.0;                          % 相关性拟合线宽度

% 峰值指标配置:
%   'joint'       : 散点位置与大小对应显著簇内的【多频段联合解码峰值】(推荐，统计检验基础)
%   'single_band' : 散点位置与大小对应显著簇内的【最优单频段解码峰值】
cfg.peak_metric  = 'joint';

% Sigmoid 点大小映射参数: Size = S_min + (S_max - S_min) / (1 + exp(-k * (acc - acc_mid)))
cfg.acc_mid      = 0.57;                         % Sigmoid 中点正确率 (57%)
cfg.acc_k        = 35;                           % Sigmoid 增长斜率
cfg.size_min     = 70;                           % 最小点大小
cfg.size_max     = 360;                          % 最大点大小

% 频段配色字典 (Nature 规范学术配色)
cfg.bands        = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low-Gamma', 'High-Gamma'};
cfg.band_keys    = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
cfg.band_colors  = [
    0.45, 0.45, 0.45;  % Delta: 石板灰
    0.95, 0.60, 0.15;  % Theta: 琥珀橙
    0.15, 0.45, 0.80;  % Alpha: 经典蓝
    0.10, 0.65, 0.45;  % Beta: 翡翠绿
    0.50, 0.30, 0.75;  % Low-Gamma: 罗兰紫
    0.90, 0.15, 0.45   % High-Gamma: 玫瑰红
];

% 路径配置
script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(fileparts(script_dir)));
res_root   = fullfile(proj_root, 'color_analyse_0825', 'result');
c04_file   = fullfile(res_root, 'tables', 'color_effects_summary.mat');
loc_dir    = fullfile(proj_root, 'color_analyse_0825', 'metadata', 'ieeg_location');
out_fig_dir= fullfile(res_root, 'figures', 'spatiotemporal');

if ~exist(out_fig_dir, 'dir'), mkdir(out_fig_dir); end

switch data_mode
    case 'non_concordant'
        tab_files    = {fullfile(res_root, 'tables', 'non_concordant_electrodes_decoding_summary.mat')};
        mode_field   = 'non_concordant';
        out_png_name = 'cluster_spatiotemporal_distribution_non_concordant_XYZ.png';
        title_suffix = ' (Non-Concordant)';
        only_sig_cluster = true;
    case 'concordant'
        tab_files    = {fullfile(res_root, 'tables', 'concordant_electrodes_decoding_summary.mat')};
        mode_field   = 'concordant';
        out_png_name = 'cluster_spatiotemporal_distribution_XYZ.png';
        title_suffix = ' (Concordant)';
        only_sig_cluster = true;
    case {'all_significant', 'overall'}
        tab_files    = {
            fullfile(res_root, 'tables', 'concordant_electrodes_decoding_summary.mat'), ...
            fullfile(res_root, 'tables', 'non_concordant_electrodes_decoding_summary.mat')
        };
        mode_field   = 'all_significant';
        out_png_name = 'cluster_spatiotemporal_distribution_all_significant_XYZ.png';
        title_suffix = ' (Overall Significant)';
        only_sig_cluster = true;
    case {'task3', 'task3_purecolor'}
        tab_files    = {fullfile(res_root, 'tables', 'task3_purecolor_decoding_summary.mat')};
        mode_field   = 'task3_purecolor';
        out_png_name = 'cluster_spatiotemporal_distribution_task3_XYZ.png';
        title_suffix = ' (Task 3 Pure Color)';
        only_sig_cluster = true;
    case {'task3_all'}
        tab_files    = {fullfile(res_root, 'tables', 'task3_purecolor_decoding_summary.mat')};
        mode_field   = 'task3_purecolor';
        out_png_name = 'cluster_spatiotemporal_distribution_task3_all_XYZ.png';
        title_suffix = ' (Task 3 Pure Color All)';
        only_sig_cluster = false;
    otherwise
        error('未知的数据模式: %s', data_mode);
end

%% 2. 加载数据并提取聚类显著位点及其 MNI 坐标与时程
t_sum_list = {};
for tf_i = 1:numel(tab_files)
    if isfile(tab_files{tf_i})
        m_load = load(tab_files{tf_i});
        if isfield(m_load, 'summary_table')
            t_sum_list{end+1} = m_load.summary_table; %#ok<AGROW>
        elseif isfield(m_load, 'master_tbl')
            t_sum_list{end+1} = m_load.master_tbl; %#ok<AGROW>
        end
    else
        warning('未找到解码汇总表: %s', tab_files{tf_i});
    end
end

if isempty(t_sum_list)
    warning('[%s] 未加载到任何有效汇总表，跳过。', data_mode);
    return;
end

t_sum = vertcat(t_sum_list{:});

% 筛选通过聚类检验的电极 (若 only_sig_cluster=false 则全量分析)
if only_sig_cluster
    t_sig = t_sum(t_sum.has_sig_cluster == 1, :);
    fprintf('[+] [%s] 共有 %d 个具有显著时间簇的核心电极。\n', data_mode, height(t_sig));
else
    t_sig = t_sum;
    fprintf('[+] [%s] 全量模式，共有 %d 个目标电极。\n', data_mode, height(t_sig));
end
n_elecs = height(t_sig);

if n_elecs == 0
    warning('[%s] 未检测到任何聚类显著电极，不生成图表。', data_mode);
    return;
end

% 读取 C04 坐标表用于匹配
t_c04_data = load(c04_file);
if isfield(t_c04_data, 'all_tbl')
    t_c04 = t_c04_data.all_tbl;
else
    t_c04 = t_c04_data.res_table;
end

% 预加载单通道时程结构体
tc_mat_file = fullfile(res_root, 'tables', 'single_channel_decoding_timecourses.mat');
if ~isfile(tc_mat_file)
    error('未找到单通道时程汇总文件: %s', tc_mat_file);
end
tc_store = load(tc_mat_file);

if strcmp(mode_field, 'all_significant')
    tc_entries = [tc_store.concordant; tc_store.non_concordant];
else
    tc_entries = tc_store.(mode_field);
end
tc_keys = {tc_entries.key};
tc_time_ms = tc_store.time_ms;

records = struct([]);

for i = 1:n_elecs
    sub = char(t_sig.subject{i});
    ch  = char(t_sig.channel{i});
    
    % 1. 获取 MNI 坐标
    mx = NaN; my = NaN; mz = NaN;
    m_c04 = strcmp(string(t_c04.subject), string(sub)) & strcmp(string(t_c04.channel), string(ch));
    idx_c04 = find(m_c04, 1);
    if ~isempty(idx_c04) && ~isnan(t_c04.mni_x(idx_c04))
        mx = t_c04.mni_x(idx_c04);
        my = t_c04.mni_y(idx_c04);
        mz = t_c04.mni_z(idx_c04);
    end
    
    % 若 C04 为 NaN，从 ieegloc.xlsx 或 .tsv 提取 (专门适配 sub008 的 [X,Y,Z] 字符串格式)
    if isnan(mx)
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
                        mx = nums(1); my = nums(2); mz = nums(3);
                    end
                elseif width(t_loc) >= 4
                    vals = table2array(t_loc(c_idx, 2:4));
                    if isnumeric(vals) && ~any(isnan(vals))
                        mx = vals(1); my = vals(2); mz = vals(3);
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
                    mx = nums(1); my = nums(2); mz = nums(3);
                end
            end
        end
    end
    
    % 2. 获取逐通道时程数据 (从已汇总结构体直接极速读取)
    ch_key = sprintf('%s_%s', sub, ch);
    tc_idx = find(strcmp(tc_keys, ch_key), 1);
    
    t_start = NaN;
    t_end   = NaN;
    p_time  = NaN;
    p_acc   = NaN;
    best_b  = 1;
    
    if ~isempty(tc_idx)
        entry = tc_entries(tc_idx);
        sig_mask = (entry.p_pointwise < 0.05) & (tc_time_ms >= 0);
        sig_times = tc_time_ms(sig_mask);
        
        if ~isempty(sig_times)
            % 显著时间跨度 (多频段联合置换检验显著时程)
            t_start = min(sig_times);
            t_end   = max(sig_times);
            
            % 严格限定在显著簇时间窗内提取峰值指标
            c_mask  = (tc_time_ms >= t_start) & (tc_time_ms <= t_end);
            c_times = tc_time_ms(c_mask);
            
            % (1) 簇内多频段峰值 (Joint)
            [j_acc, j_idx] = max(entry.acc_joint(c_mask));
            j_time = c_times(j_idx);
            
            % (2) 簇内各单频段峰值与表现最佳单频段 (Single-Band)
            b_accs = zeros(numel(cfg.bands), 1);
            b_times = zeros(numel(cfg.bands), 1);
            for b = 1:numel(cfg.bands)
                fld_name = sprintf('acc_%s', cfg.band_keys{b});
                if isfield(entry, fld_name) && ~isempty(entry.(fld_name))
                    vals = entry.(fld_name);
                    [b_accs(b), b_idx] = max(vals(c_mask));
                    b_times(b) = c_times(b_idx);
                end
            end
            [sb_acc, best_b] = max(b_accs);
            sb_time = b_times(best_b);
            
            % 根据 cfg.peak_metric 决定散点纵轴时间与散点大小正确率
            if strcmp(cfg.peak_metric, 'joint')
                p_time = j_time;
                p_acc  = j_acc;
            else
                p_time = sb_time;
                p_acc  = sb_acc;
            end
        end
    end
    
    % 若无显著时程 (如全量模式中无显著簇电极)，退化为全时间窗峰值
    if isnan(p_time)
        p_time = max(0, t_sig.peak_time_ms(i));
        if strcmp(cfg.peak_metric, 'joint')
            p_acc = t_sig.peak_acc_joint(i);
        else
            p_acc = t_sig.best_single_band_acc(i);
        end
        t_start = p_time;
        t_end   = p_time;
        
        raw_band = char(t_sig.best_single_band{i});
        clean_band = strrep(raw_band, '_', '-');
        b_idx = find(strcmp(cfg.bands, clean_band), 1);
        if ~isempty(b_idx), best_b = b_idx; end
    end
    
    % 4. 计算 Sigmoid 映射点大小: Size = S_min + (S_max - S_min) / (1 + exp(-k*(acc - mid)))
    sigm_val = 1 / (1 + exp(-cfg.acc_k * (p_acc - cfg.acc_mid)));
    pt_size  = cfg.size_min + (cfg.size_max - cfg.size_min) * sigm_val;
    
    rec = struct();
    rec.subject     = string(sub);
    rec.channel     = string(ch);
    rec.label       = string(sprintf('%s-%s', sub, ch));
    rec.mni_x       = mx;
    rec.mni_y       = my;
    rec.mni_z       = mz;
    rec.t_start     = max(0, t_start);
    rec.t_end       = min(cfg.t_lim(2), t_end);
    rec.peak_time   = max(0, min(cfg.t_lim(2), p_time));
    rec.peak_acc    = p_acc;
    rec.pt_size     = pt_size;
    rec.band_name   = cfg.bands{best_b};
    rec.band_color  = cfg.band_colors(best_b, :);
    rec.has_coords  = ~isnan(mx) && ~isnan(my) && ~isnan(mz);
    
    records = [records; rec]; %#ok<AGROW>
end

% 仅保留具备有效 MNI 坐标的位点用于空间绘图
coord_mask = [records.has_coords];
plot_recs  = records(coord_mask);
n_plotted  = numel(plot_recs);
fprintf('[+] 共有 %d 个位点具备有效三维 MNI 坐标，进入空间映射绘图。\n', n_plotted);

if n_plotted == 0
    warning('[%s] 没有具备有效坐标的电极，跳过绘图。', data_mode);
    return;
end

%% 3. 创建宽屏 Nature 级三子图
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80, 80, 1500, 640]);

coord_fields  = {'mni_x', 'mni_y', 'mni_z'};
coord_xlabels = {
    sprintf('MNI X (mm)\n[Left < 0 < Right]'), ...
    sprintf('MNI Y (mm)\n[Posterior < 0 < Anterior]'), ...
    sprintf('MNI Z (mm)\n[Inferior < 0 < Superior]')
};

min_x = min([plot_recs.mni_x]); max_x = max([plot_recs.mni_x]);
min_y = min([plot_recs.mni_y]); max_y = max([plot_recs.mni_y]);
min_z = min([plot_recs.mni_z]); max_z = max([plot_recs.mni_z]);

coord_xlimits = {
    [min(-45, min_x - 10), max(65, max_x + 10)], ...
    [min(-95, min_y - 10), max(5,  max_y + 10)], ...
    [min(-35, min_z - 10), max(65, max_z + 10)]
};

sub_pos = {
    [0.06, 0.24, 0.28, 0.66], ...
    [0.38, 0.24, 0.28, 0.66], ...
    [0.70, 0.24, 0.28, 0.66]
};

for p_i = 1:3
    field_name = coord_fields{p_i};
    ax = axes('Position', sub_pos{p_i});
    hold(ax, 'on');
    
    % 1. 设置无方格、纯净背景
    grid(ax, 'off');
    set(ax, 'Box', 'off', 'FontSize', 12, 'LineWidth', 1.2, ...
        'XColor', [0.15, 0.15, 0.15], 'YColor', [0.15, 0.15, 0.15]);
    
    xlim(ax, coord_xlimits{p_i});
    ylim(ax, cfg.t_lim);
    set(ax, 'YTick', cfg.t_lim(1):cfg.y_step:cfg.t_lim(2));
    
    xlabel(ax, coord_xlabels{p_i}, 'FontSize', 12.5, 'FontWeight', 'bold');
    if p_i == 1
        ylabel(ax, 'Significant Time from Stimulus Onset (ms)', 'FontSize', 13, 'FontWeight', 'bold');
    else
        set(ax, 'YTickLabel', []);
    end
    
    % 2. 绘制时程跨度淡色线条 (按指示: 更淡更柔和 dur_alpha = 0.30, dur_width = 2.4)
    for k = 1:numel(plot_recs)
        r = plot_recs(k);
        c_val = r.(field_name);
        line(ax, [c_val, c_val], [r.t_start, r.t_end], ...
            'Color', [r.band_color, cfg.dur_alpha], 'LineWidth', cfg.dur_width, 'HandleVisibility', 'off');
    end
    
    % 3. 拟合相关性曲线并标注显著性
    x_all = [plot_recs.(field_name)]';
    y_all = [plot_recs.peak_time]';
    
    if numel(x_all) >= 3 && std(x_all) > 1e-4
        [r_val, p_val] = corr(x_all, y_all);
        p_fit  = polyfit(x_all, y_all, 1);
        x_grid = linspace(coord_xlimits{p_i}(1), coord_xlimits{p_i}(2), 200);
        y_grid = polyval(p_fit, x_grid);
        
        valid_fit = (y_grid >= cfg.t_lim(1)) & (y_grid <= cfg.t_lim(2));
        plot(ax, x_grid(valid_fit), y_grid(valid_fit), '--', ...
            'Color', cfg.corr_color, 'LineWidth', cfg.corr_width, 'HandleVisibility', 'off');
        
        if p_val < 0.001
            sig_str = '***';
        elseif p_val < 0.01
            sig_str = '**';
        elseif p_val < 0.05
            sig_str = '*';
        elseif p_val < 0.10
            sig_str = ' (trend)';
        else
            sig_str = ' (n.s.)';
        end
        
        stat_text = sprintf('r = %.2f, p = %.3f%s', r_val, p_val, sig_str);
        x_span = coord_xlimits{p_i}(2) - coord_xlimits{p_i}(1);
        
        box_x = coord_xlimits{p_i}(1) + x_span * 0.06;
        box_y = cfg.t_lim(2) - (cfg.t_lim(2)-cfg.t_lim(1)) * 0.08;
        ha = 'left';
        
        text(ax, box_x, box_y, stat_text, ...
            'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', ha, ...
            'BackgroundColor', [1, 1, 1, 0.85], 'EdgeColor', [0.8, 0.8, 0.8], 'LineWidth', 0.8);
    end
    
    % 4. 绘制峰值散点 (前景突出)
    for k = 1:numel(plot_recs)
        r = plot_recs(k);
        c_val = r.(field_name);
        
        scatter(ax, c_val, r.peak_time, r.pt_size, ...
            'MarkerFaceColor', r.band_color, ...
            'MarkerEdgeColor', [0.15, 0.15, 0.15], ...
            'MarkerFaceAlpha', 0.95, ...
            'LineWidth', 1.4, ...
            'HandleVisibility', 'off');
    end
end

%% 4. 添加底部规范学术图例与 Sigmoid 尺寸标尺
% (A) 频段颜色图例 (仅保留色点与频段名)
dummy_ax = axes('Position', [0.07, 0.04, 0.55, 0.08], 'Visible', 'off');
hold(dummy_ax, 'on');

h_dummy = zeros(numel(cfg.bands), 1);
for b = 1:numel(cfg.bands)
    h_dummy(b) = scatter(dummy_ax, NaN, NaN, 140, ...
        'MarkerFaceColor', cfg.band_colors(b, :), ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.1, ...
        'DisplayName', cfg.bands{b});
end

legend(dummy_ax, h_dummy, cfg.bands, ...
    'Orientation', 'horizontal', 'Box', 'off', 'FontSize', 11, 'Location', 'west');

% (B) Sigmoid 点大小尺寸标尺 (Size Scale)
scale_ax = axes('Position', [0.66, 0.03, 0.29, 0.09], 'Visible', 'off');
hold(scale_ax, 'on');
xlim(scale_ax, [0, 10]);
ylim(scale_ax, [0, 2]);

acc_samples = [0.54, 0.57, 0.60, 0.64];
x_pos = [1.5, 3.8, 6.2, 8.8];

if strcmp(cfg.peak_metric, 'joint')
    scale_str = 'Peak Accuracy Scale (Joint): Size = \sigma(Acc)';
else
    scale_str = 'Peak Accuracy Scale (Single Band): Size = \sigma(Acc)';
end
text(scale_ax, 5.0, 1.6, scale_str, ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);

for s_i = 1:numel(acc_samples)
    a = acc_samples(s_i);
    sz = cfg.size_min + (cfg.size_max - cfg.size_min) / (1 + exp(-cfg.acc_k * (a - cfg.acc_mid)));
    scatter(scale_ax, x_pos(s_i), 0.7, sz, ...
        'MarkerFaceColor', [0.6, 0.6, 0.6], 'MarkerEdgeColor', 'k', 'LineWidth', 1.1);
    text(scale_ax, x_pos(s_i), 0.15, sprintf('%.0f%%', a*100), ...
        'HorizontalAlignment', 'center', 'FontSize', 9.5, 'FontWeight', 'bold');
end

% 5. 全局主标题
if contains(data_mode, 'task3')
    main_title_str = sprintf('Significant Pure Color (Task 3) Decoding Clusters%s (N = %d)', title_suffix, n_plotted);
else
    main_title_str = sprintf('Significant Memory Color Decoding Clusters%s (N = %d)', title_suffix, n_plotted);
end
sgtitle(main_title_str, 'FontSize', 16, 'FontWeight', 'bold');

% 保存图片 (300 DPI 印刷级)
out_png = fullfile(out_fig_dir, out_png_name);
exportgraphics(fig, out_png, 'Resolution', 300);
fprintf('[+] 高清图表已成功保存至: %s\n', out_png);
close(fig);

end
