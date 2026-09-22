%% ========================================================================
% 脚本名称: C05_batch_plot_timecourses_0825.m
% 功能:
%   1. 读取 C04 导出的色彩效应汇总表 (result/tables/color_effects_summary.csv)
%   2. 筛选出显著通道 (或四类别一致性正向增强通道)
%   3. 批量调用独立绘图函数 plot_channel_timecourse.m
%   4. 输出至分类明确的结果文件夹:
%        result/figures/timecourses/<sub_id>/<band>/
% ========================================================================

clear; clc; close all;

%% 1. 参数与路径配置
cfg = struct();
cfg.target_concordance = 'Concordant_Positive'; % 可选: 'Concordant_Positive', 'Concordant_Negative', 'All_Significant'
cfg.target_bands       = {'High_Gamma', 'Low_Gamma', 'Alpha', 'Beta', 'Theta'}; % 目标频段
cfg.max_plots_per_sub  = 20;                   % 每被试最大出图数 (防止数量过多)

% 路径配置
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'plot_tool'));
proj_root  = fileparts(fileparts(script_dir));
table_file = fullfile(proj_root, 'color_analyse_0825', 'result', 'tables', 'color_effects_summary.mat');

if ~isfile(table_file)
    error('未找到汇总表，请先运行 C04_screen_color_channels_0825.m！');
end

out_fig_base = fullfile(proj_root, 'color_analyse_0825', 'result', 'figures', 'timecourses');

fprintf('========================================================================\n');
fprintf('  【C05：显著通道学术时程与柱状图批量绘制】  \n');
fprintf('========================================================================\n');

%% 2. 读取汇总表并筛选
loaded = load(table_file);
if isfield(loaded, 'all_tbl')
    summary_table = loaded.all_tbl;
else
    summary_table = loaded.res_table;
end

% 筛选显著通道
if strcmp(cfg.target_concordance, 'All_Significant')
    sel_mask = summary_table.is_significant & ismember(summary_table.freq_band, cfg.target_bands);
else
    sel_mask = summary_table.is_significant & ...
               strcmp(summary_table.concordance_type, cfg.target_concordance) & ...
               ismember(summary_table.freq_band, cfg.target_bands);
end

target_rows = summary_table(sel_mask, :);
fprintf('[+] 共筛选出 %d 个符合条件的候选目标通道。\n\n', height(target_rows));

%% 3. 批量循环出图
unique_subs = unique(target_rows.subject);

for s = 1:numel(unique_subs)
    sub_id = unique_subs{s};
    sub_rows = target_rows(strcmp(target_rows.subject, sub_id), :);
    n_plot = min(height(sub_rows), cfg.max_plots_per_sub);
    
    fprintf('>>> 正在绘制被试 [%s]，共 %d 个通道 ...\n', sub_id, n_plot);
    
    for r = 1:n_plot
        ch_name   = char(sub_rows.channel(r));
        band_name = char(sub_rows.freq_band(r));
        
        % 输出子文件夹结构: result/figures/timecourses/<sub_id>/<band>/
        ch_save_dir = fullfile(out_fig_base, sub_id, band_name);
        
        try
            fig = plot_channel_timecourse(sub_id, ch_name, band_name, ch_save_dir, 'off');
            close(fig);
        catch ME
            warning('绘制 [%s %s %s] 失败: %s', sub_id, ch_name, band_name, ME.message);
        end
    end
end

fprintf('\n========================================================================\n');
fprintf('  【C05 批量绘图完成！图片已保存至: result/figures/timecourses/】\n');
fprintf('========================================================================\n');
