%% ========================================================================
% 脚本名称: C15_decoding_significant_venn_0825.m
% 功能:
%   1. 【汇总 5 大 Decoding 分析的全部统计显著位点】
%      - Set 1: Task 2 交叉 Decoding (C06, Leave-One-Fruit-Pair-Out)
%      - Set 2: Task 2 Direct Decoding (C13, 纯红绿 5 折)
%      - Set 3: Task 3 交叉 Decoding (C07, Leave-One-Patch-Out)
%      - Set 4: Task 3 Direct Decoding (C14, 纯红绿 5 折)
%      - Set 5: Task 2 对 Task 3 Cross Decoding (C09, 跨任务 2D TGM)
%   2. 【导出显著位点汇总至独立 .mat 与 .csv 文件】
%      保存到 result/tables/decoding_significant_sites_master.mat
%   3. 【绘制高清晰 3-Circle Venn 韦恩重叠图 (PNG, 300 DPI)】
%      - 图 1: Task 2 交叉 vs Task 3 交叉 vs 跨任务 Cross Decoding
%      - 图 2: Task 2 Direct vs Task 3 Direct vs 跨任务 Cross Decoding
%      - 合并图: 双面板横向对比图
% ========================================================================

clear; clc; close all;

%% 1. 路径设置
script_dir = fileparts(mfilename('fullpath'));
work_dir   = fileparts(script_dir);
res_root   = fullfile(work_dir, 'result');
tab_dir    = fullfile(res_root, 'tables');
fig_dir    = fullfile(res_root, 'figures', 'decoding_venn_diagrams');

if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

c04_mat = fullfile(tab_dir, 'color_effects_summary.mat');
if ~isfile(c04_mat)
    error('未找到 C04 筛选汇总表: %s', c04_mat);
end
c04_data = load(c04_mat);
if isfield(c04_data, 'all_tbl'), c04_tbl = c04_data.all_tbl; else, c04_tbl = c04_data.res_table; end

% 构建解剖与坐标映射字典
anat_map = containers.Map();
for r = 1:height(c04_tbl)
    k_item = sprintf('%s_%s', char(c04_tbl.subject{r}), char(c04_tbl.channel{r}));
    if ~isKey(anat_map, k_item)
        s_info = struct();
        if ismember('dkt_anatomy', c04_tbl.Properties.VariableNames), s_info.dkt = char(c04_tbl.dkt_anatomy(r)); else, s_info.dkt = ''; end
        if ismember('aal_anatomy', c04_tbl.Properties.VariableNames), s_info.aal = char(c04_tbl.aal_anatomy(r)); else, s_info.aal = ''; end
        if ismember('stream_hierarchy', c04_tbl.Properties.VariableNames)
            s_info.stream = char(c04_tbl.stream_hierarchy(r));
        elseif ismember('stream', c04_tbl.Properties.VariableNames)
            s_info.stream = char(c04_tbl.stream(r));
        else
            s_info.stream = '';
        end
        if ismember('mni_x', c04_tbl.Properties.VariableNames), s_info.x = c04_tbl.mni_x(r); else, s_info.x = NaN; end
        if ismember('mni_y', c04_tbl.Properties.VariableNames), s_info.y = c04_tbl.mni_y(r); else, s_info.y = NaN; end
        if ismember('mni_z', c04_tbl.Properties.VariableNames), s_info.z = c04_tbl.mni_z(r); else, s_info.z = NaN; end
        anat_map(k_item) = s_info;
    end
end

%% 2. 提取 5 大分析的显著电极列表
fprintf('========================================================================\n');
fprintf('  【C15: 提取 5 大 Decoding 显著通道并构建 Venn 对比】\n');
fprintf('========================================================================\n');

% (1) Task 2 交叉 (C06, 跨水果配对)
c06_c = load(fullfile(tab_dir, 'concordant_electrodes_decoding_summary.mat'));
c06_n = load(fullfile(tab_dir, 'non_concordant_electrodes_decoding_summary.mat'));
t2_c_sig1 = c06_c.summary_table(c06_c.summary_table.has_sig_cluster == 1, :);
t2_c_sig2 = c06_n.summary_table(c06_n.summary_table.has_sig_cluster == 1, :);
keys_t2_cross = unique([strcat(t2_c_sig1.subject, '_', t2_c_sig1.channel); strcat(t2_c_sig2.subject, '_', t2_c_sig2.channel)]);

% (2) Task 2 Direct (C13, 纯红绿 5 折)
t2_d = readtable(fullfile(tab_dir, 'task2_direct_decoding_summary.csv'));
keys_t2_direct = unique(strcat(t2_d.subject(t2_d.has_sig_cluster == 1), '_', t2_d.channel(t2_d.has_sig_cluster == 1)));

% (3) Task 3 交叉 (C07, 跨色块配对)
c07 = load(fullfile(tab_dir, 'task3_purecolor_decoding_summary.mat'));
keys_t3_cross = unique(strcat(c07.summary_table.subject(c07.summary_table.has_sig_cluster == 1), '_', c07.summary_table.channel(c07.summary_table.has_sig_cluster == 1)));

% (4) Task 3 Direct (C14, 纯红绿 5 折)
t3_d = readtable(fullfile(tab_dir, 'task3_direct_decoding_summary.csv'));
keys_t3_direct = unique(strcat(t3_d.subject(t3_d.has_sig_cluster == 1), '_', t3_d.channel(t3_d.has_sig_cluster == 1)));

% (5) Task 2 对 Task 3 Cross-Task (C09, 跨任务 2D TGM Perm200)
tgm200 = readtable(fullfile(tab_dir, 'cross_decoding_tgm_perm200_summary.csv'));
keys_cross_task = unique(strcat(tgm200.subject(tgm200.tgm_has_sig_cluster_2d == 1), '_', tgm200.channel(tgm200.tgm_has_sig_cluster_2d == 1)));

fprintf('[+] 1. Task 2 交叉 (跨水果) 显著通道数 : %d\n', numel(keys_t2_cross));
fprintf('[+] 2. Task 2 Direct (纯红绿) 显著通道数: %d\n', numel(keys_t2_direct));
fprintf('[+] 3. Task 3 交叉 (跨色块) 显著通道数 : %d\n', numel(keys_t3_cross));
fprintf('[+] 4. Task 3 Direct (纯红绿) 显著通道数: %d\n', numel(keys_t3_direct));
fprintf('[+] 5. Cross-Task (跨任务 2D) 显著通道数: %d\n', numel(keys_cross_task));

%% 3. 构建全显著位点 Master 结构体与表格，保存为 .mat 与 .csv
all_sig_keys = unique([keys_t2_cross; keys_t2_direct; keys_t3_cross; keys_t3_direct; keys_cross_task]);
n_all_sig = numel(all_sig_keys);

master_list = struct([]);
for k_i = 1:n_all_sig
    k_str = all_sig_keys{k_i};
    parts = strsplit(k_str, '_');
    sub_id = parts{1};
    ch_name = parts{2};
    
    rec = struct();
    rec.subject            = string(sub_id);
    rec.channel            = string(ch_name);
    
    if isKey(anat_map, k_str)
        ainfo = anat_map(k_str);
        rec.dkt_anatomy    = string(ainfo.dkt);
        rec.aal_anatomy    = string(ainfo.aal);
        rec.stream         = string(ainfo.stream);
        rec.mni_x          = ainfo.x;
        rec.mni_y          = ainfo.y;
        rec.mni_z          = ainfo.z;
    else
        rec.dkt_anatomy    = ""; rec.aal_anatomy = ""; rec.stream = "";
        rec.mni_x          = NaN; rec.mni_y = NaN; rec.mni_z = NaN;
    end
    
    rec.sig_task2_cross    = ismember(k_str, keys_t2_cross);
    rec.sig_task2_direct   = ismember(k_str, keys_t2_direct);
    rec.sig_task3_cross    = ismember(k_str, keys_t3_cross);
    rec.sig_task3_direct   = ismember(k_str, keys_t3_direct);
    rec.sig_cross_task     = ismember(k_str, keys_cross_task);
    
    master_list = [master_list; rec]; %#ok<AGROW>
end

master_tbl = struct2table(master_list);

% 结构化输出存储
sig_sites = struct();
sig_sites.master_table   = master_tbl;
sig_sites.keys_t2_cross  = keys_t2_cross;
sig_sites.keys_t2_direct = keys_t2_direct;
sig_sites.keys_t3_cross  = keys_t3_cross;
sig_sites.keys_t3_direct = keys_t3_direct;
sig_sites.keys_cross_task= keys_cross_task;

mat_out = fullfile(tab_dir, 'decoding_significant_sites_master.mat');
csv_out = fullfile(tab_dir, 'decoding_significant_sites_master.csv');
save(mat_out, 'sig_sites', 'master_tbl');
writetable(master_tbl, csv_out);

fprintf('\n[+] 显著位点汇总已成功保存至:\n');
fprintf('    - MAT: %s\n', mat_out);
fprintf('    - CSV: %s (共 %d 个唯一显著电极)\n', csv_out, height(master_tbl));

%% 4. 绘制 Figure 1: 交叉 Decoding 三圆 Venn 图
% A = Task 2 Cross (19), B = Task 3 Cross (14), C = Cross-Task (10)
fig1_png = fullfile(fig_dir, 'figure1_venn_cross_decoding.png');
plot_venn_3circle(keys_t2_cross, keys_t3_cross, keys_cross_task, ...
    'Task 2 Cross (Cross-Fruit)', 'Task 3 Cross (Cross-Patch)', 'Task 2 vs Task 3 (Cross-Task)', ...
    'Venn Diagram: Cross-Condition & Cross-Task Decoding', anat_map, fig1_png);

%% 5. 绘制 Figure 2: 直接 (Direct) Decoding 三圆 Venn 图
% A = Task 2 Direct (13), B = Task 3 Direct (15), C = Cross-Task (10)
fig2_png = fullfile(fig_dir, 'figure2_venn_direct_decoding.png');
plot_venn_3circle(keys_t2_direct, keys_t3_direct, keys_cross_task, ...
    'Task 2 Direct (Pooled Fruits)', 'Task 3 Direct (Pooled Shapes)', 'Task 2 vs Task 3 (Cross-Task)', ...
    'Venn Diagram: Direct Red-Green & Cross-Task Decoding', anat_map, fig2_png);

%% 6. 绘制合并双面板对比图
fig_comb_png = fullfile(fig_dir, 'figure_combined_venn_summary.png');
plot_combined_venn(keys_t2_cross, keys_t3_cross, keys_cross_task, ...
                   keys_t2_direct, keys_t3_direct, keys_cross_task, ...
                   anat_map, fig_comb_png);

fprintf('\n[+] 全部 Venn 图已成功绘制并导出至: %s\n', fig_dir);

%% ========================================================================
%% 辅助绘图函数: 绘制标准单图 3-Circle Venn 及其解剖明细卡片
%% ========================================================================
function plot_venn_3circle(setA, setB, setC, labelA, labelB, labelC, main_title, anat_map, out_png)
    % 集合交集计算
    interAB  = setdiff(intersect(setA, setB), setC);
    interAC  = setdiff(intersect(setA, setC), setB);
    interBC  = setdiff(intersect(setB, setC), setA);
    interABC = intersect(intersect(setA, setB), setC);
    
    onlyA    = setdiff(setdiff(setA, setB), setC);
    onlyB    = setdiff(setdiff(setB, setA), setC);
    onlyC    = setdiff(setdiff(setC, setA), setB);
    
    nA = numel(setA); nB = numel(setB); nC = numel(setC);
    
    % 创建图窗 (左侧 Venn 圆图，右侧明细列表)
    fig = figure('Color', 'w', 'Position', [80, 80, 1250, 650]);
    
    % --- 左面板: 3-Circle Venn 图 ---
    subplot('Position', [0.04, 0.08, 0.52, 0.82]);
    hold on; axis equal; box off; axis off;
    
    R = 1.0;
    cx = [-0.60, 0.60, 0];
    cy = [0.40, 0.40, -0.48];
    th = linspace(0, 2*pi, 360);
    
    cols = [
        0.88, 0.35, 0.28;  % A: 珊瑚暖红
        0.22, 0.55, 0.85;  % B: 宁静钴蓝
        0.28, 0.72, 0.48   % C: 清新翠绿
    ];
    
    for i = 1:3
        x = cx(i) + R * cos(th);
        y = cy(i) + R * sin(th);
        fill(x, y, cols(i, :), 'FaceAlpha', 0.35, 'EdgeColor', cols(i, :) * 0.75, 'LineWidth', 2.5);
    end
    
    % 绘制数字与百分比
    text(-0.95, 0.45, sprintf('%d', numel(onlyA)), 'HorizontalAlignment', 'center', ...
        'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.4, 0.1, 0.1]);
    text(0.95, 0.45, sprintf('%d', numel(onlyB)), 'HorizontalAlignment', 'center', ...
        'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.1, 0.2, 0.5]);
    text(0, -0.92, sprintf('%d', numel(onlyC)), 'HorizontalAlignment', 'center', ...
        'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.1, 0.4, 0.2]);
    
    text(0, 0.65, sprintf('%d', numel(interAB)), 'HorizontalAlignment', 'center', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.25, 0.25, 0.55]);
    text(-0.45, -0.15, sprintf('%d', numel(interAC)), 'HorizontalAlignment', 'center', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.45, 0.25, 0.25]);
    text(0.45, -0.15, sprintf('%d', numel(interBC)), 'HorizontalAlignment', 'center', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.15, 0.45, 0.30]);
    
    text(0, 0.12, sprintf('%d', numel(interABC)), 'HorizontalAlignment', 'center', ...
        'FontSize', 15, 'FontWeight', 'bold', 'Color', 'k');
    
    % 集合名称外侧标注
    text(-0.85, 1.55, sprintf('%s\n(n = %d)', labelA, nA), 'HorizontalAlignment', 'center', ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', cols(1, :) * 0.65);
    text(0.85, 1.55, sprintf('%s\n(n = %d)', labelB, nB), 'HorizontalAlignment', 'center', ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', cols(2, :) * 0.65);
    text(0, -1.65, sprintf('%s\n(n = %d)', labelC, nC), 'HorizontalAlignment', 'center', ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', cols(3, :) * 0.65);
    
    xlim([-2.05, 2.05]); ylim([-1.90, 1.85]);
    
    % --- 右面板: 交叉通道详细解剖卡片 ---
    subplot('Position', [0.58, 0.08, 0.40, 0.82]);
    hold on; box on; set(gca, 'XTick', [], 'YTick', [], 'Color', [0.98, 0.98, 0.98]);
    
    y_pos = 0.95;
    text(0.04, y_pos, 'Shared Significant Sites & Anatomy', 'FontSize', 14, 'FontWeight', 'bold');
    y_pos = y_pos - 0.07;
    
    % 渲染各类重叠集合
    categories = {
        sprintf('A & C Overlap: %s ∩ %s (n = %d)', labelA, labelC, numel(interAC)), interAC, [0.55, 0.2, 0.2];
        sprintf('A & B Overlap: %s ∩ %s (n = %d)', labelA, labelB, numel(interAB)), interAB, [0.2, 0.25, 0.55];
        sprintf('B & C Overlap: %s ∩ %s (n = %d)', labelB, labelC, numel(interBC)), interBC, [0.15, 0.45, 0.25];
        sprintf('Triple Overlap: A ∩ B ∩ C (n = %d)', numel(interABC)), interABC, [0.1, 0.1, 0.1]
    };
    
    for c_i = 1:size(categories, 1)
        cat_title = categories{c_i, 1};
        cat_items = categories{c_i, 2};
        cat_col   = categories{c_i, 3};
        
        plot([0.04, 0.94], [y_pos, y_pos], 'Color', [0.85, 0.85, 0.85], 'LineWidth', 1.0);
        y_pos = y_pos - 0.04;
        
        text(0.04, y_pos, cat_title, 'FontSize', 11, 'FontWeight', 'bold', 'Color', cat_col);
        y_pos = y_pos - 0.045;
        
        if isempty(cat_items)
            text(0.08, y_pos, '(None)', 'FontSize', 10, 'Color', [0.5, 0.5, 0.5], 'FontAngle', 'italic');
            y_pos = y_pos - 0.045;
        else
            for item_i = 1:numel(cat_items)
                k_str = cat_items{item_i};
                anat_str = 'Unknown';
                if isKey(anat_map, k_str)
                    ainfo = anat_map(k_str);
                    if ~isempty(ainfo.dkt), anat_str = ainfo.dkt; elseif ~isempty(ainfo.aal), anat_str = ainfo.aal; end
                end
                k_disp = strrep(k_str, '_', '-');
                txt_item = sprintf('• %s  [%s]', k_disp, anat_str);
                text(0.08, y_pos, txt_item, 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.2, 0.2, 0.2]);
                y_pos = y_pos - 0.042;
            end
        end
        y_pos = y_pos - 0.025;
    end
    
    xlim([0, 1]); ylim([0, 1]);
    sgtitle(main_title, 'FontSize', 16, 'FontWeight', 'bold');
    
    exportgraphics(fig, out_png, 'Resolution', 300);
    close(fig);
end

%% ========================================================================
%% 辅助绘图函数: 绘制双面板合并对比图 (Figure 1 + Figure 2)
%% ========================================================================
function plot_combined_venn(s1A, s1B, s1C, s2A, s2B, s2C, anat_map, out_png)
    fig = figure('Color', 'w', 'Position', [50, 50, 1500, 650]);
    
    % Panel 1: Cross Decoding
    subplot('Position', [0.03, 0.08, 0.45, 0.82]);
    draw_venn_axes(s1A, s1B, s1C, 'Task 2 Cross (Fruit-Pair)', 'Task 3 Cross (Patch)', 'Cross-Task (T3 -> T2)', ...
        '(A) Cross-Condition Cross-Task Overlap', anat_map);
    
    % Panel 2: Direct Decoding
    subplot('Position', [0.52, 0.08, 0.45, 0.82]);
    draw_venn_axes(s2A, s2B, s2C, 'Task 2 Direct (Pooled)', 'Task 3 Direct (Pooled)', 'Cross-Task (T3 -> T2)', ...
        '(B) Direct Red-Green Cross-Task Overlap', anat_map);
    
    sgtitle('Comparative Venn Analysis: Significant Channel Overlap Across Within-Task and Cross-Task Decoding', ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    exportgraphics(fig, out_png, 'Resolution', 300);
    close(fig);
end

function draw_venn_axes(setA, setB, setC, labelA, labelB, labelC, panel_title, ~)
    hold on; axis equal; box off; axis off;
    
    interAB  = setdiff(intersect(setA, setB), setC);
    interAC  = setdiff(intersect(setA, setC), setB);
    interBC  = setdiff(intersect(setB, setC), setA);
    interABC = intersect(intersect(setA, setB), setC);
    onlyA    = setdiff(setdiff(setA, setB), setC);
    onlyB    = setdiff(setdiff(setB, setA), setC);
    onlyC    = setdiff(setdiff(setC, setA), setB);
    
    R = 1.0;
    cx = [-0.60, 0.60, 0];
    cy = [0.40, 0.40, -0.48];
    th = linspace(0, 2*pi, 360);
    
    cols = [
        0.88, 0.35, 0.28;
        0.22, 0.55, 0.85;
        0.28, 0.72, 0.48
    ];
    
    for i = 1:3
        x = cx(i) + R * cos(th);
        y = cy(i) + R * sin(th);
        fill(x, y, cols(i, :), 'FaceAlpha', 0.35, 'EdgeColor', cols(i, :) * 0.75, 'LineWidth', 2.2);
    end
    
    text(-0.95, 0.45, sprintf('%d', numel(onlyA)), 'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold', 'Color', [0.4, 0.1, 0.1]);
    text(0.95, 0.45, sprintf('%d', numel(onlyB)), 'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold', 'Color', [0.1, 0.2, 0.5]);
    text(0, -0.92, sprintf('%d', numel(onlyC)), 'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold', 'Color', [0.1, 0.4, 0.2]);
    
    text(0, 0.65, sprintf('%d', numel(interAB)), 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold', 'Color', [0.25, 0.25, 0.55]);
    text(-0.45, -0.15, sprintf('%d', numel(interAC)), 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold', 'Color', [0.45, 0.25, 0.25]);
    text(0.45, -0.15, sprintf('%d', numel(interBC)), 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold', 'Color', [0.15, 0.45, 0.30]);
    text(0, 0.12, sprintf('%d', numel(interABC)), 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');
    
    text(-0.85, 1.55, sprintf('%s\n(n = %d)', labelA, numel(setA)), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', 'Color', cols(1,:)*0.65);
    text(0.85, 1.55, sprintf('%s\n(n = %d)', labelB, numel(setB)), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', 'Color', cols(2,:)*0.65);
    text(0, -1.65, sprintf('%s\n(n = %d)', labelC, numel(setC)), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', 'Color', cols(3,:)*0.65);
    
    title(panel_title, 'FontSize', 13, 'FontWeight', 'bold');
    xlim([-2.05, 2.05]); ylim([-1.90, 1.85]);
end
