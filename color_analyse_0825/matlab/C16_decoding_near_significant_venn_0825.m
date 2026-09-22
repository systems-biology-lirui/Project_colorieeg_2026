%% ========================================================================
% 脚本名称: C16_decoding_near_significant_venn_0825.m
% 功能:
%   1. 【纳入“接近显著”通道的 5 大 Decoding 统计汇总】
%      - 定义严格显著: Cluster-Mass FWE 置换检验 p < 0.05
%      - 定义接近显著:
%        * 1D 曲线 (Task 2/3 Cross 与 Direct): 刺激后连续显著点数 >= 4 (持续 >= 80ms, p_pt < 0.05)
%        * 2D 泛化 (Cross-Task TGM): 2D 聚类置换检验呈边缘趋势 (p < 0.10)
%   2. 【导出全量位点汇总至独立 .mat 与 .csv 文件】
%      保存到 result/tables/decoding_near_significant_sites_master.mat
%   3. 【绘制 3-Circle Venn 韦恩重叠图 (PNG, 300 DPI, 无 FIG)】
%      - 图 1: Task 2 交叉 vs Task 3 交叉 vs 跨任务 Cross Decoding (含接近显著)
%      - 图 2: Task 2 Direct vs Task 3 Direct vs 跨任务 Cross Decoding (含接近显著)
%      - 合并图: 双面板对比图
% ========================================================================

clear; clc; close all;

%% 1. 参数与路径配置 (置顶易调，简写平铺)
cfg = struct();
cfg.min_pts_near   = 4;           % 接近显著连续时间点数门槛 (4点 = 80ms, 步长20ms)
cfg.xt_p_threshold = 0.10;        % 跨任务泛化边缘显著 p 值门槛 (p < 0.10)

script_dir = fileparts(mfilename('fullpath'));
work_dir   = fileparts(script_dir);
res_root   = fullfile(work_dir, 'result');
tab_dir    = fullfile(res_root, 'tables');
fig_dir    = fullfile(res_root, 'figures', 'decoding_venn_diagrams');

if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% 载入解剖与坐标字典
c04_mat = fullfile(tab_dir, 'color_effects_summary.mat');
if ~isfile(c04_mat), error('未找到 C04 文件: %s', c04_mat); end
c04_data = load(c04_mat);
if isfield(c04_data, 'all_tbl'), c04_tbl = c04_data.all_tbl; else, c04_tbl = c04_data.res_table; end

anat_map = containers.Map();
for r = 1:height(c04_tbl)
    k_item = sprintf('%s_%s', char(c04_tbl.subject{r}), char(c04_tbl.channel{r}));
    if ~isKey(anat_map, k_item)
        s = struct();
        if ismember('dkt_anatomy', c04_tbl.Properties.VariableNames), s.dkt = char(c04_tbl.dkt_anatomy(r)); else, s.dkt = ''; end
        if ismember('aal_anatomy', c04_tbl.Properties.VariableNames), s.aal = char(c04_tbl.aal_anatomy(r)); else, s.aal = ''; end
        if ismember('stream_hierarchy', c04_tbl.Properties.VariableNames)
            s.stream = char(c04_tbl.stream_hierarchy(r));
        elseif ismember('stream', c04_tbl.Properties.VariableNames)
            s.stream = char(c04_tbl.stream(r));
        else
            s.stream = '';
        end
        if ismember('mni_x', c04_tbl.Properties.VariableNames), s.x = c04_tbl.mni_x(r); else, s.x = NaN; end
        if ismember('mni_y', c04_tbl.Properties.VariableNames), s.y = c04_tbl.mni_y(r); else, s.y = NaN; end
        if ismember('mni_z', c04_tbl.Properties.VariableNames), s.z = c04_tbl.mni_z(r); else, s.z = NaN; end
        anat_map(k_item) = s;
    end
end

%% 2. 载入 5 大 Decoding 原始时程与统计结果
fprintf('========================================================================\n');
fprintf('  【C16: 提取包含接近显著位点的 5 大 Decoding 通道并构建 Venn 对比】\n');
fprintf('========================================================================\n');

% (1) 载入基础时程 .mat (包含 Task 2 Cross 与 Task 3 Cross 的逐点 p 值)
sc_mat = fullfile(tab_dir, 'single_channel_decoding_timecourses.mat');
sc = load(sc_mat);
time_ms = sc.time_ms;
post_mask = (time_ms >= 0);

% Task 2 Cross 严格显著结果
c06_c = load(fullfile(tab_dir, 'concordant_electrodes_decoding_summary.mat'));
c06_n = load(fullfile(tab_dir, 'non_concordant_electrodes_decoding_summary.mat'));
t2_c_sum = [c06_c.summary_table; c06_n.summary_table];

% Task 3 Cross 严格显著结果
c07 = load(fullfile(tab_dir, 'task3_purecolor_decoding_summary.mat'));
t3_c_sum = c07.summary_table;

% Task 2 Direct 汇总与时程目录
t2_d = readtable(fullfile(tab_dir, 'task2_direct_decoding_summary.csv'));
tc_t2d_dir = fullfile(tab_dir, 'decoding_task2_direct_timecourses');

% Task 3 Direct 汇总与时程目录
t3_d = readtable(fullfile(tab_dir, 'task3_direct_decoding_summary.csv'));
tc_t3d_dir = fullfile(tab_dir, 'decoding_task3_direct_timecourses');

% Cross-Task 汇总
tgm = readtable(fullfile(tab_dir, 'cross_decoding_tgm_perm200_summary.csv'));

%% 3. 提取各分析的【严格显著】与【接近显著】通道列表

% --- (1) Task 2 Cross (跨水果) ---
t2_cross_items = [sc.concordant(:); sc.non_concordant(:)];
keys_t2c_strict = {}; keys_t2c_near = {};
for i = 1:numel(t2_cross_items)
    sub = t2_cross_items(i).subject; ch = t2_cross_items(i).channel;
    k = sprintf('%s_%s', sub, ch);
    % 严格显著
    m_idx = find(strcmp(t2_c_sum.subject, sub) & strcmp(t2_c_sum.channel, ch), 1);
    is_str = (~isempty(m_idx) && t2_c_sum.has_sig_cluster(m_idx) == 1);
    % 连续显著点数
    max_c = calc_consecutive_points(t2_cross_items(i).p_pointwise(post_mask) < 0.05);
    if is_str
        keys_t2c_strict{end+1, 1} = k; %#ok<AGROW>
    elseif max_c >= cfg.min_pts_near
        keys_t2c_near{end+1, 1} = k; %#ok<AGROW>
    end
end
keys_t2c_combined = unique([keys_t2c_strict; keys_t2c_near]);

% --- (2) Task 2 Direct (纯红绿) ---
keys_t2d_strict = {}; keys_t2d_near = {};
for i = 1:height(t2_d)
    sub = t2_d.subject{i}; ch = t2_d.channel{i};
    k = sprintf('%s_%s', sub, ch);
    is_str = (t2_d.has_sig_cluster(i) == 1);
    f = fullfile(tc_t2d_dir, sprintf('%s_%s_task2_direct_timecourse.csv', sub, ch));
    max_c = 0;
    if isfile(f)
        tb = readtable(f);
        max_c = calc_consecutive_points((tb.p_pointwise < 0.05) & (tb.time_ms >= 0));
    end
    if is_str
        keys_t2d_strict{end+1, 1} = k; %#ok<AGROW>
    elseif max_c >= cfg.min_pts_near
        keys_t2d_near{end+1, 1} = k; %#ok<AGROW>
    end
end
keys_t2d_combined = unique([keys_t2d_strict; keys_t2d_near]);

% --- (3) Task 3 Cross (跨色块) ---
t3_cross_items = sc.task3_purecolor(:);
keys_t3c_strict = {}; keys_t3c_near = {};
for i = 1:numel(t3_cross_items)
    sub = t3_cross_items(i).subject; ch = t3_cross_items(i).channel;
    k = sprintf('%s_%s', sub, ch);
    m_idx = find(strcmp(t3_c_sum.subject, sub) & strcmp(t3_c_sum.channel, ch), 1);
    is_str = (~isempty(m_idx) && t3_c_sum.has_sig_cluster(m_idx) == 1);
    max_c = calc_consecutive_points(t3_cross_items(i).p_pointwise(post_mask) < 0.05);
    if is_str
        keys_t3c_strict{end+1, 1} = k; %#ok<AGROW>
    elseif max_c >= cfg.min_pts_near
        keys_t3c_near{end+1, 1} = k; %#ok<AGROW>
    end
end
keys_t3c_combined = unique([keys_t3c_strict; keys_t3c_near]);

% --- (4) Task 3 Direct (纯红绿) ---
keys_t3d_strict = {}; keys_t3d_near = {};
for i = 1:height(t3_d)
    sub = t3_d.subject{i}; ch = t3_d.channel{i};
    k = sprintf('%s_%s', sub, ch);
    is_str = (t3_d.has_sig_cluster(i) == 1);
    f = fullfile(tc_t3d_dir, sprintf('%s_%s_task3_direct_timecourse.csv', sub, ch));
    max_c = 0;
    if isfile(f)
        tb = readtable(f);
        max_c = calc_consecutive_points((tb.p_pointwise < 0.05) & (tb.time_ms >= 0));
    end
    if is_str
        keys_t3d_strict{end+1, 1} = k; %#ok<AGROW>
    elseif max_c >= cfg.min_pts_near
        keys_t3d_near{end+1, 1} = k; %#ok<AGROW>
    end
end
keys_t3d_combined = unique([keys_t3d_strict; keys_t3d_near]);

% --- (5) Cross-Task (跨任务 2D TGM) ---
keys_xt_strict = {}; keys_xt_near = {};
for i = 1:height(tgm)
    sub = tgm.subject{i}; ch = tgm.channel{i};
    k = sprintf('%s_%s', sub, ch);
    is_str = (tgm.tgm_has_sig_cluster_2d(i) == 1);
    is_trend = (tgm.tgm_cluster_p_min(i) < cfg.xt_p_threshold);
    if is_str
        keys_xt_strict{end+1, 1} = k; %#ok<AGROW>
    elseif is_trend
        keys_xt_near{end+1, 1} = k; %#ok<AGROW>
    end
end
keys_xt_combined = unique([keys_xt_strict; keys_xt_near]);

fprintf('[+] 1. Task 2 交叉 : 严格 = %d, 接近显著 = %d, 合计 = %d\n', numel(keys_t2c_strict), numel(keys_t2c_near), numel(keys_t2c_combined));
fprintf('[+] 2. Task 2 Direct: 严格 = %d, 接近显著 = %d, 合计 = %d\n', numel(keys_t2d_strict), numel(keys_t2d_near), numel(keys_t2d_combined));
fprintf('[+] 3. Task 3 交叉 : 严格 = %d, 接近显著 = %d, 合计 = %d\n', numel(keys_t3c_strict), numel(keys_t3c_near), numel(keys_t3c_combined));
fprintf('[+] 4. Task 3 Direct: 严格 = %d, 接近显著 = %d, 合计 = %d\n', numel(keys_t3d_strict), numel(keys_t3d_near), numel(keys_t3d_combined));
fprintf('[+] 5. Cross-Task   : 严格 = %d, 接近显著 = %d, 合计 = %d\n', numel(keys_xt_strict), numel(keys_xt_near), numel(keys_xt_combined));

%% 4. 构建并导出包含接近显著位点的 Master 数据表 (.mat 与 .csv)
all_comb_keys = unique([keys_t2c_combined; keys_t2d_combined; keys_t3c_combined; keys_t3d_combined; keys_xt_combined]);
n_total_comb = numel(all_comb_keys);

master_list = struct([]);
for i = 1:n_total_comb
    k_str = all_comb_keys{i};
    parts = strsplit(k_str, '_');
    sub_id = parts{1}; ch_name = parts{2};
    
    r = struct();
    r.subject = string(sub_id);
    r.channel = string(ch_name);
    
    if isKey(anat_map, k_str)
        info = anat_map(k_str);
        r.dkt_anatomy = string(info.dkt);
        r.aal_anatomy = string(info.aal);
        r.stream      = string(info.stream);
        r.mni_x       = info.x;
        r.mni_y       = info.y;
        r.mni_z       = info.z;
    else
        r.dkt_anatomy = ""; r.aal_anatomy = ""; r.stream = "";
        r.mni_x = NaN; r.mni_y = NaN; r.mni_z = NaN;
    end
    
    % 0: 未达标, 1: 接近显著, 2: 严格显著
    r.status_task2_cross  = ismember(k_str, keys_t2c_strict)*2 + (ismember(k_str, keys_t2c_near) & ~ismember(k_str, keys_t2c_strict))*1;
    r.status_task2_direct = ismember(k_str, keys_t2d_strict)*2 + (ismember(k_str, keys_t2d_near) & ~ismember(k_str, keys_t2d_strict))*1;
    r.status_task3_cross  = ismember(k_str, keys_t3c_strict)*2 + (ismember(k_str, keys_t3c_near) & ~ismember(k_str, keys_t3c_strict))*1;
    r.status_task3_direct = ismember(k_str, keys_t3d_strict)*2 + (ismember(k_str, keys_t3d_near) & ~ismember(k_str, keys_t3d_strict))*1;
    r.status_cross_task   = ismember(k_str, keys_xt_strict)*2  + (ismember(k_str, keys_xt_near)  & ~ismember(k_str, keys_xt_strict))*1;
    
    % 布尔值标记 (是否被纳入宽松集合)
    r.inc_task2_cross  = ismember(k_str, keys_t2c_combined);
    r.inc_task2_direct = ismember(k_str, keys_t2d_combined);
    r.inc_task3_cross  = ismember(k_str, keys_t3c_combined);
    r.inc_task3_direct = ismember(k_str, keys_t3d_combined);
    r.inc_cross_task   = ismember(k_str, keys_xt_combined);
    
    master_list = [master_list; r]; %#ok<AGROW>
end

master_tbl = struct2table(master_list);

% 保存文件
out_mat = fullfile(tab_dir, 'decoding_near_significant_sites_master.mat');
out_csv = fullfile(tab_dir, 'decoding_near_significant_sites_master.csv');

sig_near_master = struct();
sig_near_master.table = master_tbl;
sig_near_master.keys_task2_cross_combined  = keys_t2c_combined;
sig_near_master.keys_task2_direct_combined = keys_t2d_combined;
sig_near_master.keys_task3_cross_combined  = keys_t3c_combined;
sig_near_master.keys_task3_direct_combined = keys_t3d_combined;
sig_near_master.keys_cross_task_combined   = keys_xt_combined;

save(out_mat, 'sig_near_master', 'cfg');
writetable(master_tbl, out_csv);
fprintf('\n[+] 纳入接近显著的 Master 表格已成功保存:\n  - MAT: %s\n  - CSV: %s\n  - 纳入通道总数: %d\n', ...
    out_mat, out_csv, height(master_tbl));

%% 5. 绘制三圆环 Venn 韦恩图 (PNG, 300 DPI)

% --- 图 1: 交叉 Decoding (Cross-Fruit vs Cross-Patch vs Cross-Task) ---
png1 = fullfile(fig_dir, 'figure1_venn_cross_decoding_near_sig.png');
plot_venn_3circle_near(keys_t2c_combined, keys_t3c_combined, keys_xt_combined, ...
    keys_t2c_strict, keys_t3c_strict, keys_xt_strict, ...
    'Task 2 Cross (Memory)', 'Task 3 Cross (Color)', 'Cross-Task (T3->T2)', ...
    'Decoding Sites Overlap (Cross Decodings, Including Near-Significant)', ...
    anat_map, png1);

% --- 图 2: Direct Decoding (Direct Task 2 vs Direct Task 3 vs Cross-Task) ---
png2 = fullfile(fig_dir, 'figure2_venn_direct_decoding_near_sig.png');
plot_venn_3circle_near(keys_t2d_combined, keys_t3d_combined, keys_xt_combined, ...
    keys_t2d_strict, keys_t3d_strict, keys_xt_strict, ...
    'Task 2 Direct (Memory)', 'Task 3 Direct (Color)', 'Cross-Task (T3->T2)', ...
    'Decoding Sites Overlap (Direct Decodings, Including Near-Significant)', ...
    anat_map, png2);

% --- 组合总览图 ---
png_comb = fullfile(fig_dir, 'figure_combined_venn_near_sig.png');
plot_combined_venn_near(keys_t2c_combined, keys_t3c_combined, keys_xt_combined, ...
    keys_t2d_combined, keys_t3d_combined, keys_xt_combined, ...
    keys_t2c_strict, keys_t3c_strict, keys_xt_strict, ...
    keys_t2d_strict, keys_t3d_strict, ...
    png_comb);

fprintf('\n========================================================================\n');
fprintf('  【C16 分析全部顺利完成！】\n');
fprintf('  图 1: %s\n', png1);
fprintf('  图 2: %s\n', png2);
fprintf('  总览图: %s\n', png_comb);
fprintf('========================================================================\n');

%% ========================================================================
%% 辅助函数: 计算一维逻辑序列中的最大连续点数
%% ========================================================================
function max_c = calc_consecutive_points(m_vec)
    max_c = 0; cur = 0;
    for k = 1:numel(m_vec)
        if m_vec(k)
            cur = cur + 1;
            if cur > max_c, max_c = cur; end
        else
            cur = 0;
        end
    end
end

%% ========================================================================
%% 辅助绘图函数: 绘制包含接近显著明细的标准单图 3-Circle Venn
%% ========================================================================
function plot_venn_3circle_near(setA, setB, setC, strictA, strictB, strictC, ...
    labelA, labelB, labelC, main_title, anat_map, out_png)

    interAB  = setdiff(intersect(setA, setB), setC);
    interAC  = setdiff(intersect(setA, setC), setB);
    interBC  = setdiff(intersect(setB, setC), setA);
    interABC = intersect(intersect(setA, setB), setC);
    
    onlyA    = setdiff(setdiff(setA, setB), setC);
    onlyB    = setdiff(setdiff(setB, setA), setC);
    onlyC    = setdiff(setdiff(setC, setA), setB);
    
    nA = numel(setA); nB = numel(setB); nC = numel(setC);
    
    fig = figure('Color', 'w', 'Position', [60, 60, 1300, 680], 'Visible', 'off');
    
    % 左面板: 3-Circle Venn
    subplot('Position', [0.03, 0.08, 0.53, 0.82]);
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
    
    % 三者重叠 (核心亮点)
    text(0, 0.12, sprintf('%d', numel(interABC)), 'HorizontalAlignment', 'center', ...
        'FontSize', 17, 'FontWeight', 'bold', 'Color', [0.8, 0.05, 0.05]);
    
    text(-0.85, 1.55, sprintf('%s\n(n = %d)', labelA, nA), 'HorizontalAlignment', 'center', ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', cols(1, :) * 0.65);
    text(0.85, 1.55, sprintf('%s\n(n = %d)', labelB, nB), 'HorizontalAlignment', 'center', ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', cols(2, :) * 0.65);
    text(0, -1.65, sprintf('%s\n(n = %d)', labelC, nC), 'HorizontalAlignment', 'center', ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', cols(3, :) * 0.65);
    
    xlim([-2.05, 2.05]); ylim([-1.90, 1.85]);
    
    % 右面板: 解剖与通道明细卡片
    subplot('Position', [0.58, 0.08, 0.40, 0.82]);
    hold on; box on; set(gca, 'XTick', [], 'YTick', [], 'Color', [0.98, 0.98, 0.98]);
    
    y_pos = 0.96;
    text(0.04, y_pos, 'Shared Sites Breakdown (Strict & Near-Sig)', 'FontSize', 13, 'FontWeight', 'bold');
    y_pos = y_pos - 0.05;
    
    categories = {
        sprintf('★ Triple Overlap: A ∩ B ∩ C (n = %d)', numel(interABC)), interABC, [0.8, 0.1, 0.1];
        sprintf('A & C Overlap: %s ∩ %s (n = %d)', labelA, labelC, numel(interAC)), interAC, [0.55, 0.2, 0.2];
        sprintf('A & B Overlap: %s ∩ %s (n = %d)', labelA, labelB, numel(interAB)), interAB, [0.2, 0.25, 0.55];
        sprintf('B & C Overlap: %s ∩ %s (n = %d)', labelB, labelC, numel(interBC)), interBC, [0.15, 0.45, 0.25]
    };
    
    for c_i = 1:size(categories, 1)
        cat_title = categories{c_i, 1};
        cat_items = categories{c_i, 2};
        cat_col   = categories{c_i, 3};
        
        plot([0.04, 0.94], [y_pos, y_pos], 'Color', [0.85, 0.85, 0.85], 'LineWidth', 1.0);
        y_pos = y_pos - 0.035;
        
        text(0.04, y_pos, cat_title, 'FontSize', 10.5, 'FontWeight', 'bold', 'Color', cat_col);
        y_pos = y_pos - 0.035;
        
        if isempty(cat_items)
            text(0.08, y_pos, '(None)', 'FontSize', 9.5, 'Color', [0.6, 0.6, 0.6], 'FontAngle', 'italic');
            y_pos = y_pos - 0.035;
        else
            for item_i = 1:numel(cat_items)
                k_str = cat_items{item_i};
                p = strsplit(k_str, '_');
                sub_c = p{1}; ch_c = p{2};
                
                anat_str = '';
                if isKey(anat_map, k_str)
                    s_info = anat_map(k_str);
                    if ~isempty(s_info.dkt), anat_str = s_info.dkt; elseif ~isempty(s_info.stream), anat_str = s_info.stream; end
                end
                
                % 标注严格显著 vs 接近显著
                str_tag = '';
                if ismember(k_str, strictA), str_tag = [str_tag 'A*']; end
                if ismember(k_str, strictB), str_tag = [str_tag 'B*']; end
                if ismember(k_str, strictC), str_tag = [str_tag 'C*']; end
                if isempty(str_tag), str_tag = 'near-sig'; end
                
                line_str = sprintf('• %s-%s [%s] (%s)', sub_c, ch_c, anat_str, str_tag);
                text(0.06, y_pos, line_str, 'FontSize', 8.5, 'Color', [0.2, 0.2, 0.2]);
                y_pos = y_pos - 0.028;
                if y_pos < 0.04, break; end
            end
        end
        y_pos = y_pos - 0.02;
        if y_pos < 0.04, break; end
    end
    
    sgtitle(main_title, 'FontSize', 14, 'FontWeight', 'bold');
    exportgraphics(fig, out_png, 'Resolution', 300);
    close(fig);
end

%% ========================================================================
%% 辅助绘图函数: 绘制包含接近显著的双面板组合总览图
%% ========================================================================
function plot_combined_venn_near(setA1, setB1, setC1, setA2, setB2, setC2, ...
    strictA1, strictB1, strictC1, strictA2, strictB2, out_png)

    fig = figure('Color', 'w', 'Position', [50, 50, 1400, 650], 'Visible', 'off');
    
    cols = [
        0.88, 0.35, 0.28;
        0.22, 0.55, 0.85;
        0.28, 0.72, 0.48
    ];
    R = 1.0;
    cx = [-0.60, 0.60, 0];
    cy = [0.40, 0.40, -0.48];
    th = linspace(0, 2*pi, 360);
    
    % Panel 1: Cross
    subplot(1, 2, 1);
    hold on; axis equal; box off; axis off;
    for i = 1:3
        fill(cx(i) + R * cos(th), cy(i) + R * sin(th), cols(i, :), 'FaceAlpha', 0.35, ...
            'EdgeColor', cols(i, :) * 0.75, 'LineWidth', 2.2);
    end
    
    onlyA = setdiff(setdiff(setA1, setB1), setC1);
    onlyB = setdiff(setdiff(setB1, setA1), setC1);
    onlyC = setdiff(setdiff(setC1, setA1), setB1);
    iAB   = setdiff(intersect(setA1, setB1), setC1);
    iAC   = setdiff(intersect(setA1, setC1), setB1);
    iBC   = setdiff(intersect(setB1, setC1), setA1);
    iABC  = intersect(intersect(setA1, setB1), setC1);
    
    text(-0.95, 0.45, sprintf('%d', numel(onlyA)), 'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold');
    text(0.95, 0.45, sprintf('%d', numel(onlyB)), 'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold');
    text(0, -0.92, sprintf('%d', numel(onlyC)), 'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold');
    text(0, 0.65, sprintf('%d', numel(iAB)), 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
    text(-0.45, -0.15, sprintf('%d', numel(iAC)), 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
    text(0.45, -0.15, sprintf('%d', numel(iBC)), 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
    text(0, 0.12, sprintf('%d', numel(iABC)), 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.8, 0.05, 0.05]);
    
    text(-0.85, 1.55, sprintf('Task 2 Cross\n(n = %d)', numel(setA1)), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    text(0.85, 1.55, sprintf('Task 3 Cross\n(n = %d)', numel(setB1)), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    text(0, -1.65, sprintf('Cross-Task (T3->T2)\n(n = %d)', numel(setC1)), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    xlim([-2.05, 2.05]); ylim([-1.90, 1.85]);
    title('(A) Cross-Decoding Overlap (Strict + Near-Sig)', 'FontSize', 13, 'FontWeight', 'bold');
    
    % Panel 2: Direct
    subplot(1, 2, 2);
    hold on; axis equal; box off; axis off;
    for i = 1:3
        fill(cx(i) + R * cos(th), cy(i) + R * sin(th), cols(i, :), 'FaceAlpha', 0.35, ...
            'EdgeColor', cols(i, :) * 0.75, 'LineWidth', 2.2);
    end
    
    onlyA2 = setdiff(setdiff(setA2, setB2), setC2);
    onlyB2 = setdiff(setdiff(setB2, setA2), setC2);
    onlyC2 = setdiff(setdiff(setC2, setA2), setB2);
    iAB2   = setdiff(intersect(setA2, setB2), setC2);
    iAC2   = setdiff(intersect(setA2, setC2), setB2);
    iBC2   = setdiff(intersect(setB2, setC2), setA2);
    iABC2  = intersect(intersect(setA2, setB2), setC2);
    
    text(-0.95, 0.45, sprintf('%d', numel(onlyA2)), 'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold');
    text(0.95, 0.45, sprintf('%d', numel(onlyB2)), 'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold');
    text(0, -0.92, sprintf('%d', numel(onlyC2)), 'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold');
    text(0, 0.65, sprintf('%d', numel(iAB2)), 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
    text(-0.45, -0.15, sprintf('%d', numel(iAC2)), 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
    text(0.45, -0.15, sprintf('%d', numel(iBC2)), 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
    text(0, 0.12, sprintf('%d', numel(iABC2)), 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.8, 0.05, 0.05]);
    
    text(-0.85, 1.55, sprintf('Task 2 Direct\n(n = %d)', numel(setA2)), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    text(0.85, 1.55, sprintf('Task 3 Direct\n(n = %d)', numel(setB2)), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    text(0, -1.65, sprintf('Cross-Task (T3->T2)\n(n = %d)', numel(setC2)), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    xlim([-2.05, 2.05]); ylim([-1.90, 1.85]);
    title('(B) Direct-Decoding Overlap (Strict + Near-Sig)', 'FontSize', 13, 'FontWeight', 'bold');
    
    sgtitle('Comparative Venn Analysis of Color Decoding (Including Near-Significant Sites)', 'FontSize', 15, 'FontWeight', 'bold');
    exportgraphics(fig, out_png, 'Resolution', 300);
    close(fig);
end
