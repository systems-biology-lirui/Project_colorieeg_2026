% test_near_significant_stats.m
% 探索当统计包含“接近显著”(Near-significant / Trend) 位点时的数量与交集变化

clear; clc;
tab_dir = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';

% 1. 读取基础数据
c06_c = load(fullfile(tab_dir, 'concordant_electrodes_decoding_summary.mat'));
c06_n = load(fullfile(tab_dir, 'non_concordant_electrodes_decoding_summary.mat'));
t2_c_all = [c06_c.summary_table; c06_n.summary_table];

t2_d = readtable(fullfile(tab_dir, 'task2_direct_decoding_summary.csv'));
c07  = load(fullfile(tab_dir, 'task3_purecolor_decoding_summary.mat'));
t3_d = readtable(fullfile(tab_dir, 'task3_direct_decoding_summary.csv'));
tgm  = readtable(fullfile(tab_dir, 'cross_decoding_tgm_perm200_summary.csv'));

% 2. 检查 1D 曲线的连续点数（从各自时程 CSV 提取）
% 定义连续时间点 >= 4 (即持续 >= 80ms p_pt < 0.05) 作为近显著标准
fprintf('=== 分析 1: 基于持续时间簇 (连续 >= 4 个点, 即 >= 80ms) ===\n');

% 辅助函数式逻辑：计算每个通道时程中最大连续显著点数
get_max_consec = @(csv_dir, sub, ch) helper_consec(csv_dir, sub, ch);

% Task 2 Direct 时程
tc_t2d_dir = fullfile(tab_dir, 'decoding_task2_direct_timecourses');
t2d_consec = zeros(height(t2_d), 1);
for i = 1:height(t2_d)
    f = fullfile(tc_t2d_dir, sprintf('%s_%s_task2_direct_timecourse.csv', t2_d.subject{i}, t2_d.channel{i}));
    if isfile(f)
        tb = readtable(f);
        m = (tb.p_pointwise < 0.05) & (tb.time_ms >= 0);
        t2d_consec(i) = calc_max_consec(m);
    end
end

% Task 3 Direct 时程
tc_t3d_dir = fullfile(tab_dir, 'decoding_task3_direct_timecourses');
t3d_consec = zeros(height(t3_d), 1);
for i = 1:height(t3_d)
    f = fullfile(tc_t3d_dir, sprintf('%s_%s_task3_direct_timecourse.csv', t3_d.subject{i}, t3_d.channel{i}));
    if isfile(f)
        tb = readtable(f);
        m = (tb.p_pointwise < 0.05) & (tb.time_ms >= 0);
        t3d_consec(i) = calc_max_consec(m);
    end
end

% Task 2 Cross 时程
t2c_consec = zeros(height(t2_c_all), 1);
for i = 1:height(t2_c_all)
    sub = t2_c_all.subject{i}; ch = t2_c_all.channel{i};
    f1 = fullfile(tab_dir, 'decoding_concordant_timecourses', sprintf('%s_%s_decoding_timecourse.csv', sub, ch));
    f2 = fullfile(tab_dir, 'decoding_non_concordant_timecourses', sprintf('%s_%s_decoding_timecourse.csv', sub, ch));
    f = '';
    if isfile(f1), f = f1; elseif isfile(f2), f = f2; end
    if ~isempty(f)
        tb = readtable(f);
        m = (tb.p_pointwise < 0.05) & (tb.time_ms >= 0);
        t2c_consec(i) = calc_max_consec(m);
    end
end

% Task 3 Cross 时程
t3c_consec = zeros(height(c07.summary_table), 1);
tc_t3c_dir = fullfile(tab_dir, 'task3_purecolor_decoding_timecourses');
for i = 1:height(c07.summary_table)
    sub = c07.summary_table.subject{i}; ch = c07.summary_table.channel{i};
    f = fullfile(tc_t3c_dir, sprintf('%s_%s_task3_purecolor_decoding_timecourse.csv', sub, ch));
    if isfile(f)
        tb = readtable(f);
        m = (tb.p_pointwise < 0.05) & (tb.time_ms >= 0);
        t3c_consec(i) = calc_max_consec(m);
    end
end

% 打印连续点数门槛下的统计
fprintf('\n[按持续 >= 4 点 (>= 80ms) 包含严格显著与接近显著]:\n');
fprintf('  Task 2 Cross : 严格 = %d, 加上近显著(>=4点) = %d\n', sum(t2_c_all.has_sig_cluster == 1), sum(t2_c_all.has_sig_cluster == 1 | t2c_consec >= 4));
fprintf('  Task 2 Direct: 严格 = %d, 加上近显著(>=4点) = %d\n', sum(t2_d.has_sig_cluster == 1), sum(t2_d.has_sig_cluster == 1 | t2d_consec >= 4));
fprintf('  Task 3 Cross : 严格 = %d, 加上近显著(>=4点) = %d\n', sum(c07.summary_table.has_sig_cluster == 1), sum(c07.summary_table.has_sig_cluster == 1 | t3c_consec >= 4));
fprintf('  Task 3 Direct: 严格 = %d, 加上近显著(>=4点) = %d\n', sum(t3_d.has_sig_cluster == 1), sum(t3_d.has_sig_cluster == 1 | t3d_consec >= 4));
fprintf('  Cross-Task   : 严格(p<0.05) = %d, 趋势(p<0.10) = %d\n', sum(tgm.tgm_has_sig_cluster_2d == 1), sum(tgm.tgm_cluster_p_min < 0.10));

% 打印连续点数门槛 >= 5 点 (>= 100ms)
fprintf('\n[按持续 >= 5 点 (>= 100ms) 包含严格显著与接近显著]:\n');
fprintf('  Task 2 Cross : 严格 = %d, 加上近显著(>=5点) = %d\n', sum(t2_c_all.has_sig_cluster == 1), sum(t2_c_all.has_sig_cluster == 1 | t2c_consec >= 5));
fprintf('  Task 2 Direct: 严格 = %d, 加上近显著(>=5点) = %d\n', sum(t2_d.has_sig_cluster == 1), sum(t2_d.has_sig_cluster == 1 | t2d_consec >= 5));
fprintf('  Task 3 Cross : 严格 = %d, 加上近显著(>=5点) = %d\n', sum(c07.summary_table.has_sig_cluster == 1), sum(c07.summary_table.has_sig_cluster == 1 | t3c_consec >= 5));
fprintf('  Task 3 Direct: 严格 = %d, 加上近显著(>=5点) = %d\n', sum(t3_d.has_sig_cluster == 1), sum(t3_d.has_sig_cluster == 1 | t3d_consec >= 5));

function c = calc_max_consec(m)
    c = 0; cur = 0;
    for k = 1:numel(m)
        if m(k), cur = cur + 1; if cur > c, c = cur; end
        else, cur = 0; end
    end
end
