% test_comprehensive_near_sig.m
% 全面计算包含“接近显著”位点时，5大分析的显著通道数与Venn图交集变化

clear; clc;
tab_dir = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';

% 1. 读取 Task 2 Cross 与 Task 3 Cross 的完整时程数据
sc_data = load(fullfile(tab_dir, 'single_channel_decoding_timecourses.mat'));
time_ms = sc_data.time_ms;
post_idx = find(time_ms >= 0);

% 组装 Task 2 Cross (concordant 157 + non_concordant 72 = 229)
t2_cross_items = [sc_data.concordant(:); sc_data.non_concordant(:)];
c06_c = load(fullfile(tab_dir, 'concordant_electrodes_decoding_summary.mat'));
c06_n = load(fullfile(tab_dir, 'non_concordant_electrodes_decoding_summary.mat'));
t2_c_sum = [c06_c.summary_table; c06_n.summary_table];

% 组装 Task 3 Cross (229)
t3_cross_items = sc_data.task3_purecolor(:);
c07 = load(fullfile(tab_dir, 'task3_purecolor_decoding_summary.mat'));
t3_c_sum = c07.summary_table;

% 组装 Task 2 Direct (229)
t2_d = readtable(fullfile(tab_dir, 'task2_direct_decoding_summary.csv'));
tc_t2d_dir = fullfile(tab_dir, 'decoding_task2_direct_timecourses');

% 组装 Task 3 Direct (229)
t3_d = readtable(fullfile(tab_dir, 'task3_direct_decoding_summary.csv'));
tc_t3d_dir = fullfile(tab_dir, 'decoding_task3_direct_timecourses');

% 组装 Cross-Task (157)
tgm = readtable(fullfile(tab_dir, 'cross_decoding_tgm_perm200_summary.csv'));

% 辅助函数: 计算 post_stimulus 下的最大连续显著点数 (p < 0.05)
calc_consec = @(p_vec) get_max_consec_pts(p_vec(post_idx));

% 2. 计算每个通道的最大连续显著点数
% (1) Task 2 Cross
n_t2c = numel(t2_cross_items);
t2c_keys = cell(n_t2c, 1);
t2c_strict = zeros(n_t2c, 1);
t2c_max_pts = zeros(n_t2c, 1);
for i = 1:n_t2c
    t2c_keys{i} = sprintf('%s_%s', t2_cross_items(i).subject, t2_cross_items(i).channel);
    % 匹配严格显著标记
    m_idx = find(strcmp(t2_c_sum.subject, t2_cross_items(i).subject) & strcmp(t2_c_sum.channel, t2_cross_items(i).channel), 1);
    if ~isempty(m_idx)
        t2c_strict(i) = t2_c_sum.has_sig_cluster(m_idx);
    end
    t2c_max_pts(i) = calc_consec(t2_cross_items(i).p_pointwise);
end

% (2) Task 3 Cross
n_t3c = numel(t3_cross_items);
t3c_keys = cell(n_t3c, 1);
t3c_strict = zeros(n_t3c, 1);
t3c_max_pts = zeros(n_t3c, 1);
for i = 1:n_t3c
    t3c_keys{i} = sprintf('%s_%s', t3_cross_items(i).subject, t3_cross_items(i).channel);
    m_idx = find(strcmp(t3_c_sum.subject, t3_cross_items(i).subject) & strcmp(t3_c_sum.channel, t3_cross_items(i).channel), 1);
    if ~isempty(m_idx)
        t3c_strict(i) = t3_c_sum.has_sig_cluster(m_idx);
    end
    t3c_max_pts(i) = calc_consec(t3_cross_items(i).p_pointwise);
end

% (3) Task 2 Direct
n_t2d = height(t2_d);
t2d_keys = cell(n_t2d, 1);
t2d_strict = t2_d.has_sig_cluster;
t2d_max_pts = zeros(n_t2d, 1);
for i = 1:n_t2d
    t2d_keys{i} = sprintf('%s_%s', t2_d.subject{i}, t2_d.channel{i});
    f = fullfile(tc_t2d_dir, sprintf('%s_%s_task2_direct_timecourse.csv', t2_d.subject{i}, t2_d.channel{i}));
    if isfile(f)
        tb = readtable(f);
        t2d_max_pts(i) = calc_consec(tb.p_pointwise);
    end
end

% (4) Task 3 Direct
n_t3d = height(t3_d);
t3d_keys = cell(n_t3d, 1);
t3d_strict = t3_d.has_sig_cluster;
t3d_max_pts = zeros(n_t3d, 1);
for i = 1:n_t3d
    t3d_keys{i} = sprintf('%s_%s', t3_d.subject{i}, t3_d.channel{i});
    f = fullfile(tc_t3d_dir, sprintf('%s_%s_task3_direct_timecourse.csv', t3_d.subject{i}, t3_d.channel{i}));
    if isfile(f)
        tb = readtable(f);
        t3d_max_pts(i) = calc_consec(tb.p_pointwise);
    end
end

% (5) Cross-Task (T3 -> T2)
n_xt = height(tgm);
xt_keys = cell(n_xt, 1);
xt_strict = tgm.tgm_has_sig_cluster_2d;
xt_trend = tgm.tgm_cluster_p_min < 0.10;
for i = 1:n_xt
    xt_keys{i} = sprintf('%s_%s', tgm.subject{i}, tgm.channel{i});
end

fprintf('\n========================================================================\n');
fprintf('  【不同连续时间门槛下的通道数量对比】\n');
fprintf('========================================================================\n');
fprintf('%-15s | 严格显著 | 包含>=3点(60ms) | 包含>=4点(80ms) | 包含>=5点(100ms)\n', '分析');
fprintf('------------------------------------------------------------------------\n');
fprintf('%-15s | %8d | %15d | %15d | %16d\n', 'Task 2 Cross',  sum(t2c_strict), sum(t2c_strict | t2c_max_pts >= 3), sum(t2c_strict | t2c_max_pts >= 4), sum(t2c_strict | t2c_max_pts >= 5));
fprintf('%-15s | %8d | %15d | %15d | %16d\n', 'Task 2 Direct', sum(t2d_strict), sum(t2d_strict | t2d_max_pts >= 3), sum(t2d_strict | t2d_max_pts >= 4), sum(t2d_strict | t2d_max_pts >= 5));
fprintf('%-15s | %8d | %15d | %15d | %16d\n', 'Task 3 Cross',  sum(t3c_strict), sum(t3c_strict | t3c_max_pts >= 3), sum(t3c_strict | t3c_max_pts >= 4), sum(t3c_strict | t3c_max_pts >= 5));
fprintf('%-15s | %8d | %15d | %15d | %16d\n', 'Task 3 Direct', sum(t3d_strict), sum(t3d_strict | t3d_max_pts >= 3), sum(t3d_strict | t3d_max_pts >= 4), sum(t3d_strict | t3d_max_pts >= 5));
fprintf('%-15s | %8d | %15s | %15s | %16s\n', 'Cross-Task 2D', sum(xt_strict), sprintf('趋势(p<0.1): %d', sum(xt_trend)), sprintf('趋势(p<0.1): %d', sum(xt_trend)), sprintf('趋势(p<0.1): %d', sum(xt_trend)));

% 3. 测试以 >=4 点 (80ms，项目既有标准) + Cross-Task (p<0.10) 时的交集分析
k_t2c_relax = t2c_keys(t2c_strict == 1 | t2c_max_pts >= 4);
k_t2d_relax = t2d_keys(t2d_strict == 1 | t2d_max_pts >= 4);
k_t3c_relax = t3c_keys(t3c_strict == 1 | t3c_max_pts >= 4);
k_t3d_relax = t3d_keys(t3d_strict == 1 | t3d_max_pts >= 4);
k_xt_relax  = xt_keys(xt_trend == 1);

fprintf('\n========================================================================\n');
fprintf('  【宽松门槛下 (>=4点/80ms 或 p<0.10) 的交集分析】\n');
fprintf('========================================================================\n');

% 图 1 体系: Task 2 Cross vs Task 3 Cross vs Cross-Task
i_12 = intersect(k_t2c_relax, k_t3c_relax);
i_13 = intersect(k_t2c_relax, k_xt_relax);
i_23 = intersect(k_t3c_relax, k_xt_relax);
i_123 = intersect(i_12, k_xt_relax);

fprintf('【图 1 体系: Cross 解码重叠】\n');
fprintf('  A: Task 2 Cross (N=%d)\n', numel(k_t2c_relax));
fprintf('  B: Task 3 Cross (N=%d)\n', numel(k_t3c_relax));
fprintf('  C: Cross-Task   (N=%d)\n', numel(k_xt_relax));
fprintf('  A & B 重叠通道数: %d\n', numel(i_12));
disp(i_12');
fprintf('  A & C 重叠通道数: %d\n', numel(i_13));
disp(i_13');
fprintf('  B & C 重叠通道数: %d\n', numel(i_23));
disp(i_23');
fprintf('  A & B & C 三者重叠: %d\n', numel(i_123));
disp(i_123');

% 图 2 体系: Task 2 Direct vs Task 3 Direct vs Cross-Task
i_d12 = intersect(k_t2d_relax, k_t3d_relax);
i_d13 = intersect(k_t2d_relax, k_xt_relax);
i_d23 = intersect(k_t3d_relax, k_xt_relax);
i_d123 = intersect(i_d12, k_xt_relax);

fprintf('\n【图 2 体系: Direct 解码重叠】\n');
fprintf('  A: Task 2 Direct (N=%d)\n', numel(k_t2d_relax));
fprintf('  B: Task 3 Direct (N=%d)\n', numel(k_t3d_relax));
fprintf('  C: Cross-Task    (N=%d)\n', numel(k_xt_relax));
fprintf('  A & B 重叠通道数: %d\n', numel(i_d12));
disp(i_d12');
fprintf('  A & C 重叠通道数: %d\n', numel(i_d13));
disp(i_d13');
fprintf('  B & C 重叠通道数: %d\n', numel(i_d23));
disp(i_d23');
fprintf('  A & B & C 三者重叠: %d\n', numel(i_d123));
disp(i_d123');

function max_c = get_max_consec_pts(p_arr)
    m = (p_arr < 0.05);
    max_c = 0; cur = 0;
    for k = 1:numel(m)
        if m(k)
            cur = cur + 1;
            if cur > max_c, max_c = cur; end
        else
            cur = 0;
        end
    end
end
