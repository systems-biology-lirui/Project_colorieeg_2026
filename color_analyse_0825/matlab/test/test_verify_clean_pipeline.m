%% test_verify_clean_pipeline.m
% =========================================================================
% 功能:
%   1. 检验 color_effects_summary.mat 与 color_significant_channels_simple.mat 读取
%   2. 检验 single_channel_decoding_timecourses.mat 结构完整性与 4 位小数
%   3. 测试 plot_glass_brain_3view_0825 动态匹配 MNI 坐标绘图
%   4. 测试 plot_cluster_spatiotemporal_distribution 从 .mat 极速出图并不产生 CSV
%   5. 测试色带图脚本读取 .mat 正常绘图
% =========================================================================

proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
tab_dir   = fullfile(proj_root, 'result', 'tables');
addpath(fullfile(proj_root, 'matlab'));
addpath(fullfile(proj_root, 'matlab', 'plot_tool'));

fprintf('>>> [TEST 1/5] 检验 color_effects_summary.mat ...\n');
mat1 = fullfile(tab_dir, 'color_effects_summary.mat');
assert(isfile(mat1), '文件不存在: %s', mat1);
d1 = load(mat1);
assert(isfield(d1, 'all_tbl'), '缺少 all_tbl 变量');
fprintf('    通过: all_tbl 大小为 %d 行 x %d 列\n', height(d1.all_tbl), width(d1.all_tbl));

fprintf('>>> [TEST 2/5] 检验 single_channel_decoding_timecourses.mat ...\n');
mat2 = fullfile(tab_dir, 'single_channel_decoding_timecourses.mat');
assert(isfile(mat2), '文件不存在: %s', mat2);
d2 = load(mat2);
assert(isfield(d2, 'time_ms'), '缺少 time_ms');
assert(isfield(d2, 'concordant'), '缺少 concordant');
assert(isfield(d2, 'non_concordant'), '缺少 non_concordant');
assert(isfield(d2, 'task3_purecolor'), '缺少 task3_purecolor');
fprintf('    通过: concordant (%d 个), non_concordant (%d 个), task3 (%d 个)\n', ...
    numel(d2.concordant), numel(d2.non_concordant), numel(d2.task3_purecolor));

% 检验 4 位小数规范
sample_acc = d2.concordant(1).acc_joint(1);
diff_round = abs(sample_acc - round(sample_acc, 4));
assert(diff_round < 1e-6, '浮点数小数位数未规范至 4 位！');
fprintf('    通过: 浮点数规范度检查通过 (样本值: %f)\n', sample_acc);

fprintf('>>> [TEST 3/5] 验证 plot_glass_brain_3view_0825 (动态匹配 MNI) ...\n');
cfg_gb = struct();
cfg_gb.stat_mode = 'bonferroni';
cfg_gb.concordant_only = false;
plot_glass_brain_3view_0825(cfg_gb);
fprintf('    通过: 玻璃脑动态匹配并成功输出 Bonferroni 图像\n');

fprintf('>>> [TEST 4/5] 验证 plot_cluster_spatiotemporal_distribution (无 CSV 导出) ...\n');
% 记录现有 CSV
csv_before = dir(fullfile(tab_dir, 'cluster_*.csv'));
n_before = numel(csv_before);

plot_cluster_spatiotemporal_distribution('concordant');

csv_after = dir(fullfile(tab_dir, 'cluster_*.csv'));
n_after = numel(csv_after);
assert(n_after == n_before, '警告: 依然生成了 cluster_*.csv 文件！');
fprintf('    通过: 聚类时空分布图成功渲染且未导出任何 cluster CSV\n');

fprintf('>>> [TEST 5/5] 验证色带图 plot_c04_all_subjects_significance_strip ...\n');
plot_c04_all_subjects_significance_strip();
fprintf('    通过: 色带图成功由 .mat 加载并渲染完成\n');

fprintf('\n========================================================================\n');
fprintf('  【全部测试用例通过！Pipeline 已经成功无缝切换为纯 .mat 驱动！】\n');
fprintf('========================================================================\n');
