% inspect_c17_results.m
clear; clc;
tab_dir = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';
tbl = readtable(fullfile(tab_dir, 'task3_within_shape_decoding_summary.csv'));

fprintf('=== 严格显著通道 (Cluster-Mass FWE p < 0.05, 共 %d 个) ===\n', sum(tbl.has_sig_cluster == 1));
sig_tbl = tbl(tbl.has_sig_cluster == 1, :);
disp(sig_tbl(:, {'subject', 'channel', 'dkt_anatomy', 'stream', 'peak_acc_avg', 'peak_time_ms', 'peak_acc_shape1', 'peak_acc_shape2', 'peak_acc_shape3'}));

% 检查持续时间（连续点数 >= 4，即 >= 80ms 的接近显著通道）
tc_dir = fullfile(tab_dir, 'decoding_task3_within_shape_timecourses');
n_near = 0;
near_list = {};
for i = 1:height(tbl)
    f = fullfile(tc_dir, sprintf('%s_%s_task3_within_shape_timecourse.csv', tbl.subject{i}, tbl.channel{i}));
    if isfile(f)
        tb = readtable(f);
        m = (tb.p_pointwise < 0.05) & (tb.time_ms >= 0);
        c = 0; max_c = 0;
        for k = 1:numel(m)
            if m(k), c = c + 1; if c > max_c, max_c = c; end
            else, c = 0; end
        end
        if tbl.has_sig_cluster(i) == 0 && max_c >= 4
            n_near = n_near + 1;
            near_list{end+1, 1} = sprintf('%s-%s', tbl.subject{i}, tbl.channel{i});
        end
    end
end
fprintf('\n=== 接近显著通道 (连续 >= 4 个点, 即 >= 80ms, 共 %d 个) ===\n', n_near);
disp(near_list');
