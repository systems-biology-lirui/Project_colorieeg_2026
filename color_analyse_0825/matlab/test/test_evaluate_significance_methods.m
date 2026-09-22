% test_evaluate_significance_methods.m
% 评估对比不同显著性检验方案在当前解码数据上的表现
% 对比方案:
%   1. 当前 Cluster-Mass FWE (全时程 0-800ms)
%   2. 持续时间置换检验 (Consecutive Duration Permutation)
%   3. FDR 逐点校正 (Benjamini-Hochberg q < 0.05)
%   4. 先验时间窗受限检验 (Restricted Window 100-400ms / 200-600ms)
%   5. 宽松时间簇 (Cluster p < 0.10)

clear; clc;
tab_dir = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';

% 以 Task 3 Within-Shape (C17) 和 Task 3 Direct (C14) 为例测试
tc_dir_c17 = fullfile(tab_dir, 'decoding_task3_within_shape_timecourses');
sum_c17    = readtable(fullfile(tab_dir, 'task3_within_shape_decoding_summary.csv'));
n_elecs    = height(sum_c17);

% 统计各类别的通道数
n_cluster_05   = sum(sum_c17.has_sig_cluster == 1);
n_consec_4pts  = 0; % >= 4 点 (80ms)
n_consec_3pts  = 0; % >= 3 点 (60ms)
n_fdr_05       = 0; % FDR q < 0.05
n_fdr_10       = 0; % FDR q < 0.10
n_early_win_05 = 0; % 早期 100-400ms 内点显著
n_late_win_05  = 0; % 晚期 200-600ms 内点显著

for i = 1:n_elecs
    f = fullfile(tc_dir_c17, sprintf('%s_%s_task3_within_shape_timecourse.csv', sum_c17.subject{i}, sum_c17.channel{i}));
    if isfile(f)
        tb = readtable(f);
        post_m = tb.time_ms >= 0;
        p_post = tb.p_pointwise(post_m);
        t_post = tb.time_ms(post_m);
        
        % 1. 连续点数
        c = 0; max_c = 0;
        for k = 1:numel(p_post)
            if p_post(k) < 0.05, c = c + 1; if c > max_c, max_c = c; end
            else, c = 0; end
        end
        if max_c >= 4, n_consec_4pts = n_consec_4pts + 1; end
        if max_c >= 3, n_consec_3pts = n_consec_3pts + 1; end
        
        % 2. FDR 校正 (针对刺激后时间点)
        p_sorted = sort(p_post);
        m_pts = numel(p_post);
        fdr_thresh = (1:m_pts)' / m_pts * 0.05;
        if any(p_sorted <= fdr_thresh)
            n_fdr_05 = n_fdr_05 + 1;
        end
        fdr_thresh_10 = (1:m_pts)' / m_pts * 0.10;
        if any(p_sorted <= fdr_thresh_10)
            n_fdr_10 = n_fdr_10 + 1;
        end
        
        % 3. 先验窗 (100-400ms: 颜色知觉经典窗)
        w_early = (t_post >= 100 & t_post <= 400);
        if any(p_post(w_early) < 0.01) || sum(p_post(w_early) < 0.05) >= 2
            n_early_win_05 = n_early_win_05 + 1;
        end
        
        % 4. 先验窗 (200-600ms: 维持窗)
        w_late = (t_post >= 200 & t_post <= 600);
        if any(p_post(w_late) < 0.01) || sum(p_post(w_late) < 0.05) >= 2
            n_late_win_05 = n_late_win_05 + 1;
        end
    end
end

fprintf('========================================================================\n');
fprintf('  【Task 3 同形状内解码不同显著性方案结果对比 (总电极数 N = %d)】\n', n_elecs);
fprintf('========================================================================\n');
fprintf('1. 当前全时程 Cluster-Mass FWE (p < 0.05)           : %d 个 (占比 %.1f%%)\n', n_cluster_05, n_cluster_05/n_elecs*100);
fprintf('2. 连续时间门槛 (连续 >= 4点, 即持续 >= 80ms)        : %d 个 (占比 %.1f%%)\n', n_consec_4pts, n_consec_4pts/n_elecs*100);
fprintf('3. 连续时间门槛 (连续 >= 3点, 即持续 >= 60ms)        : %d 个 (占比 %.1f%%)\n', n_consec_3pts, n_consec_3pts/n_elecs*100);
fprintf('4. FDR 错误发现率校正 (刺激后 Benjamini-Hochberg q<0.05): %d 个 (占比 %.1f%%)\n', n_fdr_05, n_fdr_05/n_elecs*100);
fprintf('5. FDR 错误发现率校正 (刺激后 Benjamini-Hochberg q<0.10): %d 个 (占比 %.1f%%)\n', n_fdr_10, n_fdr_10/n_elecs*100);
fprintf('6. 先验时间窗受限检验 (100-400ms 知觉窗连续显著)      : %d 个 (占比 %.1f%%)\n', n_early_win_05, n_early_win_05/n_elecs*100);
fprintf('7. 先验时间窗受限检验 (200-600ms 维持窗连续显著)      : %d 个 (占比 %.1f%%)\n', n_late_win_05, n_late_win_05/n_elecs*100);
