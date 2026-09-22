% test_inspect_tc.m
clear; clc;
tab_dir = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';

% 检查 task2 direct 的 timecourse CSV
tc_dir_t2 = fullfile(tab_dir, 'decoding_task2_direct_timecourses');
files_t2 = dir(fullfile(tc_dir_t2, '*.csv'));

fprintf('Task 2 Direct: total %d CSVs\n', numel(files_t2));
max_consec_pts = zeros(numel(files_t2), 1);
for i = 1:numel(files_t2)
    tbl = readtable(fullfile(tc_dir_t2, files_t2(i).name));
    % 仅考虑刺激后 (time_ms >= 0)
    post_mask = tbl.time_ms >= 0;
    sig_pts = (tbl.p_pointwise < 0.05) & post_mask;
    
    % 计算最大连续显著点数
    consec = 0; max_c = 0;
    for k = 1:numel(sig_pts)
        if sig_pts(k)
            consec = consec + 1;
            if consec > max_c, max_c = consec; end
        else
            consec = 0;
        end
    end
    max_consec_pts(i) = max_c;
end

fprintf('Max consecutive points distribution in Task 2 Direct:\n');
for c = 0:10
    fprintf('  >= %d consecutive points (%d ms): %d channels\n', c, c*20, sum(max_consec_pts >= c));
end
