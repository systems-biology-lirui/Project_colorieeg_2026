% test_check_trials_and_chans.m
% 检查 Task 2 与 Task 3 的试次数分布以及 Task 1 显著通道列表

clear; clc;
test_dir   = fileparts(mfilename('fullpath'));
matlab_dir = fileparts(test_dir);
work_dir   = fileparts(matlab_dir); % e:/liulab_project/Project_colorieeg_2026/color_analyse_0825
data_root  = fullfile(work_dir, 'process_data_new');
res_root   = fullfile(work_dir, 'result');
c04_table  = fullfile(res_root, 'tables', 'color_effects_summary.mat');

% 1. 检查电极列表
c04_data = load(c04_table);
if isfield(c04_data, 'all_tbl')
    tbl = c04_data.all_tbl;
else
    tbl = c04_data.res_table;
end

sig_mask = (tbl.is_significant == 1);
tbl_sig = tbl(sig_mask, :);
[~, u_idx] = unique(strcat(tbl_sig.subject, '_', tbl_sig.channel), 'stable');
tbl_sig_u = tbl_sig(u_idx, :);

fprintf('Task 1 总体显著电极总数 (All Significant): %d\n', height(tbl_sig_u));

concord_mask = (tbl_sig_u.concordance_type == "Concordant_Positive") | ...
               (tbl_sig_u.concordance_type == "Concordant_Negative");
fprintf('  - 同向显著电极 (Concordant): %d\n', sum(concord_mask));
fprintf('  - 不同向显著电极 (Non-Concordant): %d\n', sum(~concord_mask));

% 2. 检查各被试在 Task 2 和 Task 3 中的试次数
subs = unique(tbl_sig_u.subject, 'stable');
fprintf('\n各被试 Task 2 (Gray) 与 Task 3 (Pure Red/Green) 试次数统计:\n');
fprintf('%-10s | %-12s | %-12s | %-12s | %-12s\n', 'Subject', 'T2 RedGray', 'T2 GreenGray', 'T3 RedPure', 'T3 GreenPure');
fprintf('----------------------------------------------------------------------\n');

for i = 1:numel(subs)
    s_id = subs{i};
    % Task 2
    f2 = fullfile(data_root, s_id, 'task2_multiband_epoched.mat');
    if isfile(f2)
        d2 = load(f2, 'epoched_data');
        ti2 = d2.epoched_data.trial_info;
        gm = strcmp(ti2.state, 'gray');
        ti2_g = ti2(gm, :);
        t2_r = sum(strcmp(ti2_g.memory_color, 'red'));
        t2_g = sum(strcmp(ti2_g.memory_color, 'green'));
    else
        t2_r = NaN; t2_g = NaN;
    end
    
    % Task 3
    f3 = fullfile(data_root, s_id, 'task3_multiband_epoched.mat');
    if isfile(f3)
        d3 = load(f3, 'epoched_data');
        ti3 = d3.epoched_data.trial_info;
        t3_r = sum(strcmp(ti3.color, 'red'));
        t3_g = sum(strcmp(ti3.color, 'green'));
    else
        t3_r = NaN; t3_g = NaN;
    end
    fprintf('%-10s | %-12d | %-12d | %-12d | %-12d\n', s_id, t2_r, t2_g, t3_r, t3_g);
end
