%% test_inspect_task2_task3_for_cross_decoding.m
% 检查 Task 2 与 Task 3 的数据结构、通道一致性与试次标签

clear; clc;
script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(script_dir));
proc_dir   = fullfile(proj_root, 'process_data_new');
res_dir    = fullfile(proj_root, 'result', 'tables');

fprintf('========================================================\n');
fprintf('  【检查 Task 1 显著电极、Task 2 与 Task 3 数据结构】\n');
fprintf('========================================================\n');

% 1. 加载 C04 显著通道表
c04_mat = fullfile(res_dir, 'color_effects_summary.mat');
if isfile(c04_mat)
    c04 = load(c04_mat);
    if isfield(c04, 'all_tbl'), tbl = c04.all_tbl; else, tbl = c04.res_table; end
    sig_tbl = tbl(tbl.is_significant == 1, :);
    fprintf('[+] C04 汇总表已加载，总体显著通道记录数: %d\n', height(sig_tbl));
    u_subs = unique(sig_tbl.subject);
    for i = 1:numel(u_subs)
        s = u_subs{i};
        st = sig_tbl(strcmp(sig_tbl.subject, s), :);
        u_ch = unique(st.channel);
        fprintf('    - %s: 显著独立电极数 = %d\n', s, numel(u_ch));
    end
else
    fprintf('[-] 未找到 C04 汇总表: %s\n', c04_mat);
end

% 2. 检查 sub001 的 Task 2 和 Task 3 数据
sub = 'sub001';
t2_file = fullfile(proc_dir, sub, 'task2_multiband_epoched.mat');
t3_file = fullfile(proc_dir, sub, 'task3_multiband_epoched.mat');

if isfile(t2_file) && isfile(t3_file)
    d2 = load(t2_file, 'epoched_data'); ep2 = d2.epoched_data;
    d3 = load(t3_file, 'epoched_data'); ep3 = d3.epoched_data;
    
    fprintf('\n[+] 被试 %s 数据检查:\n', sub);
    fprintf('    Task 2: 通道数 = %d, 采样点数 = %d, 时间范围 = [%.1f, %.1f] ms\n', ...
        numel(ep2.channels), numel(ep2.time_ms), ep2.time_ms(1), ep2.time_ms(end));
    fprintf('    Task 3: 通道数 = %d, 采样点数 = %d, 时间范围 = [%.1f, %.1f] ms\n', ...
        numel(ep3.channels), numel(ep3.time_ms), ep3.time_ms(1), ep3.time_ms(end));
    
    % 通道对齐检查
    same_ch = isequal(ep2.channels, ep3.channels);
    fprintf('    Task 2 与 Task 3 通道列表完全一致: %d\n', same_ch);
    
    % Task 3 红绿试次统计
    t3_ti = ep3.trial_info;
    n_t3_red = sum(strcmp(t3_ti.color, 'red'));
    n_t3_grn = sum(strcmp(t3_ti.color, 'green'));
    fprintf('    Task 3 试次: Red = %d, Green = %d (总红绿 = %d)\n', ...
        n_t3_red, n_t3_grn, n_t3_red + n_t3_grn);
    
    % Task 2 灰色水果试次统计
    t2_ti = ep2.trial_info;
    m_t2_gray = strcmp(t2_ti.state, 'gray');
    t2_gray = t2_ti(m_t2_gray, :);
    n_t2_red = sum(strcmp(t2_gray.memory_color, 'red'));
    n_t2_grn = sum(strcmp(t2_gray.memory_color, 'green'));
    fprintf('    Task 2 Gray 试次: Red Memory = %d, Green Memory = %d (总 Gray = %d)\n', ...
        n_t2_red, n_t2_grn, n_t2_red + n_t2_grn);
    
    % 打印 Task 2 更多状态 (True / False)
    u_states = unique(t2_ti.state);
    fprintf('    Task 2 全部状态试次分布: ');
    for st_i = 1:numel(u_states)
        st_name = u_states{st_i};
        fprintf('%s=%d  ', st_name, sum(strcmp(t2_ti.state, st_name)));
    end
    fprintf('\n');
end
fprintf('========================================================\n');
