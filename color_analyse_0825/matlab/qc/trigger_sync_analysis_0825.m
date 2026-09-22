function audit_table = trigger_sync_analysis_0825(cfg_in)
%% ========================================================================
% 函数名称: trigger_sync_analysis_0825.m
% 所在路径: color_analyse_0825/matlab/qc/
% 功能: 刺激呈现 Trigger 与 SEEG 采集端 Marker 的时序同步性与一致性分析 (QC 质检模块)
%
% 输入:
%   cfg_in (可选结构体):
%     .subjects    - 被试列表 (默认 {'test001'} 或全部)
%     .tasks       - 任务列表 (默认 [1, 2, 3])
%     .save_qc_fig - 是否保存 3合1 质检图 (默认 true)
%     .save_csv    - 是否保存审计表格 (默认 true)
%
% 输出:
%   audit_table  - 包含各任务对齐一致率、ITI 相关性、Jitter 的统计表格
%% ========================================================================

if nargin < 1, cfg_in = struct(); end

% 默认配置
cfg = struct();
cfg.subjects     = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008', 'sub009'};
cfg.tasks        = [1, 2, 3];
cfg.save_qc_fig  = true;
cfg.save_csv     = true;

% 用户参数覆盖
fn = fieldnames(cfg_in);
for i = 1:numel(fn)
    cfg.(fn{i}) = cfg_in.(fn{i});
end

script_dir = fileparts(mfilename('fullpath')); % .../color_analyse_0825/matlab/qc
matlab_dir = fileparts(script_dir);            % .../color_analyse_0825/matlab
anal_dir   = fileparts(matlab_dir);            % .../color_analyse_0825
proj_root  = fileparts(anal_dir);              % 项目根目录
qc_dir     = fullfile(anal_dir, 'qc');
meta_dir   = fullfile(anal_dir, 'metadata');
if ~exist(qc_dir, 'dir'), mkdir(qc_dir); end
if ~exist(meta_dir, 'dir'), mkdir(meta_dir); end

fprintf('========================================================================\n');
fprintf('           【QC 质检：采集端 Marker 与行为端 Trigger 一致性比对】        \n');
fprintf('========================================================================\n');

audit_results = struct([]);

for sub_idx = 1:numel(cfg.subjects)
    sub_id = cfg.subjects{sub_idx};
    sub_num_str = regexprep(sub_id, '^(test|sub)0*', '');
    
    for task_num = cfg.tasks
        fprintf('\n>>> 正在质检 [%s Task %d]...\n', sub_id, task_num);
        
        % 1. 加载 SEEG 原始事件
        seeg_file = fullfile(proj_root, 'seegdata', sub_id, sprintf('task%d.mat', task_num));
        if ~exist(seeg_file, 'file')
            fprintf('[-] 未找到 SEEG 文件: %s (跳过)\n', seeg_file);
            continue;
        end
        seeg = load(seeg_file);
        fs = double(seeg.fs);
        
        eeg_events = seeg.event;
        n_eeg_raw  = numel(eeg_events);
        eeg_trigs  = nan(1, n_eeg_raw);
        eeg_lats   = nan(1, n_eeg_raw);
        for ev_i = 1:n_eeg_raw
            ev_val = eeg_events(ev_i).type;
            if ischar(ev_val) || isstring(ev_val)
                n = regexp(char(ev_val), '\d+', 'match', 'once');
                if ~isempty(n), eeg_trigs(ev_i) = str2double(n); end
            elseif isnumeric(ev_val)
                eeg_trigs(ev_i) = double(ev_val);
            end
            
            ev_lat = eeg_events(ev_i).latency;
            if iscell(ev_lat), ev_lat = ev_lat{1}; end
            eeg_lats(ev_i) = double(ev_lat);
        end
        valid_eeg_mask = isfinite(eeg_trigs);
        eeg_trigs = eeg_trigs(valid_eeg_mask);
        eeg_lats  = eeg_lats(valid_eeg_mask);
        eeg_times = (eeg_lats - 1) / fs;
        n_eeg     = numel(eeg_trigs);
        
        % 2. 检索并加载行为日志文件
        test_id_alt = sprintf('test%03d', str2double(sub_num_str));
        beh_dir = fullfile(proj_root, 'visual_experiment', 'Data', sub_id);
        if ~exist(beh_dir, 'dir')
            beh_dir = fullfile(proj_root, 'visual_experiment', 'Data', test_id_alt);
        end
        
        switch task_num
            case 1
                beh_files = dir(fullfile(beh_dir, '*Task1*Passive*Session*.mat'));
            case 2
                beh_files = dir(fullfile(beh_dir, '*Passive*FruitFull*Session*.mat'));
                if isempty(beh_files)
                    beh_files = dir(fullfile(beh_dir, '*Task2*Passive*Session*.mat'));
                end
            case 3
                beh_files = dir(fullfile(beh_dir, '*Passive*ColorPatches*Session*.mat'));
                if isempty(beh_files)
                    beh_files = dir(fullfile(beh_dir, '*Task3*Passive*Session*.mat'));
                end
        end
        
        if isempty(beh_files)
            fprintf('[-] 未找到行为日志文件 (跳过)\n');
            continue;
        end
        
        [~, sort_idx] = sort({beh_files.name});
        beh_files = beh_files(sort_idx);
        
        all_results = [];
        beh_session_id = [];
        for f_i = 1:numel(beh_files)
            beh_mat_path = fullfile(beh_files(f_i).folder, beh_files(f_i).name);
            beh_data = load(beh_mat_path,'result');
            if isfield(beh_data, 'results')
                curr_res = beh_data.results(:);
                all_results = [all_results; curr_res]; %#ok<AGROW>
                beh_session_id = [beh_session_id, repmat(f_i, 1, numel(curr_res))]; %#ok<AGROW>
            end
        end
        
        beh_markers = [all_results.marker];
        beh_onsets  = [all_results.onsetTime];
        beh_catch   = [all_results.isCatch];
        n_beh       = numel(beh_markers);
        n_catch     = sum(beh_catch == 1);
        
        % 3. LCS 动态序列对齐
        [matched_eeg_idx, matched_beh_idx] = align_lcs_local(eeg_trigs, beh_markers);
        n_matched = numel(matched_eeg_idx);
        match_rate = (n_matched / max(n_eeg, n_beh)) * 100;
        
        % 4. 统计同 Session 内试次间期 (ITI) 精度
        matched_eeg_times = eeg_times(matched_eeg_idx);
        matched_beh_times = beh_onsets(matched_beh_idx);
        matched_sess      = beh_session_id(matched_beh_idx);
        
        same_sess_mask = (matched_sess(1:end-1) == matched_sess(2:end));
        eeg_iti_all    = diff(matched_eeg_times);
        beh_iti_all    = diff(matched_beh_times);
        
        valid_eeg_iti  = eeg_iti_all(same_sess_mask);
        valid_beh_iti  = beh_iti_all(same_sess_mask);
        
        iti_corr       = corr(valid_eeg_iti(:), valid_beh_iti(:));
        iti_diff_ms    = (valid_eeg_iti - valid_beh_iti) * 1000;
        mean_jitter_ms = mean(iti_diff_ms);
        std_jitter_ms  = std(iti_diff_ms);
        max_jitter_ms  = max(abs(iti_diff_ms));
        
        fprintf('    [统计概览] SEEG 事件: %d | 行为试次: %d (Catch: %d) | 配对率: %.2f%%\n', ...
            n_eeg, n_beh, n_catch, match_rate);
        fprintf('    [时序精度] ITI 相关系数 r = %.6f, Jitter 均值: %.3f ms, 标准差: %.3f ms\n', ...
            iti_corr, mean_jitter_ms, std_jitter_ms);
        
        % 记录结构体
        rec = struct();
        rec.subject        = sub_id;
        rec.task_num       = task_num;
        rec.n_eeg_events   = n_eeg;
        rec.n_beh_trials   = n_beh;
        rec.n_catch_trials = n_catch;
        rec.n_matched      = n_matched;
        rec.match_rate_pct = match_rate;
        rec.iti_corr_r     = iti_corr;
        rec.mean_jitter_ms = mean_jitter_ms;
        rec.std_jitter_ms  = std_jitter_ms;
        rec.max_jitter_ms  = max_jitter_ms;
        audit_results = [audit_results; rec]; %#ok<AGROW>
        
        % 5. 绘图保存
        if cfg.save_qc_fig
            fig = figure('Name', sprintf('Trigger Sync QC: %s Task %d', sub_id, task_num), ...
                'Position', [100, 100, 1200, 400], 'Visible', 'off', 'Color', 'w');
            
            subplot(1, 3, 1);
            plot(matched_beh_idx, beh_markers(matched_beh_idx), 'bo', 'MarkerSize', 5, 'DisplayName', 'Behavior Log');
            hold on;
            plot(matched_beh_idx, eeg_trigs(matched_eeg_idx), 'r.', 'MarkerSize', 8, 'DisplayName', 'SEEG Event');
            xlabel('Trial Index'); ylabel('Trigger Code');
            title(sprintf('Trigger 序列一致性 (一致率: %.1f%%)', match_rate));
            legend('Location', 'best'); grid on; box on;
            
            subplot(1, 3, 2);
            scatter(valid_beh_iti, valid_eeg_iti, 25, [0.2 0.4 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
            hold on;
            ref_line = [min([valid_beh_iti, valid_eeg_iti]), max([valid_beh_iti, valid_eeg_iti])];
            plot(ref_line, ref_line, 'r--', 'LineWidth', 1.5);
            xlabel('Behavior ITI (s)'); ylabel('SEEG ITI (s)');
            title(sprintf('试次间期相关性 (r = %.6f)', iti_corr));
            grid on; box on;
            
            subplot(1, 3, 3);
            histogram(iti_diff_ms, 25, 'FaceColor', [0.3 0.7 0.4], 'EdgeColor', 'k');
            xlabel('\Delta ITI (SEEG - Behavior, ms)'); ylabel('Count');
            title(sprintf('时序误差分布 (\\mu=%.2fms, \\sigma=%.2fms)', mean_jitter_ms, std_jitter_ms));
            grid on; box on;
            
            fig_path = fullfile(qc_dir, sprintf('trigger_sync_%s_task%d.png', sub_id, task_num));
            saveas(fig, fig_path);
            close(fig);
        end
    end
end

audit_table = struct2table(audit_results);
if cfg.save_csv && ~isempty(audit_table)
    csv_path = fullfile(meta_dir, 'C01_Trigger对齐与标签审计.csv');
    writetable(audit_table, csv_path, 'Encoding', 'UTF-8');
    fprintf('\n[+] 质检审计汇总表已更新: %s\n', csv_path);
end
end

%% LCS 局部对齐函数
function [matched_idx_a, matched_idx_b] = align_lcs_local(seq_a, seq_b)
    na = numel(seq_a);
    nb = numel(seq_b);
    
    L = zeros(na + 1, nb + 1, 'uint16');
    for i = na:-1:1
        for j = nb:-1:1
            if seq_a(i) == seq_b(j)
                L(i, j) = L(i + 1, j + 1) + 1;
            else
                L(i, j) = max(L(i + 1, j), L(i, j + 1));
            end
        end
    end
    
    matched_idx_a = [];
    matched_idx_b = [];
    i = 1; j = 1;
    while i <= na && j <= nb
        if seq_a(i) == seq_b(j) && L(i, j) == L(i + 1, j + 1) + 1
            matched_idx_a(end + 1) = i; %#ok<AGROW>
            matched_idx_b(end + 1) = j; %#ok<AGROW>
            i = i + 1;
            j = j + 1;
        elseif L(i + 1, j) >= L(i, j + 1)
            i = i + 1;
        else
            j = j + 1;
        end
    end
end
