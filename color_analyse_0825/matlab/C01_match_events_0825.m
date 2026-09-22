%% ========================================================================
% 脚本名称: C01_match_events_0825.m
% 功能:
%   1. 提取 seegdata/<sub_id>/task<task_num>.mat 中的 SEEG Trigger 事件
%   2. 提取 visual_experiment/Data/<sub_id> 中的行为日志 (*Session*.mat)
%   3. 执行 LCS 动态规划序列对齐 (剔除 Catch Trial，容忍偶发丢包与首尾截断)
%   4. 提取刺激图片名称 (img_name) 与尾号 (pic_id，例如 face_color_01.bmp -> 1)
%   5. 生成标准化试次表 (trial_info) 并保存至 process_data 与 metadata
%
% 输入:
%   - seegdata/<sub_id>/task<task_num>.mat
%   - visual_experiment/Data/<sub_id>/*Session*.mat
%
% 输出:
%   - task_info/<sub_id>/task<task_num>_trial_info.mat
%   - metadata/C01_事件图片对齐记录.csv
% ========================================================================

clear; clc; close all;
% 下面这行不要删除，是为了关闭某种警告提醒
warning('off', 'MATLAB:load:cannotInstantiateLoadedVariable');

%% 1. 参数与路径配置 (参数集中平铺，直观可调)
cfg = struct();
cfg.subjects  = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008', 'sub009'};
cfg.tasks     = [1, 2, 3];
cfg.orig_fs   = 1000;         % 原始脑电采样率 (Hz)
cfg.target_fs = 500;          % 目标重采样率 (Hz)
cfg.run_qc    = false;         % 是否先调用 trigger_sync_analysis 进行时序同步 QC

script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(script_dir)); % 项目根目录
seeg_dir      = fullfile(proj_root, 'seegdata');
beh_dir       = fullfile(proj_root, 'visual_experiment', 'Data');
fb_beh_dir    = fullfile(proj_root, 'color_analyse_0727', 'Data'); % 备选行为路径
task_info_dir = fullfile(proj_root, 'color_analyse_0825', 'task_info');
proc_dir      = fullfile(proj_root, 'color_analyse_0825', 'process_data');
meta_dir      = fullfile(proj_root, 'color_analyse_0825', 'metadata');

if ~exist(task_info_dir, 'dir'), mkdir(task_info_dir); end
if ~exist(proc_dir, 'dir'), mkdir(proc_dir); end
if ~exist(meta_dir, 'dir'), mkdir(meta_dir); end
addpath(fullfile(script_dir, 'qc'));

%% 2. 注册各任务 Trigger 与条件定义
% Task 1: 真实/灰度物体 (8种条件)
t1_trigs = [11, 12, 21, 22, 31, 32, 41, 42];
t1_conds = {'face_color', 'face_gray', 'object_color', 'object_gray', ...
            'body_color', 'body_gray', 'place_color', 'place_gray'};
map_t1 = containers.Map(t1_trigs, t1_conds);

% Task 2: 水果记忆 (12种条件)
t2_trigs = [101:103, 111:113, 121:123, 131:133];
t2_fruits = {'cabbage','cabbage','cabbage', 'kiwi','kiwi','kiwi', ...
             'strawberry','strawberry','strawberry', 'watermelon','watermelon','watermelon'};
t2_states = {'true','false','gray', 'true','false','gray', ...
             'true','false','gray', 'true','false','gray'};
t2_memcols= {'green','green','green', 'green','green','green', ...
             'red','red','red', 'red','red','red'};
map_t2_fruit  = containers.Map(t2_trigs, t2_fruits);
map_t2_state  = containers.Map(t2_trigs, t2_states);
map_t2_memcol = containers.Map(t2_trigs, t2_memcols);

% Task 3: 纯色色块 (6种颜色)
t3_trigs = [51, 52, 53, 54, 55, 56];
t3_colors= {'red', 'yellow', 'blue', 'green', 'black', 'white'};
map_t3 = containers.Map(t3_trigs, t3_colors);

%% 3. 可选阶段：执行全被试 Trigger 时序与一致性 QC 质检
if cfg.run_qc
    fprintf('========================================================================\n');
    fprintf('           【阶段一：调用 QC 模块执行 Marker 对齐与时序质检】             \n');
    fprintf('========================================================================\n');
    cfg_qc = struct();
    cfg_qc.subjects    = cfg.subjects;
    cfg_qc.tasks       = cfg.tasks;
    cfg_qc.save_qc_fig = true;
    cfg_qc.save_csv    = true;
    try
        qc_table = trigger_sync_analysis_0825(cfg_qc);
        fprintf('\n[+] QC 质检完成，共审计 %d 个 Session 数据。\n\n', height(qc_table));
    catch ME
        fprintf(2, '[!] QC 模块执行提示: %s\n', ME.message);
    end
end

%% 4. 核心对齐：提取图片信息并生成各被试/任务 trial_info
fprintf('========================================================================\n');
fprintf('     【阶段二：提取刺激图片信息 (pic_id) 并保存标准 trial_info】          \n');
fprintf('========================================================================\n');

align_records = struct([]);

for s_i = 1:numel(cfg.subjects)
    sub_id = cfg.subjects{s_i};
    sub_out_dir = fullfile(task_info_dir, sub_id);
    if ~exist(sub_out_dir, 'dir'), mkdir(sub_out_dir); end
    
    sub_num_str = regexprep(sub_id, '^(test|sub)0*', '');
    test_id_alt = sprintf('test%03d', str2double(sub_num_str));
    
    % 定位行为数据目录 
    sub_beh_dir = fullfile(beh_dir, sub_id);
    if ~exist(sub_beh_dir, 'dir'), sub_beh_dir = fullfile(beh_dir, test_id_alt); end
    
    for task_num = cfg.tasks
        seeg_file = fullfile(seeg_dir, sub_id, sprintf('task%d.mat', task_num));
        if ~exist(seeg_file, 'file')
            fprintf('[-] [%s Task%d] 未找到 SEEG 数据文件: %s (跳过)\n', sub_id, task_num, seeg_file);
            continue;
        end
        
        fprintf('>>> [%s Task%d] 正在提取与对齐事件...\n', sub_id, task_num);
        
        % (1) 读取 SEEG 原始事件
        seeg_mat = load(seeg_file, 'event', 'fs');
        if isfield(seeg_mat, 'fs') && ~isempty(seeg_mat.fs)
            cur_fs = double(seeg_mat.fs);
        else
            cur_fs = cfg.orig_fs;
        end
        raw_events = seeg_mat.event;
        n_raw_ev = numel(raw_events);
        
        eeg_trigs = nan(1, n_raw_ev);
        eeg_lats_raw = nan(1, n_raw_ev);
        eeg_lats_500 = nan(1, n_raw_ev);
        
        for ev_i = 1:n_raw_ev
            ev_val = raw_events(ev_i).type;
            if ischar(ev_val) || isstring(ev_val)
                num_str = regexp(char(ev_val), '\d+', 'match', 'once');
                if ~isempty(num_str), eeg_trigs(ev_i) = str2double(num_str); end
            elseif isnumeric(ev_val)
                eeg_trigs(ev_i) = double(ev_val);
            end
            
            ev_lat = raw_events(ev_i).latency;
            if iscell(ev_lat), ev_lat = ev_lat{1}; end
            lat_val = double(ev_lat);
            eeg_lats_raw(ev_i) = lat_val;
            eeg_lats_500(ev_i) = round((lat_val - 1) * (cfg.target_fs / cur_fs)) + 1;
        end
        
        % 筛选当前任务的目标 Trigger
        switch task_num
            case 1, valid_mask = ismember(eeg_trigs, t1_trigs);
            case 2, valid_mask = ismember(eeg_trigs, t2_trigs);
            case 3, valid_mask = ismember(eeg_trigs, t3_trigs);
        end
        
        cand_eeg_trigs    = eeg_trigs(valid_mask);
        cand_eeg_lats_raw = eeg_lats_raw(valid_mask);
        cand_eeg_lats_500 = eeg_lats_500(valid_mask);
        n_cand_eeg        = numel(cand_eeg_trigs);
        
        % (2) 检索行为日志文件
        if ~exist(sub_beh_dir, 'dir')
            fprintf('    [-] 未找到行为日志目录: %s (跳过)\n', sub_beh_dir);
            continue;
        end
        
        switch task_num
            case 1
                beh_files = dir(fullfile(sub_beh_dir, '*Task1*Passive*Session*.mat'));
            case 2
                beh_files = dir(fullfile(sub_beh_dir, '*Passive*FruitFull*Session*.mat'));
                if isempty(beh_files)
                    beh_files = dir(fullfile(sub_beh_dir, '*Task2*Passive*Session*.mat'));
                end
            case 3
                beh_files = dir(fullfile(sub_beh_dir, '*Passive*ColorPatches*Session*.mat'));
                if isempty(beh_files)
                    beh_files = dir(fullfile(sub_beh_dir, '*Task3*Passive*Session*.mat'));
                end
        end
        
        if isempty(beh_files)
            fprintf('    [-] 未找到行为日志文件 (跳过)\n');
            continue;
        end
        
        % 按文件名自然排序合并各 Session
        [~, sort_idx] = sort({beh_files.name});
        beh_files = beh_files(sort_idx);
        
        all_results = [];
        for f_i = 1:numel(beh_files)
            b_path = fullfile(beh_files(f_i).folder, beh_files(f_i).name);
            b_data = load(b_path, 'results');
            if isfield(b_data, 'results')
                all_results = [all_results; b_data.results(:)]; %#ok<AGROW>
            end
        end
        
        % 过滤 Catch Trial
        is_catch = [all_results.isCatch];
        norm_results = all_results(~is_catch);
        
        beh_trigs = [norm_results.marker];
        beh_imgs  = {norm_results.imgName};
        n_beh_norm = numel(beh_trigs);
        
        % (3) LCS 动态规划对齐
        if n_cand_eeg == n_beh_norm && all(cand_eeg_trigs(:) == beh_trigs(:))
            matched_eeg_idx = (1:n_cand_eeg)';
            matched_beh_idx = (1:n_beh_norm)';
        else
            [matched_eeg_idx, matched_beh_idx] = align_lcs_local(cand_eeg_trigs, beh_trigs);
            matched_eeg_idx = matched_eeg_idx(:);
            matched_beh_idx = matched_beh_idx(:);
            fprintf('    [LCS 对齐] EEG: %d, Beh: %d -> 匹配: %d (%.1f%%)\n', ...
                n_cand_eeg, n_beh_norm, numel(matched_eeg_idx), ...
                (numel(matched_eeg_idx) / max(n_cand_eeg, n_beh_norm)) * 100);
        end
        
        n_matched = numel(matched_eeg_idx);
        if n_matched == 0
            warning('    [!] [%s Task%d] 未能成功配对任何试次！', sub_id, task_num);
            continue;
        end
        
        % (4) 提取图片尾号 (pic_id) 与条件标签
        final_trigs    = cand_eeg_trigs(matched_eeg_idx)';
        final_lats_raw = cand_eeg_lats_raw(matched_eeg_idx)';
        final_lats_500 = cand_eeg_lats_500(matched_eeg_idx)';
        matched_imgs   = beh_imgs(matched_beh_idx)';
        
        pic_ids    = nan(n_matched, 1);
        cond_names = cell(n_matched, 1);
        
        for m_i = 1:n_matched
            cur_img = matched_imgs{m_i};
            % 提取文件名末尾数字 (例如 face_color_01.bmp -> 1)
            num_str = regexp(char(cur_img), '\d+(?=\.[a-zA-Z]+$)', 'match', 'once');
            if isempty(num_str)
                num_str = regexp(char(cur_img), '\d+', 'match', 'once');
            end
            if ~isempty(num_str)
                pic_ids(m_i) = str2double(num_str);
            end
            
            % 条件标签映射
            cur_t = final_trigs(m_i);
            switch task_num
                case 1
                    if isKey(map_t1, cur_t), cond_names{m_i} = map_t1(cur_t); else, cond_names{m_i} = 'unknown'; end
                case 2
                    if isKey(map_t2_fruit, cur_t)
                        cond_names{m_i} = sprintf('%s_%s', map_t2_fruit(cur_t), map_t2_state(cur_t));
                    else
                        cond_names{m_i} = 'unknown';
                    end
                case 3
                    if isKey(map_t3, cur_t), cond_names{m_i} = map_t3(cur_t); else, cond_names{m_i} = 'unknown'; end
            end
        end
        
        % (5) 构建结构清晰的标准 trial_info 表格
        trial_info = table();
        trial_info.trial_idx          = (1:n_matched)';
        trial_info.trigger            = final_trigs;
        trial_info.condition          = cond_names;
        trial_info.img_name           = matched_imgs;
        trial_info.pic_id             = pic_ids;
        trial_info.event_sample_raw   = final_lats_raw;
        trial_info.event_sample_500hz = final_lats_500;
        trial_info.sample_idx         = final_lats_500; % 兼容下游现有调用别名
        
        % 针对 Task 2 与 Task 3 增加专属特征列
        if task_num == 2
            trial_info.fruit        = cellfun(@(t) map_t2_fruit(t), num2cell(final_trigs), 'UniformOutput', false);
            trial_info.state        = cellfun(@(t) map_t2_state(t), num2cell(final_trigs), 'UniformOutput', false);
            trial_info.memory_color = cellfun(@(t) map_t2_memcol(t), num2cell(final_trigs), 'UniformOutput', false);
        elseif task_num == 3
            trial_info.color        = cellfun(@(t) map_t3(t), num2cell(final_trigs), 'UniformOutput', false);
        end
        
        % (6) 保存标准化 .mat 试次表文件
        out_mat = fullfile(sub_out_dir, sprintf('task%d_trial_info.mat', task_num));
        save(out_mat, 'trial_info', 'sub_id', 'task_num', 'cfg', '-v7.3');
        
        fprintf('    [+] [%s Task%d] 成功保存 trial_info (%d 试次) -> %s\n', ...
            sub_id, task_num, n_matched, out_mat);
        
        % 记录审计信息
        rec = struct();
        rec.subject        = string(sub_id);
        rec.task_num       = task_num;
        rec.n_cand_eeg     = n_cand_eeg;
        rec.n_beh_norm     = n_beh_norm;
        rec.n_matched      = n_matched;
        rec.match_rate_pct = (n_matched / max(n_cand_eeg, n_beh_norm)) * 100;
        rec.output_mat     = string(out_mat);
        align_records = [align_records; rec]; %#ok<AGROW>
    end
end

%% 5. 导出全被试对齐汇总记录表
if ~isempty(align_records)
    summary_tab = struct2table(align_records);
    summary_csv = fullfile(meta_dir, 'C01_事件图片对齐记录.csv');
    writetable(summary_tab, summary_csv, 'Encoding', 'UTF-8');
    fprintf('\n[+] 全局事件对齐记录表已保存至: %s\n', summary_csv);
end

fprintf('\n========================================================================\n');
fprintf('  【C01 事件与图片信息对齐完成！】\n');
fprintf('========================================================================\n');

%% 辅助函数: LCS 动态规划序列对齐
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
