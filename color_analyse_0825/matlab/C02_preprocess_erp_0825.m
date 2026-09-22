%% ========================================================================
% 脚本名称: C02_preprocess_erp_0825.m
% 功能:
%   1. 读取连续原始 SEEG 数据 (seegdata/<sub_id>/task<task_num>.mat)
%   2. 读取 C01 生成的已对齐试次表 (task_info/<sub_id>/task<task_num>_trial_info.mat)
%   3. 连续中位数去偏置 (消除 ADC 静态直流漂移)
%   4. 连续信号重采样至 500 Hz
%   5. 零相位时域滤波: 1–30 Hz 四阶 Butterworth 带通 (纯净 ERP 信号) + 50 Hz 陷波 (Q=30)
%   6. 剔除人工坏道，执行严格同杆相邻三联体 Laplacian 重参考: V_lap = V_n - 0.5*(V_{n-1} + V_{n+1})
%   7. 根据 trial_info 切分 [-500, 1000) ms ERP Epoch (共 750 点)
%   8. 保存至 process_data/<sub_id>/erp<task_num>_preprocessed.mat
%
% 输入:
%   - seegdata/<sub_id>/task<task_num>.mat
%   - task_info/<sub_id>/task<task_num>_trial_info.mat (来自 C01)
%   - metadata/人工坏道判定表.csv
%
% 输出:
%   - process_data/<sub_id>/erp<task_num>_preprocessed.mat
%   - metadata/C02_预处理ERP执行汇总.csv
% ========================================================================

clear; clc; close all;

%% 1. 参数与路径配置 (平铺直观，易于调整)
cfg = struct();
cfg.subjects   = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008', 'sub009'};
cfg.tasks      = [1, 2, 3];
cfg.orig_fs    = 1000;               % 原始参考采样率 (Hz)
cfg.target_fs  = 500;                % 目标重采样率 (Hz)

% ERP 时域滤波参数 (1-30 Hz 带通 + 50 Hz 陷波)
cfg.bp_freq    = [1, 30];            % ERP 经典带通范围 (Hz)
cfg.bp_order   = 4;                  % Butterworth 滤波器阶数
cfg.notch_freq = 50;                 % 工频陷波频率 (Hz)
cfg.notch_q    = 30;                 % 陷波品质因数 Q

% Epoch 与基线时间窗 (毫秒)
cfg.epoch_tmin   = -500;             % 起始时间 (ms)
cfg.epoch_tmax   = 1000;             % 结束时间 (ms，共 750 采样点)
cfg.do_baseline  = true;             % 是否执行试次内基线校准 (与 C03 统一)
cfg.base_win_ms  = [-300, -100];      % 刺激前基线校准窗口 (ms，与 C03 严格统一为 [-300, -100] ms)

% 路径设置
script_dir   = fileparts(mfilename('fullpath'));
proj_root    = fileparts(fileparts(script_dir)); % 项目根目录
seeg_dir     = fullfile(proj_root, 'seegdata');
proc_dir      = fullfile(proj_root, 'color_analyse_0825', 'process_data');
task_info_dir = fullfile(proj_root, 'color_analyse_0825', 'task_info');
meta_dir      = fullfile(proj_root, 'color_analyse_0825', 'metadata');
bad_ch_file   = fullfile(meta_dir, '人工坏道判定表.csv');

if ~exist(proc_dir, 'dir'), mkdir(proc_dir); end
if ~exist(task_info_dir, 'dir'), mkdir(task_info_dir); end
if ~exist(meta_dir, 'dir'), mkdir(meta_dir); end

%% 2. 预计算滤波与时间轴常数
nyq = cfg.target_fs / 2; % Nyquist 频率 (250 Hz)
t_offsets = round(cfg.epoch_tmin * cfg.target_fs / 1000) : (round(cfg.epoch_tmax * cfg.target_fs / 1000) - 1);
n_pts     = numel(t_offsets);
time_ms   = (t_offsets / cfg.target_fs) * 1000; % [-500, -498, ..., 998] ms

% 预构建 1-30 Hz 带通滤波器 (SOS 结构数值稳定性更佳)
[z, p_bp, k]   = butter(cfg.bp_order, cfg.bp_freq / nyq, 'bandpass');
[sos_bp, g_bp] = zp2sos(z, p_bp, k);

% 预构建 50 Hz 陷波滤波器
wo_n = cfg.notch_freq / nyq;
bw_n = wo_n / cfg.notch_q;
[b_notch, a_notch] = iirnotch(wo_n, bw_n);

%% 3. 逐被试、逐任务执行 ERP 预处理
fprintf('========================================================================\n');
fprintf('  【C02: 1–30 Hz 纯净 ERP 预处理与 Epoch 切分】\n');
fprintf('  滤波频段: [%.1f, %.1f] Hz | 采样率: %d Hz | Epoch: [%d, %d] ms\n', ...
    cfg.bp_freq(1), cfg.bp_freq(2), cfg.target_fs, cfg.epoch_tmin, cfg.epoch_tmax);
fprintf('========================================================================\n');

exec_records = struct([]);

for s_i = 1:numel(cfg.subjects)
    sub_id = cfg.subjects{s_i};
    sub_proc_dir = fullfile(proc_dir, sub_id);
    
    for task_num = cfg.tasks
        seeg_file = fullfile(seeg_dir, sub_id, sprintf('task%d.mat', task_num));
        info_file = fullfile(task_info_dir, sub_id, sprintf('task%d_trial_info.mat', task_num));
        out_file  = fullfile(sub_proc_dir, sprintf('erp%d_preprocessed.mat', task_num));
        
        if ~exist(seeg_file, 'file')
            fprintf('[-] [%s Task%d] 未找到 SEEG 数据文件: %s (跳过)\n', sub_id, task_num, seeg_file);
            continue;
        end
        if ~exist(info_file, 'file')
            % 兼顾回退旧目录
            legacy_info = fullfile(sub_proc_dir, sprintf('task%d_trial_info.mat', task_num));
            if exist(legacy_info, 'file')
                info_file = legacy_info;
            else
                fprintf('[-] [%s Task%d] 未找到前置对齐文件 task%d_trial_info.mat (请先运行 C01，跳过)\n', ...
                    sub_id, task_num, task_num);
                continue;
            end
        end
        
        fprintf('>>> [%s Task%d] 正在执行 1-30Hz ERP 预处理与切片...\n', sub_id, task_num);
        
        % -------------------------------------------------------------
        % 阶段 1: 读取 SEEG 连续信号与 C01 产生的 trial_info
        % -------------------------------------------------------------
        raw = load(seeg_file);
        raw_data = double(raw.data);
        ch_names = upper(regexprep(strtrim(raw.chanel_name(:)), '\s+', ''));
        [n_ch, ~] = size(raw_data);
        
        if isfield(raw, 'fs') && ~isempty(raw.fs)
            cur_fs = double(raw.fs);
        else
            cur_fs = cfg.orig_fs;
        end
        
        info_mat = load(info_file);
        trial_info = info_mat.trial_info;
        
        % -------------------------------------------------------------
        % 阶段 2: 连续通道中位数去偏置 (消除直流漂移)
        % -------------------------------------------------------------
        ch_medians = median(raw_data, 2, 'omitnan');
        raw_data   = raw_data - ch_medians;
        
        % -------------------------------------------------------------
        % 阶段 3: 连续信号重采样至 500 Hz
        % -------------------------------------------------------------
        if cur_fs ~= cfg.target_fs
            [p, q]  = rat(cfg.target_fs / cur_fs);
            ds_data = resample(raw_data.', p, q).';
        else
            ds_data = raw_data;
        end
        n_ds_time = size(ds_data, 2);
        
        % -------------------------------------------------------------
        % 阶段 4: 人工坏道剔除与严格同杆相邻三联体 Laplacian 重参考
        % -------------------------------------------------------------
        bad_channels = {};
        if exist(bad_ch_file, 'file')
            bad_tab = readtable(bad_ch_file, 'TextType', 'string', 'VariableNamingRule', 'preserve');
            sub_mask = (string(bad_tab.subject) == string(sub_id)) & ...
                       ismember(lower(string(bad_tab.decision)), ["exclude", "bad"]);
            bad_channels = cellstr(upper(string(bad_tab.channel(sub_mask))));
        end
        
        ch_map = containers.Map(ch_names, 1:n_ch);
        lap_centers    = {};
        lap_lefts      = {};
        lap_rights     = {};
        lap_center_idx = [];
        lap_left_idx   = [];
        lap_right_idx  = [];
        
        for i = 1:n_ch
            curr_ch = ch_names{i};
            if ismember(curr_ch, bad_channels), continue; end
            
            tok = regexp(curr_ch, '^([A-Z]+)(\d+)$', 'tokens', 'once');
            if isempty(tok), continue; end
            shaft = tok{1};
            c_num = str2double(tok{2});
            
            l_name = sprintf('%s%d', shaft, c_num - 1);
            r_name = sprintf('%s%d', shaft, c_num + 1);
            
            if isKey(ch_map, l_name) && isKey(ch_map, r_name)
                if ~ismember(l_name, bad_channels) && ~ismember(r_name, bad_channels)
                    lap_centers{end+1, 1} = curr_ch; 
                    lap_lefts{end+1, 1}   = l_name;  
                    lap_rights{end+1, 1}  = r_name;  
                    lap_center_idx(end+1) = ch_map(curr_ch); 
                    lap_left_idx(end+1)   = ch_map(l_name);  
                    lap_right_idx(end+1)  = ch_map(r_name); 
                end
            end
        end
        
        n_lap_ch = numel(lap_centers);
        if n_lap_ch == 0
            warning('[%s Task%d] 未找到符合同杆 Laplacian 条件的有效通道！', sub_id, task_num);
            continue;
        end
        
        % 执行同杆相邻 Laplacian 差分 (消除全脑远场共模传导，提取局部偶极子信号)
        lap_data = ds_data(lap_center_idx, :) - 0.5 * (ds_data(lap_left_idx, :) + ds_data(lap_right_idx, :));
        
        % -------------------------------------------------------------
        % 阶段 5: 零相位 1-30 Hz 带通滤波 + 50 Hz 陷波 (提取时域纯净 ERP)
        % -------------------------------------------------------------
        lap_data = filtfilt(sos_bp, g_bp, lap_data.').';
        lap_data = filtfilt(b_notch, a_notch, lap_data.').';
        
        % -------------------------------------------------------------
        % 阶段 6: 根据 trial_info 切分 ERP Epoch ([-500, 1000) ms)
        % -------------------------------------------------------------
        cand_samples = trial_info.event_sample_500hz;
        n_cand_tr    = height(trial_info);
        
        epoch_cell = {};
        valid_tr_mask = false(n_cand_tr, 1);
        
        for tr = 1:n_cand_tr
            tr_samp = cand_samples(tr) + t_offsets;
            % 边界安全检查
            if tr_samp(1) >= 1 && tr_samp(end) <= n_ds_time
                epoch_cell{end+1} = lap_data(:, tr_samp); 
                valid_tr_mask(tr) = true;
            end
        end
        
        n_trials = numel(epoch_cell);
        epoch_data = zeros(n_trials, n_lap_ch, n_pts, 'single');
        for tr = 1:n_trials
            epoch_data(tr, :, :) = single(epoch_cell{tr});
        end
        
        % 阶段 6.2: 试次内基线校准 (各通道减去刺激前基线均值，消除试次间直流偏置起伏)
        if cfg.do_baseline
            base_mask = (time_ms >= cfg.base_win_ms(1)) & (time_ms <= cfg.base_win_ms(2));
            base_val  = mean(epoch_data(:, :, base_mask), 3); % [n_trials x n_lap_ch]
            epoch_data = epoch_data - base_val;
        end
        
        % 同步筛选保留在有效边界内的 trial_info 并更新编号
        trial_info = trial_info(valid_tr_mask, :);
        trial_info.trial_idx = (1:n_trials)';
        
        % -------------------------------------------------------------
        % 阶段 7: 持久化保存
        % -------------------------------------------------------------
        triplet_info = table(lap_centers, lap_lefts, lap_rights, ...
            'VariableNames', {'center_channel', 'left_neighbor', 'right_neighbor'});
        
        save(out_file, 'epoch_data', 'time_ms', 'lap_centers', 'triplet_info', ...
             'trial_info', 'cfg', '-v7.3');
        
        fprintf('    [+] [%s Task%d] ERP 预处理就绪: [%d trials x %d channels x %d points] -> %s\n', ...
            sub_id, task_num, n_trials, n_lap_ch, n_pts, out_file);
        
        % 记录审计汇总
        rec = struct();
        rec.subject        = string(sub_id);
        rec.task_num       = task_num;
        rec.n_trials       = n_trials;
        rec.n_lap_channels = n_lap_ch;
        rec.n_timepoints   = n_pts;
        rec.fs             = cfg.target_fs;
        rec.bp_freq        = sprintf('%d-%dHz', cfg.bp_freq(1), cfg.bp_freq(2));
        rec.output_file    = string(out_file);
        exec_records = [exec_records; rec]; %#ok<AGROW>
    end
end

%% 4. 导出 C02 预处理执行汇总表格
if ~isempty(exec_records)
    summary_tab = struct2table(exec_records);
    summary_csv = fullfile(meta_dir, 'C02_预处理ERP执行汇总.csv');
    writetable(summary_tab, summary_csv, 'Encoding', 'UTF-8');
    fprintf('\n[+] C02 预处理汇总表已保存至: %s\n', summary_csv);
end

fprintf('\n========================================================================\n');
fprintf('  【C02 全被试 ERP 预处理完成！】\n');
fprintf('========================================================================\n');
