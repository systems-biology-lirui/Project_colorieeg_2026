%% ========================================================================
% 脚本名称: C03_extract_multiband_epochs_0825.m
% 功能:
%   1. 【多频段特征提取与持久化】
%      在连续信号上提取 6 大经典生理频段的瞬时包络功率:
%        - Delta      (1–4 Hz)
%        - Theta      (4–8 Hz)
%        - Alpha      (8–12 Hz)
%        - Beta       (13–30 Hz)
%        - Low Gamma  (30–60 Hz, 避开 50Hz 工频)
%        - High Gamma (70–140 Hz, 5 个避开谐波的纯净子频段等权平均)
%   2. 切分 Epoch ([-500, 1000] ms)，执行试次内基线对数分贝校准 ([-300, -100] ms -> 0 dB)
%   3. 将各被试全部通道的多频段 Epoched 数据一次性完整保存至:
%        color_analyse_0825/process_data_new/<sub_id>/task<task_num>_multiband_epoched.mat
%   4. 输出执行审计汇总表至:
%        color_analyse_0825/metadata/C03_多频段特征提取汇总.csv
% ========================================================================

clear; clc; close all;

%% 1. 参数与路径配置 (主参数平铺直观)
cfg = struct();
cfg.subjects      = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008', 'sub009'};
cfg.tasks         = [1, 2, 3];             % 支持单个任务如 [1] 或批量任务如 [1, 2, 3]
cfg.target_fs     = 500;                   % 目标采样率 (Hz)
cfg.epoch_ms      = [-500, 1000];          % 切片时间窗口 (ms)
cfg.base_win_ms   = [-300, -100];          % 基线校准窗口 (ms)

% 6 大经典频段定义
bands_cfg = struct();
bands_cfg.Delta      = [1, 4];
bands_cfg.Theta      = [4, 8];
bands_cfg.Alpha      = [8, 12];
bands_cfg.Beta       = [13, 30];
bands_cfg.Low_Gamma  = [30, 48; 52, 60];   % 避开 50Hz 工频
bands_cfg.High_Gamma = [70, 80; 80, 90; 110, 120; 120, 130; 130, 140]; % 避开 100/150Hz 谐波

band_names = fieldnames(bands_cfg);
n_bands    = numel(band_names);

% 路径配置
script_dir    = fileparts(mfilename('fullpath'));
proj_root     = fileparts(fileparts(script_dir));
task_info_dir = fullfile(proj_root, 'color_analyse_0825', 'task_info');
proc_orig     = fullfile(proj_root, 'color_analyse_0825', 'process_data');
proc_new      = fullfile(proj_root, 'color_analyse_0825', 'process_data_new');
meta_dir      = fullfile(proj_root, 'color_analyse_0825', 'metadata');

if ~exist(proc_new, 'dir'), mkdir(proc_new); end
if ~exist(meta_dir, 'dir'), mkdir(meta_dir); end

fprintf('========================================================================\n');
fprintf('  【C03：全任务全长连续多频段特征提取并持久化至 process_data_new】  \n');
fprintf('========================================================================\n');
fprintf('目标被试数: %d | 频段数: 6 | 切片: [%d, %d] ms | 基线: [%d, %d] ms\n\n', ...
    numel(cfg.subjects), cfg.epoch_ms(1), cfg.epoch_ms(2), cfg.base_win_ms(1), cfg.base_win_ms(2));

summary_records = struct([]);

%% 2. 逐被试、逐任务提取并持久化存储
for sub_idx = 1:numel(cfg.subjects)
    sub_id = cfg.subjects{sub_idx};
    sub_out_dir = fullfile(proc_new, sub_id);
    if ~exist(sub_out_dir, 'dir'), mkdir(sub_out_dir); end
    
    for task_num = cfg.tasks
        raw_file     = fullfile(proj_root, 'seegdata', sub_id, sprintf('task%d.mat', task_num));
        info_file    = fullfile(task_info_dir, sub_id, sprintf('task%d_trial_info.mat', task_num));
        prep_file    = fullfile(proc_orig, sub_id, sprintf('erp%d_preprocessed.mat', task_num));
        out_mat_file = fullfile(sub_out_dir, sprintf('task%d_multiband_epoched.mat', task_num));
        
        if ~isfile(raw_file) || ~isfile(prep_file) || ~isfile(info_file)
            warning('[-] [%s Task%d] 原始、预处理或试次信息文件缺失，跳过。', sub_id, task_num);
            continue;
        end
        
        fprintf('------------------------------------------------------------------------\n');
        fprintf('>>> [%s Task%d] 正在提取 6 频段连续包络并切片 ...\n', sub_id, task_num);
        
        % 加载连续原始数据、配对信息与试次表 (纯净直读，无多余兼容分支)
        f_raw  = load(raw_file);
        f_prep = load(prep_file, 'triplet_info');
        f_info = load(info_file, 'trial_info');
        
        raw_data   = double(f_raw.data);
        raw_chans  = upper(regexprep(strtrim(f_raw.chanel_name(:)), '\s+', ''));
        raw_fs     = f_raw.fs;
        
        trip_info   = f_prep.triplet_info;
        trial_info  = f_info.trial_info;
        n_lap_ch    = height(trip_info);
        n_trials    = height(trial_info);
    
    % 去直流偏置
    raw_data = raw_data - median(raw_data, 2, 'omitnan');
    
    % 陷波滤波去除 50, 100, 150 Hz 工频
    for notch_f = [50, 100, 150]
        if notch_f < (raw_fs / 2)
            wo = notch_f / (raw_fs / 2);
            bw = wo / 35;
            [b_notch, a_notch] = iirnotch(wo, bw);
            raw_data = filtfilt(b_notch, a_notch, raw_data')';
        end
    end
    
    % 时间轴与偏移索引
    t_offsets = round(cfg.epoch_ms(1) * cfg.target_fs / 1000) : (round(cfg.epoch_ms(2) * cfg.target_fs / 1000) - 1);
    time_ms   = (t_offsets / cfg.target_fs) * 1000;
    n_pts     = numel(time_ms);
    base_mask = (time_ms >= cfg.base_win_ms(1)) & (time_ms <= cfg.base_win_ms(2));
    
    % 初始化存储结构体 (使用 single 精度，兼顾高精度与极佳存储效率)
    epoched_data = struct();
    epoched_data.subject      = sub_id;
    epoched_data.task_num     = task_num;
    epoched_data.fs           = cfg.target_fs;
    epoched_data.time_ms      = time_ms;
    epoched_data.channels     = trip_info.center_channel;
    epoched_data.triplet_info = trip_info;
    epoched_data.trial_info   = trial_info;
    epoched_data.bands        = band_names;
    
    for b = 1:n_bands
        b_name = band_names{b};
        epoched_data.(b_name) = zeros(n_trials, n_lap_ch, n_pts, 'single');
    end
    
    fprintf('    [通道流式提取] 通道数: %d | 试次数: %d | 采样点: %d ...\n', n_lap_ch, n_trials, n_pts);
    
    % 逐通道提取 6 频段连续包络并分段
    for ch = 1:n_lap_ch
        c_ch = trip_info.center_channel{ch};
        l_ch = trip_info.left_neighbor{ch};
        r_ch = trip_info.right_neighbor{ch};
        
        c_idx = find(strcmp(raw_chans, c_ch), 1);
        l_idx = find(strcmp(raw_chans, l_ch), 1);
        r_idx = find(strcmp(raw_chans, r_ch), 1);
        
        if isempty(c_idx) || isempty(l_idx) || isempty(r_idx)
            continue;
        end
        
        % 连续 Laplacian 重参考
        lap_sig_cont = raw_data(c_idx, :) - (raw_data(l_idx, :) + raw_data(r_idx, :)) / 2;
        if raw_fs ~= cfg.target_fs
            lap_sig_cont = resample(lap_sig_cont, cfg.target_fs, raw_fs);
        end
        
        % 6 个频段分别滤波提取瞬时包络功率 (dB)
        for b = 1:n_bands
            b_name = band_names{b};
            sub_ranges = bands_cfg.(b_name);
            n_sub_b = size(sub_ranges, 1);
            
            cont_log_power_accum = zeros(size(lap_sig_cont));
            for sb = 1:n_sub_b
                f_l = sub_ranges(sb, 1);
                f_h = sub_ranges(sb, 2);
                [b_bp, a_bp] = butter(4, [f_l, f_h] / (cfg.target_fs / 2), 'bandpass');
                filt_cont = filtfilt(b_bp, a_bp, lap_sig_cont);
                pow_cont  = abs(hilbert(filt_cont)).^2;
                cont_log_power_accum = cont_log_power_accum + 10 * log10(max(pow_cont, eps));
            end
            cont_log_power = cont_log_power_accum / n_sub_b;
            
            % 切分 Epoch
            if ismember('event_sample_500hz', trial_info.Properties.VariableNames)
                ev_samples = trial_info.event_sample_500hz;
            elseif ismember('event_sample_raw', trial_info.Properties.VariableNames)
                ev_samples = round(trial_info.event_sample_raw * (cfg.target_fs / raw_fs));
            else
                ev_samples = round(trial_info.sample_idx * (cfg.target_fs / raw_fs));
            end
            for tr = 1:n_trials
                idx_range = ev_samples(tr) + t_offsets;
                idx_range = max(1, min(idx_range, numel(cont_log_power)));
                raw_epoch_db = cont_log_power(idx_range);
                
                % 基线对数分贝校准 (相对基线均值归零)
                base_val = mean(raw_epoch_db(base_mask), 'omitnan');
                epoched_data.(b_name)(tr, ch, :) = single(raw_epoch_db - base_val);
            end
        end
    end
    
    % 保存持久化特征缓存
    save(out_mat_file, 'epoched_data', '-v7.3');
    fprintf('    [+] 已保存多频段特征缓存: %s\n', out_mat_file);
    
    % 记录汇总信息
    rec = struct();
    rec.subject        = string(sub_id);
    rec.task_num       = task_num;
    rec.n_trials       = n_trials;
    rec.n_lap_channels = n_lap_ch;
    rec.n_timepoints   = n_pts;
    rec.fs             = cfg.target_fs;
    rec.epoch_tmin_ms  = cfg.epoch_ms(1);
    rec.epoch_tmax_ms  = cfg.epoch_ms(2);
    rec.base_tmin_ms   = cfg.base_win_ms(1);
    rec.base_tmax_ms   = cfg.base_win_ms(2);
    rec.bands          = string(strjoin(band_names, ';'));
    rec.output_file    = string(out_mat_file);
    summary_records    = [summary_records; rec]; %#ok<AGROW>
    end
end

%% 3. 保存执行汇总记录表至 metadata
if ~isempty(summary_records)
    summary_tab = struct2table(summary_records);
    csv_path = fullfile(meta_dir, 'C03_多频段特征提取汇总.csv');
    writetable(summary_tab, csv_path, 'Encoding', 'UTF-8');
    fprintf('\n[+] 多频段特征提取汇总表已保存至: %s\n', csv_path);
end

fprintf('\n========================================================================\n');
fprintf('  【C03 全部被试多频段特征提取与持久化已就绪！】\n');
fprintf('========================================================================\n');
