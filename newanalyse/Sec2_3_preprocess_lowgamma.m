function Sec2_3_preprocess_lowgamma()
% PREPROCESS_LOWGAMMA 提取每个ROI的Low-Gamma (30-70Hz) 信号功率包络。
%
% 步骤:
%   1. 识别Task1/2/3之间的公共通道
%   2. 按ROI分组通道
%   3. 对原始信号进行30-70Hz带通滤波 → Hilbert变换取能量包络
%   4. 对能量包络进行时间平滑（高斯窗）
%   5. 裁剪至 -100~1000ms 并按ROI保存
%
% 输出格式:
%   每个ROI保存为一个 .mat 文件，包含:
%     - lg_task1: [Cond, Rep, Ch, Time] 的Low-Gamma能量
%     - lg_task2: [Cond, Rep, Ch, Time] 的Low-Gamma能量
%     - lg_task3: [Cond, Rep, Ch, Time] 的Low-Gamma能量

subject = 'test001';
cfg = newanalyse_load_run_config(mfilename, {'matlab_defaults', 'sec2_defaults'});
if isfield(cfg, 'subject')
    subject = char(string(cfg.subject));
end

paths = newanalyse_paths();
base_path = paths.base_path;
data_dir = fullfile(base_path, 'processed_data', subject);
save_dir = fullfile(paths.feature_root, 'lowgamma', subject);
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

loc_file = fullfile(data_dir, sprintf('%s_ieegloc.xlsx', subject));

% 日志文件
fid = fopen(fullfile(save_dir, 'preprocess_lowgamma_log.txt'), 'w');
if fid == -1, error('无法创建日志文件'); end

fprintf(fid, '=== Low-Gamma预处理开始 ===\n');
fprintf(fid, '被试: %s\n', subject);
fprintf(fid, '频段: 30-70Hz\n\n');

try
    % ============================================================
    % Step 1: 查找公共通道
    % ============================================================
    fprintf(fid, 'Step 1: 查找公共通道...\n');
    
    % 加载Task 1通道
    f1 = fullfile(data_dir, 'task1_ERP_epoched.mat');
    if ~isfile(f1), error('Task 1 数据未找到: %s', f1); end
    t1_info = load(f1, 'epoch');
    ch1 = {t1_info.epoch.ch.labels};
    clear t1_info;
    
    % 加载Task 2通道
    f2 = fullfile(data_dir, 'task2_ERP_epoched.mat');
    if ~isfile(f2), error('Task 2 数据未找到: %s', f2); end
    t2_info = load(f2, 'epoch');
    ch2 = {t2_info.epoch.ch.labels};
    clear t2_info;
    
    % 加载Task 3通道
    f3 = fullfile(data_dir, 'task3_ERP_epoched.mat');
    if ~isfile(f3), error('Task 3 数据未找到: %s', f3); end
    t3_info = load(f3, 'epoch');
    ch3 = {t3_info.epoch.ch.labels};
    clear t3_info;
    
    common_channels = intersect(ch1, intersect(ch2, ch3));
    fprintf(fid, '公共通道数: %d\n', length(common_channels));
    
    if isempty(common_channels)
        error('Task 1/2/3 之间无公共通道');
    end
    
    % ============================================================
    % Step 2: 解析ROI
    % ============================================================
    if exist('get_roi_map', 'file') ~= 2
        addpath(paths.analysis_code_dir);
    end
    roi_map = get_roi_map(loc_file, common_channels);
    rois = keys(roi_map);
    fprintf(fid, 'ROI数量: %d\n\n', length(rois));
    
    % ============================================================
    % Step 3: 滤波参数配置
    % ============================================================
    fs = 500;                       % 采样率 (已降采样至500Hz)
    lg_band = [30, 70];            % Low-Gamma频段
    time_idx = 201:750;             % -100ms to 1000ms（500Hz下，起始-500ms=第1点，-100ms=第201点，1000ms=第750点）
    smooth_win = 10;                % 高斯平滑窗宽度（10个采样点 = 20ms @500Hz）
    if isfield(cfg, 'smooth_win')
        smooth_win = double(cfg.smooth_win);
    end
    
    % 设计Butterworth带通滤波器
    filter_order = 4;
    [b_filt, a_filt] = butter(filter_order, lg_band / (fs/2), 'bandpass');
    
    fprintf(fid, '滤波器阶数: %d\n', filter_order);
    fprintf(fid, '平滑窗宽度: %d ms\n\n', smooth_win);
    
    % ============================================================
    % Step 4: 逐Task处理
    % ============================================================
    tasks = {'task1', 'task2', 'task3'};
    
    for t = 1:length(tasks)
        task_name = tasks{t};
        fprintf(fid, '处理 %s...\n', task_name);
        fprintf('处理 %s...\n', task_name);
        
        % 加载TFA数据（1-150Hz，保留了high-gamma频段信息）
        % 注意：不能使用ERP数据，因为ERP分支已做1-30Hz带通滤波，
        %       30-70Hz成分已被完全去除！
        data_file = fullfile(data_dir, sprintf('%s_TFA_epoched.mat', task_name));
        loaded = load(data_file, 'epoch');
        epoch = loaded.epoch;
        
        all_channels = {epoch.ch.labels};
        
        % 逐ROI处理
        for r = 1:length(rois)
            roi_name = rois{r};
            if strcmp(roi_name, 'Unknown') || strcmp(roi_name, 'N_A') || strcmp(roi_name, 'Calcarine_R') | strcmp(roi_name, 'Precuneus_R')
                fprintf(fid, '  跳过未知/其他ROI: %s\n', roi_name);
                continue;
            end
            roi_chans = roi_map(roi_name);
            
            [~, ch_idxs] = ismember(roi_chans, all_channels);
            if any(ch_idxs == 0)
                ch_idxs = ch_idxs(ch_idxs > 0);
            end
            if isempty(ch_idxs), continue; end
            
            % 提取ROI数据 [Cond, Rep, Ch, Time]
            raw_roi = epoch.data(:, :, ch_idxs, :);
            [n_cond, n_rep, n_ch, n_time_total] = size(raw_roi);
            
            % 重塑为 [Time, N_trials] 用于批量滤波
            % N_trials = Cond * Rep * Ch
            temp_data = reshape(raw_roi, [], n_time_total)';  % [Time, N]
            
            % --- 带通滤波 (30-70Hz) ---
            filtered_data = filtfilt(b_filt, a_filt, temp_data);
            
            % --- Hilbert变换取能量包络 ---
            analytic_signal = hilbert(filtered_data);
            power_envelope = abs(analytic_signal).^2;
            
            % --- 时间平滑（高斯窗） ---
            power_smooth = smoothdata(power_envelope, 1, 'gaussian', smooth_win);
            
            % --- 裁剪时间窗 ---
            if n_time_total < max(time_idx)
                fprintf(fid, '  警告: %s ROI %s 时间维度不足, 裁剪至末尾\n', task_name, roi_name);
                curr_time_idx = 401:n_time_total;
            else
                curr_time_idx = time_idx;
            end
            power_cropped = power_smooth(curr_time_idx, :);
            
            % --- 重塑回 [Time, Cond, Rep, Ch] ---
            power_reshaped = reshape(power_cropped, [length(curr_time_idx), n_cond, n_rep, n_ch]);
            
            % --- 转换为 [Cond, Rep, Ch, Time] ---
            lg_data = permute(power_reshaped, [2, 3, 4, 1]);
            
            % --- 保存 ---
            safe_roi = matlab.lang.makeValidName(roi_name);
            roi_file = fullfile(save_dir, sprintf('%s.mat', safe_roi));
            
            var_name = sprintf('lg_%s', task_name);
            eval(sprintf('%s = lg_data;', var_name));
            
            if exist(roi_file, 'file')
                save(roi_file, var_name, '-append');
            else
                save(roi_file, var_name);
            end
        end
        clear epoch loaded;
    end
    
    fprintf(fid, '\nLow-Gamma预处理完成!\n');
    fprintf('Low-Gamma预处理完成!\n');
    
catch ME
    fprintf(fid, '\n错误: %s\n', ME.message);
    fprintf(fid, '堆栈:\n');
    for k = 1:length(ME.stack)
        fprintf(fid, '  %s: 第%d行\n', ME.stack(k).name, ME.stack(k).line);
    end
    rethrow(ME);
end
fclose(fid);
end

