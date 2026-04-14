function Sec2_4_preprocess_tfa()
% PREPROCESS_TFA 从已有的 TFA epoched 数据中提取每个 ROI 的特征。
%
% 输出:
%   每个 ROI 一个 .mat 文件，包含:
%     - tfa_task1: [Cond, Rep, Ch, Time]
%     - tfa_task2: [Cond, Rep, Ch, Time]
%     - tfa_task3: [Cond, Rep, Ch, Time]
%     - tfa_roi_channels / tfa_time_ms
%
% 说明:
%   - 使用 task*_TFA_epoched.mat 作为输入
%   - ROI 分组与 ERP 预处理完全一致
%   - 特征选取方式也与 ERP 一致：直接提取 ROI 内已有通道数据并裁剪时间窗
%   - 不在这里额外进行频带展开或 Band x Channel 拼接
%   - 时间窗裁剪到 -100~1000ms，与现有 decoding 脚本一致

subject = 'test001';
cfg = newanalyse_load_run_config(mfilename, {'matlab_defaults', 'sec2_defaults'});
if isfield(cfg, 'subject')
    subject = char(string(cfg.subject));
end

paths = newanalyse_paths();
base_path = paths.base_path;
data_dir = fullfile(base_path, 'processed_data', subject);
save_dir = fullfile(paths.feature_root, 'tfa', subject);
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

loc_file = fullfile(data_dir, sprintf('%s_ieegloc.xlsx', subject));

fid = fopen(fullfile(save_dir, 'preprocess_tfa_log.txt'), 'w');
if fid == -1, error('无法创建日志文件'); end

fprintf(fid, '=== TFA特征提取开始 ===\n');
fprintf(fid, '被试: %s\n', subject);

try
    % ============================================================
    % Step 1: 查找 Task1/2/3 公共通道
    % ============================================================
    fprintf(fid, 'Step 1: 查找公共通道...\n');

    f1 = fullfile(data_dir, 'task1_TFA_epoched.mat');
    f2 = fullfile(data_dir, 'task2_TFA_epoched.mat');
    f3 = fullfile(data_dir, 'task3_TFA_epoched.mat');

    if ~isfile(f1), error('Task1 TFA数据未找到: %s', f1); end
    if ~isfile(f2), error('Task2 TFA数据未找到: %s', f2); end
    if ~isfile(f3), error('Task3 TFA数据未找到: %s', f3); end

    t1_info = load(f1, 'epoch'); ch1 = {t1_info.epoch.ch.labels}; clear t1_info;
    t2_info = load(f2, 'epoch'); ch2 = {t2_info.epoch.ch.labels}; clear t2_info;
    t3_info = load(f3, 'epoch'); ch3 = {t3_info.epoch.ch.labels}; clear t3_info;

    common_channels = intersect(ch1, intersect(ch2, ch3));
    fprintf(fid, '公共通道数: %d\n', length(common_channels));
    if isempty(common_channels)
        error('Task1/2/3之间无公共通道');
    end

    % ============================================================
    % Step 2: 解析 ROI
    % ============================================================
    fprintf(fid, 'Step 2: 解析ROI...\n');
    if exist('get_roi_map', 'file') ~= 2
        addpath(paths.analysis_code_dir);
    end
    roi_map = get_roi_map(loc_file, common_channels);
    rois = keys(roi_map);
    fprintf(fid, 'ROI数量: %d\n', length(rois));

    % ============================================================
    % Step 3: 时间参数配置
    % ============================================================
    fprintf(fid, 'Step 3: 配置时间参数...\n');
    fs = 500;
    time_idx = 201:750;  % -100~1000ms
    skip_rois = {};

    fprintf(fid, '原始时间窗: -500~1000ms\n');
    fprintf(fid, '裁剪时间窗: -100~1000ms（索引%d~%d，共%d点）\n\n', ...
            time_idx(1), time_idx(end), length(time_idx));

    % ============================================================
    % Step 4: 逐 Task / ROI 提取特征
    % ============================================================
    tasks = {'task1', 'task2', 'task3'};

    for t = 1:length(tasks)
        task_name = tasks{t};
        fprintf(fid, '处理 %s...\n', task_name);
        fprintf('处理 %s...\n', task_name);

        data_file = fullfile(data_dir, sprintf('%s_TFA_epoched.mat', task_name));
        loaded = load(data_file, 'epoch');
        epoch = loaded.epoch;
        all_channels = {epoch.ch.labels};

        for r = 1:length(rois)
            roi_name = rois{r};
            if any(strcmp(roi_name, skip_rois))
                fprintf(fid, '  跳过 ROI: %s\n', roi_name);
                continue;
            end

            roi_chans = roi_map(roi_name);
            [~, ch_idxs] = ismember(roi_chans, all_channels);
            ch_idxs = ch_idxs(ch_idxs > 0);
            if isempty(ch_idxs), continue; end

            raw_roi = epoch.data(:, :, ch_idxs, :);  % [Cond, Rep, Ch, Time]
            [~, ~, ~, n_time_total] = size(raw_roi);

            if n_time_total < max(time_idx)
                fprintf(fid, '  警告: 时间维度不足(%d < %d)，跳过 ROI %s\n', ...
                        n_time_total, max(time_idx), roi_name);
                continue;
            end

            % 与 ERP 一致：仅裁剪时间窗，不做额外特征展开
            tfa_cropped = raw_roi(:, :, :, time_idx);  % [Cond, Rep, Ch, Time]
            tfa_roi_channels = all_channels(ch_idxs);
            tfa_time_ms = linspace(-100, 1000, length(time_idx));

            safe_roi = matlab.lang.makeValidName(roi_name);
            roi_file = fullfile(save_dir, sprintf('%s.mat', safe_roi));
            var_name = sprintf('tfa_%s', task_name);

            eval(sprintf('%s = tfa_cropped;', var_name));

            if exist(roi_file, 'file')
                save(roi_file, var_name, 'tfa_roi_channels', 'tfa_time_ms', '-append');
            else
                save(roi_file, var_name, 'tfa_roi_channels', 'tfa_time_ms');
            end
        end

        clear epoch loaded;
        fprintf(fid, '  完成\n');
    end

    fprintf(fid, '\nTFA特征提取完成!\n');
    fprintf('TFA特征提取完成!\n');

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

