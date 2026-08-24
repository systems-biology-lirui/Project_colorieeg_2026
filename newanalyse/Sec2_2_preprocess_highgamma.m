function Sec2_2_preprocess_highgamma()
% PREPROCESS_HIGHGAMMA 从已有的HG epoched数据中提取每个ROI的High-Gamma特征。
%
% 重构后的流程:
%   - Sec1_preanalyse.m 已在连续数据上完成多频段(70-150Hz, 8×10Hz子带)
%     Hilbert包络提取 → 对数化 → 多频段平均 → epoch → z-score基线校正
%   - 本脚本仅做 ROI 通道提取 + 时间窗裁剪，与 Sec2_1_preprocess_erp.m 对齐
%
% 输入:
%   processed_data/{subject}/task*_HG_epoched.mat
%     epoch.data: [Cond, Rep, Ch, Time], z-score基线校正后的多频段对数功率包络
%
% 输出格式（与旧版完全一致，Sec3_2可直接使用）:
%   每个ROI保存为一个 .mat 文件，包含:
%     - hg_task1: [Cond, Rep, Ch, Time] 的High-Gamma特征
%     - hg_task2: [Cond, Rep, Ch, Time] 的High-Gamma特征
%     - hg_task3: [Cond, Rep, Ch, Time] 的High-Gamma特征
%
% 注意:
%   - 旧版 Sec2_2（在epoched数据上做滤波+Hilbert）已备份为
%     Sec2_2_preprocess_highgamma_legacy.m

run_timer = tic;

subject = 'test001';
cfg = newanalyse_load_run_config(mfilename, {'matlab_defaults', 'sec2_defaults'});
if isfield(cfg, 'subject')
    subject = char(string(cfg.subject));
end

paths = newanalyse_paths();
base_path = paths.base_path;
data_dir  = fullfile(base_path, 'processed_data', subject);
save_dir  = fullfile(paths.feature_root, 'highgamma', subject);
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

loc_file = fullfile(data_dir, sprintf('%s_ieegloc.xlsx', subject));

% 日志文件
fid = fopen(fullfile(save_dir, 'preprocess_highgamma_log.txt'), 'w');
if fid == -1, error('无法创建日志文件'); end

fprintf(fid, '=== High-Gamma特征提取开始（重构版: 从HG epoched提取ROI） ===\n');
fprintf(fid, '被试: %s\n', subject);
fprintf(fid, '输入: task*_HG_epoched.mat（已在Sec1中完成多频段包络提取+基线校正）\n');
fprintf(fid, '处理: 仅ROI通道提取 + 时间窗裁剪\n\n');

try
    % ============================================================
    % Step 1: 查找公共通道
    % ============================================================
    fprintf(fid, 'Step 1: 查找公共通道...\n');

    f1 = fullfile(data_dir, 'task1_HG_epoched.mat');
    f2 = fullfile(data_dir, 'task2_HG_epoched.mat');
    f3 = fullfile(data_dir, 'task3_HG_epoched.mat');

    if ~isfile(f1), error('Task1 HG数据未找到: %s', f1); end
    if ~isfile(f2), error('Task2 HG数据未找到: %s', f2); end
    if ~isfile(f3), error('Task3 HG数据未找到: %s', f3); end

    t1_info = load(f1, 'epoch'); ch1 = {t1_info.epoch.ch.labels}; clear t1_info;
    t2_info = load(f2, 'epoch'); ch2 = {t2_info.epoch.ch.labels}; clear t2_info;
    t3_info = load(f3, 'epoch'); ch3 = {t3_info.epoch.ch.labels}; clear t3_info;

    common_channels = intersect(ch1, intersect(ch2, ch3));
    fprintf(fid, '公共通道数: %d\n', length(common_channels));

    if isempty(common_channels)
        error('Task1/2/3之间无公共通道');
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
    % Step 3: 时间参数配置
    % ============================================================
    % HG epoched数据的时间窗是 -500~1000ms（共750点@500Hz）
    % 裁剪至 -100~1000ms，与其他模态对齐
    time_idx = 201:750;   % 550个时间点

    fprintf(fid, '原始时间窗: -500~1000ms\n');
    fprintf(fid, '裁剪时间窗: -100~1000ms（索引%d~%d，共%d点）\n\n', ...
            time_idx(1), time_idx(end), length(time_idx));

    % ============================================================
    % Step 4: 逐Task处理
    % ============================================================
    tasks = {'task1', 'task2', 'task3'};

    for t = 1:length(tasks)
        task_name = tasks{t};
        fprintf(fid, '处理 %s...\n', task_name);
        fprintf('处理 %s...\n', task_name);

        % 加载HG数据（已完成多频段包络提取+基线校正）
        data_file = fullfile(data_dir, sprintf('%s_HG_epoched.mat', task_name));
        if ~isfile(data_file)
            fprintf(fid, '  跳过: 文件不存在 %s\n', data_file);
            continue;
        end
        loaded = load(data_file, 'epoch');
        epoch  = loaded.epoch;

        all_channels = {epoch.ch.labels};
        [~, ~, ~, n_time_total] = size(epoch.data);

        fprintf(fid, '  总时间点数: %d\n', n_time_total);

        % 检查时间窗是否足够
        if n_time_total < max(time_idx)
            fprintf(fid, '  警告: 时间维度不足(%d < %d)，跳过该task\n', ...
                    n_time_total, max(time_idx));
            fprintf('  警告: %s 时间维度不足，跳过\n', task_name);
            clear epoch loaded;
            continue;
        end

        % 逐ROI处理
        for r = 1:length(rois)
            roi_name  = rois{r};
            if strcmp(roi_name, 'Unknown') || strcmp(roi_name, 'N_A')
                fprintf(fid, '  跳过ROI: %s\n', roi_name);
                continue;
            end
            roi_chans = roi_map(roi_name);

            [~, ch_idxs] = ismember(roi_chans, all_channels);
            ch_idxs = ch_idxs(ch_idxs > 0);
            if isempty(ch_idxs)
                continue;
            end

            % 提取ROI数据 [Cond, Rep, Ch, Time]
            raw_roi = epoch.data(:, :, ch_idxs, :);

            % --- 裁剪时间窗至 -100~1000ms ---
            hg_cropped = raw_roi(:, :, :, time_idx);

            % --- 保存（变量名 hg_task1/2/3，追加到同一ROI文件） ---
            safe_roi  = matlab.lang.makeValidName(roi_name);
            roi_file  = fullfile(save_dir, sprintf('%s.mat', safe_roi));
            var_name  = sprintf('hg_%s', task_name);

            eval(sprintf('%s = hg_cropped;', var_name));

            if exist(roi_file, 'file')
                save(roi_file, var_name, '-append');
            else
                save(roi_file, var_name);
            end
        end

        clear epoch loaded;
        fprintf(fid, '  完成\n');
    end

    fprintf(fid, '\nHigh-Gamma特征提取完成!\n');
    fprintf('High-Gamma特征提取完成!\n');

catch ME
    fprintf(fid, '\n错误: %s\n', ME.message);
    fprintf(fid, '堆栈:\n');
    for k = 1:length(ME.stack)
        fprintf(fid, '  %s: 第%d行\n', ME.stack(k).name, ME.stack(k).line);
    end
    rethrow(ME);
end

fclose(fid);
fprintf('%s runtime: %.2f s\n', mfilename, toc(run_timer));
end
