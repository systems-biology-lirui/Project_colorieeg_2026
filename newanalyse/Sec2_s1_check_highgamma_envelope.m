function plot_envelope_comparison()
% PLOT_ENVELOPE_COMPARISON 绘制 High-Gamma 包络提取前后的对比图
%
% 功能:
%   随机选择每个 ROI 中的一个通道的一个 Trial，展示:
%   1. 原始带通滤波后的 High-Gamma 信号 (蓝色)
%   2. Hilbert 变换提取并平滑后的功率包络 (红色)
%
% 注意: 
%   由于预处理代码没有保存中间的滤波后信号(filtered_data)，
%   本脚本需要重新加载原始数据并执行相同的滤波步骤来复现对比。

subject = 'test002';
paths = newanalyse_paths();
base_path = paths.base_path;
data_dir = fullfile(base_path, 'processed_data', subject);
save_dir = fullfile(paths.feature_root, 'highgamma', subject);
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

loc_file = fullfile(data_dir, sprintf('%s_ieegloc.xlsx', subject));

% 参数配置 (与 preprocess_highgamma.m 保持一致)
fs = 500;
hg_band = [70, 150];
time_idx = 201:750; % -100ms to 1000ms
smooth_win = 10;
filter_order = 4;
[b_filt, a_filt] = butter(filter_order, hg_band / (fs/2), 'bandpass');

% 获取 ROI 信息
if exist('get_roi_map', 'file') ~= 2
    addpath(paths.analysis_code_dir);
end

% 加载 Task2 数据
fprintf('加载 Task2 数据...\n');
f2 = fullfile(data_dir, 'task2_ERP_epoched.mat');
t2_info = load(f2, 'epoch');
ch2 = {t2_info.epoch.ch.labels};
common_channels = ch2; 

roi_map = get_roi_map(loc_file, common_channels);
rois = keys(roi_map);

% 加载 TFA 数据 (宽频带)
data_file = fullfile(data_dir, 'task2_TFA_epoched.mat');
loaded = load(data_file, 'epoch');
epoch = loaded.epoch;
all_channels = {epoch.ch.labels};

% 绘图设置
n_rois = length(rois);
n_cols = 4;
n_rows = ceil(n_rois / n_cols);
figure('Position', [50, 50, 1600, 1000], 'Color', 'w');

% 目标条件: Task 2 的 Condition 4
target_cond_idx = 4; 

for r = 1:n_rois
    roi_name = rois{r};
    roi_chans = roi_map(roi_name);
    
    % 找到该 ROI 的通道索引
    [~, ch_idxs] = ismember(roi_chans, all_channels);
    if any(ch_idxs == 0), ch_idxs = ch_idxs(ch_idxs > 0); end
    if isempty(ch_idxs), continue; end
    
    % 提取该 ROI、Condition 4 的所有 Trial 数据
    % epoch.data: [Cond, Rep, Ch, Time]
    % 检查维度
    sz = size(epoch.data);
    if length(sz) == 4
        % [Cond, Rep, Ch, Time]
        % 提取 Condition 4 的所有 Repetition
        % 使用 reshape 避免 squeeze 在 Ch=1 时错误地压缩维度
        temp_data = epoch.data(target_cond_idx, :, ch_idxs, :);
        % temp_data 维度: [1, Rep, Ch, Time]
        
        [~, n_reps_load, n_ch_load, n_pts_load] = size(temp_data);
        raw_roi_data = reshape(temp_data, n_reps_load, n_ch_load, n_pts_load);
        % raw_roi_data: [n_reps, n_ch, n_time]
    else
        warning('数据维度不符合预期 [Cond, Rep, Ch, Time], 跳过 ROI %s', roi_name);
        continue;
    end
    
    % --------------------------------------------------------
    % 计算平均信号 (Mean over Reps and Chs)
    % --------------------------------------------------------
    % 我们需要对比:
    % 1. "滤波后" 的平均信号 (Raw Bandpassed -> Mean)
    % 2. "包络后" 的平均信号 (Raw Bandpassed -> Hilbert -> Power -> Mean)
    
    % 注意: 这里的平均顺序
    % 对于 Raw Bandpassed: 通常 ERP 是先平均 Trial 再滤波，或者先滤波再平均 Trial (线性操作交换律)
    % 对于 Power Envelope: 必须先计算每个 Trial 的 Power，然后再平均 (非线性操作)
    
    [n_reps, n_ch_roi, n_pts] = size(raw_roi_data);
    
    % 预分配
    filt_trials = zeros(n_reps, n_ch_roi, n_pts);
    env_trials  = zeros(n_reps, n_ch_roi, n_pts);
    
    % 逐 Trial 逐 Channel 处理 (避免内存不足，虽然慢一点)
    for rep = 1:n_reps
        for ch = 1:n_ch_roi
            sig = double(squeeze(raw_roi_data(rep, ch, :)));
            % 确保 sig 是行向量或列向量，长度为 n_pts
            if length(sig) ~= n_pts
                sig = sig(:);
            end
            
            % 1. 带通滤波
            try
                filt_sig = filtfilt(b_filt, a_filt, sig);
            catch ME
                warning('filtfilt 失败 ROI %s, Ch %d: %s', roi_name, ch, ME.message);
                continue;
            end
            
            filt_trials(rep, ch, :) = filt_sig;
            
            % 2. Hilbert 包络
            an_sig = hilbert(filt_sig);
            pow_sig = abs(an_sig).^2;
            
            % 3. 平滑
            env_trials(rep, ch, :) = smoothdata(pow_sig, 'gaussian', smooth_win);
        end
    end
    
    % 计算 ROI 平均 (Mean across Reps and Channels)
    % mean(..., 1) -> mean reps
    % mean(..., 2) -> mean channels
    mean_filt = squeeze(mean(mean(filt_trials, 1), 2));
    mean_env  = squeeze(mean(mean(env_trials, 1), 2));
    
    % 裁剪时间
    if length(mean_env) >= max(time_idx)
        plot_time = time_idx;
        y_filt = mean_filt(time_idx);
        y_env = mean_env(time_idx);
    else
        plot_time = 1:length(mean_env);
        y_filt = mean_filt;
        y_env = mean_env;
    end
    
    % --- 绘图 ---
    subplot(n_rows, n_cols, r);
    hold on;
    
    yyaxis left
    plot(y_filt, 'b-', 'LineWidth', 0.5);
    ylabel('Mean Filtered Amp (uV)');
    set(gca, 'ycolor', 'b');
    
    yyaxis right
    plot(y_env, 'r-', 'LineWidth', 1.5);
    ylabel('Mean Power (uV^2)');
    set(gca, 'ycolor', 'r');
    
    title(strrep(roi_name, '_', ' '), 'Interpreter', 'none');
    if r == 1
        legend('Mean Filtered HG', 'Mean Power Env');
    end
    grid on;
    hold off;
end

sgtitle(sprintf('Task2 Cond4 High-Gamma: Filtered vs Envelope (%s)', subject));
save_file = fullfile(save_dir, 'highgamma_envelope_check_task2_cond4.png');
saveas(gcf, save_file);
fprintf('对比图已保存: %s\n', save_file);

end
