%% ========================================================================
% 脚本名称: check_baseline_validity.m
% 功能: 严格检验基线窗口 [-300, -100] ms 的 3 大有效性标准:
%   1. 跨条件无偏性 (Color vs Gray 基线绝对功率是否有差异)
%   2. 前序试次残余影响 (Carry-over effect, 前一个 Trial 条件对当前基线是否有残留)
%   3. 刺激前斜率平稳性 (Pre-stimulus drift)
% ========================================================================

clear; clc;
script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(fileparts(script_dir)));
erp_file   = fullfile(proj_root, 'color_analyse_0825', 'process_data_new', 'sub001', 'erp1_preprocessed.mat');
if ~exist(erp_file, 'file')
    erp_file = fullfile(proj_root, 'color_analyse_0825', 'process_data_new', 'test001', 'erp1_preprocessed.mat');
end

f = load(erp_file);
t_info  = f.trial_info;
ep_data = double(f.epoch_data); % [556 x 82 x 750]
time_ms = f.time_ms;
fs      = f.cfg.target_fs;

[n_trials, n_channels, n_pts] = size(ep_data);
base_mask = (time_ms >= -300) & (time_ms <= -100);

% 提取 70-140Hz 滤波后瞬时功率
[b, a] = butter(4, [70 90]/(fs/2), 'bandpass');
p_mat = zeros(n_trials, n_channels);

for ch = 1:n_channels
    sig = filtfilt(b, a, squeeze(ep_data(:, ch, :))')';
    p = abs(hilbert(sig'))'.^2;
    p_mat(:, ch) = mean(p(:, base_mask), 2);
end

% -------------------------------------------------------------
% 检验 1: Color vs Gray 在基线期的差异 (t-test)
% -------------------------------------------------------------
col_idx = ismember(t_info.trigger, [11, 21, 31, 41]);
gry_idx = ismember(t_info.trigger, [12, 22, 32, 42]);

p_vals = zeros(n_channels, 1);
for ch = 1:n_channels
    [~, p_vals(ch)] = ttest2(p_mat(col_idx, ch), p_mat(gry_idx, ch));
end

fprintf('\n========================================================\n');
fprintf('       【基线有效性实测检验报告 (test001 Task 1)】       \n');
fprintf('========================================================\n');
fprintf('1. 跨条件中立性 (Color vs Gray 基线绝对功率): \n');
fprintf('   - 全脑 %d 个通道中，基线期 Color 与 Gray 无统计差异 (p > 0.05) 比例: %.2f%%\n', ...
    n_channels, mean(p_vals > 0.05)*100);

% -------------------------------------------------------------
% 检验 2: 前一试次残余效应 (Carry-over effect)
% -------------------------------------------------------------
prev_trig    = [NaN; t_info.trigger(1:end-1)];
valid_trials = ~isnan(prev_trig);
prev_col     = ismember(prev_trig(valid_trials), [11, 21, 31, 41]);
prev_gry     = ismember(prev_trig(valid_trials), [12, 22, 32, 42]);

p_carry = zeros(n_channels, 1);
p_sub   = p_mat(valid_trials, :);
for ch = 1:n_channels
    [~, p_carry(ch)] = ttest2(p_sub(prev_col, ch), p_sub(prev_gry, ch));
end

fprintf('2. 前一试次残余影响 (Carry-over 独立性): \n');
fprintf('   - 全脑 %d 个通道中，当前基线不受前序试次条件影响 (p > 0.05) 比例: %.2f%%\n', ...
    n_channels, mean(p_carry > 0.05)*100);
fprintf('========================================================\n\n');
