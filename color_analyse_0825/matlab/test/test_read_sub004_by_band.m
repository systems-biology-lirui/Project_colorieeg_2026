%% test_read_sub004_by_band.m
clear; clc;
t2_file = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub004/task2_multiband_epoched.mat';

fprintf('>>> 测试使用 h5read 逐频段读取 sub004 (避免一次性载入 2GB 导致 OOM) ...\n');

% 1. 读取 channels
t0 = tic;
ch_cell = h5read(t2_file, '/epoched_data/channels');
fprintf('[+] 读取 channels 成功! 通道数: %d (耗时 %.2f 秒)\n', numel(ch_cell), toc(t0));

% 2. 读取时间点
time_ms = h5read(t2_file, '/epoched_data/time_ms');
fprintf('[+] 读取 time_ms 成功! 点数: %d\n', numel(time_ms));

% 3. 测试读取单个频段 (例如 Alpha)
t_band = tic;
alpha_data = h5read(t2_file, '/epoched_data/Alpha');
fprintf('[+] 读取 Alpha 频段成功! 尺寸: %s, 耗时: %.2f 秒\n', mat2str(size(alpha_data)), toc(t_band));
clear alpha_data;

% 4. 测试 trial_info: matfile 读取
m = matfile(t2_file);
ep_data_info = m.epoched_data;
fprintf('[+] 通过 matfile 查看结构字段完成!\n');
