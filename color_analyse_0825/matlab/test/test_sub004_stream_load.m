%% test_sub004_stream_load.m
clear; clc;
t3_mat = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub004/task3_multiband_epoched.mat';
t2_mat = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub004/task2_multiband_epoched.mat';

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
sub_elecs = {'A3', 'A4', 'B5', 'C3', 'D2'}; % 虚拟 5 个目标电极

fprintf('>>> 启动 20 进程 parpool ...\n');
p = gcp('nocreate');
if isempty(p) || p.NumWorkers ~= 20
    if ~isempty(p), delete(p); end
    parpool('local', 20);
end

% 1. 读取 Task 3 流式加载
fprintf('>>> 1. 流式加载 Task 3 ...\n');
m3 = matfile(t3_mat);
ep3_struct = m3.epoched_data;
ti3 = ep3_struct.trial_info;
tr_mask = strcmp(ti3.color, 'red') | strcmp(ti3.color, 'green');
ch_list3 = ep3_struct.channels;
time_ms = ep3_struct.time_ms(:)';

[ch_m3, ch_idx3] = ismember(sub_elecs, ch_list3);
target_idx3 = ch_idx3(ch_m3);
valid_elecs = sub_elecs(ch_m3);

ep3_bands = struct();
for b = 1:numel(bands)
    b_name = bands{b};
    raw = h5read(t3_mat, ['/epoched_data/' b_name]);
    ep3_bands.(b_name) = raw(tr_mask, target_idx3, :);
    clear raw;
end
fprintf('[+] Task 3 载入完成! 尺寸: %s\n', mat2str(size(ep3_bands.Alpha)));

% 2. 读取 Task 2 流式加载
fprintf('>>> 2. 流式加载 Task 2 (158通道巨型数据) ...\n');
m2 = matfile(t2_mat);
ep2_struct = m2.epoched_data;
ti2 = ep2_struct.trial_info;
te_mask = strcmp(ti2.state, 'gray');
ch_list2 = ep2_struct.channels;

[ch_m2, ch_idx2] = ismember(valid_elecs, ch_list2);
target_idx2 = ch_idx2(ch_m2);
final_elecs = valid_elecs(ch_m2);

ep2_bands = struct();
for b = 1:numel(bands)
    b_name = bands{b};
    t0 = tic;
    raw = h5read(t2_mat, ['/epoched_data/' b_name]);
    ep2_bands.(b_name) = raw(te_mask, target_idx2, :);
    clear raw;
    fprintf('    - 频段 %s 提取耗时: %.2f 秒\n', b_name, toc(t0));
end
fprintf('[+] sub004 Task 2 巨型数据流式加载 100%% 成功! 尺寸: %s\n', mat2str(size(ep2_bands.Alpha)));

delete(gcp('nocreate'));
