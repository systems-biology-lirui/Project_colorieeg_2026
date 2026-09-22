%% test_mem_with_parpool20.m
clear; clc;
t3_file = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub001/task3_multiband_epoched.mat';

fprintf('>>> 1. 启动 20 进程 parpool ...\n');
p = gcp('nocreate');
if isempty(p) || p.NumWorkers ~= 20
    if ~isempty(p), delete(p); end
    parpool('local', 20);
end

fprintf('>>> 2. 尝试载入 task3 ...\n');
try
    t1 = tic;
    m3 = load(t3_file, 'epoched_data');
    fprintf('[+] Task 3 载入成功! 耗时: %.2f 秒\n', toc(t1));
    clear m3;
catch ME
    fprintf('[-] Task 3 载入失败: %s\n', ME.message);
end

delete(gcp('nocreate'));
