%% test_sub004_worker_threshold.m
clear; clc;
t2_mat = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub004/task2_multiband_epoched.mat';

for nw = [16, 14, 12, 10, 8]
    fprintf('\n>>> 测试 %d 进程下 sub004 读取 ...\n', nw);
    p = gcp('nocreate');
    if ~isempty(p), delete(p); end
    try
        parpool('local', nw);
        t0 = tic;
        raw = h5read(t2_mat, '/epoched_data/Alpha');
        fprintf('[+] %d 进程下成功读取 sub004 Alpha! 耗时: %.2f 秒\n', nw, toc(t0));
        clear raw;
        delete(gcp('nocreate'));
        fprintf('[+] 结论: %d 进程完全可用!\n', nw);
        break;
    catch ME
        fprintf('[-] %d 进程下失败: %s\n', nw, ME.message);
        if ~isempty(gcp('nocreate')), delete(gcp('nocreate')); end
    end
end
