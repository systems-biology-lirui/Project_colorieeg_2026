%% test_find_max_workers.m
clear; clc;
t3_file = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub001/task3_multiband_epoched.mat';

worker_tests = [16, 14, 12, 10];
for k = 1:numel(worker_tests)
    nw = worker_tests(k);
    fprintf('\n>>> 测试 %d 进程 parpool ...\n', nw);
    p = gcp('nocreate');
    if ~isempty(p), delete(p); end
    
    try
        parpool('local', nw);
        m3 = load(t3_file, 'epoched_data');
        fprintf('[+] 成功! %d 进程下能够稳定载入数据 (未发生 OOM)\n', nw);
        clear m3;
        delete(gcp('nocreate'));
        break;
    catch ME
        fprintf('[-] %d 进程下失败: %s\n', nw, ME.message);
        if ~isempty(gcp('nocreate')), delete(gcp('nocreate')); end
    end
end
