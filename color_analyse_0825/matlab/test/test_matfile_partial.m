%% test_matfile_partial.m
clear; clc;
t3_file = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub001/task3_multiband_epoched.mat';
try
    m = matfile(t3_file);
    % 测试能否直接访问子字段
    t = m.epoched_data;
    fprintf('[+] Loaded struct epoched_data successfully!\n');
catch ME
    fprintf('[-] Error: %s\n', ME.message);
end
