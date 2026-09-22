%% test_test_18_workers.m
clear; clc;
t3_file = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub001/task3_multiband_epoched.mat';
t2_file = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub001/task2_multiband_epoched.mat';

nw = 20; % 20核测试
fprintf('>>> 测试 %d 进程 parpool 下连续载入 Task 3 和 Task 2 ...\n', nw);
parpool('local', nw);

t1 = tic;
m3 = load(t3_file, 'epoched_data');
ep3_bands = struct();
for b = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'}
    ep3_bands.(b{1}) = m3.epoched_data.(b{1})(1:120, :, :);
end
clear m3;
fprintf('[+] Task 3 载入并筛选完成! 耗时: %.2f 秒\n', toc(t1));

t2 = tic;
m2 = load(t2_file, 'epoched_data');
ep2_bands = struct();
for b = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'}
    ep2_bands.(b{1}) = m2.epoched_data.(b{1})(1:240, :, :);
end
clear m2;
fprintf('[+] Task 2 载入并筛选完成! 耗时: %.2f 秒\n', toc(t2));

delete(gcp('nocreate'));
fprintf('[+] 全部成功!\n');
