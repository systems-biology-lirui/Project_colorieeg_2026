%% test_check_sub004_mat.m
clear; clc;
t2_file = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub004/task2_multiband_epoched.mat';
m = matfile(t2_file);
v = whos(m);
fprintf('[+] sub004 task2 variables: \n');
for i = 1:numel(v)
    fprintf('    %s: %s [%s]\n', v(i).name, v(i).class, mat2str(v(i).size));
end
