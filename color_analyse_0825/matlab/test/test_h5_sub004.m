%% test_h5_sub004.m
clear; clc;
t2_file = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub004/task2_multiband_epoched.mat';

info = h5info(t2_file);
fprintf('[+] Top groups and datasets in sub004 task2: \n');
for i = 1:numel(info.Groups)
    fprintf('    Group: %s\n', info.Groups(i).Name);
    for j = 1:numel(info.Groups(i).Datasets)
        ds = info.Groups(i).Datasets(j);
        fprintf('        Dataset: %s [%s]\n', ds.Name, mat2str(ds.Dataspace.Size));
    end
end
