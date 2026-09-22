% check_shape_counts.m
clear; clc;
data_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new';
subs = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008'};
for s = 1:numel(subs)
    f = fullfile(data_root, subs{s}, 'task3_multiband_epoched.mat');
    if isfile(f)
        d = load(f, 'epoched_data');
        ti = d.epoched_data.trial_info;
        rg = ti(strcmp(ti.color, 'red') | strcmp(ti.color, 'green'), :);
        fprintf('%s: total rg = %d | Shape 1: %d (R:%d, G:%d) | Shape 2: %d (R:%d, G:%d) | Shape 3: %d (R:%d, G:%d)\n', ...
            subs{s}, height(rg), ...
            sum(rg.pic_id==1), sum(rg.pic_id==1 & strcmp(rg.color,'red')), sum(rg.pic_id==1 & strcmp(rg.color,'green')), ...
            sum(rg.pic_id==2), sum(rg.pic_id==2 & strcmp(rg.color,'red')), sum(rg.pic_id==2 & strcmp(rg.color,'green')), ...
            sum(rg.pic_id==3), sum(rg.pic_id==3 & strcmp(rg.color,'red')), sum(rg.pic_id==3 & strcmp(rg.color,'green')));
    end
end
