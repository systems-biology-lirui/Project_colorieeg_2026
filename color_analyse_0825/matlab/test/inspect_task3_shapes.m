% inspect_task3_shapes.m
clear; clc;
mat_file = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub001/task3_multiband_epoched.mat';
d = load(mat_file, 'epoched_data');
ti = d.epoched_data.trial_info;
disp(ti.Properties.VariableNames);
disp(head(ti, 10));
if ismember('shape', ti.Properties.VariableNames)
    disp(tabulate(ti.shape));
elseif ismember('stimulus', ti.Properties.VariableNames)
    disp(tabulate(ti.stimulus));
end
disp(tabulate(strcat(string(ti.color), "_", string(ti.shape))));
