% inspect_sc_mat.m
clear; clc;
tab_dir = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';
mfile = fullfile(tab_dir, 'single_channel_decoding_timecourses.mat');
if isfile(mfile)
    vars = whos('-file', mfile);
    disp({vars.name});
    d = load(mfile);
    fn = fieldnames(d);
    for k = 1:numel(fn)
        fprintf('Field: %s\n', fn{k});
        disp(class(d.(fn{k})));
        if isstruct(d.(fn{k}))
            disp(fieldnames(d.(fn{k})));
            fprintf('Number of elements: %d\n', numel(d.(fn{k})));
        end
    end
end
