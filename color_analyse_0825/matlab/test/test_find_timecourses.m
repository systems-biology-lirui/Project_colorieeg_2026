% test_find_timecourses.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
tab_dir = fullfile(proj_root, 'result', 'tables');
tc_dir = fullfile(tab_dir, 'decoding_task3_purecolor_timecourses');

fprintf('tc_dir exists: %d\n', exist(tc_dir, 'dir'));
if exist(tc_dir, 'dir')
    d = dir(fullfile(tc_dir, '*.csv'));
    fprintf('Found %d csv files in %s\n', numel(d), tc_dir);
    if numel(d) > 0
        disp(d(1:min(5, numel(d))));
    end
end

% Also check single_channel_decoding_timecourses.mat
mat_file = fullfile(tab_dir, 'single_channel_decoding_timecourses.mat');
if isfile(mat_file)
    vars = who('-file', mat_file);
    disp('Variables in single_channel_decoding_timecourses.mat:');
    disp(vars);
end
