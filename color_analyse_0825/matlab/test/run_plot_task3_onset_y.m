% run_plot_task3_onset_y.m
clear; clc;
script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(script_dir));
target_m   = fullfile(proj_root, 'matlab', 'plot_tool', 'plot_task3_onset_y_correlation.m');

fprintf('Running script: %s\n', target_m);
run(target_m);
fprintf('\nCompleted successfully!\n');
