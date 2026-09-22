% test_count_files.m
clear; clc;
tab_dir = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';

fprintf('decoding_concordant_timecourses: %d files\n', numel(dir(fullfile(tab_dir, 'decoding_concordant_timecourses', '*.csv'))));
fprintf('decoding_non_concordant_timecourses: %d files\n', numel(dir(fullfile(tab_dir, 'decoding_non_concordant_timecourses', '*.csv'))));
fprintf('task3_purecolor_decoding_timecourses: %d files\n', numel(dir(fullfile(tab_dir, 'task3_purecolor_decoding_timecourses', '*.csv'))));
fprintf('decoding_task2_direct_timecourses: %d files\n', numel(dir(fullfile(tab_dir, 'decoding_task2_direct_timecourses', '*.csv'))));
fprintf('decoding_task3_direct_timecourses: %d files\n', numel(dir(fullfile(tab_dir, 'decoding_task3_direct_timecourses', '*.csv'))));
