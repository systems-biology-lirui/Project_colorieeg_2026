% test_check_sub004_loc.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
loc_file = fullfile(proj_root, 'metadata', 'ieeg_location', 'sub004_ieegloc.xlsx');

t = readtable(loc_file, 'VariableNamingRule', 'preserve');
disp('sub004 table properties:');
disp(t.Properties.VariableNames);
disp('First 10 rows:');
disp(t(1:10, :));

% Look for E7, G10
ch_col = t{:, 1};
if iscell(ch_col), ch_col = string(ch_col); end
idx_e7 = find(strcmp(ch_col, 'E7') | contains(ch_col, 'E7'));
disp('E7 match:');
disp(t(idx_e7, :));

idx_g10 = find(strcmp(ch_col, 'G10') | contains(ch_col, 'G10'));
disp('G10 match:');
disp(t(idx_g10, :));
