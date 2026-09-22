% test_list_sub004_channels.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
loc_file = fullfile(proj_root, 'metadata', 'ieeg_location', 'sub004_ieegloc.xlsx');

t = readtable(loc_file, 'VariableNamingRule', 'preserve');
chs = string(t{:, 1});
disp('All unique channel letters in sub004:');
letters = unique(regexprep(chs, '\d+', ''));
disp(letters');

disp('Channels starting with E:');
disp(chs(startsWith(chs, 'E'))');

disp('Channels starting with G:');
disp(chs(startsWith(chs, 'G'))');
