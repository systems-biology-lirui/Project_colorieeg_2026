% test_check_all_recorded.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
rec_file = fullfile(proj_root, 'result', 'tables', 'all_recorded_electrodes.mat');

m = load(rec_file);
disp(fieldnames(m));
t = m.all_elec_tbl;
disp(t.Properties.VariableNames);

% Filter sub004
sub4 = t(strcmp(t.subject, 'sub004'), :);
disp('sub004 channels in all_recorded:');
disp(sub4(1:15, {'subject', 'channel', 'mni_x', 'mni_y', 'mni_z'}));

% Check E7 and G10
e7 = sub4(strcmp(sub4.channel, 'E7'), :);
disp('sub004 E7:');
disp(e7);
g10 = sub4(strcmp(sub4.channel, 'G10'), :);
disp('sub004 G10:');
disp(g10);
