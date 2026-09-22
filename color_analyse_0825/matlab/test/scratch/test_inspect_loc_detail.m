% test_inspect_loc_detail.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
loc_dir = fullfile(proj_root, 'metadata', 'ieeg_location');

% 检查 sub001 和 sub004 的列与内容
s1 = readtable(fullfile(loc_dir, 'sub001_ieegloc.xlsx'), 'VariableNamingRule', 'preserve');
disp('sub001 first 3 rows:');
disp(s1(1:3, 1:min(6, width(s1))));

s4 = readtable(fullfile(loc_dir, 'sub004_ieegloc.xlsx'), 'VariableNamingRule', 'preserve');
disp('sub004 first 3 rows:');
disp(s4(1:3, 1:min(6, width(s4))));

% 检查 seegdata 中各被试通道数
seeg_dir = 'e:/liulab_project/Project_colorieeg_2026/seegdata';
sub_dirs = dir(fullfile(seeg_dir, 'sub*'));
fprintf('\n--- seegdata 目录通道情况 ---\n');
for i = 1:numel(sub_dirs)
    sname = sub_dirs(i).name;
    elec_file = fullfile(seeg_dir, sname, [sname, '_electrodes.tsv']);
    if isfile(elec_file)
        te = readtable(elec_file, 'FileType', 'text', 'Delimiter', '\t');
        fprintf('%s: electrodes.tsv 有 %d 个电极，列: %s\n', sname, height(te), strjoin(te.Properties.VariableNames, ', '));
    else
        fprintf('%s: 无 electrodes.tsv\n', sname);
    end
end
