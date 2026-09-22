% test_check_0727_loc.m
clear; clc;
dir_0727 = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0727/metadata/localization_original';
flist = dir(fullfile(dir_0727, '*.xlsx'));
for i = 1:numel(flist)
    fn = fullfile(dir_0727, flist(i).name);
    t = readtable(fn, 'VariableNamingRule', 'preserve');
    fprintf('%s: %d 行\n', flist(i).name, height(t));
end
