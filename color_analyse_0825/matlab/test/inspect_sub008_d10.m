% inspect_sub008_d10.m
clear; clc;
tab_dir = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';
c04 = load(fullfile(tab_dir, 'color_effects_summary.mat'));
if isfield(c04, 'all_tbl'), tbl = c04.all_tbl; else, tbl = c04.res_table; end

m = strcmp(tbl.subject, 'sub008') & strcmp(tbl.channel, 'D10');
idx = find(m, 1);
if ~isempty(idx)
    disp(tbl(idx, :));
end
