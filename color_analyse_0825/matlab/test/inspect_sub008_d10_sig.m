% inspect_sub008_d10_sig.m
clear; clc;
tab_dir = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';
c04 = load(fullfile(tab_dir, 'color_effects_summary.mat'));
if isfield(c04, 'all_tbl'), tbl = c04.all_tbl; else, tbl = c04.res_table; end

m = strcmp(tbl.subject, 'sub008') & strcmp(tbl.channel, 'D10') & tbl.is_significant == 1;
disp(tbl(m, {'subject','channel','freq_band','concordance_type','stream_hierarchy'}));

% 检查 sub008 的电极位置 tsv
elec_tsv = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub008/sub008_electrodes.tsv';
if isfile(elec_tsv)
    etbl = readtable(elec_tsv, 'FileType', 'text', 'Delimiter', '\t');
    disp(etbl(strcmp(etbl.name, 'D10'), :));
end
