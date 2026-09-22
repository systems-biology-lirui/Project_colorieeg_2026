% test_check_all_electrodes.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';

% 1. 检查 color_effects_summary.mat
summary_mat = fullfile(proj_root, 'result', 'tables', 'color_effects_summary.mat');
if isfile(summary_mat)
    d = load(summary_mat, 'all_tbl');
    t = d.all_tbl;
    % 找出独立被试和通道
    [u_ch, idx] = unique(strcat(t.subject, '_', t.channel));
    ch_tbl = t(idx, :);
    fprintf('color_effects_summary 中独立通道数: %d\n', height(ch_tbl));
    has_mni = ~isnan(ch_tbl.mni_x) & (ch_tbl.mni_x ~= 0 | ch_tbl.mni_y ~= 0 | ch_tbl.mni_z ~= 0);
    fprintf('  具备有效 MNI 坐标的通道数: %d\n', sum(has_mni));
end

% 2. 检查 ieeg_location 下的所有电极文件
loc_dir = fullfile(proj_root, 'metadata', 'ieeg_location');
loc_files = dir(fullfile(loc_dir, '*.xlsx'));
tot_contacts = 0;
fprintf('\n--- 检查 ieeg_location 原始电极定位表 ---\n');
for i = 1:numel(loc_files)
    fn = fullfile(loc_dir, loc_files(i).name);
    t_loc = readtable(fn);
    n_c = height(t_loc);
    tot_contacts = tot_contacts + n_c;
    c_names = t_loc.Properties.VariableNames;
    fprintf('%s: %d 行, 列名: %s\n', loc_files(i).name, n_c, strjoin(c_names(1:min(5, numel(c_names))), ', '));
end
fprintf('ieeg_location 总触点数: %d\n', tot_contacts);
