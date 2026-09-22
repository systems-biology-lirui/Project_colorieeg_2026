%% cleanup_redundant_tables.m
% =========================================================================
% 功能:
%   清理 result/tables/ 下已迁移为 .mat 的废弃 CSV 文件与散乱时程目录
% =========================================================================

proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
tab_dir   = fullfile(proj_root, 'result', 'tables');

fprintf('>>> 开始清理 result/tables/ 下的废弃冗余 CSV 与目录 ...\n');

% 1. 聚类明细 CSV
cluster_csvs = {
    'cluster_spatiotemporal_summary.csv', ...
    'cluster_spatiotemporal_summary_all_significant.csv', ...
    'cluster_spatiotemporal_summary_concordant.csv', ...
    'cluster_spatiotemporal_summary_non_concordant.csv', ...
    'cluster_spatiotemporal_summary_task3.csv', ...
    'cluster_spatiotemporal_summary_task3_all.csv' ...
};

% 2. 旧版 master 与 summary CSV / MAT
old_masters = {
    'significant_electrodes_master.csv', ...
    'significant_electrodes_master.mat', ...
    'significant_channels_summary.csv', ...
    'sub001_G13_decoding_timecourse.csv', ...
    'sub007_C4_decoding_timecourse.csv', ...
    'task2_memory_color_decoding_summary.csv' ...
};

% 3. 已完全转为 .mat 的单表 CSV
converted_csvs = {
    'color_effects_summary.csv', ...
    'color_significant_channels_simple.csv', ...
    'concordant_electrodes_decoding_summary.csv', ...
    'non_concordant_electrodes_decoding_summary.csv', ...
    'task3_purecolor_decoding_summary.csv', ...
    'multisite_decoding_task2_summary.csv', ...
    'multisite_decoding_task2_subjects_summary.csv', ...
    'multisite_decoding_task2_timecourses.csv' ...
};

files_to_delete = [cluster_csvs, old_masters, converted_csvs];

for i = 1:numel(files_to_delete)
    f = fullfile(tab_dir, files_to_delete{i});
    if isfile(f)
        delete(f);
        fprintf('  [-] 已删除文件: %s\n', files_to_delete{i});
    end
end

% 4. 删除已整合的时程子目录 (450+ CSV)
dirs_to_delete = {
    fullfile(tab_dir, 'decoding_concordant_timecourses'), ...
    fullfile(tab_dir, 'decoding_non_concordant_timecourses'), ...
    fullfile(tab_dir, 'decoding_task3_purecolor_timecourses') ...
};

for i = 1:numel(dirs_to_delete)
    d = dirs_to_delete{i};
    if isfolder(d)
        rmdir(d, 's');
        [~, d_name] = fileparts(d);
        fprintf('  [-] 已递归删除目录: %s\n', d_name);
    end
end

fprintf('>>> 清理完毕！当前 result/tables/ 内容清单:\n');
list = dir(tab_dir);
for i = 1:numel(list)
    if ~list(i).isdir
        fprintf('  [FILE] %-45s (%.2f KB)\n', list(i).name, list(i).bytes / 1024);
    elseif ~ismember(list(i).name, {'.', '..'})
        fprintf('  [DIR]  %s/\n', list(i).name);
    end
end
