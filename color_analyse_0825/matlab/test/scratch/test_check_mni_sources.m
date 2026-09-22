% test_check_mni_sources.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';

d = load(fullfile(proj_root, 'result', 'tables', 'color_effects_summary.mat'), 'all_tbl');
t = d.all_tbl;
[~, idx] = unique(strcat(t.subject, '_', t.channel));
ch_tbl = t(idx, :);

subs = unique(ch_tbl.subject);
fprintf('--- color_effects_summary 各被试通道与有效坐标统计 ---\n');
for i = 1:numel(subs)
    s = subs{i};
    sub_t = ch_tbl(strcmp(ch_tbl.subject, s), :);
    has_coord = ~isnan(sub_t.mni_x) & (sub_t.mni_x ~= 0 | sub_t.mni_y ~= 0 | sub_t.mni_z ~= 0);
    fprintf('%s: 总分析通道 %d, 有效 MNI 坐标 %d (缺失 %d)\n', s, height(sub_t), sum(has_coord), sum(~has_coord));
end

% 查看 process_data_new/sub001 里的文件内容
proc_sub1 = fullfile(proj_root, 'process_data_new', 'sub001');
if exist(proc_sub1, 'dir')
    dfiles = dir(proc_sub1);
    fprintf('\nprocess_data_new/sub001 文件:\n');
    for i = 1:numel(dfiles)
        if ~dfiles(i).isdir
            fprintf('  %s (%d bytes)\n', dfiles(i).name, dfiles(i).bytes);
        end
    end
end
