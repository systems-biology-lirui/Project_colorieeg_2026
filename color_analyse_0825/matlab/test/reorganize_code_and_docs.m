%% reorganize_code_and_docs.m
% =========================================================================
% 功能:
%   1. 删除 matlab/ 与 docs/ 下的废弃残留与临时无用文件
%   2. 将散落的绘图脚本归整移动到 matlab/plot_tool/
%   3. 修复 docs 中 C09 编号为 C07，并将 9月7日 过期提案归档至 docs/archive_proposals/
%   4. 将 matlab/test/ 下的一次性探测脚本归整至 matlab/test/scratch/
% =========================================================================

proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
mat_dir   = fullfile(proj_root, 'matlab');
doc_dir   = fullfile(proj_root, 'docs');
test_dir  = fullfile(mat_dir, 'test');
pt_dir    = fullfile(mat_dir, 'plot_tool');

fprintf('========================================================================\n');
fprintf('  【开始执行 docs 与 matlab 目录整理方案】\n');
fprintf('========================================================================\n');

%% 1. 删除废弃无用文件
del_files = {
    fullfile(mat_dir, 'matlab_diag.txt'), ...
    fullfile(mat_dir, 'matlab_job_done.txt'), ...
    fullfile(mat_dir, 'matlab_daemon.m'), ...
    fullfile(doc_dir, 'qc文件整理清单_20260907.json')
};

for i = 1:numel(del_files)
    f = del_files{i};
    if isfile(f)
        delete(f);
        [~, fn, ext] = fileparts(f);
        fprintf('  [-] 已删除无用文件: %s%s\n', fn, ext);
    end
end

%% 2. 移动绘图脚本到 matlab/plot_tool/ 并更新相对路径
move_plots = {
    'plot_all_recorded_electrodes_glass_brain.m', ...
    'replot_c08_figures.m'
};

for i = 1:numel(move_plots)
    src = fullfile(mat_dir, move_plots{i});
    dst = fullfile(pt_dir, move_plots{i});
    if isfile(src)
        movefile(src, dst);
        fprintf('  [->] 已将脚本归整移入 plot_tool/: %s\n', move_plots{i});
    end
end

% 适配 plot_all_recorded_electrodes_glass_brain.m 路径 (层级由 1 级变为 2 级)
p1 = fullfile(pt_dir, 'plot_all_recorded_electrodes_glass_brain.m');
if isfile(p1)
    txt = fileread(p1);
    txt = strrep(txt, 'proj_root  = fileparts(script_dir);', 'proj_root  = fileparts(fileparts(script_dir));');
    fid = fopen(p1, 'w');
    fwrite(fid, txt);
    fclose(fid);
    fprintf('  [✓] 已适配 plot_all_recorded_electrodes_glass_brain.m 相对路径\n');
end

% 适配 replot_c08_figures.m 路径
p2 = fullfile(pt_dir, 'replot_c08_figures.m');
if isfile(p2)
    txt = fileread(p2);
    old_s = sprintf('script_dir = fileparts(mfilename(''fullpath''));\nproj_root  = fileparts(fileparts(script_dir));\nres_root   = fullfile(proj_root, ''color_analyse_0825'', ''result'');');
    new_s = sprintf('script_dir = fileparts(mfilename(''fullpath''));\nproj_root  = fileparts(fileparts(fileparts(script_dir)));\nres_root   = fullfile(proj_root, ''color_analyse_0825'', ''result'');');
    txt = strrep(txt, old_s, new_s);
    fid = fopen(p2, 'w');
    fwrite(fid, txt);
    fclose(fid);
    fprintf('  [✓] 已适配 replot_c08_figures.m 相对路径\n');
end

%% 3. 修复 docs/ 编号冲突并归档过期提案草案
% (1) 重命名 C09 为 C07
c09_doc = fullfile(doc_dir, 'C09_task3_red_green_decoding_scheme.md');
c07_doc = fullfile(doc_dir, 'C07_task3_pure_color_decoding_scheme.md');
if isfile(c09_doc)
    txt = fileread(c09_doc);
    txt = strrep(txt, 'C09', 'C07');
    fid = fopen(c07_doc, 'w');
    fwrite(fid, txt);
    fclose(fid);
    delete(c09_doc);
    fprintf('  [✓] 已重命名并修正文档编号: C09 -> C07_task3_pure_color_decoding_scheme.md\n');
end

% (2) 归档 9月7日 方案草案
arch_dir = fullfile(doc_dir, 'archive_proposals');
if ~exist(arch_dir, 'dir'), mkdir(arch_dir); end

arch_docs = {
    '代码编号与QC文件整理方案_20260907.md', ...
    '新主线全面修正方案_20260907.md'
};

for i = 1:numel(arch_docs)
    src = fullfile(doc_dir, arch_docs{i});
    dst = fullfile(arch_dir, arch_docs{i});
    if isfile(src)
        movefile(src, dst);
        fprintf('  [->] 已将历史提案草案归档至 docs/archive_proposals/: %s\n', arch_docs{i});
    end
end

%% 4. 收敛 matlab/test/ 临时脚本至 scratch/
scratch_dir = fullfile(test_dir, 'scratch');
if ~exist(scratch_dir, 'dir'), mkdir(scratch_dir); end

scratch_tests = {
    'test_check_0727_loc.m', ...
    'test_check_all_electrodes.m', ...
    'test_check_mni_sources.m', ...
    'test_compare_electrodes.m', ...
    'test_extract_all_recorded_electrodes.m', ...
    'test_inspect_electrode_mapping.m', ...
    'test_inspect_loc_detail.m', ...
    'test_inspect_sub004_channels.m', ...
    'test_inspect_summary_mat.m', ...
    'test_plot_all_electrodes.m', ...
    'test_run_all_electrodes_pipeline.m'
};

for i = 1:numel(scratch_tests)
    src = fullfile(test_dir, scratch_tests{i});
    dst = fullfile(scratch_dir, scratch_tests{i});
    if isfile(src)
        movefile(src, dst);
        fprintf('  [->] 已将临时探针脚本收敛入 test/scratch/: %s\n', scratch_tests{i});
    end
end

%% 5. 修正 matlab/qc/check_baseline_validity.m 中的旧路径
qc_bl = fullfile(mat_dir, 'qc', 'check_baseline_validity.m');
if isfile(qc_bl)
    txt = fileread(qc_bl);
    txt = strrep(txt, '''process_data''', '''process_data_new''');
    fid = fopen(qc_bl, 'w');
    fwrite(fid, txt);
    fclose(fid);
    fprintf('  [✓] 已修正 check_baseline_validity.m 中的旧数据路径 (process_data -> process_data_new)\n');
end

fprintf('\n========================================================================\n');
fprintf('  【目录整理全部顺利完成！】\n');
fprintf('========================================================================\n');
