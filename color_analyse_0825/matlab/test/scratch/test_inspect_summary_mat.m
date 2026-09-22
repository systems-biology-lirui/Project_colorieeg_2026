% test_inspect_summary_mat.m
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
t_dir = fullfile(proj_root, 'result', 'tables');
files = {'concordant_electrodes_decoding_summary.csv', 'non_concordant_electrodes_decoding_summary.csv', 'task3_purecolor_decoding_summary.csv'};
for i = 1:numel(files)
    f = fullfile(t_dir, files{i});
    if isfile(f)
        t = readtable(f);
        fprintf('File %s: %d rows x %d cols\n', files{i}, height(t), width(t));
        disp(t.Properties.VariableNames');
    end
end
