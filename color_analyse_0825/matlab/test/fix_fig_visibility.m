%% fix_fig_visibility.m
clear; clc;
fig_dir = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825\result\figures\rsa_per_electrode';
fig_files = dir(fullfile(fig_dir, '*.fig'));

fprintf('Found %d .fig files in %s\n', numel(fig_files), fig_dir);

for i = 1:numel(fig_files)
    fpath = fullfile(fig_dir, fig_files(i).name);
    try
        h = openfig(fpath, 'invisible');
        set(h, 'Visible', 'on');
        savefig(h, fpath);
        close(h);
        fprintf('  [Fixed] %s -> Visible: on\n', fig_files(i).name);
    catch ME
        fprintf('  [Error] %s: %s\n', fig_files(i).name, ME.message);
    end
end
