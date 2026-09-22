%% test_open_fig.m
clear; clc;
fig_path1 = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825\result\figures\rsa_per_electrode\sub008_C10_rsa_window_avg.fig';
fig_path2 = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825\result\figures\rsa_per_electrode\sub008_C10_rsa_3d_trajectory.fig';

try
    h1 = openfig(fig_path1);
    fprintf('Fig 1 opened successfully. Visible: %s\n', get(h1, 'Visible'));
    close(h1);
catch ME
    fprintf('Error opening Fig 1: %s\n', ME.message);
end

try
    h2 = openfig(fig_path2);
    fprintf('Fig 2 opened successfully. Visible: %s\n', get(h2, 'Visible'));
    close(h2);
catch ME
    fprintf('Error opening Fig 2: %s\n', ME.message);
end
