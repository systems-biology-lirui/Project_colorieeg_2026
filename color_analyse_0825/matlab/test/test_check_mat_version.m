%% test_check_mat_version.m
clear; clc;
mat_file = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/process_data_new/sub001/task3_multiband_epoched.mat';
try
    m = matfile(mat_file);
    v = whos(m);
    fprintf('[+] MAT-file v7.3 supported! Variables: \n');
    for i = 1:numel(v)
        fprintf('    %s: %s [%s]\n', v(i).name, v(i).class, mat2str(v(i).size));
    end
catch ME
    fprintf('[-] Not v7.3: %s\n', ME.message);
end
