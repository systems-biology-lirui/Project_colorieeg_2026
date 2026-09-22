% test_inspect_timecourses_struct.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
mat_file  = fullfile(proj_root, 'result', 'tables', 'single_channel_decoding_timecourses.mat');

m = load(mat_file);
disp('m.task3_purecolor size:');
disp(size(m.task3_purecolor));

if ~isempty(m.task3_purecolor)
    disp('First entry:');
    disp(m.task3_purecolor(1));
    
    % Find sub006 G3
    keys = {m.task3_purecolor.key};
    g3_idx = find(strcmp(keys, 'sub006_G3'));
    fprintf('sub006_G3 index: %d\n', g3_idx);
    if ~isempty(g3_idx)
        disp('sub006_G3 entry:');
        disp(m.task3_purecolor(g3_idx));
        
        g3 = m.task3_purecolor(g3_idx);
        disp('g3.p_pointwise:');
        disp(g3.p_pointwise);
        
        % Check where p_pointwise < 0.05 & time >= 0
        sig_pts = find(g3.p_pointwise < 0.05 & m.time_ms >= 0);
        disp('g3 sig time points (ms):');
        disp(m.time_ms(sig_pts));
    end
end
