%% check_all_subs_conditions.m
clear; clc;
root_dir = 'e:\liulab_project\Project_colorieeg_2026\color_analyse_0825';
task_info = fullfile(root_dir, 'task_info');
subs = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008'};

fprintf('--- Task 2 Gray Fruits & Task 3 Color Patches Trial Counts ---\n');
for s = 1:numel(subs)
    sub = subs{s};
    d2 = load(fullfile(task_info, sub, 'task2_trial_info.mat'));
    ti2 = d2.trial_info;
    m2 = strcmp(ti2.state, 'gray');
    f_counts = tabulate(ti2.fruit(m2));
    
    d3 = load(fullfile(task_info, sub, 'task3_trial_info.mat'));
    ti3 = d3.trial_info;
    m3_red = strcmp(ti3.color, 'red');
    m3_grn = strcmp(ti3.color, 'green');
    
    fprintf('Subject %s:\n', sub);
    fprintf('  Task 2 Gray: straw=%d, water=%d, kiwi=%d, cabb=%d\n', ...
        sum(m2 & strcmp(ti2.fruit, 'strawberry')), ...
        sum(m2 & strcmp(ti2.fruit, 'watermelon')), ...
        sum(m2 & strcmp(ti2.fruit, 'kiwi')), ...
        sum(m2 & strcmp(ti2.fruit, 'cabbage')));
    fprintf('  Task 3 Red: p1=%d, p2=%d, p3=%d | Green: p1=%d, p2=%d, p3=%d\n', ...
        sum(m3_red & ti3.pic_id == 1), sum(m3_red & ti3.pic_id == 2), sum(m3_red & ti3.pic_id == 3), ...
        sum(m3_grn & ti3.pic_id == 1), sum(m3_grn & ti3.pic_id == 2), sum(m3_grn & ti3.pic_id == 3));
end
