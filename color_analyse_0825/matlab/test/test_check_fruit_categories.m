% test_check_fruit_categories.m
script_dir = fileparts(mfilename('fullpath'));
color_root = fileparts(fileparts(script_dir));
data_root = fullfile(color_root, 'process_data_new');

fpath = fullfile(data_root, 'sub001', 'task2_multiband_epoched.mat');
m = matfile(fpath);
ep = m.epoched_data;
ti = ep.trial_info;

gray_idx = strcmp(ti.state, 'gray');
ti_gray = ti(gray_idx, :);

fprintf('--- Gray Trials in Task 2 (sub001) ---\n');
disp(unique(ti_gray(:, {'fruit', 'memory_color', 'trigger'})));
fprintf('Total gray trials: %d\n', height(ti_gray));
for f = {'strawberry', 'watermelon', 'cabbage', 'kiwi'}
    fn = f{1};
    cnt = sum(strcmp(ti_gray.fruit, fn));
    fprintf('Fruit: %-12s, Count: %d\n', fn, cnt);
end
