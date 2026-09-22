% test_read_sub004_meta.m
script_dir = fileparts(mfilename('fullpath'));
color_root = fileparts(fileparts(script_dir));
data_root  = fullfile(color_root, 'process_data_new');

t3_mat = fullfile(data_root, 'sub001', 'task3_multiband_epoched.mat');
ch_list3_all = h5read(t3_mat, '/epoched_data/channels');

if iscell(ch_list3_all)
    ch_list3_all = cellfun(@(x) char(x(:)'), ch_list3_all, 'UniformOutput', false);
end

disp(ch_list3_all(1:5));
sub_elecs = {'B3', 'C2'};
[m, idx] = ismember(sub_elecs, ch_list3_all);
disp(m);
disp(idx);
