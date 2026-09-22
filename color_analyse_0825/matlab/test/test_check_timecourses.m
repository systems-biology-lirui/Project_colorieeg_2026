%% test_check_timecourses.m
clear; clc;
f = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables/single_channel_decoding_timecourses.mat';
d = load(f);
disp(fieldnames(d));
fprintf('time_ms: %d points, from %d to %d\n', numel(d.time_ms), d.time_ms(1), d.time_ms(end));
disp(d.bands);
fprintf('\n--- concordant ---\n');
disp(size(d.concordant));
if isstruct(d.concordant)
    disp(fieldnames(d.concordant(1)));
    fprintf('First channel: %s %s\n', d.concordant(1).subject, d.concordant(1).channel);
end
fprintf('\n--- non_concordant ---\n');
disp(size(d.non_concordant));
if isstruct(d.non_concordant)
    disp(fieldnames(d.non_concordant(1)));
end
fprintf('\n--- task3_purecolor ---\n');
disp(size(d.task3_purecolor));
if isstruct(d.task3_purecolor)
    disp(fieldnames(d.task3_purecolor(1)));
end
