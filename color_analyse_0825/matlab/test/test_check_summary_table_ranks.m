%% test_check_summary_table_ranks.m
clear; clc;

res_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';

f_t1 = fullfile(res_root, 'color_effects_summary.mat');
f_t2_c = fullfile(res_root, 'concordant_electrodes_decoding_summary.mat');
f_t2_nc = fullfile(res_root, 'non_concordant_electrodes_decoding_summary.mat');
f_t3 = fullfile(res_root, 'task3_purecolor_decoding_summary.mat');

d1 = load(f_t1);
t1 = d1.all_tbl;
t1_sig = t1(t1.is_significant == 1, :);

d2_c = load(f_t2_c); t2_c = d2_c.summary_table;
d2_nc = load(f_t2_nc); t2_nc = d2_nc.summary_table;
t2 = [t2_c; t2_nc];

d3 = load(f_t3); t3 = d3.summary_table;

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};

r2 = nan(height(t1_sig), 1);
r3 = nan(height(t1_sig), 1);

for i = 1:height(t1_sig)
    sub = t1_sig.subject{i};
    ch  = t1_sig.channel{i};
    b_screen = t1_sig.freq_band{i};
    b_idx = find(strcmp(bands, b_screen));
    
    % T2
    idx2 = find(strcmp(t2.subject, sub) & strcmp(t2.channel, ch), 1);
    if ~isempty(idx2)
        row2 = t2(idx2, :);
        accs2 = [row2.peak_acc_Delta, row2.peak_acc_Theta, row2.peak_acc_Alpha, ...
                 row2.peak_acc_Beta, row2.peak_acc_Low_Gamma, row2.peak_acc_High_Gamma];
        rnk = tiedrank(-accs2);
        r2(i) = rnk(b_idx);
    end
    
    % T3
    idx3 = find(strcmp(t3.subject, sub) & strcmp(t3.channel, ch), 1);
    if ~isempty(idx3)
        row3 = t3(idx3, :);
        accs3 = [row3.peak_acc_Delta, row3.peak_acc_Theta, row3.peak_acc_Alpha, ...
                 row3.peak_acc_Beta, row3.peak_acc_Low_Gamma, row3.peak_acc_High_Gamma];
        rnk = tiedrank(-accs3);
        r3(i) = rnk(b_idx);
    end
end

fprintf('Using summary_table peak_acc:\n');
v2 = r2(~isnan(r2));
v3 = r3(~isnan(r3));
[~, p2, ~, st2] = ttest(v2 - 3.5);
[~, p3, ~, st3] = ttest(v3 - 3.5);
fprintf('Task 2: Mean = %.3f ± %.3f, t(%d) = %.3f, p = %.4f\n', mean(v2), std(v2), st2.df, st2.tstat, p2);
fprintf('Task 3: Mean = %.3f ± %.3f, t(%d) = %.3f, p = %.4f\n', mean(v3), std(v3), st3.df, st3.tstat, p3);

% 频段明细
for b = 1:6
    b_name = bands{b};
    mask = strcmp(t1_sig.freq_band, b_name);
    sub_v2 = r2(mask & ~isnan(r2));
    sub_v3 = r3(mask & ~isnan(r3));
    [~, p2_b, ~, st2_b] = ttest(sub_v2 - 3.5);
    [~, p3_b, ~, st3_b] = ttest(sub_v3 - 3.5);
    fprintf('  [%10s] N=%2d | T2: %.2f (p=%.3f) | T3: %.2f (p=%.3f)\n', ...
        b_name, sum(mask), mean(sub_v2), p2_b, mean(sub_v3), p3_b);
end
