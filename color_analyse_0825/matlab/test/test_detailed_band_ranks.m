%% test_detailed_band_ranks.m
clear; clc;

res_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';
tc_file  = fullfile(res_root, 'single_channel_decoding_timecourses.mat');
t1_file  = fullfile(res_root, 'color_effects_summary.mat');

d_tc = load(tc_file);
d_t1 = load(t1_file);
if isfield(d_t1, 'all_tbl'), t1 = d_t1.all_tbl; else, t1 = d_t1.res_table; end

t1_sig = t1(t1.is_significant == 1, :);

t2_all = [d_tc.concordant; d_tc.non_concordant];
t3_all = d_tc.task3_purecolor;
time_ms = d_tc.time_ms;

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
n_bands = numel(bands);

w_use = (time_ms >= 50 & time_ms <= 500); % 刺激响应核心窗

t1_sig.r2 = nan(height(t1_sig), 1);
t1_sig.r3 = nan(height(t1_sig), 1);

for i = 1:height(t1_sig)
    sub = t1_sig.subject{i};
    ch  = t1_sig.channel{i};
    b_screen = t1_sig.freq_band{i};
    b_idx = find(strcmp(bands, b_screen));
    
    idx2 = find(strcmp({t2_all.subject}, sub) & strcmp({t2_all.channel}, ch), 1);
    if ~isempty(idx2)
        accs2 = nan(1, 6);
        for b = 1:6
            tc = t2_all(idx2).(sprintf('acc_%s', bands{b}));
            accs2(b) = max(tc(w_use));
        end
        r2 = tiedrank(-accs2);
        t1_sig.r2(i) = r2(b_idx);
    end
    
    idx3 = find(strcmp({t3_all.subject}, sub) & strcmp({t3_all.channel}, ch), 1);
    if ~isempty(idx3)
        accs3 = nan(1, 6);
        for b = 1:6
            tc = t3_all(idx3).(sprintf('acc_%s', bands{b}));
            accs3(b) = max(tc(w_use));
        end
        r3 = tiedrank(-accs3);
        t1_sig.r3(i) = r3(b_idx);
    end
end

fprintf('========================================================================\n');
fprintf('  【按频段细分统计: 刺激响应窗 50-500ms Peak Accuracy Rank (1=最高, 6=最低)】\n');
fprintf('========================================================================\n');
fprintf('%-11s | %4s | %-24s | %-24s\n', 'Band', 'N', 'Task 2 (Mean±SD, t, p)', 'Task 3 (Mean±SD, t, p)');
fprintf('------------------------------------------------------------------------\n');

for b = 1:6
    b_name = bands{b};
    sub_t = t1_sig(strcmp(t1_sig.freq_band, b_name), :);
    v2 = sub_t.r2(~isnan(sub_t.r2));
    v3 = sub_t.r3(~isnan(sub_t.r3));
    
    [~, p2, ~, st2] = ttest(v2 - 3.5);
    [~, p3, ~, st3] = ttest(v3 - 3.5);
    
    fprintf('%-11s | %4d | %4.2f ± %4.2f (t=%+5.2f, p=%.3f) | %4.2f ± %4.2f (t=%+5.2f, p=%.3f)\n', ...
        b_name, height(sub_t), mean(v2), std(v2), st2.tstat, p2, mean(v3), std(v3), st3.tstat, p3);
end

% 总体
[~, p2_all, ~, st2_all] = ttest(t1_sig.r2 - 3.5);
[~, p3_all, ~, st3_all] = ttest(t1_sig.r3 - 3.5);
fprintf('------------------------------------------------------------------------\n');
fprintf('%-11s | %4d | %4.2f ± %4.2f (t=%+5.2f, p=%.3f) | %4.2f ± %4.2f (t=%+5.2f, p=%.3f)\n', ...
    'Overall', height(t1_sig), mean(t1_sig.r2), std(t1_sig.r2), st2_all.tstat, p2_all, ...
    mean(t1_sig.r3), std(t1_sig.r3), st3_all.tstat, p3_all);

% 检查 concordant 子集
fprintf('\n========================================================================\n');
fprintf('  【仅 Concordant 同向电极子集 (N=%d)】\n', sum(strcmp(t1_sig.concordance_type, 'Concordant_Positive') | strcmp(t1_sig.concordance_type, 'Concordant_Negative')));
fprintf('========================================================================\n');
c_mask = ismember(t1_sig.concordance_type, {'Concordant_Positive', 'Concordant_Negative'});
t1_conc = t1_sig(c_mask, :);
for b = 1:6
    b_name = bands{b};
    sub_t = t1_conc(strcmp(t1_conc.freq_band, b_name), :);
    v2 = sub_t.r2(~isnan(sub_t.r2));
    v3 = sub_t.r3(~isnan(sub_t.r3));
    [~, p2, ~, st2] = ttest(v2 - 3.5);
    [~, p3, ~, st3] = ttest(v3 - 3.5);
    fprintf('%-11s | %4d | %4.2f ± %4.2f (t=%+5.2f, p=%.3f) | %4.2f ± %4.2f (t=%+5.2f, p=%.3f)\n', ...
        b_name, height(sub_t), mean(v2), std(v2), st2.tstat, p2, mean(v3), std(v3), st3.tstat, p3);
end
[~, p2_c, ~, st2_c] = ttest(t1_conc.r2 - 3.5);
[~, p3_c, ~, st3_c] = ttest(t1_conc.r3 - 3.5);
fprintf('------------------------------------------------------------------------\n');
fprintf('%-11s | %4d | %4.2f ± %4.2f (t=%+5.2f, p=%.3f) | %4.2f ± %4.2f (t=%+5.2f, p=%.3f)\n', ...
    'Conc Overall', height(t1_conc), mean(t1_conc.r2), std(t1_conc.r2), st2_c.tstat, p2_c, ...
    mean(t1_conc.r3), std(t1_conc.r3), st3_c.tstat, p3_c);
