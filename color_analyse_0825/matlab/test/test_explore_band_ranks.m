%% test_explore_band_ranks.m
clear; clc;

res_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';

f_t1 = fullfile(res_root, 'color_effects_summary.mat');
f_t2_c = fullfile(res_root, 'concordant_electrodes_decoding_summary.mat');
f_t2_nc = fullfile(res_root, 'non_concordant_electrodes_decoding_summary.mat');
f_t3 = fullfile(res_root, 'task3_purecolor_decoding_summary.mat');

% 1. 加载 Task 1 表
d1 = load(f_t1);
if isfield(d1, 'all_tbl'), t1 = d1.all_tbl; else, t1 = d1.res_table; end
t1_sig = t1(t1.is_significant == 1, :);
fprintf('[Task 1] 显著条目数 (频段 x 位点): %d\n', height(t1_sig));
u_t1_elecs = unique(strcat(t1_sig.subject, '_', t1_sig.channel));
fprintf('[Task 1] 唯一显著电极数: %d\n', numel(u_t1_elecs));

% 2. 加载 Task 2 表
d2_c = load(f_t2_c);
t2_c = d2_c.summary_table;
d2_nc = load(f_t2_nc);
t2_nc = d2_nc.summary_table;
t2 = [t2_c; t2_nc];
u_t2_elecs = unique(strcat(t2.subject, '_', t2.channel));
fprintf('[Task 2] 总解码电极数: %d (Concordant %d, Non-Concordant %d)\n', ...
    height(t2), height(t2_c), height(t2_nc));

% 3. 加载 Task 3 表
d3 = load(f_t3);
t3 = d3.summary_table;
u_t3_elecs = unique(strcat(t3.subject, '_', t3.channel));
fprintf('[Task 3] 总解码电极数: %d\n', height(t3));

% 检查电极交集
diff_t1_t2 = setdiff(u_t1_elecs, u_t2_elecs);
diff_t1_t3 = setdiff(u_t1_elecs, u_t3_elecs);
fprintf('T1 在 T2 中缺失的电极数: %d\n', numel(diff_t1_t2));
if ~isempty(diff_t1_t2)
    disp(diff_t1_t2);
end
fprintf('T1 在 T3 中缺失的电极数: %d\n', numel(diff_t1_t3));
if ~isempty(diff_t1_t3)
    disp(diff_t1_t3);
end

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
n_bands = numel(bands);

% 4. 计算 Rank
% 对每个 Task 1 显著条目:
% screening_band = t1_sig.freq_band{i}
% 在 Task 2 中，取出 6 个单频段 peak_acc
% 计算该 screening_band 在 6 个单频段中的 rank (1 = 最大acc, 6 = 最小acc)
% 在 Task 3 中同理

t1_sig.rank_t2 = nan(height(t1_sig), 1);
t1_sig.rank_t3 = nan(height(t1_sig), 1);
t1_sig.acc_t2_screen = nan(height(t1_sig), 1);
t1_sig.acc_t3_screen = nan(height(t1_sig), 1);

for i = 1:height(t1_sig)
    sub = t1_sig.subject{i};
    ch  = t1_sig.channel{i};
    b_screen = t1_sig.freq_band{i};
    
    b_idx = find(strcmp(bands, b_screen));
    
    % Task 2
    r2 = t2(strcmp(t2.subject, sub) & strcmp(t2.channel, ch), :);
    if ~isempty(r2)
        accs2 = [r2.peak_acc_Delta(1), r2.peak_acc_Theta(1), r2.peak_acc_Alpha(1), ...
                 r2.peak_acc_Beta(1), r2.peak_acc_Low_Gamma(1), r2.peak_acc_High_Gamma(1)];
        % 降序排名: acc 越高，rank 越靠前 (1 = 最高, 6 = 最低)
        ranks2 = tiedrank(-accs2);
        t1_sig.rank_t2(i) = ranks2(b_idx);
        t1_sig.acc_t2_screen(i) = accs2(b_idx);
    end
    
    % Task 3
    r3 = t3(strcmp(t3.subject, sub) & strcmp(t3.channel, ch), :);
    if ~isempty(r3)
        accs3 = [r3.peak_acc_Delta(1), r3.peak_acc_Theta(1), r3.peak_acc_Alpha(1), ...
                 r3.peak_acc_Beta(1), r3.peak_acc_Low_Gamma(1), r3.peak_acc_High_Gamma(1)];
        ranks3 = tiedrank(-accs3);
        t1_sig.rank_t3(i) = ranks3(b_idx);
        t1_sig.acc_t3_screen(i) = accs3(b_idx);
    end
end

% 打印总体统计
valid_t2 = ~isnan(t1_sig.rank_t2);
valid_t3 = ~isnan(t1_sig.rank_t3);

fprintf('\n=== Task 2 Rank 统计 (总体，N=%d) ===\n', sum(valid_t2));
mean_r2 = mean(t1_sig.rank_t2(valid_t2));
std_r2  = std(t1_sig.rank_t2(valid_t2));
[h2, p2, ci2, stat2] = ttest(t1_sig.rank_t2(valid_t2) - 3.5);
p2_signrank = signrank(t1_sig.rank_t2(valid_t2), 3.5);
fprintf('均值: %.3f (SD: %.3f) | vs 3.5: t(%d) = %.3f, p_ttest = %.4e, p_signrank = %.4e\n', ...
    mean_r2, std_r2, stat2.df, stat2.tstat, p2, p2_signrank);

fprintf('\n=== Task 3 Rank 统计 (总体，N=%d) ===\n', sum(valid_t3));
mean_r3 = mean(t1_sig.rank_t3(valid_t3));
std_r3  = std(t1_sig.rank_t3(valid_t3));
[h3, p3, ci3, stat3] = ttest(t1_sig.rank_t3(valid_t3) - 3.5);
p3_signrank = signrank(t1_sig.rank_t3(valid_t3), 3.5);
fprintf('均值: %.3f (SD: %.3f) | vs 3.5: t(%d) = %.3f, p_ttest = %.4e, p_signrank = %.4e\n', ...
    mean_r3, std_r3, stat3.df, stat3.tstat, p3, p3_signrank);

% 分频段统计
fprintf('\n=== 按 Task 1 筛选频段细分统计 ===\n');
for b = 1:n_bands
    b_name = bands{b};
    sub_t = t1_sig(strcmp(t1_sig.freq_band, b_name), :);
    v2 = sub_t.rank_t2(~isnan(sub_t.rank_t2));
    v3 = sub_t.rank_t3(~isnan(sub_t.rank_t3));
    
    [~, p2_b, ~, st2_b] = ttest(v2 - 3.5);
    [~, p3_b, ~, st3_b] = ttest(v3 - 3.5);
    
    fprintf('[%10s] N=%2d | Task2 mean rank: %.2f (p=%.3f) | Task3 mean rank: %.2f (p=%.3f)\n', ...
        b_name, height(sub_t), mean(v2), p2_b, mean(v3), p3_b);
end
