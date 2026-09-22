%% test_compare_rank_definitions.m
clear; clc;

res_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825/result/tables';
tc_file  = fullfile(res_root, 'single_channel_decoding_timecourses.mat');
t1_file  = fullfile(res_root, 'color_effects_summary.mat');

d_tc = load(tc_file);
d_t1 = load(t1_file);
if isfield(d_t1, 'all_tbl'), t1 = d_t1.all_tbl; else, t1 = d_t1.res_table; end

t1_sig = t1(t1.is_significant == 1, :);
fprintf('Task 1 显著条目数: %d\n', height(t1_sig));

% 合并 Task 2 的 229 个通道
t2_all = [d_tc.concordant; d_tc.non_concordant];
t3_all = d_tc.task3_purecolor;
time_ms = d_tc.time_ms;

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
n_bands = numel(bands);

% 定义不同的时间窗 / 准确率提取策略:
% 策略 1: 全时程峰值 peak_acc [-200, 800] ms
% 策略 2: 刺激后窗口 [100, 400] ms 平均准确率 (与 Task 1 筛选窗口一致)
% 策略 3: 刺激后窗口 [50, 500] ms 峰值准确率
% 策略 4: 刺激后窗口 [0, 800] ms 平均准确率

w_100_400 = (time_ms >= 100 & time_ms <= 400);
w_50_500  = (time_ms >= 50  & time_ms <= 500);
w_0_800   = (time_ms >= 0   & time_ms <= 800);

strats = {'Full_Peak', 'Mean_100_400ms', 'Peak_50_500ms', 'Mean_0_800ms'};
n_strats = numel(strats);

for s = 1:n_strats
    st_name = strats{s};
    r2_vec = nan(height(t1_sig), 1);
    r3_vec = nan(height(t1_sig), 1);
    
    for i = 1:height(t1_sig)
        sub = t1_sig.subject{i};
        ch  = t1_sig.channel{i};
        b_screen = t1_sig.freq_band{i};
        b_idx = find(strcmp(bands, b_screen));
        
        % 寻找 Task 2 通道
        idx2 = find(strcmp({t2_all.subject}, sub) & strcmp({t2_all.channel}, ch), 1);
        if ~isempty(idx2)
            c2 = t2_all(idx2);
            accs2 = nan(1, 6);
            for b = 1:6
                tc = c2.(sprintf('acc_%s', bands{b}));
                if strcmp(st_name, 'Full_Peak'),           accs2(b) = max(tc);
                elseif strcmp(st_name, 'Mean_100_400ms'), accs2(b) = mean(tc(w_100_400));
                elseif strcmp(st_name, 'Peak_50_500ms'),  accs2(b) = max(tc(w_50_500));
                elseif strcmp(st_name, 'Mean_0_800ms'),   accs2(b) = mean(tc(w_0_800));
                end
            end
            % 排名: 1 = 最高acc, 6 = 最低acc
            rnk2 = tiedrank(-accs2);
            r2_vec(i) = rnk2(b_idx);
        end
        
        % 寻找 Task 3 通道
        idx3 = find(strcmp({t3_all.subject}, sub) & strcmp({t3_all.channel}, ch), 1);
        if ~isempty(idx3)
            c3 = t3_all(idx3);
            accs3 = nan(1, 6);
            for b = 1:6
                tc = c3.(sprintf('acc_%s', bands{b}));
                if strcmp(st_name, 'Full_Peak'),           accs3(b) = max(tc);
                elseif strcmp(st_name, 'Mean_100_400ms'), accs3(b) = mean(tc(w_100_400));
                elseif strcmp(st_name, 'Peak_50_500ms'),  accs3(b) = max(tc(w_50_500));
                elseif strcmp(st_name, 'Mean_0_800ms'),   accs3(b) = mean(tc(w_0_800));
                end
            end
            rnk3 = tiedrank(-accs3);
            r3_vec(i) = rnk3(b_idx);
        end
    end
    
    % 计算统计检验 (vs 3.5)
    v2 = r2_vec(~isnan(r2_vec));
    v3 = r3_vec(~isnan(r3_vec));
    [~, p2, ~, st2] = ttest(v2 - 3.5);
    [~, p3, ~, st3] = ttest(v3 - 3.5);
    
    fprintf('\n----------------- 策略: %s -----------------\n', st_name);
    fprintf('  Task 2: 均值=%.3f (SD=%.2f) | t(%d)=%.3f, p=%.4f\n', ...
        mean(v2), std(v2), st2.df, st2.tstat, p2);
    fprintf('  Task 3: 均值=%.3f (SD=%.2f) | t(%d)=%.3f, p=%.4f\n', ...
        mean(v3), std(v3), st3.df, st3.tstat, p3);
end
