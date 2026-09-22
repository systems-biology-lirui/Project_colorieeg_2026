%% ========================================================================
% 脚本名称: C04_screen_color_channels_0825.m
% 功能:
%   1. 读取 C03 提取的各被试多频段特征缓存 (process_data_new/<sub_id>/task1_multiband_epoched.mat)
%   2. 严格执行类别内同图配对 (Face, Object, Body, Place 各自配对，总 280 对)
%   3. 计算核心主窗口 [100, 400] ms 内四大类别等权色彩效应
%   4. 执行配对符号翻转置换检验 (2000 次) 与被试内 FDR (Benjamini-Hochberg) 校正
%   5. 判定四类别效应方向一致性 (Concordant_Positive / Concordant_Negative / Category_Biased)
%   6. 输出规范汇总表至:
%        - color_analyse_0825/result/tables/color_effects_summary.csv
%        - color_analyse_0825/metadata/C04_全频段色彩统一效应汇总.csv (兼容)
% ========================================================================

clear; clc; close all;

%% 1. 参数与路径配置 (平铺直观)
cfg = struct();
cfg.subjects      = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008'};
cfg.main_win_ms   = [100, 400];        % 核心主时间窗 (ms)
cfg.n_perm        = 2000;              % 置换检验次数
cfg.alpha_sig     = 0.05;              % 显著性阈值

cat_trig_col = [11, 21, 31, 41];
cat_trig_gry = [12, 22, 32, 42];
cat_names    = {'Face', 'Object', 'Body', 'Place'};

% 路径配置
script_dir = fileparts(mfilename('fullpath'));
proj_root  = fileparts(fileparts(script_dir));
proc_new   = fullfile(proj_root, 'color_analyse_0825', 'process_data_new');
res_table_dir = fullfile(proj_root, 'color_analyse_0825', 'result', 'tables');
meta_dir   = fullfile(proj_root, 'color_analyse_0825', 'metadata');

if ~exist(res_table_dir, 'dir'), mkdir(res_table_dir); end

fprintf('========================================================================\n');
fprintf('  【C04：全频段色彩效应筛选与四类别一致性统计检验】  \n');
fprintf('========================================================================\n');
fprintf('分析窗口: [%d, %d] ms | 置换次数: %d | 显著性水平: %.2f\n\n', ...
    cfg.main_win_ms(1), cfg.main_win_ms(2), cfg.n_perm, cfg.alpha_sig);

%% 2. 加载首个文件获取通用参数与解剖映射
first_mat = fullfile(proc_new, cfg.subjects{1}, 'task1_multiband_epoched.mat');
if ~isfile(first_mat)
    error('未找到特征缓存，请先运行 C03_extract_multiband_epochs_0825.m！');
end
f0 = load(first_mat);
band_names = f0.epoched_data.bands;
n_bands    = numel(band_names);
time_ms    = f0.epoched_data.time_ms;
win_mask   = (time_ms >= cfg.main_win_ms(1)) & (time_ms <= cfg.main_win_ms(2));

% 预构建解剖定位查询字典 (若已有 master 标注则安全复用)
anatomy_map = containers.Map();
master_mat = fullfile(res_table_dir, 'significant_electrodes_master.mat');
if isfile(master_mat)
    m_data = load(master_mat, 'sig_tbl');
    if isfield(m_data, 'sig_tbl') && ismember('dkt_anatomy', m_data.sig_tbl.Properties.VariableNames)
        m_tbl = m_data.sig_tbl;
        m_keys = strcat(m_tbl.subject, '_', m_tbl.channel);
        [u_k, u_idx] = unique(m_keys);
        for mi = 1:numel(u_k)
            anatomy_map(u_k{mi}) = char(m_tbl.dkt_anatomy(u_idx(mi)));
        end
    end
end

%% 3. 逐被试、逐频段统计检验
all_records = {};

for sub_idx = 1:numel(cfg.subjects)
    sub_id = cfg.subjects{sub_idx};
    mat_file = fullfile(proc_new, sub_id, 'task1_multiband_epoched.mat');
    
    if ~isfile(mat_file)
        warning('被试 %s 数据不存在，跳过。', sub_id);
        continue;
    end
    
    fprintf('>>> [%d/%d] 正在统计分析被试: [%s] ...\n', sub_idx, numel(cfg.subjects), sub_id);
    mat_obj = load(mat_file);
    ep_data = mat_obj.epoched_data;
    
    trip_info  = ep_data.triplet_info;
    trial_info = ep_data.trial_info;
    n_ch       = height(trip_info);
    
    % 建立严格的类别内同图配对索引 (280 对图片)
    col_pair_idx = cell(1, 4);
    gry_pair_idx = cell(1, 4);
    for c = 1:4
        tc = find(trial_info.trigger == cat_trig_col(c));
        tg = find(trial_info.trigger == cat_trig_gry(c));
        [~, ic, ig] = intersect(trial_info.pic_id(tc), trial_info.pic_id(tg));
        col_pair_idx{c} = tc(ic);
        gry_pair_idx{c} = tg(ig);
    end
    
    for b = 1:n_bands
        b_name = band_names{b};
        band_power = ep_data.(b_name); % [n_trials x n_ch x n_pts]
        
        % 收集该被试该频段所有通道的 p 值用于 FDR 校正
        sub_raw_pvals = zeros(n_ch, 1);
        sub_ch_recs   = cell(n_ch, 1);
        
        for ch = 1:n_ch
            c_ch = trip_info.center_channel{ch};
            ch_epoch = double(squeeze(band_power(:, ch, :))); % [n_trials x n_pts]
            
            % 计算各类别在主窗内的配对差值
            cat_diffs = zeros(1, 4);
            all_c_vals = [];
            all_g_vals = [];
            
            for c = 1:4
                c_trials = mean(ch_epoch(col_pair_idx{c}, win_mask), 2);
                g_trials = mean(ch_epoch(gry_pair_idx{c}, win_mask), 2);
                cat_diffs(c) = mean(c_trials - g_trials, 'omitnan');
                all_c_vals = [all_c_vals; c_trials];
                all_g_vals = [all_g_vals; g_trials];
            end
            
            % 四类别等权色彩效应
            gen_eff = mean(cat_diffs);
            
            % 配对检验
            paired_diffs = all_c_vals - all_g_vals;
            n_pairs = numel(paired_diffs);
            
            % 配对符号翻转置换检验
            rng(sub_idx * 100 + b);
            signs = (rand(n_pairs, cfg.n_perm) > 0.5) * 2 - 1;
            perm_diffs = mean(paired_diffs .* signs, 1);
            p_perm = (sum(abs(perm_diffs) >= abs(gen_eff)) + 1) / (cfg.n_perm + 1);
            sub_raw_pvals(ch) = p_perm;
            
            % 方向一致性分类
            if all(cat_diffs > 0)
                concord_type = 'Concordant_Positive';
            elseif all(cat_diffs < 0)
                concord_type = 'Concordant_Negative';
            else
                concord_type = 'Category_Biased';
            end
            
            % 解剖信息安全读取
            dkt_str = '';
            if ismember('dkt_anatomy', trip_info.Properties.VariableNames)
                dkt_str = char(trip_info.dkt_anatomy(ch));
            end
            if isempty(dkt_str) && isKey(anatomy_map, sprintf('%s_%s', sub_id, c_ch))
                dkt_str = anatomy_map(sprintf('%s_%s', sub_id, c_ch));
            end
            
            rec = struct();
            rec.subject             = sub_id;
            rec.channel             = c_ch;
            rec.laplacian_triplet   = sprintf('%s-(%s+%s)/2', c_ch, trip_info.left_neighbor{ch}, trip_info.right_neighbor{ch});
            rec.freq_band           = b_name;
            rec.general_effect_dB   = gen_eff;
            rec.p_perm_100_400ms    = p_perm;
            rec.concordance_type    = concord_type;
            rec.delta_face_dB       = cat_diffs(1);
            rec.delta_object_dB     = cat_diffs(2);
            rec.delta_body_dB       = cat_diffs(3);
            rec.delta_place_dB      = cat_diffs(4);
            rec.dkt_anatomy         = dkt_str;
            
            sub_ch_recs{ch} = rec;
        end
        
        % 被试内 FDR (Benjamini-Hochberg) 校正
        [~, ~, ~, p_fdr] = fdr_bh(sub_raw_pvals, cfg.alpha_sig);
        
        for ch = 1:n_ch
            sub_ch_recs{ch}.p_fdr = p_fdr(ch);
            sub_ch_recs{ch}.is_significant = (sub_ch_recs{ch}.p_perm_100_400ms < cfg.alpha_sig);
            sub_ch_recs{ch}.is_fdr_sig     = (p_fdr(ch) < cfg.alpha_sig);
            all_records = [all_records; sub_ch_recs{ch}];
        end
    end
end

%% 4. 汇总导出结果表格
res_table = struct2table(cell2mat(all_records));

% 导出全量明细表 (.mat 原生存储，高效且自包含)
out_mat1  = fullfile(res_table_dir, 'color_effects_summary.mat');
all_tbl   = res_table;
save(out_mat1, 'all_tbl', 'res_table');
fprintf('\n[+] 全量明细汇总表(.mat)已保存至: %s\n', out_mat1);

% 导出显著电极极简清单 (.mat 格式，直观易读)
sig_mask = (res_table.is_significant == 1);
sig_sub_table = res_table(sig_mask, :);

sig_type_map = repmat({'不同向'}, height(sig_sub_table), 1);
sig_type_map(strcmp(sig_sub_table.concordance_type, 'Concordant_Positive')) = {'总体正向'};
sig_type_map(strcmp(sig_sub_table.concordance_type, 'Concordant_Negative')) = {'总体负向'};

simple_table = table();
simple_table.subject           = sig_sub_table.subject;
simple_table.channel           = sig_sub_table.channel;
simple_table.freq_band         = sig_sub_table.freq_band;
simple_table.significance_type = sig_type_map;
simple_table.general_effect_dB = round(sig_sub_table.general_effect_dB, 3);
simple_table.p_perm            = round(sig_sub_table.p_perm_100_400ms, 4);
simple_table.p_fdr             = round(sig_sub_table.p_fdr, 4);
simple_table.dkt_anatomy       = sig_sub_table.dkt_anatomy;

out_simple_mat = fullfile(res_table_dir, 'color_significant_channels_simple.mat');
save(out_simple_mat, 'simple_table');
fprintf('[+] 显著电极极简表(.mat)已保存至: %s\n', out_simple_mat);

% 打印简报
n_tot = height(res_table);
n_sig = sum(res_table.is_significant);
n_pos = sum(res_table.is_significant & strcmp(res_table.concordance_type, 'Concordant_Positive'));
n_neg = sum(res_table.is_significant & strcmp(res_table.concordance_type, 'Concordant_Negative'));
n_bias = sum(res_table.is_significant & strcmp(res_table.concordance_type, 'Category_Biased'));

fprintf('\n========================================================================\n');
fprintf('  【C04 统计筛选完成简报】\n');
fprintf('  总测试数: %d | 总体显著通道数: %d\n', n_tot, n_sig);
fprintf('  - 总体正向增强: %d\n', n_pos);
fprintf('  - 总体负向抑制: %d\n', n_neg);
fprintf('  - 类别偏向不同向: %d\n', n_bias);
fprintf('========================================================================\n');

%% 辅助函数: FDR (Benjamini-Hochberg)
function [h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(pvals, q)
    if nargin < 2, q = 0.05; end
    pvals = pvals(:);
    m = length(pvals);
    [p_sorted, sort_ids] = sort(pvals);
    [~, unsort_ids] = sort(sort_ids);
    
    i = (1:m)';
    thresh = (i / m) * q;
    h_sorted = p_sorted <= thresh;
    
    max_id = find(h_sorted, 1, 'last');
    if isempty(max_id)
        crit_p = 0;
        h = false(m, 1);
    else
        crit_p = p_sorted(max_id);
        h = pvals <= crit_p;
    end
    
    % 调整 p 值计算
    adj_p_sorted = zeros(m, 1);
    adj_p_sorted(m) = p_sorted(m);
    for k = (m - 1):-1:1
        adj_p_sorted(k) = min(adj_p_sorted(k + 1), p_sorted(k) * m / k);
    end
    adj_p = min(1, adj_p_sorted(unsort_ids));
    adj_ci_cvrg = 1 - q;
end
