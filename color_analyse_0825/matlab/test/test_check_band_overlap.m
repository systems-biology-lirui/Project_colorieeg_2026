%% test_check_band_overlap.m
% 检查 Task 1 电极在 6 个频段上的显著性交集重叠分布 (UpSet 数据统计)
clear; clc;

proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
tab_file  = fullfile(proj_root, 'result', 'tables', 'color_effects_summary.mat');

if ~isfile(tab_file)
    error('未找到数据文件: %s', tab_file);
end

d = load(tab_file);
tbl = d.all_tbl;

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
band_labels = {'\delta', '\theta', '\alpha', '\beta', '\gamma', 'High \gamma'};
n_bands = numel(bands);

% 提取唯一电极列表 (被试 + 通道)
ch_keys = unique(strcat(tbl.subject, '_', tbl.channel), 'stable');
n_chs   = numel(ch_keys);

fprintf('总有效拉普拉斯电极数: %d\n', n_chs);

% 构建 0/1 矩阵 [n_chs x n_bands]
% 模式 A: 总体显著 (is_significant == 1, 未校正 p < 0.05)
% 模式 B: 严格同向显著 (Concordant Positive or Negative)
sig_mat_all = false(n_chs, n_bands);
sig_mat_con = false(n_chs, n_bands);

for i = 1:n_chs
    key = ch_keys{i};
    parts = strsplit(key, '_');
    sub = parts{1}; ch = parts{2};
    
    sub_tbl = tbl(strcmp(tbl.subject, sub) & strcmp(tbl.channel, ch), :);
    for b = 1:n_bands
        b_name = bands{b};
        r = sub_tbl(strcmp(sub_tbl.freq_band, b_name), :);
        if ~isempty(r)
            if r.is_significant(1) == 1
                sig_mat_all(i, b) = true;
                if ismember(r.concordance_type{1}, {'Concordant_Positive', 'Concordant_Negative'})
                    sig_mat_con(i, b) = true;
                end
            end
        end
    end
end

fprintf('\n--- 模式 A: 总体显著 (is_significant == 1) ---\n');
n_sig_per_ch = sum(sig_mat_all, 2);
for k = 0:6
    fprintf('  显著频段数 = %d: %d 个电极 (%.2f%%)\n', k, sum(n_sig_per_ch == k), sum(n_sig_per_ch == k)/n_chs*100);
end

fprintf('\n--- 模式 B: 四类别严格同向显著 (Concordant) ---\n');
n_con_per_ch = sum(sig_mat_con, 2);
for k = 0:6
    fprintf('  显著频段数 = %d: %d 个电极 (%.2f%%)\n', k, sum(n_con_per_ch == k), sum(n_con_per_ch == k)/n_chs*100);
end
