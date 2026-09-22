%% test_inspect_combinations.m
% 详细列出 Task 1 中实际存在的全部频段组合与对应电极数
clear; clc;

proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
tab_file  = fullfile(proj_root, 'result', 'tables', 'color_effects_summary.mat');
d = load(tab_file);
tbl = d.all_tbl;

bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};
band_labels = {'\delta', '\theta', '\alpha', '\beta', '\gamma', 'High \gamma'};
n_bands = numel(bands);

ch_keys = unique(strcat(tbl.subject, '_', tbl.channel), 'stable');
n_chs   = numel(ch_keys);

sig_mat = false(n_chs, n_bands);
for i = 1:n_chs
    key = ch_keys{i};
    parts = strsplit(key, '_');
    sub = parts{1}; ch = parts{2};
    sub_tbl = tbl(strcmp(tbl.subject, sub) & strcmp(tbl.channel, ch), :);
    for b = 1:n_bands
        r = sub_tbl(strcmp(sub_tbl.freq_band, bands{b}), :);
        if ~isempty(r) && (r.is_significant(1) == 1)
            sig_mat(i, b) = true;
        end
    end
end

% 统计唯一组合
[u_comb, ~, ic] = unique(sig_mat, 'rows');
counts = accumarray(ic, 1);
deg = sum(u_comb, 2);

% 排序：先按 degree (0, 1, 2, ...)，再按 counts 降序
[~, sort_idx] = sortrows([deg, -counts]);
u_comb_sorted = u_comb(sort_idx, :);
counts_sorted = counts(sort_idx);
deg_sorted    = deg(sort_idx);

fprintf('=== 实际存在的频段组合明细 (共 %d 种不同组合) ===\n', numel(counts));
for c = 1:numel(counts)
    b_idx = find(u_comb_sorted(c, :));
    if isempty(b_idx)
        b_str = 'None (未响应电极)';
    else
        b_str = strjoin(bands(b_idx), ' + ');
    end
    fprintf('[Degree %d] %-40s : %3d 个电极\n', deg_sorted(c), b_str, counts_sorted(c));
end
