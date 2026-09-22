% test_task3_detailed_correlations.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
res_root  = fullfile(proj_root, 'result');
c04_file  = fullfile(res_root, 'tables', 'color_effects_summary.mat');
tc_mat    = fullfile(res_root, 'tables', 'single_channel_decoding_timecourses.mat');
loc_dir   = fullfile(proj_root, 'metadata', 'ieeg_location');
sum_mat   = fullfile(res_root, 'tables', 'task3_purecolor_decoding_summary.mat');

load(sum_mat, 'summary_table');
load(tc_mat, 'task3_purecolor', 'time_ms');

c04_data = load(c04_file);
if isfield(c04_data, 'all_tbl'), c04_tbl = c04_data.all_tbl; else, c04_tbl = c04_data.res_table; end

n_ch = height(summary_table);
records = struct([]);

for i = 1:n_ch
    sub = char(summary_table.subject{i});
    ch  = char(summary_table.channel{i});
    key = sprintf('%s_%s', sub, ch);
    
    % Find coordinate
    mx = NaN; my = NaN; mz = NaN;
    m_c04 = strcmp(string(c04_tbl.subject), string(sub)) & strcmp(string(c04_tbl.channel), string(ch));
    idx_c04 = find(m_c04, 1);
    if ~isempty(idx_c04) && ~isnan(c04_tbl.mni_y(idx_c04))
        mx = c04_tbl.mni_x(idx_c04);
        my = c04_tbl.mni_y(idx_c04);
        mz = c04_tbl.mni_z(idx_c04);
    end
    
    if isnan(my)
        f_xlsx = fullfile(loc_dir, sprintf('%s_ieegloc.xlsx', sub));
        f_tsv  = fullfile(loc_dir, sprintf('%s.tsv', sub));
        if isfile(f_xlsx)
            t_loc = readtable(f_xlsx, 'VariableNamingRule', 'preserve');
            c_m = strcmp(string(table2cell(t_loc(:, 1))), string(ch));
            c_idx = find(c_m, 1);
            if ~isempty(c_idx)
                mni_col = find(strcmpi(t_loc.Properties.VariableNames, 'MNI'), 1);
                if ~isempty(mni_col)
                    mni_raw = string(t_loc{c_idx, mni_col});
                    nums = sscanf(char(strrep(strrep(mni_raw, '[', ''), ']', '')), '%f,%f,%f');
                    if numel(nums) == 3, mx = nums(1); my = nums(2); mz = nums(3); end
                elseif width(t_loc) >= 4
                    vals = table2array(t_loc(c_idx, 2:4));
                    if isnumeric(vals) && ~any(isnan(vals)), mx = vals(1); my = vals(2); mz = vals(3); end
                end
            end
        elseif isfile(f_tsv)
            t_tsv = readtable(f_tsv, 'FileType', 'text', 'Delimiter', '\t');
            c_m = strcmp(string(t_tsv.Channel), string(ch));
            c_idx = find(c_m, 1);
            if ~isempty(c_idx) && ismember('MNI', t_tsv.Properties.VariableNames)
                mni_raw = string(t_tsv.MNI(c_idx));
                nums = sscanf(char(strrep(strrep(mni_raw, '[', ''), ']', '')), '%f,%f,%f');
                if numel(nums) == 3, mx = nums(1); my = nums(2); mz = nums(3); end
            end
        end
    end
    
    tc_idx = find(strcmp({task3_purecolor.key}, key), 1);
    if isempty(tc_idx), continue; end
    entry = task3_purecolor(tc_idx);
    
    p_pt = entry.p_pointwise;
    sig_mask = (p_pt < 0.05) & (time_ms >= 0);
    
    in_c = false; cls = []; c_s = 1;
    for w = 1:numel(time_ms)
        if sig_mask(w) && ~in_c
            in_c = true; c_s = w;
        elseif ~sig_mask(w) && in_c
            in_c = false; cls = [cls; c_s, w-1]; %#ok<AGROW>
        end
    end
    if in_c, cls = [cls; c_s, numel(time_ms)]; end
    
    n_cl = size(cls, 1);
    cl_lens = []; cl_starts = []; cl_ends = []; cl_masses = [];
    for c_i = 1:n_cl
        idx_r = cls(c_i, 1):cls(c_i, 2);
        cl_lens(c_i) = numel(idx_r);
        cl_starts(c_i) = time_ms(cls(c_i, 1));
        cl_ends(c_i) = time_ms(cls(c_i, 2));
        cl_masses(c_i) = sum(entry.acc_joint(idx_r) - 0.5);
    end
    
    max_len = 0;
    first_start_4 = NaN;
    first_end_4   = NaN;
    if ~isempty(cl_lens)
        max_len = max(cl_lens);
        idx_4 = find(cl_lens >= 4, 1);
        if ~isempty(idx_4)
            first_start_4 = cl_starts(idx_4);
            first_end_4   = cl_ends(idx_4);
        end
    end
    
    r = struct();
    r.subject = sub;
    r.channel = ch;
    r.key     = key;
    r.mni_x   = mx;
    r.mni_y   = my;
    r.mni_z   = mz;
    r.has_sig_cluster = summary_table.has_sig_cluster(i);
    r.peak_acc_joint  = summary_table.peak_acc_joint(i);
    r.peak_time_ms    = summary_table.peak_time_ms(i);
    r.best_single_band= char(summary_table.best_single_band{i});
    r.n_clusters      = n_cl;
    r.max_cl_len      = max_len;
    r.first_start_4   = first_start_4;
    r.first_end_4     = first_end_4;
    
    if r.has_sig_cluster == 1 && ~isempty(cl_starts)
        [~, max_m_i] = max(cl_masses);
        r.sig_cl_start = cl_starts(max_m_i);
        r.sig_cl_end   = cl_ends(max_m_i);
    else
        r.sig_cl_start = NaN;
        r.sig_cl_end   = NaN;
    end
    
    records = [records; r]; %#ok<AGROW>
end

% 1. Strict Set (has_sig_cluster == 1 and ~isnan(mni_y))
strict_mask = [records.has_sig_cluster] == 1 & ~isnan([records.mni_y]) & ~isnan([records.sig_cl_start]);
s_recs = records(strict_mask);
s_y = [s_recs.mni_y]';
s_t = [s_recs.sig_cl_start]';
[r_strict, p_strict] = corr(s_y, s_t);
fprintf('=======================================================\n');
fprintf('1. 严格模式 (Strict: has_sig_cluster == 1, N=%d):\n', numel(s_y));
fprintf('   Pearson r = %.4f, p = %.4f\n', r_strict, p_strict);
[r_strict_spear, p_strict_spear] = corr(s_y, s_t, 'type', 'Spearman');
fprintf('   Spearman rho = %.4f, p = %.4f\n', r_strict_spear, p_strict_spear);
fprintf('=======================================================\n');

% 2. Near-sig Set: max_cl_len >= 4 (>=4 consecutive time points of pointwise p < 0.05)
% Combined set = strict + near-sig
comb_recs = [];
for k = 1:numel(records)
    r = records(k);
    if isnan(r.mni_y), continue; end
    if r.has_sig_cluster == 1 && ~isnan(r.sig_cl_start)
        r.t_onset = r.sig_cl_start;
        r.t_offset= r.sig_cl_end;
        r.is_strict = 1;
        comb_recs = [comb_recs; r]; %#ok<AGROW>
    elseif r.has_sig_cluster == 0 && ~isnan(r.first_start_4)
        r.t_onset = r.first_start_4;
        r.t_offset= r.first_end_4;
        r.is_strict = 0;
        comb_recs = [comb_recs; r]; %#ok<AGROW>
    end
end

c_y = [comb_recs.mni_y]';
c_t = [comb_recs.t_onset]';
[r_comb, p_comb] = corr(c_y, c_t);
[r_comb_spear, p_comb_spear] = corr(c_y, c_t, 'type', 'Spearman');
fprintf('=======================================================\n');
fprintf('2. 宽松/接近显著模式 (Combined: Strict + 连续>=4点显著, N=%d):\n', numel(c_y));
fprintf('   Pearson r = %.4f, p = %.4f\n', r_comb, p_comb);
fprintf('   Spearman rho = %.4f, p = %.4f\n', r_comb_spear, p_comb_spear);
fprintf('   其中严格显著: %d, 接近显著: %d\n', sum([comb_recs.is_strict] == 1), sum([comb_recs.is_strict] == 0));
fprintf('=======================================================\n');

% Check if sub006_G3 is in comb_recs
g3_comb = find(strcmp({comb_recs.key}, 'sub006_G3'));
if ~isempty(g3_comb)
    fprintf('sub006_G3 successfully included in Combined set! MNI_Y=%.1f, t_onset=%d ms\n', ...
        comb_recs(g3_comb).mni_y, comb_recs(g3_comb).t_onset);
end

% Also test with peak_time instead of onset time just to see comparison:
c_peak = [comb_recs.peak_time_ms]';
[r_comb_peak, p_comb_peak] = corr(c_y, c_peak);
fprintf('   (参考: 若使用峰值时间 peak_time: r = %.4f, p = %.4f)\n', r_comb_peak, p_comb_peak);
s_peak = [s_recs.peak_time_ms]';
[r_strict_peak, p_strict_peak] = corr(s_y, s_peak);
fprintf('   (参考: 严格组若使用峰值时间 peak_time: r = %.4f, p = %.4f)\n', r_strict_peak, p_strict_peak);
