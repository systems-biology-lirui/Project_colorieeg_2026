% test_task3_cluster_onset_y_corr.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
res_root  = fullfile(proj_root, 'result');
c04_file  = fullfile(res_root, 'tables', 'color_effects_summary.mat');
tc_mat    = fullfile(res_root, 'tables', 'single_channel_decoding_timecourses.mat');
loc_dir   = fullfile(proj_root, 'metadata', 'ieeg_location');
sum_mat   = fullfile(res_root, 'tables', 'task3_purecolor_decoding_summary.mat');

% 1. Load summary table and timecourses
load(sum_mat, 'summary_table');
load(tc_mat, 'task3_purecolor', 'time_ms');

% Load C04 for coordinates
c04_data = load(c04_file);
if isfield(c04_data, 'all_tbl'), c04_tbl = c04_data.all_tbl; else, c04_tbl = c04_data.res_table; end

fprintf('Total channels in task3: %d\n', height(summary_table));

% Pre-extract coordinates function
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
    
    % Find timecourse entry
    tc_idx = find(strcmp({task3_purecolor.key}, key), 1);
    if isempty(tc_idx)
        continue;
    end
    entry = task3_purecolor(tc_idx);
    
    % Find clusters with p_pointwise < 0.05 and time >= 0
    p_pt = entry.p_pointwise;
    sig_mask = (p_pt < 0.05) & (time_ms >= 0);
    
    % Contiguous clusters
    in_c = false; cls = []; c_s = 1;
    for w = 1:numel(time_ms)
        if sig_mask(w) && ~in_c
            in_c = true; c_s = w;
        elseif ~sig_mask(w) && in_c
            in_c = false; cls = [cls; c_s, w-1]; %#ok<AGROW>
        end
    end
    if in_c, cls = [cls; c_s, numel(time_ms)]; end
    
    % Cluster stats
    n_cl = size(cls, 1);
    cl_lens = [];
    cl_starts = [];
    cl_ends = [];
    cl_masses = [];
    for c_i = 1:n_cl
        idx_r = cls(c_i, 1):cls(c_i, 2);
        cl_lens(c_i) = numel(idx_r);
        cl_starts(c_i) = time_ms(cls(c_i, 1));
        cl_ends(c_i) = time_ms(cls(c_i, 2));
        cl_masses(c_i) = sum(entry.acc_joint(idx_r) - 0.5);
    end
    
    max_len = 0;
    first_start_4 = NaN;
    if ~isempty(cl_lens)
        max_len = max(cl_lens);
        % First cluster with >= 4 points
        idx_4 = find(cl_lens >= 4, 1);
        if ~isempty(idx_4)
            first_start_4 = cl_starts(idx_4);
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
    r.cl_starts       = cl_starts;
    r.cl_ends         = cl_ends;
    r.cl_lens         = cl_lens;
    r.first_start_4   = first_start_4;
    
    % Find start time for significant cluster if has_sig_cluster == 1
    if r.has_sig_cluster == 1 && ~isempty(cl_starts)
        % Which cluster is the significant one? Typically the largest mass or first
        % Let's see: if multiple clusters, which one?
        [~, max_m_i] = max(cl_masses);
        r.sig_cl_start = cl_starts(max_m_i);
        r.sig_cl_end   = cl_ends(max_m_i);
    else
        r.sig_cl_start = NaN;
        r.sig_cl_end   = NaN;
    end
    
    records = [records; r]; %#ok<AGROW>
end

% 2. Inspect has_sig_cluster == 1 channels
sig_recs = records([records.has_sig_cluster] == 1);
fprintf('\n=== 14 Significant Cluster Channels ===\n');
for k = 1:numel(sig_recs)
    r = sig_recs(k);
    fprintf('%s-%s: MNI_Y=%.1f, sig_cl_start=%d ms, sig_cl_end=%d ms, peak=%d ms, max_len=%d, band=%s\n', ...
        r.subject, r.channel, r.mni_y, r.sig_cl_start, r.sig_cl_end, r.peak_time_ms, r.max_cl_len, r.best_single_band);
end

% Check correlation for strict significant channels
valid_strict = ~isnan([sig_recs.mni_y]) & ~isnan([sig_recs.sig_cl_start]);
y_strict = [sig_recs(valid_strict).mni_y]';
t_strict = [sig_recs(valid_strict).sig_cl_start]';
[r_s, p_s] = corr(y_strict, t_strict);
fprintf('\n>>> Strict (has_sig_cluster == 1, N=%d): r = %.3f, p = %.4f <<<\n', numel(y_strict), r_s, p_s);

% 3. Inspect channels that do NOT have sig cluster, but have max_cl_len >= 4
near_recs = records([records.has_sig_cluster] == 0 & [records.max_cl_len] >= 4);
fprintf('\n=== Channels with NO sig cluster, but max_cl_len >= 4 (N=%d) ===\n', numel(near_recs));
for k = 1:numel(near_recs)
    r = near_recs(k);
    fprintf('%s-%s: MNI_Y=%.1f, first_start_4=%d ms, max_len=%d, peak=%d ms, peak_acc=%.3f, band=%s\n', ...
        r.subject, r.channel, r.mni_y, r.first_start_4, r.max_cl_len, r.peak_time_ms, r.peak_acc_joint, r.best_single_band);
end

% What if we include them?
comb_recs = [sig_recs; near_recs];
valid_comb = ~isnan([comb_recs.mni_y]);
y_comb = zeros(numel(comb_recs), 1);
t_comb = zeros(numel(comb_recs), 1);
for k = 1:numel(comb_recs)
    y_comb(k) = comb_recs(k).mni_y;
    if comb_recs(k).has_sig_cluster == 1
        t_comb(k) = comb_recs(k).sig_cl_start;
    else
        t_comb(k) = comb_recs(k).first_start_4;
    end
end
[r_c, p_c] = corr(y_comb, t_comb);
fprintf('\n>>> Combined (Strict + >=4 consecutive points, N=%d): r = %.3f, p = %.4f <<<\n', numel(y_comb), r_c, p_c);
