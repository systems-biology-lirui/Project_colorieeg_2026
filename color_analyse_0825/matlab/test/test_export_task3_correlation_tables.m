% test_export_task3_correlation_tables.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
res_root  = fullfile(proj_root, 'result');
c04_file  = fullfile(res_root, 'tables', 'color_effects_summary.mat');
tc_mat    = fullfile(res_root, 'tables', 'single_channel_decoding_timecourses.mat');
loc_dir   = fullfile(proj_root, 'metadata', 'ieeg_location');
sum_mat   = fullfile(res_root, 'tables', 'task3_purecolor_decoding_summary.mat');
out_tab_dir = fullfile(res_root, 'tables');

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
    
    r = struct();
    r.subject = string(sub);
    r.channel = string(ch);
    r.key     = string(key);
    r.mni_x   = mx;
    r.mni_y   = my;
    r.mni_z   = mz;
    r.has_sig_cluster = summary_table.has_sig_cluster(i);
    r.peak_acc_joint  = summary_table.peak_acc_joint(i);
    r.peak_time_ms    = summary_table.peak_time_ms(i);
    r.best_single_band= string(summary_table.best_single_band{i});
    
    if r.has_sig_cluster == 1 && ~isempty(cl_starts)
        [~, max_m] = max(cl_masses);
        r.cluster_onset_ms  = cl_starts(max_m);
        r.cluster_offset_ms = cl_ends(max_m);
        r.cluster_dur_ms    = cl_ends(max_m) - cl_starts(max_m);
        r.cluster_n_points  = cl_lens(max_m);
        r.inclusion_group   = "Strict (Significant Cluster)";
        r.is_strict         = 1;
        r.is_valid          = ~isnan(my);
    elseif any(cl_lens >= 4)
        c_cand = find(cl_lens >= 4);
        [~, best_cand] = max(cl_masses(c_cand));
        idx_m = c_cand(best_cand);
        r.cluster_onset_ms  = cl_starts(idx_m);
        r.cluster_offset_ms = cl_ends(idx_m);
        r.cluster_dur_ms    = cl_ends(idx_m) - cl_starts(idx_m);
        r.cluster_n_points  = cl_lens(idx_m);
        r.inclusion_group   = "Near-Significant (>=4 points)";
        r.is_strict         = 0;
        r.is_valid          = ~isnan(my);
    else
        r.cluster_onset_ms  = NaN;
        r.cluster_offset_ms = NaN;
        r.cluster_dur_ms    = NaN;
        r.cluster_n_points  = 0;
        r.inclusion_group   = "Non-Significant";
        r.is_strict         = 0;
        r.is_valid          = 0;
    end
    
    records = [records; r]; %#ok<AGROW>
end

tbl_all = struct2table(records);
disp('Inclusion counts:');
disp(groupsummary(tbl_all, 'inclusion_group'));

disp('Valid with MNI_Y counts:');
disp(groupsummary(tbl_all(tbl_all.is_valid == 1, :), 'inclusion_group'));
