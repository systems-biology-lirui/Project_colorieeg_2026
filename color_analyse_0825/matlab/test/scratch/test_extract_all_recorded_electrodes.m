% test_extract_all_recorded_electrodes.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
loc_dir   = fullfile(proj_root, 'metadata', 'ieeg_location');

subs = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008'};
all_recs = [];

for s = 1:numel(subs)
    sub = subs{s};
    fn = fullfile(loc_dir, sprintf('%s_ieegloc.xlsx', sub));
    t = readtable(fn, 'VariableNamingRule', 'preserve');
    
    ch_col = 1;
    mni_col = find(strcmpi(t.Properties.VariableNames, 'MNI'), 1);
    dkt_col = find(strcmpi(t.Properties.VariableNames, 'DKT'), 1);
    aseg_col = find(strcmpi(t.Properties.VariableNames, 'ASEG'), 1);
    aal_col = find(contains(t.Properties.VariableNames, 'AAL', 'IgnoreCase', true), 1);
    
    for r = 1:height(t)
        ch_name = string(t{r, ch_col});
        mni_str = string(t{r, mni_col});
        nums = sscanf(char(strrep(strrep(mni_str, '[', ''), ']', '')), '%f,%f,%f');
        if numel(nums) == 3 && (nums(1)~=0 || nums(2)~=0 || nums(3)~=0)
            rec.electrode_label = sprintf('%s_%s', sub, char(ch_name));
            rec.subject         = sub;
            rec.channel         = char(ch_name);
            rec.mni_x           = nums(1);
            rec.mni_y           = nums(2);
            rec.mni_z           = nums(3);
            
            dkt_str = '';
            if ~isempty(dkt_col), dkt_str = char(string(t{r, dkt_col})); end
            rec.dkt_anatomy = dkt_str;
            
            aseg_str = '';
            if ~isempty(aseg_col), aseg_str = char(string(t{r, aseg_col})); end
            rec.aseg_anatomy = aseg_str;
            
            aal_str = '';
            if ~isempty(aal_col), aal_str = char(string(t{r, aal_col})); end
            rec.aal_anatomy = aal_str;
            
            all_recs = [all_recs; rec];
        end
    end
end

all_elec_tbl = struct2table(all_recs);
fprintf('成功提取已记录电极触点数: %d\n', height(all_elec_tbl));
disp(head(all_elec_tbl, 5));

% 保存到 result/tables/all_recorded_electrodes.mat
out_mat = fullfile(proj_root, 'result', 'tables', 'all_recorded_electrodes.mat');
out_csv = fullfile(proj_root, 'result', 'tables', 'all_recorded_electrodes.csv');
save(out_mat, 'all_elec_tbl', '-v7.3');
writetable(all_elec_tbl, out_csv);
fprintf('[✓] 已保存至: %s\n', out_mat);
