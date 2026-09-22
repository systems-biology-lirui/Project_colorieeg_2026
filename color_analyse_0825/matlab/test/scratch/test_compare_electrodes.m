% test_compare_electrodes.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
loc_dir   = fullfile(proj_root, 'metadata', 'ieeg_location');

% 1. 读取所有 ieeg_location 触点
loc_subs = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008'};
all_contacts = [];

for s = 1:numel(loc_subs)
    sub = loc_subs{s};
    f_xlsx = fullfile(loc_dir, sprintf('%s_ieegloc.xlsx', sub));
    f_tsv  = fullfile(loc_dir, sprintf('%s.tsv', sub));
    
    if isfile(f_xlsx)
        t = readtable(f_xlsx, 'VariableNamingRule', 'preserve');
        ch_col = 1;
        mni_col = find(strcmpi(t.Properties.VariableNames, 'MNI'), 1);
        for r = 1:height(t)
            ch_name = string(t{r, ch_col});
            mni_str = string(t{r, mni_col});
            nums = sscanf(char(strrep(strrep(mni_str, '[', ''), ']', '')), '%f,%f,%f');
            if numel(nums) == 3 && (nums(1)~=0 || nums(2)~=0 || nums(3)~=0)
                rec.subject = sub;
                rec.channel = char(ch_name);
                rec.mni_x = nums(1);
                rec.mni_y = nums(2);
                rec.mni_z = nums(3);
                rec.source = 'ieeg_location';
                all_contacts = [all_contacts; rec];
            end
        end
    end
end
t_raw = struct2table(all_contacts);
fprintf('从 ieeg_location 成功解析具有有效 MNI 坐标的原始触点数: %d\n', height(t_raw));
for s = 1:numel(loc_subs)
    sub = loc_subs{s};
    n_s = sum(strcmp(t_raw.subject, sub));
    fprintf('  %s: %d 个有效触点\n', sub, n_s);
end

% 2. 比较分析通道 (793 通道)
d = load(fullfile(proj_root, 'result', 'tables', 'color_effects_summary.mat'), 'all_tbl');
t_all = d.all_tbl;
[~, u_idx] = unique(strcat(t_all.subject, '_', t_all.channel));
t_ana = t_all(u_idx, :);
fprintf('\n分析通道数: %d\n', height(t_ana));
