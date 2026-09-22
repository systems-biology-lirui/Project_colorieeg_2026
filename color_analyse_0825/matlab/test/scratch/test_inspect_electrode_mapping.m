% test_inspect_electrode_mapping.m
clear; clc;
proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
loc_dir   = fullfile(proj_root, 'metadata', 'ieeg_location');

% 1. 载入 862 个原始触点
subs = {'sub001', 'sub002', 'sub003', 'sub004', 'sub005', 'sub006', 'sub007', 'sub008'};
raw_contacts = [];
for s = 1:numel(subs)
    sub = subs{s};
    fn = fullfile(loc_dir, sprintf('%s_ieegloc.xlsx', sub));
    t = readtable(fn, 'VariableNamingRule', 'preserve');
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
            raw_contacts = [raw_contacts; rec];
        end
    end
end
t_raw = struct2table(raw_contacts);
fprintf('1. 原始物理触点总数: %d\n', height(t_raw));

% 2. 检查 793 个分析通道在 862 原始触点中的匹配情况
d = load(fullfile(proj_root, 'result', 'tables', 'color_effects_summary.mat'), 'all_tbl');
t_all = d.all_tbl;
[~, u_idx] = unique(strcat(t_all.subject, '_', t_all.channel));
t_ana = t_all(u_idx, :);
fprintf('2. 分析通道总数: %d\n', height(t_ana));

matched_count = 0;
for i = 1:height(t_ana)
    s = t_ana.subject{i};
    c = t_ana.channel{i};
    m = strcmp(t_raw.subject, s) & strcmp(t_raw.channel, c);
    if any(m)
        matched_count = matched_count + 1;
    end
end
fprintf('   分析通道与原始触点精确匹配数: %d / %d\n', matched_count, height(t_ana));
