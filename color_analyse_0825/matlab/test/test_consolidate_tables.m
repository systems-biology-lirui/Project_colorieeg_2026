%% test_consolidate_tables.m
% =========================================================================
% 功能:
%   1. 将 450+ 个散乱的单通道时程 CSV 整合为一个结构体 single_channel_decoding_timecourses.mat
%   2. 将 concordant/non_concordant/task3 的 decoding summary CSV 转为对应的 .mat 表
%   3. 将 multisite_decoding_task2 的 3 个 CSV 合并为 multisite_decoding_task2_results.mat
%   4. 对所有浮点数准确率与统计值进行 round(val, 4) 规范化
% =========================================================================

proj_root = 'e:/liulab_project/Project_colorieeg_2026/color_analyse_0825';
tab_dir   = fullfile(proj_root, 'result', 'tables');

fprintf('>>> 开始汇总散乱表格与时程数据 ...\n');

%% 1. 转换 decoding summary CSV 为 .mat
summary_files = {
    'concordant_electrodes_decoding_summary', ...
    'non_concordant_electrodes_decoding_summary', ...
    'task3_purecolor_decoding_summary'
};

for i = 1:numel(summary_files)
    fname = summary_files{i};
    csv_f = fullfile(tab_dir, [fname, '.csv']);
    mat_f = fullfile(tab_dir, [fname, '.mat']);
    if isfile(csv_f)
        t = readtable(csv_f);
        % 对数值列进行 4 位小数四舍五入
        var_names = t.Properties.VariableNames;
        for v = 1:numel(var_names)
            vn = var_names{v};
            if isnumeric(t.(vn))
                t.(vn) = round(t.(vn), 4);
            end
        end
        summary_table = t;
        save(mat_f, 'summary_table');
        fprintf('  [+] 成功生成 %s (%d 行)\n', [fname, '.mat'], height(summary_table));
    end
end

%% 2. 汇总 multisite_decoding_task2 的 3 个 CSV
multi_summary_csv = fullfile(tab_dir, 'multisite_decoding_task2_summary.csv');
multi_subs_csv    = fullfile(tab_dir, 'multisite_decoding_task2_subjects_summary.csv');
multi_tc_csv      = fullfile(tab_dir, 'multisite_decoding_task2_timecourses.csv');
multi_mat         = fullfile(tab_dir, 'multisite_decoding_task2_results.mat');

multi_res = struct();
if isfile(multi_summary_csv)
    t = readtable(multi_summary_csv);
    for v = 1:numel(t.Properties.VariableNames)
        vn = t.Properties.VariableNames{v};
        if isnumeric(t.(vn)), t.(vn) = round(t.(vn), 4); end
    end
    multi_res.summary = t;
end

if isfile(multi_subs_csv)
    t = readtable(multi_subs_csv);
    for v = 1:numel(t.Properties.VariableNames)
        vn = t.Properties.VariableNames{v};
        if isnumeric(t.(vn)), t.(vn) = round(t.(vn), 4); end
    end
    multi_res.subjects_summary = t;
end

if isfile(multi_tc_csv)
    t = readtable(multi_tc_csv);
    for v = 1:numel(t.Properties.VariableNames)
        vn = t.Properties.VariableNames{v};
        if isnumeric(t.(vn)), t.(vn) = round(t.(vn), 4); end
    end
    multi_res.timecourses = t;
end

save(multi_mat, '-struct', 'multi_res');
fprintf('  [+] 成功生成 multisite_decoding_task2_results.mat\n');

%% 3. 汇总全部单通道时程文件
tc_dirs = struct(...
    'name', {'concordant', 'non_concordant', 'task3_purecolor'}, ...
    'folder', {
        fullfile(tab_dir, 'decoding_concordant_timecourses'), ...
        fullfile(tab_dir, 'decoding_non_concordant_timecourses'), ...
        fullfile(tab_dir, 'decoding_task3_purecolor_timecourses')
    } ...
);

all_tc_data = struct();
all_tc_data.time_ms = -200:20:800;
all_tc_data.bands   = {'Delta', 'Theta', 'Alpha', 'Beta', 'Low_Gamma', 'High_Gamma'};

for d = 1:numel(tc_dirs)
    d_name = tc_dirs(d).name;
    f_dir  = tc_dirs(d).folder;
    
    csv_list = dir(fullfile(f_dir, '*_decoding_timecourse.csv'));
    n_files  = numel(csv_list);
    fprintf('  -> 正在处理 %s (%d 个时程文件) ...\n', d_name, n_files);
    
    entries = repmat(struct(...
        'subject', '', ...
        'channel', '', ...
        'key', '', ...
        'acc_joint', [], ...
        'p_pointwise', [], ...
        'null_mean', [], ...
        'null_ci_upper', [], ...
        'null_ci_lower', [], ...
        'acc_Delta', [], ...
        'acc_Theta', [], ...
        'acc_Alpha', [], ...
        'acc_Beta', [], ...
        'acc_Low_Gamma', [], ...
        'acc_High_Gamma', [] ...
    ), n_files, 1);

    for k = 1:n_files
        csv_path = fullfile(f_dir, csv_list(k).name);
        tc_tbl   = readtable(csv_path);
        
        sub = char(tc_tbl.subject{1});
        ch  = char(tc_tbl.channel{1});
        
        entries(k).subject        = sub;
        entries(k).channel        = ch;
        entries(k).key            = sprintf('%s_%s', sub, ch);
        entries(k).acc_joint      = round(tc_tbl.acc_joint_smoothed', 4);
        entries(k).p_pointwise    = round(tc_tbl.p_pointwise', 4);
        
        if ismember('null_mean', tc_tbl.Properties.VariableNames)
            entries(k).null_mean     = round(tc_tbl.null_mean', 4);
            entries(k).null_ci_upper = round(tc_tbl.null_ci_upper', 4);
            entries(k).null_ci_lower = round(tc_tbl.null_ci_lower', 4);
        end
        
        if ismember('acc_Delta_smoothed', tc_tbl.Properties.VariableNames)
            entries(k).acc_Delta      = round(tc_tbl.acc_Delta_smoothed', 4);
            entries(k).acc_Theta      = round(tc_tbl.acc_Theta_smoothed', 4);
            entries(k).acc_Alpha      = round(tc_tbl.acc_Alpha_smoothed', 4);
            entries(k).acc_Beta       = round(tc_tbl.acc_Beta_smoothed', 4);
            entries(k).acc_Low_Gamma  = round(tc_tbl.acc_Low_Gamma_smoothed', 4);
            entries(k).acc_High_Gamma = round(tc_tbl.acc_High_Gamma_smoothed', 4);
        end
    end
    
    all_tc_data.(d_name) = entries;
end

out_tc_mat = fullfile(tab_dir, 'single_channel_decoding_timecourses.mat');
save(out_tc_mat, '-struct', 'all_tc_data');
mat_info = dir(out_tc_mat);
fprintf('  [+] 成功汇总保存 single_channel_decoding_timecourses.mat (大小: %.2f KB)\n', mat_info.bytes / 1024);

fprintf('>>> 汇总完成！\n');
