function conversion_summary_nrecord_x_field = C00_raw_to_seegdata_0825(varargin)
%RAW_TO_SEEGDATA_0825 将 Neuracle 原始数据转换为标准 .mat 格式并保存至 seegdata/。
%
% 转换后的 .mat 文件保存至 seegdata/sub00n/taskK.mat，包含以下标准变量：
%   data          [n_channels x n_timepoints] single 单精度连续电压矩阵
%   chanel_name   [n_channels x 1] cellstr 通道名称列表
%   channel_name  [n_channels x 1] cellstr 通道名称别名（确保拼写兼容）
%   event         [1 x n_events] struct 包含 type 与 latency 的事件记录
%   fs            标量数值，采样率（Hz），例如 1000
%
% 默认调用：
%   summary = C00_raw_to_seegdata_0825();
%
% 自定义参数调用：
%   summary = C00_raw_to_seegdata_0825('subjects', {'sub001'}, 'tasks', [1, 2], ...
%       'overwrite', true, 'dry_run', false);
% 'dry_run'参数代表只进行数据完备性检验

% =========================================================================
% 步骤 0. 解析路径与转换参数配置
% 输入：varargin（可选参数）
% 输出：配置变量与过滤后的任务映射表
% =========================================================================
matlab_script_dir = fileparts(mfilename('fullpath'));
analysis_0825_root = fileparts(matlab_script_dir);
project_root = fileparts(analysis_0825_root);

default_eeglab_root = fullfile('E:', 'matlab_tools', 'eeglab2026.0.0');
default_plugin_root = fullfile(default_eeglab_root, 'plugins', ...
    'NeuracleEEGFileReader1.1.1');
default_mapping_csv = fullfile(analysis_0825_root, 'metadata', ...
    'raw_to_seeg_task_map.csv');
if ~isfile(default_mapping_csv)
    default_mapping_csv = fullfile(project_root, 'color_analyse_0825', ...
        'metadata', 'raw_to_seeg_task_map.csv');
end

argument_parser = inputParser;
addParameter(argument_parser, 'project_root', project_root, ...
    @(value) ischar(value) || isstring(value));
addParameter(argument_parser, 'eeglab_root', default_eeglab_root, ...
    @(value) ischar(value) || isstring(value));
addParameter(argument_parser, 'neuracle_plugin_root', default_plugin_root, ...
    @(value) ischar(value) || isstring(value));
addParameter(argument_parser, 'mapping_csv', default_mapping_csv, ...
    @(value) ischar(value) || isstring(value));
addParameter(argument_parser, 'subjects', {}, ...
    @(value) iscell(value) || isstring(value) || ischar(value));
addParameter(argument_parser, 'tasks', 1:3, ...
    @(value) isnumeric(value) && all(ismember(value, [1, 2, 3])));
addParameter(argument_parser, 'overwrite', false, ...
    @(value) islogical(value) || (isnumeric(value) && isscalar(value)));
addParameter(argument_parser, 'dry_run', false, ...
    @(value) islogical(value) || (isnumeric(value) && isscalar(value)));
parse(argument_parser, varargin{:});

resolved_project_root = char(string(argument_parser.Results.project_root));
resolved_eeglab_root = char(string(argument_parser.Results.eeglab_root));
resolved_plugin_root = char(string(argument_parser.Results.neuracle_plugin_root));
resolved_mapping_csv = char(string(argument_parser.Results.mapping_csv));
filter_subjects_nsubject_x1 = cellstr(string(argument_parser.Results.subjects(:)));
filter_subjects_nsubject_x1 = filter_subjects_nsubject_x1( ...
    strlength(string(filter_subjects_nsubject_x1)) > 0);
% 统一将 test00n 转换为 sub00n
filter_subjects_nsubject_x1 = regexprep(filter_subjects_nsubject_x1, '^test', 'sub');
filter_tasks_1_xtask = unique(double(argument_parser.Results.tasks(:)'));
overwrite_existing_mat = logical(argument_parser.Results.overwrite);
dry_run_mode = logical(argument_parser.Results.dry_run);

assert(isfolder(resolved_eeglab_root), ...
    'EEGLAB 目录未找到: %s', resolved_eeglab_root);
assert(isfolder(resolved_plugin_root), ...
    'Neuracle 插件目录未找到: %s', resolved_plugin_root);
assert(isfile(resolved_mapping_csv), ...
    '映射表 CSV 未找到: %s', resolved_mapping_csv);

addpath(resolved_eeglab_root);
addpath(resolved_plugin_root);

mapping_table_nraw_x_field = readtable(resolved_mapping_csv, ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
required_mapping_fields_1_x5 = ["subject", "task_num", "raw_sessions", ...
    "output_dir", "output_stem"];
assert(all(ismember(required_mapping_fields_1_x5, ...
    string(mapping_table_nraw_x_field.Properties.VariableNames))), ...
    '映射表缺少必要列: %s', strjoin(required_mapping_fields_1_x5, ', '));

% =========================================================================
% 步骤 1. 初始化 EEGLAB 环境（无 GUI 模式）
% =========================================================================
if ~dry_run_mode
    [~, ~, ~, ~] = eeglab('nogui');
end

% =========================================================================
% 步骤 2. 逐行遍历映射表，转换并导出 .mat 格式
% 输入：Neuracle rawdata 目录（data.bdf, evt.bdf）
% 输出：seegdata/sub00n/taskK.mat（包含 data, chanel_name, event, fs）
% =========================================================================
records_cell_nrecord_x1 = cell(height(mapping_table_nraw_x_field), 1);
n_records_kept = 0;

for row_index = 1:height(mapping_table_nraw_x_field)
    subject_id = char(strtrim(string(mapping_table_nraw_x_field.subject(row_index))));
    task_number = double(mapping_table_nraw_x_field.task_num(row_index));
    
    % 过滤指定的被试和任务
    if ~isempty(filter_subjects_nsubject_x1) && ...
            ~any(strcmp(filter_subjects_nsubject_x1, subject_id))
        continue;
    end
    if ~isempty(filter_tasks_1_xtask) && ...
            ~any(filter_tasks_1_xtask == task_number)
        continue;
    end
    
    raw_session_names_1_xsession = split(string( ...
        mapping_table_nraw_x_field.raw_sessions(row_index)), ';');
    output_subfolder_name = char(strtrim(string( ...
        mapping_table_nraw_x_field.output_dir(row_index))));
    output_file_stem = char(strtrim(string( ...
        mapping_table_nraw_x_field.output_stem(row_index))));
    
    seegdata_target_dir = fullfile(resolved_project_root, 'seegdata', ...
        output_subfolder_name);
    output_mat_path = fullfile(seegdata_target_dir, [output_file_stem '.mat']);
    
    % 检查目标 .mat 是否已存在并获取元数据
    is_mat_exist = isfile(output_mat_path);
    mat_info = get_existing_mat_info(output_mat_path);
    
    conversion_record_1_x_field = struct( ...
        'subject', subject_id, ...
        'task_num', task_number, ...
        'output_path', output_mat_path, ...
        'path_checked', false, ...
        'data_checked', false, ...
        'converted', is_mat_exist, ...
        'n_channels', mat_info.n_channels, ...
        'n_timepoints', mat_info.n_timepoints, ...
        'n_events', mat_info.n_events, ...
        'sampling_rate_fs', mat_info.sampling_rate_fs);
    
    try
        if ~isfolder(seegdata_target_dir) && ~dry_run_mode
            mkdir(seegdata_target_dir);
        end
        
        % 检查是否跳过已有输出
        if is_mat_exist && ~overwrite_existing_mat && ~dry_run_mode
            conversion_record_1_x_field.path_checked = true;
            conversion_record_1_x_field.data_checked = true;
            conversion_record_1_x_field.converted = true;
            fprintf('[跳过已存在] %s task%d -> %s [通道=%d, 点数=%d, 事件=%d, fs=%g Hz]\n', ...
                subject_id, task_number, output_mat_path, ...
                mat_info.n_channels, mat_info.n_timepoints, ...
                mat_info.n_events, mat_info.sampling_rate_fs);
            n_records_kept = n_records_kept + 1;
            records_cell_nrecord_x1{n_records_kept} = conversion_record_1_x_field;
            continue;
        end
        
        % -----------------------------------------------------------------
        % 步骤 2.1 逐个读取 Session 并解析信号矩阵与事件
        % -----------------------------------------------------------------
        n_sessions = numel(raw_session_names_1_xsession);
        part_data_cell_1_xsession = cell(1, n_sessions);
        part_events_cell_1_xsession = cell(1, n_sessions);
        part_channellabels_cell_1_xsession = cell(1, n_sessions);
        part_fs_1_xsession = zeros(1, n_sessions);
        part_timepoints_1_xsession = zeros(1, n_sessions);
        
        for session_index = 1:n_sessions
            session_name = char(strtrim(raw_session_names_1_xsession(session_index)));
            session_data_folder = fullfile(resolved_project_root, 'rawdata', ...
                subject_id, session_name, '1', '1');
            data_bdf_path = fullfile(session_data_folder, 'data.bdf');
            evt_bdf_inner_path = fullfile(session_data_folder, 'evt.bdf');
            evt_bdf_parent_path = fullfile(resolved_project_root, 'rawdata', ...
                subject_id, session_name, '1', 'evt.bdf');
            
            assert(isfile(data_bdf_path), '缺少 data.bdf 文件: %s', data_bdf_path);
            
            if isfile(evt_bdf_inner_path)
                selected_evt_path = evt_bdf_inner_path;
                import_file_list_1_x2 = {'data.bdf', 'evt.bdf'};
                need_manual_event_attach = false;
            elseif isfile(evt_bdf_parent_path)
                selected_evt_path = evt_bdf_parent_path;
                import_file_list_1_x2 = {'data.bdf'};
                need_manual_event_attach = true;
            else
                error('缺少 evt.bdf 文件（已检查内部目录与上级目录）: %s', ...
                    session_data_folder);
            end
            
            if dry_run_mode
                fprintf('[检查通过] %s task%d session %d/%d: data=%s, evt=%s\n', ...
                    subject_id, task_number, session_index, n_sessions, ...
                    data_bdf_path, selected_evt_path);
                continue;
            end
            
            fprintf('[导入中] %s task%d session %d/%d: %s\n', ...
                subject_id, task_number, session_index, n_sessions, ...
                session_data_folder);
            
            imported_eeg_struct = pop_importNeuracle(import_file_list_1_x2, ...
                char(session_data_folder));
            
            % 如果事件文件在上级目录，则通过 read_bdf 手动读取并挂载
            if need_manual_event_attach
                event_bdf_header = read_bdf(char(selected_evt_path));
                raw_events_struct_array = cell2mat(event_bdf_header.event);
                n_raw_events = numel(raw_events_struct_array);
                attached_events_1_x_nevent = struct('type', {}, 'latency', {});
                for event_idx = 1:n_raw_events
                    attached_events_1_x_nevent(event_idx).type = ...
                        raw_events_struct_array(event_idx).eventvalue;
                    attached_events_1_x_nevent(event_idx).latency = round( ...
                        raw_events_struct_array(event_idx).offset_in_sec * ...
                        imported_eeg_struct.srate);
                end
                imported_eeg_struct.event = attached_events_1_x_nevent;
            end
            imported_eeg_struct = eeg_checkset(imported_eeg_struct);
            
            % 提取各 Session 的原始数值
            part_voltage_data_nchannel_x_ntime = single(imported_eeg_struct.data);
            part_n_channels = size(part_voltage_data_nchannel_x_ntime, 1);
            part_n_timepoints = size(part_voltage_data_nchannel_x_ntime, 2);
            
            channel_labels_nchannel_x1 = cellstr(strtrim(string( ...
                {imported_eeg_struct.chanlocs(1:part_n_channels).labels}')));
            
            part_data_cell_1_xsession{session_index} = ...
                part_voltage_data_nchannel_x_ntime;
            part_events_cell_1_xsession{session_index} = imported_eeg_struct.event;
            part_channellabels_cell_1_xsession{session_index} = ...
                channel_labels_nchannel_x1;
            part_fs_1_xsession(session_index) = double(imported_eeg_struct.srate);
            part_timepoints_1_xsession(session_index) = part_n_timepoints;
        end
        
        if dry_run_mode
            conversion_record_1_x_field.path_checked = true;
            conversion_record_1_x_field.data_checked = true;
            n_records_kept = n_records_kept + 1;
            records_cell_nrecord_x1{n_records_kept} = conversion_record_1_x_field;
            continue;
        end
        
        % -----------------------------------------------------------------
        % 步骤 2.2 多 Session 沿时间维度拼接与事件延迟（Latency）偏移修正
        % 输入 shape：多个 [n_channel x n_time_i]
        % 输出 shape：merged_data [n_channel x total_time]
        % -----------------------------------------------------------------
        merged_data_nchannel_x_ntime = part_data_cell_1_xsession{1};
        merged_channel_names_nchannel_x1 = part_channellabels_cell_1_xsession{1};
        merged_fs = part_fs_1_xsession(1);
        merged_events_1_x_nevent = part_events_cell_1_xsession{1};
        
        for session_index = 2:n_sessions
            next_session_data = part_data_cell_1_xsession{session_index};
            next_session_channellabels = ...
                part_channellabels_cell_1_xsession{session_index};
            next_session_fs = part_fs_1_xsession(session_index);
            next_session_events = part_events_cell_1_xsession{session_index};
            
            assert(size(next_session_data, 1) == size(merged_data_nchannel_x_ntime, 1), ...
                '多 session 拼接时通道数不一致: %d vs %d', ...
                size(next_session_data, 1), size(merged_data_nchannel_x_ntime, 1));
            assert(isequal(merged_channel_names_nchannel_x1, next_session_channellabels), ...
                '多 session 拼接时通道标签不一致。');
            assert(merged_fs == next_session_fs, ...
                '多 session 拼接时采样率不一致: %g vs %g', merged_fs, next_session_fs);
            
            time_sample_offset = size(merged_data_nchannel_x_ntime, 2);
            
            % 修正后续 Session 的事件延迟点
            offset_events_1_x_nevent = next_session_events;
            for event_idx = 1:numel(offset_events_1_x_nevent)
                offset_events_1_x_nevent(event_idx).latency = ...
                    offset_events_1_x_nevent(event_idx).latency + time_sample_offset;
            end
            
            % 横向拼接信号与事件
            merged_data_nchannel_x_ntime = [merged_data_nchannel_x_ntime, ...
                next_session_data]; %#ok<AGROW>
            if isempty(merged_events_1_x_nevent)
                merged_events_1_x_nevent = offset_events_1_x_nevent;
            else
                merged_events_1_x_nevent = [merged_events_1_x_nevent, ...
                    offset_events_1_x_nevent]; %#ok<AGROW>
            end
        end
        
        % -----------------------------------------------------------------
        % 步骤 2.3 组装并保存符合契约规范的 .mat 文件
        % 变量清单：
        %   data          [n_channels x n_timepoints] single
        %   chanel_name   [n_channels x 1] cellstr
        %   channel_name  [n_channels x 1] cellstr (别名)
        %   event         [1 x n_events] struct
        %   fs            标量数值
        % -----------------------------------------------------------------
        data = single(merged_data_nchannel_x_ntime); 
        chanel_name = cellstr(merged_channel_names_nchannel_x1); 
        channel_name = cellstr(merged_channel_names_nchannel_x1); 
        event = merged_events_1_x_nevent; 
        fs = double(merged_fs); 
        
        total_n_channels = size(data, 1);
        total_n_timepoints = size(data, 2);
        total_n_events = numel(event);
        
        assert(total_n_channels > 0 && total_n_timepoints > 0, ...
            '转换后数据维度为空。');
        assert(numel(chanel_name) == total_n_channels, ...
            '通道名数量与数据通道数不匹配。');
        assert(fs > 0, '采样率 fs 必须为正数。');
        
        % 原子化安全保存（写入临时文件后重命名，防止写入中断）
        temp_mat_path = [output_mat_path '.partial'];
        if isfile(temp_mat_path), delete(temp_mat_path); end
        save(temp_mat_path, 'data', 'chanel_name', 'channel_name', ...
            'event', 'fs', '-v7.3');
        movefile(temp_mat_path, output_mat_path, 'f');
        
        conversion_record_1_x_field.path_checked = true;
        conversion_record_1_x_field.data_checked = true;
        conversion_record_1_x_field.converted = true;
        conversion_record_1_x_field.n_channels = total_n_channels;
        conversion_record_1_x_field.n_timepoints = total_n_timepoints;
        conversion_record_1_x_field.n_events = total_n_events;
        conversion_record_1_x_field.sampling_rate_fs = fs;
        
        fprintf('[完成] %s task%d -> %s [通道=%d, 点数=%d, 事件=%d, fs=%g Hz]\n', ...
            subject_id, task_number, output_mat_path, ...
            total_n_channels, total_n_timepoints, total_n_events, fs);
        
    catch conversion_exception
        conversion_record_1_x_field.data_checked = false;
        conversion_record_1_x_field.converted = false;
        fprintf(2, '[错误] %s task%d: %s\n', ...
            subject_id, task_number, conversion_exception.message);
    end
    
    n_records_kept = n_records_kept + 1;
    records_cell_nrecord_x1{n_records_kept} = conversion_record_1_x_field;
end

% =========================================================================
% 步骤 3. 输出转换日志与汇总表格（增量更新，绝不删除原有其他被试记录）
% =========================================================================
log_file_path = fullfile(analysis_0825_root, 'metadata', ...
    'C00_原始数据转SEEG记录.csv');

total_rows = height(mapping_table_nraw_x_field);
full_summary_records = cell(total_rows, 1);

for row_idx = 1:total_rows
    sub_k = char(strtrim(string(mapping_table_nraw_x_field.subject(row_idx))));
    task_k = double(mapping_table_nraw_x_field.task_num(row_idx));
    out_dir_k = char(strtrim(string(mapping_table_nraw_x_field.output_dir(row_idx))));
    out_stem_k = char(strtrim(string(mapping_table_nraw_x_field.output_stem(row_idx))));
    mat_path_k = fullfile(resolved_project_root, 'seegdata', out_dir_k, [out_stem_k '.mat']);
    
    % 查找本次运行是否更新了该记录
    matched_run_idx = 0;
    for run_i = 1:n_records_kept
        if strcmp(records_cell_nrecord_x1{run_i}.subject, sub_k) && ...
                records_cell_nrecord_x1{run_i}.task_num == task_k
            matched_run_idx = run_i;
            break;
        end
    end
    
    if matched_run_idx > 0
        full_summary_records{row_idx} = records_cell_nrecord_x1{matched_run_idx};
    else
        % 未在本次运行列表中的被试，读取其当前状态并提取已有 .mat 的真实元数据
        is_converted = isfile(mat_path_k);
        mat_info = get_existing_mat_info(mat_path_k);
        
        full_summary_records{row_idx} = struct( ...
            'subject', sub_k, ...
            'task_num', task_k, ...
            'output_path', mat_path_k, ...
            'path_checked', is_converted, ...
            'data_checked', is_converted, ...
            'converted', is_converted, ...
            'n_channels', mat_info.n_channels, ...
            'n_timepoints', mat_info.n_timepoints, ...
            'n_events', mat_info.n_events, ...
            'sampling_rate_fs', mat_info.sampling_rate_fs);
    end
end

conversion_summary_nrecord_x_field = struct2table([full_summary_records{:}]);
writetable(conversion_summary_nrecord_x_field, log_file_path, 'Encoding', 'UTF-8');
fprintf('\n转换日志已保存至: %s\n', log_file_path);
fprintf('已完成转换: %d / %d 个记录。\n', ...
    sum(conversion_summary_nrecord_x_field.converted), ...
    height(conversion_summary_nrecord_x_field));
end

% =========================================================================
% 本地辅助函数：快速读取已有 .mat 的真实元数据（不加载全量矩阵，速度极快）
% =========================================================================
function meta_info = get_existing_mat_info(mat_path)
meta_info = struct('n_channels', 0, 'n_timepoints', 0, ...
    'n_events', 0, 'sampling_rate_fs', 0);
if ~isfile(mat_path)
    return;
end
try
    file_vars = whos('-file', mat_path);
    var_names = {file_vars.name};
    
    d_idx = find(strcmp(var_names, 'data'), 1);
    if ~isempty(d_idx)
        meta_info.n_channels = file_vars(d_idx).size(1);
        meta_info.n_timepoints = file_vars(d_idx).size(2);
    end
    
    e_idx = find(strcmp(var_names, 'event'), 1);
    if ~isempty(e_idx)
        meta_info.n_events = prod(file_vars(e_idx).size);
    end
    
    fs_idx = find(strcmp(var_names, 'fs'), 1);
    if ~isempty(fs_idx)
        mat_obj = matfile(mat_path);
        meta_info.sampling_rate_fs = double(mat_obj.fs);
    end
catch
    % 如果文件损坏或读取异常，保持 0
end
end
