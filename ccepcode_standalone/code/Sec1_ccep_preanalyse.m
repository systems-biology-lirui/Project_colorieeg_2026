% CCEP 预处理入口脚本。
% 这个独立版本把输入数据、元数据和输出工作区都约束在当前打包目录内，
% 便于单独拷贝到其他位置后继续使用。
clear;
clc;

script_timer = tic;
cfg = build_default_cfg(mfilename('fullpath'));
process_ccep_dataset(cfg);
fprintf('\n%s runtime: %.2f s\n', mfilename, toc(script_timer));

function cfg = build_default_cfg(script_path)
% 构建默认配置。
% 目录约定：
% 1. 原始 CCEP 数据放在 data/raw/<raw_subject_dir>/ 下。
% 2. 电极定位文件放在 data/metadata/<subject>/ 下。
% 3. 预处理输出统一写到 workspace/processed/<subject>/ccep/ 下。
project_root = fileparts(fileparts(script_path));
code_dir = fileparts(script_path);

cfg = struct();
cfg.project_root = project_root;
cfg.code_dir = code_dir;
cfg.data_root = fullfile(project_root, 'data');
cfg.workspace_root = fullfile(project_root, 'workspace');
cfg.raw_subject_dir = 'test1';
cfg.subject = 'test001';
cfg.input_set = fullfile(cfg.data_root, 'raw', cfg.raw_subject_dir, 'ccep.set');
cfg.output_dir = fullfile(cfg.workspace_root, 'processed', cfg.subject, 'ccep');
cfg.loc_file = fullfile(cfg.data_root, 'metadata', cfg.subject, sprintf('%s_ieegloc.xlsx', cfg.subject));
cfg.eeglab_root = fullfile(project_root, 'external_tools', 'eeglab');
cfg.stop_event_label = '刺激停止';
cfg.stim_frequency_hz = 1;
cfg.stim_duration_sec = 40;
cfg.resample_hz = 500;
cfg.notch_centers_hz = [50 100 150];
cfg.common_highpass_hz = 1;
cfg.common_lowpass_hz = 200;
cfg.erp_band_hz = [1 30];
cfg.tfa_band_hz = [1 150];
cfg.erp_epoch_window_sec = [-0.2 0.8];
cfg.tfa_epoch_window_sec = [-0.2 0.8];
cfg.baseline_ms = [-200 0];
end

function process_ccep_dataset(cfg)
% 执行 CCEP 连续数据到 ERP/TFA epoch 的完整预处理。
if ~exist(cfg.input_set, 'file')
    error('CCEP input file not found: %s', cfg.input_set);
end
if ~exist(cfg.output_dir, 'dir')
    mkdir(cfg.output_dir);
end

ensure_eeglab_on_path(cfg.eeglab_root);

[input_dir, input_name, input_ext] = fileparts(cfg.input_set);
EEG = pop_loadset('filename', [input_name input_ext], 'filepath', input_dir);
if EEG.trials ~= 1
    error('CCEP preprocessing expects a continuous dataset, but trials=%d.', EEG.trials);
end

fprintf('Loaded CCEP dataset: %s\n', cfg.input_set);
fprintf('Original shape: %d channels x %d samples @ %.2f Hz\n', EEG.nbchan, EEG.pnts, EEG.srate);

% 如果原始数据采样率与分析采样率不同，先统一重采样。
if EEG.srate ~= cfg.resample_hz
    fprintf('--- Resampling to %.2f Hz ---\n', cfg.resample_hz);
    EEG = pop_resample(EEG, cfg.resample_hz);
end

% 从连续刺激日志中提取有效刺激 block，并为每个 block 生成规则的 1 Hz 脉冲事件。
all_blocks = extract_ccep_blocks(EEG.event, cfg.stop_event_label, EEG.srate);
valid_blocks = all_blocks([all_blocks.is_valid]);
failed_blocks = all_blocks(~[all_blocks.is_valid]);
if isempty(valid_blocks)
    error('No CCEP start triggers were found after excluding "%s".', cfg.stop_event_label);
end

synthetic_events = build_synthetic_events(valid_blocks, EEG.srate, cfg.stim_frequency_hz, cfg.stim_duration_sec);
required_window = [min(cfg.erp_epoch_window_sec(1), cfg.tfa_epoch_window_sec(1)), ...
    max(cfg.erp_epoch_window_sec(2), cfg.tfa_epoch_window_sec(2))];
synthetic_events = keep_events_in_bounds(synthetic_events, EEG.pnts, EEG.srate, required_window);
if isempty(synthetic_events)
    error('All synthetic CCEP events were removed by boundary checks.');
end

% 记录 block 级摘要，便于后续追踪每个刺激 block 生成了多少个脉冲。
block_summary = attach_block_pulse_counts(all_blocks, synthetic_events);
write_block_summary_tsv(fullfile(cfg.output_dir, 'ccep_block_summary.tsv'), block_summary);

EEG_common = EEG;
EEG_common.etc.ccep_original_event = EEG.event;
EEG_common.etc.ccep_block_summary = block_summary;
EEG_common.etc.ccep_valid_blocks = valid_blocks;
EEG_common.etc.ccep_failed_blocks = failed_blocks;
EEG_common.event = synthetic_events;
EEG_common.urevent = synthetic_events;
EEG_common.epoch = [];
EEG_common = eeg_checkset(EEG_common, 'eventconsistency');

fprintf('Detected %d valid stimulation blocks, %d failed blocks, and generated %d synthetic pulses.\n', ...
    numel(valid_blocks), numel(failed_blocks), numel(synthetic_events));

% 通用预处理先做陷波、公共带通和局部参考，后面 ERP/TFA 分支共用这一步的结果。
fprintf('--- Applying notch filters ---\n');
for center_hz = cfg.notch_centers_hz
    EEG_common = pop_eegfiltnew(EEG_common, 'locutoff', center_hz - 1, ...
        'hicutoff', center_hz + 1, 'revfilt', 1);
end

fprintf('--- Applying common high-pass %.1f Hz ---\n', cfg.common_highpass_hz);
EEG_common = pop_eegfiltnew(EEG_common, 'locutoff', cfg.common_highpass_hz);

fprintf('--- Applying common low-pass %.1f Hz ---\n', cfg.common_lowpass_hz);
EEG_common = pop_eegfiltnew(EEG_common, 'hicutoff', cfg.common_lowpass_hz);

fprintf('--- Applying local SEEG reference ---\n');
EEG_common.data = apply_local_reference(EEG_common.data, EEG_common.chanlocs);

site_labels = stable_unique_labels({valid_blocks.site_label});

% ERP 分支使用较低频段，直接导出 EEGLAB set 和后续 MATLAB 可读的结构体。
fprintf('--- Building ERP epochs ---\n');
EEG_ERP = pop_eegfiltnew(EEG_common, 'locutoff', cfg.erp_band_hz(1), 'hicutoff', cfg.erp_band_hz(2));
EEG_ERP = pop_epoch(EEG_ERP, site_labels, cfg.erp_epoch_window_sec, 'newname', 'CCEP_ERP_epochs', 'epochinfo', 'yes');
EEG_ERP = pop_rmbase(EEG_ERP, cfg.baseline_ms);
assert_trial_count(EEG_ERP, synthetic_events, 'ERP');
pop_saveset(EEG_ERP, 'filename', 'processed_ERP.set', 'filepath', cfg.output_dir);

% TFA 分支保留更宽的频率范围，便于后续做时频响应统计。
fprintf('--- Building TFA epochs ---\n');
EEG_TFA = pop_eegfiltnew(EEG_common, 'locutoff', cfg.tfa_band_hz(1), 'hicutoff', cfg.tfa_band_hz(2));
EEG_TFA = pop_epoch(EEG_TFA, site_labels, cfg.tfa_epoch_window_sec, 'newname', 'CCEP_TFA_epochs', 'epochinfo', 'yes');
EEG_TFA = pop_rmbase(EEG_TFA, cfg.baseline_ms);
assert_trial_count(EEG_TFA, synthetic_events, 'TFA');
pop_saveset(EEG_TFA, 'filename', 'processed_TFA.set', 'filepath', cfg.output_dir);

erp_epoch = build_epoch_export(EEG_ERP, synthetic_events, block_summary, valid_blocks, failed_blocks, site_labels, 'ccep_erp', cfg.erp_epoch_window_sec, cfg.baseline_ms);
tfa_epoch = build_epoch_export(EEG_TFA, synthetic_events, block_summary, valid_blocks, failed_blocks, site_labels, 'ccep_tfa', cfg.tfa_epoch_window_sec, cfg.baseline_ms);

save(fullfile(cfg.output_dir, 'ccep_ERP_epoched.mat'), 'erp_epoch', '-v7.3');
save(fullfile(cfg.output_dir, 'ccep_TFA_epoched.mat'), 'tfa_epoch', '-v7.3');

fprintf('Saved CCEP outputs to %s\n', cfg.output_dir);
end

function ensure_eeglab_on_path(eeglab_root)
% 优先尝试打包目录旁边的 external_tools/eeglab，
% 如果不存在，再退回到当前 MATLAB 环境里已经配置好的 EEGLAB。
if exist(eeglab_root, 'dir')
    addpath(genpath(eeglab_root));
end

eeglab_file = which('eeglab');
if isempty(eeglab_file)
    error('EEGLAB is not on the MATLAB path.');
end

eeglab_root = fileparts(eeglab_file);
addpath(genpath(eeglab_root));
end

function blocks = extract_ccep_blocks(events, stop_event_label, srate)
% 把原始连续事件流整理成刺激 block 列表。
% 如果发现连续两个相同开始事件，中间没有“刺激停止”，则把前一个标记为失败 block。
blocks = struct('candidate_block_index', {}, 'block_index', {}, 'site_label', {}, 'source_event_index', {}, ...
    'start_latency', {}, 'start_time_sec', {}, 'stop_latency', {}, ...
    'stop_time_sec', {}, 'has_stop_event', {}, 'observed_duration_sec', {}, ...
    'is_valid', {}, 'block_status', {}, 'exclude_reason', {}, ...
    'superseded_by_event_index', {}, 'superseded_by_latency', {});

candidate_block_index = 0;
for i = 1:numel(events)
    event_label = event_type_to_char(events(i).type);
    if strcmp(event_label, stop_event_label)
        continue;
    end

    candidate_block_index = candidate_block_index + 1;
    stop_latency = NaN;
    stop_time_sec = NaN;
    observed_duration_sec = NaN;
    has_stop_event = false;
    is_valid = true;
    block_status = 'valid';
    exclude_reason = '';
    superseded_by_event_index = NaN;
    superseded_by_latency = NaN;

    if i < numel(events)
        next_label = event_type_to_char(events(i + 1).type);
        if strcmp(next_label, stop_event_label)
            stop_latency = double(events(i + 1).latency);
            stop_time_sec = (stop_latency - 1) / srate;
            observed_duration_sec = (stop_latency - double(events(i).latency)) / srate;
            has_stop_event = true;
        elseif strcmp(next_label, event_label)
            is_valid = false;
            block_status = 'failed_duplicate';
            exclude_reason = 'consecutive_duplicate_superseded_by_later_trigger';
            superseded_by_event_index = i + 1;
            superseded_by_latency = double(events(i + 1).latency);
        end
    end

    start_latency = double(events(i).latency);
    blocks(candidate_block_index).candidate_block_index = candidate_block_index;
    blocks(candidate_block_index).block_index = NaN;
    blocks(candidate_block_index).site_label = event_label;
    blocks(candidate_block_index).source_event_index = i;
    blocks(candidate_block_index).start_latency = start_latency;
    blocks(candidate_block_index).start_time_sec = (start_latency - 1) / srate;
    blocks(candidate_block_index).stop_latency = stop_latency;
    blocks(candidate_block_index).stop_time_sec = stop_time_sec;
    blocks(candidate_block_index).has_stop_event = has_stop_event;
    blocks(candidate_block_index).observed_duration_sec = observed_duration_sec;
    blocks(candidate_block_index).is_valid = is_valid;
    blocks(candidate_block_index).block_status = block_status;
    blocks(candidate_block_index).exclude_reason = exclude_reason;
    blocks(candidate_block_index).superseded_by_event_index = superseded_by_event_index;
    blocks(candidate_block_index).superseded_by_latency = superseded_by_latency;
end

valid_block_index = 0;
for i = 1:numel(blocks)
    if blocks(i).is_valid
        valid_block_index = valid_block_index + 1;
        blocks(i).block_index = valid_block_index;
    end
end
end

function synthetic_events = build_synthetic_events(blocks, srate, stim_frequency_hz, stim_duration_sec)
% 按固定刺激频率把每个有效 block 展开成一串规则脉冲事件。
step_samples = srate / stim_frequency_hz;
if abs(step_samples - round(step_samples)) > 1e-6
    error('Sampling rate %.6f is not divisible by stimulation frequency %.6f.', srate, stim_frequency_hz);
end

step_samples = round(step_samples);
num_pulses = stim_duration_sec * stim_frequency_hz;
synthetic_events = repmat(struct( ...
    'type', '', ...
    'latency', 0, ...
    'block_index', 0, ...
    'pulse_index', 0, ...
    'site_label', '', ...
    'source_start_latency', 0), 1, numel(blocks) * num_pulses);

cursor = 1;
for i = 1:numel(blocks)
    for pulse_index = 1:num_pulses
        synthetic_events(cursor).type = blocks(i).site_label;
        synthetic_events(cursor).latency = blocks(i).start_latency + (pulse_index - 1) * step_samples;
        synthetic_events(cursor).block_index = blocks(i).block_index;
        synthetic_events(cursor).pulse_index = pulse_index;
        synthetic_events(cursor).site_label = blocks(i).site_label;
        synthetic_events(cursor).source_start_latency = blocks(i).start_latency;
        cursor = cursor + 1;
    end
end
end

function synthetic_events = keep_events_in_bounds(synthetic_events, total_points, srate, epoch_window_sec)
% 移除那些会导致 epoch 越界的脉冲事件。
pre_samples = ceil(abs(min(0, epoch_window_sec(1))) * srate);
post_samples = ceil(max(0, epoch_window_sec(2)) * srate);
keep_mask = true(1, numel(synthetic_events));

for i = 1:numel(synthetic_events)
    latency = double(synthetic_events(i).latency);
    keep_mask(i) = (latency - pre_samples >= 1) && (latency + post_samples <= total_points);
end

removed_count = sum(~keep_mask);
if removed_count > 0
    warning('Removed %d synthetic events that would exceed epoch boundaries.', removed_count);
end

synthetic_events = synthetic_events(keep_mask);
end

function blocks = attach_block_pulse_counts(blocks, synthetic_events)
% 给每个 block 附加最终保留下来的脉冲个数和首末脉冲位置。
for i = 1:numel(blocks)
    if blocks(i).is_valid
        block_mask = [synthetic_events.block_index] == blocks(i).block_index;
    else
        block_mask = false(1, numel(synthetic_events));
    end
    pulse_latencies = [synthetic_events(block_mask).latency];
    blocks(i).kept_pulse_count = sum(block_mask);
    if isempty(pulse_latencies)
        blocks(i).first_pulse_latency = NaN;
        blocks(i).last_pulse_latency = NaN;
    else
        blocks(i).first_pulse_latency = pulse_latencies(1);
        blocks(i).last_pulse_latency = pulse_latencies(end);
    end
end
end

function write_block_summary_tsv(output_path, blocks)
% 输出 block 摘要表，便于后续人工核查刺激块质量。
fid = fopen(output_path, 'w');
if fid < 0
    error('Unable to open block summary file for writing: %s', output_path);
end

cleanup_obj = onCleanup(@() fclose(fid));
fprintf(fid, 'candidate_block_index\tblock_index\tsite_label\tstart_latency\tstart_time_sec\thas_stop_event\tstop_latency\tobserved_duration_sec\tis_valid\tblock_status\texclude_reason\tsuperseded_by_event_index\tsuperseded_by_latency\tkept_pulse_count\n');
for i = 1:numel(blocks)
    fprintf(fid, '%d\t%.0f\t%s\t%.0f\t%.6f\t%d\t%.0f\t%.6f\t%d\t%s\t%s\t%.0f\t%.0f\t%d\n', ...
        blocks(i).candidate_block_index, blocks(i).block_index, blocks(i).site_label, ...
        blocks(i).start_latency, blocks(i).start_time_sec, blocks(i).has_stop_event, ...
        blocks(i).stop_latency, blocks(i).observed_duration_sec, blocks(i).is_valid, ...
        blocks(i).block_status, blocks(i).exclude_reason, ...
        blocks(i).superseded_by_event_index, blocks(i).superseded_by_latency, ...
        blocks(i).kept_pulse_count);
end
clear cleanup_obj;
end

function reref_data = apply_local_reference(data, chanlocs)
% 针对同一根电极轴做局部参考：优先使用上下两个相邻触点的均值。
chan_labels = cell(1, numel(chanlocs));
for i = 1:numel(chanlocs)
    chan_labels{i} = chanlocs(i).labels;
end

num_chans = numel(chan_labels);
shaft_names = cell(1, num_chans);
contact_nums = nan(1, num_chans);

for i = 1:num_chans
    label = chan_labels{i};
    first_digit_idx = find(isstrprop(label, 'digit'), 1);
    if ~isempty(first_digit_idx)
        shaft_names{i} = label(1:first_digit_idx - 1);
        contact_nums(i) = str2double(label(first_digit_idx:end));
    else
        shaft_names{i} = label;
    end
end

unique_shafts = stable_unique_labels(shaft_names(~isnan(contact_nums)));
temp_data = data;
local_ref_data = zeros(size(temp_data), 'like', temp_data);

for s = 1:numel(unique_shafts)
    curr_shaft = unique_shafts{s};
    shaft_idx = find(strcmp(shaft_names, curr_shaft));

    for c = 1:numel(shaft_idx)
        curr_ch_idx = shaft_idx(c);
        curr_num = contact_nums(curr_ch_idx);
        neighbor_idx = [];

        prev_idx = shaft_idx(contact_nums(shaft_idx) == curr_num - 1);
        if ~isempty(prev_idx)
            neighbor_idx(end + 1) = prev_idx;
        end

        next_idx = shaft_idx(contact_nums(shaft_idx) == curr_num + 1);
        if ~isempty(next_idx)
            neighbor_idx(end + 1) = next_idx;
        end

        if numel(neighbor_idx) == 2
            local_ref_data(curr_ch_idx, :) = temp_data(curr_ch_idx, :) - mean(temp_data(neighbor_idx, :), 1);
        elseif numel(neighbor_idx) == 1
            local_ref_data(curr_ch_idx, :) = temp_data(curr_ch_idx, :) - temp_data(neighbor_idx, :);
        else
            local_ref_data(curr_ch_idx, :) = temp_data(curr_ch_idx, :);
        end
    end
end

non_seeg_idx = find(isnan(contact_nums));
if ~isempty(non_seeg_idx)
    local_ref_data(non_seeg_idx, :) = temp_data(non_seeg_idx, :);
end

reref_data = local_ref_data;
end

function epoch_struct = build_epoch_export(EEG_epoched, synthetic_events, block_summary, valid_blocks, failed_blocks, site_labels, name, epoch_window_sec, baseline_ms)
% 把 EEGLAB epoch 对象转换成后续脚本更容易消费的 MATLAB 结构体。
epoch_struct = struct();
epoch_struct.name = name;
epoch_struct.trigger = site_labels;
epoch_struct.ch = EEG_epoched.chanlocs;
epoch_struct.time_ms = EEG_epoched.times;
epoch_struct.srate = EEG_epoched.srate;
epoch_struct.epoch_window_sec = epoch_window_sec;
epoch_struct.baseline_ms = baseline_ms;
epoch_struct.block = block_summary;
epoch_struct.valid_block = valid_blocks;
epoch_struct.failed_block = failed_blocks;
epoch_struct.valid_block_count = numel(valid_blocks);
epoch_struct.failed_block_count = numel(failed_blocks);
epoch_struct.synthetic_event_count = numel(synthetic_events);
epoch_struct.trial_count_per_site = zeros(numel(site_labels), 1);
epoch_struct.data = cell(numel(site_labels), 1);
epoch_struct.site_trial_block_index = cell(numel(site_labels), 1);
epoch_struct.site_trial_pulse_index = cell(numel(site_labels), 1);
epoch_struct.site_trial_global_epoch_index = cell(numel(site_labels), 1);
epoch_struct.site_trial_source_start_latency = cell(numel(site_labels), 1);
epoch_struct.all_trial_site_label = {synthetic_events.site_label};
epoch_struct.all_trial_block_index = [synthetic_events.block_index];
epoch_struct.all_trial_pulse_index = [synthetic_events.pulse_index];
epoch_struct.all_trial_source_start_latency = [synthetic_events.source_start_latency];

for i = 1:numel(site_labels)
    site_mask = strcmp(site_labels{i}, epoch_struct.all_trial_site_label);
    selected_indices = find(site_mask);
    trial_data = EEG_epoched.data(:, :, selected_indices);

    epoch_struct.trial_count_per_site(i) = numel(selected_indices);
    epoch_struct.data{i} = permute(trial_data, [3, 1, 2]);
    epoch_struct.site_trial_block_index{i} = [synthetic_events(selected_indices).block_index];
    epoch_struct.site_trial_pulse_index{i} = [synthetic_events(selected_indices).pulse_index];
    epoch_struct.site_trial_global_epoch_index{i} = selected_indices;
    epoch_struct.site_trial_source_start_latency{i} = [synthetic_events(selected_indices).source_start_latency];
end
end

function assert_trial_count(EEG_epoched, synthetic_events, branch_name)
% 防止事件数与 epoch 数不一致，保证后续 site/block 对齐关系正确。
if EEG_epoched.trials ~= numel(synthetic_events)
    error('%s epochs (%d) do not match synthetic event count (%d).', ...
        branch_name, EEG_epoched.trials, numel(synthetic_events));
end
end

function labels = stable_unique_labels(label_list)
% 保持原始顺序地去重，避免 ROI 或刺激标签被排序打乱。
labels = {};
for i = 1:numel(label_list)
    current_label = label_list{i};
    if ~any(strcmp(current_label, labels))
        labels{end + 1} = current_label; %#ok<AGROW>
    end
end
end

function value = event_type_to_char(event_type)
% 把 EEGLAB 里可能出现的不同事件类型统一转成字符形式。
if ischar(event_type)
    value = event_type;
elseif isstring(event_type)
    value = char(event_type);
elseif isnumeric(event_type)
    value = num2str(event_type);
elseif iscell(event_type) && isscalar(event_type)
    value = event_type_to_char(event_type{1});
else
    error('Unsupported event type class: %s', class(event_type));
end
end