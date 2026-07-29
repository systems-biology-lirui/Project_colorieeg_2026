% CCEP ROI 特征整理脚本。
% 这个独立版本只依赖当前打包目录中的 epoched 数据、定位表和 get_roi_map.m。
function Sec2_ccep_preprocess_roi_features()
run_timer = tic;
cfg = build_default_cfg(mfilename('fullpath'));

process_modality(cfg, 'ERP', 'ccep_ERP_epoched.mat', 'erp_epoch', fullfile(cfg.feature_root, 'ccep_erp', cfg.subject));
process_modality(cfg, 'TFA', 'ccep_TFA_epoched.mat', 'tfa_epoch', fullfile(cfg.feature_root, 'ccep_tfa', cfg.subject));

fprintf('%s runtime: %.2f s\n', mfilename, toc(run_timer));
end

function cfg = build_default_cfg(script_path)
% 构建本地独立目录下的默认路径。
project_root = fileparts(fileparts(script_path));
code_dir = fileparts(script_path);

cfg = struct();
cfg.project_root = project_root;
cfg.code_dir = code_dir;
cfg.subject = 'test001';
cfg.processed_dir = fullfile(project_root, 'workspace', 'processed', cfg.subject, 'ccep');
cfg.feature_root = fullfile(project_root, 'workspace', 'feature');
cfg.loc_file = fullfile(project_root, 'data', 'metadata', cfg.subject, sprintf('%s_ieegloc.xlsx', cfg.subject));
cfg.roi_map_dir = code_dir;
end

function process_modality(cfg, modality_label, epoch_filename, epoch_var_name, save_dir)
% 把 site 维度的 CCEP epoch 数据转换成以 ROI 为单位的特征文件。
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

epoch_path = fullfile(cfg.processed_dir, epoch_filename);
if ~isfile(epoch_path)
    error('CCEP %s epoched file not found: %s', modality_label, epoch_path);
end
if ~isfile(cfg.loc_file)
    error('Location file not found: %s', cfg.loc_file);
end

ensure_roi_map_on_path(cfg.roi_map_dir);

loaded = load(epoch_path, epoch_var_name);
if ~isfield(loaded, epoch_var_name)
    error('Expected variable %s in %s.', epoch_var_name, epoch_path);
end
epoch = loaded.(epoch_var_name);

all_channel_labels = {epoch.ch.labels};
roi_map = get_roi_map(cfg.loc_file, all_channel_labels);
roi_names = sort(keys(roi_map));

[excluded_channels_by_site, excluded_masks_by_site] = build_site_exclusion_metadata(epoch.trigger, all_channel_labels);

fprintf('Processing CCEP %s ROI features for %s (%d ROIs).\n', modality_label, cfg.subject, numel(roi_names));
for roi_index = 1:numel(roi_names)
    roi_name = roi_names{roi_index};
    roi_channels = roi_map(roi_name);

    [is_member, ch_indices] = ismember(roi_channels, all_channel_labels);
    ch_indices = ch_indices(is_member);
    roi_channel_labels = all_channel_labels(ch_indices);
    if isempty(ch_indices)
        continue;
    end

    % 每个 ROI 文件都保留原始 trial、site、block 对应关系，
    % 后续 Python 统计脚本直接读取这些字段即可完成逐电极统计。
    roi_feature = struct();
    roi_feature.subject = cfg.subject;
    roi_feature.modality = lower(modality_label);
    roi_feature.roi_name = roi_name;
    roi_feature.site_labels = epoch.trigger;
    roi_feature.channel_labels = roi_channel_labels;
    roi_feature.time_ms = epoch.time_ms;
    roi_feature.srate = epoch.srate;
    roi_feature.epoch_window_sec = epoch.epoch_window_sec;
    roi_feature.baseline_ms = epoch.baseline_ms;
    roi_feature.trial_count_per_site = epoch.trial_count_per_site(:);
    roi_feature.valid_block = epoch.valid_block;
    roi_feature.failed_block = epoch.failed_block;
    roi_feature.block = epoch.block;
    roi_feature.site_data = cell(numel(epoch.trigger), 1);
    roi_feature.site_trial_block_index = epoch.site_trial_block_index;
    roi_feature.site_trial_pulse_index = epoch.site_trial_pulse_index;
    roi_feature.site_trial_global_epoch_index = epoch.site_trial_global_epoch_index;
    roi_feature.site_trial_source_start_latency = epoch.site_trial_source_start_latency;
    roi_feature.excluded_channels_by_site = cell(numel(epoch.trigger), 1);
    roi_feature.included_channels_by_site = cell(numel(epoch.trigger), 1);
    roi_feature.excluded_channel_mask_by_site = false(numel(epoch.trigger), numel(roi_channel_labels));
    roi_feature.global_excluded_channel_mask_by_site = excluded_masks_by_site(:, ch_indices);

    for site_index = 1:numel(epoch.trigger)
        roi_feature.site_data{site_index} = epoch.data{site_index}(:, ch_indices, :);

        % 对每个刺激 site 单独记录在该 ROI 内应该剔除和保留的电极。
        site_excluded_mask = excluded_masks_by_site(site_index, ch_indices);
        site_excluded_labels = roi_channel_labels(site_excluded_mask);
        site_included_labels = roi_channel_labels(~site_excluded_mask);

        roi_feature.excluded_channels_by_site{site_index} = site_excluded_labels;
        roi_feature.included_channels_by_site{site_index} = site_included_labels;
        roi_feature.excluded_channel_mask_by_site(site_index, :) = site_excluded_mask;
    end

    roi_feature.site_exclusion_source_labels = excluded_channels_by_site;

    roi_file = fullfile(save_dir, sprintf('%s.mat', matlab.lang.makeValidName(roi_name)));
    save(roi_file, 'roi_feature');
end
end

function ensure_roi_map_on_path(roi_map_dir)
% 确保打包目录内复制过来的 get_roi_map.m 可以被 MATLAB 找到。
if exist('get_roi_map', 'file') ~= 2
    addpath(roi_map_dir);
end
end

function [excluded_channels_by_site, excluded_masks_by_site] = build_site_exclusion_metadata(site_labels, all_channel_labels)
% 为每个刺激 site 预先构建“应剔除的刺激相关电极”掩码。
n_sites = numel(site_labels);
n_channels = numel(all_channel_labels);
excluded_channels_by_site = cell(n_sites, 1);
excluded_masks_by_site = false(n_sites, n_channels);

for site_index = 1:n_sites
    site_label = char(string(site_labels{site_index}));
    excluded_channels = derive_excluded_channels(site_label, all_channel_labels);
    excluded_channels_by_site{site_index} = excluded_channels;
    excluded_masks_by_site(site_index, :) = ismember(all_channel_labels, excluded_channels);
end
end

function excluded_channels = derive_excluded_channels(site_label, all_channel_labels)
% 对刺激标签里的每个触点，排除本触点及其相邻触点，避免刺激伪迹污染。
parts = strsplit(site_label, '-');
candidate_labels = {};

for part_index = 1:numel(parts)
    [shaft_name, contact_num, parsed_ok] = parse_contact_label(strtrim(parts{part_index}));
    if ~parsed_ok
        if any(strcmp(parts{part_index}, all_channel_labels))
            candidate_labels{end + 1} = parts{part_index}; %#ok<AGROW>
        end
        continue;
    end

    for offset = -1:1
        candidate_label = sprintf('%s%d', shaft_name, contact_num + offset);
        if any(strcmp(candidate_label, all_channel_labels))
            candidate_labels{end + 1} = candidate_label; %#ok<AGROW>
        end
    end
end

excluded_channels = stable_unique_labels(candidate_labels);
end

function [shaft_name, contact_num, parsed_ok] = parse_contact_label(channel_label)
% 把电极标签拆成轴名和触点编号，例如 A12 -> A 和 12。
shaft_name = '';
contact_num = NaN;
parsed_ok = false;

first_digit_idx = find(isstrprop(channel_label, 'digit'), 1);
if isempty(first_digit_idx)
    return;
end

shaft_name = channel_label(1:first_digit_idx - 1);
contact_num = str2double(channel_label(first_digit_idx:end));
parsed_ok = ~isempty(shaft_name) && ~isnan(contact_num);
end

function labels = stable_unique_labels(label_list)
% 按原顺序去重，避免后续索引和原数据顺序不一致。
labels = {};
for i = 1:numel(label_list)
    current_label = label_list{i};
    if ~any(strcmp(current_label, labels))
        labels{end + 1} = current_label; %#ok<AGROW>
    end
end
end