%% ========================================================================
%  重新提取 60-150Hz Subband High Gamma
%% ========================================================================
clear; clc;
script_timer = tic;
addpath('newanalyse');
% addpath(genpath('python_libs')); % Removed legacy Linux python_libs

cfg = newanalyse_load_run_config('Sec1_preanalyse.m', {'matlab_defaults', 'sec1_defaults'});
subject = 'test001';
if isfield(cfg, 'subject'), subject = char(string(cfg.subject)); end

paths = newanalyse_paths();
project_root = paths.base_path;
raw_subject_dir = regexprep(subject, '^test0*', 'test');
if isfield(cfg, 'raw_subject_dir'), raw_subject_dir = char(string(cfg.raw_subject_dir)); end
raw_data_dir = fullfile(project_root, 'seegdata', raw_subject_dir);
if isfield(cfg, 'raw_data_dir'), raw_data_dir = char(string(cfg.raw_data_dir)); end

% Create new output directories
feature_dir = fullfile(project_root, 'color_cognition_pipeline', 'feature');
subband_dir = fullfile(feature_dir, 'subband_60_150');
% Subject-specific output matches the analyse_0617 Step0 convention.
subband_dir = fullfile(subband_dir, subject);
if ~exist(subband_dir, 'dir'), mkdir(subband_dir); end

task_ids = 1:3; % Task 1--3 required by analyse_0617
if isfield(cfg, 'tasks'), task_ids = double(cfg.tasks); end
for i = task_ids
    task_label = i;
    preprocess_analyse(task_label, raw_data_dir, subband_dir)
end

fprintf('\n%s runtime: %.2f s\n', mfilename, toc(script_timer));

function preprocess_analyse(task_label, raw_data_dir, subband_dir)
clc;
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
data_path = raw_data_dir;
file_name = sprintf('erp%d.set', task_label);

fprintf('--- 加载数据 %s ---\n', file_name);
EEG = pop_loadset('filename', file_name, 'filepath', data_path);
EEG = pop_resample(EEG, 500);

target_event = unique({EEG.event.type});
% Exclude invalid triggers
target_event = setdiff(target_event, {'Trigger-In:99', 'boundary'});

% 陷波 & 高通
EEG = pop_eegfiltnew(EEG, 'locutoff', 49, 'hicutoff', 51, 'revfilt', 1);
EEG = pop_eegfiltnew(EEG, 'locutoff', 99, 'hicutoff', 101, 'revfilt', 1);
EEG = pop_eegfiltnew(EEG, 'locutoff', 149, 'hicutoff', 151, 'revfilt', 1);
EEG = pop_eegfiltnew(EEG, 'locutoff', 1);

% SEEG 局部重参考
chan_labels = {EEG.chanlocs.labels};
num_chans = length(chan_labels);
shaft_names = cell(1, num_chans); contact_nums = nan(1, num_chans);
for i = 1:num_chans
    label = chan_labels{i};
    first_digit_idx = find(isstrprop(label, 'digit'), 1);
    if ~isempty(first_digit_idx)
        shaft_names{i} = label(1:first_digit_idx-1);
        contact_nums(i) = str2double(label(first_digit_idx:end));
    else
        shaft_names{i} = label;
    end
end
unique_shafts = unique(shaft_names(~isnan(contact_nums)));
temp_data = EEG.data; local_ref_data = zeros(size(temp_data));
for s = 1:length(unique_shafts)
    curr_shaft = unique_shafts{s};
    shaft_idx = find(strcmp(shaft_names, curr_shaft));
    for c = 1:length(shaft_idx)
        curr_ch_idx = shaft_idx(c); curr_num = contact_nums(curr_ch_idx);
        neighbor_idx = [];
        prev_idx = shaft_idx(contact_nums(shaft_idx) == curr_num - 1);
        if ~isempty(prev_idx), neighbor_idx(end+1) = prev_idx; end
        next_idx = shaft_idx(contact_nums(shaft_idx) == curr_num + 1);
        if ~isempty(next_idx), neighbor_idx(end+1) = next_idx; end
        
        if length(neighbor_idx) == 2
            local_ref_data(curr_ch_idx, :) = temp_data(curr_ch_idx, :) - mean(temp_data(neighbor_idx, :), 1);
        elseif length(neighbor_idx) == 1
            local_ref_data(curr_ch_idx, :) = temp_data(curr_ch_idx, :) - temp_data(neighbor_idx, :);
        else
            local_ref_data(curr_ch_idx, :) = temp_data(curr_ch_idx, :);
        end
    end
end
non_seeg_idx = find(isnan(contact_nums));
if ~isempty(non_seeg_idx), local_ref_data(non_seeg_idx, :) = temp_data(non_seeg_idx, :); end
EEG.data = local_ref_data;

% 200Hz 低通
EEG = pop_eegfiltnew(EEG, 'hicutoff', 200);
EEG_Ref = EEG;

%% Pipeline: Subband 60-150Hz
fprintf('\n>>> 开始执行 Subband 60-150Hz 管线 <<<\n');
EEG_Sub = EEG_Ref;
sub_bands = [60 70; 70 80; 80 90; 90 100; 100 110; 110 120; 120 130; 130 140; 140 150];
n_bands = size(sub_bands, 1);
hg_accum = zeros(size(EEG_Sub.data), 'double');
hg_filter_order = 4;
for b = 1:n_bands
    [b_filt, a_filt] = butter(hg_filter_order, sub_bands(b,:) / (EEG_Sub.srate/2), 'bandpass');
    for ch = 1:EEG_Sub.nbchan
        filtered = filtfilt(b_filt, a_filt, double(EEG_Sub.data(ch, :)));
        analytic = hilbert(filtered);
        hg_accum(ch, :) = hg_accum(ch, :) + sqrt(abs(analytic));
    end
end
EEG_Sub.data = single(hg_accum / n_bands);

% Epoch
EEG_Sub = pop_epoch(EEG_Sub, target_event, [-0.5 1.0], 'newname', 'Sub_epochs', 'epochinfo', 'yes');

% Z-score (-250~-50ms)
bl_pts = find(EEG_Sub.times >= -200 & EEG_Sub.times <= 0);
for ch = 1:EEG_Sub.nbchan
    for ep = 1:EEG_Sub.trials
        bl_seg = double(EEG_Sub.data(ch, bl_pts, ep));
        bl_mean = mean(bl_seg); bl_std = std(bl_seg);
        if bl_std > eps
            EEG_Sub.data(ch, :, ep) = single((double(EEG_Sub.data(ch, :, ep)) - bl_mean) / bl_std);
        else
            EEG_Sub.data(ch, :, ep) = single(double(EEG_Sub.data(ch, :, ep)) - bl_mean);
        end
    end
end

% Save Subband MAT
epoch = build_save_struct_cell(EEG_Sub, target_event, 'hg_subband60_150');
mat_filename = fullfile(subband_dir, sprintf('task%d_hg_subband.mat', task_label));
save(mat_filename, 'epoch', '-v7.3');
fprintf('成功保存 Subband HG: %s\n', mat_filename);

end

function epoch = build_save_struct_cell(EEG, target_event, name_prefix)
    epoch = struct();
    epoch.ch = EEG.chanlocs;
    epoch.name = name_prefix;
    epoch.time_ms = EEG.times;
    epoch.trigger = target_event;
    
    n_conds = length(target_event);
    epoch.data_cell = cell(1, n_conds);
    for i = 1:n_conds
        event_indices = find(strcmp(target_event{i}, {EEG.epoch.eventtype}));
        trial_data = EEG.data(:, :, event_indices);
        % permute to [Rep, Ch, Time]
        epoch.data_cell{i} = permute(trial_data, [3, 1, 2]);
    end
end
