%% ========================================================================
%  Extract 60-150Hz Subband High Gamma for test002 and test003
%% ========================================================================
clear; clc;
script_timer = tic;
addpath('newanalyse');
addpath(genpath('python_libs'));

subjects = {'test002', 'test003'};
for s = 1:length(subjects)
    subject = subjects{s};
    raw_subject_dir = regexprep(subject, '^test0*', 'test');
    raw_data_dir = fullfile('seegdata', raw_subject_dir);
    feature_dir = fullfile('color_cognition_pipeline', 'feature');
    subband_dir = fullfile(feature_dir, 'subband_60_150', subject);
    if ~exist(subband_dir, 'dir'), mkdir(subband_dir); end

    for task_label = [1, 2, 3]
        try
            preprocess_analyse(task_label, raw_data_dir, subband_dir);
        catch ME
            fprintf('Error processing Task %d for %s: %s\n', task_label, subject, ME.message);
        end
    end
end

fprintf('\nAll HG extraction runtime: %.2f s\n', toc(script_timer));
exit;

function preprocess_analyse(task_label, raw_data_dir, subband_dir)
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    file_name = sprintf('erp%d.set', task_label);
    if ~exist(fullfile(raw_data_dir, file_name), 'file')
        fprintf('File %s not found in %s\n', file_name, raw_data_dir);
        return;
    end
    fprintf('--- 加载数据 %s ---\n', file_name);
    EEG = pop_loadset('filename', file_name, 'filepath', raw_data_dir);
    EEG = pop_resample(EEG, 500);

    target_event = unique({EEG.event.type});
    target_event = setdiff(target_event, {'Trigger-In:99', 'boundary'});

    EEG = pop_eegfiltnew(EEG, 'locutoff', 49, 'hicutoff', 51, 'revfilt', 1);
    EEG = pop_eegfiltnew(EEG, 'locutoff', 99, 'hicutoff', 101, 'revfilt', 1);
    EEG = pop_eegfiltnew(EEG, 'locutoff', 149, 'hicutoff', 151, 'revfilt', 1);
    EEG = pop_eegfiltnew(EEG, 'locutoff', 1);

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

    EEG = pop_eegfiltnew(EEG, 'hicutoff', 200);
    EEG_Sub = EEG;
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

    EEG_Sub = pop_epoch(EEG_Sub, target_event, [-0.5 1.0], 'newname', 'Sub_epochs', 'epochinfo', 'yes');
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

    epoch = struct();
    epoch.ch = EEG_Sub.chanlocs;
    epoch.name = 'hg_subband60_150';
    epoch.time_ms = EEG_Sub.times;
    epoch.trigger = target_event;
    n_conds = length(target_event);
    epoch.data_cell = cell(1, n_conds);
    for i = 1:n_conds
        event_indices = find(strcmp(target_event{i}, {EEG_Sub.epoch.eventtype}));
        trial_data = EEG_Sub.data(:, :, event_indices);
        epoch.data_cell{i} = permute(trial_data, [3, 1, 2]);
    end

    mat_filename = fullfile(subband_dir, sprintf('task%d_hg_subband.mat', task_label));
    save(mat_filename, 'epoch', '-v7.3');
    fprintf('成功保存 Subband HG: %s\n', mat_filename);
end
