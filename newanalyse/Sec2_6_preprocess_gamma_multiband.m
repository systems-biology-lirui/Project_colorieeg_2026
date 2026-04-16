function Sec2_6_preprocess_gamma_multiband()
% PREPROCESS_GAMMA_MULTIBAND extracts multi-band gamma features from TFA input.
%
% Input:
%   processed_data/{subject}/task*_TFA_epoched.mat
%
% Output for each ROI:
%   gmb_task1 / gmb_task2 / gmb_task3: [Cond, Rep, Feature, Time]
%   gmb_roi_channels: ROI channel labels
%   gmb_band_names: band names
%   gmb_band_ranges: [n_bands, 2]
%   gmb_feature_labels: band-channel labels
%   gmb_feature_band_index / gmb_feature_channel_index
%   gmb_time_ms

run_timer = tic;

subject = 'test001';
cfg = newanalyse_load_run_config(mfilename, {'matlab_defaults', 'sec2_defaults'});
if isfield(cfg, 'subject')
    subject = char(string(cfg.subject));
end

paths = newanalyse_paths();
base_path = paths.base_path;
data_dir = fullfile(base_path, 'processed_data', subject);
save_dir = fullfile(paths.feature_root, 'gamma_multiband', subject);
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

loc_file = fullfile(data_dir, sprintf('%s_ieegloc.xlsx', subject));

fid = fopen(fullfile(save_dir, 'preprocess_gamma_multiband_log.txt'), 'w');
if fid == -1, error('Unable to create log file'); end

fprintf(fid, '=== Gamma multiband preprocessing start ===\n');
fprintf(fid, 'Subject: %s\n', subject);

try
    fprintf(fid, 'Step 1: Find common channels...\n');

    f1 = fullfile(data_dir, 'task1_TFA_epoched.mat');
    f2 = fullfile(data_dir, 'task2_TFA_epoched.mat');
    f3 = fullfile(data_dir, 'task3_TFA_epoched.mat');

    if ~isfile(f1), error('Task1 TFA data not found: %s', f1); end
    if ~isfile(f2), error('Task2 TFA data not found: %s', f2); end
    if ~isfile(f3), error('Task3 TFA data not found: %s', f3); end

    t1_info = load(f1, 'epoch'); ch1 = {t1_info.epoch.ch.labels}; clear t1_info;
    t2_info = load(f2, 'epoch'); ch2 = {t2_info.epoch.ch.labels}; clear t2_info;
    t3_info = load(f3, 'epoch'); ch3 = {t3_info.epoch.ch.labels}; clear t3_info;

    common_channels = intersect(ch1, intersect(ch2, ch3));
    fprintf(fid, 'Common channel count: %d\n', length(common_channels));
    if isempty(common_channels)
        error('No common channels across Task1/2/3');
    end

    fprintf(fid, 'Step 2: Parse ROI map...\n');
    if exist('get_roi_map', 'file') ~= 2
        addpath(paths.analysis_code_dir);
    end
    roi_map = get_roi_map(loc_file, common_channels);
    rois = keys(roi_map);
    fprintf(fid, 'ROI count: %d\n', length(rois));

    fprintf(fid, 'Step 3: Configure gamma bands...\n');
    fs = 500;
    time_idx = 201:750;
    smooth_win = 10;
    if isfield(cfg, 'smooth_win')
        smooth_win = double(cfg.smooth_win);
    end
    filter_order = 4;
    gamma_range = [30, 100];
    n_bands = 8;
    band_edges = logspace(log10(gamma_range(1)), log10(gamma_range(2)), n_bands + 1);
    gmb_band_ranges = zeros(n_bands, 2);
    gmb_band_names = cell(n_bands, 1);
    for b = 1:n_bands
        gmb_band_ranges(b, :) = [band_edges(b), band_edges(b + 1)];
        gmb_band_names{b} = sprintf('GammaBand%02d_%0.1f_%0.1fHz', b, band_edges(b), band_edges(b + 1));
    end
    fprintf(fid, 'Gamma range: %.1f-%.1f Hz | bands: %d\n', gamma_range(1), gamma_range(2), n_bands);

    tasks = {'task1', 'task2', 'task3'};
    skip_rois = {'Unknown', 'N_A'};

    for t = 1:length(tasks)
        task_name = tasks{t};
        fprintf(fid, 'Processing %s...\n', task_name);
        fprintf('Processing %s...\n', task_name);

        data_file = fullfile(data_dir, sprintf('%s_TFA_epoched.mat', task_name));
        loaded = load(data_file, 'epoch');
        epoch = loaded.epoch;
        all_channels = {epoch.ch.labels};

        for r = 1:length(rois)
            roi_name = rois{r};
            if any(strcmp(roi_name, skip_rois))
                fprintf(fid, '  Skip ROI: %s\n', roi_name);
                continue;
            end

            roi_chans = roi_map(roi_name);
            [~, ch_idxs] = ismember(roi_chans, all_channels);
            ch_idxs = ch_idxs(ch_idxs > 0);
            if isempty(ch_idxs), continue; end

            raw_roi = epoch.data(:, :, ch_idxs, :);
            [n_cond, n_rep, n_ch, n_time_total] = size(raw_roi);
            if n_time_total < max(time_idx)
                fprintf(fid, '  Warning: %s %s time dimension too short (%d)\n', task_name, roi_name, n_time_total);
                continue;
            end

            curr_time_idx = time_idx;
            n_time_keep = length(curr_time_idx);
            n_features = n_bands * n_ch;
            gmb_data = zeros(n_cond, n_rep, n_features, n_time_keep, 'single');
            gmb_feature_labels = cell(n_features, 1);
            gmb_feature_band_index = zeros(n_features, 1, 'uint16');
            gmb_feature_channel_index = zeros(n_features, 1, 'uint16');
            gmb_roi_channels = all_channels(ch_idxs);

            for ch = 1:n_ch
                channel_trials = reshape(raw_roi(:, :, ch, :), [n_cond, n_rep, n_time_total]);
                temp_channel = reshape(channel_trials, [], n_time_total)';

                for b = 1:n_bands
                    band_range = gmb_band_ranges(b, :);
                    [b_filt, a_filt] = butter(filter_order, band_range / (fs / 2), 'bandpass');

                    filtered_data = filtfilt(b_filt, a_filt, temp_channel);
                    analytic_signal = hilbert(filtered_data);
                    power_envelope = abs(analytic_signal).^2;
                    power_smooth = smoothdata(power_envelope, 1, 'gaussian', smooth_win);
                    power_cropped = power_smooth(curr_time_idx, :);

                    power_reshaped = reshape(power_cropped, [n_time_keep, n_cond, n_rep]);
                    band_data = permute(power_reshaped, [2, 3, 1]);

                    feat_idx = (b - 1) * n_ch + ch;
                    gmb_data(:, :, feat_idx, :) = reshape(single(band_data), [n_cond, n_rep, 1, n_time_keep]);

                    gmb_feature_labels{feat_idx} = sprintf('%s__%s', gmb_band_names{b}, gmb_roi_channels{ch});
                    gmb_feature_band_index(feat_idx) = b;
                    gmb_feature_channel_index(feat_idx) = ch;

                    clear b_filt a_filt filtered_data analytic_signal power_envelope power_smooth power_cropped power_reshaped band_data feat_idx;
                end

                clear channel_trials temp_channel;
            end

            gmb_time_ms = linspace(-100, 1000, n_time_keep);
            safe_roi = matlab.lang.makeValidName(roi_name);
            roi_file = fullfile(save_dir, sprintf('%s.mat', safe_roi));
            var_name = sprintf('gmb_%s', task_name);
            eval(sprintf('%s = gmb_data;', var_name));

            if exist(roi_file, 'file')
                save(roi_file, var_name, 'gmb_roi_channels', 'gmb_band_names', 'gmb_band_ranges', ...
                    'gmb_feature_labels', 'gmb_feature_band_index', 'gmb_feature_channel_index', ...
                    'gmb_time_ms', '-append');
            else
                save(roi_file, var_name, 'gmb_roi_channels', 'gmb_band_names', 'gmb_band_ranges', ...
                    'gmb_feature_labels', 'gmb_feature_band_index', 'gmb_feature_channel_index', ...
                    'gmb_time_ms');
            end

            clear raw_roi gmb_data gmb_feature_labels gmb_feature_band_index gmb_feature_channel_index gmb_roi_channels gmb_time_ms;
            clear n_cond n_rep n_ch n_time_total n_time_keep n_features curr_time_idx roi_file safe_roi var_name;
        end

        clear epoch loaded;
        fprintf(fid, '  Completed %s\n', task_name);
    end

    fprintf(fid, '\nGamma multiband preprocessing completed!\n');
    fprintf('Gamma multiband preprocessing completed!\n');

catch ME
    fprintf(fid, '\nError: %s\n', ME.message);
    fprintf(fid, 'Stack:\n');
    for k = 1:length(ME.stack)
        fprintf(fid, '  %s: line %d\n', ME.stack(k).name, ME.stack(k).line);
    end
    rethrow(ME);
end

fclose(fid);
fprintf('%s runtime: %.2f s\n', mfilename, toc(run_timer));
end
