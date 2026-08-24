classdef IEEGDataAnalyzer
    properties
        EpochData      % Struct: epoch data loaded from .mat
        LocTable       % Table: electrode location info
        TimeVector     % Double: time vector
        ChannelLabels  % Cell: channel names
    end
    
    methods
        function obj = IEEGDataAnalyzer()
            % Constructor
        end
        
        function obj = loadData(obj, filepath)
            % loadData Loads .mat file containing 'epoch' struct
            %
            % Input:
            %   filepath: Full path to the .mat file
            
            if ~isfile(filepath)
                error('File not found: %s', filepath);
            end
            fprintf('Loading data from %s...\n', filepath);
            load(filepath, 'epoch');          

            obj.EpochData = epoch;
            
            obj.TimeVector = 1:size(obj.EpochData.data,4);

            
            % Extract Channel Labels
            if isfield(obj.EpochData, 'ch') && isfield(obj.EpochData.ch, 'labels')
                obj.ChannelLabels = {obj.EpochData.ch.labels};
            elseif isfield(obj.EpochData, 'labels')
                 obj.ChannelLabels = obj.EpochData.labels;
            else
                 warning('Channel labels not found in epoch.ch.label or epoch.label.');
            end
            
            fprintf('Data loaded successfully.\n');
        end
        
        function obj = loadLoc(obj, filepath)
            % loadLoc Loads electrode location Excel file
            %
            % Input:
            %   filepath: Full path to the .xlsx file
            
            if ~isfile(filepath)
                error('File not found: %s', filepath);
            end
            fprintf('Loading location info from %s...\n', filepath);
            obj.LocTable = readtable(filepath);
            fprintf('Location info loaded successfully.\n');
        end
        
        function plotERP(obj, cond1_idxs, cond2_idxs, stim_window, channels, save_prefix)
            % plotERP Plots ERP comparison using doubleline_with_shades
            %
            % Inputs:
            %   cond1_idxs: Indices of conditions to merge for Group 1
            %   cond2_idxs: Indices of conditions to merge for Group 2
            %   stim_window: [start, end] time for stimulus
            %   channels: (Optional) Cell array of channel names or indices to average.
            %             If empty/omitted, averages across ALL channels.
            %   save_prefix: (Optional) String to prepend to saved filenames.
            
            if nargin < 6
                save_prefix = '';
            end

            if nargin < 5
                channels = [];
            end
            
            % 1. Extract and Merge Data
            [data_c1, data_c2] = obj.extractMergedData(cond1_idxs, cond2_idxs);
            % data_c1/c2: [TotalRepeats x nCh x nTime]
            
            % 2. Select Channels
            if ~isempty(channels)
                if iscell(channels)
                    % Find indices
                    [~, ch_idxs] = ismember(channels, obj.ChannelLabels);
                    ch_idxs = ch_idxs(ch_idxs > 0);
                    if isempty(ch_idxs)
                        error('No matching channels found.');
                    end
                else
                    ch_idxs = channels;
                end
                data_c1 = data_c1(:, ch_idxs, :);
                data_c2 = data_c2(:, ch_idxs, :);
            end
            
            % 3. Average across Channels (to get [Repeats x Time])
            % mean over dimension 2 (Channels)
            mean_c1 = squeeze(mean(data_c1, 2)); % [Repeats x Time]
            mean_c2 = squeeze(mean(data_c2, 2)); % [Repeats x Time]
            
            % 4. Prepare for doubleline_with_shades
            % Input must be 2 x nRepeat x nTime
            % Note: nRepeat might differ between conditions. 
            % doubleline_with_shades assumes matrix input, so nRepeat must be same?
            % Let's check the function. It uses `reshape(data(1,:,:))` and assumes `dims(2)` is nRepeat.
            % It implies equal repeats.
            % If repeats differ, we can't put them in a single matrix easily without NaN padding.
            % However, the user's function `erp1` / `doubleline_with_shades` enforces `dims(2)` as nRepeat.
            % So it strictly requires equal repeats.
            % If repeats differ, we should truncate to the minimum or error.
            % Let's truncate to minimum for now to be safe.
            
            n_rep1 = size(mean_c1, 1);
            n_rep2 = size(mean_c2, 1);
            min_rep = min(n_rep1, n_rep2);
            
            if n_rep1 ~= n_rep2
                warning('Number of repeats differ (%d vs %d). Truncating to %d.', n_rep1, n_rep2, min_rep);
                mean_c1 = mean_c1(1:min_rep, :);
                mean_c2 = mean_c2(1:min_rep, :);
            end
            
            plot_data = zeros(2, min_rep, length(obj.TimeVector));
            plot_data(1, :, :) = mean_c1;
            plot_data(2, :, :) = mean_c2;
            
            
            IeegPlot.doubleline_with_shades(plot_data, obj.TimeVector, stim_window);
            xline(500,'Color','k','LineStyle','--');
            
            % Save Figure if prefix provided
            if ~isempty(save_prefix)
                filename = sprintf('%s_ERP.png', save_prefix);
                fprintf('Saving figure to %s\n', filename);
                saveas(gcf, filename);
            end
        end
        
        function plotHeatmap(obj, cond1_idxs, cond2_idxs, baseline_window, clim_range, save_prefix)
            % plotHeatmap Plots Difference Heatmap using plot_erp_heatmap
            %
            % Inputs:
            %   cond1_idxs: Indices of conditions to merge for Group 1
            %   cond2_idxs: Indices of conditions to merge for Group 2
            %   baseline_window: [start, end] for Z-score
            %   clim_range: (Optional) [min, max]
            %   save_prefix: (Optional) String to prepend to saved filenames.
            
            if nargin < 6
                save_prefix = '';
            end

            if nargin < 5
                clim_range = [];
            end
            
            % 1. Extract and Merge Data
            [data_c1, data_c2] = obj.extractMergedData(cond1_idxs, cond2_idxs);
            % data_c1/c2: [TotalRepeats x nCh x nTime]
            
            % 2. Handle unequal repeats
            n_rep1 = size(data_c1, 1);
            n_rep2 = size(data_c2, 1);
            min_rep = min(n_rep1, n_rep2);
             if n_rep1 ~= n_rep2
                warning('Number of repeats differ (%d vs %d). Truncating to %d.', n_rep1, n_rep2, min_rep);
                data_c1 = data_c1(1:min_rep, :, :);
                data_c2 = data_c2(1:min_rep, :, :);
            end
            
            % 3. Combine to 4D: [2 x nRepeat x nCh x nTime]
            combined_data = zeros(2, min_rep, size(data_c1, 2), size(data_c1, 3));
            combined_data(1, :, :, :) = data_c1;
            combined_data(2, :, :, :) = data_c2;
            
            % 4. Call Plotting Function
            
            IeegPlot.plot_erp_heatmap(combined_data, obj.TimeVector, baseline_window, clim_range, obj.ChannelLabels, obj.LocTable);
            
            % Add custom xticks
            time_labels = cellstr(string(-500:100:1000));
            fig = gcf;
            axes_handles = findobj(fig, 'Type', 'axes');
            for i = 1:length(axes_handles)
                set(fig, 'CurrentAxes', axes_handles(i));
                xticks(0:100:1500); % Adjust these values based on actual TimeVector mapping if needed
                xticklabels(time_labels);
                xline(500,'Color','k','LineStyle','--');
            end

            % Save Figure if prefix provided
            if ~isempty(save_prefix)
                filename = sprintf('%s_Heatmap.png', save_prefix);
                fprintf('Saving figure to %s\n', filename);
                saveas(fig, filename);
            end
        end
        
        function plotROIAnalysis(obj, cond1_idxs, cond2_idxs, stim_window, baseline_window, save_prefix)
            % plotROIAnalysis Plots Heatmap and ERP for each ROI separately
            %
            % Inputs:
            %   cond1_idxs, cond2_idxs: Condition indices
            %   stim_window: [start, end] for ERP shading
            %   baseline_window: [start, end] for Z-score in Heatmap
            %   save_prefix: (Optional) String to prepend to saved filenames.
            %                If provided, figures will be saved as PNG.
            
            if nargin < 6
                save_prefix = '';
            end
            
            if isempty(obj.LocTable)
                error('Location table not loaded. Call loadLoc first.');
            end
            
            % 1. Identify Unique ROIs

            unique_rois = unique(obj.LocTable.AAL3_MNI_linear_);
            
            % 2. Extract Data
            [data_c1, data_c2] = obj.extractMergedData(cond1_idxs, cond2_idxs);
            % [Repeats x nCh x nTime]
            
            % Handle unequal repeats for Heatmap (requires 4D)
            n_rep1 = size(data_c1, 1);
            n_rep2 = size(data_c2, 1);
            min_rep = min(n_rep1, n_rep2);
            
            data_c1_trunc = data_c1(1:min_rep, :, :);
            data_c2_trunc = data_c2(1:min_rep, :, :);
            
            combined_data = zeros(2, min_rep, size(data_c1, 2), size(data_c1, 3));
            combined_data(1, :, :, :) = data_c1_trunc;
            combined_data(2, :, :, :) = data_c2_trunc;

            % 3. Iterate ROIs
            for i = 1:length(unique_rois)
                roi_name = char(unique_rois{i});
                fprintf('Plotting ROI: %s\n', roi_name);
                
                % Find channels for this ROI
                % Match obj.ChannelLabels with obj.LocTable.all3 where AAL3_MNI_linear_ == roi_name
                roi_mask = strcmp(obj.LocTable.AAL3_MNI_linear_, roi_name);
                roi_channels_in_table = obj.LocTable.Channel(roi_mask);
                
                % Find indices in data
                [found, ch_idxs] = ismember(roi_channels_in_table, obj.ChannelLabels);
                ch_idxs = ch_idxs(found); % Keep valid indices
                current_labels = obj.ChannelLabels(ch_idxs);
                
                if isempty(ch_idxs)
                    warning('No channels found for ROI: %s', roi_name);
                    continue;
                end
                
                % --- Figure Setup ---
                fig = figure('Name', ['ROI: ' roi_name], 'Color', 'w', 'Position', [100, 100, 1000, 800]);
                
                % 1. Heatmap (Top)
                % Extract ROI data
                roi_combined = combined_data(:, :, ch_idxs, :);
                
                % Call plot_erp_heatmap (Reusing existing function logic but need to adapt to subplot)
                % plot_erp_heatmap creates a new figure by default. 
                % We can either modify it to accept parent or copy logic.
                % Since we can't easily modify the external function without changing its signature significantly,
                % let's call it and then copy objects or just rely on separate figures?
                % User asked for "一张时空热图...以及一个通道平均之后的erp图" (One heatmap ... and one ERP).
                % Ideally in one figure.
                
                % Let's create a custom subplot layout here using the logic from plot_erp_heatmap
                % but localized for this ROI.
                
                % -- Heatmap (Difference) --
                subplot(2, 1, 1);
                % Calculate Difference (Raw)
                % Use reshape instead of squeeze to handle single-channel ROIs correctly
                % roi_combined: [2, Repeats, nCh, nTime]
                m1_raw = mean(roi_combined(1, :, :, :), 2); % [1, 1, nCh, nTime]
                m2_raw = mean(roi_combined(2, :, :, :), 2); % [1, 1, nCh, nTime]
                
                n_ch_roi = length(ch_idxs);
                n_time = length(obj.TimeVector);
                
                m1 = reshape(m1_raw, [n_ch_roi, n_time]);
                m2 = reshape(m2_raw, [n_ch_roi, n_time]);
                
                diff_map = m1 - m2;
                
                % T-test for Heatmap Mask
                % Replicate t-test logic from plot_erp_heatmap
                r1_flat = reshape(roi_combined(1, :, :, :), [min_rep, length(ch_idxs)*length(obj.TimeVector)]);
                r2_flat = reshape(roi_combined(2, :, :, :), [min_rep, length(ch_idxs)*length(obj.TimeVector)]);
                [h_map, ~] = ttest2(r1_flat, r2_flat, 'Alpha', 0.05);
                
                % Exclude first 500 points
                if length(obj.TimeVector) > 500
                    h_map_reshaped = reshape(h_map, [length(ch_idxs), length(obj.TimeVector)]);
                    h_map_reshaped(:, 1:500) = 0;
                    h_map = h_map_reshaped(:); % Flatten back if needed or use reshaped
                else
                    h_map_reshaped = reshape(h_map, [length(ch_idxs), length(obj.TimeVector)]);
                end
                
                imagesc(obj.TimeVector, 1:length(ch_idxs), diff_map);
                colormap('jet'); colorbar;
                set(gca, 'YDir', 'reverse');
                set(gca, 'YTick', 1:length(ch_idxs), 'YTickLabel', current_labels);
                title(['ROI: ' roi_name ' - Difference Heatmap']);
                xlabel('Time (ms)');
                
                % Plot Significance Dots on Heatmap
                hold on;
                [r_sig, c_sig] = find(h_map_reshaped);
                if ~isempty(r_sig)
                    plot(obj.TimeVector(c_sig), r_sig, 'k.', 'MarkerSize', 4);
                end
                hold off;
                
                % Apply xticks logic
                time_labels = cellstr(string(-500:100:1000));
                % Note: xticks(0:100:1500) assumes time indices or specific ms mapping.
                % If TimeVector is in ms (e.g., -500 to 1000), then xticks should match.
                % If TimeVector is indices, we need to map.
                % Assuming TimeVector is actual time in ms/sec.
                % The user snippet `xticks(0:100:1500)` suggests specific sampling points?
                % If TimeVector is 1:N, then 0:100:1500 are indices.
                % If TimeVector is -500:1000, then 0 is time 0.
                % I'll trust the user's snippet logic but apply it carefully.
                try
                    xticks(0:100:1500);
                    xticklabels(time_labels);
                catch
                end
                
                % -- ERP Plot (Bottom) --
                subplot(2, 1, 2);
                % Average across ROI channels first
                % data_c1: [Rep x nCh x nTime] -> [Rep x nTime]
                roi_c1_avg = squeeze(mean(data_c1(:, ch_idxs, :), 2)); 
                roi_c2_avg = squeeze(mean(data_c2(:, ch_idxs, :), 2));
                
                % Prepare for doubleline
                % Truncate repeats
                n_r1 = size(roi_c1_avg, 1);
                n_r2 = size(roi_c2_avg, 1);
                min_r = min(n_r1, n_r2);
                
                plot_erp_data = zeros(2, min_r, length(obj.TimeVector));
                plot_erp_data(1, :, :) = roi_c1_avg(1:min_r, :);
                plot_erp_data(2, :, :) = roi_c2_avg(1:min_r, :);
                
                % We need to use doubleline_with_shades but it creates a new figure.
                % We want it in subplot.
                % If doubleline_with_shades calls `figure;`, we can't put it in subplot easily.
                % Strategy: Temporarily override `figure` or copy code.
                % Or, simpler: Just plot standard ERP here manually since we have the data.
                
                % Calculate Mean and SEM
                mu1 = mean(roi_c1_avg, 1); sem1 = std(roi_c1_avg, 0, 1)/sqrt(n_r1);
                mu2 = mean(roi_c2_avg, 1); sem2 = std(roi_c2_avg, 0, 1)/sqrt(n_r2);
                
                % T-test for ERP
                [h_erp, ~] = ttest2(roi_c1_avg, roi_c2_avg, 'Alpha', 0.05);
                if length(obj.TimeVector) > 500
                    h_erp(1:500) = 0;
                end
                
                hold on;
                % Shade
                fill([obj.TimeVector fliplr(obj.TimeVector)], [mu1+sem1 fliplr(mu1-sem1)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
                fill([obj.TimeVector fliplr(obj.TimeVector)], [mu2+sem2 fliplr(mu2-sem2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
                % Lines
                plot(obj.TimeVector, mu1, 'b', 'LineWidth', 2);
                plot(obj.TimeVector, mu2, 'r', 'LineWidth', 2);
                
                % Stim Window
                if ~isempty(stim_window)
                    yl = ylim;
                    patch([stim_window(1) stim_window(2) stim_window(2) stim_window(1)], [yl(1) yl(1) yl(2) yl(2)], 'y', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
                end
                
                % Significant dots
                sig_idx = find(h_erp);
                if ~isempty(sig_idx)
                    yl = ylim;
                    plot(obj.TimeVector(sig_idx), repmat(yl(1), size(sig_idx)), 'k.', 'MarkerSize', 5);
                end
                
                title(['ROI: ' roi_name ' - Average ERP']);
                grid on;
                
                % Apply xticks logic
                try
                    xticks(0:100:1500);
                    xticklabels(time_labels);
                catch
                end
                
                % Save Figure if prefix provided
                if ~isempty(save_prefix)
                    if strcmp(roi_name,'N/A')
                        filename = sprintf('%s_unknown.png', save_prefix);
                    else
                        filename = sprintf('%s_%s.png', save_prefix, roi_name);
                    end
                    fprintf('Saving figure to %s\n', filename);
                    saveas(fig, filename);
                end
            end
        end
    end
    
    methods (Access = private)
        function [merged_c1, merged_c2] = extractMergedData(obj, idxs1, idxs2)
            % Extracts and merges repeats for given condition indices
            % Returns: [TotalRepeats x nCh x nTime] matrices
            
            % Check dimensions
            [nCond, nRep, nCh, nTime] = size(obj.EpochData.data);
            
            % Helper to extract
            function out = get_merged(idxs)
                % Extract: [nSelectedConds x nRep x nCh x nTime]
                selected = obj.EpochData.data(idxs, :, :, :);
                % Permute to move Cond and Rep adjacent: [nSelectedConds x nRep x nCh x nTime]
                % We want to merge dim 1 and 2.
                % Reshape to [(nSelectedConds * nRep) x nCh x nTime]
                out = reshape(selected, [], nCh, nTime);
            end
            
            merged_c1 = get_merged(idxs1);
            merged_c2 = get_merged(idxs2);
        end
    end
end
