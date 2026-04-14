function roi_map = get_roi_map(loc_file, channel_labels)
    % GET_ROI_MAP Parses the location file and maps channels to ROIs.
    % Based on logic from run_all_tasks.m

    roi_map = containers.Map();
    
    if ~isfile(loc_file)
        warning('Location file not found: %s', loc_file);
        return;
    end
    
    loc_table = readtable(loc_file);
    cols = loc_table.Properties.VariableNames;
    normalized_cols = regexprep(lower(cols), '[^a-z0-9]+', '');
    
    % Find Name Column
    name_idx = find(ismember(normalized_cols, {'name', 'channel', 'electrode', 'label'}), 1);
    if isempty(name_idx), name_idx = 1; end
    name_col = cols{name_idx};
    
    % Find ROI Column (Prioritize AAL3)
    roi_idx = find(ismember(normalized_cols, {'aal3', 'aal3mnilinear', 'aal3label', 'aal3mnisegment'}), 1);
    if isempty(roi_idx)
        roi_idx = find(startsWith(normalized_cols, 'aal3'), 1);
    end
    if isempty(roi_idx)
        roi_idx = find(ismember(normalized_cols, {'roi', 'region', 'anatomy', 'dklobe', 'lobe'}), 1);
    end
    if isempty(roi_idx)
        roi_idx = find(contains(normalized_cols, 'roi') | contains(normalized_cols, 'region') | contains(normalized_cols, 'anatomy'), 1);
    end
    
    if isempty(roi_idx)
        warning('Could not identify an ROI column.');
        return;
    end
    roi_col = cols{roi_idx};

    fprintf('Mapping Channels using Name: "%s" and ROI: "%s"\n', name_col, roi_col);
    
    % Build Map
    for i = 1:length(channel_labels)
        ch_name = channel_labels{i};
        
        % Find all occurrences in table
        row_idxs = find(strcmpi(loc_table.(name_col), ch_name));
        
        target_rois = {};
        if isempty(row_idxs)
            target_rois = {'Unknown'};
        else
            for r = 1:length(row_idxs)
                idx = row_idxs(r);
                val = loc_table.(roi_col)(idx);
                
                roi_name = '';
                if iscell(val), roi_name = val{1};
                elseif iscategorical(val), roi_name = char(val);
                elseif isnumeric(val), roi_name = num2str(val);
                else, roi_name = string(val);
                end
                
                % Sanitize ROI name
                roi_key = regexprep(char(roi_name), '[^a-zA-Z0-9_]', '_');
                roi_key = regexprep(roi_key, '^_+|_+$', '');
                if isempty(roi_key), roi_key = 'Unknown'; end
                
                target_rois{end+1} = roi_key;
            end
            % Unique ROIs for this channel to avoid duplicates
            target_rois = unique(target_rois);
        end
        
        % Add channel to all identified ROIs
        for k = 1:length(target_rois)
            r_key = target_rois{k};
            if ~isKey(roi_map, r_key)
                roi_map(r_key) = {};
            end
            roi_map(r_key) = [roi_map(r_key), {ch_name}];
        end
    end
end
