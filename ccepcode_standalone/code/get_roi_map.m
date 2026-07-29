function roi_map = get_roi_map(loc_file, channel_labels)
    % 根据定位表把电极名称映射到 ROI。
    % 这个版本是从主项目复制出来的独立副本，
    % 供 ccepcode_standalone/code 下的 MATLAB 脚本直接调用。

    roi_map = containers.Map();
    
    if ~isfile(loc_file)
        warning('Location file not found: %s', loc_file);
        return;
    end
    
    loc_table = readtable(loc_file);
    cols = loc_table.Properties.VariableNames;
    normalized_cols = regexprep(lower(cols), '[^a-z0-9]+', '');
    
    % 优先识别电极名称列。
    name_idx = find(ismember(normalized_cols, {'name', 'channel', 'electrode', 'label'}), 1);
    if isempty(name_idx), name_idx = 1; end
    name_col = cols{name_idx};
    
    % 优先识别 AAL3 列，如果没有，再退回到更泛化的 ROI/region/anatomy 列。
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
    
    % 逐个电极读取定位表，并允许一个电极同时映射到多个 ROI。
    for i = 1:length(channel_labels)
        ch_name = channel_labels{i};
        
        % 在表里查找该电极的所有匹配行。
        row_idxs = find(strcmpi(loc_table.(name_col), ch_name));
        
        target_rois = {};
        if isempty(row_idxs)
            target_rois = {'Unknown'};
        else
            for r = 1:length(row_idxs)
                idx = row_idxs(r);
                val = loc_table.(roi_col)(idx);
                
                if iscell(val), roi_name = val{1};
                elseif iscategorical(val), roi_name = char(val);
                elseif isnumeric(val), roi_name = num2str(val);
                else, roi_name = string(val);
                end
                
                % 规范化 ROI 名称，避免非法字符影响后续保存文件。
                roi_key = regexprep(char(roi_name), '[^a-zA-Z0-9_]', '_');
                roi_key = regexprep(roi_key, '^_+|_+$', '');
                if isempty(roi_key), roi_key = 'Unknown'; end
                
                target_rois{end+1} = roi_key;
            end
            % 对同一个电极的重复 ROI 名称去重。
            target_rois = unique(target_rois);
        end
        
        % 把当前电极加入它命中的每一个 ROI。
        for k = 1:length(target_rois)
            r_key = target_rois{k};
            if ~isKey(roi_map, r_key)
                roi_map(r_key) = {};
            end
            roi_map(r_key) = [roi_map(r_key), {ch_name}];
        end
    end
end
