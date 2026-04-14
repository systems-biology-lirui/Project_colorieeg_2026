function cfg = newanalyse_load_run_config(script_name, sections)
cfg = struct();

if nargin < 2
    sections = {};
end
if ischar(sections) || isstring(sections)
    sections = cellstr(sections);
end

use_config = getenv('NEWANALYSE_USE_CONFIG');
if isempty(use_config) || ~is_truthy(use_config)
    return;
end

config_path = getenv('NEWANALYSE_CONFIG_PATH');
if isempty(config_path) || ~isfile(config_path)
    return;
end

root = jsondecode(fileread(config_path));
cfg = merge_named_struct(cfg, root, 'global');
for idx = 1:numel(sections)
    cfg = merge_named_struct(cfg, root, sections{idx});
end

if isfield(root, 'steps') && isstruct(root.steps)
    step_struct = root.steps;
    safe_name = matlab.lang.makeValidName(script_name);
    if isfield(step_struct, script_name)
        cfg = merge_structs(cfg, step_struct.(script_name));
    elseif isfield(step_struct, safe_name)
        cfg = merge_structs(cfg, step_struct.(safe_name));
    end
end
end


function tf = is_truthy(value)
if isstring(value)
    value = char(value);
end
value = lower(strtrim(value));
tf = any(strcmp(value, {'1', 'true', 'yes', 'y', 'on'}));
end


function out = merge_named_struct(base, root, field_name)
out = base;
if isfield(root, field_name) && isstruct(root.(field_name))
    out = merge_structs(out, root.(field_name));
end
end


function out = merge_structs(base, extra)
out = base;
if ~isstruct(extra)
    return;
end
names = fieldnames(extra);
for idx = 1:numel(names)
    out.(names{idx}) = extra.(names{idx});
end
end
