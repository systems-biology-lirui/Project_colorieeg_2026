function paths = newanalyse_paths(base_path)
if nargin < 1 || isempty(base_path)
    current_dir = fileparts(mfilename('fullpath'));
    base_path = fileparts(current_dir);
else
    current_dir = fullfile(base_path, 'newanalyse');
end

paths = struct();
paths.base_path = base_path;
paths.analysis_code_dir = current_dir;
paths.feature_root = fullfile(base_path, 'feature');
paths.result_root = fullfile(base_path, 'result');
end