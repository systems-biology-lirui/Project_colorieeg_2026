function plot_epoch_examples()
%PLOT_EPOCH_EXAMPLES Create mean +/- SEM epoch figures from rebuilt HDF5 files.
%
% This is a small MATLAB bridge for environments where the Python plotting
% dependencies are not installed. The reusable Python implementation lives
% in pipeline/epoch_plots.py; both versions use the same (trial mean + SEM)
% definition.

root = fileparts(fileparts(mfilename('fullpath')));
outDir = fullfile(root, 'result', 'epoch_examples');
if ~exist(outDir, 'dir'), mkdir(outDir); end
set(groot, 'defaultFigureVisible', 'off');

task1Path = fullfile(root, 'process_data', 'test001', 'task1_epoched_1_200Hz.h5');
task2Path = fullfile(root, 'process_data', 'test001', 'task2_epoched_1_200Hz.h5');

plot_one_file(task1Path, {'face_color', 'face_gray'}, ...
    {'A2', 'D3', 'G3', 'G6'}, ...
    fullfile(outDir, 'test001_task1_face_color_vs_gray.png'), ...
    'test001 Task 1: face color vs gray');

plot_one_file(task2Path, ...
    {'cabbage_gray', 'kiwi_gray', 'strawberry_gray', 'watermelon_gray'}, ...
    {'A2', 'D3', 'G3', 'G6'}, ...
    fullfile(outDir, 'test001_task2_gray_fruits.png'), ...
    'test001 Task 2: gray fruit conditions');

fprintf('Epoch example figures written to %s\n', outDir);
end

function plot_one_file(path, requestedConditions, requestedChannels, outputPath, figureTitle)
conditionNames = read_strings(path, '/condition_names');
labels = read_strings(path, '/labels');
timeMs = double(h5read(path, '/time_ms'));

conditions = requestedConditions(ismember(requestedConditions, conditionNames));
if isempty(conditions)
    error('None of the requested conditions are present in %s', path);
end
availableChannels = requestedChannels(ismember(upper(requestedChannels), upper(labels)));
if isempty(availableChannels)
    availableChannels = labels(1:min(4, numel(labels)));
end

channelIndices = zeros(1, numel(availableChannels));
for k = 1:numel(availableChannels)
    channelIndices(k) = find(strcmpi(labels, availableChannels{k}), 1, 'first');
end

fig = figure('Color', 'w', 'Position', [100, 100, 1100, 220 * numel(availableChannels)]);
tiledlayout(numel(availableChannels), 1, 'TileSpacing', 'compact', 'Padding', 'compact');
colors = lines(max(numel(conditions), 2));
for channelIndex = 1:numel(availableChannels)
    nexttile;
    hold on;
    for conditionIndex = 1:numel(conditions)
        data = double(h5read(path, ['/epochs/' conditions{conditionIndex}]));
        % MATLAB exposes the HDF5 dimensions as (time, channel, trial).
        trialValues = squeeze(data(:, channelIndices(channelIndex), :));
        if isvector(trialValues), trialValues = reshape(trialValues, [], 1); end
        meanValues = mean(trialValues, 2, 'omitnan');
        semValues = std(trialValues, 0, 2, 'omitnan') ./ sqrt(size(trialValues, 2));
        fill([timeMs(:); flipud(timeMs(:))], ...
            [meanValues(:) - semValues(:); flipud(meanValues(:) + semValues(:))], ...
            colors(conditionIndex, :), 'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
            'HandleVisibility', 'off');
        plot(timeMs(:), meanValues(:), 'Color', colors(conditionIndex, :), 'LineWidth', 1.5, ...
            'DisplayName', conditions{conditionIndex});
    end
    xline(0, '--', 'Color', [0.35, 0.35, 0.35], 'LineWidth', 0.8, ...
        'HandleVisibility', 'off');
    yline(0, '-', 'Color', [0.75, 0.75, 0.75], 'LineWidth', 0.6, ...
        'HandleVisibility', 'off');
    ylabel(availableChannels{channelIndex}, 'Interpreter', 'none');
    box off;
    if channelIndex == 1, legend('Location', 'best', 'Box', 'off', 'Interpreter', 'none'); end
end
xlabel('Time (ms)');
sgtitle(figureTitle, 'Interpreter', 'none');
exportgraphics(fig, outputPath, 'Resolution', 160);
close(fig);
fprintf('  %s | conditions=%s | channels=%s\n', outputPath, ...
    strjoin(conditions, ', '), strjoin(availableChannels, ', '));
end

function values = read_strings(path, datasetPath)
raw = h5read(path, datasetPath);
if iscell(raw)
    values = cellfun(@(value) strtrim(char(value)), raw, 'UniformOutput', false);
elseif isstring(raw)
    values = cellstr(raw(:));
elseif ischar(raw)
    values = cellstr(raw');
else
    values = cellstr(string(raw(:)));
end
values = values(:)';
end
