function summary = raw_to_seegdata(project_root, eeglab_root, neuracle_plugin_root, mapping_csv, varargin)
%RAW_TO_SEEGDATA Convert mapped Neuracle raw sessions into EEGLAB datasets.
%
% The conversion is mapping-driven rather than subject-specific. The mapping
% CSV must contain: subject, task_num, raw_sessions, output_dir, output_stem.
% Multiple raw sessions in one row are separated with semicolons and are
% merged in the listed order (currently used by test005 Task 2).
%
% Example:
%   summary = raw_to_seegdata(project_root, eeglab_root, plugin_root, mapping_csv);
%   summary = raw_to_seegdata(..., 'overwrite', true, 'dry_run', false);

project_root = string(project_root);
eeglab_root = string(eeglab_root);
neuracle_plugin_root = string(neuracle_plugin_root);
mapping_csv = string(mapping_csv);
parser = inputParser;
addParameter(parser, 'overwrite', false, @(value) islogical(value) || isnumeric(value));
addParameter(parser, 'dry_run', false, @(value) islogical(value) || isnumeric(value));
parse(parser, varargin{:});
opts = parser.Results;

if ~isfolder(eeglab_root), error('EEGLAB directory not found: %s', eeglab_root); end
if ~isfolder(neuracle_plugin_root), error('Neuracle plugin directory not found: %s', neuracle_plugin_root); end
if ~isfile(mapping_csv), error('Mapping CSV not found: %s', mapping_csv); end

addpath(eeglab_root);
addpath(neuracle_plugin_root);
mapping = readtable(mapping_csv, 'TextType', 'string', 'VariableNamingRule', 'preserve');
required = ["subject", "task_num", "raw_sessions", "output_dir", "output_stem"];
if ~all(ismember(required, string(mapping.Properties.VariableNames)))
    error('Mapping CSV must contain columns: %s', strjoin(required, ', '));
end

if ~opts.dry_run
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab('nogui'); %#ok<ASGLU>
end

summary = struct('subject', {}, 'task_num', {}, 'status', {}, 'output_path', {}, 'message', {});
for rowIndex = 1:height(mapping)
    subject = string(mapping.subject(rowIndex));
    taskNum = double(mapping.task_num(rowIndex));
    sessionNames = split(string(mapping.raw_sessions(rowIndex)), ';');
    outputDir = fullfile(project_root, 'seegdata', string(mapping.output_dir(rowIndex)));
    outputStem = string(mapping.output_stem(rowIndex));
    outputSet = fullfile(outputDir, outputStem + '.set');
    outputFdt = fullfile(outputDir, outputStem + '.fdt');
    item = struct('subject', char(subject), 'task_num', taskNum, 'status', '', ...
        'output_path', char(outputSet), 'message', '');

    try
        if ~exist(outputDir, 'dir') && ~opts.dry_run
            mkdir(outputDir);
        end
        if (isfile(outputSet) || isfile(outputFdt)) && ~opts.overwrite && ~opts.dry_run
            item.status = 'skipped_existing';
            item.message = 'Output exists; set overwrite=true to regenerate.';
            summary(end+1) = item; %#ok<AGROW>
            fprintf('[SKIP] %s task%d -> %s\n', subject, taskNum, outputSet);
            continue;
        end

        partSets = cell(1, numel(sessionNames));
        for partIndex = 1:numel(sessionNames)
            sessionName = strtrim(sessionNames(partIndex));
            foldname = fullfile(project_root, 'rawdata', subject, sessionName, '1', '1');
            dataFile = fullfile(foldname, 'data.bdf');
            eventFile = fullfile(foldname, 'evt.bdf');
            eventSource = fullfile(project_root, 'rawdata', subject, sessionName, '1', 'evt.bdf');
            if ~isfile(dataFile)
                error('Missing data.bdf: %s', dataFile);
            end
            if ~isfile(eventFile) && isfile(eventSource) && ~opts.dry_run
                copyfile(eventSource, eventFile);
            end
            if ~isfile(eventFile)
                error('Missing evt.bdf: %s', eventFile);
            end
            if opts.dry_run
                fprintf('[CHECK] %s task%d part%d: %s\n', subject, taskNum, partIndex, foldname);
                continue;
            end

            fprintf('[IMPORT] %s task%d part%d: %s\n', subject, taskNum, partIndex, foldname);
            partSets{partIndex} = pop_importNeuracle({'data.bdf', 'evt.bdf'}, char(foldname));
            partSets{partIndex} = eeg_checkset(partSets{partIndex});
        end

        if opts.dry_run
            item.status = 'checked';
            item.message = sprintf('%d raw session(s) found', numel(sessionNames));
        else
            EEG = partSets{1};
            for partIndex = 2:numel(partSets)
                EEG = pop_mergeset(EEG, partSets{partIndex}, 1);
                EEG = eeg_checkset(EEG);
            end
            EEG.setname = char(outputStem);
            EEG = pop_saveset(EEG, 'filename', char(outputStem + '.set'), 'filepath', char(outputDir));
            EEG = [];
            ALLEEG = [];
            CURRENTSET = [];
            item.status = 'converted';
            item.message = sprintf('%d raw session(s) merged', numel(sessionNames));
            fprintf('[DONE] %s task%d -> %s\n', subject, taskNum, outputSet);
        end
    catch ME
        item.status = 'error';
        item.message = ME.message;
        fprintf('[ERROR] %s task%d: %s\n', subject, taskNum, ME.message);
    end
    summary(end+1) = item; %#ok<AGROW>
end

logPath = fullfile(project_root, 'color_analyse_0727', 'metadata', 'raw_to_seeg_conversion_log.tsv');
fid = fopen(logPath, 'w');
if fid > 0
    fprintf(fid, 'subject\ttask_num\tstatus\toutput_path\tmessage\n');
    for index = 1:numel(summary)
        fprintf(fid, '%s\t%d\t%s\t%s\t%s\n', summary(index).subject, summary(index).task_num, ...
            summary(index).status, summary(index).output_path, strrep(summary(index).message, sprintf('\n'), ' '));
    end
    fclose(fid);
end
fprintf('Conversion log written to %s\n', logPath);
end
