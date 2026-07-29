files = dir('color_cognition_pipeline/feature/subband_60_150/test00*/task*_hg_subband.mat');
for i = 1:length(files)
    filepath = fullfile(files(i).folder, files(i).name);
    fprintf('Converting %s...\n', filepath);
    load(filepath, 'epoch');
    save(filepath, 'epoch', '-v7');
end
fprintf('Done!\n');
exit;
