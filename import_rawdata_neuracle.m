% import_rawdata_neuracle.m
% 批量将 rawdata 下 test005, test006 的 Neuracle EEG 数据导入并另存为 EEGLAB dataset (.set)

clear;
clc;

% 添加 eeglab 及其插件路径
addpath('/home/lirui/matlab_tools/eeglab2025.1.0');
addpath('/home/lirui/matlab_tools/eeglab2025.1.0/plugins/NeuracleEEGFileReader1.1.1');

% 初始化 EEGLAB (不显示 GUI)
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% 路径定义
raw_root = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026/rawdata';
save_root = '/home/lirui/liulab_project/ieeg/Project_colorieeg_2026/seegdata';

subjects = {'test005', 'test006'};

for s = 1:length(subjects)
    subj = subjects{s};
    subj_raw_dir = fullfile(raw_root, subj);
    subj_save_dir = fullfile(save_root, subj);
    
    if ~exist(subj_save_dir, 'dir')
        mkdir(subj_save_dir);
        fprintf('Created save directory: %s\n', subj_save_dir);
    end
    
    % 获取被试子目录下的所有目录
    dir_info = dir(subj_raw_dir);
    for d = 1:length(dir_info)
        if dir_info(d).isdir && ~strcmp(dir_info(d).name, '.') && ~strcmp(dir_info(d).name, '..')
            sub_folder_name = dir_info(d).name;
            
            % 各种可能的路径
            foldname = fullfile(subj_raw_dir, sub_folder_name, '1', '1');
            data_file = fullfile(foldname, 'data.bdf');
            evt_file_target = fullfile(foldname, 'evt.bdf');
            
            % 如果 data.bdf 存在，检查/准备 evt.bdf
            if exist(data_file, 'file')
                % 检查 1/evt.bdf 是否存在于上一级目录
                evt_file_source = fullfile(subj_raw_dir, sub_folder_name, '1', 'evt.bdf');
                
                if ~exist(evt_file_target, 'file') && exist(evt_file_source, 'file')
                    fprintf('Copying evt.bdf from 1/ to 1/1/ for: %s\n', sub_folder_name);
                    try
                        copyfile(evt_file_source, evt_file_target);
                    catch ME_copy
                        fprintf('Failed to copy evt.bdf: %s\n', ME_copy.message);
                    end
                end
                
                % 此时如果 evt_file_target 存在，开始导入
                if exist(evt_file_target, 'file')
                    fprintf('\n==================================================\n');
                    fprintf('Processing: %s / %s\n', subj, sub_folder_name);
                    fprintf('Path: %s\n', foldname);
                    
                    try
                        % 导入
                        EEG = pop_importNeuracle({'data.bdf', 'evt.bdf'}, foldname);
                        EEG = eeg_checkset(EEG);
                        
                        % 使用 rawdata 文件夹名称命名 dataset
                        EEG.setname = sub_folder_name;
                        
                        % 保存 dataset
                        fprintf('Saving to %s ...\n', fullfile(subj_save_dir, [sub_folder_name '.set']));
                        EEG = pop_saveset(EEG, 'filename', [sub_folder_name '.set'], 'filepath', subj_save_dir);
                        
                        % 清除内存中的当前 dataset 以防累积
                        EEG = [];
                        ALLEEG = [];
                        CURRENTSET = [];
                        
                        fprintf('Finished successfully: %s\n', sub_folder_name);
                    catch ME
                        fprintf('Error occurred during processing %s:\n', sub_folder_name);
                        fprintf('%s\n', ME.message);
                    end
                else
                    fprintf('\nSkipped: %s / %s (evt.bdf not found in %s or %s)\n', subj, sub_folder_name, foldname, fileparts(foldname));
                end
            else
                fprintf('\nSkipped: %s / %s (data.bdf not found in %s)\n', subj, sub_folder_name, foldname);
            end
        end
    end
end

fprintf('\nAll subjects processed successfully!\n');
exit;
