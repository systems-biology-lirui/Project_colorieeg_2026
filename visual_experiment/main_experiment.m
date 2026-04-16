function main_experiment()
% MAIN_EXPERIMENT Main Entry Point
% Structure:
%   1. Select Task
%   2. Load Config
%   3. Init PTB
%   4. Run Phase (Passive or E-Stim)
    % Clear
    sca; close all; clear; clc;

    % Add paths
    addpath(genpath(fullfile(pwd, 'Configs')));
    addpath(genpath(fullfile(pwd, 'Utils')));

    % 1. GUI Task Selection
    tasks = {
        'Task 1 (Passive - Real/Gray)', ...
        'Task 2 (Passive - Color Patches)', ...
        'Task 3 (Passive - Fruit Full)', ...
        'Task 4 (Passive - Fruit Gray)', ...
        'Task 5 (E-Stim Support)', ...
        'Task 6 (E-Stim Gray Only)', ...
        'Task 7 (white blank)', ...
        'Task 8 (单个水果的多电极)'
    };
    
    [idx, tf] = listdlg('ListString', tasks, 'SelectionMode', 'single', 'ListSize', [300, 150], 'Name', '选择测试任务');
    
    if ~tf
        disp('用户取消');
        return;
    end
    
    % 2. Load Config
    try
        switch idx
            case 1, cfg = config_task1();
            case 2, cfg = config_task2();
            case 3, cfg = config_task3();
            case 4, cfg = config_task4();
            case 5, cfg = config_task5();
            case 6, cfg = config_task6();
            case 7, cfg = config_task7();
            case 8, cfg = config_task8();
        end
        fprintf('Loaded Config: %s\n', cfg.taskName);
    catch ME
        errordlg(['加载配置失败: ' ME.message]);
        return;
    end
    
    % 3. Input Subject Info
    prompt = {'被试编号 (Subject ID):', 'Session 编号:'};
    def = {cfg.subID, '1'};
    answer = inputdlg(prompt, '输入被试信息', 1, def);
    if isempty(answer), return; end
    cfg.subID = answer{1};
    cfg.sessionNum = str2double(answer{2});
    
    % 4. Init Hardware/PTB
    try
        % IO
        if cfg.io.useSerial
            cfg.io.obj = io_utils.init_port(cfg.io.portName, cfg.io.baudRate);
        end
        
        % PTB
        PsychDefaultSetup(2);
        Screen('Preference', 'SkipSyncTests', 1);
        Screen('Preference', 'TextRenderer', 1);
        
        % Open Window
        [window, rect] = Screen('OpenWindow', max(Screen('Screens')), cfg.screen.bgColor, cfg.screen.fullsize);
        Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
        
        % Font
        try
            Screen('TextFont', window, 'Microsoft YaHei');
        catch
            Screen('TextFont', window, 'Arial');
        end
        Screen('TextSize', window, 30);
        
        % 5. Run Phase
        HideCursor;
        
        if idx >4
            % Task 5 & 6: E-Stim
            run_estim_phase(cfg, window);
        else
            % Tasks 1-4: Passive
            run_passive_phase(cfg, window);
        end
        
        % Cleanup
        ShowCursor;
        sca;
        
        if cfg.io.useSerial
            io_utils.close(cfg.io.obj);
        end
        
    catch ME
        sca;
        ShowCursor;
        if isfield(cfg, 'io') && cfg.io.useSerial && isfield(cfg.io, 'obj')
            io_utils.close(cfg.io.obj);
        end
        rethrow(ME);
    end

end
