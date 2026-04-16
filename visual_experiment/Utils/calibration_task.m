function calibration_task()
% CALIBRATION_TASK Subjective Isoluminance Calibration
%
% Output: Data/color_calibration.mat
% Structure: calibrated_colors.(ColorName).scaleFactor

    sca; close all; clc;
    
    % Add Configs to path
    addpath(genpath(fullfile(pwd, 'Configs')));
    addpath(genpath( fullfile(pwd, 'Utils')));

    % Load Task 2 Config to get path
    cfg = config_task2(); 
    
    % Stimuli Directory
    stimDir = 'D:\Desktop\seeg\stimuli_pic\Stimuli_Task2';
    
    % Define Colors to Calibrate
    % Format: {CategoryName, FileName}
    colorMap = {
        'Red',    'Red_Color_01.bmp';
        'Yellow', 'Yellow_Color_01.bmp';
        'Blue',   'Blue_Color_01.bmp';
        'Green',  'Green_Color_01.bmp'
    };
    refFile = 'blank_01.bmp';
    
    % PTB Setup
    PsychDefaultSetup(2);
    Screen('Preference', 'SkipSyncTests', 1);
    screens = Screen('Screens');
    screenNumber = max(screens);
    
    try
        [window, windowRect] = Screen('OpenWindow', screenNumber, [0 0 0]); % Black background for calibration
        [xCenter, yCenter] = RectCenter(windowRect);
        Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
        Screen('TextSize', window, 30);
        
        % Load Reference
        refPath = fullfile(stimDir, refFile);
        if ~exist(refPath, 'file')
            error('Reference file not found: %s', refPath);
        end
        refImg = imread(refPath);
        [h, w, ~] = size(refImg);
        % Use center pixel
        refRGB = double(squeeze(refImg(round(h/2), round(w/2), :)))';
        
        % Load Targets
        targets = struct();
        for i = 1:size(colorMap, 1)
            cName = colorMap{i, 1};
            fName = colorMap{i, 2};
            fPath = fullfile(stimDir, fName);
            if ~exist(fPath, 'file')
                warning('File not found: %s', fPath);
                continue;
            end
            img = imread(fPath);
            [h, w, ~] = size(img);
            targets.(cName).baseRGB = double(squeeze(img(round(h/2), round(w/2), :)))';
            targets.(cName).name = cName;
        end
        
        % Prepare Trials (1 rep per side)
        trialList = {};
        cNames = fieldnames(targets);
        for i = 1:length(cNames)
            for side = {'Left', 'Right'}
                trial.name = cNames{i};
                trial.side = side{1};
                trialList{end+1} = trial;
            end
        end
        trialList = trialList(randperm(length(trialList)));
        
        % Instructions
        msg = double('使用 上/下 键调节亮度。按空格键确认。按任意键开始。');
        DrawFormattedText(window, msg, 'center', 'center', [255 255 255]);
        Screen('Flip', window);
        KbStrokeWait;
        
        % Loop
        results = struct();
        rectSize = 300;
        baseRect = [0 0 rectSize rectSize];
        centeredRect = CenterRectOnPointd(baseRect, xCenter, yCenter);
        
        for t = 1:length(trialList)
            curr = trialList{t};
            baseRGB = targets.(curr.name).baseRGB;
            scale = 1.0;
            confirmed = false;
            
            while ~confirmed
                % Calculate Color
                adjRGB = baseRGB * scale;
                adjRGB(adjRGB>255)=255; adjRGB(adjRGB<0)=0;
                
                if strcmp(curr.side, 'Right')
                    colRight = adjRGB;
                    colLeft = refRGB;
                else
                    colRight = refRGB;
                    colLeft = adjRGB;
                end
                
                % Draw Semicircles
                % Right (0-180), Left (180-180)
                Screen('FillArc', window, colRight, centeredRect, 0, 180);
                Screen('FillArc', window, colLeft, centeredRect, 180, 180);
                
                DrawFormattedText(window, double(sprintf('Trial %d/%d: %s', t, length(trialList), curr.name)), 'center', yCenter - 200, [255 255 255]);
                Screen('Flip', window);
                
                [keyIsDown, ~, keyCode] = KbCheck;
                if keyIsDown
                    if keyCode(KbName('UpArrow'))
                        scale = scale + 0.01;
                    elseif keyCode(KbName('DownArrow'))
                        scale = max(0, scale - 0.01);
                    elseif keyCode(KbName('Space'))
                        confirmed = true;
                    elseif keyCode(KbName('ESCAPE'))
                        sca; return;
                    end
                    WaitSecs(0.1);
                end
            end
            
            results(t).name = curr.name;
            results(t).scaleFactor = scale;
            
            Screen('Flip', window);
            WaitSecs(0.5);
        end
        
        % Save Results
        calibrated_colors = struct();
        for i = 1:length(cNames)
            cName = cNames{i};
            idxs = strcmp({results.name}, cName);
            avgScale = mean([results(idxs).scaleFactor]);
            calibrated_colors.(cName).scaleFactor = avgScale;
            % For compatibility/info
            calibrated_colors.(cName).rgb_adjusted = targets.(cName).baseRGB * avgScale;
        end
        
        savePath = fullfile(cfg.dataDir, 'color_calibration.mat');
        save(savePath, 'calibrated_colors');
        
        DrawFormattedText(window, double('校准已保存!'), 'center', 'center', [255 255 255]);
        Screen('Flip', window);
        WaitSecs(2);
        sca;
        
    catch ME
        sca;
        rethrow(ME);
    end
end
