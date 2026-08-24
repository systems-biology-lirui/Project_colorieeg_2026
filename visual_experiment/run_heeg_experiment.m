function run_heeg_experiment()
% RUN_HEEG_EXPERIMENT
% 新构建的代码：
% 1. 白色背景
% 2. 屏幕中央显示“开始”
% 3. 主试按键后，文字消失，注视点出现，同时开始UDP刺激（间隔1s，重复40次，Output=1）
% 4. 运行完毕1s后，注视点消失，文字出现，等待被试按键
% 5. 循环直到主试退出
% 6. 保存数据

    % --- Parameters ---
    ipOutput = '192.168.1.2';
    portOutput = 9080;
    stimOutputVal = 1;
    stimReps = 1;
    stimInterval = 41;
    
    bgColor = [255 255 255]; % White
    textColor = [0 0 0];     % Black
    fixColor = [0 0 0];
    
    dataDir = fullfile(pwd, 'Data_HEEG');
    if ~exist(dataDir, 'dir'), mkdir(dataDir); end
    
    % --- Input Subject Info ---
    prompt = {'Subject ID:', 'Session ID:'};
    def = {'test001', '1'};
    answer = inputdlg(prompt, 'HEEG Experiment', 1, def);
    if isempty(answer), return; end
    subID = answer{1};
    sessionID = answer{2};
    
    % --- Setup UDP ---
    try
        udpObj = udp(ipOutput, portOutput);
        fopen(udpObj);
        fprintf('UDP initialized: %s:%d\n', ipOutput, portOutput);
    catch ME
        errordlg(['UDP Init Failed: ' ME.message]);
        return;
    end
    
    % --- Setup IO ---
    useSerial = false; % Set to true to enable serial trigger
    ioObj = [];
    if useSerial
        try
            % Use Config's default port or hardcoded here
            % Ideally we should load config_common but let's keep this standalone-ish or hardcode COM3
            portName = 'COM3';
            baudRate = 115200;
            ioObj = io_utils.init_port(portName, baudRate);
            fprintf('Serial Port initialized: %s\n', portName);
        catch ME
            warning(ME.identifier, 'Serial Init Failed: %s. Triggers disabled.', ME.message);
            useSerial = false;
        end
    end
    
    % --- Setup PTB ---
    sca;
    PsychDefaultSetup(2);
    Screen('Preference', 'SkipSyncTests', 1);
    Screen('Preference', 'TextRenderer', 1);
    
    try
        [window, rect] = Screen('OpenWindow', max(Screen('Screens')), bgColor);
        [xC, yC] = RectCenter(rect);
        Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
        
        try
            Screen('TextFont', window, 'Microsoft YaHei');
        catch
            Screen('TextFont', window, 'Arial');
        end
        Screen('TextSize', window, 40);
        
        KbName('UnifyKeyNames');
        keySpace = KbName('SPACE');
        keyEsc = KbName('ESCAPE');
        keyEnter = KbName('Return');
        
        % Data Storage
        results = struct('trialNum', {}, 'startTime', {}, 'outputTimes', {});
        trialCount = 0;
        
        HideCursor;
        running = true;
        while running
            trialCount = trialCount + 1;
            
            % --- Phase 1: Ready (Experimenter Control) ---
            msg = double('按空格开始');
            Screen('DrawText', window, msg, xC - 40, yC+100, textColor);
            % Screen('DrawText', window, double('(按空格开始，ESC退出)'), xC - 200, yC + 100, [100 100 100]);
            Screen('Flip', window);
            
            % Wait for Experimenter
            while true
                [keyIsDown, ~, keyCode] = KbCheck;
                if keyIsDown
                    if keyCode(keySpace)
                        break; % Start Trial
                    elseif keyCode(keyEsc)
                        running = false;
                        break; % Quit
                    end
                end
                WaitSecs(0.01);
            end
            
            if ~running, break; end
            
            % --- Phase 2: Stimulation ---
            % Fixation (Pre-Stimulation 200ms)
            Screen('DrawLines', window, [-20 20 0 0; 0 0 -20 20], 4, fixColor, [xC yC]);
            Screen('Flip', window);
            
            WaitSecs(0.2); % 200ms Pre-stim fixation
            
            trialStartTime = GetSecs; % Record Stimulation Start
            
            outputTimestamps = [];
            
            % UDP Loop
            nextStimTime = trialStartTime;
            for i = 1:stimReps
                % Draw Fixation
                Screen('DrawLines', window, [-20 20 0 0; 0 0 -20 20], 4, fixColor, [xC yC]);
                
                % Draw Counter (Bottom Right, Small, Light Color)
                % "Light" on White background = Light Gray (e.g., [180 180 180])
                oldSize = Screen('TextSize', window, 20);
                txt = num2str(i);
                [normRect, ~] = Screen('TextBounds', window, txt);
                textW = normRect(3);
                textH = normRect(4);
                % Position: Bottom Right with margin
                drawX = rect(3) - textW - 20; 
                drawY = rect(4) - textH - 20;
                Screen('DrawText', window, txt, drawX, drawY, [180 180 180]);
                Screen('TextSize', window, oldSize);
                
                % Flip to show stimuli
                vbl = Screen('Flip', window);
                
                % Send UDP
                fwrite(udpObj, num2str(stimOutputVal));
                
                % Send Trigger 200
                if useSerial && ~isempty(ioObj)
                    io_utils.send_trigger(ioObj, 200);
                end
                
                tSent = GetSecs;
                outputTimestamps(end+1) = tSent - trialStartTime;
                
                % Wait Interval
                % Use absolute timing for drift correction
                nextStimTime = nextStimTime + stimInterval;
                WaitSecs('UntilTime', nextStimTime);
                
                % Check Escape
                [k, ~, kc] = KbCheck;
                if k && kc(keyEsc), running = false; break; end
            end
            
            if ~running, break; end
            
            % Wait 1s after done
            WaitSecs(1.0);
            
            % --- Phase 3: Subject Response ---
            % Fixation gone, Text appears
            msg = double('请按键继续...');
            Screen('DrawText', window, msg, xC - 100, yC, textColor);
            Screen('Flip', window);
            
            % Wait for Subject Key (Any key)
            KbStrokeWait;
            
            % --- Save Data for this trial ---
            results(trialCount).trialNum = trialCount;
            results(trialCount).startTime = trialStartTime;
            results(trialCount).outputTimes = outputTimestamps;
            
            % Save to file incrementally
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            fname = sprintf('%s_HEEG_Session%s_%s.mat', subID, sessionID, timestamp);
            savePath = fullfile(dataDir, fname);
            save(savePath, 'results');
            
            % Brief gap
            Screen('Flip', window);
            WaitSecs(0.5);
        end
        
        % Cleanup
        if useSerial && ~isempty(ioObj)
            clear ioObj; % Triggers auto-close in destructor usually, or we can explicit close if needed
        end
        fclose(udpObj);
        delete(udpObj);
        ShowCursor;
        sca;
        fprintf('Experiment Finished. Data saved to %s\n', dataDir);
        
    catch ME
        if exist('udpObj', 'var'), fclose(udpObj); delete(udpObj); end
        ShowCursor;
        sca;
        rethrow(ME);
    end
end
