function run_estim_phase(cfg, window)
% RUN_ESTIM_PHASE Runs Branch 2 (E-Stim Support)
% Logic: Wait Trigger -> Show -> Disappear (Auto)
% Strict frame-based timing for Duration.

    % Setup Timing
    ifi = Screen('GetFlipInterval', window);
    stimFrames = round(cfg.timing.stimulus / ifi); % e.g. 2.0s
    
    [w, h] = Screen('WindowSize', window);
    xC = w/2; yC = h/2;

    % Load Stimuli
    stimuli = [];
    fprintf('Loading Stimuli for E-Stim...\n');
    if isfield(cfg, 'task') && isfield(cfg.task, 'ordered_sequence') && cfg.task.ordered_sequence
        % Ordered Sequence Logic
        % Task 8: 2 Fruits
        % Task 5: 4 Fruits
        
        nFruits = length(cfg.fruitParams);
        
        % Sequence definitions: {FruitIndex, TypeName}
        % Logic: F1_G, F2_G... Fn_G, F1_T, F2_T... Fn_T, F1_F, F2_F... Fn_F
        seqDef = {};
        
        % Gray Pass
        for f = 1:nFruits
            seqDef{end+1, 1} = cfg.fruitParams(f);
            seqDef{end, 2} = 'Gray';
        end
        % True Pass
        for f = 1:nFruits
            seqDef{end+1, 1} = cfg.fruitParams(f);
            seqDef{end, 2} = 'True';
        end
        % False Pass
        for f = 1:nFruits
            seqDef{end+1, 1} = cfg.fruitParams(f);
            seqDef{end, 2} = 'False';
        end
        
        for idx = 1:cfg.img.countPerCategory
            for s = 1:size(seqDef, 1)
                fruit = seqDef{s, 1};
                typeName = seqDef{s, 2};
                
                % Construct Name: Name_Type_XX.bmp
                fNameBase = sprintf('%s_%s_%02d', fruit.name, typeName, idx);
                
                % Find file
                fPath = '';
                exts = {'.bmp', '.jpg', '.png'};
                for e = 1:length(exts)
                    p = fullfile(cfg.stimDir, [fNameBase exts{e}]);
                    if exist(p, 'file')
                        fPath = p;
                        break;
                    end
                end
                
                if isempty(fPath)
                    warning('File not found: %s (Skipping)', fNameBase);
                    continue;
                end
                
                % Load
                img = imread(fPath);
                tex = Screen('MakeTexture', window, img);
                
                % Rect
                [iH, iW, ~] = size(img);
                sW = iW * cfg.img.scaleFactor;
                sH = iH * cfg.img.scaleFactor;
                dstRect = CenterRectOnPoint([0 0 sW sH], xC + cfg.img.xOffset, yC + cfg.img.yOffset);
                
                item.texture = tex;
                item.rect = dstRect;
                item.name = fNameBase;
                item.filename = fNameBase;
                item.catName = [fruit.name '_' typeName];
                item.marker = fruit.codes.(typeName);
                
                stimuli = [stimuli; item];
            end
        end
        
    else
        % Standard Random Loading
        for c = 1:length(cfg.categories)
            cat = cfg.categories(c);
            d = dir(fullfile(cfg.stimDir, ['*', cat.name, '*.bmp']));
            if isempty(d), d = dir(fullfile(cfg.stimDir, ['*', cat.name, '*.jpg'])); end
            if isempty(d), d = dir(fullfile(cfg.stimDir, ['*', cat.name, '*.png'])); end
            
            if isempty(d), continue; end
            
            % Select subset or cycle if not enough
            nAvailable = length(d);
            nRequired = cfg.img.countPerCategory;
            
            if nAvailable >= nRequired
                selIdx = randperm(nAvailable, nRequired);
                selected = d(selIdx);
            else
                % Cycle back to head logic: 1..N, 1..N, ...
                fullCycles = repmat(1:nAvailable, 1, floor(nRequired / nAvailable));
                remainder = 1:mod(nRequired, nAvailable);
                selIdx = [fullCycles, remainder];
                selected = d(selIdx);
            end
            
            for k = 1:length(selected)
                fPath = fullfile(selected(k).folder, selected(k).name);
                img = imread(fPath);
                tex = Screen('MakeTexture', window, img);
                
                % Rect
                [iH, iW, ~] = size(img);
                sW = iW * cfg.img.scaleFactor;
                sH = iH * cfg.img.scaleFactor;
                dstRect = CenterRectOnPoint([0 0 sW sH], xC + cfg.img.xOffset, yC + cfg.img.yOffset);
                
                item.texture = tex;
                item.rect = dstRect;
                item.name = selected(k).name;
                item.filename = selected(k).name; 
                item.catName = cat.name;
                item.marker = cat.code;
                stimuli = [stimuli; item];
            end
        end
    end
    if isempty(stimuli)
        warning('No stimuli found in directory: %s', cfg.stimDir);
        return;
    end
    % =========================================================================
    % Fixed-Order Block Design (固定顺序组块设计)
    % 逻辑：严格按照 Cat1 -> Cat2 -> Cat3 ... 的顺序循环，不进行随机打乱
    % =========================================================================
    if ~(isfield(cfg, 'task') && isfield(cfg.task, 'ordered_sequence') && cfg.task.ordered_sequence)
        
        fprintf('Applying Fixed-Order Block Design (No Randomization)...\n');
        
        nCats = length(cfg.categories);
        nPerCat = cfg.img.countPerCategory;
        
        % 检查：确保试次总数符合计算要求
        % 此时 stimuli 列表是按类别成堆排列的：[Cat1_All... , Cat2_All... , Cat3_All...]
        if length(stimuli) == nCats * nPerCat
            
            blockedStimuli = [];
            
            % 外层循环：遍历每一个 Block (第1轮, 第2轮...)
            for i = 1:nPerCat
                % 内层循环：遍历每一个类别
                for c = 1:nCats
                    % 计算索引：取第 c 个类别的第 i 张图片
                    % 公式说明：(c-1)*nPerCat 跳过前几个类别的所有图，+i 取当前类别的当前进度
                    idx = (c - 1) * nPerCat + i;
                    
                    % 直接加入队列，【不进行 randperm 打乱】
                    blockedStimuli = [blockedStimuli; stimuli(idx)];
                end
            end
            
            stimuli = blockedStimuli;
            fprintf('Sequence generated: Repeated %d blocks of [Cat1...Cat%d].\n', nPerCat, nCats);
            
        else
            % 异常保护：如果数量对不上，回退到普通随机
            warning('Trial count mismatch. Falling back to full random.');
            stimuli = stimuli(randperm(length(stimuli)));
        end
    end
    % =========================================================================
    
    % Prompt in Chinese
    msg = double('开始');
    Screen('DrawText', window, msg, xC - 200, yC, [255 255 255]);
    Screen('Flip', window);
    KbStrokeWait;
    Screen('Flip', window); % Clear
    
    currIdx = 1;
    running = true;
    
    results = struct('trialNum', {}, 'catName', {}, 'filename', {}, 'marker', {}, 'stimOnset', {});
    
    % Setup UDP (Optional)
    udpObj = [];
    if isfield(cfg, 'udp') && isfield(cfg.udp, 'enabled') && cfg.udp.enabled
        try
            udpObj = udp(cfg.udp.ip, cfg.udp.port);
            fopen(udpObj);
            fprintf('UDP Trigger Enabled: %s:%d (Delay: %.2fs)\n', cfg.udp.ip, cfg.udp.port, cfg.udp.delay);
        catch ME
            warning(ME.identifier, '%s', ME.message);
            udpObj = [];
        end
    end
    
    % 3. Main Loop
    try
        HideCursor;
        startTime = GetSecs;
        
        for i = 1:length(stimuli)
            trial = stimuli(i);
            
            % Wait for Trigger
            msg = double(sprintf('Trial %d/%d\n等待触发 (空格键)...', i, length(stimuli)));
            Screen('DrawText', window, msg, xC - 150, yC, [0 0 0]);
            Screen('Flip', window);
            
            while true
                [keyIsDown, ~, keyCode] = KbCheck;
                if keyIsDown
                    if keyCode(cfg.keys.space), break; end
                    if keyCode(cfg.keys.escape), error('User Exit'); end
                end
                WaitSecs(0.01);
            end
            
            % Fixation Only
            Screen('DrawLines', window, [-cfg.fixation.size cfg.fixation.size 0 0; 0 0 -cfg.fixation.size cfg.fixation.size], cfg.fixation.lineWidth, cfg.fixation.color, [xC yC]);
            Screen('Flip', window);
            WaitSecs(cfg.task.fixationDuration);
            
            % Stimulus
            Screen('DrawTexture', window, trial.texture, [], trial.rect);
            % Draw Fixation on top
            Screen('DrawLines', window, [-cfg.fixation.size cfg.fixation.size 0 0; 0 0 -cfg.fixation.size cfg.fixation.size], cfg.fixation.lineWidth, cfg.fixation.color, [xC yC]);
            
            [vbl, onsetTime] = Screen('Flip', window);
            
            % Trigger
            if cfg.io.useSerial
                io_utils.send_trigger(cfg.io.obj, trial.marker);
            end
            
            % UDP Trigger State
            udpSent = false;
            
            % Disappear Logic (Always Manual: Wait for Space)
            while true
                % Check UDP
                if ~isempty(udpObj) && ~udpSent
                    if (GetSecs - onsetTime) >= cfg.udp.delay
                        try
                            fwrite(udpObj, cfg.udp.message);
                            % fprintf('UDP Sent: %s\n', cfg.udp.message); % Optional debug
                        catch
                            fprintf('UDP Send Failed\n');
                        end
                        udpSent = true;
                    end
                end

                [keyIsDown, ~, keyCode] = KbCheck;
                if keyIsDown
                    if keyCode(cfg.keys.space) || keyCode(cfg.keys.next), break; end
                    if keyCode(cfg.keys.escape), error('User Exit'); end
                end
                WaitSecs(0.005);
            end
            [vbl, offsetTime] = Screen('Flip', window);
            actualDur = offsetTime - onsetTime;
            
            % Record
            idx = length(results) + 1;
            results(idx).trialNum = i;
            results(idx).imgName = trial.filename;
            results(idx).category = trial.catName;
            results(idx).marker = trial.marker;
            results(idx).onsetTime = onsetTime - startTime;
            results(idx).offsetTime = offsetTime - startTime;
            results(idx).duration = actualDur;
            
            WaitSecs(0.2);
        end
        
        save_data(cfg, results, stimuli);
        
    catch ME
        save_data(cfg, results, stimuli);
        if ~isempty(udpObj)
            fclose(udpObj);
            delete(udpObj);
        end
        ShowCursor;
        rethrow(ME);
    end
    
    if ~isempty(stimuli)
        Screen('Close', [stimuli.texture]);
    end
    
    if ~isempty(udpObj)
        fclose(udpObj);
        delete(udpObj);
    end
    ShowCursor;

end

function save_data(cfg, results, stimData)
    if isempty(results), return; end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    taskSuffix = regexprep(cfg.taskName, '[^a-zA-Z0-9]', '');
    fname = sprintf('%s_%s_Estim_Session%d_%s.mat', cfg.subID, taskSuffix, cfg.sessionNum, timestamp);
    savePath = fullfile(cfg.dataDir, fname);
    save(savePath, 'results', 'cfg', 'stimData');
    fprintf('Data saved to %s\n', savePath);
end
