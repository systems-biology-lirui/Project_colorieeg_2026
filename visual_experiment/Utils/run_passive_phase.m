function run_passive_phase(cfg, window)
    % RUN_PASSIVE_PHASE Runs Branch 1 (Passive Viewing)
    % 【修改版】实现了 Block Design 和 Catch Trial 均匀分布
    
    % 1. Setup Timing
    ifi = Screen('GetFlipInterval', window);
    stimFrames = round(cfg.timing.stimulus / ifi);
    blankOptions = cfg.timing.blankRange(1):cfg.timing.blankStep:cfg.timing.blankRange(2);
    
    fprintf('Passive Phase: Stim = %d frames\n', stimFrames);
    
    % 2. Load Calibration if needed
    calibData = [];
    if cfg.calibration.enabled && exist(cfg.calibration.file, 'file')
        tmp = load(cfg.calibration.file);
        if isfield(tmp, 'calibrated_colors')
            calibData = tmp.calibrated_colors;
            fprintf('Loaded Calibration Data.\n');
        end
    end
    
    % 3. History & Stimulus Selection
    taskSuffix = regexprep(cfg.taskName, '[^a-zA-Z0-9]', '');
    historyFileName = sprintf('session_history_%s.mat', taskSuffix);
    historyFilePath = fullfile(cfg.dataDir, historyFileName);
    
    sessionHistory = struct();
    if exist(historyFilePath, 'file')
        try
            loadedHist = load(historyFilePath);
            sessionHistory = loadedHist.sessionHistory;
            fprintf('Loaded History: %s\n', historyFileName);
        catch
            warning('Failed to load history, starting fresh.');
        end
    end
    
    stimuli = [];
    [w, h] = Screen('WindowSize', window);
    xC = w/2; yC = h/2;
    fprintf('Loading Stimuli...\n');
    
    % 注意：为了后续 Block Design 逻辑，我们必须保证 stimuli 数组是按类别顺序排列的
    % 代码目前的加载逻辑正是如此：先加载完 Cat1 的所有图片，再加载 Cat2...
    
    for c = 1:length(cfg.categories)
        cat = cfg.categories(c);
    
        d = dir(fullfile(cfg.stimDir, ['*.bmp']));
        if isempty(d), d = dir(fullfile(cfg.stimDir, ['*.jpg'])); end
        if isempty(d), d = dir(fullfile(cfg.stimDir, ['*.png'])); end
    
        matches = [];
        for k = 1:length(d)
            if contains(d(k).name, cat.name, 'IgnoreCase', true)
                matches = [matches; d(k)];
            end
        end
    
        if isempty(matches)
            warning('No images found for category: %s', cat.name);
            continue;
        end
    
        [~, sortIdx] = sort({matches.name});
        matches = matches(sortIdx);
        nTotal = length(matches);
    
        lastIdx = 0;
        if isfield(sessionHistory, cat.name)
            lastIdx = sessionHistory.(cat.name).lastIdx;
        end
    
        selected = [];
        currIdx = lastIdx;
        for k = 1:cfg.img.countPerCategory
            currIdx = mod(currIdx, nTotal) + 1;
            selected = [selected; matches(currIdx)];
        end
    
        sessionHistory.(cat.name).lastIdx = currIdx;
    
        for k = 1:length(selected)
            fPath = fullfile(selected(k).folder, selected(k).name);
            img = imread(fPath);
    
            % Calibration Logic
            currentScale = cfg.img.scaleFactor;
            if cfg.calibration.enabled && ~isempty(calibData) && isfield(calibData, cat.name)
                sf = calibData.(cat.name).scaleFactor;
                imgDouble = double(img);
                bgRef = 100; tol = 15;
                isBg = abs(imgDouble(:,:,1) - bgRef) <= tol & ...
                    abs(imgDouble(:,:,2) - bgRef) <= tol & ...
                    abs(imgDouble(:,:,3) - bgRef) <= tol;
                imgScaled = imgDouble * sf;
                for ch = 1:3
                    chanS = imgScaled(:,:,ch); chanO = imgDouble(:,:,ch);
                    chanS(isBg) = chanO(isBg); imgScaled(:,:,ch) = chanS;
                end
                imgScaled(imgScaled>255)=255; imgScaled(imgScaled<0)=0;
                img = uint8(imgScaled);
            end
    
            tex = Screen('MakeTexture', window, img);
            [iH, iW, ~] = size(img);
            sW = iW * currentScale; sH = iH * currentScale;
            dstRect = CenterRectOnPoint([0 0 sW sH], xC + cfg.img.xOffset, yC + cfg.img.yOffset);
    
            item.texture = tex;
            item.rect = dstRect;
            item.marker = cat.code;
            item.catName = cat.name;
            item.filename = selected(k).name;
            item.isCatch = false; % 默认非 Catch
            stimuli = [stimuli; item];
        end
    end
    
    try
        save(historyFilePath, 'sessionHistory');
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end
    
    % =========================================================================
    % 4. Block Design Randomization & Uniform Catch Trials (完全重写部分)
    % =========================================================================
    if isempty(stimuli)
        error('No stimuli loaded!');
    end
    
    % --- 步骤 A: 实现 Block Design (小循环内各类别各出现一次) ---
    % 目前 stimuli 是按类别顺序排列的：[Cat1_1...Cat1_20, Cat2_1...Cat2_20, ...]
    
    normalTrials = stimuli; % 此时全是正常试次
    nCats = length(cfg.categories);
    nPerCat = cfg.img.countPerCategory;
    
    % 安全检查：确保加载的试次数量符合预期
    if length(normalTrials) ~= nCats * nPerCat
        warning('Loaded trials count mismatch. Randomization might be imperfect.');
    end
    
    blockedNormalTrials = [];
    
    % 循环每一轮 (Block)，每一轮包含所有类别各一张
    for i = 1:nPerCat
        miniBlock = [];
    
        % 从每个类别中提取第 i 张图片
        for c = 1:nCats
            % 计算该类别在数组中的起始位置偏移量
            offset = (c - 1) * nPerCat;
            idx = offset + i;
    
            if idx <= length(normalTrials)
                miniBlock = [miniBlock; normalTrials(idx)];
            end
        end
    
        % 打乱这个 mini-Block (如: [Cat1, Cat2, Cat3] -> [Cat2, Cat1, Cat3])
        shuffledBlock = miniBlock(randperm(length(miniBlock)));
    
        % 加入到临时总表
        blockedNormalTrials = [blockedNormalTrials; shuffledBlock];
    end
    
    % 此时 blockedNormalTrials 已经是 "Interleaved Block Design" 了
    
    % --- 步骤 B: 生成并均匀插入 Catch Trials ---
    finalStimuli = blockedNormalTrials;
    
    if isfield(cfg, 'catch') && cfg.catch.probability > 0
        % 计算 Catch Trial 数量
        nNormal = length(blockedNormalTrials);
        nCatch = round(nNormal * cfg.catch.probability);
    
        fprintf('Structure: %d Blocks (size %d), Adding %d Catch Trials uniformly.\n', ...
            nPerCat, nCats, nCatch);
    
        % 生成 Catch Trial 模板
        catchItemTemplate = struct();
        catchItemTemplate.texture = [];
        catchItemTemplate.rect = [];
        catchItemTemplate.marker = cfg.catch.trigger;
        catchItemTemplate.catName = 'CatchTrial';
        catchItemTemplate.filename = 'Catch_Trial';
        catchItemTemplate.isCatch = true;
    
        % --- 均匀插入算法 ---
        % 我们有 nCatch 个球，要插入到 nNormal 个位置的队列中
        % 使用 linspace 找到均匀的插入索引
    
        % 创建一个包含所有位置的空数组 (0代表正常试次，1代表Catch)
        totalSlots = nNormal + nCatch;
        trialSequenceType = zeros(totalSlots, 1);
    
        % 计算 Catch Trial 应该出现的位置索引 (均匀分布)
        % 例如: linspace(2, Total-1, nCatch) 避免开头和结尾
        if nCatch > 0
            catchIndices = round(linspace(2, totalSlots-1, nCatch));
            % 去重 (以防计算重叠)
            catchIndices = unique(catchIndices);
            % 标记这些位置为 Catch Trial
            trialSequenceType(catchIndices) = 1;
        end
    
        % --- 根据标记组装最终列表 ---
        newStimuliList = [];
        normalCounter = 1;
    
        for k = 1:length(trialSequenceType)
            if trialSequenceType(k) == 1
                % 这是一个 Catch Trial 位置
                newStimuliList = [newStimuliList; catchItemTemplate];
            else
                % 这是一个正常试次位置，从 blockedNormalTrials 取下一个
                if normalCounter <= length(blockedNormalTrials)
                    newStimuliList = [newStimuliList; blockedNormalTrials(normalCounter)];
                    normalCounter = normalCounter + 1;
                end
            end
        end
    
        finalStimuli = newStimuliList;
    end
    
    stimuli = finalStimuli; % 更新主变量
    fprintf('Final Trial List: %d trials generated.\n', length(stimuli));
    
    % =========================================================================
    
    % Setup KbQueue
    KbQueueCreate();
    KbQueueStart();
    
    results = struct('trialNum', {}, 'catName', {}, 'filename', {}, 'marker', {}, ...
        'stimOnset', {}, 'duration', {}, 'blankDuration', {}, 'response', {}, 'isCatch', {});
    
    % 5. Run Loop
    try
        HideCursor;
        startTime = GetSecs;
    
        % Initial Fixation
        Screen('DrawLines', window, [-cfg.fixation.size cfg.fixation.size 0 0; 0 0 -cfg.fixation.size cfg.fixation.size], cfg.fixation.lineWidth, cfg.fixation.color, [xC yC]);
        Screen('Flip', window);
    
        while true
            [keyIsDown, ~, keyCode] = KbCheck;
            if keyIsDown && keyCode(cfg.keys.space)
                break;
            end
            if keyIsDown && keyCode(cfg.keys.escape)
                error('User Abort');
            end
            WaitSecs(0.01);
        end
    
        WaitSecs(cfg.timing.initialWait);
        vbl = Screen('Flip', window);
    
        for i = 1:length(stimuli)
            trial = stimuli(i);
    
            thisBlankSecs = blankOptions(randi(length(blankOptions)));
            blankFrames = round(thisBlankSecs / ifi);
    
            % Draw Stimulus
            if trial.isCatch
                r = cfg.catch.radius;
                rect = CenterRectOnPoint([0 0 r*2 r*2], xC, yC);
                Screen('FrameOval', window, cfg.catch.color, rect, cfg.catch.penWidth);
            else
                Screen('DrawTexture', window, trial.texture, [], trial.rect);
                Screen('DrawLines', window, [-cfg.fixation.size cfg.fixation.size 0 0; 0 0 -cfg.fixation.size cfg.fixation.size], cfg.fixation.lineWidth, cfg.fixation.color, [xC yC]);
            end
    
            KbQueueFlush();
    
            % Stim Onset
            vbl = Screen('Flip', window, vbl + (blankFrames - 0.5) * ifi);
            stimOnsetTime = vbl;
    
            if cfg.io.useSerial
                io_utils.send_trigger(cfg.io.obj, trial.marker);
            end
    
            % Prepare Blank
            Screen('DrawLines', window, [-cfg.fixation.size cfg.fixation.size 0 0; 0 0 -cfg.fixation.size cfg.fixation.size], cfg.fixation.lineWidth, cfg.fixation.color, [xC yC]);
    
            % Stim Offset
            if trial.isCatch
                thisStimFrames = round(1.0 / ifi);
            else
                thisStimFrames = stimFrames;
            end
    
            vbl = Screen('Flip', window, vbl + (thisStimFrames - 0.5) * ifi);
            stimOffsetTime = vbl;
    
            % Response Check
            response = 0;
            [pressed, firstPress] = KbQueueCheck();
            if pressed
                if trial.isCatch && firstPress(cfg.keys.space) > 0
                    response = 1;
                end
                if firstPress(cfg.keys.escape) > 0
                    save_data(cfg, results, stimuli);
                    break;
                end
            end
    
            idx = length(results) + 1;
            results(idx).trialNum = i;
            results(idx).imgName = trial.filename;
            results(idx).category = trial.catName;
            results(idx).marker = trial.marker;
            results(idx).onsetTime = stimOnsetTime - startTime;
            results(idx).offsetTime = stimOffsetTime - startTime;
            results(idx).duration = stimOffsetTime - stimOnsetTime;
            results(idx).isCatch = trial.isCatch;
            results(idx).response = response;
        end
    
        save_data(cfg, results, stimuli);
    
    catch ME
        save_data(cfg, results, stimuli);
        KbQueueStop(); KbQueueRelease();
    
        % 【修改 catch 块中的清理逻辑】
        if ~isempty(stimuli)
            texs = [stimuli.texture];
            if ~isempty(texs)
                Screen('Close', texs);
            end
        end
    
        ShowCursor;
        rethrow(ME);
    end
    
    KbQueueStop(); KbQueueRelease();
    if ~isempty(stimuli)
        % [stimuli.texture] 会自动忽略空的 []，只提取有效的纹理句柄
        texs = [stimuli.texture];
    
        % 如果存在有效的纹理，则关闭它们
        if ~isempty(texs)
            Screen('Close', texs);
        end
    end
    ShowCursor;
    end
    
    function save_data(cfg, results, stimData)
    if isempty(results), return; end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    taskSuffix = regexprep(cfg.taskName, '[^a-zA-Z0-9]', '');
    fname = sprintf('%s_%s_Session%d_%s.mat', cfg.subID, taskSuffix, cfg.sessionNum, timestamp);
    savePath = fullfile(cfg.dataDir, fname);
    save(savePath, 'results', 'cfg', 'stimData');
    fprintf('Data saved to %s\n', savePath);
end