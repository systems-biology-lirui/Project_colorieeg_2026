function cfg = config_task7()

    cfg = config_common();
    cfg.screen.bgColor = [255 255 255]; % Match blank_01.bmp
    cfg.taskName = 'Task 7: white blank';
    cfg.stimDir = fullfile(cfg.picdir, 'Stimuli_Task7');
    if ~exist(cfg.stimDir, 'dir')
        % Fallback to Task 5 dir if Task 6 dir doesn't exist, just in case?
        % No, better to be explicit. User can create dir.
        % Or maybe I should default to Stimuli_Task5 but filter?
        % User said "only gray images", implying the content.
        % Let's assume Stimuli_Task6 exists or will be created.
    end
    
    cfg.img.xOffset = -300;             % Centered
    cfg.img.yOffset = -50;
    % Timing
    cfg.timing.stimulus = 2.0; 
    cfg.timing.modeA_duration = cfg.timing.stimulus;

    % Task Specific settings
    cfg.task.fixationDuration = 1.0;
    cfg.task.disappearMode = 'auto'; 
    
    % UDP Delayed Trigger (Optional)
    cfg.udp.enabled = false;
    cfg.udp.ip = '192.168.1.2';
    cfg.udp.port = 9080;
    cfg.udp.message = '1';
    cfg.udp.delay = 1.3; % seconds after stimulus onset
    
    % Categories (Gray Only)
    cfg.categories = struct('name', {}, 'code', {});

    cfg.categories(1) = struct('name', 'white',  'code', 200);
    
    % Image Selection
    cfg.img.countPerCategory = 20; 
    cfg.img.scaleFactor = 1.5;
    
    % Calibration
    cfg.calibration.enabled = false;

end
