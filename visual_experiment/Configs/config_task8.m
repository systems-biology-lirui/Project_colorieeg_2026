function cfg = config_task8()

    cfg = config_common();
    
    cfg.taskName = 'Task 8: real_false_gray fruit';
    cfg.stimDir = fullfile(cfg.picdir, 'Stimuli_Task8'); % Use Task 5 stimuli
    
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
    
    % Categories (Specific Sequence for Task 8)
    cfg.task.ordered_sequence = true;
    
    cfg.fruitParams = struct();
    % Fruit 1: Strawberry
    cfg.fruitParams(1).name = 'Kiwi';
    cfg.fruitParams(1).codes = struct('Gray', 123, 'True', 121, 'False', 122);
    
    % Fruit 2: Cabbage (Using Cabbage as Fruit 2 based on existing files)
    cfg.fruitParams(2).name = 'Strawberry';
    cfg.fruitParams(2).codes = struct('Gray', 103, 'True', 101, 'False', 102);
    
    % Standard categories for reference (ignored by ordered logic)
    cfg.categories = []; 
    
    % Image Selection
    cfg.img.countPerCategory = 15; 
    cfg.img.scaleFactor = 1.5;
    
    % Calibration
    cfg.calibration.enabled = false;

end
