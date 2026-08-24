function cfg = config_task1()
% CONFIG_TASK1 Task 1: Passive - Real/Gray
% Logic: Continuous flow (Fixation -> Stim -> Blank)
% Calibration: None

    cfg = config_common();
    
    cfg.taskName = 'Task 1: Passive (Real/Gray)';
    cfg.stimDir = fullfile(cfg.picdir, 'Stimuli_Task1');
    
    % Timing
    cfg.timing.stimulus = 0.3; % seconds
    cfg.timing.blank = 1.05;    % seconds
    cfg.img.xOffset = -350;             % Centered
    cfg.img.yOffset = -50;
    % Categories
    cfg.categories = struct('name', {}, 'code', {});
    cfg.categories(1) = struct('name', 'Face_Color',   'code', 11);
    cfg.categories(2) = struct('name', 'Face_Gray',    'code', 12);
    cfg.categories(3) = struct('name', 'Object_Color', 'code', 21);
    cfg.categories(4) = struct('name', 'Object_Gray',  'code', 22);
    cfg.categories(5) = struct('name', 'Body_Color',   'code', 31);
    cfg.categories(6) = struct('name', 'Body_Gray',    'code', 32);
    cfg.categories(7) = struct('name', 'Place_Color',   'code', 41);
    cfg.categories(8) = struct('name', 'Place_Gray',    'code', 42);  
    
    % Image Selection
    cfg.img.countPerCategory = 35;
    cfg.img.scaleFactor = 0.6; % From old config
    
    % Calibration
    cfg.calibration.enabled = false;

end
