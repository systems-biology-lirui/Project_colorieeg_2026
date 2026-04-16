function cfg = config_task2()
% CONFIG_TASK2 Task 2: Passive - Color Patches
% Logic: Continuous flow
% Calibration: YES (Apply scaleFactor from calibration_task)

    cfg = config_common();
    
    cfg.taskName = 'Task 2: Passive (Color Patches)';
    cfg.stimDir = fullfile(cfg.picdir, 'Stimuli_Task2');
    
    cfg.img.xOffset = -300;             % Centered
    cfg.img.yOffset = -50;
    % Timing
    cfg.timing.stimulus = 0.3; % seconds
    cfg.timing.blank = 1.05;    % seconds
    
    % Categories
    cfg.categories = struct('name', {}, 'code', {});
    cfg.categories(1) = struct('name', 'Red',    'code', 51);
    cfg.categories(2) = struct('name', 'Yellow', 'code', 52);
    cfg.categories(3) = struct('name', 'Blue',   'code', 53);
    cfg.categories(4) = struct('name', 'Green',  'code', 54);
    % Optional
    cfg.categories(5) = struct('name', 'Black',  'code', 55);
    cfg.categories(6) = struct('name', 'White',  'code', 56);
    
    % Image Selection
    cfg.img.countPerCategory = 30;
    cfg.img.scaleFactor = 1.5; % Base scale
    
    % Calibration
    cfg.calibration.enabled = true;

end
