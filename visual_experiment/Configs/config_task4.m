function cfg = config_task4()
% CONFIG_TASK4 Task 4: Passive - Fruit Gray
% Logic: Continuous flow
% Materials: Gray fruits only
% Calibration: None

    cfg = config_common();
    
    cfg.taskName = 'Task 4: Passive (Fruit Gray)';
    cfg.stimDir = fullfile(pwd, 'Stimuli_Task4');
    
    % Timing
    cfg.timing.stimulus = 0.3; % seconds
    cfg.timing.blank = 1.05;    % seconds
    
    % Categories
    cfg.categories = struct('name', {}, 'code', {});
    
    cfg.categories(1) = struct('name', 'Cabbage_Gray',    'code', 103);
    cfg.categories(2) = struct('name', 'Kiwi_Gray',       'code', 113);
    cfg.categories(3) = struct('name', 'Strawberry_Gray', 'code', 123);
    cfg.categories(4) = struct('name', 'Watermelon_Gray', 'code', 133);
    
    % Image Selection
    cfg.img.countPerCategory = 7;
    cfg.img.scaleFactor = 1.0;
    
    % Calibration
    cfg.calibration.enabled = false;

end
