function cfg = config_task3()
% CONFIG_TASK3 Task 3: Passive - Fruit Full
% Logic: Continuous flow
% Materials: Real, False, Gray fruits
% Calibration: None

    cfg = config_common();
    
    cfg.taskName = 'Task 3: Passive (Fruit Full)';
    cfg.stimDir = fullfile(cfg.picdir, 'Stimuli_Task3');
    
    % Timing
    cfg.timing.stimulus = 0.3; % seconds
    cfg.timing.blank = 1.05;    % seconds
    cfg.img.xOffset = -300;             % Centered
    cfg.img.yOffset = -50;
    % Categories
    cfg.categories = struct('name', {}, 'code', {});
    
    % Cabbage
    cfg.categories(1) = struct('name', 'Cabbage_True',  'code', 101);
    cfg.categories(2) = struct('name', 'Cabbage_False', 'code', 102);
    cfg.categories(3) = struct('name', 'Cabbage_Gray',  'code', 103);
    
    % Kiwi
    cfg.categories(4) = struct('name', 'Kiwi_True',     'code', 111);
    cfg.categories(5) = struct('name', 'Kiwi_False',    'code', 112);
    cfg.categories(6) = struct('name', 'Kiwi_Gray',     'code', 113);
    
    % Strawberry
    cfg.categories(7) = struct('name', 'Strawberry_True',  'code', 121);    
    cfg.categories(8) = struct('name', 'Strawberry_False', 'code', 122);
    cfg.categories(9) = struct('name', 'Strawberry_Gray',  'code', 123);
    
    % Watermelon
    cfg.categories(10) = struct('name', 'Watermelon_True',  'code', 131);
    cfg.categories(11) = struct('name', 'Watermelon_False', 'code', 132);
    cfg.categories(12) = struct('name', 'Watermelon_Gray',  'code', 133);
    
    % Image Selection
    cfg.img.countPerCategory = 20;
    cfg.img.scaleFactor = 1.5;
    
    % Calibration
    cfg.calibration.enabled = false;

end
