function cfg = config_task5()
% CONFIG_TASK5 Task 5: E-Stim Support
% Logic: Branch 2 (Wait Trigger -> Show -> Disappear)
% Materials: Same as Task 3
% Calibration: None

    cfg = config_common();
    
    cfg.taskName = 'Task 5: E-Stim Support';
    cfg.stimDir = fullfile(cfg.picdir, 'Stimuli_Task5');
    cfg.img.xOffset = -300;             % Centered
    cfg.img.yOffset = -50;
    % Timing
    cfg.timing.stimulus = 2.0; % Auto disappear duration
    cfg.timing.modeA_duration = cfg.timing.stimulus;

    % Task 5 Specific settings
    cfg.task.fixationDuration = 1.0;
    cfg.task.disappearMode = 'auto'; % 'auto' or 'manual'
    
    % UDP Delayed Trigger (Optional)
    cfg.udp.enabled = true;
    cfg.udp.ip = '192.168.1.2';
    cfg.udp.port = 9080;
    cfg.udp.message = '1';
    cfg.udp.delay = 1.3; % seconds after stimulus onset
    
    % Categories (Specific Sequence for Task 5: 4 Fruits)
    cfg.task.ordered_sequence = true;
    
    cfg.fruitParams = struct();
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
    
    % Standard categories for reference (ignored by ordered logic)
    cfg.categories = []; 
    
    % Image Selection
    cfg.img.countPerCategory = 15; % Same as Task 8? User didn't specify, but 2 is likely too small for ordered sequence. Assuming 15 like Task 8.
    cfg.img.scaleFactor = 1.5;
    
    % Calibration
    cfg.calibration.enabled = false;

end
