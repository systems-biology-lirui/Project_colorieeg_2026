function cfg = config_common()
% CONFIG_COMMON 所有任务的通用配置

    %% 1. 被试信息与路径
    cfg.subID = 'test001';
    cfg.dataDir = fullfile(pwd, 'Data');
    if ~exist(cfg.dataDir, 'dir'), mkdir(cfg.dataDir); end
    cfg.picdir = 'D:\Desktop\seeg\stimuli_pic';
    cfg.session.historyFile = fullfile(cfg.dataDir, 'session_history.mat');

    %% 2. 屏幕设置
    cfg.screen.bgColor = [100 100 100]; % 与 blank_01.bmp 匹配的灰色背景
    cfg.screen.dist = 60;            % 被试与屏幕的距离 (厘米)
    cfg.screen.fullsize = [];        % [] 表示全屏，或设置为 [0 0 800 600] 用于调试
    
    % 图像几何参数
    cfg.img.xOffset = -350;             % 水平偏移量（相对于屏幕中心）
    cfg.img.yOffset = -50;              % 垂直偏移量
    cfg.img.scaleFactor = 1.5;       % 默认缩放因子，可被具体任务覆盖

    %% 3. 串口通信与Trigger设置
    cfg.io.useSerial = true;        % 是否启用串口Trigger，默认开启
    cfg.io.portName = 'COM3';       % 串口名称
    cfg.io.baudRate = 115200;       % 波特率

    %% 4. 按键设置
    KbName('UnifyKeyNames');
    cfg.keys.escape = KbName('ESCAPE');     % 退出键
    cfg.keys.space = KbName('SPACE');       % 空格键（开始/响应）
    cfg.keys.up = KbName('UpArrow');        % 上箭头（亮度调节等）
    cfg.keys.down = KbName('DownArrow');    % 下箭头
    cfg.keys.next = KbName('RightArrow');   % 右箭头（手动推进下一试次）

    %% 5. 时序参数（默认值）
    cfg.timing.ifi = 0;              % 帧间隔，运行时自动测量
    cfg.timing.stimulus = 0.3;       % 刺激呈现时间 (秒)
    cfg.timing.blank = 1.05;         % 空白期基准时间 (秒)
    cfg.timing.blankJitter = 0.3;    % 抖动范围（已弃用，改用下方参数）
    cfg.timing.blankRange = [0.9 1.2]; % 空白期抖动范围 [最小值 最大值] (秒)
    cfg.timing.blankStep = 0.05;     % 抖动步长，每隔50ms一档
    cfg.timing.initialWait = 2.0;    % 实验开始前的等待时间 (秒)

    %% 6. 注视点设置
    cfg.fixation.size = 40;          % 注视十字的臂长 (像素)
    cfg.fixation.color = [0 0 0];    % 注视点颜色 (黑色)
    cfg.fixation.lineWidth = 6;      % 注视十字的线宽

    %% 7. 光电二极管反馈区
    cfg.feedback.rectSize = 50;      % 反馈方块大小 (像素)
    cfg.feedback.color = [255 255 255]; % 反馈方块颜色 (白色)

    %% 8. 亮度校准设置
    cfg.calibration.enabled = false; % 是否启用校准
    cfg.calibration.file = fullfile(cfg.dataDir, 'color_calibration.mat'); % 校准数据文件路径

    %% 9. Catch试次设置
    cfg.catch.probability = 0.10;     % Catch试次出现概率 (10%)
    cfg.catch.trigger = 99;           % Catch试次的Trigger标记
    cfg.catch.responseKey = cfg.keys.space; % 被试响应按键
    cfg.catch.color = [0 0 0];       % 圆环颜色 (黑色)
    cfg.catch.radius = 40;           % 圆环半径 (像素)
    cfg.catch.penWidth = 5;          % 圆环线宽

end
