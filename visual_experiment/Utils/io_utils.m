classdef io_utils
    % IO_UTILS 用于处理串口通信 (Serial Port Trigger) 的工具类
    % 使用方法:
    %   io = io_utils.init_port('COM3', 115200);
    %   io_utils.send_trigger(io, 10);
    %   io_utils.close(io);

    methods (Static)
        function io = init_port(portName, baudRate)
            % INIT_PORT 初始化串口对象
            % portName: 串口名称 (例如 'COM3')
            % baudRate: 波特率 (例如 115200)
            % 返回一个结构体，包含 serialport 对象
            
            io.enabled = true;
            io.portName = portName;
            io.baudRate = baudRate;
            io.obj = [];
            
            try
                % 初始化 serialport
                io.obj = serialport(portName, baudRate);
                fprintf('Serial Port %s Initialized successfully at %d baud.\n', portName, baudRate);
                
                % 发送复位信号测试
                write(io.obj, [1 225 1 0 0], 'uint8');
                
            catch ME
                warning('Error initializing Serial Port %s: %s. Triggers will be disabled.', portName, ME.message);
                io.enabled = false;
            end
        end

        function send_trigger(io, trigVal)
            % SEND_TRIGGER 发送 Trigger Packet
            % io: init_port 返回的结构体
            % trigVal: 要发送的整数 Marker (0-255)
            %
            % 协议逻辑:
            % 1. Header/Reset: [1 225 1 0 0]
            % 2. Pause 5ms
            % 3. Trigger: [1 225 1 0 trigVal]
            
            if io.enabled
                try
                    % 1. 复位/Header
                    write(io.obj, [1 225 1 0 0], 'uint8');
                    
                    % 2. 暂停 5ms (使用 WaitSecs 以获得更高精度)
                    WaitSecs(0.005); 
                    
                    % 3. 发送真实 Trigger
                    write(io.obj, [1 225 1 0 trigVal], 'uint8');
                    
                catch ME
                    % 忽略发送错误，避免中断测试，但打印警告
                    fprintf('Trigger Send Failed: %s\n', ME.message);
                end
            else
                % 调试模式下仅打印
                fprintf('[Simulated Trigger] Code: %d\n', trigVal);
            end
        end
        
        function close(io)
            % CLOSE 关闭/清理
            if io.enabled && ~isempty(io.obj)
                try
                    delete(io.obj);
                    fprintf('Serial Port Closed.\n');
                catch
                end
            end
        end
    end
end
