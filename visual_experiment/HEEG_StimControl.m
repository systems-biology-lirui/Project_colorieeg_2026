
%% 控制端口参数设置
ipOutput='192.168.1.2'; 
portOutput=9080; 
%Output=0;%可选0/1/2：0不刺激，1单通道刺激，2多通道同时刺激

%% 声明与开启UDP
udpOutput=udp(ipOutput, portOutput);
fopen(udpOutput);

%% 传输控制命令
% 修改为：间隔1s，重复40次，output是1
Output = 1;
numReps = 40;
interval = 1.0;

fprintf('Starting Stimulation: Output=%d, Reps=%d, Interval=%.2fs\n', Output, numReps, interval);

for idx = 1:numReps
    % 传输控制命令
    fwrite(udpOutput, num2str(Output)); % 注意传输的是字符串
    
    currTime = now;
    disp([datestr(currTime,'yyyymmdd-HHMMSS') '，输出命令' num2str(Output) '，第 ' num2str(idx) '/' num2str(numReps) ' 次']);
    
    pause(interval); % 暂停1s
end

%% 关闭/释放UDP
fclose(udpOutput);
delete(udpOutput);
disp('Done.');
