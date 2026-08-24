function trigger_test_tool()
    % TRIGGER_TEST_TOOL Simple utility to test serial triggers
    % Usage: trigger_test_tool()
    
    clc;
    fprintf('--- Trigger Test Tool ---\n');
    
    % Check if io_utils is available
    if isempty(which('io_utils'))
        error('io_utils.m not found. Make sure it is in the path.');
    end
    
    % Get Port
    defaultPort = 'COM3';
    p = input(sprintf('Enter COM port (default %s): ', defaultPort), 's');
    if isempty(p), p = defaultPort; end
    
    % Init
    try
        io = io_utils.init_port(p, 115200);
    catch ME
        fprintf('Failed to init port: %s\n', ME.message);
        return;
    end
    
    if ~io.enabled
        fprintf('Port init failed or simulated. Triggers will be fake.\n');
    end
    
    fprintf('Enter a number (0-255) to send trigger.\n');
    fprintf('Enter "q" to quit.\n');
    
    while true
        userIn = input('Trigger > ', 's');
        if strcmpi(userIn, 'q') || strcmpi(userIn, 'exit')
            break;
        end
        
        val = str2double(userIn);
        if isnan(val) || val < 0 || val > 255
            fprintf('Invalid number. Please enter 0-255.\n');
            continue;
        end
        
        io_utils.send_trigger(io, val);
        fprintf('Sent: %d\n', val);
    end
    
    io_utils.close(io);
    fprintf('Done.\n');
end
