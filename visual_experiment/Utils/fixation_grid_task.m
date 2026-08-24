% Clear the workspace and the screen
sca;
close all;
clear;

% Here we call some default settings for setting up Psychtoolbox
PsychDefaultSetup(2);

% Skip sync tests for demo purposes (comment out for actual experiment if timing is critical)
Screen('Preference', 'SkipSyncTests', 1);

% Get the screen numbers
screens = Screen('Screens');

% Draw to the external screen if avaliable
screenNumber = max(screens);

% Define black and white
white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);

% Open an on screen window
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, white);

% Get the size of the on screen window
[screenXpixels, screenYpixels] = Screen('WindowSize', window);

% Get the centre coordinate of the window
[xCenter, yCenter] = RectCenter(windowRect);

% ----------------------------------------------------------------------
%                       Fixation Cross Setup
% ----------------------------------------------------------------------

% Here we set the size of the arms of our fixation cross
fixCrossDimPix = 40;

% Now we set the coordinates (all relative to zero)
xCoords = [-fixCrossDimPix fixCrossDimPix 0 0];
yCoords = [0 0 -fixCrossDimPix fixCrossDimPix];
allCoords = [xCoords; yCoords];

% Set the line width for our fixation cross
lineWidthPix = 4;

% ----------------------------------------------------------------------
%                       Grid (5x8 Dots) Setup
% ----------------------------------------------------------------------

% Define the number of rows and columns
numRows = 5;
numCols = 8;

% Define the grid area (e.g., 80% of screen width and height)
gridWidth = screenXpixels * 0.8;
gridHeight = screenYpixels * 0.8;

% Calculate x and y positions for the dots
% linspace generates linearly spaced vectors
xPos = linspace(xCenter - gridWidth/2, xCenter + gridWidth/2, numCols);
yPos = linspace(yCenter - gridHeight/2, yCenter + gridHeight/2, numRows);

% Create a meshgrid to get all combinations of x and y
[xGrid, yGrid] = meshgrid(xPos, yPos);

% Reshape into a 2xN matrix for DrawDots
% xGrid(:)' flattens the matrix into a row vector
dotPositionMatrix = [xGrid(:)'; yGrid(:)'];

% Dot size in pixels
dotSizePix = 20;

% ----------------------------------------------------------------------
%                       Experimental Loop
% ----------------------------------------------------------------------

% Define exit key
KbName('UnifyKeyNames');
exitKey = KbName('ESCAPE');

disp('Press any key to switch screens. Press ESC to exit.');

running = true;
state = 1; % 1 = Fixation, 2 = Grid

while running
    
    if state == 1
        % --- State 1: Fixation Cross ---
        
        % Draw the fixation cross in black, centered on the screen
        Screen('DrawLines', window, allCoords, lineWidthPix, black, [xCenter yCenter], 0);
        
        % Flip to the screen
        Screen('Flip', window);
        
        % Wait for a key press
        [secs, keyCode, deltaSecs] = KbWait(-1, 2);
        
        % Check for exit key
        if keyCode(exitKey)
            running = false;
        else
            % Switch to next state
            state = 2;
        end
        
    elseif state == 2
        % --- State 2: 5x8 Dot Grid ---
        
        % Draw the dots
        % dotPositionMatrix contains coordinates relative to (0,0) if we use center [0,0]
        % but our matrix already has absolute screen coordinates.
        % So we pass [0 0] as the center offset.
        Screen('DrawDots', window, dotPositionMatrix, dotSizePix, black, [0 0], 0);
        
        % Flip to the screen
        Screen('Flip', window);
        
        % Wait for a key press
        [secs, keyCode, deltaSecs] = KbWait(-1, 2);
        
        % Check for exit key
        if keyCode(exitKey)
            running = false;
        else
            % Switch back to first state
            state = 1;
        end
    end
    
    % Small pause to prevent double detection if keys are held down (though KbWait usually handles key down)
    % WaitSecs(0.2); 
end

% Clear the screen
sca;
